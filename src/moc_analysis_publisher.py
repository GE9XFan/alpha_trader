"""Premium MOC imbalance analysis publisher.

Schedules intraday analysis drops for the premium Discord analysis channel.
Fetches synthetic MOC projections from Redis, enriches them with historical
context, and enqueues formatted payloads for the Discord relay workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import redis.asyncio as aioredis

from redis_keys import analytics_metric_key

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


@dataclass
class MOCSymbolSnapshot:
    symbol: str
    imbalance_total: float
    imbalance_ratio: float
    imbalance_side: str
    indicative_price: Optional[float]
    near_close_offset_bps: Optional[float]
    projected_volume_shares: Optional[float]
    gamma_factor: Optional[float]
    minutes_to_close: Optional[float]
    change_vs_yesterday: Optional[float]
    timestamp: float

    def to_payload(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'imbalance_total': self.imbalance_total,
            'imbalance_ratio': self.imbalance_ratio,
            'imbalance_side': self.imbalance_side,
            'indicative_price': self.indicative_price,
            'near_close_offset_bps': self.near_close_offset_bps,
            'projected_volume_shares': self.projected_volume_shares,
            'gamma_factor': self.gamma_factor,
            'minutes_to_close': self.minutes_to_close,
            'change_vs_yesterday': self.change_vs_yesterday,
        }


class MOCAnalysisPublisher:
    """Coordinates premium analysis drops for market-on-close projections."""

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        module_cfg = (
            config.get('modules', {})
            .get('analysis_publisher', {})
            .get('moc', {})
        )

        self.enabled = config.get('modules', {}).get('analysis_publisher', {}).get('enabled', False)
        self.symbols: Sequence[str] = module_cfg.get('symbols') or []
        self.schedule_strings: Sequence[str] = module_cfg.get('schedule') or []
        self.window_start_str: str = module_cfg.get('window_start', '08:00')
        self.min_notional: float = float(module_cfg.get('min_notional', 5e8))
        self.min_watchlist_ratio: float = float(module_cfg.get('min_watchlist_ratio', 0.40))
        self.stale_after_seconds: int = int(module_cfg.get('stale_after_seconds', 180))
        self.history_ttl_hours: int = int(module_cfg.get('historical_ttl_hours', 72))
        ping_cfg = module_cfg.get('ping_thresholds', {}) or {}
        self.ping_extreme: float = float(ping_cfg.get('extreme_notional', 5e9))
        self.ping_final: float = float(ping_cfg.get('final_notional', 2e9))
        self.market_holidays: Set[date] = self._parse_holidays(module_cfg.get('market_holidays') or [])

        self.queue_name: str = module_cfg.get('queue', 'analysis:premium:moc')
        self.dedupe_ttl: int = 7200  # two hours
        self.poll_interval: int = 30

        try:
            self.market_tz = ZoneInfo('US/Eastern')
        except Exception:  # pragma: no cover
            self.market_tz = ZoneInfo('UTC')

        self.schedule_times: List[dtime] = self._parse_schedule(self.schedule_strings)
        self.schedule_fractions: List[float] = [t.hour + t.minute / 60 for t in self.schedule_times]
        self.window_start = self._parse_time(self.window_start_str)

        self._stop_event = asyncio.Event()
        self._last_run_slot: Optional[str] = None
        self.metrics_prefix = 'metrics:analysis:moc'

    async def start(self) -> None:
        """Begin the polling loop for scheduled analysis drops."""

        if not self.enabled:
            self.logger.info("MOC analysis publisher disabled")
            return
        if not self.schedule_times:
            self.logger.warning("MOC analysis publisher has no schedule configured")
            return
        if not self.symbols:
            self.logger.warning("MOC analysis publisher has no symbols configured")
            return

        self.logger.info(
            "Starting MOC analysis publisher",
            extra={'schedule': self.schedule_strings, 'queue': self.queue_name, 'symbols': list(self.symbols)},
        )

        while not self._stop_event.is_set():
            try:
                await self._maybe_publish()
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.exception("Error during MOC analysis publishing: %s", exc)
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._stop_event.set()

    async def _maybe_publish(self) -> None:
        now = datetime.now(self.market_tz)
        slot = self._resolve_slot(now)
        if slot is None:
            return

        slot_code, slot_dt = slot

        if not self.should_run_now(slot_dt):
            return

        dedupe_key = f"analysis:moc:{slot_dt.strftime('%Y%m%d_%H')}:sent"
        if not await self.redis.setnx(dedupe_key, "1"):
            self.logger.debug("MOC analysis already sent within this hour", extra={'slot': slot_code})
            return
        await self.redis.expire(dedupe_key, self.dedupe_ttl)

        payload = await self.build_analysis_payload(slot_dt, slot_code)
        if payload is None:
            return

        await self.redis.lpush(self.queue_name, json.dumps(payload))
        await self.redis.incr(f"{self.metrics_prefix}:sent")
        self._last_run_slot = slot_code
        self.logger.info("Queued MOC analysis payload", extra={'slot': slot_code, 'status': payload.get('status')})

    async def build_analysis_payload(self, slot_dt: datetime, slot_code: str) -> Optional[Dict[str, Any]]:
        data, stale_symbols = await self._gather_moc_data(self.symbols, slot_dt, slot_code)

        timestamp_utc = datetime.now(self.market_tz).astimezone(ZoneInfo('UTC')).isoformat()
        slot_label = slot_dt.strftime('%H:%M ET')
        next_update_text = self.get_next_update_text(slot_dt)

        if not data:
            note = 'MOC data temporarily unavailable - models recalibrating'
            if stale_symbols:
                note = f"MOC data temporarily unavailable for {', '.join(stale_symbols)} - models recalibrating"
            return {
                'type': 'analysis.moc',
                'id': f"moc:{slot_code}",
                'status': 'stale',
                'slot_label': slot_label,
                'timestamp': timestamp_utc,
                'featured': [],
                'watchlist': [],
                'note': note,
                'next_update_text': next_update_text,
                'mention_everyone': False,
                'dedupe_token': f"analysis.moc:{slot_code}",
                'extreme': False,
            }

        sorted_data = sorted(data, key=lambda entry: entry.imbalance_total, reverse=True)
        featured_snapshots, watchlist_snapshots = self.select_symbols_for_display(sorted_data)

        featured_payload = [snap.to_payload() for snap in featured_snapshots]
        watchlist_payload = [snap.to_payload() for snap in watchlist_snapshots]

        note: Optional[str] = None
        if not featured_payload:
            note = f"No qualifying imbalances ≥ {_format_currency(self.min_notional)} — monitoring continues"

        max_imbalance = max(entry.imbalance_total for entry in sorted_data)
        mention_all = self.should_ping_everyone(sorted_data, slot_dt)
        extreme_flag = max_imbalance >= self.ping_extreme

        # Persist history for tomorrow's comparison
        await self._store_history(sorted_data, slot_code, slot_dt)

        return {
            'type': 'analysis.moc',
            'id': f"moc:{slot_code}",
            'status': 'ok',
            'slot_label': slot_label,
            'timestamp': timestamp_utc,
            'featured': featured_payload,
            'watchlist': watchlist_payload,
            'note': note,
            'next_update_text': next_update_text,
            'mention_everyone': mention_all,
            'dedupe_token': f"analysis.moc:{slot_code}",
            'extreme': extreme_flag,
        }

    def should_run_now(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        if now.date() in self.market_holidays:
            return False

        earliest = now.replace(hour=self.window_start.hour, minute=self.window_start.minute, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return earliest <= now <= market_close

    def select_symbols_for_display(
        self,
        data: Sequence[MOCSymbolSnapshot],
    ) -> Tuple[List[MOCSymbolSnapshot], List[MOCSymbolSnapshot]]:
        qualifying = [snap for snap in data if snap.imbalance_total >= self.min_notional]

        featured = qualifying[:3]
        watchlist = [snap for snap in qualifying[3:8] if snap.imbalance_ratio >= self.min_watchlist_ratio]

        return featured, watchlist

    def get_next_update_text(self, current_dt: datetime) -> str:
        current_fraction = current_dt.hour + current_dt.minute / 60

        for idx, slot_fraction in enumerate(self.schedule_fractions):
            if slot_fraction > current_fraction + 1e-9:
                hours_until = slot_fraction - current_fraction
                label = self._format_fraction(slot_fraction)
                if idx == len(self.schedule_fractions) - 1 or hours_until < 1:
                    return f"Final update at {label}"
                return f"Next update in ~{int(round(hours_until))}h"

        return "Market closes at 4:00 PM ET"

    def should_ping_everyone(self, data: Sequence[MOCSymbolSnapshot], current_dt: datetime) -> bool:
        if not data:
            return False

        max_imbalance = max(snap.imbalance_total for snap in data)
        if max_imbalance >= self.ping_extreme:
            return True

        current_fraction = current_dt.hour + current_dt.minute / 60
        if current_fraction >= 15.0 and max_imbalance >= self.ping_final:
            return True

        return False

    async def _gather_moc_data(
        self,
        symbols: Sequence[str],
        slot_dt: datetime,
        slot_code: str,
    ) -> Tuple[List[MOCSymbolSnapshot], List[str]]:
        if not symbols:
            return [], []

        pipe = self.redis.pipeline()
        hour_suffix = slot_code[-4:]
        for symbol in symbols:
            pipe.get(analytics_metric_key(symbol, 'moc'))
            pipe.get(self._history_key(symbol, hour_suffix))

        results = await pipe.execute()

        utc_now = datetime.now(self.market_tz).astimezone(ZoneInfo('UTC'))
        now_ts = utc_now.timestamp()
        yesterday = (slot_dt.date() - timedelta(days=1)).isoformat()

        snapshots: List[MOCSymbolSnapshot] = []
        stale_symbols: List[str] = []

        for idx, symbol in enumerate(symbols):
            metric_raw = results[idx * 2]
            history_raw = results[idx * 2 + 1]

            metric = self._decode_json(metric_raw)
            if not isinstance(metric, dict):
                stale_symbols.append(symbol)
                continue

            try:
                ts = float(metric.get('timestamp') or 0.0)
            except (TypeError, ValueError):
                ts = 0.0

            if ts <= 0 or now_ts - ts > self.stale_after_seconds:
                stale_symbols.append(symbol)
                continue

            change_vs_yesterday: Optional[float] = None
            history = self._decode_json(history_raw)
            if isinstance(history, dict) and history.get('date') == yesterday:
                try:
                    change_vs_yesterday = float(metric.get('imbalance_total', 0.0)) - float(history.get('imbalance_total', 0.0))
                except (TypeError, ValueError):
                    change_vs_yesterday = None

            try:
                snapshots.append(
                    MOCSymbolSnapshot(
                        symbol=symbol.upper(),
                        imbalance_total=float(metric.get('imbalance_total', 0.0)),
                        imbalance_ratio=float(metric.get('imbalance_ratio', 0.0)),
                        imbalance_side=str(metric.get('imbalance_side') or 'FLAT').upper(),
                        indicative_price=self._to_float(metric.get('indicative_price')),
                        near_close_offset_bps=self._to_float(metric.get('near_close_offset_bps')),
                        projected_volume_shares=self._to_float(metric.get('projected_volume_shares')),
                        gamma_factor=self._to_float(metric.get('gamma_factor')),
                        minutes_to_close=self._to_float(metric.get('minutes_to_close')),
                        change_vs_yesterday=change_vs_yesterday,
                        timestamp=ts,
                    )
                )
            except (TypeError, ValueError):
                stale_symbols.append(symbol)
                continue

        return snapshots, stale_symbols

    async def _store_history(self, snapshots: Sequence[MOCSymbolSnapshot], slot_code: str, slot_dt: datetime) -> None:
        if not snapshots or self.history_ttl_hours <= 0:
            return

        ttl_seconds = int(self.history_ttl_hours * 3600)
        hour_suffix = slot_code[-4:]
        pipe = self.redis.pipeline()
        for snap in snapshots:
            history_payload = {
                'date': slot_dt.date().isoformat(),
                'imbalance_total': snap.imbalance_total,
            }
            pipe.setex(self._history_key(snap.symbol, hour_suffix), ttl_seconds, json.dumps(history_payload))
        await pipe.execute()

    def _resolve_slot(self, now: datetime) -> Optional[Tuple[str, datetime]]:
        for slot_time in self.schedule_times:
            target = now.replace(hour=slot_time.hour, minute=slot_time.minute, second=0, microsecond=0)
            delta = abs((now - target).total_seconds())
            if delta <= self.poll_interval:
                slot_code = target.strftime('%Y%m%d%H%M')
                if slot_code != self._last_run_slot:
                    return slot_code, target
        return None

    def _parse_schedule(self, schedule: Sequence[str]) -> List[dtime]:
        parsed: List[dtime] = []
        for entry in schedule:
            try:
                parsed.append(self._parse_time(entry))
            except ValueError:
                self.logger.warning("Invalid schedule entry for MOC publisher", extra={'entry': entry})
        return sorted(parsed)

    @staticmethod
    def _parse_time(value: str) -> dtime:
        parts = value.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid time value: {value}")
        hour = int(parts[0])
        minute = int(parts[1])
        return dtime(hour=hour, minute=minute)

    @staticmethod
    def _format_fraction(fraction: float) -> str:
        hour = int(fraction)
        minute = int(round((fraction - hour) * 60))
        if minute >= 60:
            hour += 1
            minute -= 60
        return f"{hour:02d}:{minute:02d} ET"

    def _history_key(self, symbol: str, hour_suffix: str) -> str:
        return f"analysis:moc:{symbol}:{hour_suffix}:history"

    def _decode_json(self, raw: Any) -> Any:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return raw

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_holidays(values: Sequence[str]) -> Set[date]:
        holidays: Set[date] = set()
        for entry in values:
            try:
                holidays.add(datetime.strptime(entry, '%Y-%m-%d').date())
            except ValueError:
                continue
        return holidays


def _format_currency(value: float) -> str:
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:.2f}"


__all__ = ['MOCAnalysisPublisher']
