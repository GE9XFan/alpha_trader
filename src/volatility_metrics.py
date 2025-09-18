#!/usr/bin/env python3
"""Volatility metrics ingestion and analytics (VIX1D integration)."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import redis.asyncio as aioredis

import redis_keys as rkeys


class VolatilityMetrics:
    """Fetch and persist VIX1D metrics with simple regime classification."""

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        analytics_cfg = config.get('modules', {}).get('analytics', {})
        store_ttls = analytics_cfg.get('store_ttls', {})
        vol_cfg = analytics_cfg.get('volatility', {})
        self.ttl = int(vol_cfg.get('ttl', store_ttls.get('analytics', 60)))
        self.history_window = int(vol_cfg.get('history_window', 480))
        self.timeout = float(vol_cfg.get('timeout', 10.0))
        self.session: Optional[aiohttp.ClientSession] = None

        thresholds = vol_cfg.get('vix1d_thresholds', {})
        self.shock_threshold = float(thresholds.get('shock', 35.0))
        self.elevated_threshold = float(thresholds.get('elevated', 25.0))
        self.benign_threshold = float(thresholds.get('benign', 18.0))

        change_cfg = vol_cfg.get('change_thresholds', {})
        self.shock_change = float(change_cfg.get('shock', 5.0))
        self.elevated_change = float(change_cfg.get('elevated', 2.0))

        data_sources_cfg = vol_cfg.get('data_sources', {})
        self.primary_source = str(data_sources_cfg.get('primary', 'cboe')).lower()
        self.enable_fallback = bool(data_sources_cfg.get('enable_fallback', True))
        self.cache_duration = int(data_sources_cfg.get('cache_duration', 300))

        self.cboe_cache_key = 'volatility:vix1d:cboe:last'
        self.cboe_history_cache_key = 'volatility:vix1d:cboe:history'
        self.yahoo_backoff_key = 'volatility:vix1d:yahoo_backoff'
        self.yahoo_retry_count = 0

    async def update_vix1d(self) -> Dict[str, Any]:
        fetched = await self._fetch_vix1d()
        if isinstance(fetched, tuple):
            value, source = fetched
        else:  # Backwards compatibility with monkeypatched helpers returning raw floats
            value, source = fetched, None
        if value is None:
            cached = await self.redis.get(rkeys.analytics_vix1d_key())
            if cached:
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    pass
            return {'error': 'unavailable'}

        as_of = datetime.now(timezone.utc)
        history_stats, change_5m, change_1h, percentile = await self._update_history(value, as_of.timestamp())
        regime = self._classify_regime(value, history_stats.get('zscore', 0.0), change_5m, change_1h)

        payload = self._build_payload(
            value=value,
            history_stats=history_stats,
            change_5m=change_5m,
            change_1h=change_1h,
            percentile=percentile,
            regime=regime,
            as_of=as_of,
            source=source or 'auto',
        )

        await self.redis.setex(rkeys.analytics_vix1d_key(), self.ttl, json.dumps(payload))
        return payload

    async def ingest_manual(
        self,
        value: float,
        *,
        as_of: Optional[datetime] = None,
        source: str = 'manual',
    ) -> Dict[str, Any]:
        """Persist an externally supplied VIX1D reading for backfills.

        Args:
            value: The VIX1D level to store.
            as_of: Timestamp associated with the reading (defaults to current UTC time).
            source: Label describing the upstream provider for auditing.

        Returns:
            The payload written to Redis, mirroring :meth:`update_vix1d`.
        """

        if as_of is None:
            as_of = datetime.now(timezone.utc)
        elif as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        history_stats, change_5m, change_1h, percentile = await self._update_history(value, as_of.timestamp())
        regime = self._classify_regime(value, history_stats.get('zscore', 0.0), change_5m, change_1h)

        payload = self._build_payload(
            value=value,
            history_stats=history_stats,
            change_5m=change_5m,
            change_1h=change_1h,
            percentile=percentile,
            regime=regime,
            as_of=as_of,
            source=source,
        )

        await self.redis.setex(rkeys.analytics_vix1d_key(), self.ttl, json.dumps(payload))
        return payload

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _fetch_vix1d(self) -> Tuple[Optional[float], Optional[str]]:
        for source in self._ordered_sources():
            if source == 'cboe':
                value = await self._fetch_vix1d_cboe()
            elif source == 'ibkr':
                value = await self._fetch_vix1d_ibkr()
            elif source == 'yahoo':
                value = await self._fetch_vix1d_yahoo_with_backoff()
            else:
                continue

            if value is not None:
                self.logger.debug("VIX1D from %s: %.4f", source, value)
                return value, source

        self.logger.warning("VIX1D unavailable from configured sources")
        return None, None

    def _ordered_sources(self) -> List[str]:
        universe = ['cboe', 'ibkr', 'yahoo']
        primary = self.primary_source if self.primary_source in universe else 'cboe'
        if not self.enable_fallback:
            return [primary]

        order: List[str] = [primary]
        for candidate in universe:
            if candidate not in order:
                order.append(candidate)
        return order

    async def _fetch_vix1d_cboe(self) -> Optional[float]:
        cached_value = await self._get_cached_cboe_value()
        if cached_value is not None:
            return cached_value

        history = await self._fetch_and_cache_cboe_history()
        if not history:
            return None

        latest = history[-1]
        try:
            close_value = float(latest['close'])
        except (KeyError, TypeError, ValueError):
            return None

        date_str = latest.get('date')
        if date_str:
            try:
                data_date = self._parse_cboe_date(date_str)
            except ValueError:
                data_date = None
            else:
                today = datetime.now().date()
                if data_date and data_date < today - timedelta(days=2):
                    # Too stale to trust
                    self.logger.debug("Discarding stale CBOE VIX1D value from %s", date_str)
                    return None

        payload = {
            'timestamp': time.time(),
            'value': close_value,
            'date': date_str,
        }
        await self.redis.setex(self.cboe_cache_key, max(self.cache_duration, 60), json.dumps(payload))
        return close_value

    async def _get_cached_cboe_value(self) -> Optional[float]:
        raw = await self.redis.get(self.cboe_cache_key)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            ts = float(payload.get('timestamp', 0))
            if time.time() - ts > self.cache_duration:
                return None
            return float(payload.get('value'))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    async def _fetch_and_cache_cboe_history(self) -> Optional[List[Dict[str, Any]]]:
        raw_cache = await self.redis.get(self.cboe_history_cache_key)
        if raw_cache:
            try:
                payload = json.loads(raw_cache)
                cached_ts = float(payload.get('timestamp', 0))
                if time.time() - cached_ts < 4 * 3600:
                    data = payload.get('data')
                    if isinstance(data, list) and data:
                        return data
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX1D_History.csv'
        try:
            session = await self._get_session()
            async with session.get(url, timeout=self.timeout) as response:
                if response.status != 200:
                    self.logger.debug("CBOE VIX1D history request failed: %s", response.status)
                    return None
                text = await response.text()
        except Exception as exc:
            self.logger.debug("CBOE VIX1D fetch failed: %s", exc)
            return None

        reader = csv.DictReader(io.StringIO(text))
        history: List[Dict[str, Any]] = []
        for row in reader:
            if not row:
                continue
            try:
                history.append({
                    'date': row.get('DATE') or row.get('Date'),
                    'open': float(row.get('OPEN') or row.get('Open')),
                    'high': float(row.get('HIGH') or row.get('High')),
                    'low': float(row.get('LOW') or row.get('Low')),
                    'close': float(row.get('CLOSE') or row.get('Close')),
                })
            except (TypeError, ValueError):
                continue

        if not history:
            return None

        cache_payload = {
            'timestamp': time.time(),
            'data': history,
        }
        await self.redis.setex(self.cboe_history_cache_key, 4 * 3600, json.dumps(cache_payload))
        return history

    def _parse_cboe_date(self, date_str: str):
        for fmt in ('%m/%d/%Y', '%Y-%m-%d'):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unknown CBOE date format: {date_str}")

    async def _fetch_vix1d_ibkr(self) -> Optional[float]:
        try:
            raw = await self.redis.get('ibkr:market_data:VIX1D')
        except Exception as exc:
            self.logger.debug("IBKR VIX1D fetch failed: %s", exc)
            return None

        if not raw:
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None

        value = payload.get('last') or payload.get('close') or payload.get('value')
        ts = payload.get('timestamp')
        if value is None:
            return None
        if ts is not None:
            try:
                age = time.time() - float(ts)
            except (TypeError, ValueError):
                age = None
            else:
                if age is not None and age > 300:
                    return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def _fetch_vix1d_yahoo_with_backoff(self) -> Optional[float]:
        state = await self._load_yahoo_backoff_state()
        now = time.time()
        if state is not None:
            retries, last_ts = state
            backoff_seconds = min(3600, 60 * (2 ** retries))
            if now - last_ts < backoff_seconds:
                return None
        else:
            retries = self.yahoo_retry_count

        url = 'https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX1D?range=1d&interval=1m'
        try:
            session = await self._get_session()
            async with session.get(url, timeout=5) as response:
                if response.status == 429:
                    await self._store_yahoo_backoff_state(retries + 1, now)
                    self.yahoo_retry_count = retries + 1
                    return None
                response.raise_for_status()
                payload = await response.json()
        except Exception as exc:
            self.logger.debug("Yahoo VIX1D fetch failed: %s", exc)
            await self._store_yahoo_backoff_state(retries + 1, now)
            self.yahoo_retry_count = retries + 1
            return None

        result = (payload.get('chart') or {}).get('result') or []
        if not result:
            return None

        meta = result[0].get('meta') or {}
        value = meta.get('regularMarketPrice')
        if value is None:
            closes = result[0].get('indicators', {}).get('quote', [{}])[0].get('close') or []
            value = next((c for c in reversed(closes) if isinstance(c, (int, float))), None)
        if value is None:
            return None

        await self._clear_yahoo_backoff_state()
        self.yahoo_retry_count = 0
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def _load_yahoo_backoff_state(self) -> Optional[Tuple[int, float]]:
        raw = await self.redis.get(self.yahoo_backoff_key)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            return int(payload.get('retries', 1)), float(payload.get('ts', 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    async def _store_yahoo_backoff_state(self, retries: int, timestamp: float) -> None:
        payload = json.dumps({'retries': retries, 'ts': timestamp})
        await self.redis.setex(self.yahoo_backoff_key, 3600, payload)

    async def _clear_yahoo_backoff_state(self) -> None:
        await self.redis.delete(self.yahoo_backoff_key)

    async def _update_history(
        self,
        value: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[Dict[str, float], float, float, float]:
        history_key = f"{rkeys.analytics_vix1d_key()}:history"
        ts = int(timestamp if timestamp is not None else time.time())
        entry = json.dumps({'ts': ts, 'value': float(value)})

        async with self.redis.pipeline(transaction=False) as pipe:
            await pipe.lpush(history_key, entry)
            await pipe.ltrim(history_key, 0, self.history_window - 1)
            await pipe.expire(history_key, max(self.history_window * 5, 3600))
            await pipe.lrange(history_key, 0, self.history_window - 1)
            _, _, _, raw_history = await pipe.execute()

        timestamps: List[int] = []
        values: List[float] = []
        for raw in raw_history:
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='ignore')
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts = int(record.get('ts', 0))
            val = record.get('value')
            if isinstance(val, (int, float)):
                timestamps.append(ts)
                values.append(float(val))

        stats = {
            'zscore': 0.0,
            'mean': float(value),
            'stdev': 0.0,
            'samples': len(values),
            'window': self.history_window,
        }

        change_5m = 0.0
        change_1h = 0.0
        percentile = 0.5

        if values:
            mean = sum(values) / len(values)
            stats['mean'] = mean
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                stdev = math.sqrt(max(variance, 1e-12))
                stats['stdev'] = stdev
                if stdev > 0:
                    stats['zscore'] = (value - mean) / stdev

            percentile = sum(1 for v in values if v <= value) / len(values)

            now = int(timestamp if timestamp is not None else time.time())
            change_5m = self._compute_change(now, timestamps, values, 300, value)
            change_1h = self._compute_change(now, timestamps, values, 3600, value)

        return stats, change_5m, change_1h, percentile

    def _build_payload(
        self,
        *,
        value: float,
        history_stats: Dict[str, float],
        change_5m: float,
        change_1h: float,
        percentile: float,
        regime: str,
        as_of: datetime,
        source: str,
    ) -> Dict[str, Any]:
        return {
            'timestamp': int(as_of.timestamp() * 1000),
            'as_of': as_of.isoformat(),
            'value': value,
            'change_5m': change_5m,
            'change_1h': change_1h,
            'zscore': history_stats.get('zscore'),
            'mean': history_stats.get('mean'),
            'stdev': history_stats.get('stdev'),
            'samples': history_stats.get('samples'),
            'percentile': percentile,
            'regime': regime,
            'source': source,
            'thresholds': {
                'shock': self.shock_threshold,
                'elevated': self.elevated_threshold,
                'benign': self.benign_threshold,
            },
        }

    def _compute_change(
        self,
        now: int,
        timestamps: List[int],
        values: List[float],
        lookback_s: int,
        current_value: float,
    ) -> float:
        threshold = now - lookback_s
        for ts, val in zip(timestamps, values):
            if ts <= threshold:
                return current_value - val
        return 0.0

    def _classify_regime(self, value: float, zscore: float, change_5m: float, change_1h: float) -> str:
        if value >= self.shock_threshold or abs(change_5m) >= self.shock_change:
            return 'SHOCK'
        if value >= self.elevated_threshold or abs(change_1h) >= self.elevated_change or zscore >= 1.5:
            return 'ELEVATED'
        if value <= self.benign_threshold and zscore <= -1.0:
            return 'BENIGN'
        return 'NORMAL'
