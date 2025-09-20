"""Kelly sizing utilities for AlphaTraderPro execution layer.

Derives dynamic position sizing fractions from historical fills stored in
Redis. The estimator computes win probability, payoff ratio, and determines
an adjusted Kelly fraction per strategy while enforcing configured caps and
fallback defaults for sparse datasets (e.g. fresh paper trading accounts).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import redis.asyncio as aioredis


@dataclass
class KellyStats:
    """Aggregate performance statistics used for Kelly sizing."""

    sample_size: int = 0
    wins: int = 0
    losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    payoff_ratio: float = 0.0
    kelly_fraction: float = 0.0


class KellySizer:
    """Compute Kelly fractions from historical position outcomes."""

    def __init__(self, redis_conn: aioredis.Redis, config: Dict[str, Any]):
        self.redis = redis_conn
        self.config = config
        risk_cfg = (config.get('risk_management') or {}).get('kelly', {})
        self.lookback_days = int(risk_cfg.get('lookback_days', 20))
        self.min_trades = int(risk_cfg.get('min_trades', 15))
        self.base_fraction = float(risk_cfg.get('base_fraction', 0.02))
        self.max_fraction = float(risk_cfg.get('max_fraction', 0.15))
        self.max_buying_power_fraction = float(risk_cfg.get('max_buying_power_fraction', 0.25))
        self.refresh_interval = int(risk_cfg.get('refresh_interval_seconds', 300))
        self.lookback_seconds = self.lookback_days * 86400

        self._stats: Dict[str, KellyStats] = {}
        self._last_refresh = 0.0
        self._refresh_lock = asyncio.Lock()

    async def ensure_stats(self) -> None:
        """Refresh cached Kelly statistics if the cache is stale."""
        async with self._refresh_lock:
            now = time.time()
            if now - self._last_refresh < self.refresh_interval:
                return
            await self._refresh_stats()
            self._last_refresh = now

    async def invalidate(self) -> None:
        """Force stats to refresh on next request."""
        self._last_refresh = 0.0

    def stats_for(self, strategy: Optional[str]) -> KellyStats:
        key = (strategy or 'GLOBAL').upper() or 'GLOBAL'
        return self._stats.get(key, self._stats.get('GLOBAL', KellyStats()))

    async def suggest_notional(
        self,
        *,
        strategy: Optional[str],
        confidence: float,
        account_value: float,
        buying_power: float,
    ) -> Tuple[float, KellyStats]:
        """Return target notional exposure and raw stats for diagnostics."""
        await self.ensure_stats()
        stats = self.stats_for(strategy)

        # Fallback to base fraction when data is sparse or Kelly is zero/negative.
        fraction = stats.kelly_fraction if stats.sample_size >= self.min_trades else 0.0
        if fraction <= 0:
            fraction = self.base_fraction

        # Confidence dampening (confidence in range 0-100).
        confidence = max(0.0, min(confidence, 100.0))
        confidence_scalar = 0.25 + 0.75 * (confidence / 100.0)
        fraction *= confidence_scalar

        fraction = max(0.0, min(fraction, self.max_fraction))

        notional = fraction * account_value
        bp_cap = buying_power * self.max_buying_power_fraction
        if bp_cap > 0:
            notional = min(notional, bp_cap)

        return notional, stats

    async def _refresh_stats(self) -> None:
        now_ts = time.time()
        lower_bound = now_ts - self.lookback_seconds

        raw_keys = await self.redis.zrangebyscore('positions:closed:index', lower_bound, now_ts)
        keys = [key.decode('utf-8') if isinstance(key, bytes) else key for key in raw_keys]

        fallback_used = False
        if not keys:
            keys = await self._scan_closed_keys(lower_bound)
            fallback_used = bool(keys)

        if not keys:
            self._stats = {}
            return

        payloads = await self.redis.mget(*keys)

        agg: Dict[str, List[float]] = defaultdict(list)
        losses: Dict[str, List[float]] = defaultdict(list)
        agg['GLOBAL'] = []
        losses['GLOBAL'] = []

        index_updates: Dict[str, float] = {}

        for key, blob in zip(keys, payloads):
            if not blob:
                continue
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError:
                continue

            strategy = str(payload.get('strategy') or 'GLOBAL').upper()
            realized = _safe_float(payload.get('realized_pnl'))
            notional = _safe_float(payload.get('position_notional'))
            if notional <= 0:
                entry_price = _safe_float(payload.get('entry_price'))
                qty = _safe_float(payload.get('quantity'), allow_zero=True)
                multiplier = 100.0 if _is_option(payload) else 1.0
                notional = abs(entry_price * qty * multiplier)
            if notional <= 0:
                continue

            ret = realized / notional
            if math.isnan(ret) or math.isinf(ret):
                continue

            agg['GLOBAL'].append(ret)
            if ret < 0:
                losses['GLOBAL'].append(abs(ret))

            agg[strategy].append(ret)
            if ret < 0:
                losses[strategy].append(abs(ret))

            if fallback_used:
                ts = _extract_exit_timestamp(payload, default=now_ts)
                index_updates[key] = ts

        if fallback_used and index_updates:
            await self.redis.zadd('positions:closed:index', index_updates)

        computed: Dict[str, KellyStats] = {}
        for key, returns in agg.items():
            if not returns:
                continue
            wins = [r for r in returns if r > 0]
            loss_values = losses.get(key, []) or [abs(r) for r in returns if r < 0]
            stats = KellyStats()
            stats.sample_size = len(returns)
            stats.wins = len(wins)
            stats.losses = len(loss_values)
            stats.avg_win = sum(wins) / len(wins) if wins else 0.0
            stats.avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
            stats.win_rate = stats.wins / stats.sample_size if stats.sample_size else 0.0

            if stats.avg_loss <= 0 or stats.win_rate <= 0:
                stats.payoff_ratio = 0.0
                stats.kelly_fraction = 0.0
            else:
                stats.payoff_ratio = stats.avg_win / stats.avg_loss if stats.avg_loss else 0.0
                q = 1.0 - stats.win_rate
                b = stats.payoff_ratio
                if b <= 0:
                    stats.kelly_fraction = 0.0
                else:
                    stats.kelly_fraction = max(0.0, (stats.win_rate - q / b))

            computed[key] = stats

        self._stats = computed

    async def _scan_closed_keys(self, window_start_ts: float) -> List[str]:
        cutoff_date = datetime.fromtimestamp(window_start_ts, timezone.utc).date()
        today = datetime.now(timezone.utc).date()
        results: List[str] = []

        day = cutoff_date
        while day <= today:
            pattern = f'positions:closed:{day.strftime("%Y%m%d")}:*'
            cursor = b'0'
            while True:
                cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=500)
                for key in keys:
                    results.append(key.decode('utf-8') if isinstance(key, bytes) else key)
                if cursor == b'0':
                    break
            day += timedelta(days=1)

        return results


def _safe_float(value: Any, *, allow_zero: bool = False) -> float:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not allow_zero and fval == 0.0:
        return 0.0
    if math.isnan(fval) or math.isinf(fval):
        return 0.0
    return fval


def _extract_exit_timestamp(payload: Dict[str, Any], default: float) -> float:
    exit_time = payload.get('exit_time') or payload.get('timestamp')
    if exit_time:
        try:
            return datetime.fromisoformat(str(exit_time).replace('Z', '+00:00')).timestamp()
        except ValueError:
            pass
    return default


def _is_option(position_payload: Dict[str, Any]) -> bool:
    contract = position_payload.get('contract') or {}
    sec_type = str(contract.get('type') or contract.get('secType') or '').lower()
    return sec_type in {'opt', 'option', 'options'}
