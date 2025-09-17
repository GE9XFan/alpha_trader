#!/usr/bin/env python3
"""Volatility metrics ingestion and analytics (VIX1D integration)."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
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
        self.ttl = int(store_ttls.get('analytics', 60))

        vol_cfg = analytics_cfg.get('volatility', {})
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

    async def update_vix1d(self) -> Dict[str, Any]:
        value = await self._fetch_vix1d()
        if value is None:
            cached = await self.redis.get(rkeys.analytics_vix1d_key())
            if cached:
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    pass
            return {'error': 'unavailable'}

        history_stats, change_5m, change_1h, percentile = await self._update_history(value)
        regime = self._classify_regime(value, history_stats.get('zscore', 0.0), change_5m, change_1h)

        payload = {
            'timestamp': int(time.time() * 1000),
            'as_of': datetime.now(timezone.utc).isoformat(),
            'value': value,
            'change_5m': change_5m,
            'change_1h': change_1h,
            'zscore': history_stats.get('zscore'),
            'mean': history_stats.get('mean'),
            'stdev': history_stats.get('stdev'),
            'samples': history_stats.get('samples'),
            'percentile': percentile,
            'regime': regime,
            'thresholds': {
                'shock': self.shock_threshold,
                'elevated': self.elevated_threshold,
                'benign': self.benign_threshold,
            },
        }

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

    async def _fetch_vix1d(self) -> Optional[float]:
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX1D?range=1d&interval=1m'
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                response.raise_for_status()
                payload = await response.json()
        except Exception as exc:
            self.logger.debug("Failed to fetch VIX1D: %s", exc)
            return None

        try:
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
            return float(value)
        except Exception:
            return None

    async def _update_history(self, value: float) -> Tuple[Dict[str, float], float, float, float]:
        history_key = f"{rkeys.analytics_vix1d_key()}:history"
        entry = json.dumps({'ts': int(time.time()), 'value': float(value)})

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

            now = int(time.time())
            change_5m = self._compute_change(now, timestamps, values, 300, value)
            change_1h = self._compute_change(now, timestamps, values, 3600, value)

        return stats, change_5m, change_1h, percentile

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
