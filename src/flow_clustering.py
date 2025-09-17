#!/usr/bin/env python3
"""Flow clustering analytics for intraday trade classification."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as aioredis
from sklearn.cluster import KMeans

import redis_keys as rkeys


@dataclass
class TradeFeature:
    price: float
    size: float
    timestamp: float
    direction: float
    notional: float
    sweep: float


class FlowClusterModel:
    """Cluster recent trades to infer strategy archetypes and participant mix."""

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        analytics_cfg = config.get('modules', {}).get('analytics', {})
        store_ttls = analytics_cfg.get('store_ttls', {})
        self.ttl = int(store_ttls.get('analytics', 60))

        cluster_cfg = analytics_cfg.get('flow_clustering', {})
        self.window_trades = int(cluster_cfg.get('window_trades', 150))
        self.min_trades = int(cluster_cfg.get('min_trades', 30))
        self.institutional_size = float(cluster_cfg.get('institutional_size', 250.0))
        self.seed = int(cluster_cfg.get('random_state', 42))

    async def classify_flows(self, symbol: str) -> Dict[str, Any]:
        trades = await self._load_trades(symbol)
        if len(trades) < self.min_trades:
            payload = self._empty_payload(symbol)
            await self.redis.setex(rkeys.analytics_flow_clusters_key(symbol), self.ttl, json.dumps(payload))
            return payload

        features = self._build_feature_matrix(trades)
        if features.size == 0:
            payload = self._empty_payload(symbol)
            await self.redis.setex(rkeys.analytics_flow_clusters_key(symbol), self.ttl, json.dumps(payload))
            return payload

        assignments, centroids = self._cluster(features)
        stats = self._compute_cluster_stats(trades, features, assignments)
        payload = self._build_payload(symbol, trades, assignments, centroids, stats)

        await self.redis.setex(rkeys.analytics_flow_clusters_key(symbol), self.ttl, json.dumps(payload))
        return payload

    async def _load_trades(self, symbol: str) -> List[TradeFeature]:
        try:
            raw_trades = await self.redis.lrange(rkeys.market_trades_key(symbol), -self.window_trades, -1)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Failed to load trades for %s: %s", symbol, exc)
            return []

        trades: List[TradeFeature] = []
        last_price = None
        last_ts = None

        for raw in raw_trades:
            trade = self._decode(raw)
            if not isinstance(trade, dict):
                continue

            price = self._to_float(trade.get('price') or trade.get('last') or trade.get('p'))
            size = self._to_float(trade.get('size') or trade.get('qty') or trade.get('volume') or trade.get('s'))
            ts = trade.get('timestamp') or trade.get('ts')
            if ts is None:
                ts = trade.get('time')
            ts = self._to_float(ts)

            if price is None or price <= 0 or size is None or size <= 0 or ts is None:
                continue

            direction_field = trade.get('side') or trade.get('direction') or trade.get('taker_side')
            direction = 0.0
            if isinstance(direction_field, str):
                if direction_field.lower() in {'buy', 'b', 'bid'}:
                    direction = 1.0
                elif direction_field.lower() in {'sell', 's', 'ask'}:
                    direction = -1.0
            elif isinstance(direction_field, (int, float)):
                direction = 1.0 if float(direction_field) > 0 else -1.0

            if direction == 0.0 and trade.get('buyer_maker') is not None:
                direction = -1.0 if trade['buyer_maker'] else 1.0

            if direction == 0.0:
                # Infer from price movement relative to last trade
                if last_price is not None:
                    direction = 1.0 if price >= last_price else -1.0
                else:
                    direction = 1.0

            sweep = float(trade.get('sweep', 0) or trade.get('is_sweep', 0) or 0)
            notional = price * size

            trades.append(TradeFeature(
                price=price,
                size=size,
                timestamp=ts,
                direction=direction,
                notional=notional,
                sweep=sweep,
            ))

            last_price = price
            last_ts = ts

        return trades

    def _build_feature_matrix(self, trades: List[TradeFeature]) -> np.ndarray:
        if not trades:
            return np.empty((0, 6))

        features: List[List[float]] = []
        prev_ts = trades[0].timestamp
        prev_price = trades[0].price

        for trade in trades:
            time_gap = max(trade.timestamp - prev_ts, 0.0)
            price_change = trade.price - prev_price
            volatility_scaled = abs(price_change) / max(prev_price, 1e-6)

            features.append([
                math.log1p(trade.size),
                trade.direction,
                volatility_scaled,
                math.log1p(abs(trade.notional)),
                trade.sweep,
                math.log1p(time_gap + 1e-6),
            ])

            prev_ts = trade.timestamp
            prev_price = trade.price

        return np.asarray(features, dtype=float)

    def _cluster(self, features: np.ndarray) -> (np.ndarray, np.ndarray):
        kmeans = KMeans(n_clusters=3, random_state=self.seed, n_init='auto', max_iter=100)
        assignments = kmeans.fit_predict(features)
        return assignments, kmeans.cluster_centers_

    def _compute_cluster_stats(
        self,
        trades: List[TradeFeature],
        features: np.ndarray,
        assignments: np.ndarray,
    ) -> List[Dict[str, Any]]:
        stats: List[Dict[str, Any]] = []
        for cluster_id in range(3):
            mask = assignments == cluster_id
            if not np.any(mask):
                stats.append({
                    'count': 0,
                    'mean_direction': 0.0,
                    'mean_abs_return': 0.0,
                    'mean_notional': 0.0,
                    'mean_size': 0.0,
                    'mean_time_gap': 0.0,
                    'sweep_ratio': 0.0,
                })
                continue

            subset = features[mask]
            trade_subset = [trades[i] for i, flag in enumerate(mask) if flag]

            mean_direction = float(np.mean(subset[:, 1]))
            mean_abs_return = float(np.mean(subset[:, 2]))
            mean_notional = float(np.mean([t.notional for t in trade_subset]))
            mean_size = float(np.mean([t.size for t in trade_subset]))
            mean_time_gap = float(np.mean(subset[:, 5]))
            sweep_ratio = float(np.mean([1.0 if t.sweep else 0.0 for t in trade_subset]))

            stats.append({
                'count': int(mask.sum()),
                'mean_direction': mean_direction,
                'mean_abs_return': mean_abs_return,
                'mean_notional': mean_notional,
                'mean_size': mean_size,
                'mean_time_gap': mean_time_gap,
                'sweep_ratio': sweep_ratio,
            })

        return stats

    def _build_payload(
        self,
        symbol: str,
        trades: List[TradeFeature],
        assignments: np.ndarray,
        centroids: np.ndarray,
        stats: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total = max(len(trades), 1)

        # Determine strategy mapping heuristically
        momentum_idx = max(range(3), key=lambda idx: abs(stats[idx]['mean_direction']) * (stats[idx]['mean_abs_return'] + 1e-6))
        hedging_idx = max(range(3), key=lambda idx: stats[idx]['mean_notional'] * (1.0 - min(abs(stats[idx]['mean_direction']), 1.0)))
        remaining = {0, 1, 2} - {momentum_idx, hedging_idx}
        mean_rev_idx = remaining.pop() if remaining else momentum_idx

        strategy_counts = {
            'momentum': stats[momentum_idx]['count'],
            'mean_reversion': stats[mean_rev_idx]['count'],
            'hedging': stats[hedging_idx]['count'],
        }

        strategy_distribution = {
            name: (count / total if total else 0.0)
            for name, count in strategy_counts.items()
        }

        institutional = sum(
            stats[idx]['count']
            for idx in range(3)
            if stats[idx]['mean_size'] >= self.institutional_size
        )
        retail = total - institutional
        participant_distribution = {
            'institutional': institutional / total if total else 0.0,
            'retail': retail / total if total else 0.0,
        }

        payload = {
            'symbol': symbol,
            'timestamp': time.time(),
            'samples': total,
            'strategy_distribution': strategy_distribution,
            'participant_distribution': participant_distribution,
            'cluster_stats': stats,
            'centroids': centroids.tolist(),
        }
        return payload

    def _empty_payload(self, symbol: str) -> Dict[str, Any]:
        return {
            'symbol': symbol,
            'timestamp': time.time(),
            'samples': 0,
            'strategy_distribution': {
                'momentum': 0.0,
                'mean_reversion': 0.0,
                'hedging': 0.0,
            },
            'participant_distribution': {
                'institutional': 0.0,
                'retail': 0.0,
            },
            'cluster_stats': [],
            'centroids': [],
        }

    def _decode(self, raw: Any) -> Any:
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

    def _to_float(self, value: Any) -> Optional[float]:
        if value in (None, '', 'null'):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
