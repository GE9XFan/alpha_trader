#!/usr/bin/env python3
"""
Analytics Engine - Main Analytics Coordinator
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- market:{symbol}:ticker: Market data
- market:{symbol}:trades: Trade data
- market:{symbol}:book: Order book data
- analytics:{symbol}:*: Symbol analytics outputs
- analytics:portfolio:*: Portfolio and correlation artifacts
- discovered:*: Discovered parameters
- module:heartbeat:analytics: Heartbeat status
"""

import pandas as pd
import json
import redis.asyncio as aioredis
import time
import asyncio
import pytz
from datetime import datetime, time as datetime_time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import traceback

from logging_utils import get_logger

import redis_keys as rkeys
from redis_keys import get_ttl
from vpin_calculator import VPINCalculator
from gex_dex_calculator import GEXDEXCalculator
from pattern_analyzer import PatternAnalyzer
from dealer_flow_calculator import DealerFlowCalculator
from flow_clustering import FlowClusterModel
from volatility_metrics import VolatilityMetrics


class MetricsAggregator:
    """Aggregate symbol analytics into portfolio, sector, and correlation views."""

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        self.config = config
        self.redis = redis_conn
        self.logger = get_logger(__name__, component="analytics", subsystem="metrics")

        symbols_cfg = config.get('symbols', {})
        self.symbols = sorted({*symbols_cfg.get('standard', []), *symbols_cfg.get('level2', [])})

        analytics_cfg = config.get('modules', {}).get('analytics', {})
        store_ttls = analytics_cfg.get('store_ttls', {})
        self.ttls = {
            'symbol': store_ttls.get('analytics', get_ttl('analytics')),
            'portfolio': store_ttls.get('portfolio', get_ttl('analytics_portfolio')),
            'sector': store_ttls.get('sector', get_ttl('analytics_sector')),
            'correlation': store_ttls.get('correlation', get_ttl('analytics_correlation')),
        }

        sector_map_cfg = analytics_cfg.get('sector_map', {})
        self.sector_map = {symbol: sector_map_cfg.get(symbol, 'OTHER') for symbol in self.symbols}

        tox_cfg = config.get('parameter_discovery', {}).get('toxicity_detection', {}).get('vpin_thresholds', {})
        self.toxic_threshold = float(tox_cfg.get('toxic', 0.7))

        corr_cfg = config.get('parameter_discovery', {}).get('correlation', {})
        self.correlation_window = int(corr_cfg.get('calculation_window', 500))
        self.min_correlation_symbols = int(corr_cfg.get('min_correlation_symbols', 3))
        risk_cfg = config.get('risk_management', {})
        self.correlation_threshold = float(risk_cfg.get('correlation_limit', 0.7))

        self._last_snapshots: Dict[str, Dict[str, Any]] = {}

    async def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Compute and persist portfolio-wide analytics aggregates."""
        snapshots = await self._gather_symbol_snapshots()
        self._last_snapshots = {snap['symbol']: snap for snap in snapshots}

        vpins = [snap['vpin'] for snap in snapshots if snap.get('vpin') is not None]
        max_symbol = None
        max_vpin = 0.0
        min_vpin = 1.0 if vpins else 0.0

        for snap in snapshots:
            vpin_value = snap.get('vpin')
            if vpin_value is None:
                continue
            if vpin_value > max_vpin:
                max_vpin = vpin_value
                max_symbol = snap['symbol']
            if vpin_value < min_vpin:
                min_vpin = vpin_value

        metrics: Dict[str, Any] = {
            'timestamp': time.time(),
            'symbol_count': len(snapshots),
            'avg_vpin': round(sum(vpins) / len(vpins), 4) if vpins else None,
            'max_vpin': round(max_vpin, 4) if vpins else None,
            'min_vpin': round(min_vpin, 4) if vpins else None,
            'max_vpin_symbol': max_symbol,
            'toxic_count': sum(1 for snap in snapshots if (snap.get('vpin') or 0) >= self.toxic_threshold),
            'total_gex': float(sum(snap.get('gex', 0.0) for snap in snapshots)),
            'total_dex': float(sum(snap.get('dex', 0.0) for snap in snapshots)),
            'total_vanna_notional': float(sum(snap.get('vanna_notional', 0.0) or 0.0 for snap in snapshots)),
            'total_charm_notional': float(sum(snap.get('charm_notional', 0.0) or 0.0 for snap in snapshots)),
            'total_hedging_notional': float(sum(snap.get('hedging_notional', 0.0) or 0.0 for snap in snapshots)),
            'sector_flows': {},
            'vix1d': snapshots[0].get('vix1d') if snapshots else None,
        }

        metrics['sector_flows'] = self._build_sector_summary(snapshots)

        await self.redis.setex(
            rkeys.analytics_portfolio_summary_key(),
            self.ttls['portfolio'],
            json.dumps(metrics)
        )

        return metrics

    async def calculate_sector_flows(self) -> Dict[str, Any]:
        """Persist sector-level aggregates derived from the most recent snapshots."""
        snapshots = list(self._last_snapshots.values()) or await self._gather_symbol_snapshots()
        summary = self._build_sector_summary(snapshots)
        ttl = self.ttls['sector']

        for sector, payload in summary.items():
            await self.redis.setex(rkeys.analytics_sector_key(sector), ttl, json.dumps(payload))

        return summary

    async def calculate_cross_asset_correlations(self) -> Dict[str, Any]:
        """Publish a correlation matrix using discovered data or on-the-fly calculations."""
        payload: Dict[str, Any] = {
            'timestamp': time.time(),
            'source': None,
            'matrix': {},
            'pair_count': 0,
            'high_pairs': [],
        }

        discovered = await self._fetch_json('discovered:correlation_matrix')
        matrix: Optional[Dict[str, Dict[str, float]]] = None

        if discovered:
            payload['source'] = 'discovered'
            matrix = self._sanitize_correlation_matrix(discovered)
        else:
            payload['source'] = 'computed'
            matrix = await self._compute_correlation_from_bars()

        if not matrix:
            payload['source'] = payload['source'] or 'unavailable'
        else:
            payload['matrix'] = matrix
            symbols = sorted(matrix.keys())
            seen_pairs = set()
            for base in symbols:
                for peer, value in matrix.get(base, {}).items():
                    if base == peer:
                        continue
                    pair = tuple(sorted((base, peer)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    payload['pair_count'] += 1
                    if abs(value) >= self.correlation_threshold:
                        payload['high_pairs'].append({
                            'pair': list(pair),
                            'correlation': value,
                        })

        await self.redis.setex(
            rkeys.analytics_portfolio_correlation_key(),
            self.ttls['correlation'],
            json.dumps(payload)
        )

        return payload

    async def _gather_symbol_snapshots(self) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        vix1d_data = await self._fetch_json(rkeys.analytics_vix1d_key())

        for symbol in self.symbols:
            snapshot: Dict[str, Any] = {
                'symbol': symbol,
                'sector': self.sector_map.get(symbol, 'OTHER'),
                'timestamp': time.time(),
                'vpin': None,
                'gex': 0.0,
                'dex': 0.0,
                'call_dex': 0.0,
                'put_dex': 0.0,
                'volume': 0.0,
                'vanna_notional': 0.0,
                'vanna_z': 0.0,
                'charm_notional': 0.0,
                'charm_z': 0.0,
                'hedging_notional': 0.0,
                'hedging_shares_per_pct': 0.0,
                'skew': None,
                'skew_z': 0.0,
                'flow_clusters': {},
                'vix1d': vix1d_data,
            }

            vpin_data = await self._fetch_json(rkeys.analytics_vpin_key(symbol))
            if vpin_data:
                try:
                    snapshot['vpin'] = float(vpin_data.get('value'))
                except (TypeError, ValueError):
                    snapshot['vpin'] = None

            gex_data = await self._fetch_json(rkeys.analytics_gex_key(symbol))
            if gex_data:
                snapshot['gex'] = float(gex_data.get('total_gex', 0.0))

            dex_data = await self._fetch_json(rkeys.analytics_dex_key(symbol))
            if dex_data:
                snapshot['dex'] = float(dex_data.get('total_dex', 0.0))
                snapshot['call_dex'] = float(dex_data.get('call_dex', 0.0))
                snapshot['put_dex'] = float(dex_data.get('put_dex', 0.0))

            vanna_data = await self._fetch_json(rkeys.analytics_vanna_key(symbol))
            if vanna_data:
                snapshot['vanna'] = vanna_data
                snapshot['vanna_notional'] = float(vanna_data.get('total_vanna_notional_per_pct_vol', 0.0))
                history = vanna_data.get('history') or {}
                try:
                    snapshot['vanna_z'] = float(history.get('zscore', 0.0))
                except (TypeError, ValueError):
                    snapshot['vanna_z'] = 0.0

            charm_data = await self._fetch_json(rkeys.analytics_charm_key(symbol))
            if charm_data:
                snapshot['charm'] = charm_data
                snapshot['charm_notional'] = float(charm_data.get('total_charm_notional_per_day', 0.0))
                history = charm_data.get('history') or {}
                try:
                    snapshot['charm_z'] = float(history.get('zscore', 0.0))
                except (TypeError, ValueError):
                    snapshot['charm_z'] = 0.0

            hedging_data = await self._fetch_json(rkeys.analytics_hedging_impact_key(symbol))
            if hedging_data:
                snapshot['hedging'] = hedging_data
                snapshot['hedging_notional'] = float(hedging_data.get('notional_per_pct_move', 0.0))
                snapshot['hedging_shares_per_pct'] = float(hedging_data.get('shares_per_pct_move', 0.0))

            skew_data = await self._fetch_json(rkeys.analytics_skew_key(symbol))
            if skew_data:
                snapshot['skew'] = skew_data.get('skew')
                history = skew_data.get('history') or {}
                try:
                    snapshot['skew_z'] = float(history.get('zscore', 0.0))
                except (TypeError, ValueError):
                    snapshot['skew_z'] = 0.0
                snapshot['skew_data'] = skew_data

            clusters_data = await self._fetch_json(rkeys.analytics_flow_clusters_key(symbol))
            if clusters_data:
                snapshot['flow_clusters'] = clusters_data

            ticker = await self._fetch_json(rkeys.market_ticker_key(symbol))
            if ticker:
                try:
                    snapshot['volume'] = float(ticker.get('volume') or ticker.get('vol') or 0.0)
                except (TypeError, ValueError):
                    snapshot['volume'] = 0.0

            snapshots.append(snapshot)

        return snapshots

    def _build_sector_summary(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}

        for snap in snapshots:
            sector = snap.get('sector', 'OTHER')
            data = summary.setdefault(sector, {
                'symbols': [],
                'avg_vpin': [],
                'total_volume': 0.0,
                'total_gex': 0.0,
                'total_dex': 0.0,
                'call_dex': 0.0,
                'put_dex': 0.0,
                'toxic_symbols': [],
                'timestamp': time.time(),
                'total_vanna_notional': 0.0,
                'total_charm_notional': 0.0,
                'total_hedging_notional': 0.0,
                'hedging_shares_per_pct': 0.0,
                'skew_values': [],
                'momentum_probs': [],
                'hedging_probs': [],
            })

            data['symbols'].append(snap['symbol'])
            data['total_volume'] += snap.get('volume', 0.0) or 0.0
            data['total_gex'] += snap.get('gex', 0.0) or 0.0
            data['total_dex'] += snap.get('dex', 0.0) or 0.0
            data['call_dex'] += snap.get('call_dex', 0.0) or 0.0
            data['put_dex'] += snap.get('put_dex', 0.0) or 0.0
            data['total_vanna_notional'] += snap.get('vanna_notional', 0.0) or 0.0
            data['total_charm_notional'] += snap.get('charm_notional', 0.0) or 0.0
            data['total_hedging_notional'] += snap.get('hedging_notional', 0.0) or 0.0
            data['hedging_shares_per_pct'] += snap.get('hedging_shares_per_pct', 0.0) or 0.0

            vpin_value = snap.get('vpin')
            if vpin_value is not None:
                data['avg_vpin'].append(vpin_value)
                if vpin_value >= self.toxic_threshold:
                    data['toxic_symbols'].append(snap['symbol'])

            skew_value = snap.get('skew')
            if isinstance(skew_value, (int, float)):
                data['skew_values'].append(float(skew_value))

            clusters = snap.get('flow_clusters') or {}
            strategy_dist = clusters.get('strategy_distribution') or {}
            if strategy_dist:
                momentum = strategy_dist.get('momentum')
                hedging = strategy_dist.get('hedging')
                if isinstance(momentum, (int, float)):
                    data['momentum_probs'].append(float(momentum))
                if isinstance(hedging, (int, float)):
                    data['hedging_probs'].append(float(hedging))

        for sector, data in summary.items():
            vpins = data.pop('avg_vpin')
            avg_vpin = sum(vpins) / len(vpins) if vpins else None
            data['avg_vpin'] = round(avg_vpin, 4) if avg_vpin is not None else None
            data['symbol_count'] = len(data['symbols'])
            data['total_volume'] = float(data['total_volume'])
            data['total_gex'] = float(data['total_gex'])
            data['total_dex'] = float(data['total_dex'])
            data['call_dex'] = float(data['call_dex'])
            data['put_dex'] = float(data['put_dex'])
            data['net_flow'] = float(data['total_dex'])
            data['toxic_symbols'] = sorted(set(data['toxic_symbols']))
            skew_values = data.pop('skew_values')
            if skew_values:
                data['avg_skew'] = sum(skew_values) / len(skew_values)
            else:
                data['avg_skew'] = None
            momentum_probs = data.pop('momentum_probs')
            hedging_probs = data.pop('hedging_probs')
            data['avg_momentum_prob'] = sum(momentum_probs) / len(momentum_probs) if momentum_probs else None
            data['avg_hedging_prob'] = sum(hedging_probs) / len(hedging_probs) if hedging_probs else None
            data['total_vanna_notional'] = float(data['total_vanna_notional'])
            data['total_charm_notional'] = float(data['total_charm_notional'])
            data['total_hedging_notional'] = float(data['total_hedging_notional'])
            data['hedging_shares_per_pct'] = float(data['hedging_shares_per_pct'])

        return summary

    async def _fetch_json(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = await self.redis.get(key)
        except Exception as exc:
            self.logger.error(f"Failed to fetch {key}: {exc}")
            return None

        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON payload for {key}")
            return None

    def _sanitize_correlation_matrix(self, matrix: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        sanitized: Dict[str, Dict[str, float]] = {}

        for symbol, row in matrix.items():
            if not isinstance(row, dict):
                continue
            sanitized[symbol] = {}
            for peer, value in row.items():
                try:
                    sanitized[symbol][peer] = round(float(value), 3)
                except (TypeError, ValueError):
                    sanitized[symbol][peer] = 0.0

        return sanitized

    async def _compute_correlation_from_bars(self) -> Optional[Dict[str, Dict[str, float]]]:
        price_returns: Dict[str, pd.Series] = {}

        for symbol in self.symbols:
            bars_raw = await self.redis.lrange(rkeys.market_bars_key(symbol, '1min'), -self.correlation_window, -1)
            if not bars_raw:
                bars_raw = await self.redis.lrange(rkeys.market_bars_key(symbol), -self.correlation_window, -1)
            if not bars_raw:
                continue

            closes: List[float] = []
            for raw_entry in bars_raw:
                if isinstance(raw_entry, bytes):
                    raw_entry = raw_entry.decode('utf-8', errors='ignore')
                try:
                    entry = json.loads(raw_entry)
                except json.JSONDecodeError:
                    continue

                close = None
                if isinstance(entry, dict):
                    close = entry.get('close') or entry.get('c')
                elif isinstance(entry, (list, tuple)):
                    close = entry[4] if len(entry) >= 5 else None
                if close is None:
                    continue
                try:
                    closes.append(float(close))
                except (TypeError, ValueError):
                    continue

            if len(closes) < 2:
                continue

            series = pd.Series(closes).pct_change().dropna()
            if series.empty:
                continue

            price_returns[symbol] = series

        if len(price_returns) < self.min_correlation_symbols:
            return None

        frame = pd.DataFrame(price_returns).dropna()
        if frame.shape[0] < 2:
            return None

        corr = frame.corr()
        matrix: Dict[str, Dict[str, float]] = {}

        for symbol in corr.columns:
            matrix[symbol] = {}
            for peer in corr.columns:
                value = corr.at[symbol, peer]
                if pd.isna(value):
                    continue
                matrix[symbol][peer] = round(float(value), 3)

        return matrix if matrix else None


class AnalyticsEngine:
    """
    Main analytics engine that coordinates all calculations and updates.
    Processes market data and produces trading signals.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """
        Initialize analytics engine with async Redis.
        Note: Expects async Redis connection for all operations.
        """
        self.config = config
        self.redis = redis_conn  # This is async Redis
        self.logger = get_logger(__name__, component="analytics", subsystem="engine")

        # Load symbols from config - combine standard and level2
        syms = config.get('symbols', {})
        self.symbols = sorted({*syms.get('standard', []), *syms.get('level2', [])})

        analytics_cfg = config.get('modules', {}).get('analytics', {})

        default_ttls = {
            'analytics': get_ttl('analytics'),
            'portfolio': get_ttl('analytics_portfolio'),
            'sector': get_ttl('analytics_sector'),
            'correlation': get_ttl('analytics_correlation'),
            'heartbeat': get_ttl('heartbeat'),
        }
        store_ttls = analytics_cfg.get('store_ttls', {})
        self.ttls = {**default_ttls, **store_ttls}

        default_intervals = {
            'vpin': 5,
            'gex_dex': 30,
            'flow_toxicity': 60,
            'order_book': 1,
            'sweep_detection': 2,
            'portfolio': 15,
            'sectors': 30,
            'correlation': 300,
            'dealer_flows': 60,
            'flow_clustering': 90,
            'volatility': 60,
        }
        configured_intervals = analytics_cfg.get('update_intervals', {})
        self.update_intervals = {**default_intervals, **configured_intervals}

        signals_cfg = config.get('modules', {}).get('signals', {})
        cluster_symbols = signals_cfg.get('strategies', {}).get('0dte', {}).get('symbols', self.symbols)
        if isinstance(cluster_symbols, list):
            self.flow_cluster_symbols = sorted({s for s in cluster_symbols if isinstance(s, str)})
        else:
            self.flow_cluster_symbols = list(self.symbols)

        # Track last update times and error counts
        self.last_updates: Dict[str, float] = defaultdict(float)
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Scheduler cadence and runtime behaviour
        self.cadence_hz = max(float(analytics_cfg.get('cadence_hz', 2) or 0), 0.0)
        self.analytics_rth_only = bool(analytics_cfg.get('analytics_rth_only', False))
        self._sleep_interval = (1.0 / self.cadence_hz) if self.cadence_hz > 0 else 0.5
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

        self.market_extended_hours = bool(config.get('market', {}).get('extended_hours', False))

        # Market hours
        self.market_tz = pytz.timezone('US/Eastern')

        # Initialize calculator instances (will be imported from separate modules)
        self.vpin_calculator: Optional[VPINCalculator] = None
        self.gex_dex_calculator: Optional[GEXDEXCalculator] = None
        self.pattern_analyzer: Optional[PatternAnalyzer] = None
        self.dealer_flow_calculator: Optional[DealerFlowCalculator] = None
        self.flow_cluster_model: Optional[FlowClusterModel] = None
        self.volatility_metrics: Optional[VolatilityMetrics] = None
        self.flow_cluster_symbols: List[str] = []
        self.aggregator = MetricsAggregator(config, redis_conn)

        self.logger.info(
            "analytics_engine_initialized",
            extra={"action": "init", "symbol_count": len(self.symbols)}
        )

    async def start(self):
        """Start the analytics engine."""
        if self._running:
            self.logger.warning("Analytics engine already running")
            return

        self.logger.info("analytics_engine_start", extra={"action": "start"})

        try:
            self.vpin_calculator = VPINCalculator(self.config, self.redis)
            self.gex_dex_calculator = GEXDEXCalculator(self.config, self.redis)
            self.pattern_analyzer = PatternAnalyzer(self.config, self.redis)
            self.dealer_flow_calculator = DealerFlowCalculator(self.config, self.redis)
            self.flow_cluster_model = FlowClusterModel(self.config, self.redis)
            self.volatility_metrics = VolatilityMetrics(self.config, self.redis)

            self._running = True
            self._loop_task = asyncio.create_task(self._calculation_loop(), name="analytics_calculation_loop")
            await self._loop_task

        except asyncio.CancelledError:
            self.logger.info("analytics_engine_start_cancelled", extra={"action": "start", "status": "cancelled"})
            raise
        except Exception as exc:
            self.logger.error(f"Error starting analytics engine: {exc}")
            self.logger.error(traceback.format_exc())
        finally:
            self._running = False
            self._loop_task = None

    async def _calculation_loop(self):
        """Main loop that triggers calculations based on configured intervals."""
        self.logger.info("analytics_loop_start", extra={"action": "loop_start"})

        try:
            while self._running:
                try:
                    current_time = time.time()

                    if self.analytics_rth_only and not self.is_market_hours(self.market_extended_hours):
                        await self._update_heartbeat(idle=True)
                        await asyncio.sleep(self._sleep_interval)
                        continue

                    for calc_type, interval in self.update_intervals.items():
                        last_update = self.last_updates[calc_type]
                        if current_time - last_update >= interval:
                            executed = await self._run_calculation(calc_type)
                            if executed:
                                self.last_updates[calc_type] = current_time

                    await self._update_heartbeat()
                    await asyncio.sleep(self._sleep_interval)

                except Exception as exc:
                    self.logger.error(f"Error in calculation loop: {exc}")
                    self.logger.error(traceback.format_exc())
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            self.logger.info("analytics_loop_cancelled", extra={"action": "loop_cancelled"})
            raise
        finally:
            await self._update_heartbeat(idle=True)

    async def _run_calculation(self, calc_type: str) -> bool:
        """Run specific calculation type for all symbols."""
        tasks = []

        try:
            if calc_type == 'vpin' and self.vpin_calculator:
                tasks = [self.vpin_calculator.calculate_vpin(symbol) for symbol in self.symbols]

            elif calc_type == 'gex_dex' and self.gex_dex_calculator:
                for symbol in self.symbols:
                    tasks.append(self.gex_dex_calculator.calculate_gex(symbol))
                    tasks.append(self.gex_dex_calculator.calculate_dex(symbol))

            elif calc_type == 'flow_toxicity' and self.pattern_analyzer:
                tasks = [self.pattern_analyzer.analyze_flow_toxicity(symbol) for symbol in self.symbols]

            elif calc_type == 'order_book' and self.pattern_analyzer:
                tasks = [self.pattern_analyzer.calculate_order_book_imbalance(symbol) for symbol in self.symbols]

            elif calc_type == 'sweep_detection' and self.pattern_analyzer:
                tasks = [self.pattern_analyzer.detect_sweeps(symbol) for symbol in self.symbols]

            elif calc_type == 'dealer_flows' and self.dealer_flow_calculator:
                tasks = [self.dealer_flow_calculator.calculate_dealer_metrics(symbol) for symbol in self.symbols]

            elif calc_type == 'flow_clustering' and self.flow_cluster_model:
                symbols = self.flow_cluster_symbols or self.symbols
                tasks = [self.flow_cluster_model.classify_flows(symbol) for symbol in symbols]

            elif calc_type == 'volatility' and self.volatility_metrics:
                await self.volatility_metrics.update_vix1d()
                return True

            elif calc_type == 'portfolio':
                await self.aggregator.calculate_portfolio_metrics()
                return True

            elif calc_type == 'sectors':
                await self.aggregator.calculate_sector_flows()
                return True

            elif calc_type == 'correlation':
                await self.aggregator.calculate_cross_asset_correlations()
                return True

            if not tasks:
                return True

            results = await asyncio.gather(*tasks, return_exceptions=True)
            success = True

            for result in results:
                if isinstance(result, Exception):
                    success = False
                    self.error_counts[calc_type] += 1
                    self.logger.error(f"Calculation error for {calc_type}: {result}")

            return success

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error_counts[calc_type] += 1
            self.logger.error(f"Failed to execute {calc_type}: {exc}")
            self.logger.error(traceback.format_exc())
            return False

    async def _update_heartbeat(self, idle: bool = False):
        """Update heartbeat in Redis."""
        heartbeat = {
            'ts': int(datetime.now().timestamp() * 1000),
            'symbols': len(self.symbols),
            'cadence_hz': self.cadence_hz,
            'running': self._running,
            'idle': idle,
            'update_intervals': self.update_intervals,
            'last_updates': {k: self.last_updates[k] for k in self.update_intervals},
            'error_counts': dict(self.error_counts),
        }

        await self.redis.setex(
            rkeys.heartbeat_key('analytics'),
            self.ttls.get('heartbeat', get_ttl('heartbeat')),
            json.dumps(heartbeat)
        )

    def is_market_hours(self, extended: bool = False) -> bool:
        """Check if currently in market hours."""
        now = datetime.now(self.market_tz)

        # Check weekend
        if now.weekday() >= 5:
            return False

        # Check time
        current_time = now.time()
        if extended:
            return datetime_time(4, 0) <= current_time <= datetime_time(20, 0)
        else:
            return datetime_time(9, 30) <= current_time <= datetime_time(16, 0)

    async def stop(self):
        """Stop the analytics engine."""
        self.logger.info("analytics_engine_stop", extra={"action": "stop"})
        self._running = False

        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        self._loop_task = None
        await self._update_heartbeat(idle=True)
        if self.volatility_metrics:
            await self.volatility_metrics.close()
