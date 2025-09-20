#!/usr/bin/env python3
"""Signal Generator - Main Signal Generation Coordinator."""

from __future__ import annotations

import asyncio
import json
import random
import time
import traceback
from datetime import datetime, timedelta, time as datetime_time
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import pytz
import redis.asyncio as aioredis

from dte_strategies import DTEStrategies
from moc_strategy import MOCStrategy
from option_utils import normalize_expiry
import redis_keys as rkeys
from signal_deduplication import SignalDeduplication, contract_fingerprint
from logging_utils import get_logger


class StrategyEvaluator(Protocol):
    """Protocol describing the strategy interface used by :class:`SignalGenerator`."""

    async def evaluate(self, strategy: str, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """Return ``(confidence, reasons, side)`` for the given symbol."""

    async def select_contract(
        self,
        symbol: str,
        strategy: str,
        side: str,
        spot: float,
        options_chain: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return a fully described contract for the proposed trade."""


class FeatureReader(Protocol):
    """Protocol for feature loader dependency injection."""

    async def __call__(self, redis_conn: aioredis.Redis, symbol: str) -> Dict[str, Any]:
        """Load analytics features for ``symbol`` from Redis."""


class SignalGenerator:
    """Generate trading signals based on analytics metrics and strategy rules."""

    def __init__(
        self,
        config: Dict[str, Any],
        redis_conn: aioredis.Redis,
        *,
        feature_reader: Optional[FeatureReader] = None,
        strategy_factories: Optional[Dict[str, Callable[[Dict[str, Any], aioredis.Redis], StrategyEvaluator]]] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.config = config
        self.redis = redis_conn
        self.logger = get_logger(__name__, component="signals", subsystem="generator")

        # Signal configuration
        self.signal_config = config.get('modules', {}).get('signals', {})
        self.enabled = self.signal_config.get('enabled', False)
        self.dry_run = self.signal_config.get('dry_run', True)

        # Strategy configurations
        self.strategies = self.signal_config.get('strategies', {})

        # Guardrail parameters
        self.max_staleness_s = float(self.signal_config.get('max_staleness_s', 5))
        self.min_confidence = float(self.signal_config.get('min_confidence', 0.60))
        self.min_refresh_s = float(self.signal_config.get('min_refresh_s', 2))
        self.cooldown_s = int(self.signal_config.get('cooldown_s', 30))
        self.ttl_seconds = int(self.signal_config.get('ttl_seconds', 300))
        self.version = self.signal_config.get('version', 'D6.0.1')

        # Loop configuration
        self.loop_interval_s = float(self.signal_config.get('loop_interval_s', 0.5))
        self.loop_jitter_s = float(self.signal_config.get('loop_jitter_s', 0.05))
        self.max_backoff_s = float(self.signal_config.get('max_backoff_s', 5.0))
        self._consecutive_errors = 0
        self._circuit_open_until: Optional[float] = None

        # Track last evaluation time per symbol
        self.last_eval: Dict[str, float] = {}

        # Eastern timezone for market hours
        self.eastern = pytz.timezone('US/Eastern')

        # Strategy implementations
        self.strategy_handlers: Dict[str, StrategyEvaluator] = {}
        self._strategy_factories = strategy_factories or {
            '0dte': DTEStrategies,
            '1dte': DTEStrategies,
            '14dte': DTEStrategies,
            'moc': MOCStrategy,
        }

        # Feature reader dependency (allows deterministic tests without Redis)
        self._feature_reader: FeatureReader = feature_reader or default_feature_reader

        # Deduplication/emission helper shared across strategies
        self._deduper = SignalDeduplication(config, redis_conn)

        # Shutdown coordination
        self._stop_event = stop_event or asyncio.Event()

        # Lua SHA cached by SignalDeduplication; maintain for backwards compat metrics
        self.lua_sha = None

    async def start(self, stop_event: Optional[asyncio.Event] = None) -> None:
        """Main signal generation loop.

        Args:
            stop_event: Optional external stop event. When provided the loop exits when
                either the internal or external event is set.
        """

        if not self.enabled:
            self.logger.info("signal_generator_disabled", extra={"action": "disabled"})
            return

        event = stop_event or self._stop_event
        self.logger.info(
            "signal_generator_start",
            extra={"action": "start", "dry_run": self.dry_run}
        )
        self._initialize_strategy_handlers()

        while not event.is_set():
            cycle_started = time.perf_counter()

            try:
                if self._circuit_open_until and time.time() < self._circuit_open_until:
                    await asyncio.sleep(min(1.0, self._circuit_open_until - time.time()))
                    continue

                current_time = datetime.now(self.eastern)

                # Update heartbeat for liveness monitoring
                await self.redis.setex('health:signals:heartbeat', 15, current_time.isoformat())

                for strategy_name, strategy_config in self.strategies.items():
                    if not strategy_config.get('enabled', False):
                        continue

                    if not self.is_strategy_active(strategy_name, current_time):
                        continue

                    symbols = strategy_config.get('symbols', [])
                    for symbol in symbols:
                        await self._evaluate_symbol(strategy_name, strategy_config, symbol)

                self._consecutive_errors = 0

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                self._consecutive_errors += 1
                await self.redis.incr('metrics:signals:loop_errors')
                self.logger.error("Error in signal generation loop", exc_info=exc)
                if self._consecutive_errors >= 5:
                    backoff = min(self.max_backoff_s, 2 ** self._consecutive_errors)
                    self._circuit_open_until = time.time() + backoff
                    self.logger.warning(
                        "Opening signal generation circuit breaker",
                        extra={"cooldown_s": backoff},
                    )

            delay = self._compute_sleep_delay(time.perf_counter() - cycle_started)
            if delay > 0:
                await asyncio.sleep(delay)

        self.logger.info("signal_generator_stop", extra={"action": "stop"})

    def stop(self) -> None:
        """Request cooperative shutdown of the signal loop."""

        self._stop_event.set()

    def get_strategy_handler(self, strategy_name: str):
        """Get the appropriate strategy handler."""
        if strategy_name in ['0dte', '1dte', '14dte']:
            return self.strategy_handlers.get('0dte')  # DTEStrategies handles all DTE variants
        else:
            return self.strategy_handlers.get(strategy_name)

    def _initialize_strategy_handlers(self) -> None:
        """Instantiate strategy handlers using the configured factories."""

        if self.strategy_handlers:
            return

        for strategy_name, factory in self._strategy_factories.items():
            if strategy_name in self.strategy_handlers:
                continue

            handler = factory(self.config, self.redis)

            if strategy_name in {'0dte', '1dte', '14dte'}:
                for alias in ('0dte', '1dte', '14dte'):
                    self.strategy_handlers.setdefault(alias, handler)
            else:
                self.strategy_handlers[strategy_name] = handler

    async def _evaluate_symbol(self, strategy_name: str, strategy_config: Dict[str, Any], symbol: str) -> None:
        """Evaluate a single symbol for a strategy."""

        now = time.time()
        cache_key = f"{symbol}:{strategy_name}"
        last_time = self.last_eval.get(cache_key, 0)
        if now - last_time < self.min_refresh_s:
            return

        try:
            features = await self._feature_reader(self.redis, symbol)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to load features", exc_info=exc, extra={"symbol": symbol})
            await self.redis.incr('metrics:signals:feature_errors')
            return

        if not features:
            await self.redis.incr('metrics:signals:blocked:no_features')
            return

        if not self.check_freshness(features):
            await self.redis.incr('metrics:signals:skipped_stale')
            await self.redis.incr('metrics:signals:blocked:stale_features')
            await self.redis.hincrby('metrics:signals:blocked_by_reason', 'stale_features', 1)
            return

        if not self.check_schema(features):
            await self.redis.incr('metrics:signals:skipped_schema')
            await self.redis.hincrby('metrics:signals:blocked_by_reason', 'schema', 1)
            return

        handler = self.get_strategy_handler(strategy_name)
        if handler is None:
            self.logger.error("No handler registered for strategy", extra={"strategy": strategy_name})
            return

        try:
            confidence, reasons, side = await handler.evaluate(strategy_name, symbol, features)
        except Exception as exc:  # pragma: no cover - surfaces strategy bugs
            self.logger.error(
                "Strategy evaluation failed",
                exc_info=exc,
                extra={"strategy": strategy_name, "symbol": symbol},
            )
            await self.redis.incr('metrics:signals:strategy_errors')
            return

        await self.redis.incr('metrics:signals:considered')

        min_conf = self._normalize_confidence_threshold(
            strategy_config.get('thresholds', {}).get('min_confidence')
        )

        if side == "FLAT" or confidence < min_conf:
            await self.redis.hincrby('metrics:signals:blocked_by_reason', 'low_confidence', 1)
            await self._debug_rejected_signal(symbol, strategy_name, confidence, min_conf, side, reasons, features)
            self.last_eval[cache_key] = now
            return

        await self._process_valid_signal(
            symbol, strategy_name, confidence, reasons, features, side, strategy_config
        )

        self.last_eval[cache_key] = now

    async def _process_valid_signal(
        self,
        symbol: str,
        strategy_name: str,
        confidence: int,
        reasons: List[str],
        features: Dict[str, Any],
        side: str,
        strategy_config: Dict[str, Any],
    ) -> None:
        """Process a validated signal by selecting a contract and emitting it atomically."""

        options_chain = features.get('options_chain')
        contract = await self.select_contract(symbol, strategy_name, side, features.get('price', 0), options_chain)
        contract_fp = contract_fingerprint(symbol, strategy_name, side, contract)

        signal_id = self._deduper.generate_signal_id(
            symbol, side, contract_fp, contract.get('expiry')
        )

        signal = await self.create_signal_with_contract(
            symbol,
            strategy_name,
            int(confidence),
            reasons,
            features,
            side,
            contract=contract,
            emit_id=signal_id,
            contract_fp=contract_fp,
        )

        exposure_state = await self._gather_exposure_state(symbol, contract_fp)
        exposure_violation = await self._check_exposure_caps(
            symbol, strategy_name, contract_fp, exposure_state, signal
        )
        if exposure_violation:
            return

        if not await self._deduper.check_material_change(
            contract_fp,
            int(confidence),
            signal=signal,
            signal_id=signal_id,
            exposure_state=exposure_state,
        ):
            return

        dynamic_ttl = self.calculate_dynamic_ttl(contract)

        emit_result = await self._deduper.atomic_emit(signal_id, contract_fp, symbol, signal, dynamic_ttl)
        self.lua_sha = self._deduper.lua_sha

        await self._handle_emit_result(
            emit_result,
            signal,
            signal_id,
            contract_fp,
            symbol,
            int(confidence),
            contract,
            dynamic_ttl,
            features,
        )

    async def _handle_emit_result(self, emit_result: int, signal: Dict, signal_id: str,
                                 contract_fp: str, symbol: str, confidence: int,
                                 contract: Dict, dynamic_ttl: int, features: Dict):
        """Handle the result of atomic signal emission."""
        if emit_result == 0:
            await self.redis.incr('metrics:signals:duplicates')
            await self.redis.incr('metrics:signals:blocked:duplicate')
            await self._deduper.add_audit_entry(
                contract_fp,
                "blocked",
                "duplicate",
                {"conf": confidence, "signal_id": signal_id},
            )

        elif emit_result == -1:
            await self.redis.incr('metrics:signals:cooldown_blocked')
            await self.redis.incr('metrics:signals:blocked:cooldown')
            await self._deduper.add_audit_entry(
                contract_fp,
                "blocked",
                "cooldown",
                {"conf": confidence, "signal_id": signal_id},
            )

        elif emit_result == -2:
            await self.redis.incr('metrics:signals:blocked:live')
            await self._deduper.add_audit_entry(
                contract_fp,
                "blocked",
                "live_lock",
                {"conf": confidence, "signal_id": signal_id},
            )

        elif emit_result == 1:
            # Successfully enqueued
            ts = int(time.time() * 1000)
            await self.redis.setex(f'signals:out:{symbol}:{ts}', dynamic_ttl, json.dumps(signal))
            await self.redis.setex(f'signals:latest:{symbol}', dynamic_ttl, json.dumps(signal))

            await self._deduper.add_audit_entry(
                contract_fp,
                "emitted",
                "",
                {
                    "conf": confidence,
                    "signal_id": signal_id,
                    "ttl": dynamic_ttl,
                    "contract": contract,
                },
            )

            # Increment emitted counter
            await self.redis.incr('metrics:signals:emitted')

            # Log signal
            self.logger.info(
                "signal_emitted",
                extra={
                    "action": "emit",
                    "symbol": symbol,
                    "side": signal['side'],
                    "confidence_pct": confidence,
                    "is_rth": self.is_rth(datetime.now(self.eastern)),
                    "features": {
                        "vpin": features.get('vpin', 0),
                        "obi": features.get('obi', 0),
                        "gex_z": features.get('gex_z', 0),
                        "dex_z": features.get('dex_z', 0),
                    },
                    "signal_id": signal_id,
                    "ttl": dynamic_ttl,
                },
            )

    async def _debug_rejected_signal(self, symbol: str, strategy: str, confidence: int,
                                    min_conf: float, side: str, reasons: List[str], features: Dict):
        """Store debug info for rejected signals."""
        await self.redis.setex(
            f'signals:debug:{symbol}:{strategy}',
            60,
            json.dumps({
                'rejected': True,
                'confidence': confidence,
                'min_conf': min_conf,
                'side': side,
                'reasons': reasons,
                'features': {
                    'vpin': features.get('vpin', 0),
                    'obi': features.get('obi', 0),
                    'price': features.get('price', 0),
                    'age_s': features.get('age_s', 999),
                    'gamma_pin_proximity': features.get('gamma_pin_proximity', 0),
                    'gex_strikes': len(features.get('gex_by_strike', [])),
                    'gex_total': features.get('gex', 0)
                },
                'timestamp': time.time()
            })
        )

    def is_strategy_active(self, strategy: str, current_time: datetime) -> bool:
        """Check if strategy is within its active time window."""
        strategy_config = self.strategies.get(strategy, {})
        time_window = strategy_config.get('time_window', {})

        if not time_window:
            return True  # No time window means always active

        start_str = time_window.get('start', '09:30')
        end_str = time_window.get('end', '16:00')

        # Parse time strings
        start_hour, start_min = map(int, start_str.split(':'))
        end_hour, end_min = map(int, end_str.split(':'))

        # Create time objects for comparison
        start_time = datetime_time(start_hour, start_min)
        end_time = datetime_time(end_hour, end_min)
        current_time_only = current_time.time()

        # Check if within window
        return start_time <= current_time_only <= end_time

    def is_rth(self, current_time: datetime) -> bool:
        """Check if within regular trading hours."""
        return (current_time.weekday() < 5 and
                datetime_time(9, 30) <= current_time.time() <= datetime_time(16, 0))

    def _normalize_confidence_threshold(self, value: Optional[float]) -> int:
        """Normalize configuration confidence thresholds to an integer percentage."""

        if value is None:
            value = self.min_confidence

        if value <= 1:
            return int(round(value * 100))
        return int(round(value))

    def _compute_sleep_delay(self, cycle_duration: float) -> float:
        """Compute sleep delay with jitter and back-pressure handling."""

        base_delay = max(0.0, self.loop_interval_s - cycle_duration)
        jitter = random.uniform(0, max(self.loop_jitter_s, 0.0))
        delay = base_delay + jitter

        if self._consecutive_errors:
            delay = min(self.max_backoff_s, delay + (2 ** self._consecutive_errors) * 0.1)

        return max(0.0, delay)

    def check_freshness(self, features: Dict[str, Any]) -> bool:
        """Check if features are fresh enough."""
        return features.get('age_s', 999) <= self.max_staleness_s

    def check_schema(self, features: Dict[str, Any]) -> bool:
        """Check if features have required fields."""
        required = ['price', 'vpin', 'obi', 'timestamp']
        return all(k in features for k in required)

    def calculate_dynamic_ttl(self, contract: Dict[str, Any], *, now: Optional[datetime] = None) -> int:
        """Calculate TTL based on contract expiry and market close timing."""

        now = now or datetime.now(self.eastern)

        def _ttl_for_dte(dte_days: int) -> int:
            if dte_days <= 0:
                return 86400  # 24 hours
            if dte_days == 1:
                return 172800  # 48 hours
            return max((dte_days + 1) * 86400, 604800)  # At least a week

        ttl = max(60, int(self.ttl_seconds))

        expiry_str = normalize_expiry(contract.get('expiry'))
        dte = None
        if expiry_str:
            try:
                expiry_dt = datetime.strptime(expiry_str, "%Y%m%d").date()
                dte = (expiry_dt - now.date()).days
            except ValueError:
                dte = None

        if dte is None:
            dte_band = str(contract.get('dte_band') or '').strip()
            try:
                if dte_band:
                    dte = int(dte_band)
            except ValueError:
                dte = None

        if dte is not None:
            ttl = max(ttl, _ttl_for_dte(dte))
        else:
            ttl = max(ttl, 86400)

        return int(ttl)

    def _resolve_exposure_cap(self, symbol: str, strategy: str, cap_key: str) -> Optional[int]:
        caps_cfg = self.signal_config.get('exposure_caps', {}) or {}
        default_cfg = caps_cfg.get('default', {}) or {}
        strategy_cfg = (caps_cfg.get('strategies') or {}).get(strategy, {}) or {}
        symbol_cfg = (caps_cfg.get('symbols') or {}).get(symbol, {}) or {}
        symbol_strategy_cfg = (symbol_cfg.get('strategies') or {}).get(strategy, {}) or {}

        for scope in (symbol_strategy_cfg, symbol_cfg, strategy_cfg, default_cfg):
            value = scope.get(cap_key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "invalid_exposure_cap_value",
                    extra={
                        "action": "exposure_cap_invalid",
                        "symbol": symbol,
                        "strategy": strategy,
                        "cap_key": cap_key,
                        "value": value,
                    },
                )
        return None

    async def _count_pending_orders(self, symbol: str) -> int:
        symbol_upper = (symbol or '').upper()
        count = 0
        async for key in self.redis.scan_iter(match='orders:pending:*'):
            payload = await self.redis.get(key)
            if not payload:
                continue
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8', errors='ignore')
            try:
                data = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
            order_symbol = data.get('symbol') or (
                (data.get('contract') or {}).get('symbol')
            )
            if isinstance(order_symbol, str) and order_symbol.upper() == symbol_upper:
                count += 1
        return count

    async def _gather_exposure_state(self, symbol: str, contract_fp: str) -> Dict[str, Any]:
        live_key = f'signals:live:{symbol}:{contract_fp}'
        live_lock = bool(await self.redis.exists(live_key))

        live_count = 0
        async for key in self.redis.scan_iter(match=f'signals:live:{symbol}:*'):
            live_count += 1

        positions = await self.redis.scard(f'positions:by_symbol:{symbol}')
        pending_orders = await self._count_pending_orders(symbol)

        return {
            'symbol': symbol,
            'live_key': live_key,
            'live_lock': live_lock,
            'live_count': live_count,
            'positions': int(positions or 0),
            'pending_orders': pending_orders,
        }

    async def _check_exposure_caps(
        self,
        symbol: str,
        strategy: str,
        contract_fp: str,
        exposure_state: Dict[str, Any],
        signal: Dict[str, Any],
    ) -> bool:
        caps = [
            ('max_positions', exposure_state.get('positions', 0)),
            ('max_pending_orders', exposure_state.get('pending_orders', 0)),
            ('max_live_signals', exposure_state.get('live_count', 0)),
        ]

        for cap_key, value in caps:
            limit = self._resolve_exposure_cap(symbol, strategy, cap_key)
            if limit is None or limit <= 0:
                continue

            adjusted_value = value
            if cap_key == 'max_live_signals' and exposure_state.get('live_lock'):
                adjusted_value = value

            if adjusted_value >= limit:
                details = {
                    'cap': cap_key,
                    'limit': limit,
                    'value': adjusted_value,
                    'symbol': symbol,
                    'strategy': strategy,
                }
                self.logger.info(
                    "signal_exposure_block",
                    extra={
                        "action": "exposure_block",
                        **details,
                    },
                )
                await self.redis.incr('metrics:signals:blocked:exposure')
                await self._deduper.add_audit_entry(
                    contract_fp,
                    "blocked",
                    "exposure_cap",
                    details,
                )
                await self._deduper.publish_update(
                    dict(signal),
                    contract_fp=contract_fp,
                    reason=cap_key,
                    exposure_state=exposure_state,
                )
                return True

        return False

    async def select_contract(self, symbol: str, strategy: str, side: str,
                            spot: float, options_chain=None) -> Dict[str, Any]:
        """Select appropriate contract for the strategy."""
        # Delegate to strategy handlers
        handler = self.get_strategy_handler(strategy)
        if handler and hasattr(handler, 'select_contract'):
            return await handler.select_contract(symbol, strategy, side, spot, options_chain)

        # Default contract if no handler
        return {
            'symbol': symbol,
            'expiry': strategy.upper(),
            'strike': round(spot),
            'right': 'C' if side == 'LONG' else 'P',
            'multiplier': 100,
            'exchange': 'SMART'
        }

    async def create_signal_with_contract(self, symbol: str, strategy: str, confidence: int,
                                         reasons: List[str], features: Dict, side: str,
                                         contract: Dict, emit_id: str, contract_fp: str) -> Dict:
        """Create signal object with contract details."""
        timestamp_ms = int(time.time() * 1000)
        return {
            'id': emit_id,
            'contract_fp': contract_fp,
            'timestamp': timestamp_ms,
            'ts': timestamp_ms,
            'symbol': symbol,
            'strategy': strategy,
            'side': side,
            'confidence': int(confidence),
            'reasons': reasons,
            'contract': contract,
            'features': {
                'vpin': features.get('vpin', 0),
                'obi': features.get('obi', 0),
                'gex': features.get('gex', 0),
                'dex': features.get('dex', 0),
                'price': features.get('price', 0)
            },
            'version': self.version,
            'dry_run': self.dry_run
        }


async def default_feature_reader(redis_conn: aioredis.Redis, symbol: str) -> Dict[str, Any]:
    """Load analytics features for a symbol from Redis.

    Missing or stale data falls back to neutral defaults so the generator can
    continue operating during transient upstream disruptions.
    """

    logger = get_logger(__name__, component="signals", subsystem="features")

    async with redis_conn.pipeline() as pipe:
        pipe.get(rkeys.market_ticker_key(symbol))
        pipe.lrange(rkeys.market_bars_key(symbol, '1min'), -25, -1)
        pipe.get(rkeys.analytics_vpin_key(symbol))
        pipe.get(rkeys.analytics_obi_key(symbol))
        pipe.get(rkeys.analytics_gex_key(symbol))
        pipe.get(rkeys.analytics_dex_key(symbol))
        pipe.get(rkeys.analytics_toxicity_key(symbol))
        pipe.get(rkeys.analytics_metric_key(symbol, 'sweep'))
        pipe.get(rkeys.analytics_metric_key(symbol, 'hidden'))
        pipe.get(rkeys.analytics_metric_key(symbol, 'unusual'))
        pipe.get(rkeys.analytics_metric_key(symbol, 'institutional_flow'))
        pipe.get(rkeys.analytics_metric_key(symbol, 'retail_flow'))
        pipe.get(rkeys.analytics_metric_key(symbol, 'gamma_pin'))
        pipe.get(rkeys.analytics_metric_key(symbol, 'gamma_pull'))
        pipe.get(rkeys.analytics_metric_key(symbol, 'moc'))
        pipe.get(rkeys.options_chain_key(symbol))
        pipe.get(rkeys.analytics_vanna_key(symbol))
        pipe.get(rkeys.analytics_charm_key(symbol))
        pipe.get(rkeys.analytics_hedging_impact_key(symbol))
        pipe.get(rkeys.analytics_skew_key(symbol))
        pipe.get(rkeys.analytics_flow_clusters_key(symbol))
        pipe.get(rkeys.analytics_vix1d_key())
        raw_results = await pipe.execute()

    def _decode(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    try:
        (
            ticker_raw,
            bars_raw,
            vpin_raw,
            obi_raw,
            gex_raw,
            dex_raw,
            toxicity_raw,
            sweep_raw,
            hidden_raw,
            unusual_raw,
            institutional_raw,
            retail_raw,
            gamma_pin_raw,
            gamma_pull_raw,
            moc_raw,
            options_chain_raw,
            vanna_raw,
            charm_raw,
            hedging_raw,
            skew_raw,
            flow_clusters_raw,
            vix1d_raw,
        ) = raw_results
    except ValueError:  # pragma: no cover - defensive
        logger.debug("Unexpected feature payload length for %s", symbol)
        ticker_raw = bars_raw = vpin_raw = obi_raw = gex_raw = dex_raw = None
        toxicity_raw = sweep_raw = hidden_raw = unusual_raw = None
        institutional_raw = retail_raw = gamma_pin_raw = gamma_pull_raw = None
        moc_raw = options_chain_raw = None
        vanna_raw = charm_raw = hedging_raw = skew_raw = flow_clusters_raw = vix1d_raw = None

    ticker = _decode(ticker_raw) or {}
    bars_payload = bars_raw or []
    bars: List[Dict[str, Any]] = []
    for entry in bars_payload:
        parsed = _decode(entry)
        if isinstance(parsed, dict):
            bars.append(parsed)

    options_chain_value = _decode(options_chain_raw)
    if isinstance(options_chain_value, dict):
        by_contract = options_chain_value.get('by_contract')
        if isinstance(by_contract, dict):
            options_chain = [c for c in by_contract.values() if isinstance(c, dict)]
        else:
            contracts_list = options_chain_value.get('contracts') or options_chain_value.get('raw') or []
            options_chain = [c for c in contracts_list if isinstance(c, dict)]
    elif isinstance(options_chain_value, list):
        options_chain = [c for c in options_chain_value if isinstance(c, dict)]
    else:
        options_chain = []

    timestamp_ms = int(ticker.get('timestamp') or ticker.get('ts') or 0)
    if timestamp_ms and timestamp_ms < 1e12:
        timestamp_ms = int(timestamp_ms * 1000)

    price = ticker.get('mid') or ticker.get('last') or ticker.get('close') or ticker.get('price')
    if price is None:
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        if bid is not None and ask is not None:
            price = (float(bid) + float(ask)) / 2
    price = float(price or 0)

    now_ms = int(time.time() * 1000)
    age_s = (now_ms - timestamp_ms) / 1000 if timestamp_ms else 999

    def _extract_metric(payload: Any, *fields: str, default: float = 0.0) -> float:
        data = _decode(payload)
        if isinstance(data, (int, float)):
            return float(data)
        if isinstance(data, dict):
            for field in fields:
                if field in data:
                    try:
                        return float(data[field])
                    except (TypeError, ValueError):
                        continue
        return float(default)

    features: Dict[str, Any] = {
        'price': price,
        'timestamp': timestamp_ms or now_ms,
        'age_s': max(age_s, 0),
        'bars': bars,
        'options_chain': options_chain,
        'vpin': _extract_metric(vpin_raw, 'value', 'vpin', 'score'),
        'obi': _extract_metric(
            obi_raw,
            'level1_imbalance',
            'level5_imbalance',
            'value',
            'obi',
            default=0.5,
        ),
        'sweep': _extract_metric(sweep_raw, 'score', 'value'),
        'hidden_orders': _extract_metric(hidden_raw, 'score', 'value'),
        'unusual_activity': _extract_metric(unusual_raw, 'score', 'value'),
        'institutional_flow': _extract_metric(institutional_raw, 'value'),
        'retail_flow': _extract_metric(retail_raw, 'value'),
    }

    toxicity_data = _decode(toxicity_raw) or {}
    if isinstance(toxicity_data, dict):
        raw_score = toxicity_data.get('toxicity_score')
        if raw_score is None:
            raw_score = toxicity_data.get('toxicity') or toxicity_data.get('score')
        adjusted_score = toxicity_data.get('toxicity_adjusted')

        try:
            if raw_score is not None:
                features['toxicity_raw'] = float(raw_score)
        except (TypeError, ValueError):
            pass

        try:
            if adjusted_score is not None:
                features['toxicity_adjusted'] = float(adjusted_score)
        except (TypeError, ValueError):
            pass

        try:
            features['toxicity_confidence'] = float(toxicity_data.get('confidence', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['toxicity_confidence'] = 0.0

        try:
            features['aggressor_ratio'] = float(toxicity_data.get('aggressor_ratio', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['aggressor_ratio'] = 0.0

        try:
            features['large_trade_ratio'] = float(toxicity_data.get('large_trade_ratio', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['large_trade_ratio'] = 0.0

        try:
            features['institutional_score'] = float(toxicity_data.get('institutional_score', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['institutional_score'] = 0.0

        features['toxicity'] = (
            features.get('toxicity_adjusted')
            or features.get('toxicity_raw')
            or 0.5
        )
        features['toxicity_level'] = toxicity_data.get('toxicity_adjusted_level') or toxicity_data.get('toxicity_level')
        features['venue_mix'] = toxicity_data.get('derived_venue_mix') or {}
    else:
        features['toxicity'] = _extract_metric(toxicity_raw, 'score', 'toxicity', default=0.5)

    gex_data = _decode(gex_raw) or {}
    if isinstance(gex_data, dict):
        features['gex'] = gex_data.get('total_gex') or gex_data.get('gex') or 0
        features['gex_z'] = gex_data.get('zscore') or gex_data.get('z') or 0
        by_strike = gex_data.get('gex_by_strike') or {}
        if isinstance(by_strike, dict):
            parsed_strikes = []
            for strike, value in by_strike.items():
                try:
                    parsed_strikes.append({'strike': float(strike), 'gex': float(value)})
                except (TypeError, ValueError):
                    continue
            features['gex_by_strike'] = parsed_strikes
        pin_strike = gex_data.get('max_gex_strike') or gex_data.get('zero_gamma_strike')
        spot = gex_data.get('spot') or price
        if pin_strike and spot:
            try:
                distance_pct = abs(float(pin_strike) - float(spot)) / max(float(spot), 1e-6)
                features['gamma_pin_proximity'] = max(0.0, min(1.0, 1 - distance_pct / 0.02))
                features['gamma_pull_dir'] = 'UP' if pin_strike > spot else 'DOWN'
            except (TypeError, ValueError):
                pass

    dex_data = _decode(dex_raw) or {}
    if isinstance(dex_data, dict):
        features['dex'] = dex_data.get('total_dex') or dex_data.get('dex') or 0
        features['dex_z'] = dex_data.get('zscore') or dex_data.get('z') or 0

    moc_data = _decode(moc_raw) or {}
    if isinstance(moc_data, dict):
        features.update({
            'imbalance_total': moc_data.get('imbalance_total', 0),
            'imbalance_ratio': moc_data.get('imbalance_ratio', 0),
            'imbalance_side': moc_data.get('imbalance_side'),
            'imbalance_paired': moc_data.get('imbalance_paired', 0),
            'indicative_price': moc_data.get('indicative_price'),
            'near_close_offset_bps': moc_data.get('near_close_offset_bps', 0),
        })

    gamma_pin_data = _decode(gamma_pin_raw)
    if isinstance(gamma_pin_data, dict):
        features.setdefault('gamma_pin_proximity', gamma_pin_data.get('proximity', 0))

    gamma_pull_data = _decode(gamma_pull_raw)
    if isinstance(gamma_pull_data, dict):
        features.setdefault('gamma_pull_dir', gamma_pull_data.get('direction'))

    vanna_data = _decode(vanna_raw) or {}
    if isinstance(vanna_data, dict):
        features['vanna_notional'] = float(vanna_data.get('total_vanna_notional_per_pct_vol', 0.0) or 0.0)
        history = vanna_data.get('history') or {}
        try:
            features['vanna_z'] = float(history.get('zscore', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['vanna_z'] = 0.0
        features['vanna_data'] = vanna_data

    charm_data = _decode(charm_raw) or {}
    if isinstance(charm_data, dict):
        features['charm_notional'] = float(charm_data.get('total_charm_notional_per_day', 0.0) or 0.0)
        features['charm_shares_per_day'] = float(charm_data.get('total_charm_shares_per_day', 0.0) or 0.0)
        history = charm_data.get('history') or {}
        try:
            features['charm_z'] = float(history.get('zscore', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['charm_z'] = 0.0
        features['charm_data'] = charm_data

    hedging_data = _decode(hedging_raw) or {}
    if isinstance(hedging_data, dict):
        features['hedging_notional_per_pct'] = float(hedging_data.get('notional_per_pct_move', 0.0) or 0.0)
        features['hedging_shares_per_pct'] = float(hedging_data.get('shares_per_pct_move', 0.0) or 0.0)
        features['hedging_gamma_shares'] = float(hedging_data.get('gamma_component_shares', 0.0) or 0.0)
        features['hedging_vanna_shares'] = float(hedging_data.get('vanna_component_shares', 0.0) or 0.0)
        features['charm_notional_per_day'] = float(hedging_data.get('charm_notional_per_day', 0.0) or 0.0)
        features['hedging_data'] = hedging_data

    skew_data = _decode(skew_raw) or {}
    if isinstance(skew_data, dict):
        features['skew'] = skew_data.get('skew')
        history = skew_data.get('history') or {}
        try:
            features['skew_z'] = float(history.get('zscore', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['skew_z'] = 0.0
        features['skew_ratio'] = skew_data.get('skew_ratio')
        features['skew_data'] = skew_data

    flow_clusters = _decode(flow_clusters_raw) or {}
    if isinstance(flow_clusters, dict):
        strategy_dist = flow_clusters.get('strategy_distribution') or {}
        participants = flow_clusters.get('participant_distribution') or {}
        features['flow_momentum'] = float(strategy_dist.get('momentum', 0.0) or 0.0)
        features['flow_mean_reversion'] = float(strategy_dist.get('mean_reversion', 0.0) or 0.0)
        features['flow_hedging'] = float(strategy_dist.get('hedging', 0.0) or 0.0)
        features['flow_institutional'] = float(participants.get('institutional', 0.0) or 0.0)
        features['flow_retail'] = float(participants.get('retail', 0.0) or 0.0)
        features['flow_clusters'] = flow_clusters

    vix1d_data = _decode(vix1d_raw) or {}
    if isinstance(vix1d_data, dict):
        features['vix1d_value'] = float(vix1d_data.get('value', 0.0) or 0.0)
        features['vix1d_regime'] = str(vix1d_data.get('regime') or '')
        try:
            features['vix1d_z'] = float(vix1d_data.get('zscore', 0.0) or 0.0)
        except (TypeError, ValueError):
            features['vix1d_z'] = 0.0
        features['vix1d'] = vix1d_data

    return features
