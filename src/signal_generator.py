#!/usr/bin/env python3
"""
Signal Generator - Main Signal Generation Coordinator
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- metrics:{symbol}:*: All analytics metrics
- signals:pending:{symbol}: Signal queue
- signals:emitted:*: Idempotency tracking
- signals:cooldown:*: Cooldown tracking
- health:signals:heartbeat: Heartbeat status
"""

import asyncio
import json
import time
import redis
import redis.asyncio as aioredis
import uuid
import hashlib
import numpy as np
import pytz
from datetime import datetime, timedelta, time as datetime_time
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback


def contract_fingerprint(symbol: str, strategy: str, side: str, contract: dict) -> str:
    """
    Stable identity for the specific option contract choice.
    Includes multiplier and exchange to handle edge cases (minis, different venues).
    """
    parts = (
        symbol,
        strategy,
        side,
        str(contract.get('expiry')),
        str(contract.get('right')),
        str(contract.get('strike')),
        str(contract.get('multiplier', 100)),  # Default 100 for standard equity options
        str(contract.get('exchange', 'SMART')),  # Default SMART for IB routing
    )
    return "sigfp:" + hashlib.sha1(":".join(parts).encode()).hexdigest()[:20]


def trading_day_bucket(ts: float = None) -> str:
    """
    Return YYYYMMDD in US/Eastern; aligns with the trading session day.
    Market day changes at 4PM ET, not midnight.
    """
    ET = pytz.timezone("America/New_York")
    dt = datetime.fromtimestamp(ts or time.time(), tz=ET)
    return dt.strftime("%Y%m%d")


class SignalGenerator:
    """
    Generate trading signals based on analytics metrics and strategy rules.
    Coordinates all signal generation strategies.
    """

    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize signal generator with configuration.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Signal configuration
        self.signal_config = config.get('modules', {}).get('signals', {})
        self.enabled = self.signal_config.get('enabled', False)
        self.dry_run = self.signal_config.get('dry_run', True)

        # Strategy configurations
        self.strategies = self.signal_config.get('strategies', {})

        # Guardrail parameters
        self.max_staleness_s = self.signal_config.get('max_staleness_s', 5)
        self.min_confidence = self.signal_config.get('min_confidence', 0.60)
        self.min_refresh_s = self.signal_config.get('min_refresh_s', 2)
        self.cooldown_s = self.signal_config.get('cooldown_s', 30)
        self.ttl_seconds = self.signal_config.get('ttl_seconds', 300)
        self.version = self.signal_config.get('version', 'D6.0.1')

        # Track last evaluation time per symbol
        self.last_eval = {}

        # Eastern timezone for market hours
        self.eastern = pytz.timezone('US/Eastern')

        # Strategy implementations (will be imported from separate modules)
        self.strategy_handlers = {}

        # Atomic Redis Lua script for idempotency + enqueue + cooldown
        self.LUA_ATOMIC_EMIT = """
        -- KEYS[1] = idempotency_key "signals:emitted:<emit_id>"
        -- KEYS[2] = cooldown_key    "signals:cooldown:<contract_fp>"
        -- KEYS[3] = queue_key       "signals:pending:<symbol>"
        -- ARGV[1] = signal_json
        -- ARGV[2] = idempotency_ttl_seconds
        -- ARGV[3] = cooldown_ttl_seconds

        if redis.call('SETNX', KEYS[1], '1') == 1 then
            redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) * 1000)
            if redis.call('EXISTS', KEYS[2]) == 0 then
                redis.call('LPUSH', KEYS[3], ARGV[1])
                redis.call('PEXPIRE', KEYS[2], tonumber(ARGV[3]) * 1000)
                return 1  -- Signal enqueued
            else
                return -1  -- Blocked by cooldown
            end
        else
            return 0  -- Duplicate signal
        end
        """
        self.lua_sha = None  # Will be loaded on first use

    async def start(self):
        """
        Main signal generation loop.
        Processing frequency: Every 500ms
        """
        if not self.enabled:
            self.logger.info("Signal generator disabled in config")
            return

        self.logger.info(f"Starting signal generator (dry_run={self.dry_run})...")

        # Import and initialize strategy handlers
        from dte_strategies import DTEStrategies
        from moc_strategy import MOCStrategy

        self.strategy_handlers['0dte'] = DTEStrategies(self.config, self.redis)
        self.strategy_handlers['1dte'] = DTEStrategies(self.config, self.redis)
        self.strategy_handlers['14dte'] = DTEStrategies(self.config, self.redis)
        self.strategy_handlers['moc'] = MOCStrategy(self.config, self.redis)

        while True:
            try:
                current_time = datetime.now(self.eastern)

                # Update heartbeat
                await self.redis.setex('health:signals:heartbeat', 15, current_time.isoformat())

                # Process each enabled strategy
                for strategy_name, strategy_config in self.strategies.items():
                    if not strategy_config.get('enabled', False):
                        continue

                    # Check if strategy is active
                    if not self.is_strategy_active(strategy_name, current_time):
                        continue

                    # Process each symbol for this strategy
                    symbols = strategy_config.get('symbols', [])
                    for symbol in symbols:
                        try:
                            # Check minimum refresh interval
                            last_time = self.last_eval.get(f"{symbol}:{strategy_name}", 0)
                            if time.time() - last_time < self.min_refresh_s:
                                continue

                            # Read features from Redis
                            features = await self.read_features(symbol)

                            # Check freshness gate
                            if not self.check_freshness(features):
                                await self.redis.incr('metrics:signals:skipped_stale')
                                await self.redis.incr('metrics:signals:blocked:stale_features')
                                continue

                            # Check schema gate
                            if not self.check_schema(features):
                                await self.redis.incr('metrics:signals:skipped_schema')
                                continue

                            # Evaluate strategy conditions using appropriate handler
                            handler = self.get_strategy_handler(strategy_name)
                            confidence, reasons, side = await handler.evaluate(strategy_name, symbol, features)

                            # Increment considered counter
                            await self.redis.incr('metrics:signals:considered')

                            # Check if signal meets threshold
                            min_conf = strategy_config.get('thresholds', {}).get('min_confidence', self.min_confidence * 100)
                            if side == "FLAT" or confidence < min_conf:
                                await self._debug_rejected_signal(symbol, strategy_name, confidence, min_conf, side, reasons, features)
                                continue

                            # Process valid signal
                            await self._process_valid_signal(
                                symbol, strategy_name, confidence, reasons, features, side, strategy_config
                            )

                            # Update last evaluation time
                            self.last_eval[f"{symbol}:{strategy_name}"] = time.time()

                        except Exception as e:
                            self.logger.error(f"Error processing {symbol} for {strategy_name}: {e}")
                            self.logger.error(traceback.format_exc())

                await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Error in signal generation loop: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(1)

    def get_strategy_handler(self, strategy_name: str):
        """Get the appropriate strategy handler."""
        if strategy_name in ['0dte', '1dte', '14dte']:
            return self.strategy_handlers.get('0dte')  # DTEStrategies handles all DTE variants
        else:
            return self.strategy_handlers.get(strategy_name)

    async def _process_valid_signal(self, symbol: str, strategy_name: str, confidence: int,
                                   reasons: List[str], features: Dict, side: str, strategy_config: Dict):
        """Process a valid signal that meets confidence threshold."""
        # Select contract first
        options_chain = features.get('options_chain') if strategy_name == 'moc' else None
        contract = await self.select_contract(symbol, strategy_name, side, features.get('price', 0), options_chain)
        contract_fp = contract_fingerprint(symbol, strategy_name, side, contract)

        # Generate deterministic signal ID
        signal_id = self.generate_signal_id(symbol, side, contract_fp)

        # Check for material change
        if not await self._check_material_change(confidence, contract_fp):
            await self.redis.incr('metrics:signals:thin_update_blocked')
            return

        # Update last confidence
        await self.redis.setex(f"signals:last_conf:{contract_fp}", 900, int(confidence))

        # Create signal object
        signal = await self.create_signal_with_contract(
            symbol, strategy_name, confidence, reasons, features, side,
            contract=contract, emit_id=signal_id, contract_fp=contract_fp
        )

        # Calculate dynamic TTL
        dynamic_ttl = self.calculate_dynamic_ttl(contract)

        # Atomic emit
        emit_result = await self.atomic_emit_signal(signal, signal_id, contract_fp, symbol, dynamic_ttl)

        # Handle emit result
        await self._handle_emit_result(emit_result, signal, signal_id, contract_fp, symbol, confidence, contract, dynamic_ttl, features)

    async def _check_material_change(self, confidence: int, contract_fp: str) -> bool:
        """Check if confidence change is material enough to emit."""
        last_c_key = f"signals:last_conf:{contract_fp}"
        last_conf = await self.redis.get(last_c_key)
        if last_conf is not None:
            last_conf_val = int(last_conf)
            delta = abs(confidence - last_conf_val)
            threshold = max(3, 0.05 * max(1, last_conf_val))  # 3 pts or 5%
            if delta < threshold:
                # Add to audit trail
                await self._add_audit_entry(contract_fp, "blocked", "thin_update", confidence,
                                          extra={"last_conf": last_conf_val, "delta": delta, "threshold": threshold})
                return False
        return True

    async def _handle_emit_result(self, emit_result: int, signal: Dict, signal_id: str,
                                 contract_fp: str, symbol: str, confidence: int,
                                 contract: Dict, dynamic_ttl: int, features: Dict):
        """Handle the result of atomic signal emission."""
        if emit_result == 0:
            # Duplicate signal
            await self.redis.incr('metrics:signals:duplicates')
            await self.redis.incr('metrics:signals:blocked:duplicate')
            await self._add_audit_entry(contract_fp, "blocked", "duplicate", confidence,
                                       extra={"signal_id": signal_id})

        elif emit_result == -1:
            # Blocked by cooldown
            await self.redis.incr('metrics:signals:cooldown_blocked')
            await self.redis.incr('metrics:signals:blocked:cooldown')
            await self._add_audit_entry(contract_fp, "blocked", "cooldown", confidence,
                                       extra={"signal_id": signal_id})

        elif emit_result == 1:
            # Successfully enqueued
            ts = int(time.time() * 1000)
            await self.redis.setex(f'signals:out:{symbol}:{ts}', dynamic_ttl, json.dumps(signal))
            await self.redis.setex(f'signals:latest:{symbol}', dynamic_ttl, json.dumps(signal))

            await self._add_audit_entry(contract_fp, "emitted", None, confidence,
                                       extra={"signal_id": signal_id, "ttl": dynamic_ttl, "contract": contract})

            # Increment emitted counter
            await self.redis.incr('metrics:signals:emitted')

            # Log signal
            self.logger.info(
                f"signals DECIDE symbol={symbol} side={signal['side']} conf={confidence/100:.2f} "
                f"vpin={features.get('vpin', 0):.2f} obi={features.get('obi', 0):.2f} "
                f"gexZ={features.get('gex_z', 0):.1f} dexZ={features.get('dex_z', 0):.1f} "
                f"rth={self.is_rth(datetime.now(self.eastern))}"
            )

    async def _add_audit_entry(self, contract_fp: str, action: str, reason: Optional[str],
                              confidence: int, extra: Dict = None):
        """Add entry to audit trail for a contract fingerprint."""
        audit_key = f"signals:audit:{contract_fp}"
        entry = {
            "ts": time.time(),
            "action": action,
            "conf": confidence
        }
        if reason:
            entry["reason"] = reason
        if extra:
            entry.update(extra)

        await self.redis.lpush(audit_key, json.dumps(entry))
        await self.redis.ltrim(audit_key, 0, 50)  # Keep last 50 entries
        await self.redis.expire(audit_key, 3600)  # 1 hour TTL

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

    def check_freshness(self, features: Dict[str, Any]) -> bool:
        """Check if features are fresh enough."""
        return features.get('age_s', 999) <= self.max_staleness_s

    def check_schema(self, features: Dict[str, Any]) -> bool:
        """Check if features have required fields."""
        required = ['price', 'vpin', 'obi', 'timestamp']
        return all(k in features for k in required)

    async def read_features(self, symbol: str) -> Dict[str, Any]:
        """Read all required features from Redis for signal evaluation."""
        # This is a large method that reads various metrics
        # Moving to separate module for clarity
        from signal_deduplication import read_features_from_redis
        return await read_features_from_redis(self.redis, symbol)

    def generate_signal_id(self, symbol: str, side: str, contract_fp: str) -> str:
        """Generate deterministic signal ID."""
        day_bucket = trading_day_bucket()
        return f"{day_bucket}:{contract_fp}"

    def calculate_dynamic_ttl(self, contract: dict) -> int:
        """Calculate TTL based on contract expiry."""
        now = datetime.now(self.eastern)

        # Get market close time (4:00 PM ET)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now >= market_close:
            # If after market close, use next trading day close
            next_day = market_close + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip weekends
                next_day += timedelta(days=1)
            market_close = next_day

        # Calculate time to market close
        time_to_close = (market_close - now).total_seconds()

        # Set TTL based on contract type
        if contract.get('expiry') == '0DTE':
            ttl = min(time_to_close, self.ttl_seconds)
        elif contract.get('expiry') == '1DTE':
            ttl = min(time_to_close + 86400, self.ttl_seconds)
        else:
            ttl = self.ttl_seconds

        return max(60, int(ttl))

    async def atomic_emit_signal(self, signal: dict, signal_id: str, contract_fp: str,
                                symbol: str, ttl: int = None) -> int:
        """Atomically check idempotency, enqueue signal, and set cooldown."""
        # Load Lua script if not already loaded
        if not self.lua_sha:
            self.lua_sha = await self.redis.script_load(self.LUA_ATOMIC_EMIT)

        # Use provided TTL or default
        if ttl is None:
            ttl = self.ttl_seconds

        # Prepare keys and arguments
        idempotency_key = f'signals:emitted:{signal_id}'
        cooldown_key = f'signals:cooldown:{contract_fp}'
        queue_key = f'signals:pending:{symbol}'

        signal_json = json.dumps(signal)

        try:
            # Execute atomic operation
            result = await self.redis.evalsha(
                self.lua_sha,
                3,  # Number of keys
                idempotency_key,
                cooldown_key,
                queue_key,
                signal_json,
                str(ttl),
                str(self.cooldown_s)
            )
            return int(result)

        except redis.NoScriptError:
            # Script was evicted, reload and retry
            self.lua_sha = await self.redis.script_load(self.LUA_ATOMIC_EMIT)
            result = await self.redis.evalsha(
                self.lua_sha,
                3,
                idempotency_key,
                cooldown_key,
                queue_key,
                signal_json,
                str(ttl),
                str(self.cooldown_s)
            )
            return int(result)

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
        return {
            'id': emit_id,
            'contract_fp': contract_fp,
            'timestamp': int(time.time() * 1000),
            'symbol': symbol,
            'strategy': strategy,
            'side': side,
            'confidence': confidence / 100.0,  # Convert to 0-1 scale
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