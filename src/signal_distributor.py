"""
Signal Distribution Module

Manages the distribution of trading signals to different subscription tiers
with appropriate delays. Implements persistent scheduling to survive process
restarts and ensures reliable signal delivery.

Redis Keys Used:
    Read:
        - signals:distribution:pending (executed signal queue)
        - distribution:scheduled:basic (scheduled basic tier signals)
        - distribution:scheduled:free (scheduled free tier signals)
    Write:
        - distribution:premium:queue (premium signals)
        - distribution:basic:queue (basic signals)
        - distribution:free:queue (free signals)
        - distribution:scheduled:basic (scheduled basic signals)
        - distribution:scheduled:free (scheduled free signals)

Author: QuantiCity Capital
Version: 3.0.0
"""

import asyncio
import json
import time
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
import pytz

import redis_keys as rkeys
from redis_keys import get_system_key


class SignalDistributor:
    """
    Distribute signals to different subscription tiers with appropriate delays.
    Manages signal queuing and delivery to various platforms.
    """

    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize signal distributor with configuration.

        Args:
            config: System configuration
            redis_conn: Redis connection for queue management
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Signal configuration
        self.signal_config = config.get('modules', {}).get('signals', {})
        self.enabled = self.signal_config.get('enabled', False)

        self.validator = SignalValidator(config, redis_conn)
        self.dead_letter_key = 'distribution:dead_letter'
        self._consecutive_failures = 0

        # Execution statuses that qualify for downstream fan-out
        self.allowed_statuses = {'FILLED', 'CLOSED', 'PARTIAL', 'ADJUSTMENT'}

        # Distribution tiers from config
        distribution_config = self.signal_config.get('distribution', {})
        self.tiers = distribution_config.get('tiers', {
            'premium': {'delay_seconds': 0, 'include_all_details': True},
            'basic': {'delay_seconds': 60, 'include_all_details': False},
            'free': {'delay_seconds': 300, 'include_all_details': False}
        })

    async def start(self):
        """
        Main distribution loop for signals.
        Processing frequency: Every 1 second
        """
        if not self.enabled:
            self.logger.info("Signal distributor disabled in config")
            return

        self.logger.info("Starting signal distributor...")

        # Start the scheduler task for delayed signals
        asyncio.create_task(self.process_scheduled_signals())

        distribution_queue = 'signals:distribution:pending'

        while True:
            try:
                result = await self.redis.brpop(distribution_queue, timeout=2)
                if not result:
                    await asyncio.sleep(0.2)
                    continue

                _, signal_payload = result
                try:
                    if isinstance(signal_payload, bytes):
                        signal_payload = signal_payload.decode('utf-8')
                    signal = json.loads(signal_payload)
                    execution_info = signal.get('execution') or {}
                    status = str(execution_info.get('status', '')).upper()
                    if status not in self.allowed_statuses:
                        self.logger.debug(
                            "Skipping signal due to unsupported execution status",
                            extra={'signal_id': signal.get('id'), 'status': status}
                        )
                        continue

                    await self.distribute_signal(signal)
                    self._consecutive_failures = 0
                except Exception as exc:  # pragma: no cover - defensive guard
                    self._consecutive_failures += 1
                    await self.redis.lpush(self.dead_letter_key, signal_payload)
                    backoff = min(5, 0.5 * self._consecutive_failures)
                    self.logger.error("Failed to distribute signal", exc_info=exc)
                    if backoff:
                        await asyncio.sleep(backoff)

            except Exception as e:
                self.logger.error(f"Error in distribution loop: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(1)

    async def process_scheduled_signals(self):
        """
        Process scheduled signals from Redis sorted sets.
        This ensures signals are published even after process restarts.
        """
        self.logger.info("Starting scheduled signal processor...")

        while True:
            try:
                current_time = time.time()

                # Process basic tier scheduled signals
                basic_ready = await self.redis.zrangebyscore(
                    'distribution:scheduled:basic',
                    min=0,
                    max=current_time,
                    withscores=False,
                    start=0,
                    num=100
                )

                for signal_json in basic_ready:
                    await self.redis.lpush('distribution:basic:queue', signal_json)
                    await self.redis.zrem('distribution:scheduled:basic', signal_json)
                    self.logger.debug(f"Published scheduled basic signal")
                    await self.redis.incr('metrics:signals:distributed:basic')

                # Process free tier scheduled signals
                free_ready = await self.redis.zrangebyscore(
                    'distribution:scheduled:free',
                    min=0,
                    max=current_time,
                    withscores=False,
                    start=0,
                    num=100
                )

                for signal_json in free_ready:
                    await self.redis.lpush('distribution:free:queue', signal_json)
                    await self.redis.zrem('distribution:scheduled:free', signal_json)
                    self.logger.debug(f"Published scheduled free signal")
                    await self.redis.incr('metrics:signals:distributed:free')

                # Check every second
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error processing scheduled signals: {e}")
                await asyncio.sleep(1)

    async def distribute_signal(self, signal: Dict[str, Any]):
        """
        Distribute signal to appropriate tiers with delays.
        Uses Redis sorted sets for persistent scheduling to survive restarts.

        Args:
            signal: Signal to distribute
        """
        try:
            symbol = signal.get('symbol')
            if not symbol:
                self.logger.warning("Signal missing symbol, skipping distribution")
                return

            if 'ts' not in signal:
                if signal.get('timestamp'):
                    signal['ts'] = signal['timestamp']
                else:
                    signal['ts'] = int(time.time() * 1000)

            if not self.validator.validate_signal(signal):
                await self.redis.incr('metrics:signals:validator_rejects')
                return

            if not signal.get('execution'):
                if not await self.validator.validate_market_conditions(symbol):
                    await self.redis.incr('metrics:signals:validator_rejects')
                    return

            current_time = time.time()

            # Premium tier - immediate
            premium_signal = self.format_premium_signal(signal)
            await self.redis.lpush('distribution:premium:queue', json.dumps(premium_signal))
            await self.redis.incr('metrics:signals:distributed:premium')

            # Basic tier - 60s delay (use sorted set for persistence)
            basic_signal = self.format_basic_signal(signal)
            basic_delay = self.tiers.get('basic', {}).get('delay_seconds', 60)
            basic_publish_time = current_time + basic_delay
            await self.redis.zadd(
                'distribution:scheduled:basic',
                {json.dumps(basic_signal): basic_publish_time}
            )
            await self.redis.incr('metrics:signals:scheduled:basic')

            # Free tier - 300s delay (use sorted set for persistence)
            free_signal = self.format_free_signal(signal)
            free_delay = self.tiers.get('free', {}).get('delay_seconds', 300)
            free_publish_time = current_time + free_delay
            await self.redis.zadd(
                'distribution:scheduled:free',
                {json.dumps(free_signal): free_publish_time}
            )
            await self.redis.incr('metrics:signals:scheduled:free')

            self.logger.info(f"Distributed signal {signal['id']} for {signal['symbol']}")

        except Exception as e:
            self.logger.error(f"Error distributing signal: {e}")

    def format_premium_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format signal with full details for premium subscribers.

        Args:
            signal: Original signal

        Returns:
            Premium formatted signal with all details
        """
        formatted = dict(signal)
        formatted.setdefault('action_type', signal.get('action_type', 'ENTRY'))
        formatted.setdefault('id', signal.get('id'))
        return formatted

    def format_basic_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format signal with limited details for basic subscribers.

        Args:
            signal: Original signal

        Returns:
            Basic formatted signal with confidence bands
        """
        # Determine confidence band
        confidence = signal.get('confidence', 0)
        if confidence >= 80:
            conf_band = 'HIGH'
        elif confidence >= 65:
            conf_band = 'MEDIUM'
        else:
            conf_band = 'LOW'

        execution = self._sanitize_execution(signal.get('execution'))
        lifecycle = self._sanitize_lifecycle(signal.get('lifecycle'))

        return {
            'id': signal.get('id'),
            'symbol': signal.get('symbol'),
            'side': signal.get('side'),
            'strategy': signal.get('strategy'),
            'confidence_band': conf_band,
            'action_type': signal.get('action_type', 'ENTRY'),
            'ts': signal.get('ts'),
            'execution': execution,
            'lifecycle': lifecycle,
        }

    def format_free_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format signal teaser for free tier.

        Args:
            signal: Original signal

        Returns:
            Free tier teaser with minimal details
        """
        side = signal.get('side', '')
        sentiment = 'bullish' if side == 'LONG' else 'bearish' if side == 'SHORT' else 'neutral'

        lifecycle = self._sanitize_lifecycle(signal.get('lifecycle'))

        return {
            'id': signal.get('id'),
            'symbol': signal.get('symbol'),
            'sentiment': sentiment,
            'message': f"New {sentiment} signal on {signal.get('symbol')}. Upgrade for full details!",
            'action_type': signal.get('action_type', 'ENTRY'),
            'ts': signal.get('ts'),
            'execution': self._sanitize_execution(signal.get('execution')),
            'lifecycle': lifecycle,
        }

    @staticmethod
    def _sanitize_execution(execution: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not execution:
            return None

        sanitized = {
            'status': execution.get('status'),
            'executed_at': execution.get('executed_at'),
        }

        return sanitized

    @staticmethod
    def _sanitize_lifecycle(lifecycle: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not lifecycle:
            return None

        sanitized = {
            key: lifecycle.get(key)
            for key in (
                'result',
                'return_pct',
                'holding_period_minutes',
                'reason',
                'remaining_quantity',
            )
            if lifecycle.get(key) is not None
        }

        return sanitized or None


class SignalValidator:
    """
    Validate signals before distribution to ensure quality.
    """

    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize signal validator.

        Args:
            config: System configuration
            redis_conn: Redis connection
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Signal configuration
        self.signal_config = config.get('modules', {}).get('signals', {})
        distribution_config = self.signal_config.get('distribution', {})
        self.max_spread_bps = distribution_config.get('max_spread_bps', 50)
        self.min_daily_volume = distribution_config.get('min_daily_volume', 200000)

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate signal meets quality standards.

        Args:
            signal: Signal to validate

        Returns:
            True if valid, False otherwise
        """
        # Check confidence
        min_conf_raw = self.signal_config.get('min_confidence', 0.60)
        min_confidence = min_conf_raw if min_conf_raw > 1 else min_conf_raw * 100
        if float(signal.get('confidence', 0)) < min_confidence:
            return False

        # Check stop distance
        entry = signal.get('entry', 0)
        stop = signal.get('stop', 0)
        if entry and stop:
            stop_distance = abs(stop - entry) / entry
            if stop_distance > 0.05:  # 5% max
                return False

        # Check risk/reward
        targets = signal.get('targets', [])
        if targets and stop and entry:
            reward = abs(targets[0] - entry)
            risk = abs(entry - stop)
            if risk > 0 and reward / risk < 1.5:
                return False

        return True

    async def validate_market_conditions(self, symbol: str) -> bool:
        """
        Check if market conditions are suitable for trading.

        Args:
            symbol: Symbol to validate

        Returns:
            True if conditions suitable, False otherwise
        """
        # Check market hours
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)

        if current_time.weekday() >= 5:  # Weekend
            return False

        from datetime import time as datetime_time
        market_open = datetime_time(9, 30)
        market_close = datetime_time(16, 0)
        current_time_only = current_time.time()

        if not (market_open <= current_time_only <= market_close):
            # Check if extended hours are enabled
            if not self.config.get('market', {}).get('extended_hours', False):
                return False

        # Global halt checks
        system_halt = await self.redis.get(get_system_key('halt'))
        system_halt_val = system_halt.decode('utf-8', errors='ignore') if isinstance(system_halt, bytes) else system_halt
        if system_halt_val and str(system_halt_val).lower() not in ('false', '0', 'none'):
            return False

        risk_halt = await self.redis.get('risk:halt:status')
        if risk_halt and str(risk_halt).lower() in ('true', '1'):
            return False

        ticker_raw = await self.redis.get(rkeys.market_ticker_key(symbol))
        if ticker_raw:
            if isinstance(ticker_raw, bytes):
                ticker_raw = ticker_raw.decode('utf-8', errors='ignore')
            try:
                ticker = json.loads(ticker_raw)
            except json.JSONDecodeError:
                ticker = {}

            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if bid and ask:
                try:
                    mid = (float(bid) + float(ask)) / 2
                    spread_bps = abs(float(ask) - float(bid)) / max(mid, 1e-6) * 10000
                    if spread_bps > self.max_spread_bps:
                        return False
                except (TypeError, ValueError):
                    pass

            volume = ticker.get('volume') or ticker.get('dayVolume')
            try:
                if volume is not None and float(volume) < self.min_daily_volume:
                    return False
            except (TypeError, ValueError):
                pass

        return True
