"""
Signal Distribution Module

Manages the distribution of trading signals to different subscription tiers
with appropriate delays. Implements persistent scheduling to survive process
restarts and ensures reliable signal delivery.

Redis Keys Used:
    Read:
        - signals:pending:{symbol} (pending signals queue)
        - distribution:scheduled:basic (scheduled basic tier signals)
        - distribution:scheduled:free (scheduled free tier signals)
    Write:
        - distribution:premium:queue (premium signals)
        - distribution:basic:queue (basic signals)
        - distribution:free:queue (free signals)
        - distribution:scheduled:basic (scheduled basic signals)
        - distribution:scheduled:free (scheduled free signals)

Author: AlphaTrader Pro
Version: 3.0.0
"""

import asyncio
import json
import time
import logging
import traceback
from typing import Dict, Any, List
from datetime import datetime
import pytz


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

        while True:
            try:
                # Get symbols from configuration (bug fix #5)
                level2_symbols = self.config.get('symbols', {}).get('level2', [])
                standard_symbols = self.config.get('symbols', {}).get('standard', [])
                all_symbols = list(set(level2_symbols + standard_symbols))

                # Build list of all pending queues
                pending_queues = [f'signals:pending:{symbol}' for symbol in all_symbols]

                if pending_queues:
                    # Use BRPOP with multiple queues and timeout (bug fix #1)
                    result = await self.redis.brpop(pending_queues, timeout=2)
                    if result:
                        queue_name, signal_json = result
                        signal = json.loads(signal_json)
                        await self.distribute_signal(signal)
                else:
                    # No symbols configured, sleep longer
                    await asyncio.sleep(5)

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
            current_time = time.time()

            # Premium tier - immediate
            premium_signal = self.format_premium_signal(signal)
            await self.redis.lpush('distribution:premium:queue', json.dumps(premium_signal))

            # Basic tier - 60s delay (use sorted set for persistence)
            basic_signal = self.format_basic_signal(signal)
            basic_delay = self.tiers.get('basic', {}).get('delay_seconds', 60)
            basic_publish_time = current_time + basic_delay
            await self.redis.zadd(
                'distribution:scheduled:basic',
                {json.dumps(basic_signal): basic_publish_time}
            )

            # Free tier - 300s delay (use sorted set for persistence)
            free_signal = self.format_free_signal(signal)
            free_delay = self.tiers.get('free', {}).get('delay_seconds', 300)
            free_publish_time = current_time + free_delay
            await self.redis.zadd(
                'distribution:scheduled:free',
                {json.dumps(free_signal): free_publish_time}
            )

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
        # Premium gets everything
        return signal

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

        return {
            'symbol': signal.get('symbol'),
            'side': signal.get('side'),
            'strategy': signal.get('strategy'),
            'confidence_band': conf_band,
            'ts': signal.get('ts')
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

        return {
            'symbol': signal.get('symbol'),
            'sentiment': sentiment,
            'message': f"New {sentiment} signal on {signal.get('symbol')}. Upgrade for full details!",
            'ts': signal.get('ts')
        }


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

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate signal meets quality standards.

        Args:
            signal: Signal to validate

        Returns:
            True if valid, False otherwise
        """
        # Check confidence
        min_confidence = self.signal_config.get('min_confidence', 0.60) * 100
        if signal.get('confidence', 0) < min_confidence:
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

        # TODO: Check for halts, spread, liquidity

        return True