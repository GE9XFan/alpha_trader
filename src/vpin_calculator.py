#!/usr/bin/env python3
"""
VPIN Calculator - Volume-Synchronized Probability of Informed Trading
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- market:{symbol}:ticker: Market data for midpoint
- market:{symbol}:trades: Trade data
- analytics:{symbol}:vpin: VPIN metrics output
- discovered:vpin_bucket_size: Discovered optimal bucket size
"""

import numpy as np
import json
import redis.asyncio as aioredis
import time
from typing import Dict, List, Any
import logging
import traceback

import redis_keys as rkeys


class VPINCalculator:
    """
    Calculates VPIN (Volume-Synchronized Probability of Informed Trading).
    VPIN measures the probability of informed trading based on order flow imbalance.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize VPIN calculator."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # TTLs for storing metrics
        self.ttls = config.get('modules', {}).get('analytics', {}).get('store_ttls', {
            'metrics': 60,
            'alerts': 3600,
            'heartbeat': 15
        })

    async def calculate_vpin(self, symbol: str) -> float:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

        VPIN measures the probability of informed trading based on order flow imbalance.
        Higher VPIN (>0.7) indicates toxic/informed flow.
        Lower VPIN (<0.3) indicates uninformed/retail flow.

        Returns:
            VPIN score between 0 and 1
        """
        try:
            # 1. Get discovered bucket size from Redis
            bucket_size_str = await self.redis.get('discovered:vpin_bucket_size')
            if bucket_size_str:
                bucket_size = int(bucket_size_str)
                self.logger.debug(f"Using discovered bucket size: {bucket_size}")
            else:
                # Fallback to config default
                bucket_size = self.config.get('parameter_discovery', {}).get('vpin', {}).get('default_bucket_size', 100)
                self.logger.warning(f"No discovered bucket size, using default: {bucket_size}")

            # 2. Fetch recent trades from Redis (last 1000 trades)
            trades_json = await self.redis.lrange(rkeys.market_trades_key(symbol), -1000, -1)

            # Check minimum trades requirement
            min_trades = self.config.get('parameter_discovery', {}).get('vpin', {}).get('min_trades_required', 100)
            if len(trades_json) < min_trades:
                self.logger.debug(f"Insufficient trades for {symbol}: {len(trades_json)} < {min_trades}")
                return 0.5  # Neutral VPIN if insufficient data

            # 3. Parse trades
            trades = []
            for trade_str in trades_json:
                try:
                    trade = json.loads(trade_str)
                    trades.append(trade)
                except json.JSONDecodeError:
                    continue

            if not trades:
                return 0.5

            # 4. Classify trades as buy/sell
            classified_trades = await self._classify_trades(trades, symbol)

            # 5. Create volume buckets
            buckets = self._create_volume_buckets(classified_trades, bucket_size)

            # 6. Calculate VPIN from buckets
            vpin = self._calculate_vpin_from_buckets(buckets)

            # 7. Store in Redis with configured TTL
            ttl = self.ttls.get('analytics', self.ttls.get('metrics', 60))
            vpin_data = {
                'value': round(float(vpin), 4),
                'buckets': len(buckets),
                'bucket_size': bucket_size,
                'trades_analyzed': len(trades),
                'timestamp': time.time()
            }

            await self.redis.setex(
                rkeys.analytics_vpin_key(symbol),
                ttl,
                json.dumps(vpin_data)
            )

            # Log if VPIN is extreme
            if vpin > 0.7:
                self.logger.info(f"HIGH VPIN for {symbol}: {vpin:.3f} (toxic flow detected)")
            elif vpin < 0.3:
                self.logger.info(f"LOW VPIN for {symbol}: {vpin:.3f} (uninformed flow)")

            return float(vpin)

        except Exception as e:
            self.logger.error(f"Error calculating VPIN for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return 0.5  # Default neutral VPIN on error

    async def _classify_trades(self, trades: List[dict], symbol: str) -> List[dict]:
        """
        Classify trades as buy or sell using Lee-Ready algorithm.

        Lee-Ready Algorithm:
        1. If trade price > midpoint: BUY
        2. If trade price < midpoint: SELL
        3. If trade price = midpoint: Use tick test
           - If price > previous price: BUY
           - If price < previous price: SELL
           - If price = previous price: Use previous classification

        Returns trades with 'side' field added ('buy' or 'sell')
        """
        if not trades:
            return []

        classified = []
        prev_price = None
        prev_side = 'buy'  # Default assumption

        # Get current bid/ask for midpoint calculation
        ticker_json = await self.redis.get(rkeys.market_ticker_key(symbol))
        if ticker_json:
            ticker = json.loads(ticker_json)
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            midpoint = (bid + ask) / 2 if bid and ask else 0
        else:
            midpoint = 0

        # If no midpoint, try to estimate from trade prices
        if midpoint == 0 and trades:
            prices = [t.get('price', 0) for t in trades if t.get('price', 0) > 0]
            if prices:
                # Use median as proxy for midpoint
                midpoint = np.median(prices)

        buy_count = 0
        sell_count = 0

        for i, trade in enumerate(trades):
            price = trade.get('price', 0)

            # Lee-Ready classification
            if midpoint > 0:
                # Add small tolerance for floating point comparison
                tolerance = 0.001 * midpoint

                if price > midpoint + tolerance:
                    side = 'buy'
                elif price < midpoint - tolerance:
                    side = 'sell'
                else:
                    # Price at midpoint - use tick test
                    if prev_price is not None:
                        if price > prev_price:
                            side = 'buy'
                        elif price < prev_price:
                            side = 'sell'
                        else:
                            # Alternate between buy/sell to avoid all same classification
                            side = 'sell' if prev_side == 'buy' else 'buy'
                    else:
                        side = 'buy'  # Default for first trade
            else:
                # No midpoint available - use tick test with alternation
                if prev_price is not None:
                    if price > prev_price:
                        side = 'buy'
                    elif price < prev_price:
                        side = 'sell'
                    else:
                        # Alternate to ensure mix
                        side = 'sell' if prev_side == 'buy' else 'buy'
                else:
                    # Start with random to ensure variety
                    side = 'buy' if i % 2 == 0 else 'sell'

            # Track counts for balance check
            if side == 'buy':
                buy_count += 1
            else:
                sell_count += 1

            # Add classification to trade
            classified_trade = trade.copy()
            classified_trade['side'] = side
            classified.append(classified_trade)

            # Update for next iteration
            prev_price = price
            prev_side = side

        # Log if severely imbalanced (for debugging)
        total = len(classified)
        if total > 0:
            buy_ratio = buy_count / total
            if buy_ratio > 0.95 or buy_ratio < 0.05:
                self.logger.debug(f"Trade classification imbalanced for {symbol}: "
                                f"{buy_count} buys, {sell_count} sells")

        return classified

    def _create_volume_buckets(self, trades: List[dict], bucket_size: int) -> List[dict]:
        """
        Create volume-synchronized buckets for VPIN calculation.
        Each bucket contains trades until volume reaches bucket_size.

        Returns list of buckets with buy/sell volumes.
        """
        if not trades:
            return []

        buckets = []
        current_bucket = {
            'buy_volume': 0,
            'sell_volume': 0,
            'total_volume': 0,
            'trades': []
        }

        for trade in trades:
            size = trade.get('size', 0)
            side = trade.get('side', 'buy')

            # Check if adding this trade exceeds bucket size
            if current_bucket['total_volume'] + size > bucket_size and current_bucket['total_volume'] > 0:
                # Save current bucket
                buckets.append(current_bucket)
                # Start new bucket
                current_bucket = {
                    'buy_volume': 0,
                    'sell_volume': 0,
                    'total_volume': 0,
                    'trades': []
                }

            # Add trade to current bucket
            if side == 'buy':
                current_bucket['buy_volume'] += size
            else:
                current_bucket['sell_volume'] += size

            current_bucket['total_volume'] += size
            current_bucket['trades'].append(trade)

        # Add final bucket if it has trades
        if current_bucket['total_volume'] > 0:
            buckets.append(current_bucket)

        return buckets

    def _calculate_vpin_from_buckets(self, buckets: List[dict]) -> float:
        """
        Calculate VPIN from volume buckets.

        VPIN = Mean(|Buy Volume - Sell Volume| / Total Volume) across buckets

        Returns VPIN score between 0 and 1.
        """
        if not buckets:
            return 0.5  # Neutral VPIN

        # Need at least 50 buckets for reliable VPIN (from research)
        min_buckets = self.config.get('parameter_discovery', {}).get('vpin', {}).get('min_buckets_for_vpin', 50)

        if len(buckets) < min_buckets:
            # Not enough data - calculate simple imbalance
            total_buy = sum(b['buy_volume'] for b in buckets)
            total_sell = sum(b['sell_volume'] for b in buckets)
            total_volume = total_buy + total_sell

            if total_volume > 0:
                simple_vpin = abs(total_buy - total_sell) / total_volume
                # Weight by data availability (less data = closer to neutral)
                weight = len(buckets) / min_buckets
                return 0.5 + (simple_vpin - 0.5) * weight
            return 0.5

        # Calculate VPIN across all buckets
        imbalances = []
        for bucket in buckets:
            total = bucket['total_volume']
            if total > 0:
                imbalance = abs(bucket['buy_volume'] - bucket['sell_volume']) / total
                imbalances.append(imbalance)

        if imbalances:
            # VPIN is the mean absolute imbalance
            vpin = np.mean(imbalances)
            # Ensure VPIN is between 0 and 1
            return float(np.clip(vpin, 0.0, 1.0))

        return 0.5  # Neutral if no valid imbalances