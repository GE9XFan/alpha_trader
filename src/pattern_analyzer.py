#!/usr/bin/env python3
"""
Pattern Analyzer - Toxicity Detection and Order Book Analysis
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- market:{symbol}:book: Order book data
- market:{symbol}:trades: Trade data
- analytics:{symbol}:obi: Order book imbalance metrics
- analytics:{symbol}:hidden: Hidden order detection
- analytics:{symbol}:sweep: Sweep detection
- discovered:flow_toxicity: Flow toxicity patterns
"""

import numpy as np
import json
import redis.asyncio as aioredis
import time
from typing import Dict, List, Any, Optional
import logging
import traceback

from redis_keys import Keys


class PatternAnalyzer:
    """
    Analyzes trading patterns for toxicity, order book imbalance, and sweep detection.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize pattern analyzer."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # TTLs for storing metrics
        self.ttls = config.get('modules', {}).get('analytics', {}).get('store_ttls', {
            'metrics': 60,
            'alerts': 3600,
            'heartbeat': 15
        })

    async def analyze_flow_toxicity(self, symbol: str) -> dict:
        """
        Analyze flow toxicity from trading patterns.
        This is simplified from the full parameter discovery version.
        """
        try:
            # Get VPIN data as primary toxicity indicator
            vpin_json = await self.redis.get(Keys.analytics_vpin(symbol))
            if not vpin_json:
                return {'error': 'No VPIN data available'}

            vpin_data = json.loads(vpin_json)
            vpin_value = vpin_data.get('value', 0.5)

            # Classify toxicity based on VPIN
            if vpin_value > 0.7:
                toxicity_level = 'high'
                toxicity_score = vpin_value
            elif vpin_value > 0.5:
                toxicity_level = 'moderate'
                toxicity_score = vpin_value
            else:
                toxicity_level = 'low'
                toxicity_score = vpin_value

            result = {
                'symbol': symbol,
                'toxicity_score': round(toxicity_score, 4),
                'toxicity_level': toxicity_level,
                'vpin': vpin_value,
                'timestamp': time.time()
            }

            # Store result
            await self.redis.setex(
                Keys.analytics_toxicity(symbol),
                self.ttls.get('analytics', self.ttls.get('metrics', 60)),
                json.dumps(result)
            )

            if toxicity_level == 'high':
                self.logger.info(f"HIGH flow toxicity for {symbol}: {toxicity_score:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing flow toxicity for {symbol}: {e}")
            return {'error': str(e)}

    async def calculate_order_book_imbalance(self, symbol: str) -> dict:
        """
        Calculate comprehensive order book imbalance metrics.

        These metrics help identify:
        - Short-term price direction (micro-price)
        - Buying/selling pressure (imbalance)
        - Potential support/resistance (pressure ratio)

        Returns:
            Dictionary with multiple imbalance metrics
        """
        try:
            # 1. Fetch current order book
            book_json = await self.redis.get(Keys.market_book(symbol))
            if not book_json:
                return {'error': 'No order book data'}

            book = json.loads(book_json)

            # 2. Calculate various imbalance metrics
            l1_imbalance = self._calculate_l1_imbalance(book)
            l5_imbalance = self._calculate_l5_imbalance(book)
            pressure_ratio = self._calculate_pressure_ratio(book)
            micro_price = self._calculate_micro_price(book)

            # 3. Calculate book velocity (requires history)
            # For now, store current imbalance for future velocity calculation
            velocity_key = f'obi_history:{symbol}'
            history_json = await self.redis.get(velocity_key)

            if history_json:
                history = json.loads(history_json)
            else:
                history = []

            # Add current imbalance to history
            current_data = {
                'l1': l1_imbalance,
                'l5': l5_imbalance,
                'timestamp': time.time()
            }
            history.append(current_data)

            # Keep only last 20 samples (10 seconds at 2Hz)
            history = history[-20:]

            # Calculate velocity if we have enough history
            book_velocity = 0.0
            if len(history) >= 5:
                # Rate of change of imbalance
                old_l5 = history[-5]['l5']
                new_l5 = history[-1]['l5']
                time_diff = history[-1]['timestamp'] - history[-5]['timestamp']

                if time_diff > 0:
                    book_velocity = (new_l5 - old_l5) / time_diff

            # Store history for next calculation (short TTL for velocity tracking)
            velocity_ttl = min(30, self.ttls.get('analytics', self.ttls.get('metrics', 60)))  # Use shorter of 30s or config
            await self.redis.setex(velocity_key, velocity_ttl, json.dumps(history))

            # 4. Determine market state based on metrics
            if abs(l5_imbalance) > 0.7:
                state = 'extreme_imbalance'
            elif abs(l5_imbalance) > 0.4:
                state = 'moderate_imbalance'
            else:
                state = 'balanced'

            # 5. Prepare result
            result = {
                'level1_imbalance': round(l1_imbalance, 4),
                'level5_imbalance': round(l5_imbalance, 4),
                'pressure_ratio': round(pressure_ratio, 2),
                'micro_price': round(micro_price, 2) if micro_price else 0,
                'book_velocity': round(book_velocity, 4),
                'state': state,
                'timestamp': time.time()
            }

            # 6. Store in Redis
            ttl = self.ttls.get('analytics', self.ttls.get('metrics', 60))
            await self.redis.setex(
                Keys.analytics_obi(symbol),
                ttl,
                json.dumps(result)
            )

            # Log extreme imbalances
            if abs(l5_imbalance) > 0.7:
                direction = 'BID' if l5_imbalance > 0 else 'ASK'
                self.logger.info(f"EXTREME {direction} imbalance for {symbol}: L5={l5_imbalance:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Error calculating OBI for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

    def _calculate_l1_imbalance(self, book: dict) -> float:
        """Calculate Level 1 (best bid/ask) imbalance."""
        try:
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            if not bids or not asks:
                return 0.0

            best_bid_size = bids[0].get('size', 0) if bids else 0
            best_ask_size = asks[0].get('size', 0) if asks else 0

            total = best_bid_size + best_ask_size
            if total == 0:
                return 0.0

            # Imbalance: positive = more bids, negative = more asks
            return (best_bid_size - best_ask_size) / total

        except Exception:
            return 0.0

    def _calculate_l5_imbalance(self, book: dict) -> float:
        """Calculate Level 5 (top 5 levels) weighted imbalance."""
        try:
            bids = book.get('bids', [])[:5]
            asks = book.get('asks', [])[:5]

            if not bids or not asks:
                return 0.0

            # Weight by inverse distance from mid (closer levels have more weight)
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]

            weighted_bid_volume = 0
            weighted_ask_volume = 0

            for i, (bid, ask) in enumerate(zip(bids[:5], asks[:5])):
                weight = weights[i] if i < len(weights) else 0.1
                weighted_bid_volume += bid.get('size', 0) * weight
                weighted_ask_volume += ask.get('size', 0) * weight

            total = weighted_bid_volume + weighted_ask_volume
            if total == 0:
                return 0.0

            return (weighted_bid_volume - weighted_ask_volume) / total

        except Exception:
            return 0.0

    def _calculate_pressure_ratio(self, book: dict) -> float:
        """Calculate bid/ask pressure ratio."""
        try:
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            # Sum all visible bid and ask volumes
            total_bid_volume = sum(level.get('size', 0) for level in bids)
            total_ask_volume = sum(level.get('size', 0) for level in asks)

            if total_ask_volume == 0:
                return 10.0 if total_bid_volume > 0 else 1.0

            return total_bid_volume / total_ask_volume

        except Exception:
            return 1.0

    def _calculate_micro_price(self, book: dict) -> Optional[float]:
        """
        Calculate micro-price: size-weighted mid price.
        Better predictor of short-term price movement than simple mid.
        """
        try:
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            if not bids or not asks:
                return None

            best_bid = bids[0].get('price', 0)
            best_ask = asks[0].get('price', 0)
            best_bid_size = bids[0].get('size', 0)
            best_ask_size = asks[0].get('size', 0)

            if best_bid_size + best_ask_size == 0:
                return (best_bid + best_ask) / 2

            # Micro-price: weighted by opposite side size
            # Intuition: larger ask size pushes price toward bid
            micro_price = (best_bid * best_ask_size + best_ask * best_bid_size) / (best_bid_size + best_ask_size)

            return micro_price

        except Exception:
            return None

    async def detect_sweeps(self, symbol: str) -> float:
        """
        Detect sweep orders - rapid executions across multiple price levels.
        Returns 0 or 1 (binary detection).
        """
        try:
            # Get recent trades (last 1 second worth)
            trades_json = await self.redis.lrange(f'market:{symbol}:trades', -50, -1)

            if len(trades_json) < 3:
                return 0

            # Parse trades
            trades = []
            current_time = time.time() * 1000

            for trade_str in trades_json:
                try:
                    trade = json.loads(trade_str)
                    # Only consider trades from last 1 second
                    trade_time = trade.get('time', trade.get('timestamp', 0))
                    if current_time - trade_time <= 1000:
                        trades.append(trade)
                except (json.JSONDecodeError, KeyError):
                    continue

            if len(trades) < 3:
                return 0

            # Check for sweep characteristics
            prices = [t.get('price', 0) for t in trades]
            sizes = [t.get('size', 0) for t in trades]

            # Sweep detection criteria:
            # 1. At least 3 trades in 1 second
            # 2. Prices walk at least 3 levels
            # 3. Cumulative size > threshold
            unique_prices = len(set(prices))
            total_size = sum(sizes)

            # Price direction consistency
            if len(prices) >= 3:
                price_moves = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
                same_direction = all(m >= 0 for m in price_moves) or all(m <= 0 for m in price_moves)
            else:
                same_direction = False

            # Detect sweep
            is_sweep = (
                unique_prices >= 3 and
                total_size > 1000 and
                same_direction
            )

            # Store result
            sweep_value = 1.0 if is_sweep else 0.0
            await self.redis.setex(Keys.analytics_metric(symbol, 'sweep'), 5, str(sweep_value))

            if is_sweep:
                self.logger.info(f"SWEEP detected for {symbol}: {unique_prices} levels, {total_size} shares")

            return sweep_value

        except Exception as e:
            self.logger.error(f"Error detecting sweep for {symbol}: {e}")
            return 0

    async def detect_hidden_orders(self, symbol: str) -> dict:
        """
        Detect potential hidden/iceberg orders from market patterns.

        Hidden orders are detected by:
        1. Trades executing beyond best bid/ask (dark pool)
        2. Persistent refills at same price level (iceberg)
        3. Volume spikes without corresponding book depth

        Returns:
            Dictionary with hidden order detection results
        """
        try:
            # 1. Get current order book and recent trades
            book_json = await self.redis.get(f'market:{symbol}:book')
            trades_json = await self.redis.lrange(f'market:{symbol}:trades', -100, -1)

            if not book_json or not trades_json:
                return {'error': 'Insufficient data'}

            book = json.loads(book_json)
            trades = [json.loads(t) for t in trades_json]

            if not trades:
                return {'error': 'No trades'}

            # Extract best bid/ask
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            if not bids or not asks:
                return {'error': 'Invalid book'}

            best_bid = bids[0].get('price', 0)
            best_ask = asks[0].get('price', 0)

            # 2. Detect trades beyond NBBO (potential hidden orders)
            hidden_trades = []
            for trade in trades[-50:]:  # Last 50 trades
                price = trade.get('price', 0)
                size = trade.get('size', 0)

                # Trade outside NBBO suggests hidden liquidity
                if price > best_ask * 1.001 or price < best_bid * 0.999:
                    hidden_trades.append({
                        'price': price,
                        'size': size,
                        'type': 'dark' if price > best_ask or price < best_bid else 'midpoint'
                    })

            # 3. Detect iceberg orders (persistent refills)
            price_refills = {}
            for i in range(1, len(trades)):
                if trades[i].get('price') == trades[i-1].get('price'):
                    price = trades[i].get('price')
                    if price not in price_refills:
                        price_refills[price] = 0
                    price_refills[price] += 1

            # Identify potential icebergs (>5 refills at same price)
            icebergs = []
            for price, refills in price_refills.items():
                if refills >= 5:
                    icebergs.append({
                        'price': price,
                        'refills': refills,
                        'confidence': min(refills / 10, 1.0)  # Confidence score
                    })

            # 4. Detect volume spikes without depth
            recent_volume = sum(t.get('size', 0) for t in trades[-20:])
            visible_depth = sum(l.get('size', 0) for l in bids[:5]) + \
                           sum(l.get('size', 0) for l in asks[:5])

            volume_depth_ratio = recent_volume / visible_depth if visible_depth > 0 else 0

            # High ratio suggests hidden liquidity
            hidden_liquidity_likely = volume_depth_ratio > 2.0

            # 5. Calculate hidden order probability
            hidden_score = 0.0

            # Weight different signals
            if hidden_trades:
                hidden_score += 0.4 * min(len(hidden_trades) / 10, 1.0)
            if icebergs:
                hidden_score += 0.4 * min(len(icebergs) / 3, 1.0)
            if hidden_liquidity_likely:
                hidden_score += 0.2

            # 6. Prepare result
            result = {
                'hidden_trades': len(hidden_trades),
                'hidden_trade_volume': sum(t['size'] for t in hidden_trades),
                'iceberg_levels': len(icebergs),
                'top_icebergs': sorted(icebergs, key=lambda x: x['confidence'], reverse=True)[:3],
                'volume_depth_ratio': round(volume_depth_ratio, 2),
                'hidden_score': round(hidden_score, 3),
                'hidden_likely': hidden_score > 0.5,
                'timestamp': time.time()
            }

            # 7. Store in Redis
            ttl = self.ttls.get('analytics', self.ttls.get('metrics', 60))
            await self.redis.setex(
                Keys.analytics_metric(symbol, 'hidden'),
                ttl,
                json.dumps(result)
            )

            # Log significant hidden order detection
            if hidden_score > 0.7:
                self.logger.info(f"HIGH hidden order probability for {symbol}: score={hidden_score:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Error detecting hidden orders for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}