#!/usr/bin/env python3
"""
Market Microstructure Analysis Module
Calculates institutional-grade metrics: VPIN, order book imbalance, toxicity, etc.
"""

import time
import json
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import logging

import redis
import numpy as np
import orjson
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


class RunningStats:
    """Efficient running statistics calculator"""
    
    def __init__(self, window_size: int = 1000):
        self.window = deque(maxlen=window_size)
        self.sum = 0.0
        self.sum_sq = 0.0
        
    def update(self, value: float):
        """Add new value and update statistics"""
        if len(self.window) == self.window.maxlen:
            # Remove oldest value
            old = self.window[0]
            self.sum -= old
            self.sum_sq -= old * old
            
        self.window.append(value)
        self.sum += value
        self.sum_sq += value * value
        
    @property
    def mean(self) -> float:
        return self.sum / len(self.window) if self.window else 0.0
        
    @property
    def std(self) -> float:
        if len(self.window) < 2:
            return 0.0
        variance = (self.sum_sq / len(self.window)) - (self.mean ** 2)
        return np.sqrt(max(0, variance))
        
    @property
    def percentiles(self) -> Dict[str, float]:
        if not self.window:
            return {'p50': 0, 'p95': 0, 'p99': 0}
        sorted_data = sorted(self.window)
        return {
            'p50': np.percentile(sorted_data, 50),
            'p95': np.percentile(sorted_data, 95),
            'p99': np.percentile(sorted_data, 99)
        }


class MarketMicrostructure:
    """Advanced market microstructure analytics"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.symbols = config['trading']['symbols']
        
        # VPIN calculation parameters (discovered dynamically)
        self.vpin_params = {
            symbol: {
                'bucket_size': 1000,  # Will be updated by parameter discovery
                'num_buckets': 50,
                'buckets': deque(maxlen=50),
                'total_volume': 0,
                'buy_volume': 0,
                'sell_volume': 0
            } for symbol in self.symbols
        }
        
        # Order book imbalance tracking
        self.book_imbalance = {
            symbol: {
                'volume_imbalance': RunningStats(100),
                'order_imbalance': RunningStats(100),
                'weighted_imbalance': RunningStats(100),
                'slope_imbalance': RunningStats(100)
            } for symbol in self.symbols
        }
        
        # Microstructure metrics
        self.micro_metrics = {
            symbol: {
                'spread': RunningStats(1000),
                'depth': RunningStats(1000),
                'resilience': RunningStats(100),
                'toxicity': RunningStats(100),
                'realized_spread': RunningStats(1000),
                'effective_spread': RunningStats(1000),
                'price_impact': RunningStats(1000)
            } for symbol in self.symbols
        }
        
        # Trade flow tracking for VPIN
        self.trade_flow = defaultdict(lambda: deque(maxlen=10000))
        
        # Book snapshots for resilience calculation
        self.book_snapshots = defaultdict(lambda: deque(maxlen=100))
        
        # Load discovered parameters if available
        self._load_discovered_parameters()
        
    def _load_discovered_parameters(self):
        """Load parameters from discovery module if available"""
        
        try:
            # Try to load discovered VPIN bucket size
            bucket_size = self.redis.get('discovered:vpin_bucket_size')
            if bucket_size:
                bucket_data = json.loads(bucket_size)
                for symbol in self.symbols:
                    if symbol in bucket_data:
                        self.vpin_params[symbol]['bucket_size'] = bucket_data[symbol]
                        logger.info(f"Loaded discovered VPIN bucket size for {symbol}: {bucket_data[symbol]}")
                        
        except Exception as e:
            logger.warning(f"Could not load discovered parameters: {e}")
            
    def process_book_snapshot(self, symbol: str, book: Dict):
        """Process order book snapshot and calculate microstructure metrics"""
        
        try:
            # Extract bids and asks
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            if not bids or not asks:
                return
                
            # Calculate spread metrics
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            # Update spread statistics
            self.micro_metrics[symbol]['spread'].update(spread)
            
            # Calculate depth at best
            bid_depth = sum(level['size'] for level in bids[:3])  # Top 3 levels
            ask_depth = sum(level['size'] for level in asks[:3])
            total_depth = bid_depth + ask_depth
            
            self.micro_metrics[symbol]['depth'].update(total_depth)
            
            # Calculate order book imbalance (multiple methods)
            self._calculate_book_imbalance(symbol, bids, asks)
            
            # Store snapshot for resilience calculation
            snapshot = {
                'time': time.time_ns(),
                'mid_price': mid_price,
                'spread': spread,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'bids': bids[:5],  # Top 5 levels
                'asks': asks[:5]
            }
            self.book_snapshots[symbol].append(snapshot)
            
            # Calculate resilience if we have enough snapshots
            if len(self.book_snapshots[symbol]) >= 10:
                resilience = self._calculate_resilience(symbol)
                self.micro_metrics[symbol]['resilience'].update(resilience)
                
            # Store metrics in Redis
            self._store_microstructure_metrics(symbol)
            
        except Exception as e:
            logger.error(f"Error processing book snapshot for {symbol}: {e}")
            
    def _calculate_book_imbalance(self, symbol: str, bids: List, asks: List):
        """Calculate various order book imbalance metrics"""
        
        # Volume imbalance (raw size)
        bid_volume = sum(level['size'] for level in bids)
        ask_volume = sum(level['size'] for level in asks)
        volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Order count imbalance
        order_imbalance = (len(bids) - len(asks)) / (len(bids) + len(asks)) if (len(bids) + len(asks)) > 0 else 0
        
        # Weighted imbalance (by distance from mid)
        mid_price = (bids[0]['price'] + asks[0]['price']) / 2
        
        weighted_bid = sum(
            level['size'] / (1 + abs(level['price'] - mid_price))
            for level in bids
        )
        weighted_ask = sum(
            level['size'] / (1 + abs(level['price'] - mid_price))
            for level in asks
        )
        weighted_imbalance = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask) if (weighted_bid + weighted_ask) > 0 else 0
        
        # Slope imbalance (depth decay rate)
        bid_slope = self._calculate_depth_slope(bids)
        ask_slope = self._calculate_depth_slope(asks)
        slope_imbalance = (bid_slope - ask_slope) / (abs(bid_slope) + abs(ask_slope)) if (abs(bid_slope) + abs(ask_slope)) > 0 else 0
        
        # Update statistics
        self.book_imbalance[symbol]['volume_imbalance'].update(volume_imbalance)
        self.book_imbalance[symbol]['order_imbalance'].update(order_imbalance)
        self.book_imbalance[symbol]['weighted_imbalance'].update(weighted_imbalance)
        self.book_imbalance[symbol]['slope_imbalance'].update(slope_imbalance)
        
        # Store in Redis
        imbalance_data = {
            'volume': volume_imbalance,
            'order': order_imbalance,
            'weighted': weighted_imbalance,
            'slope': slope_imbalance,
            'composite': (volume_imbalance + weighted_imbalance + slope_imbalance) / 3
        }
        
        self.redis.setex(
            f'market:{symbol}:book:imbalance',
            1,
            orjson.dumps(imbalance_data).decode('utf-8')
        )
        
    def _calculate_depth_slope(self, levels: List) -> float:
        """Calculate the slope of depth decay across price levels"""
        
        if len(levels) < 2:
            return 0.0
            
        # Use log of size to handle large variations
        sizes = [np.log(level['size'] + 1) for level in levels[:5]]  # Top 5 levels
        prices = [level['price'] for level in levels[:5]]
        
        # Calculate slope using linear regression
        if len(sizes) >= 2:
            slope, _ = np.polyfit(prices, sizes, 1)
            return slope
            
        return 0.0
        
    def _calculate_resilience(self, symbol: str) -> float:
        """Calculate order book resilience (speed of recovery after trades)"""
        
        snapshots = list(self.book_snapshots[symbol])
        
        if len(snapshots) < 10:
            return 0.0
            
        # Look for price moves and recovery
        resilience_scores = []
        
        for i in range(1, len(snapshots)):
            prev = snapshots[i-1]
            curr = snapshots[i]
            
            # Check if there was a price move
            price_change = abs(curr['mid_price'] - prev['mid_price'])
            
            if price_change > prev['spread']:  # Significant move
                # Check how quickly depth recovers
                depth_recovery = curr['bid_depth'] + curr['ask_depth'] - (prev['bid_depth'] + prev['ask_depth'])
                spread_recovery = prev['spread'] - curr['spread']
                
                # Resilience score (higher is better)
                score = (depth_recovery / (prev['bid_depth'] + prev['ask_depth'] + 1)) + \
                        (spread_recovery / (prev['spread'] + 0.01))
                        
                resilience_scores.append(score)
                
        return np.mean(resilience_scores) if resilience_scores else 0.0
        
    def process_trade(self, symbol: str, trade: Dict):
        """Process trade for VPIN calculation and impact analysis"""
        
        try:
            # Add to trade flow
            self.trade_flow[symbol].append(trade)
            
            # Update VPIN
            self._update_vpin(symbol, trade)
            
            # Calculate price impact
            self._calculate_price_impact(symbol, trade)
            
            # Calculate realized spread
            self._calculate_realized_spread(symbol, trade)
            
        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")
            
    def _update_vpin(self, symbol: str, trade: Dict):
        """Update VPIN (Volume-Synchronized Probability of Informed Trading)"""
        
        params = self.vpin_params[symbol]
        
        # Classify trade as buy or sell
        # Using tick rule: compare to previous trade
        is_buy = self._classify_trade_direction(symbol, trade)
        
        # Update volume counters
        trade_volume = trade.get('size', 0)
        params['total_volume'] += trade_volume
        
        if is_buy:
            params['buy_volume'] += trade_volume
        else:
            params['sell_volume'] += trade_volume
            
        # Check if we've filled a bucket
        if params['total_volume'] >= params['bucket_size']:
            # Calculate bucket imbalance
            bucket_imbalance = abs(params['buy_volume'] - params['sell_volume'])
            
            # Add to buckets
            params['buckets'].append({
                'imbalance': bucket_imbalance,
                'total_volume': params['total_volume'],
                'time': time.time()
            })
            
            # Reset counters
            params['total_volume'] = 0
            params['buy_volume'] = 0
            params['sell_volume'] = 0
            
            # Calculate VPIN if we have enough buckets
            if len(params['buckets']) >= 10:
                total_imbalance = sum(b['imbalance'] for b in params['buckets'])
                total_volume = sum(b['total_volume'] for b in params['buckets'])
                
                vpin = total_imbalance / total_volume if total_volume > 0 else 0
                
                # Adjust for market maker participation (reduces toxicity)
                mm_adjustment = self._get_mm_adjustment(symbol)
                adjusted_vpin = vpin * (1 - mm_adjustment)
                
                # Update toxicity metric
                self.micro_metrics[symbol]['toxicity'].update(adjusted_vpin)
                
                # Store VPIN in Redis
                vpin_data = {
                    'vpin': vpin,
                    'adjusted_vpin': adjusted_vpin,
                    'mm_adjustment': mm_adjustment,
                    'buckets': len(params['buckets']),
                    'time': time.time()
                }
                
                self.redis.setex(
                    f'metrics:{symbol}:vpin',
                    5,
                    orjson.dumps(vpin_data).decode('utf-8')
                )
                
    def _classify_trade_direction(self, symbol: str, trade: Dict) -> bool:
        """Classify trade as buy or sell using tick rule and quote rule"""
        
        # First try quote rule (most accurate)
        if 'bid' in trade and 'ask' in trade:
            mid = (trade['bid'] + trade['ask']) / 2
            if trade['price'] > mid:
                return True  # Buy
            elif trade['price'] < mid:
                return False  # Sell
                
        # Fall back to tick rule
        trade_flow = self.trade_flow[symbol]
        if len(trade_flow) >= 2:
            prev_price = trade_flow[-2].get('price', trade['price'])
            if trade['price'] > prev_price:
                return True  # Buy
            elif trade['price'] < prev_price:
                return False  # Sell
                
        # Default to buy if uncertain
        return True
        
    def _get_mm_adjustment(self, symbol: str) -> float:
        """Get market maker participation adjustment for VPIN"""
        
        # Check if market makers are actively providing liquidity
        mm_active = self.redis.get(f'market:{symbol}:mm:classification')
        
        if mm_active:
            mm_data = json.loads(mm_active)
            # More MMs = less toxic flow
            num_active_mms = len(mm_data.get('active_mms', []))
            return min(0.5, num_active_mms * 0.1)  # Up to 50% reduction
            
        return 0.0
        
    def _calculate_price_impact(self, symbol: str, trade: Dict):
        """Calculate the price impact of trades"""
        
        # Get pre-trade mid price
        book_snapshot = self.book_snapshots[symbol]
        if not book_snapshot:
            return
            
        pre_trade_mid = book_snapshot[-1]['mid_price'] if book_snapshot else trade['price']
        
        # Calculate impact (in basis points)
        impact = abs(trade['price'] - pre_trade_mid) / pre_trade_mid * 10000
        
        # Adjust for trade size
        size_adjusted_impact = impact * np.log(1 + trade.get('size', 1))
        
        self.micro_metrics[symbol]['price_impact'].update(size_adjusted_impact)
        
    def _calculate_realized_spread(self, symbol: str, trade: Dict):
        """Calculate realized spread (execution quality metric)"""
        
        # Get quote at time of trade
        if 'bid' in trade and 'ask' in trade:
            quoted_spread = trade['ask'] - trade['bid']
            mid = (trade['bid'] + trade['ask']) / 2
            
            # Realized spread = 2 * |trade_price - mid|
            realized = 2 * abs(trade['price'] - mid)
            
            # Effective spread (what was actually paid)
            effective = 2 * (trade['price'] - mid) if trade['price'] > mid else 2 * (mid - trade['price'])
            
            self.micro_metrics[symbol]['realized_spread'].update(realized)
            self.micro_metrics[symbol]['effective_spread'].update(effective)
            
    def process_bar(self, symbol: str, bar: Dict):
        """Process bar data for additional metrics"""
        
        try:
            # Calculate bar-based toxicity indicators
            if bar.get('volume', 0) > 0:
                # Volume-weighted average price deviation
                vwap = bar.get('wap', (bar['high'] + bar['low']) / 2)
                close = bar['close']
                
                # VWAP deviation indicates informed trading
                vwap_deviation = abs(close - vwap) / vwap
                
                # High-low range indicates volatility/uncertainty
                range_pct = (bar['high'] - bar['low']) / bar['low'] if bar['low'] > 0 else 0
                
                # Update toxicity with these indicators
                toxicity_score = vwap_deviation * 100 + range_pct * 50
                self.micro_metrics[symbol]['toxicity'].update(toxicity_score)
                
        except Exception as e:
            logger.error(f"Error processing bar for {symbol}: {e}")
            
    def _store_microstructure_metrics(self, symbol: str):
        """Store all microstructure metrics in Redis"""
        
        try:
            # Compile all metrics
            metrics = {
                'spread': {
                    'current': self.micro_metrics[symbol]['spread'].window[-1] if self.micro_metrics[symbol]['spread'].window else 0,
                    'mean': self.micro_metrics[symbol]['spread'].mean,
                    'std': self.micro_metrics[symbol]['spread'].std
                },
                'depth': {
                    'current': self.micro_metrics[symbol]['depth'].window[-1] if self.micro_metrics[symbol]['depth'].window else 0,
                    'mean': self.micro_metrics[symbol]['depth'].mean
                },
                'resilience': {
                    'score': self.micro_metrics[symbol]['resilience'].mean
                },
                'toxicity': {
                    'vpin_adjusted': self.micro_metrics[symbol]['toxicity'].mean,
                    'percentiles': self.micro_metrics[symbol]['toxicity'].percentiles
                },
                'price_impact': {
                    'mean_bps': self.micro_metrics[symbol]['price_impact'].mean,
                    'p95_bps': self.micro_metrics[symbol]['price_impact'].percentiles.get('p95', 0)
                },
                'spreads': {
                    'realized': self.micro_metrics[symbol]['realized_spread'].mean,
                    'effective': self.micro_metrics[symbol]['effective_spread'].mean
                },
                'timestamp': time.time_ns()
            }
            
            # Store in Redis
            pipe = self.redis.pipeline()
            
            # Store complete metrics
            pipe.setex(
                f'market:{symbol}:micro:metrics',
                5,
                orjson.dumps(metrics).decode('utf-8')
            )
            
            # Store individual metrics for quick access
            pipe.setex(f'market:{symbol}:micro:spread', 5, metrics['spread']['current'])
            pipe.setex(f'market:{symbol}:micro:depth', 5, metrics['depth']['current'])
            pipe.setex(f'market:{symbol}:micro:resilience', 5, metrics['resilience']['score'])
            pipe.setex(f'market:{symbol}:micro:toxicity', 5, metrics['toxicity']['vpin_adjusted'])
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Error storing microstructure metrics for {symbol}: {e}")