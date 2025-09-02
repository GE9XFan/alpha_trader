#!/usr/bin/env python3
"""
Hidden Order Detector Module
Detects iceberg orders, reserve orders, and other hidden liquidity
"""

import time
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional
import logging

import redis
import orjson
import numpy as np

logger = logging.getLogger(__name__)


class HiddenOrderDetector:
    """Detects hidden orders using multiple signals"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        
        # Detection signals per symbol
        self.detection_signals = defaultdict(lambda: {
            'between_spread': deque(maxlen=100),
            'size_refreshes': defaultdict(int),
            'time_violations': deque(maxlen=50),
            'pegged_patterns': deque(maxlen=100),
            'iceberg_candidates': {}
        })
        
        # Track executions for hidden order detection
        self.recent_executions = defaultdict(lambda: deque(maxlen=1000))
        
    def analyze_depth_update(self, symbol: str, depth):
        """Analyze depth update for hidden order signals"""
        
        # Track size refreshes at same price level
        price_key = f"{depth.price:.2f}"
        side = 'bid' if depth.side == 0 else 'ask'
        
        signals = self.detection_signals[symbol]
        
        # Check for size refresh pattern (iceberg indicator)
        if price_key in signals['size_refreshes']:
            if depth.operation == 1:  # Update operation
                signals['size_refreshes'][price_key] += 1
                
                # Multiple refreshes indicate iceberg
                if signals['size_refreshes'][price_key] >= 3:
                    self._flag_iceberg_candidate(symbol, depth.price, side)
                    
    def analyze_trade_vs_book(self, symbol: str, trade: Dict, book: Dict):
        """Detect hidden orders by comparing trades to visible book"""
        
        signals = self.detection_signals[symbol]
        
        # Signal 1: Trade between spread with no visible liquidity
        if self._trade_between_spread(trade, book):
            signals['between_spread'].append({
                'price': trade['price'],
                'size': trade['size'],
                'time': time.time()
            })
            
        # Signal 2: Large trade with minimal impact
        if self._large_trade_no_impact(trade, book):
            self._flag_hidden_liquidity(symbol, trade)
            
        # Store execution for pattern analysis
        self.recent_executions[symbol].append(trade)
        
        # Analyze execution patterns
        if len(self.recent_executions[symbol]) >= 10:
            self._analyze_execution_patterns(symbol)
            
    def _trade_between_spread(self, trade: Dict, book: Dict) -> bool:
        """Check if trade executed between the spread"""
        
        if not book.get('bids') or not book.get('asks'):
            return False
            
        best_bid = book['bids'][0]['price']
        best_ask = book['asks'][0]['price']
        
        return best_bid < trade['price'] < best_ask
        
    def _large_trade_no_impact(self, trade: Dict, book: Dict) -> bool:
        """Check if large trade had minimal price impact"""
        
        if trade['size'] < 1000:  # Not large enough
            return False
            
        # Would need pre/post trade book snapshots for accurate detection
        # Simplified check: large trade at same price level
        return True
        
    def _flag_iceberg_candidate(self, symbol: str, price: float, side: str):
        """Flag potential iceberg order"""
        
        signals = self.detection_signals[symbol]
        
        price_key = f"{price:.2f}"
        signals['iceberg_candidates'][price_key] = {
            'side': side,
            'confidence': 0.75,
            'detected_time': time.time(),
            'refresh_count': signals['size_refreshes'][price_key]
        }
        
        # Publish detection
        self._publish_hidden_order_signal(symbol, 'iceberg', price, side)
        
    def _flag_hidden_liquidity(self, symbol: str, trade: Dict):
        """Flag hidden liquidity detection"""
        
        self._publish_hidden_order_signal(symbol, 'hidden_liquidity', 
                                         trade['price'], 
                                         'buy' if trade['price'] > trade.get('mid', trade['price']) else 'sell')
                                         
    def _analyze_execution_patterns(self, symbol: str):
        """Analyze recent executions for hidden order patterns"""
        
        executions = list(self.recent_executions[symbol])[-50:]
        
        # Look for consistent execution at specific price levels
        price_counts = defaultdict(int)
        for exec in executions:
            price_key = f"{exec['price']:.2f}"
            price_counts[price_key] += 1
            
        # Persistent execution at same price indicates hidden order
        for price_key, count in price_counts.items():
            if count >= 10:
                price = float(price_key)
                self._flag_iceberg_candidate(symbol, price, 'unknown')
                
    def _publish_hidden_order_signal(self, symbol: str, order_type: str, price: float, side: str):
        """Publish hidden order detection to Redis"""
        
        signal = {
            'symbol': symbol,
            'type': order_type,
            'price': price,
            'side': side,
            'timestamp': time.time_ns(),
            'detection_signals': self._get_detection_summary(symbol)
        }
        
        self.redis.setex(
            f'market:{symbol}:hidden:{order_type}',
            30,
            orjson.dumps(signal).decode('utf-8')
        )
        
        logger.info(f"Hidden order detected: {order_type} on {symbol} at {price}")
        
    def _get_detection_summary(self, symbol: str) -> Dict:
        """Get summary of detection signals"""
        
        signals = self.detection_signals[symbol]
        
        return {
            'between_spread_count': len(signals['between_spread']),
            'refresh_locations': len(signals['size_refreshes']),
            'iceberg_candidates': len(signals['iceberg_candidates']),
            'time_violations': len(signals['time_violations'])
        }
        
    def get_hidden_order_locations(self, symbol: str) -> Dict:
        """Get current hidden order candidate locations"""
        
        return dict(self.detection_signals[symbol]['iceberg_candidates'])
        
    def _flag_iceberg_candidate(self, symbol: str, price: float, side: str):
        """Flag potential iceberg order at price level"""
        
        price_key = f"{price:.2f}"
        
        self.detection_signals[symbol]['iceberg_candidates'][price_key] = {
            'price': price,
            'side': side,
            'confidence': 0.8,
            'detection_time': time.time(),
            'refresh_count': self.detection_signals[symbol]['size_refreshes'].get(price_key, 0)
        }
        
        # Store in Redis for other modules
        self.redis.setex(
            f'market:{symbol}:hidden:iceberg:{price_key}',
            30,
            json.dumps(self.detection_signals[symbol]['iceberg_candidates'][price_key])
        )
        
        logger.info(f"Iceberg order detected on {symbol} at {price} ({side})")
        
    def _flag_hidden_liquidity(self, symbol: str, trade: Dict):
        """Flag hidden liquidity detection"""
        
        alert = {
            'symbol': symbol,
            'price': trade['price'],
            'size': trade['size'],
            'type': 'hidden_liquidity',
            'confidence': 0.7,
            'time': time.time()
        }
        
        # Store alert
        self.redis.setex(
            f'market:{symbol}:hidden:alert',
            10,
            json.dumps(alert)
        )
        
    def _analyze_execution_patterns(self, symbol: str):
        """Analyze recent executions for hidden order patterns"""
        
        executions = list(self.recent_executions[symbol])
        
        # Pattern 1: Repeated executions at same price (iceberg)
        price_counts = defaultdict(int)
        for exec in executions:
            price_key = f"{exec['price']:.2f}"
            price_counts[price_key] += 1
            
        # Prices with multiple executions suggest iceberg
        for price_key, count in price_counts.items():
            if count >= 5:  # 5+ executions at same price
                price = float(price_key)
                
                # Determine side based on recent trades
                recent_at_price = [e for e in executions if f"{e['price']:.2f}" == price_key]
                avg_direction = sum(1 if e.get('direction') == 'buy' else -1 for e in recent_at_price) / len(recent_at_price)
                side = 'bid' if avg_direction > 0 else 'ask'
                
                self._flag_iceberg_candidate(symbol, price, side)
                
        # Pattern 2: Pegged order detection (follows NBBO)
        self._detect_pegged_orders(symbol, executions)
        
        # Pattern 3: Reserve order detection (partial fills)
        self._detect_reserve_orders(symbol, executions)
        
    def _detect_pegged_orders(self, symbol: str, executions: List[Dict]):
        """Detect pegged orders that follow NBBO"""
        
        if len(executions) < 5:
            return
            
        # Get NBBO history
        nbbo_data = self.redis.get(f'market:{symbol}:nbbo')
        if not nbbo_data:
            return
            
        nbbo = json.loads(nbbo_data)
        
        # Check if executions follow NBBO movement
        for i in range(1, len(executions)):
            curr = executions[i]
            prev = executions[i-1]
            
            # If price moved exactly with NBBO, likely pegged
            if abs(curr['price'] - nbbo['mid']) < 0.01:  # Within penny of mid
                self.detection_signals[symbol]['pegged_patterns'].append({
                    'price': curr['price'],
                    'time': curr.get('time', time.time()),
                    'nbbo_mid': nbbo['mid']
                })
                
    def _detect_reserve_orders(self, symbol: str, executions: List[Dict]):
        """Detect reserve orders (show small, have large reserve)"""
        
        # Look for patterns of consistent small fills followed by large
        sizes = [e['size'] for e in executions]
        
        if len(sizes) < 5:
            return
            
        # Calculate statistics
        median_size = np.median(sizes)
        max_size = np.max(sizes)
        
        # Reserve order pattern: many small fills, occasional large
        if max_size > median_size * 5:  # Large fill 5x bigger than median
            small_fills = sum(1 for s in sizes if s <= median_size)
            large_fills = sum(1 for s in sizes if s > median_size * 3)
            
            if small_fills > large_fills * 3:  # Many more small than large
                # Likely reserve order pattern
                self.detection_signals[symbol]['reserve_order_signals'].append({
                    'median_size': median_size,
                    'max_size': max_size,
                    'pattern': 'reserve',
                    'confidence': 0.6,
                    'time': time.time()
                })
                
                # Store detection
                self.redis.setex(
                    f'market:{symbol}:hidden:reserve',
                    30,
                    json.dumps({
                        'detected': True,
                        'visible_size': median_size,
                        'estimated_reserve': max_size * 10
                    })
                )
                
    def generate_hidden_order_report(self, symbol: str) -> Dict:
        """Generate comprehensive hidden order report"""
        
        signals = self.detection_signals[symbol]
        
        report = {
            'symbol': symbol,
            'timestamp': time.time(),
            'summary': {
                'iceberg_locations': len(signals['iceberg_candidates']),
                'between_spread_count': len(signals['between_spread']),
                'pegged_orders': len(signals['pegged_patterns']),
                'reserve_orders': len(signals.get('reserve_order_signals', [])),
                'time_violations': len(signals['time_violations'])
            },
            'iceberg_candidates': list(signals['iceberg_candidates'].values()),
            'high_confidence_hidden': []
        }
        
        # Identify high confidence hidden orders
        for price_key, iceberg in signals['iceberg_candidates'].items():
            if iceberg['confidence'] > 0.7:
                report['high_confidence_hidden'].append({
                    'price': iceberg['price'],
                    'side': iceberg['side'],
                    'confidence': iceberg['confidence']
                })
                
        # Store report
        self.redis.setex(
            f'market:{symbol}:hidden:report',
            60,
            orjson.dumps(report).decode('utf-8')
        )
        
        return report