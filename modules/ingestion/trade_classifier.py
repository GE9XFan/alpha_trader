#!/usr/bin/env python3
"""
Trade Classifier Module
Classifies trades with condition codes and determines trade direction
Critical for VPIN calculation and sweep detection
"""

import time
import json
from collections import deque, defaultdict
from typing import Dict, Optional
import logging

import redis
import orjson

logger = logging.getLogger(__name__)


class TradeClassifier:
    """Classifies trades based on condition codes and market context"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        
        # Complete FINRA/CTA trade condition code mappings
        self.condition_codes = {
            # Primary sale conditions
            '@': {'name': 'Regular Sale', 'include_vpin': True},
            'A': {'name': 'Acquisition', 'include_vpin': False},
            'B': {'name': 'Bunched', 'include_vpin': True},
            'C': {'name': 'Cash Sale', 'include_vpin': False},
            'D': {'name': 'Distribution', 'include_vpin': False},
            'E': {'name': 'Placeholder', 'include_vpin': False},
            'F': {'name': 'Intermarket Sweep', 'include_vpin': False, 'is_sweep': True},
            'G': {'name': 'Bunched Sold', 'include_vpin': True},
            'H': {'name': 'Price Variation', 'include_vpin': False},
            'I': {'name': 'Odd Lot', 'include_vpin': False, 'is_odd_lot': True},
            'J': {'name': 'Rule 127 or 155', 'include_vpin': True},
            'K': {'name': 'Rule 155', 'include_vpin': True},
            'L': {'name': 'Sold Last', 'include_vpin': True},
            'M': {'name': 'Market Center Close', 'include_vpin': False},
            'N': {'name': 'Next Day', 'include_vpin': False},
            'O': {'name': 'Opening Prints', 'include_vpin': True, 'is_opening': True},
            'P': {'name': 'Prior Reference Price', 'include_vpin': False},
            'Q': {'name': 'Market Center Open', 'include_vpin': True},
            'R': {'name': 'Seller', 'include_vpin': True},
            'S': {'name': 'Split Trade', 'include_vpin': True},
            'T': {'name': 'Form T', 'include_vpin': True, 'is_block': True},
            'U': {'name': 'Extended Hours', 'include_vpin': False, 'is_extended': True},
            'V': {'name': 'Contingent Trade', 'include_vpin': False},
            'W': {'name': 'Average Price Trade', 'include_vpin': False, 'is_dark': True},
            'X': {'name': 'Cross Trade', 'include_vpin': True, 'is_cross': True},
            'Y': {'name': 'Yellow Flag', 'include_vpin': False},
            'Z': {'name': 'Sold Out of Sequence', 'include_vpin': False},
            '1': {'name': 'Stopped Stock', 'include_vpin': True},
            '2': {'name': 'Stopped Stock ETH', 'include_vpin': False},
            '3': {'name': 'Stopped Stock Out of Sequence', 'include_vpin': False},
            '4': {'name': 'Derivatively Priced', 'include_vpin': False},
            '5': {'name': 'Re-Opening Prints', 'include_vpin': True},
            '6': {'name': 'Closing Prints', 'include_vpin': True, 'is_closing': True},
            '7': {'name': 'Qualified Contingent', 'include_vpin': True},
            '8': {'name': 'Placeholder for 611', 'include_vpin': False},
            '9': {'name': 'Corrected Consolidated Close', 'include_vpin': False}
        }
        
        # Trade history for tick rule
        self.trade_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Sweep detection buffer
        self.sweep_buffer = defaultdict(lambda: deque(maxlen=100))
        
    def classify_trade(self, symbol: str, trade: Dict) -> Dict:
        """Classify trade with all attributes"""
        
        # Start with base trade data
        classified = trade.copy()
        
        # Parse condition codes if present
        conditions = trade.get('conditions', '')
        if conditions:
            classified.update(self._parse_conditions(conditions))
        else:
            # Default classifications
            classified['include_vpin'] = True
            classified['is_sweep'] = False
            classified['is_odd_lot'] = False
            classified['is_block'] = False
            classified['is_dark'] = False
            classified['is_extended'] = False
            
        # Determine trade direction
        classified['direction'] = self._determine_direction(symbol, trade)
        
        # Check for sweep pattern
        if not classified.get('is_sweep'):
            classified['is_sweep'] = self._detect_sweep_pattern(symbol, trade)
            
        # Add to history
        self.trade_history[symbol].append(classified)
        
        return classified
        
    def classify_tick_trade(self, trade_detail: Dict) -> Dict:
        """Classify tick-by-tick trade"""
        
        symbol = trade_detail['symbol']
        
        # Parse special conditions
        conditions = trade_detail.get('conditions', '')
        
        classified = {
            'symbol': symbol,
            'price': trade_detail['price'],
            'size': trade_detail['size'],
            'time': trade_detail['time'],
            'exchange': trade_detail.get('exchange', 'UNKNOWN'),
            'past_limit': trade_detail.get('past_limit', False),
            'unreported': trade_detail.get('unreported', False)
        }
        
        # Parse conditions
        if conditions:
            classified.update(self._parse_conditions(conditions))
        else:
            classified['include_vpin'] = not classified['unreported']
            classified['is_sweep'] = False
            
        # Determine direction
        classified['direction'] = self._determine_direction(symbol, classified)
        
        return classified
        
    def _parse_conditions(self, conditions: str) -> Dict:
        """Parse trade condition codes"""
        
        result = {
            'condition_codes': conditions,
            'condition_names': [],
            'include_vpin': True,
            'is_sweep': False,
            'is_odd_lot': False,
            'is_block': False,
            'is_dark': False,
            'is_extended': False,
            'is_opening': False,
            'is_cross': False
        }
        
        for char in conditions:
            if char in self.condition_codes:
                cond_info = self.condition_codes[char]
                result['condition_names'].append(cond_info['name'])
                
                # Update include_vpin flag (AND logic - exclude if any condition says exclude)
                if 'include_vpin' in cond_info and not cond_info['include_vpin']:
                    result['include_vpin'] = False
                    
                # Set special flags
                for key in ['is_sweep', 'is_odd_lot', 'is_block', 'is_dark', 
                           'is_extended', 'is_opening', 'is_cross']:
                    if key in cond_info and cond_info[key]:
                        result[key] = True
                        
        return result
        
    def _determine_direction(self, symbol: str, trade: Dict) -> str:
        """Determine if trade is buy or sell"""
        
        # Method 1: Quote rule (most accurate)
        if 'bid' in trade and 'ask' in trade:
            mid = (trade['bid'] + trade['ask']) / 2
            if trade['price'] > mid:
                return 'buy'
            elif trade['price'] < mid:
                return 'sell'
                
        # Method 2: Tick rule
        history = self.trade_history[symbol]
        if len(history) >= 2:
            prev_price = history[-1]['price'] if history else trade['price']
            if trade['price'] > prev_price:
                return 'buy'
            elif trade['price'] < prev_price:
                return 'sell'
            else:
                # Zero tick - use previous direction
                return history[-1].get('direction', 'buy') if history else 'buy'
                
        # Method 3: Lee-Ready rule (if we have previous quote)
        # Would need quote history for this
        
        # Default
        return 'buy'
        
    def _detect_sweep_pattern(self, symbol: str, trade: Dict) -> bool:
        """Detect sweep pattern from trade sequence"""
        
        # Add to sweep buffer
        self.sweep_buffer[symbol].append({
            'price': trade['price'],
            'size': trade['size'],
            'time': trade.get('time', time.time()),
            'exchange': trade.get('exchange', 'UNKNOWN')
        })
        
        # Need at least 3 trades to detect sweep
        if len(self.sweep_buffer[symbol]) < 3:
            return False
            
        recent = list(self.sweep_buffer[symbol])[-5:]
        
        # Sweep indicators:
        # 1. Multiple trades at same/similar price
        # 2. Within short time window (< 1 second)
        # 3. Different exchanges (if available)
        # 4. Large total size
        
        # Check time window
        time_span = recent[-1]['time'] - recent[0]['time']
        if time_span > 1.0:  # More than 1 second
            return False
            
        # Check price similarity
        prices = [t['price'] for t in recent]
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        
        if price_range / avg_price > 0.001:  # More than 0.1% price range
            return False
            
        # Check total size
        total_size = sum(t['size'] for t in recent)
        if total_size < 5000:  # Threshold for sweep
            return False
            
        # Check exchanges (if available)
        exchanges = set(t['exchange'] for t in recent if t['exchange'] != 'UNKNOWN')
        multi_exchange = len(exchanges) > 1
        
        # Sweep detected if multi-exchange or large size
        return multi_exchange or total_size >= 10000
        
    def get_trade_statistics(self, symbol: str) -> Dict:
        """Get trade classification statistics"""
        
        history = list(self.trade_history[symbol])
        
        if not history:
            return {}
            
        total = len(history)
        
        return {
            'total_trades': total,
            'buy_trades': sum(1 for t in history if t.get('direction') == 'buy'),
            'sell_trades': sum(1 for t in history if t.get('direction') == 'sell'),
            'sweep_trades': sum(1 for t in history if t.get('is_sweep')),
            'odd_lot_trades': sum(1 for t in history if t.get('is_odd_lot')),
            'block_trades': sum(1 for t in history if t.get('is_block')),
            'dark_trades': sum(1 for t in history if t.get('is_dark')),
            'vpin_eligible': sum(1 for t in history if t.get('include_vpin'))
        }