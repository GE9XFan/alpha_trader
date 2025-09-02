#!/usr/bin/env python3
"""
Options Market Maker Detector Module
Detects when options market makers are hedging gamma exposure
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


class OptionsMMDetector:
    """Detects options market maker hedging patterns"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        
        # Known options market maker IDs
        self.options_mms = {
            'CDRG': 'Citadel', 'SUSQ': 'Susquehanna', 'WLVN': 'Wolverine',
            'JANE': 'Jane Street', 'FLOW': 'Flow Traders', 'OPTV': 'Optiver',
            'GCSI': 'GTS', 'VIRT': 'Virtu', 'JPMM': 'JPMorgan', 'GSCO': 'Goldman Sachs'
        }
        
        # MM activity tracking
        self.mm_activity = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.hedging_patterns = defaultdict(lambda: {'score': 0.0, 'events': deque(maxlen=100)})
        
    def analyze_market_maker(self, symbol: str, depth):
        """Analyze market maker quote for hedging patterns"""
        
        if not hasattr(depth, 'marketMaker'):
            return
            
        mm_id = depth.marketMaker
        if mm_id not in self.options_mms:
            return
            
        mm_name = self.options_mms[mm_id]
        
        # Track MM quote
        quote = {
            'price': depth.price,
            'size': depth.size,
            'side': 'bid' if depth.side == 0 else 'ask',
            'time': time.time_ns(),
            'aggressive': self._is_aggressive_quote(symbol, depth)
        }
        
        self.mm_activity[symbol][mm_name].append(quote)
        
        # Detect hedging pattern
        if self._detect_hedging_pattern(symbol, mm_name):
            self._publish_hedging_signal(symbol, mm_name)
            
    def _is_aggressive_quote(self, symbol: str, depth) -> bool:
        """Check if quote is aggressive (improving NBBO)"""
        
        nbbo = self.redis.get(f'market:{symbol}:nbbo')
        if not nbbo:
            return False
            
        nbbo_data = json.loads(nbbo)
        
        if depth.side == 0:  # Bid
            return depth.price > nbbo_data['bid']['price']
        else:  # Ask
            return depth.price < nbbo_data['ask']['price']
            
    def _detect_hedging_pattern(self, symbol: str, mm_name: str) -> bool:
        """Detect if MM is hedging gamma"""
        
        quotes = list(self.mm_activity[symbol][mm_name])
        if len(quotes) < 10:
            return False
            
        recent = quotes[-10:]
        
        # Pattern indicators:
        # 1. Rapid quote updates
        time_diffs = [recent[i]['time'] - recent[i-1]['time'] for i in range(1, len(recent))]
        avg_time_diff = np.mean(time_diffs)
        rapid_updates = avg_time_diff < 100_000_000  # < 100ms between updates
        
        # 2. Aggressive pricing
        aggressive_count = sum(1 for q in recent if q['aggressive'])
        aggressive_pattern = aggressive_count >= 5
        
        # 3. Size changes
        sizes = [q['size'] for q in recent]
        size_volatility = np.std(sizes) / (np.mean(sizes) + 1)
        volatile_sizes = size_volatility > 0.3
        
        # Combine signals
        hedging_detected = rapid_updates and (aggressive_pattern or volatile_sizes)
        
        if hedging_detected:
            self.hedging_patterns[symbol]['score'] = 0.85
            self.hedging_patterns[symbol]['events'].append({
                'mm': mm_name,
                'time': time.time(),
                'confidence': 0.85
            })
            
        return hedging_detected
        
    def _publish_hedging_signal(self, symbol: str, mm_name: str):
        """Publish MM hedging detection to Redis"""
        
        signal = {
            'symbol': symbol,
            'mm': mm_name,
            'hedging_detected': True,
            'confidence': self.hedging_patterns[symbol]['score'],
            'timestamp': time.time_ns()
        }
        
        self.redis.setex(
            f'market:{symbol}:mm:hedging:{mm_name}',
            30,
            orjson.dumps(signal).decode('utf-8')
        )
        
        logger.info(f"MM hedging detected: {mm_name} on {symbol}")
        
    def get_active_mms(self, symbol: str) -> List[str]:
        """Get list of currently active market makers"""
        
        active = []
        for mm_name, quotes in self.mm_activity[symbol].items():
            if quotes:
                last_quote_time = quotes[-1]['time']
                if time.time_ns() - last_quote_time < 5_000_000_000:  # Active in last 5 seconds
                    active.append(mm_name)
                    
        return active
        
    def _publish_hedging_signal(self, symbol: str, mm_name: str):
        """Publish market maker hedging signal"""
        
        signal = {
            'symbol': symbol,
            'mm': mm_name,
            'type': 'gamma_hedging',
            'confidence': self.hedging_patterns[symbol]['score'],
            'time': time.time_ns()
        }
        
        # Store in Redis
        self.redis.setex(
            f'market:{symbol}:mm:hedging:{mm_name}',
            30,
            json.dumps(signal)
        )
        
        # Add to event log
        self.hedging_patterns[symbol]['events'].append(signal)
        
        logger.info(f"MM hedging detected: {mm_name} on {symbol}")
        
    def analyze_gamma_exposure(self, symbol: str, gex_data: Dict):
        """Analyze gamma exposure and MM positioning"""
        
        try:
            # Get current MM activity
            active_mms = self.get_active_market_makers(symbol)
            
            if not active_mms:
                return
                
            # High GEX suggests MMs need to hedge
            gex = gex_data.get('total_gex', 0)
            spot = gex_data.get('spot_price', 0)
            
            if abs(gex) > 1000000:  # Significant gamma exposure
                # Check if MMs are adjusting quotes
                for mm_name in active_mms:
                    quotes = list(self.mm_activity[symbol][mm_name])
                    if len(quotes) >= 5:
                        recent = quotes[-5:]
                        
                        # Look for directional bias in quoting
                        bid_aggressive = sum(1 for q in recent if q['side'] == 'bid' and q['aggressive'])
                        ask_aggressive = sum(1 for q in recent if q['side'] == 'ask' and q['aggressive'])
                        
                        # Gamma hedging pattern
                        if gex > 0:  # Positive gamma - sell rallies, buy dips
                            if spot > gex_data.get('flip_level', spot) and ask_aggressive > bid_aggressive:
                                # MM selling into strength
                                self.hedging_patterns[symbol]['score'] = 0.8
                                self._publish_hedging_signal(symbol, mm_name)
                        else:  # Negative gamma - buy rallies, sell dips
                            if spot < gex_data.get('flip_level', spot) and bid_aggressive > ask_aggressive:
                                # MM buying into weakness
                                self.hedging_patterns[symbol]['score'] = 0.8
                                self._publish_hedging_signal(symbol, mm_name)
                                
        except Exception as e:
            logger.error(f"Error analyzing gamma exposure for {symbol}: {e}")
            
    def detect_quote_stuffing(self, symbol: str) -> bool:
        """Detect quote stuffing patterns (rapid quote updates with no trades)"""
        
        # Check each MM for quote stuffing
        for mm_name, quotes in self.mm_activity[symbol].items():
            if len(quotes) < 50:
                continue
                
            recent = list(quotes)[-50:]
            
            # Calculate quote rate
            time_span = (recent[-1]['time'] - recent[0]['time']) / 1e9  # Convert to seconds
            if time_span > 0:
                quote_rate = len(recent) / time_span
                
                # More than 100 quotes per second suggests stuffing
                if quote_rate > 100:
                    logger.warning(f"Quote stuffing detected: {mm_name} on {symbol} ({quote_rate:.0f} quotes/sec)")
                    
                    # Store alert
                    self.redis.setex(
                        f'market:{symbol}:alert:quote_stuffing',
                        60,
                        json.dumps({
                            'mm': mm_name,
                            'rate': quote_rate,
                            'time': time.time()
                        })
                    )
                    
                    return True
                    
        return False
        
    def detect_spoofing(self, symbol: str, depth_updates: List[Dict]) -> bool:
        """Detect spoofing patterns (large orders that disappear quickly)"""
        
        # Track large orders that appear and disappear
        for update in depth_updates:
            if update.get('size', 0) > 10000:  # Large order
                # Check if it was cancelled quickly
                if update.get('operation') == 2:  # Delete operation
                    # Check how long the order was live
                    order_key = f"{update['price']}_{update['side']}"
                    
                    # Would need order tracking to fully implement
                    # For now, flag rapid large order cancellations
                    logger.warning(f"Potential spoofing on {symbol}: Large order cancelled at {update['price']}")
                    
                    return True
                    
        return False
        
    def generate_mm_report(self, symbol: str) -> Dict:
        """Generate comprehensive market maker activity report"""
        
        report = {
            'symbol': symbol,
            'timestamp': time.time_ns(),
            'active_mms': self.get_active_market_makers(symbol),
            'hedging_detected': False,
            'quote_stuffing': False,
            'spoofing': False,
            'mm_concentration': 0.0
        }
        
        # Check for hedging
        if self.hedging_patterns[symbol]['score'] > 0.7:
            report['hedging_detected'] = True
            report['hedging_confidence'] = self.hedging_patterns[symbol]['score']
            
        # Check for manipulation
        report['quote_stuffing'] = self.detect_quote_stuffing(symbol)
        
        # Calculate MM concentration (HHI)
        if report['active_mms']:
            total_quotes = sum(
                len(self.mm_activity[symbol][mm]) 
                for mm in report['active_mms']
            )
            
            if total_quotes > 0:
                hhi = sum(
                    (len(self.mm_activity[symbol][mm]) / total_quotes) ** 2
                    for mm in report['active_mms']
                )
                report['mm_concentration'] = hhi
                
        # Store report
        self.redis.setex(
            f'market:{symbol}:mm:report',
            60,
            orjson.dumps(report).decode('utf-8')
        )
        
        return report