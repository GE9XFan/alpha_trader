#!/usr/bin/env python3
"""
Auction Processor Module
Handles opening and closing auction imbalances for MOC strategy
Processes auction data from IBKR generic tick 233
"""

import time
import json
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, time as dt_time

import redis
import orjson
import numpy as np

logger = logging.getLogger(__name__)


class AuctionProcessor:
    """Processes auction imbalance data for opening and closing auctions"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.symbols = config['trading']['symbols']
        
        # Auction data storage
        self.auction_data = {
            symbol: {
                'opening': AuctionData('opening'),
                'closing': AuctionData('closing')
            } for symbol in self.symbols
        }
        
        # Historical imbalance tracking for pattern detection
        self.imbalance_history = defaultdict(lambda: deque(maxlen=20))  # 20 days
        
        # MOC signal parameters
        self.moc_params = {
            'min_imbalance': 100000,  # Minimum shares imbalanced
            'min_notional': 1000000,  # Minimum $1M notional
            'signal_time_start': dt_time(15, 30),  # 3:30 PM
            'signal_time_end': dt_time(15, 50),  # 3:50 PM
            'confidence_threshold': 0.7
        }
        
        # Auction patterns for signal generation
        self.auction_patterns = AuctionPatternDetector()
        
        # Track auction performance
        self.performance_tracker = defaultdict(lambda: {
            'correct_direction': 0,
            'total_signals': 0,
            'avg_move': 0.0
        })
        
    def process_auction_update(self, symbol: str, ticker):
        """Process auction imbalance update from IBKR"""
        
        try:
            current_time = datetime.now().time()
            
            # Determine if this is opening or closing auction
            if current_time < dt_time(10, 0):
                auction_type = 'opening'
            elif current_time > dt_time(15, 0):
                auction_type = 'closing'
            else:
                # Mid-day auction updates are rare but possible
                return
                
            auction = self.auction_data[symbol][auction_type]
            
            # Extract auction data from ticker
            # IBKR provides these via generic tick 233
            if hasattr(ticker, 'auctionVolume'):
                auction.imbalance_qty = ticker.auctionVolume
                
            if hasattr(ticker, 'auctionPrice'):
                auction.indicative_price = ticker.auctionPrice
                
            if hasattr(ticker, 'auctionImbalance'):
                auction.imbalance_qty = ticker.auctionImbalance
                auction.imbalance_side = 'buy' if ticker.auctionImbalance > 0 else 'sell'
                
            if hasattr(ticker, 'regulatoryImbalance'):
                auction.regulatory_imbalance = ticker.regulatoryImbalance
                
            # Calculate paired and unpaired quantities
            auction.calculate_paired_unpaired()
            
            # Update timestamp
            auction.last_update = time.time_ns()
            
            # For closing auction, generate MOC signal if within time window
            if auction_type == 'closing' and self._in_moc_window(current_time):
                moc_signal = self._generate_moc_signal(symbol, auction)
                
                if moc_signal['confidence'] >= self.moc_params['confidence_threshold']:
                    self._publish_moc_signal(symbol, moc_signal)
                    
            # Store auction data in Redis
            self._store_auction_data(symbol, auction_type, auction)
            
            # Track patterns for future analysis
            self.auction_patterns.update(symbol, auction_type, auction)
            
        except Exception as e:
            logger.error(f"Error processing auction update for {symbol}: {e}")
            
    def _in_moc_window(self, current_time: dt_time) -> bool:
        """Check if we're in MOC signal generation window"""
        
        return (self.moc_params['signal_time_start'] <= current_time <= 
                self.moc_params['signal_time_end'])
                
    def _generate_moc_signal(self, symbol: str, auction: 'AuctionData') -> Dict:
        """Generate MOC trading signal based on auction imbalance"""
        
        signal = {
            'symbol': symbol,
            'strategy': 'MOC',
            'direction': None,
            'confidence': 0.0,
            'imbalance_qty': auction.imbalance_qty,
            'imbalance_side': auction.imbalance_side,
            'indicative_price': auction.indicative_price,
            'factors': {}
        }
        
        # Factor 1: Imbalance magnitude
        if abs(auction.imbalance_qty) >= self.moc_params['min_imbalance']:
            signal['factors']['size'] = 0.3
        else:
            signal['factors']['size'] = abs(auction.imbalance_qty) / self.moc_params['min_imbalance'] * 0.3
            
        # Factor 2: Notional value
        if auction.indicative_price > 0:
            notional = abs(auction.imbalance_qty) * auction.indicative_price
            if notional >= self.moc_params['min_notional']:
                signal['factors']['notional'] = 0.2
            else:
                signal['factors']['notional'] = notional / self.moc_params['min_notional'] * 0.2
        else:
            signal['factors']['notional'] = 0.0
            
        # Factor 3: Historical pattern
        pattern_score = self.auction_patterns.get_pattern_score(symbol, 'closing')
        signal['factors']['pattern'] = pattern_score * 0.25
        
        # Factor 4: Market regime
        regime_score = self._get_market_regime_score(symbol)
        signal['factors']['regime'] = regime_score * 0.15
        
        # Factor 5: Regulatory imbalance alignment
        if auction.regulatory_imbalance != 0:
            if np.sign(auction.regulatory_imbalance) == np.sign(auction.imbalance_qty):
                signal['factors']['regulatory'] = 0.1
            else:
                signal['factors']['regulatory'] = -0.05  # Conflicting signals
        else:
            signal['factors']['regulatory'] = 0.0
            
        # Calculate total confidence
        signal['confidence'] = sum(signal['factors'].values())
        
        # Determine direction
        # Fade large imbalances (they tend to revert)
        if auction.imbalance_side == 'buy':
            signal['direction'] = 'sell'  # Fade the buy imbalance
        else:
            signal['direction'] = 'buy'  # Fade the sell imbalance
            
        # Adjust for extreme imbalances (don't fade if too extreme)
        if abs(auction.imbalance_qty) > self.moc_params['min_imbalance'] * 5:
            # Extreme imbalance - go with the flow instead of fading
            signal['direction'] = auction.imbalance_side
            signal['factors']['extreme'] = 0.1
            signal['confidence'] += 0.1
            
        return signal
        
    def _get_market_regime_score(self, symbol: str) -> float:
        """Get market regime score for MOC signal"""
        
        try:
            # Check volatility regime
            regime = self.redis.get(f'metrics:{symbol}:regime')
            if regime:
                regime_data = json.loads(regime)
                
                # High volatility favors MOC strategies
                if regime_data.get('volatility', 'normal') == 'high':
                    return 0.8
                elif regime_data.get('volatility', 'normal') == 'low':
                    return 0.3
                    
            return 0.5  # Neutral
            
        except Exception:
            return 0.5
            
    def _publish_moc_signal(self, symbol: str, signal: Dict):
        """Publish MOC signal to Redis for execution"""
        
        # Add metadata
        signal['timestamp'] = time.time_ns()
        signal['ttl'] = 60  # Valid for 60 seconds
        
        # Store in Redis
        signal_json = orjson.dumps(signal).decode('utf-8')
        
        # Add to pending signals queue
        self.redis.lpush(f'signals:{symbol}:pending', signal_json)
        self.redis.expire(f'signals:{symbol}:pending', 60)
        
        # Store as latest MOC signal
        self.redis.setex(
            f'market:{symbol}:auction:moc_signal',
            60,
            signal_json
        )
        
        # Track performance
        self.performance_tracker[symbol]['total_signals'] += 1
        
        logger.info(f"MOC signal generated for {symbol}: {signal['direction']} with confidence {signal['confidence']:.2f}")
        
    def _store_auction_data(self, symbol: str, auction_type: str, auction: 'AuctionData'):
        """Store auction data in Redis"""
        
        auction_dict = auction.to_dict()
        
        # Store main auction data
        self.redis.setex(
            f'market:{symbol}:auction:{auction_type}',
            60 if auction_type == 'closing' else 300,  # Closing auction data expires faster
            orjson.dumps(auction_dict).decode('utf-8')
        )
        
        # Store imbalance for quick access
        if auction.imbalance_qty != 0:
            self.redis.setex(
                f'market:{symbol}:auction:imbalance',
                60,
                json.dumps({
                    'qty': auction.imbalance_qty,
                    'side': auction.imbalance_side,
                    'price': auction.indicative_price
                })
            )
            
    def track_auction_result(self, symbol: str, auction_type: str, close_price: float):
        """Track auction results for performance analysis"""
        
        try:
            auction = self.auction_data[symbol][auction_type]
            
            if auction.indicative_price > 0:
                # Calculate how close indicative was to actual
                price_diff = close_price - auction.indicative_price
                pct_diff = (price_diff / auction.indicative_price) * 100
                
                # Track if our signal was correct
                if auction.imbalance_side == 'buy' and price_diff > 0:
                    self.performance_tracker[symbol]['correct_direction'] += 1
                elif auction.imbalance_side == 'sell' and price_diff < 0:
                    self.performance_tracker[symbol]['correct_direction'] += 1
                    
                # Update average move
                tracker = self.performance_tracker[symbol]
                tracker['avg_move'] = (
                    tracker['avg_move'] * (tracker['total_signals'] - 1) + abs(pct_diff)
                ) / tracker['total_signals']
                
                # Store historical data
                self.imbalance_history[symbol].append({
                    'date': datetime.now().date().isoformat(),
                    'imbalance': auction.imbalance_qty,
                    'indicative': auction.indicative_price,
                    'actual': close_price,
                    'pct_move': pct_diff
                })
                
        except Exception as e:
            logger.error(f"Error tracking auction result for {symbol}: {e}")


class AuctionData:
    """Container for auction imbalance data"""
    
    def __init__(self, auction_type: str):
        self.auction_type = auction_type
        self.imbalance_qty = 0
        self.imbalance_side = None
        self.indicative_price = 0.0
        self.reference_price = 0.0
        self.near_price = 0.0
        self.far_price = 0.0
        self.paired_qty = 0
        self.unpaired_qty = 0
        self.regulatory_imbalance = 0
        self.last_update = 0
        self.auction_time = None
        
    def calculate_paired_unpaired(self):
        """Calculate paired and unpaired quantities"""
        
        # Unpaired is the absolute imbalance
        self.unpaired_qty = abs(self.imbalance_qty)
        
        # Paired quantity would need order book depth
        # For now, estimate based on typical patterns
        if self.imbalance_qty != 0:
            # Rough estimate: paired is usually 2-5x the imbalance
            self.paired_qty = abs(self.imbalance_qty) * 3
            
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        
        return {
            'auction_type': self.auction_type,
            'imbalance_qty': self.imbalance_qty,
            'imbalance_side': self.imbalance_side,
            'indicative_price': self.indicative_price,
            'reference_price': self.reference_price,
            'near_price': self.near_price,
            'far_price': self.far_price,
            'paired_qty': self.paired_qty,
            'unpaired_qty': self.unpaired_qty,
            'regulatory_imbalance': self.regulatory_imbalance,
            'last_update': self.last_update,
            'auction_time': self.auction_time.isoformat() if self.auction_time else None
        }


class AuctionPatternDetector:
    """Detects patterns in auction imbalances for improved signals"""
    
    def __init__(self):
        self.patterns = defaultdict(lambda: {
            'imbalance_reversals': deque(maxlen=20),
            'imbalance_continuations': deque(maxlen=20),
            'large_imbalances': deque(maxlen=20),
            'pattern_score': 0.5
        })
        
    def update(self, symbol: str, auction_type: str, auction: AuctionData):
        """Update pattern tracking with new auction data"""
        
        if auction_type != 'closing':
            return  # Only track closing auction patterns for MOC
            
        patterns = self.patterns[symbol]
        
        # Track if this is a large imbalance
        if abs(auction.imbalance_qty) > 500000:
            patterns['large_imbalances'].append({
                'qty': auction.imbalance_qty,
                'side': auction.imbalance_side,
                'time': time.time()
            })
            
        # Pattern detection would be enhanced with historical data
        # For now, track basic patterns
        self._calculate_pattern_score(symbol)
        
    def _calculate_pattern_score(self, symbol: str):
        """Calculate pattern score based on historical patterns"""
        
        patterns = self.patterns[symbol]
        
        # Simple scoring based on recent patterns
        if len(patterns['large_imbalances']) >= 5:
            # Check for consistency in imbalance direction
            recent_sides = [p['side'] for p in list(patterns['large_imbalances'])[-5:]]
            
            # Same direction imbalances suggest trend
            if all(s == recent_sides[0] for s in recent_sides):
                patterns['pattern_score'] = 0.8
            # Alternating suggests mean reversion
            elif recent_sides == ['buy', 'sell', 'buy', 'sell', 'buy']:
                patterns['pattern_score'] = 0.7
            else:
                patterns['pattern_score'] = 0.5
        else:
            patterns['pattern_score'] = 0.5
            
    def get_pattern_score(self, symbol: str, auction_type: str) -> float:
        """Get pattern score for signal generation"""
        
        if symbol in self.patterns:
            return self.patterns[symbol]['pattern_score']
            
        return 0.5  # Neutral