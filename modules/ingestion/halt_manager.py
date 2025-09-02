#!/usr/bin/env python3
"""
Halt and LULD Manager Module
Manages trading halts, circuit breakers, and Limit Up/Limit Down bands
Critical for regulatory compliance and risk management
"""

import time
import json
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from enum import Enum

import redis
import orjson
import numpy as np

logger = logging.getLogger(__name__)


class HaltType(Enum):
    """Trading halt types"""
    LULD_PAUSE = 'M'  # Limit Up/Limit Down pause
    NEWS_PENDING = 'T1'  # News pending
    NEWS_RELEASED = 'T2'  # News released
    REGULATORY = 'H4'  # SEC regulatory halt
    OPERATIONAL = 'O'  # Operational halt
    CIRCUIT_BREAKER_L1 = 'MW1'  # Market-wide circuit breaker level 1
    CIRCUIT_BREAKER_L2 = 'MW2'  # Market-wide circuit breaker level 2
    CIRCUIT_BREAKER_L3 = 'MW3'  # Market-wide circuit breaker level 3
    SINGLE_STOCK_PAUSE = 'T5'  # Single stock trading pause
    SUB_PENNY = 'Y'  # Sub-penny trading halt


class HaltManager:
    """Manages trading halts and LULD bands"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.symbols = config['trading']['symbols']
        
        # LULD band parameters (configured per symbol tier)
        self.luld_params = self._initialize_luld_params()
        
        # Current halt states
        self.halt_states = defaultdict(lambda: {
            'halted': False,
            'halt_type': None,
            'halt_code': None,
            'halt_time': None,
            'resume_time': None,
            'pre_halt_price': None,
            'pre_halt_bid': None,
            'pre_halt_ask': None
        })
        
        # LULD bands tracking
        self.luld_bands = defaultdict(lambda: {
            'upper': None,
            'lower': None,
            'reference_price': None,
            'band_width_pct': None,
            'last_update': None
        })
        
        # Circuit breaker levels (S&P 500)
        self.circuit_breaker_levels = {
            'level1': 0.07,  # 7% decline
            'level2': 0.13,  # 13% decline
            'level3': 0.20   # 20% decline
        }
        
        # Market-wide halt tracking
        self.market_wide_halt = False
        self.market_halt_level = 0
        
        # Halt history for pattern analysis
        self.halt_history = defaultdict(lambda: deque(maxlen=100))
        
        # LULD straddle tracking (distance to bands)
        self.luld_straddle = defaultdict(lambda: {
            'upper_distance': deque(maxlen=100),
            'lower_distance': deque(maxlen=100),
            'band_tests': deque(maxlen=20)
        })
        
        # Halt code mappings
        self.halt_codes = {
            'D': 'News Released',
            'E': 'Order Imbalance',
            'F': 'News Pending',
            'H': 'Halt - Regulatory',
            'I': 'Order Influx',
            'M': 'LULD Trading Pause',
            'N': 'Non-Compliance',
            'O': 'Operations Halt',
            'P': 'News and Resumption Times',
            'T': 'Single Stock Trading Pause',
            'X': 'Operational Halt',
            'Y': 'Sub-Penny Trading',
            'Z': 'No Open/No Resume',
            '1': 'Market Wide Circuit Breaker Level 1',
            '2': 'Market Wide Circuit Breaker Level 2',
            '3': 'Market Wide Circuit Breaker Level 3'
        }
        
    def _initialize_luld_params(self) -> Dict:
        """Initialize LULD parameters based on symbol tiers"""
        
        # Tier 1: S&P 500 and Russell 1000 stocks > $3
        # Tier 2: Other NMS stocks > $0.75
        # Tier 3: Other NMS stocks <= $0.75
        
        params = {}
        
        for symbol in self.symbols:
            # Determine tier (would normally come from reference data)
            if symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']:
                tier = 1
            else:
                tier = 2
                
            if tier == 1:
                # Tier 1 LULD parameters
                params[symbol] = {
                    'percentage_bands': {
                        'open_first_15min': 0.10,  # 10% for first 15 minutes
                        'open_next_10min': 0.05,   # 5% for next 10 minutes
                        'regular': 0.05,            # 5% regular trading
                        'close_last_25min': 0.10   # 10% for last 25 minutes
                    },
                    'price_bands': {
                        'above_3': 0.05,
                        'above_1_below_3': 0.10,
                        'above_0.75_below_1': 0.20,
                        'below_0.75': 0.28
                    },
                    'pause_time': 5  # 5 minute pause
                }
            else:
                # Tier 2 LULD parameters
                params[symbol] = {
                    'percentage_bands': {
                        'open_first_15min': 0.20,  # 20% for first 15 minutes
                        'open_next_10min': 0.10,   # 10% for next 10 minutes
                        'regular': 0.10,            # 10% regular trading
                        'close_last_25min': 0.20   # 20% for last 25 minutes
                    },
                    'price_bands': {
                        'above_3': 0.10,
                        'above_1_below_3': 0.20,
                        'above_0.75_below_1': 0.28,
                        'below_0.75': 0.30
                    },
                    'pause_time': 5  # 5 minute pause
                }
                
        return params
        
    def process_halt(self, symbol: str, ticker):
        """Process halt notification"""
        
        try:
            # Extract halt information
            halt_code = None
            if hasattr(ticker, 'haltCode'):
                halt_code = ticker.haltCode
            elif hasattr(ticker, 'lastTradingStatus'):
                halt_code = ticker.lastTradingStatus
                
            if not halt_code:
                return
                
            # Get current market data before halt
            pre_halt_data = self._capture_pre_halt_state(symbol, ticker)
            
            # Update halt state
            self.halt_states[symbol] = {
                'halted': True,
                'halt_type': self._determine_halt_type(halt_code),
                'halt_code': halt_code,
                'halt_description': self.halt_codes.get(halt_code, 'Unknown'),
                'halt_time': time.time_ns(),
                'resume_time': None,
                'pre_halt_price': pre_halt_data['price'],
                'pre_halt_bid': pre_halt_data['bid'],
                'pre_halt_ask': pre_halt_data['ask'],
                'pre_halt_volume': pre_halt_data['volume']
            }
            
            # Store halt state in Redis
            self._store_halt_state(symbol)
            
            # Add to halt history
            self.halt_history[symbol].append(self.halt_states[symbol].copy())
            
            # Notify risk manager
            self._notify_risk_manager(symbol, 'HALT')
            
            # Log critical event
            logger.critical(f"TRADING HALT: {symbol} - Code: {halt_code} - {self.halt_codes.get(halt_code, 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error processing halt for {symbol}: {e}")
            
    def process_resume(self, symbol: str):
        """Process trading resume after halt"""
        
        try:
            if self.halt_states[symbol]['halted']:
                # Update resume time
                self.halt_states[symbol]['resume_time'] = time.time_ns()
                self.halt_states[symbol]['halted'] = False
                
                # Calculate halt duration
                duration = (self.halt_states[symbol]['resume_time'] - 
                           self.halt_states[symbol]['halt_time']) / 1e9  # Convert to seconds
                           
                logger.info(f"Trading resumed for {symbol} after {duration:.1f} seconds")
                
                # Store resume state
                self._store_halt_state(symbol)
                
                # Notify risk manager
                self._notify_risk_manager(symbol, 'RESUME')
                
        except Exception as e:
            logger.error(f"Error processing resume for {symbol}: {e}")
            
    def update_luld_bands(self, symbol: str, data: Dict):
        """Update LULD bands for a symbol"""
        
        try:
            # Extract LULD data (from generic tick 232)
            upper_band = data.get('luld_high')
            lower_band = data.get('luld_low')
            reference_price = data.get('reference_price')
            
            if not all([upper_band, lower_band, reference_price]):
                return
                
            # Calculate band width percentage
            band_width_pct = ((upper_band - lower_band) / reference_price) * 100
            
            # Update bands
            self.luld_bands[symbol] = {
                'upper': upper_band,
                'lower': lower_band,
                'reference_price': reference_price,
                'band_width_pct': band_width_pct,
                'last_update': time.time_ns()
            }
            
            # Check current price proximity to bands
            self._check_luld_proximity(symbol)
            
            # Store in Redis
            self._store_luld_bands(symbol)
            
        except Exception as e:
            logger.error(f"Error updating LULD bands for {symbol}: {e}")
            
    def calculate_dynamic_luld_bands(self, symbol: str, current_price: float, current_time: datetime) -> Dict:
        """Calculate dynamic LULD bands based on time of day and price level"""
        
        try:
            params = self.luld_params.get(symbol, self.luld_params[self.symbols[0]])
            
            # Determine time-based percentage
            market_open = datetime.now().replace(hour=9, minute=30, second=0)
            market_close = datetime.now().replace(hour=16, minute=0, second=0)
            
            time_since_open = (current_time - market_open).total_seconds() / 60  # Minutes
            time_to_close = (market_close - current_time).total_seconds() / 60  # Minutes
            
            if time_since_open < 15:
                band_pct = params['percentage_bands']['open_first_15min']
            elif time_since_open < 25:
                band_pct = params['percentage_bands']['open_next_10min']
            elif time_to_close < 25:
                band_pct = params['percentage_bands']['close_last_25min']
            else:
                band_pct = params['percentage_bands']['regular']
                
            # Adjust for price level
            if current_price >= 3.00:
                band_pct = min(band_pct, params['price_bands']['above_3'])
            elif current_price >= 1.00:
                band_pct = min(band_pct, params['price_bands']['above_1_below_3'])
            elif current_price >= 0.75:
                band_pct = min(band_pct, params['price_bands']['above_0.75_below_1'])
            else:
                band_pct = params['price_bands']['below_0.75']
                
            # Calculate bands
            upper_band = current_price * (1 + band_pct)
            lower_band = current_price * (1 - band_pct)
            
            # Apply minimum tick constraints
            if current_price < 1.00:
                # Sub-dollar stocks: $0.0001 minimum tick
                upper_band = round(upper_band, 4)
                lower_band = round(lower_band, 4)
            else:
                # Regular stocks: $0.01 minimum tick
                upper_band = round(upper_band, 2)
                lower_band = round(lower_band, 2)
                
            return {
                'upper': upper_band,
                'lower': lower_band,
                'reference_price': current_price,
                'band_width_pct': band_pct * 100,
                'calculation_time': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating LULD bands for {symbol}: {e}")
            return {}
            
    def _check_luld_proximity(self, symbol: str):
        """Check if price is approaching LULD bands"""
        
        try:
            bands = self.luld_bands[symbol]
            if not bands['upper'] or not bands['lower']:
                return
                
            # Get current price
            last_price = float(self.redis.get(f'market:{symbol}:last') or 0)
            if not last_price:
                return
                
            # Calculate distance to bands
            upper_distance = (bands['upper'] - last_price) / last_price
            lower_distance = (last_price - bands['lower']) / last_price
            
            # Track straddle distances
            self.luld_straddle[symbol]['upper_distance'].append((upper_distance, time.time()))
            self.luld_straddle[symbol]['lower_distance'].append((lower_distance, time.time()))
            
            # Warning thresholds
            warning_threshold = 0.01  # 1% from band
            critical_threshold = 0.005  # 0.5% from band
            
            # Check for band approach
            if upper_distance < critical_threshold:
                self._trigger_luld_alert(symbol, 'upper', 'critical', upper_distance)
                self.luld_straddle[symbol]['band_tests'].append({
                    'band': 'upper',
                    'price': last_price,
                    'distance': upper_distance,
                    'time': time.time()
                })
            elif upper_distance < warning_threshold:
                self._trigger_luld_alert(symbol, 'upper', 'warning', upper_distance)
                
            if lower_distance < critical_threshold:
                self._trigger_luld_alert(symbol, 'lower', 'critical', lower_distance)
                self.luld_straddle[symbol]['band_tests'].append({
                    'band': 'lower',
                    'price': last_price,
                    'distance': lower_distance,
                    'time': time.time()
                })
            elif lower_distance < warning_threshold:
                self._trigger_luld_alert(symbol, 'lower', 'warning', lower_distance)
                
        except Exception as e:
            logger.error(f"Error checking LULD proximity for {symbol}: {e}")
            
    def _trigger_luld_alert(self, symbol: str, band: str, severity: str, distance: float):
        """Trigger LULD proximity alert"""
        
        alert = {
            'symbol': symbol,
            'alert_type': 'LULD_PROXIMITY',
            'band': band,
            'severity': severity,
            'distance_pct': distance * 100,
            'timestamp': time.time_ns()
        }
        
        # Store alert in Redis
        self.redis.setex(
            f'alerts:{symbol}:luld:{band}',
            30,
            orjson.dumps(alert).decode('utf-8')
        )
        
        if severity == 'critical':
            logger.warning(f"LULD CRITICAL: {symbol} within {distance*100:.2f}% of {band} band")
            
    def check_circuit_breaker(self, index_value: float, open_value: float) -> Optional[int]:
        """Check if market-wide circuit breaker should trigger"""
        
        try:
            if not open_value:
                return None
                
            decline = (open_value - index_value) / open_value
            
            current_time = datetime.now()
            
            # Level 1 & 2: Only before 3:25 PM
            if current_time.hour < 15 or (current_time.hour == 15 and current_time.minute < 25):
                if decline >= self.circuit_breaker_levels['level2']:
                    return 2
                elif decline >= self.circuit_breaker_levels['level1']:
                    return 1
                    
            # Level 3: Any time
            if decline >= self.circuit_breaker_levels['level3']:
                return 3
                
            return None
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return None
            
    def _capture_pre_halt_state(self, symbol: str, ticker) -> Dict:
        """Capture market state before halt"""
        
        try:
            return {
                'price': ticker.last if hasattr(ticker, 'last') else 0,
                'bid': ticker.bid if hasattr(ticker, 'bid') else 0,
                'ask': ticker.ask if hasattr(ticker, 'ask') else 0,
                'volume': ticker.volume if hasattr(ticker, 'volume') else 0
            }
        except:
            return {'price': 0, 'bid': 0, 'ask': 0, 'volume': 0}
            
    def _determine_halt_type(self, halt_code: str) -> HaltType:
        """Determine halt type from code"""
        
        halt_map = {
            'M': HaltType.LULD_PAUSE,
            'T1': HaltType.NEWS_PENDING,
            'T2': HaltType.NEWS_RELEASED,
            'H4': HaltType.REGULATORY,
            'O': HaltType.OPERATIONAL,
            'T5': HaltType.SINGLE_STOCK_PAUSE,
            'Y': HaltType.SUB_PENNY,
            '1': HaltType.CIRCUIT_BREAKER_L1,
            '2': HaltType.CIRCUIT_BREAKER_L2,
            '3': HaltType.CIRCUIT_BREAKER_L3
        }
        
        return halt_map.get(halt_code, HaltType.REGULATORY)
        
    def _store_halt_state(self, symbol: str):
        """Store halt state in Redis"""
        
        halt_data = self.halt_states[symbol].copy()
        
        # Convert enum to string
        if halt_data['halt_type']:
            halt_data['halt_type'] = halt_data['halt_type'].value
            
        self.redis.setex(
            f'market:{symbol}:halt:state',
            3600,  # 1 hour TTL
            orjson.dumps(halt_data).decode('utf-8')
        )
        
        # Set halt flag for quick checking
        if halt_data['halted']:
            self.redis.set(f'market:{symbol}:halted', 'true')
        else:
            self.redis.delete(f'market:{symbol}:halted')
            
    def _store_luld_bands(self, symbol: str):
        """Store LULD bands in Redis"""
        
        bands = self.luld_bands[symbol]
        
        pipe = self.redis.pipeline()
        
        # Store complete band data
        pipe.setex(
            f'market:{symbol}:luld:bands',
            5,
            orjson.dumps(bands).decode('utf-8')
        )
        
        # Store individual values for quick access
        if bands['upper']:
            pipe.setex(f'market:{symbol}:luld:upper', 5, bands['upper'])
        if bands['lower']:
            pipe.setex(f'market:{symbol}:luld:lower', 5, bands['lower'])
            
        pipe.execute()
        
    def _notify_risk_manager(self, symbol: str, event_type: str):
        """Notify risk manager of halt events"""
        
        notification = {
            'symbol': symbol,
            'event': event_type,
            'halt_state': self.halt_states[symbol],
            'timestamp': time.time_ns()
        }
        
        # Push to risk event queue
        self.redis.lpush(
            'risk:events:halts',
            orjson.dumps(notification).decode('utf-8')
        )
        self.redis.expire('risk:events:halts', 300)
        
    def is_halted(self, symbol: str) -> bool:
        """Check if symbol is currently halted"""
        
        return self.halt_states[symbol]['halted'] or self.market_wide_halt
        
    def get_halt_info(self, symbol: str) -> Dict:
        """Get current halt information"""
        
        return self.halt_states[symbol].copy()
        
    def get_luld_bands(self, symbol: str) -> Dict:
        """Get current LULD bands"""
        
        return self.luld_bands[symbol].copy()