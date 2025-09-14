#!/usr/bin/env python3
"""
Signals Module - Signal Generation and Distribution
Day 6 Implementation: Dry-run signal generation with four strategies

Strategies: 0DTE, 1DTE, 14DTE, MOC
Distribution: Premium (real-time), Basic (60s delay), Free (5min delay)
Guardrails: Freshness, cooldown, idempotency, TTL management
"""

import asyncio
import json
import time
import redis
import redis.asyncio as aioredis
import uuid
import hashlib
import numpy as np
import pytz
from datetime import datetime, timedelta, time as datetime_time
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback


class SignalGenerator:
    """
    Generate trading signals based on analytics metrics and strategy rules.
    Supports multiple strategies with different time horizons and risk profiles.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize signal generator with configuration.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        
        # Signal configuration
        self.signal_config = config.get('modules', {}).get('signals', {})
        self.enabled = self.signal_config.get('enabled', False)
        self.dry_run = self.signal_config.get('dry_run', True)
        
        # Strategy configurations
        self.strategies = self.signal_config.get('strategies', {})
        
        # Guardrail parameters
        self.max_staleness_s = self.signal_config.get('max_staleness_s', 5)
        self.min_confidence = self.signal_config.get('min_confidence', 0.60)
        self.min_refresh_s = self.signal_config.get('min_refresh_s', 2)
        self.cooldown_s = self.signal_config.get('cooldown_s', 30)
        self.ttl_seconds = self.signal_config.get('ttl_seconds', 300)
        self.version = self.signal_config.get('version', 'D6.0.1')
        
        # Track last evaluation time per symbol
        self.last_eval = {}
        
        # Eastern timezone for market hours
        self.eastern = pytz.timezone('US/Eastern')
    
    async def start(self):
        """
        Main signal generation loop.
        Processing frequency: Every 500ms
        """
        if not self.enabled:
            self.logger.info("Signal generator disabled in config")
            return
            
        self.logger.info(f"Starting signal generator (dry_run={self.dry_run})...")
        
        while True:
            try:
                current_time = datetime.now(self.eastern)
                
                # Update heartbeat
                await self.redis.setex('health:signals:heartbeat', 15, current_time.isoformat())
                
                # Process each enabled strategy
                for strategy_name, strategy_config in self.strategies.items():
                    if not strategy_config.get('enabled', False):
                        continue
                        
                    # Check if strategy is active
                    if not self.is_strategy_active(strategy_name, current_time):
                        continue
                    
                    # Process each symbol for this strategy
                    symbols = strategy_config.get('symbols', [])
                    for symbol in symbols:
                        try:
                            # Check minimum refresh interval
                            last_time = self.last_eval.get(f"{symbol}:{strategy_name}", 0)
                            if time.time() - last_time < self.min_refresh_s:
                                continue
                            
                            # Read features from Redis
                            features = await self.read_features(symbol)
                            
                            # Check freshness gate
                            if not self.check_freshness(features):
                                await self.redis.incr('metrics:signals:skipped_stale')
                                continue
                            
                            # Check schema gate
                            if not self.check_schema(features):
                                await self.redis.incr('metrics:signals:skipped_schema')
                                continue
                            
                            # Evaluate strategy conditions
                            confidence, reasons, side = await self.evaluate_strategy(strategy_name, symbol, features)
                            
                            # Increment considered counter
                            await self.redis.incr('metrics:signals:considered')
                            
                            # Check if signal meets threshold
                            min_conf = strategy_config.get('thresholds', {}).get('min_confidence', self.min_confidence * 100)
                            if side == "FLAT" or confidence < min_conf:
                                continue
                            
                            # Check cooldown
                            if not await self.check_cooldown(symbol, side):
                                await self.redis.incr('metrics:signals:cooldown_blocked')
                                continue
                            
                            # Check idempotency
                            signal_id = self.generate_signal_id(symbol, side, features.get('price', 0))
                            if not await self.check_idempotency(signal_id):
                                await self.redis.incr('metrics:signals:duplicates')
                                continue
                            
                            # Create signal object
                            signal = await self.create_signal(symbol, strategy_name, confidence, reasons, features, side)
                            
                            # Enqueue signal for distribution
                            await self.redis.lpush(f'signals:pending:{symbol}', json.dumps(signal))
                            
                            # Write convenience keys
                            ts = int(time.time() * 1000)
                            await self.redis.setex(f'signals:out:{symbol}:{ts}', self.ttl_seconds, json.dumps(signal))
                            await self.redis.setex(f'signals:latest:{symbol}', self.ttl_seconds, json.dumps(signal))
                            
                            # Set cooldown
                            await self.redis.setex(f'signals:cooldown:{symbol}:{side}', self.cooldown_s, '1')
                            
                            # Mark as emitted
                            await self.redis.setex(f'signals:emitted:{signal_id}', self.ttl_seconds, '1')
                            
                            # Increment emitted counter
                            await self.redis.incr('metrics:signals:emitted')
                            
                            # Update last evaluation time
                            self.last_eval[f"{symbol}:{strategy_name}"] = time.time()
                            
                            # Log signal
                            self.logger.info(
                                f"signals DECIDE symbol={symbol} side={side} conf={confidence/100:.2f} "
                                f"vpin={features.get('vpin', 0):.2f} obi={features.get('obi', 0):.2f} "
                                f"gexZ={features.get('gex_z', 0):.1f} dexZ={features.get('dex_z', 0):.1f} "
                                f"rth={self.is_rth(current_time)}"
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Error processing {symbol} for {strategy_name}: {e}")
                            self.logger.error(traceback.format_exc())
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in signal generation loop: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(1)
    
    def get_symbol_strategies(self, symbol: str) -> List[str]:
        """
        Get applicable strategies for a symbol.
        """
        applicable = []
        for strategy_name, strategy_config in self.strategies.items():
            if symbol in strategy_config.get('symbols', []):
                applicable.append(strategy_name)
        return applicable
    
    def is_strategy_active(self, strategy: str, current_time: datetime) -> bool:
        """
        Check if strategy is within its active time window.
        """
        strategy_config = self.strategies.get(strategy, {})
        time_window = strategy_config.get('time_window', {})
        
        if not time_window:
            return True  # No time window means always active
        
        start_str = time_window.get('start', '09:30')
        end_str = time_window.get('end', '16:00')
        
        # Parse time strings
        start_hour, start_min = map(int, start_str.split(':'))
        end_hour, end_min = map(int, end_str.split(':'))
        
        # Create time objects for comparison
        start_time = datetime_time(start_hour, start_min)
        end_time = datetime_time(end_hour, end_min)
        current_time_only = current_time.time()
        
        # Check if within window
        return start_time <= current_time_only <= end_time
    
    async def read_features(self, symbol: str) -> Dict[str, Any]:
        """
        Read all required features from Redis for signal evaluation.
        """
        features = {}
        
        try:
            # Get basic metrics (using async with for proper resource management)
            async with self.redis.pipeline() as pipe:
                pipe.get(f'metrics:{symbol}:vpin')
                pipe.get(f'metrics:{symbol}:obi')
                pipe.get(f'metrics:{symbol}:gex')
                pipe.get(f'metrics:{symbol}:dex')
                pipe.get(f'metrics:{symbol}:gex_z')
                pipe.get(f'metrics:{symbol}:dex_z')
                pipe.get(f'metrics:{symbol}:toxicity')
                pipe.get(f'metrics:{symbol}:regime')
                pipe.get(f'market:{symbol}:last')
                pipe.lrange(f'market:{symbol}:bars', -14, -1)  # Last 14 bars for ATR
                pipe.get(f'options:{symbol}:sweep')
                pipe.get(f'options:{symbol}:unusual_activity')
                pipe.get(f'orderflow:{symbol}:hidden_orders')
                pipe.get(f'gex:{symbol}:by_strike')
                pipe.get(f'imbalance:{symbol}:raw')
                pipe.get(f'imbalance:{symbol}:indicative')
                pipe.get(f'options:{symbol}:chain')  # For MOC delta calculation
                pipe.get(f'market:{symbol}:vwap')  # VWAP if available
                
                results = await pipe.execute()
            
            # Parse basic metrics
            features['vpin'] = float(results[0] or 0)
            
            # OBI might be JSON or float
            if results[1]:
                try:
                    # Try parsing as JSON first (from calculate_order_book_imbalance)
                    obi_data = json.loads(results[1])
                    # Use level1_imbalance and normalize to 0-1 range
                    raw_imbalance = obi_data.get('level1_imbalance', 0)
                    features['obi'] = (raw_imbalance + 1.0) / 2.0  # Map [-1,1] to [0,1]
                except (json.JSONDecodeError, TypeError):
                    # Fall back to float if not JSON
                    features['obi'] = float(results[1])
            else:
                features['obi'] = 0
            
            features['gex'] = float(results[2] or 0)
            features['dex'] = float(results[3] or 0)
            features['gex_z'] = float(results[4] or 0)
            features['dex_z'] = float(results[5] or 0)
            features['toxicity'] = float(results[6] or 0)
            features['regime'] = results[7] or 'NORMAL'
            
            # Parse market data - could be JSON or direct float
            if results[8]:
                try:
                    # Try parsing as JSON first
                    if isinstance(results[8], str) and results[8].startswith('{'):
                        market_data = json.loads(results[8])
                        features['price'] = market_data.get('price', 0)
                        features['timestamp'] = market_data.get('ts', 0)
                    else:
                        # Direct float value - DO NOT stamp with 'now' (bug fix #2)
                        features['price'] = float(results[8])
                        features['timestamp'] = 0  # Mark as unknown timestamp
                    features['age_s'] = (time.time() * 1000 - features['timestamp']) / 1000 if features['timestamp'] else 999
                except (json.JSONDecodeError, ValueError):
                    features['price'] = 0
                    features['timestamp'] = 0
                    features['age_s'] = 999
            else:
                features['price'] = 0
                features['timestamp'] = 0
                features['age_s'] = 999
            
            # Parse bars for ATR calculation
            try:
                features['bars'] = [json.loads(bar) for bar in results[9] if bar]
            except (json.JSONDecodeError, TypeError):
                features['bars'] = []
            
            # Parse options flow indicators
            features['sweep'] = float(results[10] or 0)
            features['unusual_activity'] = float(results[11] or 0)
            
            # Hidden orders might be JSON or float
            if results[12]:
                try:
                    if isinstance(results[12], str) and results[12].startswith('{'):
                        hidden_data = json.loads(results[12])
                        features['hidden_orders'] = hidden_data.get('score', 0)
                    else:
                        features['hidden_orders'] = float(results[12])
                except (json.JSONDecodeError, ValueError):
                    features['hidden_orders'] = 0
            else:
                features['hidden_orders'] = 0
            
            # Parse GEX strike data
            if results[13]:
                try:
                    features['gex_by_strike'] = json.loads(results[13])
                except (json.JSONDecodeError, TypeError):
                    features['gex_by_strike'] = []
            else:
                features['gex_by_strike'] = []
            
            # Parse imbalance data for MOC
            if results[14]:
                try:
                    imb = json.loads(results[14])
                    features['imbalance_side'] = imb.get('side', '')
                    features['imbalance_total'] = imb.get('total', 0)
                    features['imbalance_ratio'] = imb.get('ratio', 0)
                    features['imbalance_paired'] = imb.get('paired', 0)
                except (json.JSONDecodeError, TypeError):
                    features['imbalance_side'] = ''
                    features['imbalance_total'] = 0
                    features['imbalance_ratio'] = 0
                    features['imbalance_paired'] = 0
            else:
                features['imbalance_side'] = ''
                features['imbalance_total'] = 0
                features['imbalance_ratio'] = 0
                features['imbalance_paired'] = 0
            
            if results[15]:
                try:
                    ind = json.loads(results[15])
                    features['indicative_price'] = ind.get('price', 0)
                    features['near_close_offset_bps'] = ind.get('near_close_offset_bps', 0)
                except (json.JSONDecodeError, TypeError):
                    features['indicative_price'] = 0
                    features['near_close_offset_bps'] = 0
            else:
                features['indicative_price'] = 0
                features['near_close_offset_bps'] = 0
            
            # Parse options chain for MOC (result 16)
            if results[16]:
                try:
                    features['options_chain'] = json.loads(results[16])
                except (json.JSONDecodeError, TypeError):
                    features['options_chain'] = None
            else:
                features['options_chain'] = None
            
            # Parse VWAP (result 17)
            if results[17]:
                try:
                    features['vwap'] = float(results[17])
                except (ValueError, TypeError):
                    features['vwap'] = None
            else:
                features['vwap'] = None
            
            # Calculate additional features
            features['atr'] = await self.calculate_atr(features['bars'])
            features['gamma_pin_proximity'] = self.calculate_gamma_pin_proximity(features)
            features['gamma_pull_dir'] = self.calculate_gamma_pull_direction(features)
            
        except Exception as e:
            self.logger.error(f"Error reading features for {symbol}: {e}")
            features['age_s'] = 999  # Mark as stale on error
        
        return features
    
    async def evaluate_strategy(self, strategy: str, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Route to appropriate strategy evaluator.
        """
        if strategy == '0dte':
            return self.evaluate_0dte_conditions(symbol, features)
        elif strategy == '1dte':
            return self.evaluate_1dte_conditions(symbol, features)
        elif strategy == '14dte':
            return self.evaluate_14dte_conditions(symbol, features)
        elif strategy == 'moc':
            return self.evaluate_moc_conditions(symbol, features)
        else:
            return 0, [], "FLAT"
    
    def evaluate_0dte_conditions(self, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Evaluate 0DTE strategy conditions (intraday gamma-driven moves).
        Enhanced with gamma squeeze detection and dealer positioning.
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('0dte', {})
        thresholds = strategy_config.get('thresholds', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # 1. VPIN pressure analysis (30 points max)
        vpin = features.get('vpin', 0)
        vpin_min = thresholds.get('vpin_min', 0.40)
        if vpin >= vpin_min:
            # Scale points based on VPIN intensity
            intensity = min((vpin - vpin_min) / (0.8 - vpin_min), 1.0)  # 0.4->0.8 maps to 0->1
            points = int(weights.get('vpin', 30) * (0.5 + 0.5 * intensity))  # 15-30 points
            confidence += points
            if vpin > 0.7:
                reasons.append(f"Strong VPIN pressure ({vpin:.2f})")
            else:
                reasons.append(f"VPIN pressure ({vpin:.2f})")
        
        # 2. Order Book Imbalance with depth analysis (25 points max)
        obi = features.get('obi', 0.5)
        obi_min = thresholds.get('obi_min', 0.30)
        obi_deviation = abs(obi - 0.5)  # Distance from neutral
        if obi_deviation >= obi_min:
            # Scale based on imbalance strength
            points = int(weights.get('obi', 25) * (obi_deviation / 0.5))  # Max at 0 or 1
            confidence += points
            if obi > 0.7:
                reasons.append(f"Strong bid imbalance ({obi:.2f})")
            elif obi < 0.3:
                reasons.append(f"Strong ask imbalance ({obi:.2f})")
            else:
                reasons.append(f"OBI skew ({obi:.2f})")
        
        # 3. Enhanced Gamma Analysis (30 points max)
        gamma_proximity = features.get('gamma_pin_proximity', 0)
        gex_by_strike = features.get('gex_by_strike', [])
        price = features.get('price', 0)
        gamma_pin_distance = thresholds.get('gamma_pin_distance', 0.005)
        
        if gamma_proximity > 0 and gex_by_strike and price > 0:
            # Find nearest major gamma strike
            pin_strike = self.find_gamma_pin(features)
            
            # Calculate total gamma above and below current price
            gamma_above = sum(s.get('gex', 0) for s in gex_by_strike if s.get('strike', 0) > price)
            gamma_below = sum(s.get('gex', 0) for s in gex_by_strike if s.get('strike', 0) < price)
            gamma_total = abs(gamma_above) + abs(gamma_below)
            
            if gamma_total > 0:
                # Detect gamma squeeze (high concentration near price)
                gamma_concentration = gamma_proximity  # Already 0-1 score
                squeeze_detected = gamma_concentration > 0.7 and abs(price - pin_strike) / price < 0.003
                
                if squeeze_detected:
                    points = weights.get('gamma_proximity', 30)
                    confidence += points
                    reasons.append(f"Gamma squeeze at {pin_strike:.0f}")
                elif gamma_proximity > 0.5:
                    points = int(weights.get('gamma_proximity', 30) * gamma_proximity)
                    confidence += points
                    reasons.append(f"Near gamma pin {pin_strike:.0f}")
                else:
                    # Mild gamma influence
                    points = int(weights.get('gamma_proximity', 30) * gamma_proximity * 0.5)
                    confidence += points
                    reasons.append(f"Gamma influence")
        
        # 4. Sweep detection with size analysis (15 points max)
        sweep_score = features.get('sweep', 0)
        if sweep_score >= 1:
            # Check sweep size and aggressiveness
            bars = features.get('bars', [])
            if len(bars) >= 3:
                # Calculate recent volatility for context
                recent_moves = [abs(bars[i].get('close', 0) - bars[i-1].get('close', 0)) 
                               for i in range(1, min(4, len(bars)))]
                avg_move = sum(recent_moves) / len(recent_moves) if recent_moves else 0
                current_move = abs(bars[-1].get('close', 0) - bars[-2].get('close', 0)) if len(bars) >= 2 else 0
                
                if avg_move > 0 and current_move > 2 * avg_move:
                    # Large aggressive sweep
                    points = weights.get('sweep', 15)
                    reasons.append("Aggressive sweep detected")
                else:
                    # Normal sweep
                    points = int(weights.get('sweep', 15) * 0.7)
                    reasons.append("Sweep detected")
            else:
                points = int(weights.get('sweep', 15) * 0.5)
                reasons.append("Sweep activity")
            confidence += points
        
        # 5. Determine direction with enhanced logic
        side = self._determine_0dte_direction(features, obi, price)
        
        # 6. Apply gamma-based direction override for strong setups
        if gamma_proximity > 0.8 and side != "FLAT":
            gamma_pull_dir = features.get('gamma_pull_dir', '')
            if gamma_pull_dir == 'UP' and side == 'SHORT':
                # Gamma fighting our direction, reduce confidence
                confidence = int(confidence * 0.7)
                reasons.append("(gamma resistance)")
            elif gamma_pull_dir == 'DOWN' and side == 'LONG':
                # Gamma fighting our direction, reduce confidence
                confidence = int(confidence * 0.7)
                reasons.append("(gamma resistance)")
            elif gamma_pull_dir and gamma_pull_dir == ('UP' if side == 'LONG' else 'DOWN'):
                # Gamma supporting our direction
                confidence = min(100, int(confidence * 1.2))
                reasons.append("(gamma support)")
        
        return confidence, reasons, side
    
    def _determine_0dte_direction(self, features: Dict[str, Any], obi: float, price: float) -> str:
        """
        Determine 0DTE trade direction using multiple factors.
        """
        vwap = features.get('vwap')
        bars = features.get('bars', [])
        toxicity = features.get('toxicity', 0.5)
        
        # Direction scoring system
        long_score = 0
        short_score = 0
        
        # 1. OBI signal (strongest weight)
        if obi > 0.65:
            long_score += 3
        elif obi > 0.55:
            long_score += 1
        elif obi < 0.35:
            short_score += 3
        elif obi < 0.45:
            short_score += 1
        
        # 2. VWAP position (if available)
        if vwap and vwap > 0:
            vwap_diff = (price - vwap) / vwap
            if abs(vwap_diff) > 0.001:  # 0.1% threshold
                if price > vwap and obi > 0.5:
                    long_score += 2
                elif price < vwap and obi < 0.5:
                    short_score += 2
                elif price > vwap * 1.005:  # Extended above VWAP
                    short_score += 1  # Mean reversion
                elif price < vwap * 0.995:  # Extended below VWAP
                    long_score += 1  # Mean reversion
        
        # 3. Recent momentum
        if len(bars) >= 5:
            # 5-bar momentum
            momentum_5bar = (bars[-1].get('close', 0) - bars[-5].get('close', 0)) / bars[-5].get('close', 1)
            # 2-bar momentum
            momentum_2bar = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
            
            if momentum_5bar > 0.002 and momentum_2bar > 0:
                long_score += 1
            elif momentum_5bar < -0.002 and momentum_2bar < 0:
                short_score += 1
            
            # Momentum divergence (potential reversal)
            if momentum_5bar > 0.003 and momentum_2bar < -0.001:
                short_score += 1  # Exhaustion
            elif momentum_5bar < -0.003 and momentum_2bar > 0.001:
                long_score += 1  # Exhaustion
        
        # 4. Toxicity adjustment
        if toxicity > 0.7:
            # High toxicity suggests informed flow
            if obi > 0.5:
                long_score += 1
            else:
                short_score += 1
        
        # Final decision
        if long_score > short_score + 1:
            return "LONG"
        elif short_score > long_score + 1:
            return "SHORT"
        else:
            return "FLAT"
    
    def evaluate_1dte_conditions(self, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Evaluate 1DTE strategy conditions (overnight positioning).
        Enhanced with end-of-day momentum, overnight gap probability, and event analysis.
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('1dte', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # Get current time for EOD analysis
        current_time = datetime.now(self.eastern)
        minutes_to_close = (16 * 60) - (current_time.hour * 60 + current_time.minute)
        is_power_hour = 15 <= current_time.hour < 16
        is_last_30min = minutes_to_close <= 30
        
        # 1. Volatility regime analysis (20 points max)
        regime = features.get('regime', 'NORMAL')
        if regime == 'HIGH':
            # High volatility favors overnight moves
            points = weights.get('volatility_regime', 20)
            confidence += points
            reasons.append("HIGH vol regime (gap likely)")
            
            # Check if volatility is expanding or contracting
            bars = features.get('bars', [])
            if len(bars) >= 10:
                recent_ranges = [(b.get('high', 0) - b.get('low', 0)) / b.get('low', 1) 
                               for b in bars[-10:] if b.get('low', 0) > 0]
                if len(recent_ranges) >= 5:
                    vol_trend = recent_ranges[-1] / (sum(recent_ranges[:5]) / 5) if sum(recent_ranges[:5]) > 0 else 1
                    if vol_trend > 1.5:
                        confidence += 5
                        reasons.append("Vol expanding")
        elif regime == 'LOW':
            # Low volatility might mean compression before expansion
            points = int(weights.get('volatility_regime', 20) * 0.5)
            confidence += points
            reasons.append("LOW vol (breakout setup)")
        
        # 2. Order Book Imbalance EOD analysis (30 points max)
        obi = features.get('obi', 0.5)
        if is_power_hour and abs(obi - 0.5) >= 0.15:
            # Strong EOD imbalance
            imbalance_strength = abs(obi - 0.5) * 2  # Scale 0-1
            points = int(weights.get('obi', 30) * imbalance_strength)
            confidence += points
            
            if obi > 0.65:
                reasons.append(f"Strong EOD buying ({obi:.2f})")
            elif obi < 0.35:
                reasons.append(f"Strong EOD selling ({obi:.2f})")
            else:
                reasons.append(f"EOD imbalance ({obi:.2f})")
        
        # 3. GEX positioning for overnight risk (25 points max)
        gex_z = features.get('gex_z', 0)
        gex = features.get('gex', 0)
        
        if abs(gex_z) >= 0.5:
            # Significant GEX positioning
            if gex < 0:
                # Negative GEX = dealers short gamma = higher overnight volatility
                points = int(weights.get('gex', 25) * min(abs(gex_z) / 2, 1))
                confidence += points
                reasons.append(f"Negative GEX ({gex_z:.1f}σ) - volatile overnight")
            else:
                # Positive GEX = dealers long gamma = dampened moves
                points = int(weights.get('gex', 25) * min(abs(gex_z) / 2, 1) * 0.7)
                confidence += points
                reasons.append(f"Positive GEX ({gex_z:.1f}σ) - pinned overnight")
        
        # 4. VPIN flow analysis for institutional positioning (25 points max)
        vpin = features.get('vpin', 0.5)
        if vpin >= 0.35:
            # Calculate flow intensity
            flow_intensity = min((vpin - 0.35) / 0.35, 1)  # 0.35->0.70 maps to 0->1
            
            # Check if flow is accelerating into close
            if is_last_30min:
                flow_intensity *= 1.3  # Boost for last 30 minutes
            
            points = int(weights.get('vpin', 25) * flow_intensity)
            confidence += points
            
            if vpin > 0.6:
                reasons.append(f"Strong VPIN flow ({vpin:.2f})")
            else:
                reasons.append(f"VPIN accumulation ({vpin:.2f})")
        
        # 5. End-of-day momentum analysis (bonus points)
        bars = features.get('bars', [])
        if len(bars) >= 14 and is_power_hour:
            # Calculate momentum over different timeframes
            momentum_14bar = (bars[-1].get('close', 0) - bars[-14].get('close', 0)) / bars[-14].get('close', 1)
            momentum_5bar = (bars[-1].get('close', 0) - bars[-5].get('close', 0)) / bars[-5].get('close', 1)
            momentum_2bar = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
            
            # Check for momentum acceleration
            if abs(momentum_2bar) > abs(momentum_5bar) and abs(momentum_5bar) > abs(momentum_14bar):
                # Accelerating momentum into close
                confidence += 10
                reasons.append("Momentum accelerating")
            elif abs(momentum_14bar) > 0.005:  # 0.5% move
                # Strong trend day
                confidence += 5
                reasons.append("Trend day")
        
        # 6. Determine direction with overnight gap probability
        side = self._determine_1dte_direction(features, vpin, obi, is_last_30min)
        
        # 7. Friday/Monday adjustments
        if current_time.weekday() == 4:  # Friday
            # Weekend risk adjustment
            if side != "FLAT" and confidence < 70:
                confidence = int(confidence * 0.85)
                reasons.append("(weekend risk discount)")
        elif current_time.weekday() == 0:  # Monday
            # Monday gap tendency
            if side != "FLAT":
                confidence = min(100, int(confidence * 1.1))
                reasons.append("(Monday gap boost)")
        
        return confidence, reasons, side
    
    def _determine_1dte_direction(self, features: Dict[str, Any], vpin: float, obi: float, is_last_30min: bool) -> str:
        """
        Determine 1DTE overnight direction using EOD flow analysis.
        """
        bars = features.get('bars', [])
        dex = features.get('dex', 0)
        toxicity = features.get('toxicity', 0.5)
        
        # Direction scoring
        long_score = 0
        short_score = 0
        
        # 1. EOD momentum (strongest signal for overnight)
        if len(bars) >= 10:
            # Last 10 bars momentum (about 50 minutes)
            eod_momentum = (bars[-1].get('close', 0) - bars[-10].get('close', 0)) / bars[-10].get('close', 1)
            # Last 2 bars momentum
            recent_momentum = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
            
            if eod_momentum > 0.003:  # 0.3% move
                long_score += 3
                if recent_momentum > 0:  # Still moving up
                    long_score += 1
            elif eod_momentum < -0.003:
                short_score += 3
                if recent_momentum < 0:  # Still moving down
                    short_score += 1
            
            # Check for reversal patterns
            if len(bars) >= 14:
                high_14 = max(b.get('high', 0) for b in bars[-14:])
                low_14 = min(b.get('low', 0) for b in bars[-14:])
                close = bars[-1].get('close', 0)
                
                if close > high_14 * 0.998:  # Near highs
                    long_score += 2  # Breakout continuation
                elif close < low_14 * 1.002:  # Near lows
                    short_score += 2  # Breakdown continuation
        
        # 2. VPIN directional pressure
        if vpin > 0.6:
            # Strong selling pressure
            if obi < 0.4:
                short_score += 3
            else:
                short_score += 1
        elif vpin < 0.4:
            # Strong buying pressure
            if obi > 0.6:
                long_score += 3
            else:
                long_score += 1
        
        # 3. Order book EOD positioning
        if is_last_30min:
            # Last 30 minutes matter most
            if obi > 0.65:
                long_score += 2
            elif obi < 0.35:
                short_score += 2
        
        # 4. Delta exposure lean
        if abs(dex) > 1e9:  # Significant delta exposure
            if dex > 0:
                long_score += 1  # Positive delta build
            else:
                short_score += 1  # Negative delta build
        
        # 5. Smart money detection via toxicity
        if toxicity > 0.65 and is_last_30min:
            # Informed flow in last 30 minutes
            if obi > 0.5:
                long_score += 2
            else:
                short_score += 2
        
        # Final decision with higher threshold for overnight risk
        if long_score >= short_score + 2:
            return "LONG"
        elif short_score >= long_score + 2:
            return "SHORT"
        else:
            return "FLAT"
    
    def evaluate_14dte_conditions(self, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Evaluate 14DTE strategy conditions (swing trades on unusual activity).
        Enhanced with institutional flow detection, sweep characterization, and smart money tracking.
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('14dte', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # Track institutional flow characteristics
        institutional_signals = 0
        retail_signals = 0
        
        # 1. Unusual options activity analysis (40 points max)
        unusual = features.get('unusual_activity', 0)
        options_chain = features.get('options_chain')
        
        if unusual >= 0.6:
            # Analyze the nature of unusual activity
            unusual_intensity = min((unusual - 0.6) / 0.4, 1.0)  # 0.6->1.0 maps to 0->1
            
            # Check if we have options chain for deeper analysis
            if options_chain and isinstance(options_chain, list):
                # Analyze put/call skew
                calls_oi = sum(opt.get('open_interest', 0) for opt in options_chain 
                              if opt.get('type') == 'CALL')
                puts_oi = sum(opt.get('open_interest', 0) for opt in options_chain 
                             if opt.get('type') == 'PUT')
                
                total_oi = calls_oi + puts_oi
                if total_oi > 0:
                    call_ratio = calls_oi / total_oi
                    
                    # Extreme skew detection
                    if call_ratio > 0.7:
                        institutional_signals += 2
                        reasons.append(f"Heavy call skew ({call_ratio:.0%})")
                    elif call_ratio < 0.3:
                        institutional_signals += 2
                        reasons.append(f"Heavy put skew ({(1-call_ratio):.0%})")
                    
                    # Check for large individual positions
                    large_positions = [opt for opt in options_chain 
                                     if opt.get('open_interest', 0) > 5000]
                    if large_positions:
                        institutional_signals += 1
                        reasons.append(f"{len(large_positions)} large OI strikes")
            
            # Base points for unusual activity
            base_points = 24 + (unusual_intensity * 16)  # 24-40 range
            points = min(int(base_points), weights.get('unusual_options', 40))
            confidence += points
            
            if unusual > 0.85:
                reasons.append("Extreme unusual options")
            else:
                reasons.append("Unusual options activity")
        
        # 2. Sweep detection and characterization (30 points max)
        sweep_score = features.get('sweep', 0)
        if sweep_score >= 1:
            # Analyze sweep characteristics
            bars = features.get('bars', [])
            
            if len(bars) >= 5:
                # Calculate sweep aggressiveness
                recent_volumes = [b.get('volume', 0) for b in bars[-5:]]
                avg_volume = sum(recent_volumes[:-1]) / 4 if len(recent_volumes) > 1 else 1
                current_volume = recent_volumes[-1] if recent_volumes else 0
                
                if avg_volume > 0:
                    volume_spike = current_volume / avg_volume
                    
                    if volume_spike > 3:
                        # Aggressive institutional sweep
                        points = weights.get('sweep', 30)
                        institutional_signals += 2
                        reasons.append(f"Aggressive sweep ({volume_spike:.1f}x vol)")
                    elif volume_spike > 2:
                        # Moderate sweep
                        points = int(weights.get('sweep', 30) * 0.8)
                        institutional_signals += 1
                        reasons.append(f"Sweep detected ({volume_spike:.1f}x vol)")
                    else:
                        # Mild sweep
                        points = int(weights.get('sweep', 30) * 0.6)
                        reasons.append("Sweep activity")
                else:
                    points = int(weights.get('sweep', 30) * 0.5)
                    reasons.append("Sweep pattern")
            else:
                points = int(weights.get('sweep', 30) * 0.4)
                reasons.append("Possible sweep")
            
            confidence += points
        
        # 3. Hidden order detection (20 points max)
        hidden = features.get('hidden_orders', 0)
        if hidden >= 0.5:
            # Hidden orders suggest institutional activity
            hidden_intensity = min((hidden - 0.5) / 0.3, 1.0)  # 0.5->0.8 maps to 0->1
            
            if hidden > 0.75:
                # Strong hidden order presence
                points = weights.get('hidden_orders', 20)
                institutional_signals += 2
                reasons.append(f"Strong hidden orders ({hidden:.2f})")
            elif hidden > 0.65:
                # Moderate hidden orders
                points = int(weights.get('hidden_orders', 20) * 0.8)
                institutional_signals += 1
                reasons.append(f"Hidden order flow ({hidden:.2f})")
            else:
                # Light hidden orders
                points = int(weights.get('hidden_orders', 20) * 0.6)
                reasons.append("Some hidden orders")
            
            confidence += points
        
        # 4. Delta exposure analysis (10 points max)
        dex = features.get('dex', 0)
        dex_z = features.get('dex_z', 0)
        
        if abs(dex_z) >= 0.5:
            # Significant delta positioning
            dex_intensity = min(abs(dex_z) / 2, 1.0)  # 0.5->2.0 z-score maps to 0->1
            
            if abs(dex) > 10e9:  # >$10B delta
                # Massive positioning
                points = weights.get('dex', 10)
                institutional_signals += 1
                if dex > 0:
                    reasons.append(f"Massive call delta (${dex/1e9:.1f}B)")
                else:
                    reasons.append(f"Massive put delta (${abs(dex)/1e9:.1f}B)")
            elif abs(dex) > 5e9:  # >$5B delta
                # Large positioning
                points = int(weights.get('dex', 10) * 0.7)
                if dex_z > 1:
                    reasons.append(f"Large delta build ({dex_z:.1f}σ)")
                else:
                    reasons.append("Delta positioning")
            else:
                # Notable positioning
                points = int(weights.get('dex', 10) * dex_intensity * 0.5)
                if points > 0:
                    reasons.append("Delta flow")
            
            confidence += points
        
        # 5. Additional smart money indicators
        toxicity = features.get('toxicity', 0.5)
        vpin = features.get('vpin', 0.5)
        
        if toxicity > 0.7 and vpin < 0.4:
            # Low VPIN with high toxicity = smart money accumulation
            confidence += 5
            institutional_signals += 1
            reasons.append("Smart money pattern")
        elif toxicity > 0.8:
            # Very high toxicity = informed flow
            confidence += 3
            reasons.append("Informed flow")
        
        # 6. Time of day adjustments
        current_time = datetime.now(self.eastern)
        if 9.5 <= current_time.hour + current_time.minute/60 <= 10.5:
            # First hour often has institutional positioning
            if institutional_signals >= 2:
                confidence = min(100, int(confidence * 1.15))
                reasons.append("(morning institutional)")
        elif 15 <= current_time.hour < 16:
            # Power hour positioning
            if sweep_score >= 1 or hidden > 0.6:
                confidence = min(100, int(confidence * 1.1))
                reasons.append("(power hour setup)")
        
        # 7. Determine direction with institutional vs retail analysis
        side = self._determine_14dte_direction(
            features, unusual, dex, institutional_signals, retail_signals
        )
        
        # 8. Confidence boost for strong institutional consensus
        if side != "FLAT" and institutional_signals >= 3:
            confidence = min(100, int(confidence * 1.2))
            reasons.append("(institutional consensus)")
        elif side != "FLAT" and institutional_signals == 0 and retail_signals > 2:
            # Retail-driven, reduce confidence
            confidence = int(confidence * 0.8)
            reasons.append("(retail-driven)")
        
        return confidence, reasons, side
    
    def _determine_14dte_direction(self, features: Dict[str, Any], unusual: float, 
                                   dex: float, inst_signals: int, retail_signals: int) -> str:
        """
        Determine 14DTE direction using institutional flow analysis.
        """
        bars = features.get('bars', [])
        options_chain = features.get('options_chain')
        
        # Weighted voting system
        long_score = 0
        short_score = 0
        
        # 1. Delta exposure direction (strongest signal)
        if dex > 5e9:
            long_score += 3
        elif dex > 1e9:
            long_score += 2
        elif dex < -5e9:
            short_score += 3
        elif dex < -1e9:
            short_score += 2
        
        # 2. Unusual activity with options chain analysis
        if unusual > 0.6 and options_chain:
            # Analyze strike distribution
            price = features.get('price', 0)
            if price > 0:
                otm_calls = sum(1 for opt in options_chain 
                              if opt.get('type') == 'CALL' and opt.get('strike', 0) > price * 1.02)
                otm_puts = sum(1 for opt in options_chain 
                             if opt.get('type') == 'PUT' and opt.get('strike', 0) < price * 0.98)
                
                if otm_calls > otm_puts * 1.5:
                    long_score += 2
                elif otm_puts > otm_calls * 1.5:
                    short_score += 2
        
        # 3. Price action and momentum
        if len(bars) >= 20:
            # 20-bar trend (roughly 100 minutes)
            trend_20 = (bars[-1].get('close', 0) - bars[-20].get('close', 0)) / bars[-20].get('close', 1)
            # 5-bar momentum
            momentum_5 = (bars[-1].get('close', 0) - bars[-5].get('close', 0)) / bars[-5].get('close', 1)
            
            if trend_20 > 0.005 and momentum_5 > 0:  # Uptrend with momentum
                long_score += 2
            elif trend_20 < -0.005 and momentum_5 < 0:  # Downtrend with momentum
                short_score += 2
            elif abs(trend_20) < 0.002:  # Consolidation
                # Look for breakout direction
                if momentum_5 > 0.002:
                    long_score += 1
                elif momentum_5 < -0.002:
                    short_score += 1
        
        # 4. Sweep direction
        if features.get('sweep', 0) >= 1 and len(bars) >= 2:
            last_move = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
            if last_move > 0.001:
                long_score += 1
            elif last_move < -0.001:
                short_score += 1
        
        # 5. Hidden order direction
        hidden = features.get('hidden_orders', 0)
        if hidden > 0.6:
            # Hidden orders with price action
            if len(bars) >= 3:
                recent_trend = (bars[-1].get('close', 0) - bars[-3].get('close', 0)) / bars[-3].get('close', 1)
                if recent_trend > 0:
                    long_score += 1
                elif recent_trend < 0:
                    short_score += 1
        
        # 6. Institutional vs retail weighting
        if inst_signals > retail_signals:
            # Trust institutional direction more
            threshold = 1
        else:
            # Need stronger signal if retail-driven
            threshold = 2
        
        # Final decision
        if long_score > short_score + threshold:
            return "LONG"
        elif short_score > long_score + threshold:
            return "SHORT"
        else:
            return "FLAT"
    
    def evaluate_moc_conditions(self, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Evaluate MOC strategy conditions (market-on-close imbalance options play).
        Enhanced with gamma magnet analysis, imbalance dynamics, and dealer hedging effects.
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('moc', {})
        thresholds = strategy_config.get('thresholds', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # Get current time for MOC window validation
        current_time = datetime.now(self.eastern)
        minutes_to_close = (16 * 60) - (current_time.hour * 60 + current_time.minute)
        
        # 1. Imbalance analysis with hard gates
        imbalance_total = features.get('imbalance_total', 0)
        imbalance_ratio = features.get('imbalance_ratio', 0)
        imbalance_side = features.get('imbalance_side', '')
        imbalance_paired = features.get('imbalance_paired', 0)
        indicative_price = features.get('indicative_price', 0)
        near_close_offset_bps = features.get('near_close_offset_bps', 0)
        
        min_notional = thresholds.get('min_imbalance_notional', 2e9)
        min_ratio = thresholds.get('min_imbalance_ratio', 0.60)
        
        # Hard gate: insufficient imbalance
        if imbalance_total < min_notional or imbalance_ratio < min_ratio:
            return 0, [], "FLAT"
        
        # Calculate imbalance strength with nuanced scoring (45 points max)
        # Consider both absolute size and ratio
        size_score = min(1.0, imbalance_total / 5e9)  # Max at $5B
        ratio_score = min(1.0, (imbalance_ratio - 0.60) / 0.30)  # 60-90% range
        
        # Check if imbalance is growing (compare to paired)
        if imbalance_paired > 0:
            unpaired_pct = 1 - (imbalance_paired / imbalance_total)
            if unpaired_pct > 0.7:  # 70%+ unpaired is strong
                imbalance_multiplier = 1.2
                reasons.append(f"{unpaired_pct:.0%} unpaired")
            else:
                imbalance_multiplier = 1.0
        else:
            imbalance_multiplier = 1.0
        
        imbalance_points = int(weights.get('imbalance_strength', 45) * 
                               (size_score * 0.6 + ratio_score * 0.4) * imbalance_multiplier)
        confidence += imbalance_points
        
        # Format imbalance message
        if imbalance_total >= 1e9:
            reasons.append(f"${imbalance_total/1e9:.1f}B {imbalance_side} imbalance")
        else:
            reasons.append(f"${imbalance_total/1e6:.0f}M {imbalance_side} imbalance")
        reasons.append(f"ratio {imbalance_ratio:.0%}")
        
        # 2. Enhanced gamma pin analysis (25 points max)
        price = features.get('price', 0)
        gex_by_strike = features.get('gex_by_strike', [])
        gamma_proximity = features.get('gamma_pin_proximity', 0)
        
        if price > 0 and gex_by_strike:
            pin_strike = self.find_gamma_pin(features)
            
            if pin_strike > 0:
                pin_distance_pct = abs(price - pin_strike) / price
                gamma_pull_dir = features.get('gamma_pull_dir', '')
                
                # Calculate gamma concentration around pin
                gamma_concentration = self._calculate_gamma_concentration(gex_by_strike, pin_strike, price)
                
                if pin_distance_pct <= 0.005:  # Within 0.5%
                    # Strong gamma magnet effect
                    if gamma_concentration > 0.5:  # >50% of gamma within 1% of pin
                        gamma_points = weights.get('gamma_pull', 25)
                        reasons.append(f"Strong gamma magnet at {pin_strike:.0f}")
                        
                        # Check if we're fighting or following gamma
                        if indicative_price > 0:
                            indicative_vs_pin = (indicative_price - pin_strike) / pin_strike
                            if abs(indicative_vs_pin) > 0.002:  # 0.2% difference
                                if (indicative_vs_pin > 0 and imbalance_side == 'SELL') or \
                                   (indicative_vs_pin < 0 and imbalance_side == 'BUY'):
                                    # Imbalance fighting gamma
                                    gamma_points = int(gamma_points * 0.7)
                                    reasons.append("(imbalance vs gamma)")
                    else:
                        # Moderate gamma influence
                        gamma_points = int(weights.get('gamma_pull', 25) * 
                                         (1 - pin_distance_pct / 0.005) * gamma_concentration)
                        reasons.append(f"Gamma influence to {pin_strike:.0f}")
                    
                    confidence += gamma_points
                
                # Dealer hedging dynamics
                if minutes_to_close <= 10:  # Last 10 minutes
                    # Dealers actively hedge in final minutes
                    total_gamma = sum(abs(s.get('gex', 0)) for s in gex_by_strike)
                    if total_gamma > 1e9:  # Significant gamma exposure
                        confidence += 5
                        reasons.append("Dealer hedge pressure")
        
        # 3. Order book imbalance near close (20 points max)
        obi = features.get('obi', 0.5)
        obi_deviation = abs(obi - 0.5)
        
        if obi_deviation > 0.15 and minutes_to_close <= 20:
            # OBI matters more in final 20 minutes
            time_weight = 1.0 if minutes_to_close <= 10 else 0.7
            obi_points = int(weights.get('obi', 20) * (obi_deviation * 2) * time_weight)
            confidence += obi_points
            
            if obi > 0.65:
                reasons.append(f"Bid pressure ({obi:.2f})")
            elif obi < 0.35:
                reasons.append(f"Ask pressure ({obi:.2f})")
            else:
                reasons.append(f"OBI skew ({obi:.2f})")
        
        # 4. Near indicative price analysis
        if indicative_price > 0 and price > 0:
            indicative_diff_pct = (indicative_price - price) / price
            
            if abs(indicative_diff_pct) > 0.002:  # >0.2% difference
                # Large indicative move expected
                confidence += 5
                if indicative_diff_pct > 0:
                    reasons.append(f"Indicative {indicative_diff_pct:.1%} above")
                else:
                    reasons.append(f"Indicative {abs(indicative_diff_pct):.1%} below")
        
        # 5. Day-of-week factors (10 points max)
        weekday = current_time.weekday()
        if weekday == 4:  # Friday
            # Friday: index rebalancing, weekly options expiry
            friday_points = weights.get('friday_factor', 10)
            confidence += friday_points
            reasons.append("Friday dynamics")
            
            # Check for OPEX (3rd Friday)
            if 15 <= current_time.day <= 21:
                confidence += 5
                reasons.append("(OPEX week)")
        elif weekday == 0:  # Monday
            # Monday: weekend gap risk realized
            confidence += 3
            reasons.append("Monday reposition")
        
        # 6. Determine direction with sophisticated logic
        side = self._determine_moc_direction(
            features, imbalance_side, imbalance_total, 
            gamma_proximity, indicative_price, price
        )
        
        # 7. Final adjustments based on time to close
        if minutes_to_close > 20:
            # Too early, reduce confidence
            confidence = int(confidence * 0.8)
            reasons.append(f"({minutes_to_close}min early)")
        elif minutes_to_close <= 5:
            # Very close to close, boost confidence if setup is clean
            if side != "FLAT" and gamma_proximity > 0.7:
                confidence = min(100, int(confidence * 1.15))
                reasons.append("(final minutes)")
        
        return confidence, reasons, side
    
    def _determine_moc_direction(self, features: Dict[str, Any], imbalance_side: str,
                                 imbalance_total: float, gamma_proximity: float,
                                 indicative_price: float, price: float) -> str:
        """
        Determine MOC trade direction considering imbalance, gamma, and indicative price.
        """
        # Primary signal: imbalance direction
        if not imbalance_side:
            return "FLAT"
        
        base_direction = "LONG" if imbalance_side == 'BUY' else "SHORT"
        
        # Check for contradicting factors
        contradictions = 0
        confirmations = 0
        
        # 1. Gamma pin analysis
        if gamma_proximity > 0.7:
            pin_strike = self.find_gamma_pin(features)
            if pin_strike > 0:
                expected_move = pin_strike - price
                imbalance_move = indicative_price - price if indicative_price > 0 else 0
                
                if expected_move != 0 and imbalance_move != 0:
                    # Check if gamma and imbalance agree on direction
                    if (expected_move > 0 and imbalance_move > 0) or \
                       (expected_move < 0 and imbalance_move < 0):
                        confirmations += 1
                    else:
                        contradictions += 1
        
        # 2. Order book analysis
        obi = features.get('obi', 0.5)
        if (imbalance_side == 'BUY' and obi > 0.6) or \
           (imbalance_side == 'SELL' and obi < 0.4):
            confirmations += 1
        elif (imbalance_side == 'BUY' and obi < 0.4) or \
             (imbalance_side == 'SELL' and obi > 0.6):
            contradictions += 1
        
        # 3. Recent price action
        bars = features.get('bars', [])
        if len(bars) >= 5:
            recent_move = (bars[-1].get('close', 0) - bars[-5].get('close', 0)) / bars[-5].get('close', 1)
            if (imbalance_side == 'BUY' and recent_move > 0.002) or \
               (imbalance_side == 'SELL' and recent_move < -0.002):
                confirmations += 1
            elif (imbalance_side == 'BUY' and recent_move < -0.003) or \
                 (imbalance_side == 'SELL' and recent_move > 0.003):
                # Strong contradiction
                contradictions += 2
        
        # Final decision
        if contradictions > confirmations + 1:
            return "FLAT"  # Too many contradictions
        elif imbalance_total > 3e9 and contradictions == 0:
            # Very large imbalance with no contradictions
            return base_direction
        else:
            return base_direction
    
    def _calculate_gamma_concentration(self, gex_by_strike: List[Dict], 
                                      pin_strike: float, current_price: float) -> float:
        """
        Calculate gamma concentration around the pin strike.
        Returns 0-1 score for how concentrated gamma is near the pin.
        """
        if not gex_by_strike or pin_strike <= 0:
            return 0
        
        total_gamma = sum(abs(s.get('gex', 0)) for s in gex_by_strike)
        if total_gamma == 0:
            return 0
        
        # Calculate gamma within 1% of pin
        pin_range = pin_strike * 0.01
        near_pin_gamma = sum(abs(s.get('gex', 0)) for s in gex_by_strike 
                           if abs(s.get('strike', 0) - pin_strike) <= pin_range)
        
        concentration = near_pin_gamma / total_gamma
        return min(1.0, concentration)
    
    async def create_signal(self, symbol: str, strategy: str, confidence: int, 
                           reasons: List[str], features: Dict[str, Any], side: str) -> Dict[str, Any]:
        """
        Create a complete signal object with all trading parameters.
        """
        # Generate unique ID
        signal_id = str(uuid.uuid4())
        
        # Get current price
        entry_price = features.get('price', 0)
        
        # Calculate ATR-based stops and targets
        atr = features.get('atr', entry_price * 0.01)  # 1% default
        
        if side == "LONG":
            stop_loss = entry_price - (1.5 * atr)
            targets = [
                entry_price + (1.0 * atr),
                entry_price + (2.0 * atr),
                entry_price + (3.0 * atr)
            ]
        else:  # SHORT
            stop_loss = entry_price + (1.5 * atr)
            targets = [
                entry_price - (1.0 * atr),
                entry_price - (2.0 * atr),
                entry_price - (3.0 * atr)
            ]
        
        # Select contract (pass options chain for MOC)
        options_chain = features.get('options_chain') if strategy == 'moc' else None
        contract = self.select_contract(symbol, strategy, side, entry_price, options_chain)
        
        # Calculate position size (placeholder for now)
        position_size = self.calculate_position_size(confidence, strategy)
        
        # Build signal object
        signal = {
            'id': signal_id,
            'symbol': symbol,
            'strategy': strategy,
            'side': side,
            'confidence': confidence,
            'reasons': reasons,
            'entry': round(entry_price, 2),
            'stop': round(stop_loss, 2),
            'targets': [round(t, 2) for t in targets],
            'contract': contract,
            'rth': self.is_rth(datetime.now(self.eastern)),
            'ts': int(time.time() * 1000),
            'version': self.version
        }
        
        return signal
    
    def select_contract(self, symbol: str, strategy: str, side: str, spot: float, options_chain=None) -> Dict[str, Any]:
        """
        Select specific options contract for the signal.
        """
        contract = {
            'type': 'OPT',
            'right': 'C' if side == 'LONG' else 'P'
        }
        
        if strategy == '0dte':
            # First OTM strike expiring today
            contract['expiry'] = '0DTE'
            if side == 'LONG':
                contract['strike'] = round(spot + 1, 0)  # Next dollar strike up
            else:
                contract['strike'] = round(spot - 1, 0)  # Next dollar strike down
                
        elif strategy == '1dte':
            # 1% OTM expiring tomorrow
            contract['expiry'] = '1DTE'
            if side == 'LONG':
                contract['strike'] = round(spot * 1.01, 0)
            else:
                contract['strike'] = round(spot * 0.99, 0)
                
        elif strategy == '14dte':
            # 2% OTM or follow unusual activity
            contract['expiry'] = '14DTE'
            if side == 'LONG':
                contract['strike'] = round(spot * 1.02, 0)
            else:
                contract['strike'] = round(spot * 0.98, 0)
                
        elif strategy == 'moc':
            # MOC uses 0DTE options with proper delta calculation (bug fix #3)
            contract['expiry'] = '0DTE'
            strategy_config = self.strategies.get('moc', {})
            options_config = strategy_config.get('options', {})
            thresholds = strategy_config.get('thresholds', {})
            
            # Check if near pin for delta adjustment
            pin_dist = spot * 0.01  # Placeholder
            if pin_dist <= 0.003:
                target_delta = options_config.get('alt_delta_if_pin', 0.15)
            else:
                target_delta = options_config.get('target_delta', 0.25)
            
            # Find actual option contract with target delta if chain available
            if options_chain and isinstance(options_chain, list):
                selected_strike = None
                min_delta_diff = float('inf')
                best_contract_info = None
                
                for opt in options_chain:
                    if opt.get('expiry') != '0DTE':
                        continue
                    if (side == 'LONG' and opt.get('type') == 'CALL') or \
                       (side == 'SHORT' and opt.get('type') == 'PUT'):
                        opt_delta = abs(opt.get('delta', 0))
                        delta_diff = abs(opt_delta - target_delta)
                        
                        # Check liquidity requirements
                        if opt.get('open_interest', 0) >= thresholds.get('min_option_oi', 2000) and \
                           opt.get('spread_bps', 100) <= thresholds.get('max_spread_bps', 8):
                            if delta_diff < min_delta_diff:
                                min_delta_diff = delta_diff
                                selected_strike = opt.get('strike')
                                best_contract_info = {
                                    'oi': opt.get('open_interest'),
                                    'spread_bps': opt.get('spread_bps'),
                                    'actual_delta': opt_delta
                                }
                
                if selected_strike:
                    contract['strike'] = selected_strike
                    contract['liquidity'] = best_contract_info
                else:
                    # Fallback to approximation if no suitable contract found
                    if side == 'LONG':
                        contract['strike'] = round(spot * (1 + target_delta * 0.01), 0)
                    else:
                        contract['strike'] = round(spot * (1 - target_delta * 0.01), 0)
                    contract['liquidity'] = {'oi': 0, 'spread_bps': 999}  # Mark as poor liquidity
            else:
                # No chain available, use approximation
                if side == 'LONG':
                    contract['strike'] = round(spot * (1 + target_delta * 0.01), 0)
                else:
                    contract['strike'] = round(spot * (1 - target_delta * 0.01), 0)
                contract['liquidity'] = {'oi': 0, 'spread_bps': 999}
            
            contract['target_delta'] = target_delta
        
        return contract
    
    def calculate_position_size(self, confidence: int, strategy: str) -> float:
        """
        Calculate position size using confidence-based scaling.
        """
        # Base allocation (placeholder - would come from account size)
        base_allocation = 10000  # $10k base
        
        # Strategy-specific max positions
        max_positions = {
            '0dte': 0.05,
            '1dte': 0.07,
            '14dte': 0.10,
            'moc': 0.15
        }
        
        # Scale by confidence (60-100 -> 0.5-1.0)
        confidence_scale = 0.5 + 0.5 * max(0, (confidence - 60) / 40)
        
        # Calculate position
        max_pct = max_positions.get(strategy, 0.05)
        position = base_allocation * max_pct * confidence_scale
        
        return round(position, 2)
    
    async def calculate_atr(self, bars: List[Dict[str, Any]]) -> float:
        """
        Calculate Average True Range for stop/target placement.
        """
        if len(bars) < 2:
            return 1.0  # Default ATR
        
        true_ranges = []
        for i in range(1, min(len(bars), 15)):  # Use up to 14 periods
            high = bars[i].get('high', 0)
            low = bars[i].get('low', 0)
            prev_close = bars[i-1].get('close', 0)
            
            if high and low and prev_close:
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)
        
        if not true_ranges:
            return 1.0
        
        # Simple average (could use EMA for production)
        atr = sum(true_ranges) / len(true_ranges)
        
        # Apply minimum of 0.5
        return max(0.5, atr)
    
    def calculate_gamma_pin_proximity(self, features: Dict[str, Any]) -> float:
        """
        Calculate proximity to gamma pin (0-1 score).
        """
        gex_by_strike = features.get('gex_by_strike', [])
        if not gex_by_strike:
            return 0
        
        price = features.get('price', 0)
        if not price:
            return 0
        
        # Find strike with minimum absolute GEX (the pin)
        pin_strike = self.find_gamma_pin(features)
        if not pin_strike:
            return 0
        
        # Calculate distance
        pin_dist = abs(price - pin_strike) / price
        
        # Convert to proximity score (closer = higher)
        gamma_pin_distance = 0.005  # 0.5% threshold
        proximity = max(0, 1 - pin_dist / gamma_pin_distance)
        
        return proximity
    
    def find_gamma_pin(self, features: Dict[str, Any]) -> float:
        """
        Find the gamma pin strike.
        """
        gex_by_strike = features.get('gex_by_strike', [])
        if not gex_by_strike:
            return 0
        
        # Find strike with minimum absolute GEX
        min_gex = float('inf')
        pin_strike = 0
        
        for strike_data in gex_by_strike:
            strike = strike_data.get('strike', 0)
            gex = abs(strike_data.get('gex', float('inf')))
            if gex < min_gex:
                min_gex = gex
                pin_strike = strike
        
        return pin_strike
    
    def calculate_gamma_pull_direction(self, features: Dict[str, Any]) -> str:
        """
        Calculate direction of gamma pull.
        """
        gex_by_strike = features.get('gex_by_strike', [])
        if not gex_by_strike:
            return ''
        
        price = features.get('price', 0)
        if not price:
            return ''
        
        pin_strike = self.find_gamma_pin(features)
        if not pin_strike:
            return ''
        
        # Determine pull direction
        if pin_strike > price:
            return 'UP'
        elif pin_strike < price:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def check_freshness(self, features: Dict[str, Any]) -> bool:
        """
        Check if features are fresh enough.
        """
        age_s = features.get('age_s', 999)
        return age_s <= self.max_staleness_s
    
    def check_schema(self, features: Dict[str, Any]) -> bool:
        """
        Check if features have required fields and valid values.
        """
        # Check required fields
        required = ['vpin', 'obi', 'price', 'timestamp']
        for field in required:
            if field not in features:
                return False
        
        # Check numeric validity
        numeric_fields = ['vpin', 'obi', 'gex', 'dex', 'price']
        for field in numeric_fields:
            value = features.get(field, 0)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                return False
        
        # Clamp VPIN and OBI to [0,1]
        features['vpin'] = max(0, min(1, features.get('vpin', 0)))
        features['obi'] = max(0, min(1, features.get('obi', 0)))
        
        return True
    
    async def check_cooldown(self, symbol: str, side: str) -> bool:
        """
        Check if cooldown allows new signal.
        """
        cooldown_key = f'signals:cooldown:{symbol}:{side}'
        exists = await self.redis.exists(cooldown_key)
        return not exists
    
    async def check_idempotency(self, signal_id: str) -> bool:
        """
        Check if signal is duplicate.
        """
        emitted_key = f'signals:emitted:{signal_id}'
        # Use SET NX (set if not exists)
        result = await self.redis.set(emitted_key, '1', nx=True, ex=self.ttl_seconds)
        return result is not None
    
    def generate_signal_id(self, symbol: str, side: str, price: float) -> str:
        """
        Generate idempotent signal ID.
        """
        # Round time to 5-second bucket
        time_bucket = int(time.time() * 1000 / 5000) * 5000
        
        # Round price to penny
        price_bucket = round(price * 100) / 100
        
        # Create hash
        components = f"{symbol}:{side}:{time_bucket}:{self.version}:{price_bucket}"
        signal_id = hashlib.sha1(components.encode()).hexdigest()[:16]
        
        return signal_id
    
    def is_rth(self, current_time: datetime) -> bool:
        """
        Check if within regular trading hours.
        """
        # Check weekday
        if current_time.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check time (9:30 AM - 4:00 PM ET)
        market_open = datetime_time(9, 30)
        market_close = datetime_time(16, 0)
        current_time_only = current_time.time()
        
        return market_open <= current_time_only <= market_close


class SignalDistributor:
    """
    Distribute signals to different subscription tiers with appropriate delays.
    Manages signal queuing and delivery to various platforms.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize signal distributor with configuration.
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
    
    async def distribute_signal(self, signal: Dict[str, Any]):
        """
        Distribute signal to appropriate tiers with delays.
        """
        try:
            # Premium tier - immediate
            premium_signal = self.format_premium_signal(signal)
            await self.redis.lpush('distribution:premium:queue', json.dumps(premium_signal))
            
            # Basic tier - 60s delay
            basic_signal = self.format_basic_signal(signal)
            basic_delay = self.tiers.get('basic', {}).get('delay_seconds', 60)
            asyncio.create_task(self.delayed_publish('distribution:basic:queue', basic_signal, basic_delay))
            
            # Free tier - 300s delay
            free_signal = self.format_free_signal(signal)
            free_delay = self.tiers.get('free', {}).get('delay_seconds', 300)
            asyncio.create_task(self.delayed_publish('distribution:free:queue', free_signal, free_delay))
            
            self.logger.info(f"Distributed signal {signal['id']} for {signal['symbol']}")
            
        except Exception as e:
            self.logger.error(f"Error distributing signal: {e}")
    
    def format_premium_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format signal with full details for premium subscribers.
        """
        # Premium gets everything
        return signal
    
    def format_basic_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format signal with limited details for basic subscribers.
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
        """
        side = signal.get('side', '')
        sentiment = 'bullish' if side == 'LONG' else 'bearish' if side == 'SHORT' else 'neutral'
        
        return {
            'symbol': signal.get('symbol'),
            'sentiment': sentiment,
            'message': f"New {sentiment} signal on {signal.get('symbol')}. Upgrade for full details!",
            'ts': signal.get('ts')
        }
    
    async def delayed_publish(self, queue: str, data: Dict[str, Any], delay: int):
        """
        Publish data to queue after specified delay.
        """
        await asyncio.sleep(delay)
        await self.redis.lpush(queue, json.dumps(data))


class SignalValidator:
    """
    Validate signals before distribution to ensure quality.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize signal validator.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        
        # Signal configuration
        self.signal_config = config.get('modules', {}).get('signals', {})
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate signal meets quality standards.
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
        """
        # Check market hours
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)
        
        if current_time.weekday() >= 5:  # Weekend
            return False
        
        market_open = datetime_time(9, 30)
        market_close = datetime_time(16, 0)
        current_time_only = current_time.time()
        
        if not (market_open <= current_time_only <= market_close):
            # Check if extended hours are enabled
            if not self.config.get('market', {}).get('extended_hours', False):
                return False
        
        # TODO: Check for halts, spread, liquidity
        
        return True


class PerformanceTracker:
    """
    Track signal and strategy performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize performance tracker.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
    
    async def track_signal_performance(self, signal_id: str, outcome: Dict[str, Any]):
        """
        Track individual signal performance.
        """
        try:
            # Store performance data
            perf_key = f'performance:signal:{signal_id}'
            await self.redis.hset(perf_key, mapping=outcome)
            await self.redis.expire(perf_key, 86400 * 30)  # Keep for 30 days
            
            # Update strategy statistics
            strategy = outcome.get('strategy', '')
            if outcome.get('pnl', 0) > 0:
                await self.redis.incr(f'performance:strategy:{strategy}:wins')
            else:
                await self.redis.incr(f'performance:strategy:{strategy}:losses')
            
            self.logger.info(f"Tracked performance for signal {signal_id}")
            
        except Exception as e:
            self.logger.error(f"Error tracking performance: {e}")
    
    async def calculate_strategy_metrics(self, strategy: str) -> Dict[str, Any]:
        """
        Calculate performance metrics for a strategy.
        """
        try:
            # Get win/loss counts
            wins = int(await self.redis.get(f'performance:strategy:{strategy}:wins') or 0)
            losses = int(await self.redis.get(f'performance:strategy:{strategy}:losses') or 0)
            
            total = wins + losses
            if total == 0:
                return {'win_rate': 0, 'total_trades': 0}
            
            win_rate = wins / total
            
            # TODO: Calculate additional metrics
            # - Average win/loss
            # - Profit factor
            # - Sharpe ratio
            # - Maximum drawdown
            
            return {
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses,
                'total_trades': total
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'strategies': {}
        }
        
        strategies = ['0dte', '1dte', '14dte', 'moc']
        for strategy in strategies:
            report['strategies'][strategy] = await self.calculate_strategy_metrics(strategy)
        
        return report