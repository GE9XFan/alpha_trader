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
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('0dte', {})
        thresholds = strategy_config.get('thresholds', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # VPIN pressure (30 points max)
        vpin_min = thresholds.get('vpin_min', 0.40)
        if features.get('vpin', 0) >= vpin_min:
            points = weights.get('vpin', 30)
            confidence += points
            reasons.append("VPIN pressure")
        
        # OBI imbalance (25 points max)
        obi_min = thresholds.get('obi_min', 0.30)
        if features.get('obi', 0) >= obi_min:
            points = weights.get('obi', 25)
            confidence += points
            reasons.append("OBI imbalance")
        
        # Gamma pin proximity (30 points max)
        gamma_proximity = features.get('gamma_pin_proximity', 0)
        if gamma_proximity > 0:
            points = int(weights.get('gamma_proximity', 30) * gamma_proximity)
            confidence += points
            reasons.append("Near gamma pin")
        
        # Sweep detection (15 points max)
        if features.get('sweep', 0) >= 1:
            points = weights.get('sweep', 15)
            confidence += points
            reasons.append("Sweep detected")
        
        # Determine direction
        obi = features.get('obi', 0.5)
        price = features.get('price', 0)
        vwap = features.get('vwap')  # Don't fallback to price (bug fix #4)
        
        # Only use VWAP if available, otherwise use OBI alone
        if vwap:
            if obi > 0.5 and price >= vwap:
                side = "LONG"
            elif obi < 0.5 and price < vwap:
                side = "SHORT"
            else:
                # Tie-break with recent price movement
                bars = features.get('bars', [])
                if len(bars) >= 2:
                    last_return = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
                    side = "LONG" if last_return > 0 else "SHORT"
                else:
                    side = "FLAT"
        else:
            # Use OBI thresholds without VWAP
            if obi > 0.65:
                side = "LONG"
            elif obi < 0.35:
                side = "SHORT"
            else:
                # Tie-break with recent price movement
                bars = features.get('bars', [])
                if len(bars) >= 2:
                    last_return = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
                    side = "LONG" if last_return > 0 else "SHORT"
                else:
                    side = "FLAT"
        
        return confidence, reasons, side
    
    def evaluate_1dte_conditions(self, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Evaluate 1DTE strategy conditions (overnight positioning).
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('1dte', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # Volatility regime (20 points max)
        if features.get('regime', '') == 'HIGH':
            points = weights.get('volatility_regime', 20)
            confidence += points
            reasons.append("HIGH volatility regime")
        
        # OBI pressure (30 points max)
        if features.get('obi', 0) >= 0.20:
            points = int(weights.get('obi', 30) * min(features.get('obi', 0) / 0.5, 1))
            confidence += points
            reasons.append("OBI pressure")
        
        # GEX positioning (25 points max)
        if features.get('gex_z', 0) >= 0.5:
            points = int(weights.get('gex', 25) * min(features.get('gex_z', 0) / 2, 1))
            confidence += points
            reasons.append("GEX positioning")
        
        # VPIN flow (25 points max)
        if features.get('vpin', 0) >= 0.35:
            points = int(weights.get('vpin', 25) * min((features.get('vpin', 0) - 0.35) / 0.35, 1))
            confidence += points
            reasons.append("VPIN flow")
        
        # Direction based on 1-minute return
        bars = features.get('bars', [])
        if len(bars) >= 2:
            last_return = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
            
            # Check if VPIN strongly contradicts
            vpin = features.get('vpin', 0.5)
            if vpin > 0.7 and last_return < 0:
                side = "FLAT"  # Strong sell pressure contradicts bullish signal
            elif vpin < 0.3 and last_return > 0:
                side = "FLAT"  # Strong buy pressure contradicts bearish signal
            else:
                side = "LONG" if last_return > 0 else "SHORT"
        else:
            side = "FLAT"
        
        return confidence, reasons, side
    
    def evaluate_14dte_conditions(self, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Evaluate 14DTE strategy conditions (swing trades on unusual activity).
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('14dte', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # Unusual options activity (40 points max)
        unusual = features.get('unusual_activity', 0)
        if unusual >= 0.6:
            # Scale from 0.6->1.0 to 24->40 points
            points = int(24 + (unusual - 0.6) * 40)
            points = min(points, weights.get('unusual_options', 40))
            confidence += points
            reasons.append("Unusual options activity")
        
        # Sweep detection (30 points max)
        if features.get('sweep', 0) >= 1:
            points = weights.get('sweep', 30)
            confidence += points
            reasons.append("Sweep detected")
        
        # Hidden orders (20 points max)
        hidden = features.get('hidden_orders', 0)
        if hidden >= 0.5:
            points = int(weights.get('hidden_orders', 20) * min(hidden / 0.8, 1))
            confidence += points
            reasons.append("Hidden order flow")
        
        # DEX flow (10 points max)
        if features.get('dex_z', 0) >= 0.5:
            points = int(weights.get('dex', 10) * min(features.get('dex_z', 0) / 2, 1))
            confidence += points
            reasons.append("DEX flow")
        
        # Direction based on majority vote
        votes = []
        
        # Unusual activity direction (inferred from call/put ratio)
        if unusual > 0.6:
            # For now, use DEX sign as proxy
            votes.append("LONG" if features.get('dex', 0) > 0 else "SHORT")
        
        # Sweep direction (use recent price movement)
        if features.get('sweep', 0) >= 1:
            bars = features.get('bars', [])
            if len(bars) >= 2:
                last_return = (bars[-1].get('close', 0) - bars[-2].get('close', 0)) / bars[-2].get('close', 1)
                votes.append("LONG" if last_return > 0 else "SHORT")
        
        # DEX sign
        votes.append("LONG" if features.get('dex', 0) > 0 else "SHORT")
        
        # Determine side by majority
        if len(votes) == 0:
            side = "FLAT"
        else:
            long_votes = sum(1 for v in votes if v == "LONG")
            short_votes = sum(1 for v in votes if v == "SHORT")
            if long_votes > short_votes:
                side = "LONG"
            elif short_votes > long_votes:
                side = "SHORT"
            else:
                side = "FLAT"  # Tie
        
        return confidence, reasons, side
    
    def evaluate_moc_conditions(self, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """
        Evaluate MOC strategy conditions (market-on-close imbalance options play).
        """
        confidence = 0
        reasons = []
        
        strategy_config = self.strategies.get('moc', {})
        thresholds = strategy_config.get('thresholds', {})
        weights = strategy_config.get('confidence_weights', {})
        
        # Hard gates for imbalance
        imbalance_total = features.get('imbalance_total', 0)
        imbalance_ratio = features.get('imbalance_ratio', 0)
        min_notional = thresholds.get('min_imbalance_notional', 2e9)
        min_ratio = thresholds.get('min_imbalance_ratio', 0.60)
        
        if imbalance_total < min_notional or imbalance_ratio < min_ratio:
            return 0, [], "FLAT"
        
        # Imbalance strength (45 points max)
        base_points = min(30, int(30 * (imbalance_total / 2e9)))
        ratio_points = int(15 * max(0, (imbalance_ratio - 0.60) / 0.40))
        imbalance_points = min(weights.get('imbalance_strength', 45), base_points + ratio_points)
        confidence += imbalance_points
        reasons.append(f"${imbalance_total/1e9:.1f}B {features.get('imbalance_side', '')} imbalance")
        reasons.append(f"ratio {imbalance_ratio:.2f}")
        
        # Gamma pull (25 points max)
        gamma_pull_dir = features.get('gamma_pull_dir', '')
        pin_dist = features.get('gamma_pin_proximity', 1.0) * thresholds.get('gamma_pin_distance', 0.005)
        
        if gamma_pull_dir and pin_dist <= 0.005:
            gamma_points = int(weights.get('gamma_pull', 25) * (1 - pin_dist / 0.005))
            confidence += gamma_points
            price = features.get('price', 0)
            pin_strike = self.find_gamma_pin(features)
            reasons.append(f"gamma pull to {pin_strike:.0f}")
        
        # OBI (20 points max)
        obi = features.get('obi', 0)
        if obi > 0:
            obi_points = int(weights.get('obi', 20) * min(obi, 1.0))
            confidence += obi_points
            reasons.append(f"OBI skew {obi:.2f}")
        
        # Friday factor (10 points max)
        if datetime.now(self.eastern).weekday() == 4:  # Friday
            confidence += weights.get('friday_factor', 10)
            reasons.append("Friday uplift")
        
        # Determine direction based on imbalance
        imbalance_side = features.get('imbalance_side', '')
        if imbalance_side == 'BUY':
            side = "LONG"  # Buy imbalance -> calls
        elif imbalance_side == 'SELL':
            side = "SHORT"  # Sell imbalance -> puts
        else:
            side = "FLAT"
        
        # Check for gamma contradiction
        if gamma_pull_dir and pin_dist <= 0.003:
            if (gamma_pull_dir == 'UP' and side == 'SHORT') or (gamma_pull_dir == 'DOWN' and side == 'LONG'):
                # Strong gamma contradiction
                min_conf_override = thresholds.get('min_confidence', 75) + 10
                if confidence < min_conf_override:
                    side = "FLAT"
        
        return confidence, reasons, side
    
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