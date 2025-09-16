#!/usr/bin/env python3
"""
DTE Strategies - 0DTE, 1DTE, and 14DTE Option Strategies
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Handles all day-to-expiry based trading strategies.
"""

import json
import time
import redis.asyncio as aioredis
import numpy as np
import pytz
import hashlib
from datetime import datetime, time as datetime_time
from typing import Dict, List, Any, Tuple, Optional
import logging


class DTEStrategies:
    """
    Implements 0DTE, 1DTE, and 14DTE options strategies.
    All methods moved exactly from SignalGenerator class in signals.py.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize DTE strategies handler."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Get strategy configurations
        self.signal_config = config.get('modules', {}).get('signals', {})
        self.strategies = self.signal_config.get('strategies', {})

        # Eastern timezone for market hours
        self.eastern = pytz.timezone('US/Eastern')

        # Guardrail parameters
        self.ttl_seconds = self.signal_config.get('ttl_seconds', 300)

    async def evaluate(self, strategy: str, symbol: str, features: Dict[str, Any]) -> Tuple[int, List[str], str]:
        """Route to appropriate DTE strategy evaluator."""
        if strategy == '0dte':
            return self.evaluate_0dte_conditions(symbol, features)
        elif strategy == '1dte':
            return self.evaluate_1dte_conditions(symbol, features)
        elif strategy == '14dte':
            return self.evaluate_14dte_conditions(symbol, features)
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
        vpin_min = float(thresholds.get('vpin_min', 0.40))
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
        obi_min = float(thresholds.get('obi_min', 0.30))
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
        gamma_pin_distance = float(thresholds.get('gamma_pin_distance', 0.005))

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
                    points = int(float(weights.get('gamma_proximity', 30)))
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
                    points = int(float(weights.get('sweep', 15)))
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
            points = int(float(weights.get('volatility_regime', 20)))
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
                        points = int(float(weights.get('sweep', 30)))
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
                points = int(float(weights.get('hidden_orders', 20)))
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
                points = int(float(weights.get('dex', 10)))
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

    async def select_contract(self, symbol: str, strategy: str, side: str, spot: float, options_chain=None) -> Dict[str, Any]:
        """
        Select specific options contract for the signal with hysteresis to prevent strike bouncing.
        """
        contract = {
            'type': 'OPT',
            'right': 'C' if side == 'LONG' else 'P'
        }

        # Determine DTE band for hysteresis tracking
        dte_band = None

        if strategy == '0dte':
            # First OTM strike expiring today
            contract['expiry'] = '0DTE'
            dte_band = '0'
            if side == 'LONG':
                contract['strike'] = round(spot + 1, 0)  # Next dollar strike up
            else:
                contract['strike'] = round(spot - 1, 0)  # Next dollar strike down

        elif strategy == '1dte':
            # 1% OTM expiring tomorrow
            contract['expiry'] = '1DTE'
            dte_band = '1'
            if side == 'LONG':
                contract['strike'] = round(spot * 1.01, 0)
            else:
                contract['strike'] = round(spot * 0.99, 0)

        elif strategy == '14dte':
            # 2% OTM or follow unusual activity
            contract['expiry'] = '14DTE'
            dte_band = '14'
            if side == 'LONG':
                contract['strike'] = round(spot * 1.02, 0)
            else:
                contract['strike'] = round(spot * 0.98, 0)

        # Add DTE band to contract for tracking
        if dte_band:
            contract['dte_band'] = dte_band

        # Add hysteresis to prevent strike bouncing (keyed by DTE band)
        fp_key = f"signals:last_contract:{symbol}:{strategy}:{side}:{dte_band or 'NA'}"
        last_contract_str = await self.redis.get(fp_key)

        if last_contract_str:
            try:
                last_contract = json.loads(last_contract_str)
                last_strike = last_contract.get('strike', 0)

                # If we're only 1 strike away, require spot to move past midpoint before switching
                if abs(contract['strike'] - last_strike) == 1:
                    midpoint = (contract['strike'] + last_strike) / 2
                    if abs(spot - midpoint) < 0.3:  # Within 30 cents of midpoint, stick with previous
                        contract = last_contract
            except (json.JSONDecodeError, TypeError):
                pass

        # Remember this contract for next time (10 minute TTL)
        await self.redis.setex(fp_key, 600, json.dumps(contract))

        return contract

    def find_gamma_pin(self, features: Dict[str, Any]) -> float:
        """
        Find the gamma pin strike (strike with maximum absolute GEX).
        """
        gex_by_strike = features.get('gex_by_strike', [])
        if not gex_by_strike:
            return 0

        # Find strike with MAXIMUM absolute GEX (the pin)
        max_gex = 0
        pin_strike = 0

        for strike_data in gex_by_strike:
            strike = strike_data.get('strike', 0)
            gex = abs(strike_data.get('gex', 0))
            if gex > max_gex:
                max_gex = gex
                pin_strike = strike

        return pin_strike