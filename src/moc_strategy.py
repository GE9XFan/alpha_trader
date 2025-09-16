"""
MOC (Market-on-Close) Strategy Module

Implements market-on-close imbalance options plays. This module evaluates MOC
conditions, detects large imbalances, analyzes gamma magnets, and determines
optimal entry points for closing auction plays.

Redis Keys Used:
    Read:
        - metrics:{symbol}:features (market features)
        - options:chain:{symbol} (options chain data)
        - signals:last_contract:{symbol}:moc:{side}:{dte_band} (hysteresis)
    Write:
        - signals:last_contract:{symbol}:moc:{side}:{dte_band} (contract memory)

Author: AlphaTrader Pro
Version: 3.0.0
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, time as datetime_time
import pytz
import json


class MOCStrategy:
    """
    Market-on-close imbalance options strategy implementation.

    Detects and trades large MOC imbalances using options to capture
    the expected move into the close. Considers gamma effects, dealer
    positioning, and imbalance dynamics.
    """

    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize MOC strategy with configuration.

        Args:
            config: Strategy configuration
            redis_conn: Redis connection for data access
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Strategy configuration
        self.strategies = config.get('modules', {}).get('signals', {}).get('strategies', {})
        self.eastern = pytz.timezone('US/Eastern')

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

        # Convert to float to handle string values from config
        min_notional = float(thresholds.get('min_imbalance_notional', 2e9))
        min_ratio = float(thresholds.get('min_imbalance_ratio', 0.60))

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
                        gamma_points = int(float(weights.get('gamma_pull', 25)))
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
                        reasons.append("Dealer hedging active")

        # 3. Indicative price divergence (15 points max)
        if indicative_price > 0 and price > 0:
            divergence_bps = abs(indicative_price - price) / price * 10000
            min_divergence = float(thresholds.get('min_divergence_bps', 10))

            if divergence_bps >= min_divergence:
                # Scale by divergence magnitude
                divergence_score = min(1.0, (divergence_bps - min_divergence) / 40)  # 10-50bps range
                divergence_points = int(weights.get('indicative_divergence', 15) * divergence_score)
                confidence += divergence_points

                # Determine expected direction from divergence
                if indicative_price > price:
                    reasons.append(f"Indicative +{divergence_bps:.0f}bps")
                else:
                    reasons.append(f"Indicative -{divergence_bps:.0f}bps")

        # 4. Time window validation (10 points max)
        optimal_start = 15  # 15 minutes before close
        optimal_end = 2     # 2 minutes before close

        if optimal_end <= minutes_to_close <= optimal_start:
            # Perfect timing window
            time_score = 1.0
            confidence += int(float(weights.get('time_window', 10)))
            reasons.append(f"{minutes_to_close}min to close")
        elif minutes_to_close <= 20:
            # Acceptable but not optimal
            time_score = 0.5
            confidence += int(weights.get('time_window', 10) * 0.5)
            reasons.append(f"{minutes_to_close}min (early)")
        else:
            # Too early, penalize
            time_score = 0
            return 0, [], "FLAT"  # Don't trade too early

        # 5. Volume and liquidity check (5 points)
        volume = features.get('volume', 0)
        avg_volume = features.get('avg_volume_20d', 0)

        if avg_volume > 0 and volume > avg_volume * 1.5:
            confidence += int(float(weights.get('volume', 5)))
            reasons.append("High volume")

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