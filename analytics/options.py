#!/usr/bin/env python3
"""
Options Analytics Module - Gamma Exposure and Greeks Analysis
Uses PROVIDED Greeks from Alpha Vantage - NO Black-Scholes calculation needed!
Institutional-grade GEX calculation for market maker positioning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger
import json
from enum import Enum

# Import from existing core modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core import CacheManager
from core.models import OptionsChain, OptionContract, OptionType


class GammaProfile(str, Enum):
    """Gamma exposure profile classification"""
    LONG_GAMMA = "LONG_GAMMA"      # MMs are long gamma (stabilizing)
    SHORT_GAMMA = "SHORT_GAMMA"    # MMs are short gamma (destabilizing)
    NEUTRAL = "NEUTRAL"            # Balanced gamma
    SQUEEZE = "SQUEEZE"            # Potential gamma squeeze setup
    PIN = "PIN"                    # Strong pin risk at strike


@dataclass
class GammaExposureMetrics:
    """Comprehensive gamma exposure metrics"""
    symbol: str
    spot_price: float
    timestamp: int

    # Core GEX metrics
    total_gamma_exposure: float      # Total market maker gamma in millions
    call_gamma_exposure: float       # Call GEX
    put_gamma_exposure: float        # Put GEX
    net_gamma_exposure: float        # Net GEX (calls - puts)

    # Key levels
    zero_gamma_level: float          # Price where gamma flips
    max_gamma_strike: float          # Strike with highest gamma
    pin_strike: float               # Most likely pin level

    # Gamma profile
    gamma_profile: GammaProfile
    gamma_tilt: str                 # CALL_HEAVY, PUT_HEAVY, BALANCED

    # Risk metrics
    gamma_risk_score: float         # 0-100 risk score
    squeeze_probability: float      # Probability of gamma squeeze
    pin_risk: float                # Strength of pin risk

    # Distribution
    gamma_by_strike: Dict[float, float]
    cumulative_gamma: Dict[float, float]

    # Hedging flows
    expected_hedging_direction: str  # BUY or SELL
    hedging_intensity: float         # Expected hedging strength

    # Greeks aggregates
    total_delta: float
    total_vega: float
    total_theta: float


@dataclass
class OptionsFlowMetrics:
    """Options flow and unusual activity metrics"""
    symbol: str
    timestamp: int

    # Flow metrics
    put_call_ratio: float
    put_call_volume_ratio: float

    # Unusual activity
    unusual_options_activity: List[Dict]
    sweep_orders: List[Dict]
    block_trades: List[Dict]

    # Sentiment indicators
    options_sentiment: str          # BULLISH, BEARISH, NEUTRAL
    smart_money_positioning: str    # Based on large trades

    # IV metrics
    iv_rank: float                  # 0-100 percentile
    iv_percentile: float
    term_structure_slope: float     # Front-month vs back-month IV


class GammaExposureCalculator:
    """
    Institutional-grade Gamma Exposure (GEX) Calculator
    Uses PROVIDED Greeks from Alpha Vantage - no computation needed!
    """

    def __init__(self, cache: CacheManager, config: Dict, av_client=None):
        """Initialize GEX calculator with AV client for historical data"""
        self.cache = cache
        self.config = config
        self.av_client = av_client  # Alpha Vantage client for historical IV
        
        # Load analytics configuration
        analytics_config = config.get('analytics', {})
        self.gex_config = analytics_config.get('gex', {})
        self.cache_limits = analytics_config.get('cache_limits', {})
        self.options_flow_config = analytics_config.get('options_flow', {})

        # Configuration (from config file)
        self.contract_multiplier = 100  # Standard equity option multiplier
        self.market_maker_hedge_ratio = self.gex_config.get('mm_hedge_ratio', 0.85)
        self.pin_range_pct = self.gex_config.get('pin_range_pct', 0.02)
        self.high_gamma_threshold = self.gex_config.get('high_gamma_threshold', 100)
        self.distance_decay_factor = self.gex_config.get('distance_decay_factor', 50)
        
        # Historical IV configuration
        self.historical_iv_days = self.gex_config.get('historical_iv_days', 252)
        self.historical_sample_frequency = self.gex_config.get('historical_sample_frequency', 7)

        # Historical tracking (config-driven limits)
        self.gex_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.cache_limits.get('gex_history', 500))
        )
        self.pin_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.cache_limits.get('pin_history', 100))
        )
        
        # Historical IV cache
        self.historical_iv_cache: Dict[str, Dict] = {}

        # Metrics
        self.calculation_metrics = {
            'calculations': 0,
            'avg_calc_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("Gamma Exposure calculator initialized - using PROVIDED Greeks!")

    async def calculate_gamma_exposure(self, symbol: str) -> GammaExposureMetrics:
        """
        Calculate comprehensive gamma exposure using PROVIDED Greeks
        No Black-Scholes calculation needed - Greeks come from Alpha Vantage!
        """
        import time
        start_time = time.time()

        try:
            # Get options chain from cache (populated by Alpha Vantage client)
            options_data = self.cache.get_options_chain(symbol)

            if not options_data:
                logger.warning(f"No options data for {symbol}")
                self.calculation_metrics['cache_misses'] += 1
                return self._empty_metrics(symbol)

            self.calculation_metrics['cache_hits'] += 1

            # Parse options chain
            spot_price = options_data.get('spot_price', 0)
            options = options_data.get('options', [])

            if not options:
                return self._empty_metrics(symbol, spot_price)

            # Calculate gamma exposure for each strike
            gamma_by_strike = defaultdict(float)
            call_gex_by_strike = defaultdict(float)
            put_gex_by_strike = defaultdict(float)

            total_call_gamma = 0
            total_put_gamma = 0
            total_delta = 0
            total_vega = 0
            total_theta = 0

            for option_data in options:
                # Parse option (Greeks are PROVIDED!)
                strike = option_data.get('strike', 0)
                opt_type = option_data.get('type', 'CALL')

                # PROVIDED Greeks - no calculation needed!
                gamma = option_data.get('gamma', 0)
                delta = option_data.get('delta', 0)
                vega = option_data.get('vega', 0)
                theta = option_data.get('theta', 0)

                # Open interest and volume
                open_interest = option_data.get('open_interest', 0)
                volume = option_data.get('volume', 0)

                # Calculate GEX for this contract
                # GEX = Spot * Gamma * Open Interest * Contract Multiplier * Spot / 100
                contract_gex = (spot_price * gamma * open_interest * self.contract_multiplier * spot_price) / 100

                # Convert to millions for readability
                contract_gex = contract_gex / 1_000_000

                # Market makers are SHORT options (negative gamma for calls, positive for puts)
                if opt_type == 'CALL':
                    # MMs short calls = negative gamma exposure
                    gamma_by_strike[strike] -= contract_gex
                    call_gex_by_strike[strike] -= contract_gex
                    total_call_gamma -= contract_gex
                else:  # PUT
                    # MMs short puts = positive gamma exposure
                    gamma_by_strike[strike] += contract_gex
                    put_gex_by_strike[strike] += contract_gex
                    total_put_gamma += contract_gex

                # Aggregate Greeks
                total_delta += delta * open_interest * self.contract_multiplier
                total_vega += vega * open_interest * self.contract_multiplier
                total_theta += theta * open_interest * self.contract_multiplier

            # Calculate total and net gamma exposure
            total_gamma = abs(total_call_gamma) + abs(total_put_gamma)
            net_gamma = total_call_gamma + total_put_gamma

            # Find key strikes
            max_gamma_strike = self._find_max_gamma_strike(gamma_by_strike)
            zero_gamma_level = self._calculate_zero_gamma_level(gamma_by_strike, spot_price)
            pin_strike = self._identify_pin_strike(gamma_by_strike, spot_price, options)

            # Determine gamma profile
            gamma_profile = self._classify_gamma_profile(net_gamma, total_gamma, spot_price, zero_gamma_level)

            # Calculate gamma tilt
            gamma_tilt = self._calculate_gamma_tilt(total_call_gamma, total_put_gamma)

            # Calculate risk metrics
            gamma_risk_score = self._calculate_gamma_risk_score(
                net_gamma, total_gamma, spot_price, zero_gamma_level
            )

            squeeze_probability = self._calculate_squeeze_probability(
                gamma_by_strike, spot_price, net_gamma
            )

            pin_risk = self._calculate_pin_risk(gamma_by_strike, pin_strike, spot_price)

            # Calculate expected hedging flows
            hedging_direction, hedging_intensity = self._calculate_hedging_flows(
                net_gamma, spot_price, zero_gamma_level
            )

            # Build cumulative gamma profile
            cumulative_gamma = self._build_cumulative_gamma(gamma_by_strike)

            # Create metrics object
            metrics = GammaExposureMetrics(
                symbol=symbol,
                spot_price=spot_price,
                timestamp=int(datetime.now().timestamp() * 1000),
                total_gamma_exposure=round(total_gamma, 2),
                call_gamma_exposure=round(total_call_gamma, 2),
                put_gamma_exposure=round(total_put_gamma, 2),
                net_gamma_exposure=round(net_gamma, 2),
                zero_gamma_level=round(zero_gamma_level, 2),
                max_gamma_strike=max_gamma_strike,
                pin_strike=pin_strike,
                gamma_profile=gamma_profile,
                gamma_tilt=gamma_tilt,
                gamma_risk_score=round(gamma_risk_score, 2),
                squeeze_probability=round(squeeze_probability, 3),
                pin_risk=round(pin_risk, 3),
                gamma_by_strike={k: round(v, 2) for k, v in gamma_by_strike.items()},
                cumulative_gamma=cumulative_gamma,
                expected_hedging_direction=hedging_direction,
                hedging_intensity=round(hedging_intensity, 2),
                total_delta=round(total_delta, 0),
                total_vega=round(total_vega, 0),
                total_theta=round(total_theta, 0)
            )

            # Track history
            self.gex_history[symbol].append({
                'timestamp': metrics.timestamp,
                'net_gamma': metrics.net_gamma_exposure,
                'spot_price': spot_price
            })

            self.pin_history[symbol].append({
                'timestamp': metrics.timestamp,
                'pin_strike': pin_strike,
                'pin_risk': pin_risk
            })

            # Cache results
            self.cache.set_metrics(symbol, {
                'gamma_exposure': metrics.net_gamma_exposure,
                'zero_gamma': metrics.zero_gamma_level,
                'pin_strike': metrics.pin_strike,
                'gamma_profile': metrics.gamma_profile.value
            })

            # Update calculation metrics
            calc_time = (time.time() - start_time) * 1000
            self.calculation_metrics['calculations'] += 1
            self.calculation_metrics['avg_calc_time_ms'] = (
                (self.calculation_metrics['avg_calc_time_ms'] * (self.calculation_metrics['calculations'] - 1) + calc_time)
                / self.calculation_metrics['calculations']
            )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating gamma exposure: {e}")
            return self._empty_metrics(symbol)

    def _find_max_gamma_strike(self, gamma_by_strike: Dict[float, float]) -> float:
        """Find strike with maximum absolute gamma"""
        if not gamma_by_strike:
            return 0.0

        max_strike = max(gamma_by_strike.items(), key=lambda x: abs(x[1]))
        return max_strike[0]

    def _calculate_zero_gamma_level(self, gamma_by_strike: Dict[float, float], spot_price: float) -> float:
        """
        Calculate the price level where net gamma crosses zero
        This is where market maker hedging flips from buying to selling
        """
        if not gamma_by_strike:
            return spot_price

        # Sort strikes
        sorted_strikes = sorted(gamma_by_strike.keys())

        # Find where gamma changes sign
        cumulative_gamma = 0
        prev_strike = sorted_strikes[0]

        for strike in sorted_strikes:
            cumulative_gamma += gamma_by_strike[strike]

            # Check if we crossed zero
            if cumulative_gamma > 0 and prev_strike < spot_price <= strike:
                # Interpolate the exact crossing point
                if gamma_by_strike[strike] != gamma_by_strike[prev_strike]:
                    ratio = abs(gamma_by_strike[prev_strike]) / (abs(gamma_by_strike[prev_strike]) + abs(gamma_by_strike[strike]))
                    zero_gamma = prev_strike + (strike - prev_strike) * ratio
                    return zero_gamma

            prev_strike = strike

        # If no crossing found, estimate based on weighted average
        total_gamma = sum(abs(g) for g in gamma_by_strike.values())
        if total_gamma > 0:
            weighted_strike = sum(strike * abs(gamma) for strike, gamma in gamma_by_strike.items()) / total_gamma
            return weighted_strike

        return spot_price

    def _identify_pin_strike(self, gamma_by_strike: Dict[float, float],
                            spot_price: float, options: List[Dict]) -> float:
        """
        Identify the most likely pin strike
        Usually the strike with highest open interest near spot
        """
        # Find strikes within configured range of spot
        pin_range = spot_price * self.pin_range_pct
        nearby_strikes = {
            strike: gamma for strike, gamma in gamma_by_strike.items()
            if abs(strike - spot_price) <= pin_range
        }

        if not nearby_strikes:
            # If no nearby strikes, use max gamma strike
            return self._find_max_gamma_strike(gamma_by_strike)

        # Weight by gamma magnitude and proximity to spot
        best_pin = spot_price
        best_score = 0

        for strike, gamma in nearby_strikes.items():
            # Score based on gamma magnitude and proximity
            distance_score = 1 - (abs(strike - spot_price) / pin_range)
            gamma_score = abs(gamma) / max(abs(g) for g in gamma_by_strike.values())

            total_score = gamma_score * 0.7 + distance_score * 0.3

            if total_score > best_score:
                best_score = total_score
                best_pin = strike

        return best_pin

    def _classify_gamma_profile(self, net_gamma: float, total_gamma: float,
                               spot_price: float, zero_gamma: float) -> GammaProfile:
        """Classify the gamma profile"""

        # Check for squeeze conditions
        if net_gamma < -total_gamma * 0.3 and abs(spot_price - zero_gamma) / spot_price < 0.01:
            return GammaProfile.SQUEEZE

        # Check for pin risk
        if total_gamma > self.high_gamma_threshold:  # High absolute gamma
            return GammaProfile.PIN

        # Check net positioning
        if net_gamma > total_gamma * 0.2:
            return GammaProfile.LONG_GAMMA
        elif net_gamma < -total_gamma * 0.2:
            return GammaProfile.SHORT_GAMMA
        else:
            return GammaProfile.NEUTRAL

    def _calculate_gamma_tilt(self, call_gamma: float, put_gamma: float) -> str:
        """Determine if gamma is call or put heavy"""
        total = abs(call_gamma) + abs(put_gamma)

        if total == 0:
            return "BALANCED"

        call_pct = abs(call_gamma) / total

        if call_pct > 0.6:
            return "CALL_HEAVY"
        elif call_pct < 0.4:
            return "PUT_HEAVY"
        else:
            return "BALANCED"

    def _calculate_gamma_risk_score(self, net_gamma: float, total_gamma: float,
                                   spot_price: float, zero_gamma: float) -> float:
        """
        Calculate gamma risk score (0-100)
        Higher = more risk of volatile moves
        """
        risk_score = 0

        # Factor 1: Absolute gamma magnitude (40%)
        gamma_magnitude_score = min(100, abs(net_gamma) / 10) * 0.4
        risk_score += gamma_magnitude_score

        # Factor 2: Distance from zero gamma (30%)
        distance_pct = abs(spot_price - zero_gamma) / spot_price
        distance_score = max(0, 100 - distance_pct * 1000) * 0.3
        risk_score += distance_score

        # Factor 3: Net/Total ratio (20%)
        if total_gamma > 0:
            imbalance = abs(net_gamma) / total_gamma
            imbalance_score = imbalance * 100 * 0.2
            risk_score += imbalance_score

        # Factor 4: Negative gamma (10%)
        if net_gamma < 0:
            risk_score += 10

        return min(100, risk_score)

    def _calculate_squeeze_probability(self, gamma_by_strike: Dict[float, float],
                                      spot_price: float, net_gamma: float) -> float:
        """
        Calculate probability of gamma squeeze
        Based on concentration and positioning
        """
        squeeze_prob = 0

        # Negative gamma increases squeeze probability
        if net_gamma < 0:
            squeeze_prob += 0.3

        # Check gamma concentration
        if gamma_by_strike:
            strikes_near_spot = [
                strike for strike in gamma_by_strike.keys()
                if 0.98 * spot_price <= strike <= 1.02 * spot_price
            ]

            if strikes_near_spot:
                near_gamma = sum(abs(gamma_by_strike[s]) for s in strikes_near_spot)
                total_gamma = sum(abs(g) for g in gamma_by_strike.values())

                concentration = near_gamma / total_gamma if total_gamma > 0 else 0

                # High concentration near spot = higher squeeze risk
                if concentration > 0.5:
                    squeeze_prob += 0.4
                elif concentration > 0.3:
                    squeeze_prob += 0.2

        # Check for call heavy positioning
        call_gamma = sum(g for g in gamma_by_strike.values() if g < 0)
        put_gamma = sum(g for g in gamma_by_strike.values() if g > 0)

        if abs(call_gamma) > abs(put_gamma) * 1.5:
            squeeze_prob += 0.3

        return min(1.0, squeeze_prob)

    def _calculate_pin_risk(self, gamma_by_strike: Dict[float, float],
                           pin_strike: float, spot_price: float) -> float:
        """
        Calculate strength of pin risk
        Higher value = stronger magnet effect
        """
        if not gamma_by_strike or pin_strike == 0:
            return 0.0

        # Get gamma at pin strike
        pin_gamma = gamma_by_strike.get(pin_strike, 0)

        # Calculate relative gamma concentration
        total_gamma = sum(abs(g) for g in gamma_by_strike.values())

        if total_gamma == 0:
            return 0.0

        concentration = abs(pin_gamma) / total_gamma

        # Adjust for distance from spot
        distance = abs(pin_strike - spot_price) / spot_price
        distance_factor = max(0, 1 - distance * self.distance_decay_factor)  # Decay over 2% distance

        pin_risk = concentration * distance_factor

        return min(1.0, pin_risk)

    def _calculate_hedging_flows(self, net_gamma: float, spot_price: float,
                                zero_gamma: float) -> Tuple[str, float]:
        """
        Calculate expected market maker hedging direction and intensity
        """
        # Determine direction based on gamma positioning
        if spot_price > zero_gamma:
            # Above zero gamma, negative gamma means MMs must buy on rallies
            if net_gamma < 0:
                direction = "BUY"
                intensity = abs(net_gamma)
            else:
                direction = "SELL"
                intensity = abs(net_gamma)
        else:
            # Below zero gamma, negative gamma means MMs must sell on declines
            if net_gamma < 0:
                direction = "SELL"
                intensity = abs(net_gamma)
            else:
                direction = "BUY"
                intensity = abs(net_gamma)

        # Normalize intensity (0-100)
        intensity = min(100, intensity)

        return direction, intensity

    def _build_cumulative_gamma(self, gamma_by_strike: Dict[float, float]) -> Dict[float, float]:
        """Build cumulative gamma profile"""
        if not gamma_by_strike:
            return {}

        sorted_strikes = sorted(gamma_by_strike.keys())
        cumulative = {}
        running_total = 0

        for strike in sorted_strikes:
            running_total += gamma_by_strike[strike]
            cumulative[strike] = round(running_total, 2)

        return cumulative

    def _empty_metrics(self, symbol: str, spot_price: float = 0) -> GammaExposureMetrics:
        """Return empty metrics when no data available"""
        return GammaExposureMetrics(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=int(datetime.now().timestamp() * 1000),
            total_gamma_exposure=0.0,
            call_gamma_exposure=0.0,
            put_gamma_exposure=0.0,
            net_gamma_exposure=0.0,
            zero_gamma_level=spot_price,
            max_gamma_strike=0.0,
            pin_strike=spot_price,
            gamma_profile=GammaProfile.NEUTRAL,
            gamma_tilt="BALANCED",
            gamma_risk_score=0.0,
            squeeze_probability=0.0,
            pin_risk=0.0,
            gamma_by_strike={},
            cumulative_gamma={},
            expected_hedging_direction="NEUTRAL",
            hedging_intensity=0.0,
            total_delta=0.0,
            total_vega=0.0,
            total_theta=0.0
        )

    async def calculate_options_flow(self, symbol: str) -> OptionsFlowMetrics:
        """
        Calculate options flow metrics and unusual activity
        """
        try:
            options_data = self.cache.get_options_chain(symbol)

            if not options_data:
                return self._empty_flow_metrics(symbol)

            options = options_data.get('options', [])

            # Calculate put/call ratios
            put_volume = sum(opt.get('volume', 0) for opt in options if opt.get('type') == 'PUT')
            call_volume = sum(opt.get('volume', 0) for opt in options if opt.get('type') == 'CALL')

            put_oi = sum(opt.get('open_interest', 0) for opt in options if opt.get('type') == 'PUT')
            call_oi = sum(opt.get('open_interest', 0) for opt in options if opt.get('type') == 'CALL')

            pc_ratio = put_oi / call_oi if call_oi > 0 else 0
            pc_volume_ratio = put_volume / call_volume if call_volume > 0 else 0

            # Detect unusual activity
            unusual_options = self._detect_unusual_options(options)
            sweep_orders = self._detect_sweep_orders(options)
            block_trades = self._detect_block_trades(options)

            # Determine sentiment
            sentiment = self._determine_options_sentiment(pc_ratio, pc_volume_ratio, unusual_options)

            # Analyze smart money
            smart_money = self._analyze_smart_money(block_trades, sweep_orders)

            # Calculate IV metrics
            iv_metrics = self._calculate_iv_metrics(options)

            return OptionsFlowMetrics(
                symbol=symbol,
                timestamp=int(datetime.now().timestamp() * 1000),
                put_call_ratio=round(pc_ratio, 3),
                put_call_volume_ratio=round(pc_volume_ratio, 3),
                unusual_options_activity=unusual_options,
                sweep_orders=sweep_orders,
                block_trades=block_trades,
                options_sentiment=sentiment,
                smart_money_positioning=smart_money,
                iv_rank=iv_metrics['iv_rank'],
                iv_percentile=iv_metrics['iv_percentile'],
                term_structure_slope=iv_metrics['term_slope']
            )

        except Exception as e:
            logger.error(f"Error calculating options flow: {e}")
            return self._empty_flow_metrics(symbol)

    def _detect_unusual_options(self, options: List[Dict]) -> List[Dict]:
        """Detect unusual options activity"""
        unusual = []

        for opt in options:
            volume = opt.get('volume', 0)
            open_interest = opt.get('open_interest', 0)

            # Unusual if volume > 2x open interest
            if open_interest > 0 and volume > open_interest * 2:
                unusual.append({
                    'strike': opt.get('strike'),
                    'type': opt.get('type'),
                    'expiration': opt.get('expiration'),
                    'volume': volume,
                    'open_interest': open_interest,
                    'volume_oi_ratio': round(volume / open_interest, 2),
                    'premium': volume * opt.get('ask', 0) * 100
                })

        # Sort by premium
        unusual.sort(key=lambda x: x['premium'], reverse=True)

        return unusual[:10]  # Top 10 unusual

    def _detect_sweep_orders(self, options: List[Dict]) -> List[Dict]:
        """Detect potential sweep orders"""
        sweeps = []

        # Get sweep threshold from config
        sweep_threshold = self.options_flow_config.get('volume_thresholds', {}).get('sweep', 500)
        
        for opt in options:
            # Sweep indicators: high volume, at ask, large premium
            volume = opt.get('volume', 0)
            if volume > sweep_threshold:  # Significant volume
                sweeps.append({
                    'strike': opt.get('strike'),
                    'type': opt.get('type'),
                    'volume': volume,
                    'urgency_score': min(1.0, volume / 1000)
                })

        return sweeps[:5]  # Top 5 sweeps

    def _detect_block_trades(self, options: List[Dict]) -> List[Dict]:
        """Detect large block trades"""
        blocks = []

        # Get block threshold from config
        block_threshold = self.options_flow_config.get('volume_thresholds', {}).get('block', 1000)
        
        for opt in options:
            volume = opt.get('volume', 0)
            # Block trade threshold
            if volume >= block_threshold:
                blocks.append({
                    'strike': opt.get('strike'),
                    'type': opt.get('type'),
                    'volume': volume,
                    'estimated_premium': volume * opt.get('ask', 0) * 100
                })

        return blocks

    def _determine_options_sentiment(self, pc_ratio: float, pc_volume: float,
                                    unusual: List[Dict]) -> str:
        """Determine overall options sentiment"""

        # Basic P/C analysis
        if pc_ratio > 1.2:
            base_sentiment = "BEARISH"
        elif pc_ratio < 0.8:
            base_sentiment = "BULLISH"
        else:
            base_sentiment = "NEUTRAL"

        # Adjust for unusual activity
        if unusual:
            call_unusual = sum(1 for u in unusual if u['type'] == 'CALL')
            put_unusual = sum(1 for u in unusual if u['type'] == 'PUT')

            if call_unusual > put_unusual * 2:
                return "BULLISH"
            elif put_unusual > call_unusual * 2:
                return "BEARISH"

        return base_sentiment

    def _analyze_smart_money(self, blocks: List[Dict], sweeps: List[Dict]) -> str:
        """Analyze smart money positioning"""

        if not blocks and not sweeps:
            return "NEUTRAL"

        # Analyze block trades
        call_blocks = sum(b['volume'] for b in blocks if b['type'] == 'CALL')
        put_blocks = sum(b['volume'] for b in blocks if b['type'] == 'PUT')

        if call_blocks > put_blocks * 1.5:
            return "BULLISH_POSITIONING"
        elif put_blocks > call_blocks * 1.5:
            return "BEARISH_POSITIONING"
        else:
            return "MIXED_POSITIONING"

    def _calculate_iv_metrics(self, options: List[Dict]) -> Dict[str, float]:
        """Calculate IV rank and percentile using historical data"""

        # Get current IV levels
        current_ivs = [opt.get('implied_volatility', 0) for opt in options if opt.get('implied_volatility', 0) > 0]

        if not current_ivs:
            return {'iv_rank': 50, 'iv_percentile': 50, 'term_slope': 0}

        avg_current_iv = np.mean(current_ivs)
        
        # Get symbol from first option
        symbol = options[0].get('symbol', 'SPY') if options else 'SPY'
        
        # Fetch historical IVs using existing AV client
        historical_ivs = self._fetch_historical_ivs(symbol)
        
        if historical_ivs:
            # Calculate IV rank: (Current - Min) / (Max - Min) * 100
            min_iv = min(historical_ivs)
            max_iv = max(historical_ivs)
            
            if max_iv > min_iv:
                iv_rank = (avg_current_iv - min_iv) / (max_iv - min_iv) * 100
            else:
                iv_rank = 50
            
            # Calculate IV percentile: % of days in past year IV was lower
            iv_percentile = (sum(1 for iv in historical_ivs if iv < avg_current_iv) / len(historical_ivs)) * 100
        else:
            # Fallback to simple estimate if no historical data
            iv_rank = min(100, avg_current_iv * 100)
            iv_percentile = iv_rank

        # Calculate term structure slope
        term_slope = self._calculate_term_structure_slope(options)

        return {
            'iv_rank': round(iv_rank, 1),
            'iv_percentile': round(iv_percentile, 1),
            'term_slope': round(term_slope, 3)
        }
    
    def _fetch_historical_ivs(self, symbol: str) -> List[float]:
        """Fetch historical IVs using existing Alpha Vantage client"""
        
        # Check cache first
        cache_key = f"historical_iv:{symbol}"
        if cache_key in self.historical_iv_cache:
            cached_data = self.historical_iv_cache[cache_key]
            # Check if cache is still valid (less than 1 day old)
            if cached_data.get('timestamp', 0) > datetime.now().timestamp() - 86400:
                return cached_data.get('ivs', [])
        
        if not self.av_client:
            logger.warning("No Alpha Vantage client available for historical IV")
            return []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.historical_iv_days)
            
            # Fetch historical options data using existing AV client method
            historical_data = self.av_client.get_historical_options(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if not historical_data:
                return []
            
            # Extract IVs at regular intervals
            ivs = []
            for data_point in historical_data:
                if 'implied_volatility' in data_point:
                    ivs.append(data_point['implied_volatility'])
            
            # Cache the results
            self.historical_iv_cache[cache_key] = {
                'ivs': ivs,
                'timestamp': datetime.now().timestamp()
            }
            
            return ivs
            
        except Exception as e:
            logger.error(f"Error fetching historical IVs: {e}")
            return []
    
    def _calculate_term_structure_slope(self, options: List[Dict]) -> float:
        """Calculate the slope of IV term structure"""
        
        # Group options by expiration
        expirations = {}
        for opt in options:
            exp = opt.get('expiration')
            if exp and opt.get('implied_volatility', 0) > 0:
                if exp not in expirations:
                    expirations[exp] = []
                expirations[exp].append(opt.get('implied_volatility'))
        
        if len(expirations) < 2:
            return 0.0
        
        # Sort by expiration date
        sorted_exps = sorted(expirations.items())
        
        # Get average IV for front month and back month
        front_iv = np.mean(sorted_exps[0][1])
        back_iv = np.mean(sorted_exps[-1][1])
        
        # Calculate slope (positive = backwardation, negative = contango)
        slope = (back_iv - front_iv) / front_iv if front_iv > 0 else 0
        
        return slope
    
    def calculate_cross_strike_correlation(self, symbol: str) -> Dict[str, Any]:
        """Calculate correlation between strikes for arbitrage detection"""
        
        try:
            options_data = self.cache.get_options_chain(symbol)
            
            if not options_data:
                return {'correlations': {}, 'arbitrage_opportunities': []}
            
            options = options_data.get('options', [])
            
            # Group by expiration and type
            strikes_by_exp = defaultdict(lambda: {'calls': {}, 'puts': {}})
            
            for opt in options:
                exp = opt.get('expiration')
                strike = opt.get('strike')
                opt_type = opt.get('type')
                price = opt.get('last', opt.get('mid', 0))
                
                if exp and strike and price > 0:
                    if opt_type == 'CALL':
                        strikes_by_exp[exp]['calls'][strike] = price
                    else:
                        strikes_by_exp[exp]['puts'][strike] = price
            
            correlations = {}
            arbitrage_opportunities = []
            
            # Check for arbitrage opportunities
            for exp, strikes in strikes_by_exp.items():
                calls = strikes['calls']
                puts = strikes['puts']
                
                # Check put-call parity violations
                for strike in set(calls.keys()) & set(puts.keys()):
                    call_price = calls[strike]
                    put_price = puts[strike]
                    spot = options_data.get('spot_price', 0)
                    
                    if spot > 0:
                        # Simplified put-call parity check (ignoring interest rates)
                        theoretical_diff = spot - strike
                        actual_diff = call_price - put_price
                        
                        violation = abs(actual_diff - theoretical_diff)
                        
                        if violation > spot * 0.01:  # More than 1% violation
                            arbitrage_opportunities.append({
                                'type': 'PUT_CALL_PARITY',
                                'strike': strike,
                                'expiration': exp,
                                'violation_amount': round(violation, 2),
                                'profit_potential': round(violation * self.contract_multiplier, 2)
                            })
                
                # Calculate strike correlations
                if len(calls) >= 3:
                    sorted_strikes = sorted(calls.keys())
                    prices = [calls[s] for s in sorted_strikes]
                    
                    # Simple correlation metric
                    if len(prices) > 1:
                        price_diffs = np.diff(prices)
                        smoothness = 1 - (np.std(price_diffs) / np.mean(np.abs(price_diffs)) if np.mean(np.abs(price_diffs)) > 0 else 0)
                        correlations[exp] = round(smoothness, 3)
            
            return {
                'correlations': correlations,
                'arbitrage_opportunities': arbitrage_opportunities[:5],  # Top 5 opportunities
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error calculating cross-strike correlation: {e}")
            return {'correlations': {}, 'arbitrage_opportunities': []}

    def _empty_flow_metrics(self, symbol: str) -> OptionsFlowMetrics:
        """Return empty flow metrics"""
        return OptionsFlowMetrics(
            symbol=symbol,
            timestamp=int(datetime.now().timestamp() * 1000),
            put_call_ratio=0.0,
            put_call_volume_ratio=0.0,
            unusual_options_activity=[],
            sweep_orders=[],
            block_trades=[],
            options_sentiment="NEUTRAL",
            smart_money_positioning="NEUTRAL",
            iv_rank=50.0,
            iv_percentile=50.0,
            term_structure_slope=0.0
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get calculator performance metrics"""
        return {
            **self.calculation_metrics,
            'symbols_tracked': len(self.gex_history),
            'cache_hit_rate': (
                self.calculation_metrics['cache_hits'] /
                (self.calculation_metrics['cache_hits'] + self.calculation_metrics['cache_misses'])
                if (self.calculation_metrics['cache_hits'] + self.calculation_metrics['cache_misses']) > 0
                else 0
            )
        }


# Module initialization
async def initialize_options_analytics(cache: CacheManager, config: Dict, av_client=None) -> Dict[str, Any]:
    """
    Initialize options analytics components with AV client for historical data
    """
    try:
        gex_calculator = GammaExposureCalculator(cache, config, av_client)

        return {
            'gex_calculator': gex_calculator,
            'status': 'initialized',
            'metrics': gex_calculator.get_metrics(),
            'message': 'Using PROVIDED Greeks from Alpha Vantage with historical IV support!'
        }

    except Exception as e:
        logger.error(f"Failed to initialize options analytics: {e}")
        raise
