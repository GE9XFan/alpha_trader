#!/usr/bin/env python3
"""
Technical Indicators and Order Book Analytics Module
Institutional-grade order book imbalance and microstructure indicators
Integrates with existing cache and IBKR Level 2 data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
from loguru import logger
import json
from enum import Enum

# Import from existing core modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core import CacheManager
from core.models import OrderBook, OrderBookLevel, Bar


class BookPressure(str, Enum):
    """Order book pressure classification"""
    HEAVY_BUYING = "HEAVY_BUYING"
    MODERATE_BUYING = "MODERATE_BUYING"
    BALANCED = "BALANCED"
    MODERATE_SELLING = "MODERATE_SELLING"
    HEAVY_SELLING = "HEAVY_SELLING"


@dataclass
class OrderBookMetrics:
    """Comprehensive order book metrics"""
    symbol: str
    timestamp: int

    # Basic imbalance
    volume_imbalance: float  # -1 to 1
    value_imbalance: float   # -1 to 1

    # Depth metrics
    bid_depth: float
    ask_depth: float
    depth_ratio: float

    # Weighted metrics
    weighted_mid_price: float
    micro_price: float  # Depth-weighted price

    # Pressure indicators
    book_pressure: BookPressure
    pressure_score: float  # -100 to 100

    # Advanced metrics
    spread_bps: float  # Spread in basis points
    book_skew: float
    liquidity_score: float

    # Level-specific metrics
    top_level_ratio: float
    deep_book_ratio: float

    # Market maker metrics
    mm_participation: Dict[str, float]
    spoofing_probability: float


class OrderBookImbalance:
    """
    Advanced Order Book Imbalance Calculator
    Used by institutional traders to predict short-term price movements
    """

    def __init__(self, cache: CacheManager, config: Dict):
        """Initialize OBI calculator with cache integration"""
        self.cache = cache
        self.config = config

        # Load analytics configuration
        analytics_config = config.get('analytics', {})
        self.obi_config = analytics_config.get('obi', {})
        self.market_patterns = analytics_config.get('market_patterns', {})
        self.cache_limits = analytics_config.get('cache_limits', {})
        self.volatility_config = analytics_config.get('volatility', {})
        self.spoofing_config = analytics_config.get('spoofing', {})

        # Configuration
        self.depth_levels = config.get('ibkr', {}).get('market_data', {}).get('level2_depth', 10)
        
        # VAMP configuration
        self.enable_vamp = self.obi_config.get('enable_vamp', True)
        self.vamp_levels = self.obi_config.get('vamp_levels', 5)

        # Historical tracking for pattern detection (config-driven limits)
        self.imbalance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.obi_config.get('imbalance_history_size', 1000))
        )
        self.pressure_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.obi_config.get('pressure_history_size', 100))
        )

        # Market maker tracking (from Level 2 data)
        self.market_maker_activity: Dict[str, Dict] = defaultdict(dict)

        # Metrics
        self.calculation_metrics = {
            'calculations': 0,
            'avg_calc_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info(f"Order Book Imbalance calculator initialized with {self.depth_levels} levels")

    async def calculate_order_book_imbalance(self, symbol: str, order_book: Optional[Dict] = None) -> OrderBookMetrics:
        """
        Calculate comprehensive order book imbalance metrics
        
        Args:
            symbol: Trading symbol
            order_book: Optional order book dictionary with 'bids' and 'asks'. If not provided, fetches from cache
            
        Returns:
            OrderBookMetrics with normalized imbalance score and predictive metrics
        """
        import time
        start_time = time.time()

        try:
            # Use provided order book or fetch from cache
            if order_book is not None:
                # Order book provided directly - institutional-grade flexibility
                order_book_data = order_book
                logger.debug(f"Using provided order book for OBI calculation")
            else:
                # Fetch from cache (backward compatibility)
                order_book_data = self.cache.get_order_book(symbol)

                if not order_book_data:
                    logger.warning(f"No order book data for {symbol}")
                    self.calculation_metrics['cache_misses'] += 1
                    return self._empty_metrics(symbol)

                self.calculation_metrics['cache_hits'] += 1

            # Extract bid/ask levels
            bids = order_book_data.get('bids', [])
            asks = order_book_data.get('asks', [])

            if not bids or not asks:
                return self._empty_metrics(symbol)

            # Calculate volume imbalance
            volume_imbalance = self._calculate_volume_imbalance(bids, asks)

            # Calculate value imbalance (size * price weighted)
            value_imbalance = self._calculate_value_imbalance(bids, asks)

            # Calculate depth metrics
            bid_depth = sum(level['size'] for level in bids)
            ask_depth = sum(level['size'] for level in asks)
            depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 0

            # Calculate weighted mid price
            weighted_mid = self._calculate_weighted_mid_price(bids, asks)

            # Calculate micro price (depth-weighted)
            micro_price = self._calculate_micro_price(bids, asks)

            # Calculate spread
            spread = asks[0]['price'] - bids[0]['price']
            mid_price = (asks[0]['price'] + bids[0]['price']) / 2
            spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0

            # Calculate book skew
            book_skew = self._calculate_book_skew(bids, asks)

            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(bids, asks, spread)

            # Level-specific ratios
            top_level_ratio = bids[0]['size'] / asks[0]['size'] if asks[0]['size'] > 0 else 0

            # Deep book ratio (levels 5-10 vs 1-4)
            deep_book_ratio = 0
            if len(bids) >= 5 and len(asks) >= 5:
                deep_bid = sum(b['size'] for b in bids[4:])
                deep_ask = sum(a['size'] for a in asks[4:])
                shallow_bid = sum(b['size'] for b in bids[:4])
                shallow_ask = sum(a['size'] for a in asks[:4])

                if (shallow_bid + shallow_ask) > 0:
                    deep_book_ratio = (deep_bid + deep_ask) / (shallow_bid + shallow_ask)

            # Determine book pressure
            pressure_score = self._calculate_pressure_score(
                volume_imbalance, value_imbalance, book_skew, top_level_ratio
            )
            book_pressure = self._classify_pressure(pressure_score)

            # Detect market maker activity
            mm_participation = self._detect_market_maker_activity(bids, asks)

            # Calculate spoofing probability
            spoofing_prob = self._calculate_spoofing_probability(
                bids, asks, self.imbalance_history[symbol]
            )

            # Create metrics object
            metrics = OrderBookMetrics(
                symbol=symbol,
                timestamp=int(datetime.now().timestamp() * 1000),
                volume_imbalance=round(volume_imbalance, 4),
                value_imbalance=round(value_imbalance, 4),
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                depth_ratio=round(depth_ratio, 2),
                weighted_mid_price=round(weighted_mid, 2),
                micro_price=round(micro_price, 2),
                book_pressure=book_pressure,
                pressure_score=round(pressure_score, 2),
                spread_bps=round(spread_bps, 2),
                book_skew=round(book_skew, 4),
                liquidity_score=round(liquidity_score, 2),
                top_level_ratio=round(top_level_ratio, 2),
                deep_book_ratio=round(deep_book_ratio, 2),
                mm_participation=mm_participation,
                spoofing_probability=round(spoofing_prob, 3)
            )

            # Track history
            self.imbalance_history[symbol].append({
                'timestamp': metrics.timestamp,
                'volume_imbalance': metrics.volume_imbalance,
                'pressure_score': metrics.pressure_score
            })

            self.pressure_history[symbol].append(metrics.pressure_score)

            # Cache the result
            self.cache.set_metrics(symbol, {
                'order_book_imbalance': metrics.volume_imbalance,
                'book_pressure': metrics.book_pressure.value,
                'pressure_score': metrics.pressure_score,
                'micro_price': metrics.micro_price,
                'liquidity_score': metrics.liquidity_score
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
            logger.error(f"Error calculating order book imbalance: {e}")
            return self._empty_metrics(symbol)

    def _calculate_volume_imbalance(self, bids: List[Dict], asks: List[Dict]) -> float:
        """
        Calculate volume imbalance (-1 to 1)
        Positive = more bid volume (buying pressure)
        Negative = more ask volume (selling pressure)
        """
        bid_volume = sum(level['size'] for level in bids)
        ask_volume = sum(level['size'] for level in asks)

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / total_volume

    def _calculate_value_imbalance(self, bids: List[Dict], asks: List[Dict]) -> float:
        """
        Calculate value-weighted imbalance
        Accounts for price levels, not just size
        """
        bid_value = sum(level['size'] * level['price'] for level in bids)
        ask_value = sum(level['size'] * level['price'] for level in asks)

        total_value = bid_value + ask_value
        if total_value == 0:
            return 0.0

        return (bid_value - ask_value) / total_value

    def _calculate_weighted_mid_price(self, bids: List[Dict], asks: List[Dict]) -> float:
        """
        Calculate size-weighted mid price
        Gives more weight to side with more liquidity
        """
        if not bids or not asks:
            return 0.0

        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        bid_size = bids[0]['size']
        ask_size = asks[0]['size']

        total_size = bid_size + ask_size
        if total_size == 0:
            return (best_bid + best_ask) / 2

        return (best_bid * ask_size + best_ask * bid_size) / total_size

    def _calculate_micro_price(self, bids: List[Dict], asks: List[Dict]) -> float:
        """
        Calculate micro price using multiple levels
        Better predictor of short-term price movement
        Uses VAMP (Volume Adjusted Mid Price) when enabled
        """
        if not bids or not asks:
            return 0.0

        # Use VAMP if enabled
        if self.enable_vamp:
            return self._calculate_vamp(bids, asks)

        # Standard micro price calculation
        levels_to_use = min(3, len(bids), len(asks))

        weighted_bid = 0
        weighted_ask = 0
        total_bid_size = 0
        total_ask_size = 0

        for i in range(levels_to_use):
            # Inverse distance weighting (closer levels matter more)
            weight = 1 / (i + 1)

            weighted_bid += bids[i]['price'] * bids[i]['size'] * weight
            total_bid_size += bids[i]['size'] * weight

            weighted_ask += asks[i]['price'] * asks[i]['size'] * weight
            total_ask_size += asks[i]['size'] * weight

        if total_bid_size == 0 or total_ask_size == 0:
            return (bids[0]['price'] + asks[0]['price']) / 2

        avg_bid = weighted_bid / total_bid_size
        avg_ask = weighted_ask / total_ask_size

        # Weight by relative sizes
        total_size = total_bid_size + total_ask_size
        return (avg_bid * total_ask_size + avg_ask * total_bid_size) / total_size

    def _calculate_vamp(self, bids: List[Dict], asks: List[Dict]) -> float:
        """
        Calculate Volume Adjusted Mid Price (VAMP)
        
        VAMP adjusts the mid-price based on order book imbalance using
        cumulative volumes at each price level. This provides a better
        estimate of fair value when liquidity is imbalanced.
        
        Formula:
        VAMP = Σ(P_bid[i] × Q_ask_cum[i] + P_ask[i] × Q_bid_cum[i]) / Σ(Q_bid_cum[i] + Q_ask_cum[i])
        
        Reference: High-frequency trading literature on microstructure pricing
        """
        if not bids or not asks:
            return 0.0
        
        import numpy as np
        
        # Use configured number of levels or available depth
        levels = min(self.vamp_levels, len(bids), len(asks))
        
        if levels == 0:
            return 0.0
        
        # Extract prices and sizes
        bid_prices = np.array([float(bids[i]['price']) for i in range(levels)])
        ask_prices = np.array([float(asks[i]['price']) for i in range(levels)])
        bid_sizes = np.array([float(bids[i]['size']) for i in range(levels)])
        ask_sizes = np.array([float(asks[i]['size']) for i in range(levels)])
        
        # VAMP calculation: cross-multiply prices with opposite side volumes
        # This weights each price by the liquidity available on the opposite side
        # Formula: VAMP = Σ(P_bid[i] × Q_ask[i] + P_ask[i] × Q_bid[i]) / Σ(Q_bid[i] + Q_ask[i])
        vamp_numerator = np.sum(bid_prices * ask_sizes + ask_prices * bid_sizes)
        vamp_denominator = np.sum(bid_sizes + ask_sizes)
        
        if vamp_denominator > 0:
            vamp = vamp_numerator / vamp_denominator
            
            # Sanity check: VAMP should be between best bid and best ask
            if vamp < bid_prices[0] or vamp > ask_prices[0]:
                logger.warning(f"VAMP {vamp:.4f} outside bid-ask spread [{bid_prices[0]:.4f}, {ask_prices[0]:.4f}]")
                # Fall back to simple mid
                return (bid_prices[0] + ask_prices[0]) / 2
            
            return vamp
        else:
            # No volume - use simple mid
            return (bid_prices[0] + ask_prices[0]) / 2

    def _calculate_book_skew(self, bids: List[Dict], asks: List[Dict]) -> float:
        """
        Calculate order book skew
        Measures asymmetry in book depth distribution
        """
        # Calculate cumulative depth at each level
        bid_cumulative = []
        ask_cumulative = []

        cum_bid = 0
        cum_ask = 0

        max_levels = max(len(bids), len(asks))

        for i in range(max_levels):
            if i < len(bids):
                cum_bid += bids[i]['size']
            if i < len(asks):
                cum_ask += asks[i]['size']

            bid_cumulative.append(cum_bid)
            ask_cumulative.append(cum_ask)

        # Calculate skew as difference in cumulative distributions
        total_bid = bid_cumulative[-1]
        total_ask = ask_cumulative[-1]

        if total_bid == 0 or total_ask == 0:
            return 0.0

        # Normalize and calculate area between curves
        skew = 0
        for i in range(max_levels):
            bid_pct = bid_cumulative[i] / total_bid
            ask_pct = ask_cumulative[i] / total_ask
            skew += (bid_pct - ask_pct) / max_levels

        return skew

    def _calculate_liquidity_score(self, bids: List[Dict], asks: List[Dict], spread: float) -> float:
        """
        Calculate overall liquidity score (0-100)
        Higher score = better liquidity
        """
        if not bids or not asks:
            return 0.0

        # Factors for liquidity score
        scores = []

        # 1. Tight spread (40% weight)
        mid_price = (bids[0]['price'] + asks[0]['price']) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 10000
        spread_score = max(0, 100 - spread_bps)  # 0 bps = 100 score, 100 bps = 0 score
        scores.append(spread_score * 0.4)

        # 2. Depth (30% weight)
        total_depth = sum(b['size'] for b in bids) + sum(a['size'] for a in asks)
        normalization_factor = self.obi_config.get('normalization_factor', 100)
        depth_score = min(100, total_depth / normalization_factor)  # Normalize to 0-100
        scores.append(depth_score * 0.3)

        # 3. Balance (20% weight)
        bid_depth = sum(b['size'] for b in bids)
        ask_depth = sum(a['size'] for a in asks)
        balance = 1 - abs(bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        balance_score = balance * 100
        scores.append(balance_score * 0.2)

        # 4. Level consistency (10% weight)
        # Check if sizes decrease gradually (natural book) vs cliff (potential spoofing)
        consistency_score = self._calculate_level_consistency(bids, asks) * 100
        scores.append(consistency_score * 0.1)

        return sum(scores)

    def _calculate_level_consistency(self, bids: List[Dict], asks: List[Dict]) -> float:
        """
        Check if order book levels decrease naturally
        Returns 0-1 (1 = very consistent/natural)
        """
        consistency_scores = []

        # Check bid side
        if len(bids) > 1:
            bid_changes = []
            for i in range(1, min(5, len(bids))):
                if bids[i-1]['size'] > 0:
                    change = abs(bids[i]['size'] - bids[i-1]['size']) / bids[i-1]['size']
                    bid_changes.append(min(change, 1))

            if bid_changes:
                # Natural books have gradual changes
                bid_consistency = 1 - (sum(bid_changes) / len(bid_changes))
                consistency_scores.append(bid_consistency)

        # Check ask side
        if len(asks) > 1:
            ask_changes = []
            for i in range(1, min(5, len(asks))):
                if asks[i-1]['size'] > 0:
                    change = abs(asks[i]['size'] - asks[i-1]['size']) / asks[i-1]['size']
                    ask_changes.append(min(change, 1))

            if ask_changes:
                ask_consistency = 1 - (sum(ask_changes) / len(ask_changes))
                consistency_scores.append(ask_consistency)

        return np.mean(consistency_scores) if consistency_scores else 0.5

    def _calculate_pressure_score(self, volume_imb: float, value_imb: float,
                                 skew: float, top_ratio: float) -> float:
        """
        Calculate composite pressure score (-100 to 100)
        Positive = buying pressure, Negative = selling pressure
        """
        # Weight different factors
        score = (
            volume_imb * 40 +  # 40% weight
            value_imb * 30 +   # 30% weight
            skew * 20 +        # 20% weight
            (top_ratio - 1) * 10  # 10% weight (normalized)
        )

        # Cap at -100 to 100
        return max(-100, min(100, score))

    def _classify_pressure(self, pressure_score: float) -> BookPressure:
        """Classify book pressure based on score"""
        if pressure_score > 50:
            return BookPressure.HEAVY_BUYING
        elif pressure_score > 20:
            return BookPressure.MODERATE_BUYING
        elif pressure_score > -20:
            return BookPressure.BALANCED
        elif pressure_score > -50:
            return BookPressure.MODERATE_SELLING
        else:
            return BookPressure.HEAVY_SELLING

    def _detect_market_maker_activity(self, bids: List[Dict], asks: List[Dict]) -> Dict[str, float]:
        """
        Detect market maker participation from order sizes and patterns
        """
        mm_patterns = {
            'IBEOS': 0.0,    # IB Smart Router
            'CDRG': 0.0,     # Citadel
            'OVERNIGHT': 0.0  # Extended hours specialist
        }

        # Analyze order sizes for MM patterns
        all_sizes = [b['size'] for b in bids] + [a['size'] for a in asks]

        # Use config-driven patterns
        retail_sizes = self.market_patterns.get('retail_sizes', [100, 200, 300, 500])
        institutional_threshold = self.market_patterns.get('institutional_threshold', 1000)
        odd_lot_threshold = self.market_patterns.get('odd_lot_threshold', 100)
        
        for size in all_sizes:
            # Retail flow patterns
            if size in retail_sizes:
                mm_patterns['IBEOS'] += 1
            # Institutional patterns
            elif size >= institutional_threshold:
                mm_patterns['CDRG'] += 1
            # Odd lots often from overnight
            elif size < odd_lot_threshold:
                mm_patterns['OVERNIGHT'] += 1

        # Normalize to percentages
        total = sum(mm_patterns.values())
        if total > 0:
            for mm in mm_patterns:
                mm_patterns[mm] = round(mm_patterns[mm] / total, 3)

        return mm_patterns

    def _calculate_spoofing_probability(self, bids: List[Dict], asks: List[Dict],
                                       history: deque) -> float:
        """
        Calculate probability of spoofing/layering
        Based on order book patterns and historical behavior
        """
        spoofing_score = 0.0

        # Pattern 1: Large orders away from best price
        if len(bids) > 3 and len(asks) > 3:
            # Check for unusually large orders at levels 3-5
            deep_bid_sizes = [b['size'] for b in bids[2:5] if len(bids) > 2]
            deep_ask_sizes = [a['size'] for a in asks[2:5] if len(asks) > 2]

            if deep_bid_sizes or deep_ask_sizes:
                top_sizes = [bids[0]['size'], asks[0]['size']]
                deep_sizes = deep_bid_sizes + deep_ask_sizes

                # Large deep orders relative to top = potential spoofing
                deep_order_multiplier = self.spoofing_config.get('deep_order_multiplier', 3)
                if max(deep_sizes) > np.mean(top_sizes) * deep_order_multiplier:
                    spoofing_score += self.spoofing_config.get('deep_order_penalty', 0.3)

        # Pattern 2: Sudden imbalance changes
        if len(history) >= 10:
            recent_imbalances = [h['volume_imbalance'] for h in list(history)[-10:]]
            imb_std = np.std(recent_imbalances)

            # High volatility in imbalance = potential manipulation
            imbalance_volatility_threshold = self.spoofing_config.get('imbalance_volatility_threshold', 0.3)
            if imb_std > imbalance_volatility_threshold:
                spoofing_score += self.spoofing_config.get('imbalance_volatility_penalty', 0.2)

        # Pattern 3: Round number sizes at multiple levels
        round_sizes = self.market_patterns.get('round_sizes', [100, 200, 500, 1000])
        round_count = 0

        for level in bids[:5] + asks[:5]:
            if level['size'] in round_sizes:
                round_count += 1

        # Too many round numbers = suspicious
        round_number_threshold = self.spoofing_config.get('round_number_threshold', 6)
        if round_count >= round_number_threshold:
            spoofing_score += self.spoofing_config.get('round_number_penalty', 0.3)

        # Pattern 4: Cliff-like depth (sudden drop in liquidity)
        if len(bids) >= 3 and len(asks) >= 3:
            bid_cliff = bids[0]['size'] > (bids[1]['size'] + bids[2]['size']) * 2
            ask_cliff = asks[0]['size'] > (asks[1]['size'] + asks[2]['size']) * 2

            if bid_cliff or ask_cliff:
                spoofing_score += self.spoofing_config.get('cliff_penalty', 0.2)

        return min(spoofing_score, 1.0)

    def _empty_metrics(self, symbol: str) -> OrderBookMetrics:
        """Return empty metrics when no data available"""
        return OrderBookMetrics(
            symbol=symbol,
            timestamp=int(datetime.now().timestamp() * 1000),
            volume_imbalance=0.0,
            value_imbalance=0.0,
            bid_depth=0,
            ask_depth=0,
            depth_ratio=0.0,
            weighted_mid_price=0.0,
            micro_price=0.0,
            book_pressure=BookPressure.BALANCED,
            pressure_score=0.0,
            spread_bps=0.0,
            book_skew=0.0,
            liquidity_score=0.0,
            top_level_ratio=0.0,
            deep_book_ratio=0.0,
            mm_participation={},
            spoofing_probability=0.0
        )

    async def get_imbalance_trend(self, symbol: str, lookback_minutes: int = 5) -> Dict[str, Any]:
        """
        Analyze imbalance trend over time
        Returns trend direction and strength
        """
        if symbol not in self.pressure_history or len(self.pressure_history[symbol]) < 2:
            return {
                'symbol': symbol,
                'trend': 'NEUTRAL',
                'strength': 0.0,
                'momentum': 0.0
            }

        history = list(self.pressure_history[symbol])

        # Calculate trend using linear regression
        x = np.arange(len(history))
        y = np.array(history)

        # Simple linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Determine trend
        if slope > 1:
            trend = 'STRONGLY_BULLISH'
        elif slope > 0.5:
            trend = 'BULLISH'
        elif slope > -0.5:
            trend = 'NEUTRAL'
        elif slope > -1:
            trend = 'BEARISH'
        else:
            trend = 'STRONGLY_BEARISH'

        # Calculate momentum (acceleration)
        if len(history) >= 10:
            first_half = np.mean(history[:len(history)//2])
            second_half = np.mean(history[len(history)//2:])
            momentum = second_half - first_half
        else:
            momentum = 0.0

        return {
            'symbol': symbol,
            'trend': trend,
            'strength': abs(slope),
            'momentum': momentum,
            'current_pressure': history[-1] if history else 0,
            'avg_pressure': np.mean(history),
            'pressure_std': np.std(history)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get calculator performance metrics"""
        return {
            **self.calculation_metrics,
            'symbols_tracked': len(self.imbalance_history),
            'total_historical_points': sum(len(h) for h in self.imbalance_history.values()),
            'cache_hit_rate': (
                self.calculation_metrics['cache_hits'] /
                (self.calculation_metrics['cache_hits'] + self.calculation_metrics['cache_misses'])
                if (self.calculation_metrics['cache_hits'] + self.calculation_metrics['cache_misses']) > 0
                else 0
            )
        }


class TechnicalIndicators:
    """
    Additional technical indicators that complement order book analysis
    """

    def __init__(self, cache: CacheManager, config: Dict = None):
        self.cache = cache
        self.config = config or {}
        
        # Load analytics configuration
        analytics_config = self.config.get('analytics', {})
        self.cache_limits = analytics_config.get('cache_limits', {})
        self.volatility_config = analytics_config.get('volatility', {})
        
        # Use config-driven cache limit
        bar_history_limit = self.cache_limits.get('bar_history', 1000)
        self.bar_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=bar_history_limit))

    async def calculate_weighted_avg_price(self, symbol: str, period_minutes: int = 30) -> Dict[str, float]:
        """
        Calculate VWAP and related metrics
        """
        # Get bars from cache or IBKR
        bars_data = self.cache.get(f"bars:{symbol}")

        if not bars_data:
            return {'vwap': 0.0, 'volume': 0}

        # Parse bars
        bars = []
        if isinstance(bars_data, str):
            bars = json.loads(bars_data)
        else:
            bars = bars_data

        if not bars:
            return {'vwap': 0.0, 'volume': 0}

        total_value = sum(bar['close'] * bar['volume'] for bar in bars)
        total_volume = sum(bar['volume'] for bar in bars)

        vwap = total_value / total_volume if total_volume > 0 else 0

        return {
            'vwap': round(vwap, 2),
            'volume': total_volume,
            'num_bars': len(bars),
            'timestamp': int(datetime.now().timestamp() * 1000)
        }

    async def calculate_volatility_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Calculate realized volatility and related metrics
        """
        bars_data = self.cache.get(f"bars:{symbol}")

        if not bars_data or len(bars_data) < 20:
            return {'volatility': 0.0, 'atr': 0.0}

        # Parse bars
        bars = []
        if isinstance(bars_data, str):
            bars = json.loads(bars_data)
        else:
            bars = bars_data

        # Calculate returns
        closes = [bar['close'] for bar in bars]
        returns = np.diff(np.log(closes))

        # Realized volatility (annualized) using config values
        trading_days = self.volatility_config.get('trading_days', 252)
        minutes_per_day = self.volatility_config.get('minutes_per_day', 390)
        annualization_factor = trading_days * minutes_per_day
        
        volatility = np.std(returns) * np.sqrt(annualization_factor / len(returns))

        # ATR (Average True Range)
        atr_values = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            atr_values.append(tr)

        atr = np.mean(atr_values[-14:]) if len(atr_values) >= 14 else 0

        return {
            'volatility': round(volatility, 4),
            'atr': round(atr, 2),
            'returns_std': round(np.std(returns), 6)
        }


# Module initialization
async def initialize_indicators(cache: CacheManager, config: Dict) -> Dict[str, Any]:
    """
    Initialize all indicator calculators
    """
    try:
        obi_calculator = OrderBookImbalance(cache, config)
        tech_indicators = TechnicalIndicators(cache, config)

        return {
            'obi_calculator': obi_calculator,
            'tech_indicators': tech_indicators,
            'status': 'initialized',
            'metrics': obi_calculator.get_metrics()
        }

    except Exception as e:
        logger.error(f"Failed to initialize indicators: {e}")
        raise
