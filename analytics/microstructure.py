#!/usr/bin/env python3
"""
Market Microstructure Analytics Module
Institutional-grade VPIN calculation and toxicity analysis
Integrates with existing cache and IBKR trade tape data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
from loguru import logger
import json
import asyncio
from enum import Enum
from scipy.stats import norm  # For BV-VPIN Z-score calculation

# Import from existing core modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core import CacheManager
from core.models import Trade, OrderBook, OrderBookLevel


class TradeSide(str, Enum):
    """Trade side classification"""
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


@dataclass
class VolumeBar:
    """Volume-synchronized bar for VPIN calculation"""
    timestamp: int
    buy_volume: float
    sell_volume: float
    total_volume: float
    order_imbalance: float
    num_trades: int
    vwap: float

    @property
    def toxicity(self) -> float:
        """Calculate toxicity as absolute order imbalance"""
        if self.total_volume > 0:
            return abs(self.buy_volume - self.sell_volume) / self.total_volume
        return 0.0


@dataclass
class MarketMakerProfile:
    """Track market maker behavior patterns"""
    name: str
    frequency: float  # Percentage of orders
    avg_duration_ms: float
    cancel_rate: float
    toxicity_score: float  # 0-1, higher = more toxic
    spoofing_score: float  # 0-1, likelihood of spoofing
    last_seen: int
    total_orders: int
    cancelled_orders: int


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading
    Used by Citadel, Two Sigma, Jump Trading
    """

    def __init__(self, cache: CacheManager, config: Dict):
        """Initialize VPIN calculator with cache integration"""
        self.cache = cache
        self.config = config
        
        # Load analytics config
        self.analytics_config = config.get('analytics', {})
        self.vpin_config = self.analytics_config.get('vpin', {})
        self.market_patterns = self.analytics_config.get('market_patterns', {})
        self.cache_limits = self.analytics_config.get('cache_limits', {})

        # Load discovered parameters or use config defaults
        self.discovered_params = self._load_discovered_parameters()

        # VPIN parameters from config
        self.bucket_size = self.discovered_params.get('vpin_bucket_size', 
                                                      self.vpin_config.get('default_bucket_size', 100))
        self.num_buckets = self.discovered_params.get('vpin_window', 50)

        # Volume bars storage with config-driven maxlen
        self.volume_bars: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.num_buckets))

        # Market maker tracking
        self.market_makers: Dict[str, MarketMakerProfile] = {}
        self._initialize_market_makers()

        # Trade classification cache with config-driven size
        self.trade_classifications: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.vpin_config.get('trade_cache_size', 10000))
        )
        
        # NEW: Quote history buffer for Lee-Ready historical classification
        self.quote_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)  # Keep last 1000 quotes per symbol
        )
        self.quote_timestamp_index: Dict[str, Dict[int, Dict]] = defaultdict(dict)
        
        # Debug mode for detailed logging
        self.debug_mode = self.config.get('system', {}).get('debug', False)

        # Metrics tracking
        self.metrics = {
            'calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_calculation_time_ms': 0
        }
        
        logger.info(f"VPIN Calculator initialized with bucket_size={self.bucket_size}")

    def _load_discovered_parameters(self) -> Dict:
        """Load discovered parameters from cache or config"""
        # First check cache for runtime discovered params
        discovered = self.cache.get('discovered_parameters')
        if discovered:
            logger.info("Loaded discovered parameters from cache")
            return discovered

        # Check for config file
        try:
            from pathlib import Path
            import yaml
            discovered_file = Path('config/discovered.yaml')
            if discovered_file.exists():
                with open(discovered_file, 'r') as f:
                    params = yaml.safe_load(f)
                    logger.info("Loaded discovered parameters from config file")
                    return params.get('optimal_parameters', {})
        except Exception as e:
            logger.warning(f"Could not load discovered parameters: {e}")

        # Return defaults
        return {}

    def _initialize_market_makers(self):
        """Initialize market maker profiles from config or discovery"""
        mm_config = self.analytics_config.get('market_makers', {})
        
        if mm_config.get('enable_discovery', False):
            # Load discovered profiles from cache
            discovered = self.cache.get('discovered_market_makers')
            if discovered:
                self.market_makers = discovered
                logger.info(f"Loaded {len(discovered)} discovered market maker profiles")
                return
        
        # Use default profiles from config
        default_profiles = mm_config.get('default_profiles', {})
        self.market_makers = {}
        
        for mm_id, profile in default_profiles.items():
            self.market_makers[mm_id] = MarketMakerProfile(
                name=profile['name'],
                frequency=0.0,  # Will be discovered
                avg_duration_ms=0.0,  # Will be discovered
                cancel_rate=0.0,  # Will be discovered
                toxicity_score=profile.get('toxicity_base', 0.5),
                spoofing_score=0.0,  # Will be discovered
                last_seen=0,
                total_orders=0,
                cancelled_orders=0
            )
        
        # Fallback if no config
        if not self.market_makers:
            self.market_makers = {
                'IBEOS': MarketMakerProfile(
                    name='IB Smart Router',
                    frequency=0.0,
                    avg_duration_ms=0.0,
                    cancel_rate=0.0,
                    toxicity_score=0.15,
                    spoofing_score=0.0,
                    last_seen=0,
                    total_orders=0,
                    cancelled_orders=0
                )
            }
    
    def store_quote_snapshot(self, symbol: str, order_book: Dict, timestamp: Optional[int] = None):
        """
        Store order book snapshot for historical Lee-Ready classification
        Should be called whenever Level 2 data is updated
        """
        if order_book and order_book.get('bids') and order_book.get('asks'):
            # Use provided timestamp or current time
            ts = timestamp if timestamp else int(datetime.now().timestamp() * 1000)
            
            quote = {
                'timestamp': ts,
                'best_bid': float(order_book['bids'][0]['price']),
                'best_ask': float(order_book['asks'][0]['price']),
                'bid_size': float(order_book['bids'][0].get('size', 0)),
                'ask_size': float(order_book['asks'][0].get('size', 0))
            }
            
            # Store in history
            self.quote_history[symbol].append(quote)
            
            # Index by timestamp for fast lookup
            self.quote_timestamp_index[symbol][ts] = quote
            
            # Clean old entries from index (keep last 1000)
            if len(self.quote_timestamp_index[symbol]) > 1000:
                old_timestamps = sorted(self.quote_timestamp_index[symbol].keys())[:-1000]
                for old_ts in old_timestamps:
                    del self.quote_timestamp_index[symbol][old_ts]
    
    def _get_quote_at_time(self, symbol: str, timestamp: int, tolerance_ms: int = 100) -> Optional[Dict]:
        """
        Get historical quote nearest to trade timestamp
        Tolerance: Accept quotes within 100ms of trade
        """
        # Check exact timestamp match first
        if timestamp in self.quote_timestamp_index[symbol]:
            return self.quote_timestamp_index[symbol][timestamp]
        
        # Find nearest quote within tolerance
        if symbol in self.quote_history and self.quote_history[symbol]:
            for quote in reversed(self.quote_history[symbol]):
                if abs(quote['timestamp'] - timestamp) <= tolerance_ms:
                    return quote
                if quote['timestamp'] < timestamp - tolerance_ms:
                    break  # No point looking further back
        
        return None

    def classify_trade(self, trade: Dict, order_book: Optional[Dict] = None) -> TradeSide:
        """
        Lee-Ready algorithm with proper temporal alignment
        Per Lee & Ready (1991) - uses quotes at time of trade
        """
        # Validate trade structure and extract fields
        # Handle different field names that might be used
        trade_price = trade.get('price', trade.get('last', trade.get('trade_price')))
        if trade_price is None:
            logger.warning(f"Trade missing price field: {trade.keys()}")
            return TradeSide.UNKNOWN
        
        trade_price = float(trade_price)
        trade_time = int(trade.get('timestamp', 0))
        symbol = trade.get('symbol', '')
        
        # Step 1: Try to get historical quote at trade time
        historical_quote = self._get_quote_at_time(symbol, trade_time)
        
        # Use provided order book, historical quote, or current quote as fallback
        if order_book and order_book.get('bids') and order_book.get('asks'):
            best_bid = float(order_book['bids'][0]['price'])
            best_ask = float(order_book['asks'][0]['price'])
            if self.debug_mode:
                logger.debug(f"Using provided order book: bid={best_bid:.2f}, ask={best_ask:.2f}")
        elif historical_quote:
            best_bid = historical_quote['best_bid']
            best_ask = historical_quote['best_ask']
            if self.debug_mode:
                logger.debug(f"Using historical quote from {trade_time - historical_quote['timestamp']}ms ago")
        else:
            # No historical quotes - try current order book as last resort
            current_book = self.cache.get_order_book(symbol)
            if current_book and current_book.get('bids') and current_book.get('asks'):
                best_bid = float(current_book['bids'][0]['price'])
                best_ask = float(current_book['asks'][0]['price'])
                if self.debug_mode:
                    logger.debug(f"WARNING: Using current order book (not historical)")
            else:
                # No quotes available - fall back to tick test
                return self._tick_test(symbol, trade_price)
        
        # Lee-Ready Quote Rule (per original 1991 paper)
        mid_price = (best_bid + best_ask) / 2
        
        # Classification logic from Lee & Ready (1991)
        if trade_price > best_ask:
            return TradeSide.BUY
        elif trade_price < best_bid:
            return TradeSide.SELL
        elif trade_price == best_ask:
            return TradeSide.BUY
        elif trade_price == best_bid:
            return TradeSide.SELL
        elif trade_price > mid_price:
            return TradeSide.BUY
        elif trade_price < mid_price:
            return TradeSide.SELL
        else:  # trade_price == mid_price
            # Use tick test for midpoint trades
            return self._tick_test(symbol, trade_price)

    def _tick_test(self, symbol: str, current_price: float) -> TradeSide:
        """Tick test for trade classification"""
        # Get last classified trade
        if symbol in self.trade_classifications and self.trade_classifications[symbol]:
            last_trade = self.trade_classifications[symbol][-1]
            if current_price > last_trade['price']:
                return TradeSide.BUY
            elif current_price < last_trade['price']:
                return TradeSide.SELL
            else:
                # Price unchanged, use last classification
                return TradeSide(last_trade.get('side', TradeSide.UNKNOWN))

        return TradeSide.UNKNOWN

    def _bulk_volume_classify(self, trades: List[Dict]) -> Tuple[float, float]:
        """
        Bulk Volume Classification per Easley, López de Prado, O'Hara (2012)
        'Flow Toxicity and Liquidity in a High-Frequency World'
        Review of Financial Studies 25(5), 1457-1493
        
        Uses standardized price changes (Z-score) and normal CDF for classification
        """
        if not trades or len(trades) < 2:
            # Not enough data for classification
            total_volume = sum(t.get('size', 0) for t in trades) if trades else 0
            return total_volume / 2, total_volume / 2
        
        # Extract prices and volumes, handling potential missing fields
        prices = []
        volumes = []
        for trade in trades:
            # Handle different field names
            price = trade.get('price', trade.get('last', trade.get('trade_price', 0)))
            size = trade.get('size', trade.get('volume', trade.get('quantity', 0)))
            if price > 0 and size > 0:
                prices.append(float(price))
                volumes.append(float(size))
        
        if len(prices) < 2:
            total_volume = sum(volumes)
            return total_volume / 2, total_volume / 2
        
        total_volume = sum(volumes)
        
        # Calculate log returns for the bucket (Easley et al. 2012 method)
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                log_return = np.log(prices[i] / prices[i-1])
                log_returns.append(log_return)
        
        if not log_returns:
            # No valid returns - split 50/50
            return total_volume / 2, total_volume / 2
        
        # Calculate standardized price change (Z-score)
        # Per Easley et al. (2012) Section 3.2
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns, ddof=1) if len(log_returns) > 1 else np.std(log_returns)
        
        # Handle zero variance case
        if std_return < 1e-10:
            # No price movement - neutral classification
            return total_volume / 2, total_volume / 2
        
        # Z-score of the bucket's price change
        # Using sample size adjustment for small samples
        z_score = mean_return / (std_return / np.sqrt(len(log_returns)))
        
        # Apply CDF of standard normal distribution
        # This gives probability of buy-initiated volume
        buy_probability = norm.cdf(z_score)
        
        # Classify entire bucket volume
        buy_volume = total_volume * buy_probability
        sell_volume = total_volume * (1 - buy_probability)
        
        # Log classification for debugging
        if self.debug_mode:
            logger.debug(f"BV-VPIN: {len(trades)} trades, {len(log_returns)} returns")
            logger.debug(f"  Mean return: {mean_return:.6f}, Std: {std_return:.6f}")
            logger.debug(f"  Z-score: {z_score:.4f}, Buy prob: {buy_probability:.4f}")
            logger.debug(f"  Buy vol: {buy_volume:.0f}, Sell vol: {sell_volume:.0f}")
        
        return buy_volume, sell_volume

    def create_volume_bar(self, trades: List[Dict], bucket_size: Optional[int] = None) -> Optional[VolumeBar]:
        """Create a volume-synchronized bar from trades"""
        if not trades:
            return None

        bucket_size = bucket_size or self.bucket_size

        # Check if bulk volume classification is enabled
        if self.vpin_config.get('bulk_volume_classification', False):
            buy_volume, sell_volume = self._bulk_volume_classify(trades)
            total_value = sum(t['price'] * t['size'] for t in trades)
        else:
            # Original trade-by-trade classification
            buy_volume = 0
            sell_volume = 0
            total_value = 0

            for trade in trades:
                side = self.classify_trade(trade)

                # Store classification for future tick tests
                self.trade_classifications[trade.get('symbol', '')].append({
                    'price': float(trade.get('price', 0)),
                    'side': side.value,
                    'timestamp': int(trade.get('timestamp', 0))
                })

                if side == TradeSide.BUY:
                    buy_volume += trade['size']
                elif side == TradeSide.SELL:
                    sell_volume += trade['size']
                else:
                    # Unknown trades split 50/50
                    buy_volume += trade['size'] / 2
                    sell_volume += trade['size'] / 2

                total_value += trade['price'] * trade['size']

        total_volume = buy_volume + sell_volume

        if total_volume > 0:
            return VolumeBar(
                timestamp=int(trades[-1].get('timestamp', 0)),
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                total_volume=total_volume,
                order_imbalance=(buy_volume - sell_volume) / total_volume,
                num_trades=len(trades),
                vwap=total_value / total_volume
            )

        return None

    async def calculate_vpin(self, symbol: str, trades: Optional[List[Dict]] = None, lookback_minutes: int = 30) -> Dict[str, Any]:
        """
        Calculate VPIN for a symbol using provided trades or recent trades from cache
        
        Args:
            symbol: Trading symbol
            trades: Optional list of trade dictionaries. If not provided, fetches from cache
            lookback_minutes: Minutes to look back if fetching from cache (ignored if trades provided)
            
        Returns:
            Dictionary with VPIN metrics and toxicity score 0-1 (>0.4 indicates toxic/informed flow)
        """
        import time
        start_time = time.time()

        try:
            # Use provided trades or fetch from cache
            if trades is not None:
                # Trades provided directly - institutional-grade flexibility
                trades_data = trades
                logger.debug(f"Using {len(trades_data)} provided trades for VPIN calculation")
            else:
                # Fetch from cache (backward compatibility)
                trades_data = self.cache.get_recent_trades(symbol, lookback_minutes * 60)

                if not trades_data:
                    logger.warning(f"No trades found for {symbol}")
                    self.metrics['cache_misses'] += 1
                    return {
                        'symbol': symbol,
                        'vpin': 0.0,
                        'confidence': 0.0,
                        'num_trades': 0,
                        'message': 'Insufficient trade data'
                    }

                self.metrics['cache_hits'] += 1

            # Parse trades if they're JSON strings
            trades = []
            for trade in trades_data:
                if isinstance(trade, str):
                    trades.append(json.loads(trade))
                else:
                    trades.append(trade)

            # Create volume bars
            current_bucket = []
            current_volume = 0

            for trade in trades:
                current_bucket.append(trade)
                current_volume += trade['size']

                # When bucket is full, create volume bar
                if current_volume >= self.bucket_size:
                    bar = self.create_volume_bar(current_bucket)
                    if bar:
                        self.volume_bars[symbol].append(bar)

                    # Start new bucket with overflow
                    overflow = current_volume - self.bucket_size
                    if overflow > 0 and current_bucket:
                        current_bucket = [current_bucket[-1]]
                        current_bucket[-1]['size'] = overflow
                        current_volume = overflow
                    else:
                        current_bucket = []
                        current_volume = 0

            # Calculate VPIN from volume bars
            bars = list(self.volume_bars[symbol])

            if len(bars) < 5:  # Need minimum bars
                return {
                    'symbol': symbol,
                    'vpin': 0.0,
                    'confidence': 0.0,
                    'num_bars': len(bars),
                    'message': 'Insufficient volume bars'
                }

            # Calculate VPIN as mean toxicity
            toxicities = [bar.toxicity for bar in bars]
            vpin = np.mean(toxicities)

            # Enhance with market maker intelligence
            vpin_enhanced = self._enhance_vpin_with_mm_intelligence(symbol, vpin, trades)

            # Calculate confidence based on data quality
            confidence = min(1.0, len(bars) / self.num_buckets)

            # Calculate additional metrics
            recent_bars = bars[-10:] if len(bars) >= 10 else bars
            trend = 'increasing' if len(recent_bars) >= 2 and recent_bars[-1].toxicity > recent_bars[0].toxicity else 'decreasing'

            # Update metrics
            calc_time = (time.time() - start_time) * 1000
            self.metrics['calculations'] += 1
            self.metrics['avg_calculation_time_ms'] = (
                (self.metrics['avg_calculation_time_ms'] * (self.metrics['calculations'] - 1) + calc_time)
                / self.metrics['calculations']
            )

            result = {
                'symbol': symbol,
                'vpin': round(vpin_enhanced, 4),
                'vpin_raw': round(vpin, 4),
                'confidence': round(confidence, 2),
                'num_bars': len(bars),
                'num_trades': sum(bar.num_trades for bar in bars),
                'toxicity_trend': trend,
                'bucket_size': self.bucket_size,
                'calculation_time_ms': round(calc_time, 2),
                'timestamp': int(datetime.now().timestamp() * 1000),
                'interpretation': self._interpret_vpin(vpin_enhanced),
                'market_maker_influence': self._get_mm_influence(symbol)
            }

            # Cache the result
            self.cache.set_vpin(symbol, vpin_enhanced)
            self.cache.set_metrics(symbol, {'vpin': vpin_enhanced, 'vpin_details': result})

            return result

        except Exception as e:
            logger.error(f"Error calculating VPIN for {symbol}: {e}")
            return {
                'symbol': symbol,
                'vpin': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

    def _enhance_vpin_with_mm_intelligence(self, symbol: str, base_vpin: float, trades: List[Dict]) -> float:
        """Adjust VPIN based on market maker toxicity"""
        # Track market maker activity
        mm_activity = defaultdict(int)
        total_trades = len(trades)

        # Count MM participation (would need Level 2 data in production)
        # For now, simulate based on known patterns
        for trade in trades:
            # In production, this would come from Level 2 market maker field
            if trade['size'] in self.market_patterns.get('retail_sizes', [100, 200, 300]):  # Typical retail sizes
                mm_activity['IBEOS'] += 1
            elif trade['size'] > self.market_patterns.get('sweep_volume_threshold', 5000):  # Large institutional
                mm_activity['CDRG'] += 1

        # Calculate toxicity adjustment
        toxicity_adjustment = 0
        for mm_id, count in mm_activity.items():
            if mm_id in self.market_makers:
                mm_profile = self.market_makers[mm_id]
                participation = count / total_trades if total_trades > 0 else 0
                toxicity_adjustment += participation * mm_profile.toxicity_score * 0.3

        # Combine base VPIN with MM toxicity
        enhanced_vpin = min(1.0, base_vpin * (1 + toxicity_adjustment))

        return enhanced_vpin

    def _interpret_vpin(self, vpin: float) -> Dict[str, str]:
        """Interpret VPIN score with institutional context"""
        if vpin < 0.2:
            return {
                'level': 'LOW',
                'interpretation': 'Balanced flow, low information asymmetry',
                'action': 'Safe to provide liquidity'
            }
        elif vpin < 0.4:
            return {
                'level': 'MODERATE',
                'interpretation': 'Normal market conditions',
                'action': 'Standard risk parameters'
            }
        elif vpin < 0.6:
            return {
                'level': 'ELEVATED',
                'interpretation': 'Increased informed trading detected',
                'action': 'Reduce position sizes, widen spreads'
            }
        elif vpin < 0.8:
            return {
                'level': 'HIGH',
                'interpretation': 'Significant toxic flow, likely institutional',
                'action': 'Defensive positioning, reduce exposure'
            }
        else:
            return {
                'level': 'CRITICAL',
                'interpretation': 'Extreme toxicity, potential adverse selection',
                'action': 'Exit positions, avoid market making'
            }

    def _get_mm_influence(self, symbol: str) -> Dict[str, Any]:
        """Get current market maker influence metrics"""
        return {
            'dominant_mm': max(self.market_makers.items(),
                             key=lambda x: x[1].frequency)[0] if self.market_makers else 'UNKNOWN',
            'toxicity_score': np.mean([mm.toxicity_score for mm in self.market_makers.values()]),
            'spoofing_risk': max([mm.spoofing_score for mm in self.market_makers.values()]) if self.market_makers else 0
        }

    async def discover_optimal_bucket_size(self, symbol: str, sample_minutes: int = 60) -> int:
        """
        Discover optimal VPIN bucket size from YOUR market data
        Not academic 50-share assumptions
        """
        try:
            # Get sample trades
            trades_data = self.cache.get_recent_trades(symbol, sample_minutes * 60)

            if not trades_data or len(trades_data) < self.vpin_config.get('min_trades', 100):
                logger.warning(f"Insufficient data for bucket size discovery")
                return self.bucket_size

            # Parse trades
            trades = []
            for trade in trades_data:
                if isinstance(trade, str):
                    trades.append(json.loads(trade))
                else:
                    trades.append(trade)

            # Analyze volume distribution
            volumes = [t['size'] for t in trades]

            # Find natural clustering in YOUR market
            percentiles = np.percentile(volumes, [10, 25, 50, 75, 90])

            # Use median as baseline
            optimal_bucket = int(percentiles[2])

            # Adjust based on market activity
            if len(trades) > self.vpin_config['activity_thresholds']['high']:  # High activity
                optimal_bucket = int(percentiles[3])  # Use 75th percentile
            elif len(trades) < self.vpin_config['activity_thresholds']['low']:  # Low activity
                optimal_bucket = int(percentiles[1])  # Use 25th percentile

            logger.info(f"Discovered optimal bucket size for {symbol}: {optimal_bucket} shares")
            logger.info(f"Volume distribution - P10: {percentiles[0]:.0f}, P50: {percentiles[2]:.0f}, P90: {percentiles[4]:.0f}")

            # Update discovered parameters
            self.discovered_params['vpin_bucket_size'] = optimal_bucket
            self.bucket_size = optimal_bucket

            # Cache discovery
            self.cache.set('discovered_parameters', self.discovered_params, ttl=3600)

            return optimal_bucket

        except Exception as e:
            logger.error(f"Error discovering bucket size: {e}")
            return self.bucket_size

    def get_metrics(self) -> Dict[str, Any]:
        """Get calculator performance metrics"""
        return {
            **self.metrics,
            'current_bucket_size': self.bucket_size,
            'discovered_params': self.discovered_params,
            'market_makers_tracked': len(self.market_makers),
            'cache_hit_rate': (self.metrics['cache_hits'] /
                             (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                             if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0)
        }


class HiddenOrderDetector:
    """
    Detect iceberg orders and hidden liquidity
    Used by HFT firms to identify large institutional orders
    """

    def __init__(self, cache: CacheManager, config: Dict = None):
        self.cache = cache
        self.config = config or {}
        
        # Load analytics config
        self.analytics_config = self.config.get('analytics', {})
        self.cache_limits = self.analytics_config.get('cache_limits', {})
        self.market_patterns = self.analytics_config.get('market_patterns', {})
        
        self.suspected_icebergs: Dict[str, List[Dict]] = defaultdict(list)
        
        # Metrics tracking
        self.metrics = {
            'detections': 0,
            'symbols_analyzed': set(),
            'total_hidden_liquidity_detected': 0,
            'avg_confidence': 0,
            'patterns_found': {'refill': 0, 'round_size': 0, 'persistent': 0}
        }

    async def detect_hidden_orders(self, symbol: str, order_book: Dict) -> Dict[str, Any]:
        """
        Detect potential hidden/iceberg orders in the book
        """
        try:
            hidden_levels = {
                'bids': [],
                'asks': []
            }

            # Analyze each side
            for side in ['bids', 'asks']:
                if side not in order_book or not order_book[side]:
                    continue

                levels = order_book[side]

                for i, level in enumerate(levels):
                    # Iceberg detection patterns
                    is_hidden = False
                    confidence = 0.0

                    # Pattern 1: Consistent refills at same price
                    if self._check_refill_pattern(symbol, side, level['price']):
                        is_hidden = True
                        confidence += 0.4

                    # Pattern 2: Round number size (100, 500, 1000)
                    if level['size'] in self.market_patterns.get('round_sizes', [100, 200, 500, 1000]):
                        confidence += 0.2

                    # Pattern 3: Size doesn't decrease despite trades
                    if self._check_persistent_size(symbol, side, level['price'], level['size']):
                        is_hidden = True
                        confidence += 0.4

                    if confidence > 0.5:
                        hidden_levels[side].append({
                            'price': level['price'],
                            'visible_size': level['size'],
                            'estimated_hidden': level['size'] * 5,  # Typical iceberg ratio
                            'confidence': min(confidence, 1.0),
                            'level': i
                        })

            result = {
                'symbol': symbol,
                'hidden_bid_levels': hidden_levels['bids'],
                'hidden_ask_levels': hidden_levels['asks'],
                'total_hidden_liquidity': sum(h['estimated_hidden'] for h in hidden_levels['bids'] + hidden_levels['asks']),
                'timestamp': int(datetime.now().timestamp() * 1000)
            }

            # Track for pattern analysis
            self.suspected_icebergs[symbol].append(result)

            # Keep only recent detections
            if len(self.suspected_icebergs[symbol]) > self.cache_limits.get('iceberg_history', 100):
                self.suspected_icebergs[symbol] = self.suspected_icebergs[symbol][-self.cache_limits.get('iceberg_history', 100):]
            
            # Update metrics
            self.metrics['detections'] += 1
            self.metrics['symbols_analyzed'].add(symbol)
            self.metrics['total_hidden_liquidity_detected'] += result['total_hidden_liquidity']
            if hidden_levels['bids'] or hidden_levels['asks']:
                all_confidences = [h['confidence'] for h in hidden_levels['bids'] + hidden_levels['asks']]
                if all_confidences:
                    self.metrics['avg_confidence'] = np.mean(all_confidences)

            return result

        except Exception as e:
            logger.error(f"Error detecting hidden orders: {e}")
            return {
                'symbol': symbol,
                'hidden_bid_levels': [],
                'hidden_ask_levels': [],
                'error': str(e)
            }

    def _check_refill_pattern(self, symbol: str, side: str, price: float) -> bool:
        """Check if price level shows refill pattern"""
        # In production, this would track order book changes over time
        # For now, return based on heuristics
        if symbol in self.suspected_icebergs:
            recent = self.suspected_icebergs[symbol][-10:]
            refills = sum(1 for r in recent if any(
                abs(h['price'] - price) < 0.01
                for h in r.get(f'hidden_{side}_levels', [])
            ))
            return refills >= 3
        return False

    def _check_persistent_size(self, symbol: str, side: str, price: float, size: int) -> bool:
        """Check if size persists despite trading"""
        # Would need historical order book snapshots in production
        # Simplified check for now
        return size in self.market_patterns.get('persistent_sizes', [100, 500, 1000]) and size > 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detector performance metrics"""
        return {
            'detections': self.metrics['detections'],
            'symbols_analyzed': len(self.metrics['symbols_analyzed']),
            'total_hidden_liquidity_detected': self.metrics['total_hidden_liquidity_detected'],
            'avg_confidence': self.metrics['avg_confidence'],
            'patterns_found': self.metrics['patterns_found'],
            'history_size': sum(len(h) for h in self.suspected_icebergs.values())
        }


class SweepDetector:
    """
    Detect sweep orders and urgent institutional flow
    """

    def __init__(self, cache: CacheManager, config: Dict = None):
        self.cache = cache
        self.config = config or {}
        
        # Load analytics config
        self.analytics_config = self.config.get('analytics', {})
        self.cache_limits = self.analytics_config.get('cache_limits', {})
        self.market_patterns = self.analytics_config.get('market_patterns', {})
        
        self.sweep_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.cache_limits.get('sweep_history', 100))
        )
        
        # Metrics tracking
        self.metrics = {
            'sweeps_detected': 0,
            'symbols_analyzed': set(),
            'total_sweep_volume': 0,
            'avg_confidence': 0,
            'urgency_scores': []
        }

    async def detect_sweeps(self, symbol: str, window_seconds: int = 5) -> Dict[str, Any]:
        """
        Detect sweep orders (urgent liquidity taking)
        """
        try:
            # Get recent trades
            trades_data = self.cache.get_recent_trades(symbol, window_seconds)

            if not trades_data or len(trades_data) < 3:
                return {
                    'symbol': symbol,
                    'sweep_detected': False,
                    'confidence': 0.0
                }

            # Parse trades
            trades = []
            for trade in trades_data:
                if isinstance(trade, str):
                    trades.append(json.loads(trade))
                else:
                    trades.append(trade)

            # Sort by timestamp
            trades.sort(key=lambda x: x['timestamp'])

            # Sweep detection criteria
            sweep_detected = False
            confidence = 0.0

            # Check for rapid succession of trades
            if len(trades) >= 3:
                time_span = (trades[-1]['timestamp'] - trades[0]['timestamp']) / 1000  # Convert to seconds

                if time_span > 0 and time_span <= window_seconds:
                    trades_per_second = len(trades) / time_span

                    # High trade frequency indicates sweep
                    if trades_per_second > 2:
                        sweep_detected = True
                        confidence += 0.3

                    # Large total volume
                    total_volume = sum(t['size'] for t in trades)
                    if total_volume > self.market_patterns.get('sweep_volume_threshold', 5000):
                        sweep_detected = True
                        confidence += 0.3

                    # Same direction (all buys or all sells)
                    order_book = self.cache.get_order_book(symbol)
                    if order_book:
                        mid_price = self._calculate_mid_price(order_book)
                        buy_trades = sum(1 for t in trades if t['price'] >= mid_price)
                        sell_trades = len(trades) - buy_trades

                        directional_ratio = max(buy_trades, sell_trades) / len(trades)
                        if directional_ratio > 0.8:
                            sweep_detected = True
                            confidence += 0.4

            result = {
                'symbol': symbol,
                'sweep_detected': sweep_detected,
                'confidence': min(confidence, 1.0),
                'num_trades': len(trades),
                'total_volume': sum(t['size'] for t in trades),
                'time_span_seconds': (trades[-1]['timestamp'] - trades[0]['timestamp']) / 1000 if len(trades) > 1 else 0,
                'direction': self._determine_sweep_direction(trades, symbol),
                'urgency_score': confidence,
                'timestamp': int(datetime.now().timestamp() * 1000)
            }

            # Track history
            if sweep_detected:
                self.sweep_history[symbol].append(result)
                
            # Update metrics
            self.metrics['symbols_analyzed'].add(symbol)
            if sweep_detected:
                self.metrics['sweeps_detected'] += 1
                self.metrics['total_sweep_volume'] += result['total_volume']
                self.metrics['urgency_scores'].append(result['urgency_score'])
                if self.metrics['urgency_scores']:
                    self.metrics['avg_confidence'] = np.mean(self.metrics['urgency_scores'][-100:])  # Last 100

            return result

        except Exception as e:
            logger.error(f"Error detecting sweeps: {e}")
            return {
                'symbol': symbol,
                'sweep_detected': False,
                'error': str(e)
            }

    def _calculate_mid_price(self, order_book: Dict) -> float:
        """Calculate mid price from order book"""
        if order_book.get('bids') and order_book.get('asks'):
            best_bid = order_book['bids'][0]['price']
            best_ask = order_book['asks'][0]['price']
            return (best_bid + best_ask) / 2
        return 0.0

    def _determine_sweep_direction(self, trades: List[Dict], symbol: str) -> str:
        """Determine sweep direction (BUY/SELL)"""
        order_book = self.cache.get_order_book(symbol)
        if not order_book:
            return 'UNKNOWN'

        mid_price = self._calculate_mid_price(order_book)
        buy_volume = sum(t['size'] for t in trades if t['price'] >= mid_price)
        sell_volume = sum(t['size'] for t in trades if t['price'] < mid_price)

        if buy_volume > sell_volume * 1.5:
            return 'BUY_SWEEP'
        elif sell_volume > buy_volume * 1.5:
            return 'SELL_SWEEP'
        else:
            return 'MIXED'
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sweep detector performance metrics"""
        return {
            'sweeps_detected': self.metrics['sweeps_detected'],
            'symbols_analyzed': len(self.metrics['symbols_analyzed']),
            'total_sweep_volume': self.metrics['total_sweep_volume'],
            'avg_confidence': self.metrics['avg_confidence'],
            'urgency_scores_count': len(self.metrics['urgency_scores']),
            'history_size': sum(len(h) for h in self.sweep_history.values())
        }


# Module-level initialization function
async def initialize_microstructure_analytics(cache: CacheManager, config: Dict) -> Dict[str, Any]:
    """
    Initialize all microstructure analytics components
    Returns initialized calculators
    """
    try:
        vpin_calc = VPINCalculator(cache, config)
        hidden_detector = HiddenOrderDetector(cache, config)
        sweep_detector = SweepDetector(cache, config)

        # Discover optimal parameters if enough data and discovery is enabled
        if config.get('analytics', {}).get('vpin', {}).get('enable_discovery', True):
            symbols = config.get('symbols', {}).get('primary', [])
            if symbols:
                primary_symbol = symbols[0].get('symbol')
                if primary_symbol:
                    optimal_bucket = await vpin_calc.discover_optimal_bucket_size(primary_symbol)
                    logger.info(f"Discovered optimal VPIN bucket size for {primary_symbol}: {optimal_bucket}")

        return {
            'vpin_calculator': vpin_calc,
            'hidden_detector': hidden_detector,
            'sweep_detector': sweep_detector,
            'status': 'initialized',
            'metrics': vpin_calc.get_metrics()
        }

    except Exception as e:
        logger.error(f"Failed to initialize microstructure analytics: {e}")
        raise
