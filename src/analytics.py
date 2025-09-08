#!/usr/bin/env python3
"""
Analytics Module - Parameter Discovery and Real-time Analytics
Handles empirical parameter discovery and calculates all trading metrics

Parameter Discovery: VPIN bucket sizing, MM profiling, volatility regimes, correlations
Analytics Engine: Enhanced VPIN, OBI, Hidden orders, GEX/DEX, Sweep detection
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import acf
from collections import defaultdict
import json
import yaml
import redis
import redis.asyncio as aioredis
import time
import asyncio
import pytz
import math
import re
from datetime import datetime, time as datetime_time
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback

# OCC option symbol regex pattern: UNDERLYING(1-6) + YYMMDD(6) + C/P + STRIKE(8)
_OCC_RE = re.compile(r'^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$')


class ParameterDiscovery:
    """
    Empirically discover optimal trading parameters from market data.
    Runs on startup and periodically to adapt to market conditions.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize parameter discovery with configuration.
        All parameters come from config.yaml - NO HARDCODING.
        Note: redis_conn can be either sync or async Redis depending on context.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        
        # Load symbols from config - extract from dict structure
        syms = config.get('symbols', {})
        self.symbols = list({*syms.get('standard', []), *syms.get('level2', [])})
        
        # Load discovery configuration
        self.discovery_config = config.get('parameter_discovery', {})
        
        # Discovery results will be stored here
        self.discovered_params = {}
        
        # Get Level 2 symbols from IBKR config
        self.level2_symbols = config.get('ibkr', {}).get('level2_symbols', ['SPY', 'QQQ', 'IWM'])
    
    async def run_discovery(self):
        """
        Run complete parameter discovery pipeline.
        This should be run on startup and periodically (e.g., daily).
        """
        self.logger.info("Starting parameter discovery...")
        start_time = time.time()
        
        try:
            # 1. Discover VPIN bucket size
            self.discovered_params['vpin_bucket_size'] = await self.discover_vpin_bucket_size()
            
            # 2. Discover temporal structure
            self.discovered_params['lookback_bars'] = await self.discover_temporal_structure()
            
            # 3. Analyze flow toxicity (replaces market maker profiling)
            self.discovered_params['flow_toxicity'] = await self.analyze_flow_toxicity()
            
            # 4. Discover volatility regimes
            self.discovered_params['vol_regimes'] = await self.discover_volatility_regimes()
            
            # 5. Calculate correlations
            self.discovered_params['correlation_matrix'] = await self.calculate_correlations()
            
            # Store all parameters in Redis
            await self.store_to_redis()
            
            # Generate config file
            await self.generate_config_file()
            
            elapsed = time.time() - start_time
            self.logger.info(f"Parameter discovery completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in parameter discovery: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _check_data_freshness(self, symbol: str) -> bool:
        """
        Check if market data is fresh (not stale weekend/holiday data).
        Returns True if data is fresh, False if stale.
        """
        # Try to get last update timestamp from various sources
        last_timestamp = None
        
        # Check trades timestamp
        trades = await self.redis.lrange(f'market:{symbol}:trades', -1, -1)
        if trades:
            try:
                trade = json.loads(trades[0])
                if 'timestamp' in trade:
                    last_timestamp = trade['timestamp']
            except:
                pass
        
        # Check bars timestamp (bars are stored as a list)
        if not last_timestamp:
            bars_list = await self.redis.lrange(f'market:{symbol}:bars', -1, -1)  # Get last bar
            if bars_list:
                try:
                    last_bar = json.loads(bars_list[0])
                    # Bars use 'time' field, not 'timestamp'
                    if 'time' in last_bar:
                        last_timestamp = last_bar['time']
                    elif 'timestamp' in last_bar:
                        last_timestamp = last_bar['timestamp']
                except:
                    pass
        
        if not last_timestamp:
            return False
        
        # Check if older than threshold
        threshold = self.discovery_config.get('data_staleness_threshold', 3600)
        # Convert timestamp from milliseconds to seconds if needed
        if last_timestamp > 1e10:  # Timestamp is in milliseconds
            last_timestamp = last_timestamp / 1000
        age = time.time() - float(last_timestamp)
        
        # If market is closed, allow data from last trading session (up to 24 hours old)
        import pytz
        from datetime import datetime
        ny_tz = pytz.timezone('US/Eastern')
        now = datetime.now(ny_tz)
        is_market_hours = now.weekday() < 5 and 9 <= now.hour < 16
        
        if not is_market_hours:
            # Market is closed - allow older data (last trading session)
            threshold = 86400  # 24 hours
        
        if age > threshold:
            self.logger.warning(f"Data for {symbol} is stale: {age:.0f}s old (threshold: {threshold}s)")
            return False
        
        return True
    
    async def discover_vpin_bucket_size(self) -> int:
        """
        Discover optimal VPIN bucket size from trade clustering.
        Uses K-means clustering to find natural trade size groups.
        """
        vpin_config = self.discovery_config.get('vpin', {})
        min_trades = vpin_config.get('min_trades_required', 1000)
        default_size = vpin_config.get('default_bucket_size', 100)
        min_size = vpin_config.get('min_bucket_size', 50)
        max_size = vpin_config.get('max_bucket_size', 500)
        num_clusters = vpin_config.get('num_clusters', 5)
        
        self.logger.info("Discovering VPIN bucket size...")
        
        # Check data freshness (now handles market hours internally)
        if not await self._check_data_freshness('SPY'):
            self.logger.warning("Using default VPIN bucket size due to stale data")
            return default_size
        
        # Fetch trades from Redis
        trades_json = await self.redis.lrange('market:SPY:trades', 0, 9999)
        
        # Check minimum trades requirement
        if len(trades_json) < min_trades:
            self.logger.warning(f"Insufficient trades: {len(trades_json)} < {min_trades}")
            return default_size
        
        # Extract trade sizes
        trade_sizes = []
        for trade_str in trades_json:
            try:
                trade = json.loads(trade_str)
                if 'size' in trade:
                    trade_sizes.append(trade['size'])
            except:
                continue
        
        if len(trade_sizes) < min_trades:
            self.logger.warning(f"Insufficient valid trades: {len(trade_sizes)}")
            return default_size
        
        # Apply K-means clustering
        from sklearn.cluster import KMeans
        sizes_array = np.array(trade_sizes).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(sizes_array)
        
        # Sort cluster centers and select median
        centers = sorted(kmeans.cluster_centers_.flatten())
        median_center = centers[num_clusters // 2]  # Middle cluster
        
        # Clamp to configured range
        bucket_size = int(np.clip(median_center, min_size, max_size))
        
        self.logger.info(f"Discovered VPIN bucket size: {bucket_size} shares")
        self.logger.info(f"  Cluster centers: {[int(c) for c in centers]}")
        self.logger.info(f"  Selected median: {int(median_center)}")
        
        return bucket_size
    
    async def discover_temporal_structure(self) -> int:
        """
        Find optimal lookback period using autocorrelation analysis.
        Uses Box-Jenkins methodology for time series analysis.
        """
        temporal_config = self.discovery_config.get('temporal', {})
        min_bars = temporal_config.get('min_bars_required', 100)
        max_lookback = temporal_config.get('max_lookback', 30)
        significance_threshold = temporal_config.get('significance_threshold', 0.05)
        default_lookback = temporal_config.get('default_lookback', 10)
        
        self.logger.info("Discovering temporal structure...")
        
        # Check data freshness
        if not await self._check_data_freshness('SPY'):
            self.logger.warning("Using default lookback due to stale data")
            return default_lookback
        
        # Fetch bars from Redis (stored as list)
        bars_list = await self.redis.lrange('market:SPY:bars', 0, -1)
        if not bars_list:
            self.logger.warning("No bar data available")
            return default_lookback
        
        bars = [json.loads(bar) for bar in bars_list]
        
        # Check minimum bars requirement
        if len(bars) < min_bars:
            self.logger.warning(f"Insufficient bars: {len(bars)} < {min_bars}")
            return default_lookback
        
        # Extract close prices and calculate log returns
        closes = [float(bar['close']) for bar in bars]
        log_returns = np.diff(np.log(closes))
        
        if len(log_returns) < min_bars:
            self.logger.warning(f"Insufficient returns: {len(log_returns)}")
            return default_lookback
        
        # Compute autocorrelation function
        from statsmodels.tsa.stattools import acf
        
        # Calculate ACF with confidence intervals
        # ACF can return 2-4 values depending on parameters
        acf_result = acf(log_returns, nlags=min(50, len(log_returns)//4), 
                        alpha=significance_threshold)
        
        # Safely unpack only what we need
        if isinstance(acf_result, tuple) and len(acf_result) >= 2:
            acf_values = acf_result[0]
            confint = acf_result[1] if len(acf_result) > 1 else None
        else:
            acf_values = acf_result
            confint = None
        
        # Find significant lags (outside confidence interval)
        significance_bound = 2 / np.sqrt(len(log_returns))
        significant_lags = []
        
        for lag in range(1, len(acf_values)):
            if abs(acf_values[lag]) > significance_bound:
                significant_lags.append(lag)
        
        # Select the last significant lag as lookback
        if significant_lags:
            lookback = min(significant_lags[-1], max_lookback)
        else:
            lookback = default_lookback
        
        self.logger.info(f"Discovered temporal lookback: {lookback} bars")
        self.logger.info(f"  Significant lags found: {significant_lags[:10]}")
        self.logger.info(f"  Significance bound: {significance_bound:.4f}")
        
        return lookback
    
    # Venue alias mapping to handle spelling variations
    VENUE_ALIASES = {
        # NASDAQ variants
        "NSDQ": "NSDQ",
        "NASDAQ": "NSDQ",
        "ISLAND": "NSDQ",  # NASDAQ's ISLAND book
        "PSX": "PSX",      # NASDAQ PSX
        
        # NYSE variants
        "NYSE": "NYSE",
        "ARCA": "ARCA",
        "ARCX": "ARCA",    # Alternative ARCA code
        "AMEX": "AMEX",    # NYSE American
        "NYSENAT": "NYSENAT",  # NYSE National
        
        # CBOE variants
        "BATS": "BATS",
        "BZX": "BATS",     # CBOE BZX is BATS
        "BYX": "BYX",      # CBOE BYX
        "BEX": "BEX",      # CBOE BEX
        "EDGX": "EDGX",    # CBOE EDGX
        "EDGEA": "EDGEA",  # CBOE EDGA
        "EDGA": "EDGEA",   # Map EDGA to EDGEA
        
        # Other exchanges
        "IEX": "IEX",
        "MEMX": "MEMX",
        "CHX": "CHX",
        "LTSE": "LTSE",
        "PEARL": "PEARL",  # MIAX PEARL
        "ISE": "ISE",      # International Securities Exchange
        "DRCTEDGE": "DRCTEDGE",
        
        # IB internal
        "IBEOS": "IBEOS",
        "IBKRATS": "IBKRATS",
        "OVERNIGHT": "OVERNIGHT",
        "SMART": "SMART",  # If we see SMART, keep it
    }
    
    def _score_venue(self, venue: str) -> float:
        """Score a single venue with alias resolution and logging."""
        tox_config = self.discovery_config.get('toxicity_detection', {})
        venue_scores = tox_config.get('venue_scores', {})
        
        # Normalize and map venue
        v = venue.strip().upper()
        v = self.VENUE_ALIASES.get(v, v)
        
        # Get score or default
        if v not in venue_scores:
            default_score = venue_scores.get('UNKNOWN', 0.5)
            # Use INFO level initially to track unseen venues, can change to DEBUG later
            self.logger.info(f"[toxicity] unseen venue '{venue}' (mapped to '{v}'), using UNKNOWN={default_score}")
            return default_score
            
        return venue_scores.get(v, 0.5)
    
    def _venue_toxicity_from_config(self, venue_counts: dict) -> float:
        """Calculate venue-based toxicity score from venue distribution."""
        if not venue_counts:
            return 0.5  # Default neutral if no venue data
            
        total = sum(venue_counts.values()) or 1
        weighted_sum = 0
        
        # Score each venue using alias mapping
        for venue, count in venue_counts.items():
            score = self._score_venue(venue)
            weighted_sum += score * count
            
        return weighted_sum / total
    
    def _detect_sweeps(self, trades: list, window_ms: float) -> int:
        """Detect sweep patterns in trades (rapid multi-level takes)."""
        if not trades:
            return 0
            
        sweeps = 0
        trades_sorted = sorted(trades, key=lambda t: t.get('time', 0))
        i = 0
        
        while i < len(trades_sorted):
            j = i + 1
            # Find all trades within sweep window
            while (j < len(trades_sorted) and 
                   trades_sorted[j].get('time', 0) - trades_sorted[i].get('time', 0) <= window_ms):
                j += 1
            
            window = trades_sorted[i:j]
            if len(window) >= 3:  # Need at least 3 trades for a sweep
                # Check if same-sided trades with price progression
                prices = [t.get('price', 0) for t in window]
                if prices == sorted(prices) or prices == sorted(prices, reverse=True):
                    sweeps += 1
            i = j
            
        return sweeps
    
    def _trade_pattern_toxicity(self, trades: list) -> float:
        """Calculate toxicity from trade patterns (odd lots, sweeps, blocks)."""
        if not trades:
            return 0.5
            
        tox_config = self.discovery_config.get('toxicity_detection', {})
        patterns = tox_config.get('trade_patterns', {})
        block_threshold = tox_config.get('block_threshold', 10000)
        sweep_window_ms = tox_config.get('sweep_window_ms', 250)
        
        # Odd lot ratio (retail indicator)
        odd_lots = sum(1 for t in trades if t.get('size', 0) % 100 != 0)
        odd_ratio = odd_lots / len(trades) if trades else 0
        
        # Sweep ratio (aggressive/toxic)
        sweeps = self._detect_sweeps(trades, sweep_window_ms)
        sweep_ratio = sweeps / max(1, len(trades) // 3)  # Normalize by potential sweep groups
        
        # Block ratio (institutional)
        blocks = sum(1 for t in trades if t.get('size', 0) >= block_threshold)
        block_ratio = blocks / len(trades) if trades else 0
        
        # Weight and combine
        toxicity = (odd_ratio * patterns.get('odd_lot_weight', 0.7) +
                   sweep_ratio * patterns.get('sweep_weight', 0.9) +
                   block_ratio * patterns.get('block_weight', -0.5))
        
        return max(0.0, min(1.0, toxicity))
    
    def _book_imbalance_volatility(self, books: list) -> float:
        """Calculate volatility of order book imbalance (manipulation indicator)."""
        if not books:
            return 0.0
            
        imbalances = []
        for book in books:
            bid_volume = sum(b.get('size', 0) for b in book.get('bids', [])[:5]) or 1
            ask_volume = sum(a.get('size', 0) for a in book.get('asks', [])[:5]) or 1
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            imbalances.append(imbalance)
            
        return float(np.std(imbalances)) if len(imbalances) > 1 else 0.0
    
    async def analyze_market_makers(self) -> dict:
        """Backward compatibility wrapper - now calls analyze_flow_toxicity."""
        return await self.analyze_flow_toxicity()
    
    async def analyze_flow_toxicity(self) -> dict:
        """
        Analyze flow toxicity using venue distribution, trade patterns, and VPIN.
        Returns comprehensive toxicity metrics for each Level 2 symbol.
        """
        from collections import Counter
        
        tox_config = self.discovery_config.get('toxicity_detection', {})
        vpin_thresholds = tox_config.get('vpin_thresholds', {'toxic': 0.7, 'informed': 0.3})
        
        self.logger.info("Analyzing venue & pattern toxicity...")
        
        toxicity_results = {}
        
        # Analyze each Level 2 symbol
        for symbol in self.level2_symbols:
            try:
                # 1. Get current VPIN if available
                vpin_str = await self.redis.get(f'vpin:{symbol}')
                vpin = float(vpin_str) if vpin_str else 0.5
                
                # 2. Analyze venue distribution from trades
                trades_json = await self.redis.lrange(f'market:{symbol}:trades', 0, 999)
                trades = []
                venue_counts = Counter()
                
                for trade_str in trades_json:
                    try:
                        trade = json.loads(trade_str)
                        trades.append(trade)
                        # Extract venue from trade or book
                        venue = trade.get('venue', 'UNKNOWN')
                        venue_counts[venue] += 1
                    except:
                        continue
                
                # If no venue info in trades, try to get from order book
                if not venue_counts:
                    book_json = await self.redis.get(f'market:{symbol}:book')
                    if book_json:
                        book = json.loads(book_json)
                        # Extract venues from venue field (or mm for backward compat)
                        for side in ['bids', 'asks']:
                            for level in book.get(side, [])[:10]:  # Sample top 10 levels
                                venue = level.get('venue') or level.get('mm', '')
                                venue = str(venue).strip()
                                if venue and venue != 'UNKNOWN':
                                    size = level.get('size', 1)
                                    venue_counts[venue] += size  # Weight by size
                
                # 3. Calculate venue toxicity
                venue_tox = self._venue_toxicity_from_config(dict(venue_counts))
                
                # 4. Calculate trade pattern toxicity
                pattern_tox = self._trade_pattern_toxicity(trades)
                
                # 5. Calculate book imbalance volatility (if we have book history)
                book_tox = 0.5  # Default neutral
                # TODO: Store book snapshots for volatility calculation
                
                # 6. Blend toxicity scores (weights can be configured)
                overall_toxicity = (
                    0.50 * vpin +           # VPIN is primary signal
                    0.25 * venue_tox +      # Venue mix
                    0.20 * pattern_tox +    # Trade patterns
                    0.05 * book_tox         # Book volatility
                )
                
                # 7. Classify based on VPIN thresholds
                if vpin >= vpin_thresholds['toxic']:
                    label = 'toxic'
                elif vpin <= vpin_thresholds['informed']:
                    label = 'informed'
                else:
                    label = 'neutral'
                
                # Store result
                toxicity_results[symbol] = {
                    'toxicity': round(overall_toxicity, 3),
                    'vpin': round(vpin, 3),
                    'venue_tox': round(venue_tox, 3),
                    'pattern_tox': round(pattern_tox, 3),
                    'book_tox': round(book_tox, 3),
                    'venue_mix': dict(venue_counts),
                    'label': label
                }
                
                # Log the result
                self.logger.info(f"  {symbol}: tox={overall_toxicity:.3f} (vpin={vpin:.3f}, "
                               f"venue={venue_tox:.3f}, pattern={pattern_tox:.3f}) "
                               f"mix={dict(list(venue_counts.most_common(3)))}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing toxicity for {symbol}: {e}")
                continue
        
        self.logger.info(f"Analyzed flow toxicity for {len(toxicity_results)} symbols")
        
        return toxicity_results
    
    async def discover_volatility_regimes(self) -> dict:
        """
        Identify current market volatility regime.
        CRITICAL: Uses CORRECTED annualization factor (4,680 bars per day).
        """
        vol_config = self.discovery_config.get('volatility', {})
        bars_per_day = vol_config.get('bars_per_day', 4680)  # CORRECTED: 6.5 hours * 60 min * 12 bars/min
        trading_days = vol_config.get('trading_days_per_year', 252)
        window = vol_config.get('window_bars', 20)
        min_bars = vol_config.get('min_bars_for_regime', 500)
        default_regime = vol_config.get('default_regime', 'NORMAL')
        low_pct = vol_config.get('low_percentile', 33)
        high_pct = vol_config.get('high_percentile', 67)
        
        self.logger.info("Discovering volatility regimes...")
        self.logger.info(f"  Using annualization factor: sqrt({bars_per_day} * {trading_days}) = {np.sqrt(bars_per_day * trading_days):.1f}")
        
        # Check data freshness
        if not await self._check_data_freshness('SPY'):
            self.logger.warning("Using default regime due to stale data")
            return {
                'current': default_regime,
                'low_threshold': 0.10,
                'high_threshold': 0.30,
                'realized_vol': None,
                'data_status': 'stale'
            }
        
        # Fetch SPY bars (stored as list)
        bars_list = await self.redis.lrange('market:SPY:bars', 0, -1)
        if not bars_list:
            self.logger.warning("No bar data for SPY")
            return {
                'current': default_regime,
                'error': 'No bar data'
            }
        
        bars = [json.loads(bar) for bar in bars_list]
        
        # Check minimum bars requirement
        if len(bars) < min_bars:
            self.logger.warning(f"Insufficient bars for regime: {len(bars)} < {min_bars}")
            return {
                'current': default_regime,
                'low_threshold': 0.10,
                'high_threshold': 0.30,
                'data_status': 'insufficient'
            }
        
        # Calculate log returns
        closes = [float(bar['close']) for bar in bars]
        log_returns = np.diff(np.log(closes))
        
        # Calculate rolling volatility series
        vol_series = []
        for i in range(window, len(log_returns)):
            window_returns = log_returns[i-window:i]
            # Calculate realized volatility
            realized_vol = np.std(window_returns)
            # CORRECTED ANNUALIZATION: sqrt(4680 * 252) not sqrt(78 * 252)
            annualized_vol = realized_vol * np.sqrt(bars_per_day * trading_days)
            vol_series.append(annualized_vol)
        
        if not vol_series:
            self.logger.warning("Could not calculate volatility series")
            return {
                'current': default_regime,
                'error': 'Insufficient data for volatility calculation'
            }
        
        # Calculate regime thresholds from historical distribution
        low_threshold = np.percentile(vol_series, low_pct)
        high_threshold = np.percentile(vol_series, high_pct)
        current_vol = vol_series[-1]
        
        # Classify current regime
        if current_vol < low_threshold:
            regime = 'LOW'
        elif current_vol > high_threshold:
            regime = 'HIGH'
        else:
            regime = 'NORMAL'
        
        self.logger.info(f"Discovered volatility regime: {regime}")
        self.logger.info(f"  Current vol: {current_vol:.2%}")
        self.logger.info(f"  Low threshold: {low_threshold:.2%}")
        self.logger.info(f"  High threshold: {high_threshold:.2%}")
        
        return {
            'current': regime,
            'low_threshold': round(low_threshold, 4),
            'high_threshold': round(high_threshold, 4),
            'realized_vol': round(current_vol, 4),
            'annualization_factor': int(np.sqrt(bars_per_day * trading_days)),
            'percentiles': {
                '10': round(np.percentile(vol_series, 10), 4),
                '25': round(np.percentile(vol_series, 25), 4),
                '50': round(np.percentile(vol_series, 50), 4),
                '75': round(np.percentile(vol_series, 75), 4),
                '90': round(np.percentile(vol_series, 90), 4)
            }
        }
    
    async def calculate_correlations(self) -> dict:
        """
        Calculate correlation matrix between symbols.
        Uses inner join for alignment and forward fill for missing data.
        """
        corr_config = self.discovery_config.get('correlation', {})
        min_bars = corr_config.get('min_bars', 10)  # Lowered for startup
        window = corr_config.get('calculation_window', 50)  # Use available bars
        fill_limit = corr_config.get('forward_fill_limit', 5)
        min_symbols = corr_config.get('min_correlation_symbols', 3)
        
        self.logger.info("Calculating correlation matrix...")
        
        returns_data = {}
        
        # Collect returns for each symbol
        for symbol in self.symbols:
            bars_list = await self.redis.lrange(f'market:{symbol}:bars', 0, -1)
            if not bars_list:
                self.logger.debug(f"No bar data for {symbol}")
                continue
            
            try:
                bars = [json.loads(bar) for bar in bars_list]
                
                # Skip if insufficient data
                if len(bars) < min_bars:
                    self.logger.warning(f"Skipping {symbol}: only {len(bars)} bars")
                    continue
                
                # Calculate log returns
                closes = [float(bar['close']) for bar in bars[-window:]]
                if len(closes) > 1:
                    log_returns = np.diff(np.log(closes))
                    returns_data[symbol] = log_returns
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Check minimum symbols requirement
        if len(returns_data) < min_symbols:
            self.logger.warning(f"Insufficient symbols for correlation: {len(returns_data)} < {min_symbols}")
            return {}
        
        # Create aligned DataFrame using inner join
        # Find minimum length for alignment
        min_length = min(len(returns) for returns in returns_data.values())
        
        if min_length < 2:
            self.logger.warning("Insufficient data for correlation calculation")
            return {}
        
        # Align all series to same length (most recent data)
        aligned_data = {}
        for symbol, returns in returns_data.items():
            aligned_data[symbol] = returns[-min_length:]
        
        # Create DataFrame
        df = pd.DataFrame(aligned_data)
        
        # Forward fill missing values with limit
        df = df.ffill(limit=fill_limit)
        
        # Drop any remaining NaN values
        df = df.dropna()
        
        if len(df) < 2:
            self.logger.warning("Insufficient valid data after cleaning")
            return {}
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Convert to nested dictionary with rounding
        # Handle potential complex values from correlation calculation
        import math
        
        correlations = {}
        for symbol1 in corr_matrix.index:
            correlations[symbol1] = {}
            for symbol2 in corr_matrix.columns:
                val = corr_matrix.loc[symbol1, symbol2]
                # Handle complex numbers or NaN
                try:
                    # Handle pandas/numpy scalar types
                    if isinstance(val, (complex, np.complexfloating)):
                        # Complex number - take real part
                        numeric_val = float(val.real)
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        numeric_val = float(val)
                    elif hasattr(val, 'item'):
                        # Handle numpy scalars
                        item_val = val.item()
                        if isinstance(item_val, complex):
                            numeric_val = float(item_val.real)
                        else:
                            numeric_val = float(item_val)
                    else:
                        # Try direct conversion as last resort
                        numeric_val = float(val)
                    
                    # Check for finite values
                    if math.isfinite(numeric_val):
                        correlations[symbol1][symbol2] = round(numeric_val, 3)
                    else:
                        correlations[symbol1][symbol2] = 0.0
                except (TypeError, ValueError):
                    correlations[symbol1][symbol2] = 0.0
        
        self.logger.info(f"Calculated correlations for {len(correlations)} symbols")
        if correlations:
            # Log sample correlations
            symbols = list(correlations.keys())
            if len(symbols) >= 2:
                sample_corr = correlations[symbols[0]][symbols[1]]
                self.logger.info(f"  Sample: {symbols[0]}-{symbols[1]} correlation = {sample_corr}")
        
        return correlations
    
    async def store_to_redis(self):
        """
        Store all discovered parameters in Redis with configured TTL.
        """
        ttl = self.discovery_config.get('ttl_seconds', 86400)  # 24 hours default
        
        # Store VPIN bucket size
        if 'vpin_bucket_size' in self.discovered_params:
            await self.redis.setex('discovered:vpin_bucket_size', ttl, 
                           self.discovered_params['vpin_bucket_size'])
            self.logger.info(f"Stored discovered:vpin_bucket_size with {ttl}s TTL")
        
        # Store lookback bars
        if 'lookback_bars' in self.discovered_params:
            await self.redis.setex('discovered:lookback_bars', ttl,
                           self.discovered_params['lookback_bars'])
            self.logger.info(f"Stored discovered:lookback_bars with {ttl}s TTL")
        
        # Store flow toxicity analysis
        if 'flow_toxicity' in self.discovered_params:
            await self.redis.setex('discovered:flow_toxicity', ttl,
                           json.dumps(self.discovered_params['flow_toxicity']))
            self.logger.info(f"Stored discovered:flow_toxicity with {ttl}s TTL")
        
        # Store volatility regimes
        if 'vol_regimes' in self.discovered_params:
            await self.redis.setex('discovered:vol_regimes', ttl,
                           json.dumps(self.discovered_params['vol_regimes']))
            self.logger.info(f"Stored discovered:vol_regimes with {ttl}s TTL")
        
        # Store correlation matrix
        if 'correlation_matrix' in self.discovered_params:
            await self.redis.setex('discovered:correlation_matrix', ttl,
                           json.dumps(self.discovered_params['correlation_matrix']))
            self.logger.info(f"Stored discovered:correlation_matrix with {ttl}s TTL")
    
    async def generate_config_file(self):
        """
        Generate discovered parameters config file.
        Saves to config/discovered.yaml with timestamp and documentation.
        """
        from pathlib import Path
        
        # Prepare output data
        output = {
            'generated_at': datetime.now().isoformat(),
            'parameters': {}
        }
        
        # Add VPIN parameters
        if 'vpin_bucket_size' in self.discovered_params:
            output['parameters']['vpin'] = {
                'bucket_size': self.discovered_params['vpin_bucket_size'],
                'description': 'Optimal VPIN bucket size in shares'
            }
        
        # Add temporal parameters
        if 'lookback_bars' in self.discovered_params:
            output['parameters']['temporal'] = {
                'lookback_bars': self.discovered_params['lookback_bars'],
                'description': 'Optimal lookback period for time series analysis'
            }
        
        # Add flow toxicity results
        if 'flow_toxicity' in self.discovered_params:
            output['parameters']['flow_toxicity'] = {
                'analysis': self.discovered_params['flow_toxicity'],
                'description': 'Venue and pattern-based flow toxicity analysis'
            }
        
        # Add volatility regime
        if 'vol_regimes' in self.discovered_params:
            output['parameters']['volatility'] = {
                'regime': self.discovered_params['vol_regimes'],
                'description': 'Current market volatility regime and thresholds'
            }
        
        # Add correlations
        if 'correlation_matrix' in self.discovered_params:
            output['parameters']['correlations'] = {
                'matrix': self.discovered_params['correlation_matrix'],
                'description': 'Symbol correlation matrix'
            }
        
        # Helper function to convert numpy types to Python built-ins
        def _to_builtin(x):
            if isinstance(x, np.generic):
                return x.item()
            return x
        
        # Clean the output to remove numpy tags
        clean_output = json.loads(json.dumps(output, default=_to_builtin))
        
        # Write to file
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        
        discovered_file = config_dir / 'discovered.yaml'
        
        with open(discovered_file, 'w') as f:
            # Add header comment
            f.write("# Discovered Trading Parameters\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# These parameters were empirically discovered from market data\n\n")
            
            # Write YAML - use safe_dump to avoid Python tags
            yaml.safe_dump(clean_output, f, sort_keys=False)
        
        self.logger.info(f"Generated config file: {discovered_file}")

class MetricsAggregator:
    """
    Aggregate metrics across symbols for portfolio-level analytics.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize metrics aggregator.
        
        TODO: Set up Redis connection
        TODO: Define aggregation rules from config
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
    
    async def calculate_portfolio_metrics(self) -> dict:
        """
        Calculate portfolio-wide metrics by aggregating across all symbols.
        
        Calculates:
        - Weighted average VPIN (by volume)
        - Total GEX/DEX exposure
        - Portfolio volatility
        - Correlation risk
        - Market regime consensus
        
        Returns:
            Portfolio-level metrics dictionary
        """
        try:
            # Get all symbols from config
            symbols = self.config.get('symbols', {})
            all_symbols = list(set(symbols.get('level2', []) + symbols.get('standard', [])))
            
            # 1. Collect VPIN and volume for weighted average
            vpins = []
            volumes = []
            vpin_symbols = []
            
            for symbol in all_symbols:
                # Get VPIN
                vpin_json = await self.redis.get(f'metrics:{symbol}:vpin')
                # Get volume from ticker
                ticker_json = await self.redis.get(f'market:{symbol}:ticker')
                
                if vpin_json and ticker_json:
                    vpin_data = json.loads(vpin_json)
                    ticker_data = json.loads(ticker_json)
                    
                    vpins.append(vpin_data.get('value', 0.5))
                    volumes.append(ticker_data.get('volume', 0))
                    vpin_symbols.append(symbol)
            
            # Calculate weighted average VPIN
            weighted_vpin = 0.5  # Default neutral
            if volumes and sum(volumes) > 0:
                weights = [v/sum(volumes) for v in volumes]
                weighted_vpin = sum(v*w for v,w in zip(vpins, weights))
            
            # 2. Aggregate GEX/DEX across portfolio
            total_gex = 0
            total_dex = 0
            gex_symbols = []
            
            for symbol in all_symbols:
                # Get GEX
                gex_json = await self.redis.get(f'metrics:{symbol}:gex')
                if gex_json:
                    gex_data = json.loads(gex_json)
                    total_gex += gex_data.get('total_gex', 0)
                    gex_symbols.append(symbol)
                
                # Get DEX
                dex_json = await self.redis.get(f'metrics:{symbol}:dex')
                if dex_json:
                    dex_data = json.loads(dex_json)
                    total_dex += dex_data.get('total_dex', 0)
            
            # 3. Calculate portfolio risk metrics
            # Get correlation matrix
            corr_json = await self.redis.get('discovered:correlation_matrix')
            max_correlation = 0
            avg_correlation = 0
            
            if corr_json:
                corr_matrix = json.loads(corr_json)
                correlations = []
                
                # Extract all pairwise correlations
                for sym1 in corr_matrix:
                    for sym2 in corr_matrix[sym1]:
                        if sym1 != sym2:
                            correlations.append(abs(corr_matrix[sym1][sym2]))
                
                if correlations:
                    max_correlation = max(correlations)
                    avg_correlation = np.mean(correlations)
            
            # 4. Get volatility regime
            vol_regime_json = await self.redis.get('discovered:vol_regimes')
            vol_regime = 'NORMAL'
            realized_vol = 0
            
            if vol_regime_json:
                vol_data = json.loads(vol_regime_json)
                vol_regime = vol_data.get('current', 'NORMAL')
                realized_vol = vol_data.get('realized_vol', 0)
            
            # 5. Calculate market regime consensus
            # Based on VPIN, GEX, and volatility
            if weighted_vpin > 0.7:
                toxicity = 'high'
            elif weighted_vpin < 0.3:
                toxicity = 'low'
            else:
                toxicity = 'moderate'
            
            if total_gex > 0:
                stability = 'stabilizing'
            else:
                stability = 'destabilizing'
            
            # Determine overall regime
            if toxicity == 'high' and stability == 'destabilizing':
                market_regime = 'high_risk'
            elif toxicity == 'low' and stability == 'stabilizing':
                market_regime = 'low_risk'
            else:
                market_regime = 'moderate_risk'
            
            # 6. Prepare result
            result = {
                'weighted_vpin': round(weighted_vpin, 4),
                'vpin_symbols_count': len(vpin_symbols),
                'total_gex': round(total_gex, 0),
                'total_dex': round(total_dex, 0),
                'gex_regime': 'stabilizing' if total_gex > 0 else 'destabilizing',
                'dex_bias': 'bullish' if total_dex > 0 else 'bearish',
                'max_correlation': round(max_correlation, 3),
                'avg_correlation': round(avg_correlation, 3),
                'vol_regime': vol_regime,
                'realized_vol': round(realized_vol, 4),
                'toxicity': toxicity,
                'stability': stability,
                'market_regime': market_regime,
                'timestamp': time.time()
            }
            
            # 7. Store in Redis with configured TTL
            ttl = self.config['modules']['data_ingestion']['store_ttls'].get('metrics', 60)
            
            await self.redis.setex(
                'metrics:portfolio:summary',
                ttl,
                json.dumps(result)
            )
            
            # Store individual metrics too
            await self.redis.setex(
                'metrics:portfolio:vpin',
                ttl,
                json.dumps({'value': weighted_vpin, 'timestamp': time.time()})
            )
            
            await self.redis.setex(
                'metrics:portfolio:gex_total',
                ttl,
                json.dumps({'value': total_gex, 'timestamp': time.time()})
            )
            
            await self.redis.setex(
                'metrics:portfolio:risk',
                ttl,
                json.dumps({
                    'max_correlation': max_correlation,
                    'avg_correlation': avg_correlation,
                    'market_regime': market_regime,
                    'timestamp': time.time()
                })
            )
            
            # Log significant portfolio states
            if market_regime == 'high_risk':
                self.logger.warning(f"HIGH RISK portfolio state: VPIN={weighted_vpin:.3f}, GEX={total_gex:.0f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            self.logger.error(traceback.format_exc())
            return {}
    
    async def calculate_sector_flows(self) -> dict:
        """
        Analyze sector-level flow patterns and rotation.
        
        Groups symbols by sector and analyzes:
        - Net flows per sector
        - Sector rotation signals
        - Risk-on vs risk-off sentiment
        
        Returns:
            Sector flow analysis dictionary
        """
        try:
            # Define sector groupings
            sectors = {
                'technology': ['AAPL', 'NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
                'broad_market': ['SPY', 'QQQ', 'IWM'],
                'consumer': ['AMZN', 'TSLA'],
                'volatility': ['VXX']
            }
            
            sector_metrics = {}
            
            # 1. Calculate metrics for each sector
            for sector_name, symbols in sectors.items():
                sector_data = {
                    'symbols': symbols,
                    'avg_vpin': 0,
                    'total_volume': 0,
                    'net_flow': 0,
                    'price_change': 0,
                    'momentum': 0
                }
                
                vpins = []
                volumes = []
                price_changes = []
                
                for symbol in symbols:
                    # Get VPIN
                    vpin_json = await self.redis.get(f'metrics:{symbol}:vpin')
                    if vpin_json:
                        vpin_data = json.loads(vpin_json)
                        vpins.append(vpin_data.get('value', 0.5))
                    
                    # Get ticker data
                    ticker_json = await self.redis.get(f'market:{symbol}:ticker')
                    if ticker_json:
                        ticker = json.loads(ticker_json)
                        volume = ticker.get('volume', 0)
                        volumes.append(volume)
                        
                        # Calculate price change (if we have previous close)
                        last = ticker.get('last')
                        if last is not None and last > 0:
                            # For simplicity, use a small change threshold
                            # In production, you'd compare to previous close
                            price_changes.append(0)  # Placeholder
                    
                    # Get OBI for flow direction
                    obi_json = await self.redis.get(f'metrics:{symbol}:obi')
                    if obi_json:
                        obi = json.loads(obi_json)
                        imbalance = obi.get('level5_imbalance', 0)
                        # Net flow approximation: volume * imbalance
                        if ticker_json:
                            ticker = json.loads(ticker_json)
                            volume = ticker.get('volume', 0)
                            sector_data['net_flow'] += volume * imbalance
                
                # Calculate sector averages
                if vpins:
                    sector_data['avg_vpin'] = round(np.mean(vpins), 4)
                
                if volumes:
                    sector_data['total_volume'] = sum(volumes)
                
                if price_changes:
                    sector_data['price_change'] = round(np.mean(price_changes), 3)
                
                # Determine sector momentum
                if sector_data['net_flow'] > 0:
                    sector_data['momentum'] = 'inflow'
                elif sector_data['net_flow'] < 0:
                    sector_data['momentum'] = 'outflow'
                else:
                    sector_data['momentum'] = 'neutral'
                
                sector_metrics[sector_name] = sector_data
            
            # 2. Detect sector rotation
            rotation_signals = []
            
            # Tech vs broad market rotation
            if 'technology' in sector_metrics and 'broad_market' in sector_metrics:
                tech_vpin = sector_metrics['technology']['avg_vpin']
                market_vpin = sector_metrics['broad_market']['avg_vpin']
                
                if tech_vpin > market_vpin + 0.1:
                    rotation_signals.append('rotate_out_of_tech')
                elif tech_vpin < market_vpin - 0.1:
                    rotation_signals.append('rotate_into_tech')
            
            # 3. Determine risk sentiment
            risk_sentiment = 'neutral'
            
            if 'volatility' in sector_metrics:
                vxx_vpin = sector_metrics['volatility']['avg_vpin']
                if vxx_vpin > 0.6:
                    risk_sentiment = 'risk_off'
                elif vxx_vpin < 0.4:
                    risk_sentiment = 'risk_on'
            
            # 4. Find leading/lagging sectors
            sorted_sectors = sorted(
                sector_metrics.items(),
                key=lambda x: x[1]['net_flow'],
                reverse=True
            )
            
            leading_sectors = [s[0] for s in sorted_sectors[:2]]
            lagging_sectors = [s[0] for s in sorted_sectors[-2:]]
            
            # 5. Prepare result
            result = {
                'sectors': sector_metrics,
                'rotation_signals': rotation_signals,
                'risk_sentiment': risk_sentiment,
                'leading_sectors': leading_sectors,
                'lagging_sectors': lagging_sectors,
                'timestamp': time.time()
            }
            
            # 6. Store in Redis
            ttl = self.config['modules']['data_ingestion']['store_ttls'].get('metrics', 60)
            
            await self.redis.setex(
                'metrics:portfolio:sectors',
                ttl,
                json.dumps(result)
            )
            
            # Log significant rotations
            if rotation_signals:
                self.logger.info(f"Sector rotation detected: {rotation_signals}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating sector flows: {e}")
            self.logger.error(traceback.format_exc())
            return {}
class AnalyticsEngine:
    """
    Real-time analytics calculation engine.
    Day 4: Basic TOB metrics only.
    Day 5: Full VPIN, OBI, and advanced analytics.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize analytics engine with async Redis."""
        self.config = config
        self.redis = redis_conn  # Async Redis client
        self.logger = logging.getLogger(__name__)
        
        # Build symbol list from config
        level2 = config.get('symbols', {}).get('level2', [])
        standard = config.get('symbols', {}).get('standard', [])
        self.symbols = level2 + standard
        
        # Configuration
        self.cadence_hz = config['modules']['analytics'].get('cadence_hz', 2)
        self.sleep_interval = 1.0 / self.cadence_hz
        
        # Use same TTLs as data_ingestion
        self.ttls = config['modules']['data_ingestion']['store_ttls']
        self.output_ttl = self.ttls.get('metrics', 10)
        
        # Market hours
        self.market_tz = pytz.timezone('US/Eastern')
        self.analytics_rth_only = config['modules']['analytics'].get('analytics_rth_only', False)
        
        # Metrics tracking
        self.last_calculation_time = {}
        self.calculation_count = 0
        self.calc_start_times = {}  # Track start times for performance metrics
        
    async def start(self):
        """
        Start the analytics calculation loop with connection pooling optimization.
        
        Calculation frequencies:
        - TOB metrics: Every cycle (2Hz)
        - Order book imbalance: Every 2 seconds (1Hz)
        - VPIN: Every 5 seconds
        - Hidden orders: Every 5 seconds
        - GEX/DEX: Every 10 seconds
        - Multi-timeframe: Every 10 seconds
        - Portfolio aggregation: Every 10 seconds
        """
        self.logger.info(f"Starting Analytics Engine (cadence: {self.cadence_hz} Hz)")
        self.logger.info("Day 5 analytics fully enabled: VPIN, GEX/DEX, OBI, Hidden Orders, MTF")
        
        # Initialize metrics aggregator for portfolio calculations
        self.aggregator = MetricsAggregator(self.config, self.redis)
        
        # Set maximum concurrent tasks to prevent connection pool exhaustion
        MAX_CONCURRENT_TASKS = 10  # Limit concurrent Redis operations
        
        while await self._should_run():
            try:
                cycle_start = time.time()
                
                # 1. Always calculate TOB metrics (every cycle - 2Hz)
                # Batch these in smaller groups to avoid connection exhaustion
                tob_tasks = []
                for symbol in self.symbols:
                    tob_tasks.append(self._calculate_tob_metrics(symbol))
                
                # Execute TOB in batches
                for i in range(0, len(tob_tasks), MAX_CONCURRENT_TASKS):
                    batch = tob_tasks[i:i + MAX_CONCURRENT_TASKS]
                    await asyncio.gather(*batch, return_exceptions=True)
                
                # 2. Order book imbalance (every 2 seconds - 1Hz)
                if self.calculation_count % 4 == 0:
                    obi_tasks = []
                    for symbol in self.symbols:
                        obi_tasks.append(self.calculate_order_book_imbalance(symbol))
                    
                    # Execute in batches
                    for i in range(0, len(obi_tasks), MAX_CONCURRENT_TASKS):
                        batch = obi_tasks[i:i + MAX_CONCURRENT_TASKS]
                        await asyncio.gather(*batch, return_exceptions=True)
                
                # 3. VPIN and hidden order detection (every 5 seconds)
                if self.calculation_count % 10 == 0:
                    analysis_tasks = []
                    for symbol in self.symbols:
                        analysis_tasks.append(self.calculate_vpin(symbol))
                        analysis_tasks.append(self.detect_hidden_orders(symbol))
                    
                    # Execute in batches
                    for i in range(0, len(analysis_tasks), MAX_CONCURRENT_TASKS):
                        batch = analysis_tasks[i:i + MAX_CONCURRENT_TASKS]
                        results = await asyncio.gather(*batch, return_exceptions=True)
                        
                        # Log any errors
                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.error(f"Analysis task error: {result}")
                
                # 4. GEX/DEX and multi-timeframe (every 10 seconds)
                if self.calculation_count % 20 == 0:
                    advanced_tasks = []
                    
                    # Calculate GEX/DEX for all Level 2 symbols (they have options)
                    level2_symbols = self.config.get('symbols', {}).get('level2', [])
                    for symbol in level2_symbols:
                        advanced_tasks.append(self.calculate_gex(symbol))
                        advanced_tasks.append(self.calculate_dex(symbol))
                    
                    # Multi-timeframe for all tracked symbols
                    for symbol in self.symbols:
                        advanced_tasks.append(self.calculate_multi_timeframe_metrics(symbol))
                    
                    # Execute in smaller batches for these heavy calculations
                    for i in range(0, len(advanced_tasks), MAX_CONCURRENT_TASKS // 2):
                        batch = advanced_tasks[i:i + MAX_CONCURRENT_TASKS // 2]
                        results = await asyncio.gather(*batch, return_exceptions=True)
                        
                        # Log any errors or successful results
                        for j, result in enumerate(results):
                            if isinstance(result, Exception):
                                # Try to identify which task failed
                                task_idx = i + j
                                if task_idx < len(level2_symbols) * 2:
                                    sym_idx = task_idx // 2
                                    task_type = "GEX" if task_idx % 2 == 0 else "DEX"
                                    failed_symbol = level2_symbols[sym_idx] if sym_idx < len(level2_symbols) else "UNKNOWN"
                                    self.logger.error(f"Advanced task error - {task_type} for {failed_symbol}: {result}")
                                else:
                                    self.logger.error(f"Advanced task error: {result}")
                            elif isinstance(result, dict) and 'error' in result:
                                # Task returned but with an error dict - silent unless debugging
                                pass
                
                # 5. Portfolio aggregation (every 10 seconds) - run sequentially
                if self.calculation_count % 20 == 0:
                    try:
                        # Add timeout to prevent hanging
                        await asyncio.wait_for(
                            self.aggregator.calculate_portfolio_metrics(),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.error("Portfolio metrics calculation timed out")
                    except Exception as e:
                        self.logger.error(f"Portfolio metrics error: {e}")
                        self.logger.error(traceback.format_exc())
                    
                    try:
                        await asyncio.wait_for(
                            self.aggregator.calculate_sector_flows(),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.error("Sector flows calculation timed out")
                    except Exception as e:
                        self.logger.error(f"Sector flows error: {e}")
                        self.logger.error(traceback.format_exc())
                
                # Update heartbeat
                await self._update_heartbeat()
                
                # Log performance metrics periodically
                if self.calculation_count % 100 == 0:
                    await self._log_performance_metrics()
                
                # Update calculation counter
                self.calculation_count += 1
                
                # Maintain cadence
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.sleep_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    # Log if we're falling behind
                    if self.calculation_count % 100 == 0:
                        self.logger.warning(f"Analytics falling behind: {elapsed:.3f}s > {self.sleep_interval:.3f}s")
                        try:
                            # Store performance warning in Redis
                            await self.redis.setex(
                                'monitoring:analytics:performance_warning',
                                60,
                                json.dumps({
                                    'elapsed': elapsed,
                                    'expected': self.sleep_interval,
                                    'timestamp': time.time()
                                })
                            )
                        except:
                            pass  # Don't fail if we can't log to Redis
                
            except Exception as e:
                self.logger.error(f"Analytics engine error: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(1)
    
    async def _should_run(self):
        """Check if analytics should be running."""
        # Check system halt flag
        halt = await self.redis.get('system:halt')
        if halt == '1':
            return False
        
        # Check market hours if configured
        if self.analytics_rth_only:
            now = datetime.now(self.market_tz)
            if now.weekday() >= 5:  # Weekend
                return False
            current_time = now.time()
            if not (datetime_time(9, 30) <= current_time <= datetime_time(16, 0)):
                return False
        
        return True
    
    async def _calculate_tob_metrics(self, symbol: str):
        """Calculate top-of-book metrics with exchange info."""
        try:
            # Get ticker data
            ticker_json = await self.redis.get(f'market:{symbol}:ticker')
            
            if not ticker_json:
                return
            
            ticker = json.loads(ticker_json)
            
            # Build metrics
            metrics = {
                'symbol': symbol,
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'last': ticker.get('last'),
                'mid': ticker.get('mid'),
                'spread': ticker.get('spread'),
                'spread_bps': ticker.get('spread_bps'),
                'volume': ticker.get('volume'),
                # Include exchange info for venue verification
                'bid_exchange': ticker.get('bid_exchange', 'UNKNOWN'),
                'ask_exchange': ticker.get('ask_exchange', 'UNKNOWN'),
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
            # Store metrics
            await self.redis.setex(
                f'metrics:{symbol}:tob',
                self.output_ttl,
                json.dumps(metrics)
            )
            
            # Track calculation time
            if symbol not in self.calc_start_times:
                self.calc_start_times[symbol] = time.time()
            calc_duration = time.time() - self.calc_start_times[symbol]
            self.last_calculation_time[symbol] = calc_duration
            self.calc_start_times[symbol] = time.time()  # Reset for next cycle
            
        except Exception as e:
            self.logger.error(f"Error calculating TOB metrics for {symbol}: {e}")
    
    async def _update_heartbeat(self):
        """Update analytics heartbeat."""
        try:
            heartbeat = {
                'ts': int(datetime.now().timestamp() * 1000),
                'count': self.calculation_count,
                'symbols': len(self.symbols)
            }
            
            ttl = self.ttls.get('heartbeat', 15)
            await self.redis.setex('hb:analytics', ttl, json.dumps(heartbeat))
                
        except Exception as e:
            self.logger.error(f"Error updating heartbeat: {e}")
    
    async def _log_performance_metrics(self):
        """Log and store performance metrics for monitoring."""
        try:
            # Calculate average calculation times
            if self.last_calculation_time:
                avg_time = np.mean(list(self.last_calculation_time.values()))
                max_time = max(self.last_calculation_time.values())
                min_time = min(self.last_calculation_time.values())
                
                metrics = {
                    'calculations_performed': self.calculation_count,
                    'symbols_tracked': len(self.symbols),
                    'avg_calc_time': round(avg_time, 3),
                    'max_calc_time': round(max_time, 3),
                    'min_calc_time': round(min_time, 3),
                    'timestamp': time.time()
                }
                
                # Store in Redis for monitoring
                ttl = self.ttls.get('monitoring', 60)
                await self.redis.setex(
                    'monitoring:analytics:performance',
                    ttl,
                    json.dumps(metrics)
                )
                
                self.logger.info(f"Performance: {self.calculation_count} calcs, avg={avg_time:.3f}s")
                
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {e}")
    
    async def stop(self):
        """Stop the analytics engine."""
        self.logger.info("Stopping Analytics Engine")
    
    async def _classify_trades(self, trades: List[dict], symbol: str) -> List[dict]:
        """
        Classify trades as buy or sell using Lee-Ready algorithm.
        
        Lee-Ready Algorithm:
        1. If trade price > midpoint: BUY
        2. If trade price < midpoint: SELL
        3. If trade price = midpoint: Use tick test
           - If price > previous price: BUY
           - If price < previous price: SELL
           - If price = previous price: Use previous classification
        
        Returns trades with 'side' field added ('buy' or 'sell')
        """
        if not trades:
            return []
        
        classified = []
        prev_price = None
        prev_side = 'buy'  # Default assumption
        
        # Get current bid/ask for midpoint calculation
        ticker_json = await self.redis.get(f'market:{symbol}:ticker')
        if ticker_json:
            ticker = json.loads(ticker_json)
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            midpoint = (bid + ask) / 2 if bid and ask else 0
        else:
            midpoint = 0
        
        # If no midpoint, try to estimate from trade prices
        if midpoint == 0 and trades:
            prices = [t.get('price', 0) for t in trades if t.get('price', 0) > 0]
            if prices:
                # Use median as proxy for midpoint
                midpoint = np.median(prices)
        
        buy_count = 0
        sell_count = 0
        
        for i, trade in enumerate(trades):
            price = trade.get('price', 0)
            
            # Lee-Ready classification
            if midpoint > 0:
                # Add small tolerance for floating point comparison
                tolerance = 0.001 * midpoint  
                
                if price > midpoint + tolerance:
                    side = 'buy'
                elif price < midpoint - tolerance:
                    side = 'sell'
                else:
                    # Price at midpoint - use tick test
                    if prev_price is not None:
                        if price > prev_price:
                            side = 'buy'
                        elif price < prev_price:
                            side = 'sell'
                        else:
                            # Alternate between buy/sell to avoid all same classification
                            side = 'sell' if prev_side == 'buy' else 'buy'
                    else:
                        side = 'buy'  # Default for first trade
            else:
                # No midpoint available - use tick test with alternation
                if prev_price is not None:
                    if price > prev_price:
                        side = 'buy'
                    elif price < prev_price:
                        side = 'sell'
                    else:
                        # Alternate to ensure mix
                        side = 'sell' if prev_side == 'buy' else 'buy'
                else:
                    # Start with random to ensure variety
                    side = 'buy' if i % 2 == 0 else 'sell'
            
            # Track counts for balance check
            if side == 'buy':
                buy_count += 1
            else:
                sell_count += 1
            
            # Add classification to trade
            classified_trade = trade.copy()
            classified_trade['side'] = side
            classified.append(classified_trade)
            
            # Update for next iteration
            prev_price = price
            prev_side = side
        
        # Log if severely imbalanced (for debugging)
        total = len(classified)
        if total > 0:
            buy_ratio = buy_count / total
            if buy_ratio > 0.95 or buy_ratio < 0.05:
                self.logger.debug(f"Trade classification imbalanced for {symbol}: "
                                f"{buy_count} buys, {sell_count} sells")
        
        return classified
    
    def _create_volume_buckets(self, trades: List[dict], bucket_size: int) -> List[dict]:
        """
        Create volume-synchronized buckets for VPIN calculation.
        Each bucket contains trades until volume reaches bucket_size.
        
        Returns list of buckets with buy/sell volumes.
        """
        if not trades:
            return []
        
        buckets = []
        current_bucket = {
            'buy_volume': 0,
            'sell_volume': 0,
            'total_volume': 0,
            'trades': []
        }
        
        for trade in trades:
            size = trade.get('size', 0)
            side = trade.get('side', 'buy')
            
            # Check if adding this trade exceeds bucket size
            if current_bucket['total_volume'] + size > bucket_size and current_bucket['total_volume'] > 0:
                # Save current bucket
                buckets.append(current_bucket)
                # Start new bucket
                current_bucket = {
                    'buy_volume': 0,
                    'sell_volume': 0,
                    'total_volume': 0,
                    'trades': []
                }
            
            # Add trade to current bucket
            if side == 'buy':
                current_bucket['buy_volume'] += size
            else:
                current_bucket['sell_volume'] += size
            
            current_bucket['total_volume'] += size
            current_bucket['trades'].append(trade)
        
        # Add final bucket if it has trades
        if current_bucket['total_volume'] > 0:
            buckets.append(current_bucket)
        
        return buckets
    
    def _calculate_vpin_from_buckets(self, buckets: List[dict]) -> float:
        """
        Calculate VPIN from volume buckets.
        
        VPIN = Mean(|Buy Volume - Sell Volume| / Total Volume) across buckets
        
        Returns VPIN score between 0 and 1.
        """
        if not buckets:
            return 0.5  # Neutral VPIN
        
        # Need at least 50 buckets for reliable VPIN (from research)
        min_buckets = self.config.get('parameter_discovery', {}).get('vpin', {}).get('min_buckets_for_vpin', 50)
        
        if len(buckets) < min_buckets:
            # Not enough data - calculate simple imbalance
            total_buy = sum(b['buy_volume'] for b in buckets)
            total_sell = sum(b['sell_volume'] for b in buckets)
            total_volume = total_buy + total_sell
            
            if total_volume > 0:
                simple_vpin = abs(total_buy - total_sell) / total_volume
                # Weight by data availability (less data = closer to neutral)
                weight = len(buckets) / min_buckets
                return 0.5 + (simple_vpin - 0.5) * weight
            return 0.5
        
        # Calculate VPIN across all buckets
        imbalances = []
        for bucket in buckets:
            total = bucket['total_volume']
            if total > 0:
                imbalance = abs(bucket['buy_volume'] - bucket['sell_volume']) / total
                imbalances.append(imbalance)
        
        if imbalances:
            # VPIN is the mean absolute imbalance
            vpin = np.mean(imbalances)
            # Ensure VPIN is between 0 and 1
            return float(np.clip(vpin, 0.0, 1.0))
        
        return 0.5  # Neutral if no valid imbalances
    
    async def calculate_vpin(self, symbol: str) -> float:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).
        
        VPIN measures the probability of informed trading based on order flow imbalance.
        Higher VPIN (>0.7) indicates toxic/informed flow.
        Lower VPIN (<0.3) indicates uninformed/retail flow.
        
        Returns:
            VPIN score between 0 and 1
        """
        try:
            # 1. Get discovered bucket size from Redis
            bucket_size_str = await self.redis.get('discovered:vpin_bucket_size')
            if bucket_size_str:
                bucket_size = int(bucket_size_str)
                self.logger.debug(f"Using discovered bucket size: {bucket_size}")
            else:
                # Fallback to config default
                bucket_size = self.config.get('parameter_discovery', {}).get('vpin', {}).get('default_bucket_size', 100)
                self.logger.warning(f"No discovered bucket size, using default: {bucket_size}")
            
            # 2. Fetch recent trades from Redis (last 1000 trades)
            trades_json = await self.redis.lrange(f'market:{symbol}:trades', -1000, -1)
            
            # Check minimum trades requirement
            min_trades = self.config.get('parameter_discovery', {}).get('vpin', {}).get('min_trades_required', 100)
            if len(trades_json) < min_trades:
                self.logger.debug(f"Insufficient trades for {symbol}: {len(trades_json)} < {min_trades}")
                return 0.5  # Neutral VPIN if insufficient data
            
            # 3. Parse trades
            trades = []
            for trade_str in trades_json:
                try:
                    trade = json.loads(trade_str)
                    trades.append(trade)
                except json.JSONDecodeError:
                    continue
            
            if not trades:
                return 0.5
            
            # 4. Classify trades as buy/sell
            classified_trades = await self._classify_trades(trades, symbol)
            
            # 5. Create volume buckets
            buckets = self._create_volume_buckets(classified_trades, bucket_size)
            
            # 6. Calculate VPIN from buckets
            vpin = self._calculate_vpin_from_buckets(buckets)
            
            # 7. Store in Redis with configured TTL
            ttl = self.ttls.get('metrics', 60)
            vpin_data = {
                'value': round(float(vpin), 4),
                'buckets': len(buckets),
                'bucket_size': bucket_size,
                'trades_analyzed': len(trades),
                'timestamp': time.time()
            }
            
            await self.redis.setex(
                f'metrics:{symbol}:vpin',
                ttl,
                json.dumps(vpin_data)
            )
            
            # Log if VPIN is extreme
            if vpin > 0.7:
                self.logger.info(f"HIGH VPIN for {symbol}: {vpin:.3f} (toxic flow detected)")
            elif vpin < 0.3:
                self.logger.info(f"LOW VPIN for {symbol}: {vpin:.3f} (uninformed flow)")
            
            return float(vpin)
            
        except Exception as e:
            self.logger.error(f"Error calculating VPIN for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return 0.5  # Default neutral VPIN on error
    
    def _extract_occ(self, key: str):
        """
        Extract option type and strike from Redis key using OCC format.
        Key format: options:{SYMBOL}:{OCC}:greeks
        Returns (cp, strike_float) or (None, 0.0) if not match.
        """
        try:
            parts = key.split(':', 3)  # Split into at most 4 parts
            if len(parts) >= 3:
                occ = parts[2]  # Third token is OCC
                m = _OCC_RE.match(occ)
                if not m:
                    return None, 0.0
                cp = m.group('cp')  # 'C' or 'P'
                strike = float(m.group('strike')) / 1000.0  # OCC 8 digits, 3 implied decimals
                return cp, strike
            return None, 0.0
        except Exception:
            return None, 0.0
    
    def _extract_option_type(self, key: str) -> str:
        """
        Return 'call' or 'put' from the OCC symbol.
        """
        cp, _ = self._extract_occ(key)
        if not cp:
            return ''
        return 'call' if cp == 'C' else 'put'
    
    def _extract_strike_from_key(self, key: str) -> float:
        """
        Extract strike price from Redis options key.
        """
        _, strike = self._extract_occ(key)
        return strike
    
    async def calculate_gex(self, symbol: str) -> dict:
        """
        Calculate Gamma Exposure (GEX) from options chain.
        
        GEX measures the hedging flow required by market makers.
        Positive GEX: Market makers buy dips, sell rallies (stabilizing)
        Negative GEX: Market makers sell dips, buy rallies (destabilizing)
        
        Returns:
            Dictionary with GEX by strike and key levels
        """
        try:
            # 1. Get current spot price
            ticker_json = await self.redis.get(f'market:{symbol}:ticker')
            if not ticker_json:
                self.logger.warning(f"No ticker data for {symbol}")
                return {'error': 'No spot price'}
            
            ticker = json.loads(ticker_json)
            spot = ticker.get('last')
            if spot is None or spot <= 0:
                return {'error': 'Invalid spot price'}
            
            # 2. Get all option Greeks keys for this symbol
            # Note: Alpha Vantage stores options differently than expected
            # We need to check for the actual format used
            greek_keys = []
            
            # Use direct KEYS command - SCAN is broken with aioredis
            pattern = f'options:{symbol}:*:greeks'
            
            try:
                # Direct KEYS command - simple and works
                greek_keys = await self.redis.keys(pattern)
                
                if not greek_keys:
                    # Try alternate patterns
                    for alt_pattern in [f'options:{symbol}:greeks:*', f'greeks:{symbol}:*']:
                        greek_keys = await self.redis.keys(alt_pattern)
                        if greek_keys:
                            break
                
                self.logger.debug(f"Found {len(greek_keys)} option contracts for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to get option keys for {symbol}: {e}")
                return {'error': f'Failed to get options for {symbol}'}
            
            if not greek_keys:
                self.logger.warning(f"No options data for {symbol}")
                return {'error': f'No options data for {symbol}'}
            
            # 3. Calculate GEX for each strike
            gex_by_strike = {}
            oi_by_strike = {}  # Track OI per strike for filtering
            total_gex = 0
            contract_multiplier = 100  # Standard equity option multiplier
            
            # Track statistics for debugging
            calls_processed = 0
            puts_processed = 0
            contracts_skipped = 0
            
            for key in greek_keys:
                try:
                    # Get Greeks data
                    greeks_json = await self.redis.get(key)
                    if not greeks_json:
                        continue
                    
                    greeks = json.loads(greeks_json)
                    # Convert string values to floats (Alpha Vantage returns strings)
                    try:
                        gamma = float(greeks.get('gamma', 0))
                    except (ValueError, TypeError):
                        gamma = 0
                    
                    # Guardrail: Skip invalid gamma values (AV quirks)
                    if gamma < 0 or gamma > 1:  # Gamma should be 0-1
                        contracts_skipped += 1
                        continue
                    
                    # Get open interest (might be in separate key or same dict)
                    try:
                        open_interest = float(greeks.get('open_interest', 0))
                    except (ValueError, TypeError):
                        open_interest = 0
                    
                    if open_interest <= 0 or gamma == 0:
                        contracts_skipped += 1
                        continue
                    
                    # Extract strike and option type
                    strike = self._extract_strike_from_key(key)
                    option_type = self._extract_option_type(key)
                    
                    if strike <= 0 or not option_type:
                        contracts_skipped += 1
                        continue
                    
                    # Calculate GEX for this contract
                    # GEX = Gamma * Open Interest * Contract Multiplier * Spot^2
                    # Alpha Vantage gamma is per $1 move, no /100 needed
                    # For calls: positive gamma exposure
                    # For puts: negative gamma exposure (dealers are short)
                    
                    if option_type == 'call':
                        contract_gex = gamma * open_interest * contract_multiplier * spot * spot
                        calls_processed += 1
                    else:  # put
                        contract_gex = -gamma * open_interest * contract_multiplier * spot * spot
                        puts_processed += 1
                    
                    # Aggregate by strike
                    if strike not in gex_by_strike:
                        gex_by_strike[strike] = 0
                    gex_by_strike[strike] += contract_gex
                    total_gex += contract_gex
                    
                    # Track OI per strike for filtering
                    if strike not in oi_by_strike:
                        oi_by_strike[strike] = 0
                    oi_by_strike[strike] += int(open_interest)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing option {key}: {e}")
                    continue
            
            # Debug logging for telemetry
            if symbol == 'SPY' or self.logger.level <= logging.DEBUG:
                self.logger.debug(f"GEX stats for {symbol}: {calls_processed} calls, {puts_processed} puts, {contracts_skipped} skipped")
                # Sample strike parsing to catch regressions
                if gex_by_strike and self.logger.isEnabledFor(logging.DEBUG):
                    sample = list(gex_by_strike.items())[:3]
                    self.logger.debug(f"[PARSE] {symbol} sample strikes: {sample}")
            
            if not gex_by_strike:
                return {'error': 'No valid GEX data'}
            
            # 4. Find key levels with OI floor to filter out ghost strikes
            MIN_OI = 5  # Minimum open interest threshold
            valid_strikes = [(s, v) for s, v in gex_by_strike.items() 
                           if oi_by_strike.get(s, 0) >= MIN_OI]
            
            if not valid_strikes:
                self.logger.warning(f"No strikes with OI >= {MIN_OI} for {symbol}")
                return {'error': 'No valid GEX data after OI filter'}
            
            sorted_strikes = sorted([s for s, _ in valid_strikes])
            
            # Find max GEX strike (biggest hedging level) from valid strikes only
            max_gex_strike, _ = max(valid_strikes, key=lambda kv: abs(kv[1]))
            
            # Find zero gamma level (flip point)
            # This is where cumulative GEX crosses zero
            cumulative_gex = 0
            zero_gamma_strike = None
            for strike in sorted_strikes:
                cumulative_gex += gex_by_strike[strike]
                if zero_gamma_strike is None and cumulative_gex >= 0:
                    zero_gamma_strike = strike
            
            # Identify support and resistance levels
            # Support: Strikes with high positive GEX below spot
            # Resistance: Strikes with high positive GEX above spot
            supports = []
            resistances = []
            
            for strike in sorted_strikes:
                gex = gex_by_strike[strike]
                if gex > total_gex * 0.1:  # Significant level (>10% of total)
                    if strike < spot:
                        supports.append(strike)
                    else:
                        resistances.append(strike)
            
            # 5. Prepare result
            result = {
                'spot': round(spot, 2),
                'total_gex': round(total_gex, 0),
                'gex_by_strike': {str(k): round(v, 0) for k, v in gex_by_strike.items()},
                'max_gex_strike': max_gex_strike,
                'zero_gamma_strike': zero_gamma_strike,
                'supports': sorted(supports, reverse=True)[:3],  # Top 3 supports
                'resistances': sorted(resistances)[:3],  # Top 3 resistances
                'regime': 'stabilizing' if total_gex > 0 else 'destabilizing',
                'units': 'dollar',  # Explicit units for downstream consumers
                'timestamp': time.time()
            }
            
            # 6. Store in Redis with configured TTL
            ttl = self.ttls.get('metrics', 60)
            await self.redis.setex(
                f'metrics:{symbol}:gex',
                ttl,
                json.dumps(result)
            )
            
            # Log significant levels
            self.logger.info(f"GEX for {symbol}: Total={total_gex:.0f}, Max Strike={max_gex_strike}, Regime={result['regime']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating GEX for {symbol}: {e}")
            self.logger.error(f"Full traceback for {symbol} GEX error:\n{traceback.format_exc()}")
            return {'error': str(e)}
    
    async def calculate_dex(self, symbol: str) -> dict:
        """
        Calculate Delta Exposure (DEX) from options chain.
        
        DEX measures directional exposure from options positioning.
        Positive DEX: Net long exposure (bullish positioning)
        Negative DEX: Net short exposure (bearish positioning)
        
        Returns:
            Dictionary with DEX by strike and directional bias
        """
        try:
            # 1. Get current spot price
            ticker_json = await self.redis.get(f'market:{symbol}:ticker')
            if not ticker_json:
                return {'error': 'No spot price'}
            
            ticker = json.loads(ticker_json)
            spot = ticker.get('last')
            if spot is None or spot <= 0:
                return {'error': 'Invalid spot price'}
            
            # 2. Get all option Greeks keys using direct KEYS command
            pattern = f'options:{symbol}:*:greeks'
            
            try:
                # Direct KEYS command - simple and works
                greek_keys = await self.redis.keys(pattern)
                
                if not greek_keys:
                    # Try alternate patterns
                    for alt_pattern in [f'options:{symbol}:greeks:*', f'greeks:{symbol}:*']:
                        greek_keys = await self.redis.keys(alt_pattern)
                        if greek_keys:
                            break
                
            except Exception as e:
                self.logger.error(f"Failed to get option keys for {symbol} DEX: {e}")
                return {'error': f'Failed to get options for {symbol}'}
            
            if not greek_keys:
                return {'error': f'No options data for {symbol}'}
            
            # 3. Calculate DEX for each strike
            dex_by_strike = {}
            oi_by_strike = {}  # Track OI per strike for filtering
            total_call_dex = 0
            total_put_dex = 0
            contract_multiplier = 100
            
            # Track statistics for debugging
            calls_processed = 0
            puts_processed = 0
            contracts_skipped = 0
            
            for key in greek_keys:
                try:
                    # Get Greeks data
                    greeks_json = await self.redis.get(key)
                    if not greeks_json:
                        continue
                    
                    greeks = json.loads(greeks_json)
                    # Convert string values to floats (Alpha Vantage returns strings)
                    try:
                        delta = float(greeks.get('delta', 0))
                    except (ValueError, TypeError):
                        delta = 0
                    
                    # Guardrail: Skip invalid delta values (AV quirks)
                    if abs(delta) > 1.2:  # Delta should be between -1 and 1
                        contracts_skipped += 1
                        continue
                    
                    try:
                        open_interest = float(greeks.get('open_interest', 0))
                    except (ValueError, TypeError):
                        open_interest = 0
                    
                    if open_interest <= 0 or delta == 0:
                        contracts_skipped += 1
                        continue
                    
                    # Extract strike and option type
                    strike = self._extract_strike_from_key(key)
                    option_type = self._extract_option_type(key)
                    
                    if strike <= 0 or not option_type:
                        contracts_skipped += 1
                        continue
                    
                    # Calculate DEX for this contract
                    # DEX = Delta * Open Interest * Contract Multiplier * Spot
                    contract_dex = delta * open_interest * contract_multiplier * spot
                    
                    # Aggregate by strike
                    if strike not in dex_by_strike:
                        dex_by_strike[strike] = {'call': 0, 'put': 0, 'net': 0}
                    
                    if option_type == 'call':
                        dex_by_strike[strike]['call'] += contract_dex
                        total_call_dex += contract_dex
                        calls_processed += 1
                    else:
                        dex_by_strike[strike]['put'] += contract_dex
                        total_put_dex += contract_dex
                        puts_processed += 1
                    
                    dex_by_strike[strike]['net'] += contract_dex
                    
                    # Track OI per strike for filtering
                    if strike not in oi_by_strike:
                        oi_by_strike[strike] = 0
                    oi_by_strike[strike] += int(open_interest)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing option {key}: {e}")
                    continue
            
            # Debug logging for telemetry
            if symbol == 'SPY' or self.logger.level <= logging.DEBUG:
                self.logger.debug(f"DEX stats for {symbol}: {calls_processed} calls, {puts_processed} puts, {contracts_skipped} skipped")
            
            if not dex_by_strike:
                return {'error': 'No valid DEX data'}
            
            # 4. Calculate total net DEX
            total_dex = total_call_dex + total_put_dex
            
            # 5. Determine directional bias
            if abs(total_dex) < abs(total_call_dex) * 0.1:
                bias = 'neutral'
            elif total_dex > 0:
                bias = 'bullish'
            else:
                bias = 'bearish'
            
            # 6. Find strikes with highest exposure (filter by OI floor)
            MIN_OI = 5  # Minimum open interest threshold
            valid_strikes = [(s, v) for s, v in dex_by_strike.items() 
                           if oi_by_strike.get(s, 0) >= MIN_OI]
            
            if not valid_strikes:
                self.logger.warning(f"No strikes with OI >= {MIN_OI} for {symbol} DEX")
                return {'error': 'No valid DEX data after OI filter'}
            
            sorted_by_exposure = sorted(valid_strikes, 
                                       key=lambda x: abs(x[1]['net']), 
                                       reverse=True)
            
            key_strikes = [strike for strike, _ in sorted_by_exposure[:5]]
            
            # 7. Calculate put/call ratio
            pc_ratio = abs(total_put_dex / total_call_dex) if total_call_dex != 0 else 0
            
            # 8. Prepare result
            result = {
                'spot': round(spot, 2),
                'total_dex': round(total_dex, 0),
                'call_dex': round(total_call_dex, 0),
                'put_dex': round(total_put_dex, 0),
                'dex_by_strike': {
                    str(k): {
                        'net': round(v['net'], 0),
                        'call': round(v['call'], 0),
                        'put': round(v['put'], 0)
                    } for k, v in dex_by_strike.items()
                },
                'bias': bias,
                'pc_ratio': round(pc_ratio, 2),
                'key_strikes': key_strikes,
                'units': 'dollar_delta',  # Explicit units for downstream consumers
                'timestamp': time.time()
            }
            
            # 9. Store in Redis
            ttl = self.ttls.get('metrics', 60)
            await self.redis.setex(
                f'metrics:{symbol}:dex',
                ttl,
                json.dumps(result)
            )
            
            # Log directional bias
            self.logger.info(f"DEX for {symbol}: Total={total_dex:.0f}, Bias={bias}, PC Ratio={pc_ratio:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating DEX for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def _calculate_l1_imbalance(self, book: dict) -> float:
        """Calculate Level 1 (best bid/ask) imbalance."""
        try:
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            best_bid_size = bids[0].get('size', 0) if bids else 0
            best_ask_size = asks[0].get('size', 0) if asks else 0
            
            total = best_bid_size + best_ask_size
            if total == 0:
                return 0.0
            
            # Imbalance: positive = more bids, negative = more asks
            return (best_bid_size - best_ask_size) / total
            
        except Exception:
            return 0.0
    
    def _calculate_l5_imbalance(self, book: dict) -> float:
        """Calculate Level 5 (top 5 levels) weighted imbalance."""
        try:
            bids = book.get('bids', [])[:5]
            asks = book.get('asks', [])[:5]
            
            if not bids or not asks:
                return 0.0
            
            # Weight by inverse distance from mid (closer levels have more weight)
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]
            
            weighted_bid_volume = 0
            weighted_ask_volume = 0
            
            for i, (bid, ask) in enumerate(zip(bids[:5], asks[:5])):
                weight = weights[i] if i < len(weights) else 0.1
                weighted_bid_volume += bid.get('size', 0) * weight
                weighted_ask_volume += ask.get('size', 0) * weight
            
            total = weighted_bid_volume + weighted_ask_volume
            if total == 0:
                return 0.0
            
            return (weighted_bid_volume - weighted_ask_volume) / total
            
        except Exception:
            return 0.0
    
    def _calculate_pressure_ratio(self, book: dict) -> float:
        """Calculate bid/ask pressure ratio."""
        try:
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            # Sum all visible bid and ask volumes
            total_bid_volume = sum(level.get('size', 0) for level in bids)
            total_ask_volume = sum(level.get('size', 0) for level in asks)
            
            if total_ask_volume == 0:
                return 2.0  # Maximum pressure (capped)
            
            ratio = total_bid_volume / total_ask_volume
            # Cap at reasonable bounds
            return min(max(ratio, 0.1), 10.0)
            
        except Exception:
            return 1.0
    
    def _calculate_micro_price(self, book: dict) -> float:
        """
        Calculate micro-price (size-weighted midpoint).
        More accurate than simple midpoint for predicting short-term price moves.
        """
        try:
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            best_bid = bids[0].get('price', 0)
            best_ask = asks[0].get('price', 0)
            best_bid_size = bids[0].get('size', 0)
            best_ask_size = asks[0].get('size', 0)
            
            total_size = best_bid_size + best_ask_size
            if total_size == 0:
                return (best_bid + best_ask) / 2
            
            # Weighted average
            micro_price = (best_bid * best_ask_size + best_ask * best_bid_size) / total_size
            return micro_price
            
        except Exception:
            return 0.0
    
    async def calculate_order_book_imbalance(self, symbol: str) -> dict:
        """
        Calculate comprehensive order book imbalance metrics.
        
        These metrics help identify:
        - Short-term price direction (micro-price)
        - Buying/selling pressure (imbalance)
        - Potential support/resistance (pressure ratio)
        
        Returns:
            Dictionary with multiple imbalance metrics
        """
        try:
            # 1. Fetch current order book
            book_json = await self.redis.get(f'market:{symbol}:book')
            if not book_json:
                return {'error': 'No order book data'}
            
            book = json.loads(book_json)
            
            # 2. Calculate various imbalance metrics
            l1_imbalance = self._calculate_l1_imbalance(book)
            l5_imbalance = self._calculate_l5_imbalance(book)
            pressure_ratio = self._calculate_pressure_ratio(book)
            micro_price = self._calculate_micro_price(book)
            
            # 3. Calculate book velocity (requires history)
            # For now, store current imbalance for future velocity calculation
            velocity_key = f'obi_history:{symbol}'
            history_json = await self.redis.get(velocity_key)
            
            if history_json:
                history = json.loads(history_json)
            else:
                history = []
            
            # Add current imbalance to history
            current_data = {
                'l1': l1_imbalance,
                'l5': l5_imbalance,
                'timestamp': time.time()
            }
            history.append(current_data)
            
            # Keep only last 20 samples (10 seconds at 2Hz)
            history = history[-20:]
            
            # Calculate velocity if we have enough history
            book_velocity = 0.0
            if len(history) >= 5:
                # Rate of change of imbalance
                old_l5 = history[-5]['l5']
                new_l5 = history[-1]['l5']
                time_diff = history[-1]['timestamp'] - history[-5]['timestamp']
                
                if time_diff > 0:
                    book_velocity = (new_l5 - old_l5) / time_diff
            
            # Store history for next calculation (short TTL for velocity tracking)
            velocity_ttl = min(30, self.ttls.get('metrics', 60))  # Use shorter of 30s or config
            await self.redis.setex(velocity_key, velocity_ttl, json.dumps(history))
            
            # 4. Determine market state based on metrics
            if abs(l5_imbalance) > 0.7:
                state = 'extreme_imbalance'
            elif abs(l5_imbalance) > 0.4:
                state = 'moderate_imbalance'
            else:
                state = 'balanced'
            
            # 5. Prepare result
            result = {
                'level1_imbalance': round(l1_imbalance, 4),
                'level5_imbalance': round(l5_imbalance, 4),
                'pressure_ratio': round(pressure_ratio, 2),
                'micro_price': round(micro_price, 2),
                'book_velocity': round(book_velocity, 4),
                'state': state,
                'timestamp': time.time()
            }
            
            # 6. Store in Redis
            ttl = self.ttls.get('metrics', 60)
            await self.redis.setex(
                f'metrics:{symbol}:obi',
                ttl,
                json.dumps(result)
            )
            
            # Log extreme imbalances
            if abs(l5_imbalance) > 0.7:
                direction = 'BID' if l5_imbalance > 0 else 'ASK'
                self.logger.info(f"EXTREME {direction} imbalance for {symbol}: L5={l5_imbalance:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating OBI for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    async def detect_hidden_orders(self, symbol: str) -> dict:
        """
        Detect potential hidden/iceberg orders from market patterns.
        
        Hidden orders are detected by:
        1. Trades executing beyond best bid/ask (dark pool)
        2. Persistent refills at same price level (iceberg)
        3. Volume spikes without corresponding book depth
        
        Returns:
            Dictionary with hidden order detection results
        """
        try:
            # 1. Get current order book and recent trades
            book_json = await self.redis.get(f'market:{symbol}:book')
            trades_json = await self.redis.lrange(f'market:{symbol}:trades', -100, -1)
            
            if not book_json or not trades_json:
                return {'error': 'Insufficient data'}
            
            book = json.loads(book_json)
            trades = [json.loads(t) for t in trades_json]
            
            if not trades:
                return {'error': 'No trades'}
            
            # Extract best bid/ask
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            if not bids or not asks:
                return {'error': 'Invalid book'}
            
            best_bid = bids[0].get('price', 0)
            best_ask = asks[0].get('price', 0)
            
            # 2. Detect trades beyond NBBO (potential hidden orders)
            hidden_trades = []
            for trade in trades[-50:]:  # Last 50 trades
                price = trade.get('price', 0)
                size = trade.get('size', 0)
                
                # Trade outside NBBO suggests hidden liquidity
                if price > best_ask * 1.001 or price < best_bid * 0.999:
                    hidden_trades.append({
                        'price': price,
                        'size': size,
                        'type': 'dark' if price > best_ask or price < best_bid else 'midpoint'
                    })
            
            # 3. Detect iceberg orders (persistent refills)
            price_refills = {}
            for i in range(1, len(trades)):
                if trades[i].get('price') == trades[i-1].get('price'):
                    price = trades[i].get('price')
                    if price not in price_refills:
                        price_refills[price] = 0
                    price_refills[price] += 1
            
            # Identify potential icebergs (>5 refills at same price)
            icebergs = []
            for price, refills in price_refills.items():
                if refills >= 5:
                    icebergs.append({
                        'price': price,
                        'refills': refills,
                        'confidence': min(refills / 10, 1.0)  # Confidence score
                    })
            
            # 4. Detect volume spikes without depth
            recent_volume = sum(t.get('size', 0) for t in trades[-20:])
            visible_depth = sum(l.get('size', 0) for l in bids[:5]) + \
                           sum(l.get('size', 0) for l in asks[:5])
            
            volume_depth_ratio = recent_volume / visible_depth if visible_depth > 0 else 0
            
            # High ratio suggests hidden liquidity
            hidden_liquidity_likely = volume_depth_ratio > 2.0
            
            # 5. Calculate hidden order probability
            hidden_score = 0.0
            
            # Weight different signals
            if hidden_trades:
                hidden_score += 0.4 * min(len(hidden_trades) / 10, 1.0)
            if icebergs:
                hidden_score += 0.4 * min(len(icebergs) / 3, 1.0)
            if hidden_liquidity_likely:
                hidden_score += 0.2
            
            # 6. Prepare result
            result = {
                'hidden_trades': len(hidden_trades),
                'hidden_trade_volume': sum(t['size'] for t in hidden_trades),
                'iceberg_levels': len(icebergs),
                'top_icebergs': sorted(icebergs, key=lambda x: x['confidence'], reverse=True)[:3],
                'volume_depth_ratio': round(volume_depth_ratio, 2),
                'hidden_score': round(hidden_score, 3),
                'hidden_likely': hidden_score > 0.5,
                'timestamp': time.time()
            }
            
            # 7. Store in Redis
            ttl = self.ttls.get('metrics', 60)
            await self.redis.setex(
                f'metrics:{symbol}:hidden',
                ttl,
                json.dumps(result)
            )
            
            # Log significant hidden order detection
            if hidden_score > 0.7:
                self.logger.info(f"HIGH hidden order probability for {symbol}: score={hidden_score:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden orders for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    async def calculate_multi_timeframe_metrics(self, symbol: str) -> dict:
        """
        Aggregate metrics across multiple timeframes for trend analysis.
        
        Timeframes analyzed:
        - 1 minute (12 bars)
        - 5 minutes (60 bars)
        - 15 minutes (180 bars)
        - 30 minutes (360 bars)
        
        Returns:
            Dictionary with trend, momentum, and alignment scores
        """
        try:
            # 1. Fetch recent bars (5-second bars from Redis)
            bars_json = await self.redis.lrange(f'market:{symbol}:bars', -360, -1)
            
            if len(bars_json) < 12:  # Need at least 1 minute of data
                return {'error': 'Insufficient bar data'}
            
            # Parse bars
            bars = []
            for bar_str in bars_json:
                try:
                    bar = json.loads(bar_str)
                    bars.append(bar)
                except json.JSONDecodeError:
                    continue
            
            if not bars:
                return {'error': 'No valid bars'}
            
            # 2. Aggregate bars into different timeframes
            timeframes = {
                '1min': 12,     # 12 x 5sec = 1 minute
                '5min': 60,     # 60 x 5sec = 5 minutes
                '15min': 180,   # 180 x 5sec = 15 minutes
                '30min': 360    # 360 x 5sec = 30 minutes
            }
            
            aggregated = {}
            
            for tf_name, num_bars in timeframes.items():
                if len(bars) < num_bars:
                    continue
                
                # Get the last num_bars bars
                tf_bars = bars[-num_bars:]
                
                # Aggregate OHLCV
                aggregated[tf_name] = {
                    'open': tf_bars[0].get('open', 0),
                    'high': max(b.get('high', 0) for b in tf_bars),
                    'low': min(b.get('low', 0) for b in tf_bars),
                    'close': tf_bars[-1].get('close', 0),
                    'volume': sum(b.get('volume', 0) for b in tf_bars),
                    'bars_count': len(tf_bars)
                }
            
            # 3. Calculate trend for each timeframe
            trends = {}
            momentum = {}
            
            for tf_name, tf_data in aggregated.items():
                # Simple trend: compare close to open
                price_change = (tf_data['close'] - tf_data['open']) / tf_data['open'] if tf_data['open'] > 0 else 0
                
                # Trend classification
                if price_change > 0.001:  # 0.1% threshold
                    trends[tf_name] = 'bullish'
                elif price_change < -0.001:
                    trends[tf_name] = 'bearish'
                else:
                    trends[tf_name] = 'neutral'
                
                # Momentum (rate of change)
                momentum[tf_name] = round(price_change * 100, 3)  # Percentage
                
                # Add SMA if we have enough data
                if tf_name == '1min' and len(bars) >= 20:
                    # Calculate 20-period SMA on 1-min timeframe
                    closes = [b.get('close', 0) for b in bars[-20:]]
                    sma20 = np.mean(closes)
                    aggregated[tf_name]['sma20'] = round(sma20, 2)
                    
                    # Price relative to SMA
                    if sma20 > 0:
                        aggregated[tf_name]['price_to_sma'] = round((tf_data['close'] - sma20) / sma20 * 100, 2)
            
            # 4. Calculate trend alignment score
            # Higher score means trends align across timeframes
            alignment_score = 0.0
            trend_values = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            
            if len(trends) >= 2:
                trend_nums = [trend_values.get(t, 0) for t in trends.values()]
                # Check if all trends point same direction
                if all(t > 0 for t in trend_nums):
                    alignment_score = 1.0  # All bullish
                elif all(t < 0 for t in trend_nums):
                    alignment_score = -1.0  # All bearish
                else:
                    # Partial alignment
                    alignment_score = np.mean(trend_nums)
            
            # 5. Detect divergences
            divergences = []
            
            # Check for trend divergence between short and long timeframes
            if '1min' in trends and '15min' in trends:
                if trends['1min'] == 'bullish' and trends['15min'] == 'bearish':
                    divergences.append('bearish_divergence')  # Short-term rally in downtrend
                elif trends['1min'] == 'bearish' and trends['15min'] == 'bullish':
                    divergences.append('bullish_divergence')  # Short-term dip in uptrend
            
            # Check for momentum divergence
            if '1min' in momentum and '5min' in momentum:
                if momentum['1min'] > 0 and momentum['5min'] < momentum['1min'] * 0.5:
                    divergences.append('momentum_weakening')
                elif momentum['1min'] < 0 and momentum['5min'] > momentum['1min'] * 0.5:
                    divergences.append('momentum_strengthening')
            
            # 6. Determine overall market state
            if alignment_score > 0.7:
                market_state = 'strong_trend_up'
            elif alignment_score < -0.7:
                market_state = 'strong_trend_down'
            elif abs(alignment_score) < 0.3:
                market_state = 'ranging'
            else:
                market_state = 'weak_trend'
            
            # 7. Prepare result
            result = {
                'timeframes': aggregated,
                'trends': trends,
                'momentum': momentum,
                'alignment_score': round(alignment_score, 3),
                'divergences': divergences,
                'market_state': market_state,
                'timestamp': time.time()
            }
            
            # 8. Store in Redis
            ttl = self.ttls.get('metrics', 60)
            await self.redis.setex(
                f'metrics:{symbol}:mtf',
                ttl,
                json.dumps(result)
            )
            
            # Log significant alignments
            if abs(alignment_score) > 0.7:
                direction = 'UP' if alignment_score > 0 else 'DOWN'
                self.logger.info(f"STRONG trend alignment {direction} for {symbol}: score={alignment_score:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating MTF metrics for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}