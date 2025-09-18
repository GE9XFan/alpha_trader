#!/usr/bin/env python3
"""
Parameter Discovery - Empirical Parameter Optimization
Part of Quantisity Capital System

This module operates independently and communicates only via Redis.
Redis keys used:
- market:{symbol}:trades: Trade data for analysis
- market:{symbol}:bars: Bar data for temporal analysis
- market:{symbol}:book: Order book data
- discovered:*: All discovered parameters output
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import acf
from collections import defaultdict, Counter
import json
import yaml
import redis.asyncio as aioredis
import time
import pytz
from datetime import datetime
from typing import Dict, List, Any
import logging
import traceback


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
                # Trades can have either 'time' or 'timestamp' field
                if 'time' in trade:
                    last_timestamp = trade['time']
                elif 'timestamp' in trade:
                    last_timestamp = trade['timestamp']
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Trade parse error for {symbol}: {e}", exc_info=True)

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
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Bar parse error for {symbol}: {e}", exc_info=True)

        if not last_timestamp:
            return False

        # Check if older than threshold
        threshold = self.discovery_config.get('data_staleness_threshold', 3600)
        # Convert timestamp from milliseconds to seconds if needed
        if last_timestamp > 1e10:  # Timestamp is in milliseconds
            last_timestamp = last_timestamp / 1000
        age = time.time() - float(last_timestamp)

        # If market is closed, allow data from last trading session (up to 24 hours old)
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
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Trade size parse error: {e}")
                continue

        if len(trade_sizes) < min_trades:
            self.logger.warning(f"Insufficient valid trades: {len(trade_sizes)}")
            return default_size

        # Apply K-means clustering
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
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(f"Venue parse error: {e}")
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

        # Create discovered config
        discovered_config = {
            'generated_at': datetime.now().isoformat(),
            'parameters': self.discovered_params,
            'metadata': {
                'symbols_analyzed': self.symbols,
                'discovery_config': self.discovery_config
            }
        }

        # Save to file
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)

        output_file = config_dir / 'discovered.yaml'

        with open(output_file, 'w') as f:
            yaml.dump(discovered_config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Generated discovered parameters config: {output_file}")

        # Also save a timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = config_dir / f'discovered_{timestamp}.yaml'

        with open(backup_file, 'w') as f:
            yaml.dump(discovered_config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Backup saved to: {backup_file}")