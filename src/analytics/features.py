#!/usr/bin/env python3
"""
Feature Engineering Module - UPDATED VERSION
Calculates features for ML model using Alpha Vantage for technical indicators.
Feeds the entire prediction pipeline with consistent feature generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from collections import deque
import asyncio

from src.data.options_data import OptionsDataManager
from src.data.market_data import MarketDataManager
from src.data.av_client import AlphaVantageClient
from src.data.fundamental_data import FundamentalDataManager
from src.data.market_regime import MarketRegimeDetector
from src.core.config import get_config, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Structured feature vector with metadata"""
    timestamp: datetime
    symbol: str
    features: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for storage"""
        return dict(zip(self.feature_names, self.features))
    
    def is_valid(self) -> bool:
        """Check if feature vector is valid"""
        return not np.any(np.isnan(self.features)) and len(self.features) == len(self.feature_names)


class FeatureEngine:
    """
    Feature engineering - FEEDS THE ML MODEL
    Now uses Alpha Vantage for technical indicators instead of TA-Lib.
    Includes sentiment and fundamental features.
    Total of 28+ features across 5 categories.
    """
    
    def __init__(self, 
                 options_manager: OptionsDataManager,
                 market_manager: MarketDataManager,
                 av_client: AlphaVantageClient,
                 fundamental_manager: Optional[FundamentalDataManager] = None,
                 regime_detector: Optional[MarketRegimeDetector] = None):
        """
        Initialize FeatureEngine with Alpha Vantage integration
        
        Args:
            options_manager: Options data manager for options features
            market_manager: Market data manager for price data
            av_client: Alpha Vantage client for technical indicators
            fundamental_manager: Optional fundamental data manager
            regime_detector: Optional market regime detector
        """
        self.options = options_manager
        self.market = market_manager
        self.av_client = av_client  # NEW: For technical indicators
        self.fundamentals = fundamental_manager
        self.regime = regime_detector
        self.config = get_config()
        
        # Define feature names for consistency - CRITICAL for ML
        self.feature_names = [
            # Price action features (5)
            'returns_5m',       # 5-minute return
            'returns_30m',      # 30-minute return
            'returns_1h',       # 1-hour return
            'volume_ratio',     # Current vs average volume
            'high_low_ratio',   # (High - Low) / Close
            
            # Technical indicators from AV (10)
            'rsi',              # RSI from Alpha Vantage
            'macd_signal',      # MACD signal from AV
            'macd_histogram',   # MACD histogram from AV
            'bb_upper',         # Bollinger Band upper from AV
            'bb_lower',         # Bollinger Band lower from AV
            'bb_position',      # Position within bands
            'atr',              # ATR from AV
            'adx',              # ADX from AV
            'obv_slope',        # OBV from AV (calculate slope)
            'vwap_distance',    # VWAP from AV (intraday only)
            
            # Options metrics (8)
            'iv_rank',          # IV rank (0-100)
            'iv_percentile',    # IV percentile
            'put_call_ratio',   # Put/Call volume ratio
            'gamma_exposure',   # Net gamma exposure
            'delta_neutral',    # Delta neutral price
            'max_pain_dist',    # Distance from max pain
            'call_volume',      # Call volume
            'put_volume',       # Put volume
            
            # Market structure (5)
            'spy_correlation',  # Correlation with SPY
            'qqq_correlation',  # Correlation with QQQ
            'vix_level',        # VIX level (normalized)
            'term_structure',   # Options term structure
            'market_regime',    # Market regime indicator
            
            # Sentiment features (3) - NEW
            'news_sentiment',   # News sentiment from AV
            'sentiment_volume', # Number of news articles
            'insider_sentiment', # Insider transaction sentiment
            
            # Fundamental features (3) - NEW
            'earnings_distance', # Days to/from earnings
            'fundamental_score', # Company fundamental health
            'pe_percentile'     # PE ratio percentile
        ]
        
        # Feature calculation cache
        self.feature_cache: Dict[str, FeatureVector] = {}
        self.cache_ttl = 60  # seconds
        
        # Technical indicator cache (from AV)
        self.technical_cache: Dict[str, Dict[str, float]] = {}
        self.technical_cache_time: Dict[str, datetime] = {}
        
        # Historical data storage for correlations
        self.correlation_window = 100  # bars
        self.historical_data: Dict[str, deque] = {}
        
        # Feature statistics for normalization
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"FeatureEngine initialized with {len(self.feature_names)} features using Alpha Vantage")
    
    async def calculate_features(self, 
                                symbol: str,
                                historical_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate all features - CRITICAL FOR ML
        Now uses Alpha Vantage for technical indicators.
        
        Args:
            symbol: Stock symbol
            historical_data: DataFrame with OHLCV data
            
        Returns:
            Feature array in consistent order
        """
        # TODO: Implement feature calculation with AV
        # 1. Check cache first
        # 2. Calculate price action features (local)
        # 3. Get technical indicators from AV
        # 4. Calculate options features (using cached AV data)
        # 5. Calculate market structure features
        # 6. Get sentiment features from AV
        # 7. Get fundamental features if available
        # 8. Combine into array
        # 9. Handle NaN values
        # 10. Cache result
        # 11. Return feature array
        pass
    
    def _calculate_price_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate price action features (local calculation)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of price features
        """
        # TODO: Implement price feature calculation
        # 1. Calculate returns at different timeframes
        # 2. Calculate volume ratio
        # 3. Calculate high-low ratio
        # 4. Handle edge cases
        # 5. Return features dict
        pass
    
    async def _get_av_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """
        Get technical indicators from Alpha Vantage instead of calculating locally
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of technical indicators
        """
        # TODO: Implement AV technical indicator fetching
        # 1. Check cache (5 min TTL)
        # 2. Fetch RSI from AV
        # 3. Fetch MACD from AV
        # 4. Fetch Bollinger Bands from AV
        # 5. Fetch ATR from AV
        # 6. Fetch ADX from AV
        # 7. Fetch OBV from AV
        # 8. Fetch VWAP from AV (if intraday)
        # 9. Cache results
        # 10. Return indicators dict
        pass
    
    async def _get_sentiment_features(self, symbol: str) -> Dict[str, float]:
        """
        Get sentiment features from Alpha Vantage NEWS_SENTIMENT
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of sentiment features
        """
        # TODO: Implement sentiment feature fetching
        # 1. Call av_client.get_news_sentiment()
        # 2. Aggregate sentiment scores
        # 3. Count article volume
        # 4. Get insider sentiment if available
        # 5. Normalize scores
        # 6. Return sentiment features
        pass
    
    async def _get_fundamental_features(self, symbol: str) -> Dict[str, float]:
        """
        Get fundamental features if manager available
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of fundamental features
        """
        if not self.fundamentals:
            return {
                'earnings_distance': 0.0,
                'fundamental_score': 0.5,
                'pe_percentile': 0.5
            }
        
        # TODO: Implement fundamental features
        # 1. Get earnings date distance
        # 2. Get fundamental score
        # 3. Get PE percentile vs peers
        # 4. Return features
        pass
    
    def _calculate_returns(self, data: pd.DataFrame, periods: int) -> float:
        """
        Calculate returns over specified periods
        
        Args:
            data: Price data
            periods: Number of periods (5-second bars)
            
        Returns:
            Percentage return
        """
        # TODO: Implement returns calculation
        # 1. Get current price
        # 2. Get price N periods ago
        # 3. Calculate percentage return
        # 4. Handle missing data
        # 5. Return return value
        pass
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """
        Calculate volume ratio vs average
        
        Args:
            data: OHLCV data
            
        Returns:
            Volume ratio
        """
        # TODO: Implement volume ratio
        # 1. Get current volume
        # 2. Calculate average volume
        # 3. Calculate ratio
        # 4. Handle zero volume
        # 5. Return ratio
        pass
    
    def _calculate_options_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate options-specific features using cached Alpha Vantage data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of options features
        """
        # TODO: Implement options feature calculation using cached AV data
        # 1. Get ATM options from options manager (has AV Greeks)
        # 2. Get IV directly from option data (from AV)
        # 3. Calculate IV rank using options manager
        # 4. Calculate IV percentile
        # 5. Get put/call ratio from options manager
        # 6. Get gamma exposure from options manager
        # 7. Find delta neutral price
        # 8. Calculate max pain distance
        # 9. Get option volumes from cached data
        # 10. Handle missing data
        # 11. Return features dict
        pass
    
    def _calculate_market_structure(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate market structure features
        
        Args:
            symbol: Stock symbol
            data: Price data
            
        Returns:
            Dictionary of market structure features
        """
        # TODO: Implement market structure calculation
        # 1. Calculate SPY correlation
        # 2. Calculate QQQ correlation
        # 3. Get VIX level (from market data or AV)
        # 4. Calculate term structure
        # 5. Get market regime if detector available
        # 6. Return features dict
        pass
    
    def _calculate_correlation(self, 
                              data1: pd.Series,
                              data2: pd.Series,
                              window: int = 100) -> float:
        """
        Calculate rolling correlation
        
        Args:
            data1: First price series
            data2: Second price series
            window: Correlation window
            
        Returns:
            Correlation coefficient
        """
        # TODO: Implement correlation calculation
        # 1. Align series
        # 2. Calculate returns
        # 3. Calculate correlation
        # 4. Handle insufficient data
        # 5. Return correlation
        pass
    
    async def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from model or pre-calculated
        
        Returns:
            Series with feature importance scores
        """
        # TODO: Implement importance retrieval
        # 1. Load from model if available
        # 2. Or use pre-calculated importance
        # 3. Create Series with names
        # 4. Sort by importance
        # 5. Return Series
        pass
    
    def validate_features(self, features: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate feature vector
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        # TODO: Implement feature validation
        # 1. Check array length
        # 2. Check for NaN values
        # 3. Check for infinite values
        # 4. Validate feature ranges
        # 5. Return validation result
        pass
    
    def calculate_feature_stats(self, 
                               training_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature statistics for normalization
        
        Args:
            training_data: Training data with features
            
        Returns:
            Statistics dictionary
        """
        # TODO: Implement statistics calculation
        # 1. Calculate mean for each feature
        # 2. Calculate std for each feature
        # 3. Calculate min/max
        # 4. Calculate percentiles
        # 5. Store and return stats
        pass
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using stored statistics
        
        Args:
            features: Raw feature array
            
        Returns:
            Normalized feature array
        """
        # TODO: Implement normalization
        # 1. Apply z-score normalization
        # 2. Clip outliers
        # 3. Handle missing stats
        # 4. Return normalized array
        pass
    
    def get_feature_names_by_category(self) -> Dict[str, List[str]]:
        """
        Get feature names organized by category
        
        Returns:
            Dictionary mapping categories to feature names
        """
        return {
            'price_action': self.feature_names[0:5],
            'technical': self.feature_names[5:15],
            'options': self.feature_names[15:23],
            'market_structure': self.feature_names[23:28],
            'sentiment': self.feature_names[28:31],
            'fundamental': self.feature_names[31:34]
        }
    
    def create_feature_vector(self,
                            symbol: str,
                            features: np.ndarray,
                            metadata: Optional[Dict[str, Any]] = None) -> FeatureVector:
        """
        Create structured feature vector
        
        Args:
            symbol: Stock symbol
            features: Feature array
            metadata: Optional metadata
            
        Returns:
            FeatureVector object
        """
        # TODO: Implement vector creation
        # 1. Create FeatureVector
        # 2. Add timestamp
        # 3. Add metadata
        # 4. Validate vector
        # 5. Return vector
        pass
    
    async def warmup_av_indicators(self, symbols: List[str]) -> bool:
        """
        Warmup Alpha Vantage technical indicators cache
        
        Args:
            symbols: List of symbols to warmup
            
        Returns:
            True if successful
        """
        # TODO: Implement AV warmup
        # 1. For each symbol:
        #    a. Fetch RSI
        #    b. Fetch MACD
        #    c. Fetch Bollinger Bands
        #    d. Cache results
        # 2. Log warmup status
        # 3. Return success
        pass
    
    def save_features(self, features: List[FeatureVector], filepath: str) -> bool:
        """
        Save features to file for analysis
        
        Args:
            features: List of feature vectors
            filepath: Output file path
            
        Returns:
            True if saved successfully
        """
        # TODO: Implement feature saving
        # 1. Convert to DataFrame
        # 2. Add metadata columns
        # 3. Save to parquet/CSV
        # 4. Return success status
        pass
    
    def load_features(self, filepath: str) -> List[FeatureVector]:
        """
        Load features from file
        
        Args:
            filepath: Input file path
            
        Returns:
            List of feature vectors
        """
        # TODO: Implement feature loading
        # 1. Load from file
        # 2. Parse into vectors
        # 3. Validate features
        # 4. Return vectors
        pass