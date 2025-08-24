#!/usr/bin/env python3
"""
Feature Engineering Module
Calculates features for ML model - feeds the entire prediction pipeline.
Reused for both training and live prediction with identical feature generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import talib
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from collections import deque

from src.data.options_data import OptionsDataManager
from src.data.market_data import MarketDataManager
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
    Calculates consistent features for training and live prediction.
    Total of 28 features across 4 categories.
    """
    
    def __init__(self, 
                 options_manager: OptionsDataManager,
                 market_manager: MarketDataManager):
        """
        Initialize FeatureEngine
        
        Args:
            options_manager: Options data manager for options features
            market_manager: Market data manager for price data
        """
        self.options = options_manager
        self.market = market_manager
        self.config = get_config()
        
        # Define feature names for consistency - CRITICAL for ML
        self.feature_names = [
            # Price action features (5)
            'returns_5m',       # 5-minute return
            'returns_30m',      # 30-minute return
            'returns_1h',       # 1-hour return
            'volume_ratio',     # Current vs average volume
            'high_low_ratio',   # (High - Low) / Close
            
            # Technical indicators (10)
            'rsi',              # Relative Strength Index
            'macd_signal',      # MACD signal line
            'macd_histogram',   # MACD histogram
            'bb_upper',         # Bollinger Band upper
            'bb_lower',         # Bollinger Band lower
            'bb_position',      # Position within bands
            'atr',              # Average True Range
            'adx',              # Average Directional Index
            'obv_slope',        # On-Balance Volume slope
            'vwap_distance',    # Distance from VWAP
            
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
            'market_regime'     # Market regime indicator
        ]
        
        # Feature calculation cache
        self.feature_cache: Dict[str, FeatureVector] = {}
        self.cache_ttl = 60  # seconds
        
        # Historical data storage for correlations
        self.correlation_window = 100  # bars
        self.historical_data: Dict[str, deque] = {}
        
        # Feature statistics for normalization
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"FeatureEngine initialized with {len(self.feature_names)} features")
    
    def calculate_features(self, 
                         symbol: str,
                         historical_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate all features - CRITICAL FOR ML
        Same features for training and live prediction.
        
        Args:
            symbol: Stock symbol
            historical_data: DataFrame with OHLCV data
            
        Returns:
            Feature array in consistent order
        """
        # TODO: Implement feature calculation
        # 1. Check cache first
        # 2. Calculate price action features
        # 3. Calculate technical indicators
        # 4. Calculate options features
        # 5. Calculate market structure features
        # 6. Combine into array
        # 7. Handle NaN values
        # 8. Cache result
        # 9. Return feature array
        pass
    
    def _calculate_price_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate price action features
        
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
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical indicators using TA-Lib
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of technical indicators
        """
        # TODO: Implement technical indicator calculation
        # 1. Extract price arrays
        # 2. Calculate RSI
        # 3. Calculate MACD
        # 4. Calculate Bollinger Bands
        # 5. Calculate ATR
        # 6. Calculate ADX
        # 7. Calculate OBV slope
        # 8. Calculate VWAP distance
        # 9. Handle TA-Lib errors
        # 10. Return indicators dict
        pass
    
    def _calculate_obv_slope(self, close: np.ndarray, volume: np.ndarray) -> float:
        """
        Calculate On-Balance Volume slope
        
        Args:
            close: Close prices
            volume: Volume data
            
        Returns:
            OBV slope
        """
        # TODO: Implement OBV slope calculation
        # 1. Calculate OBV
        # 2. Fit linear regression
        # 3. Return slope
        # 4. Handle edge cases
        pass
    
    def _calculate_vwap_distance(self, data: pd.DataFrame) -> float:
        """
        Calculate distance from VWAP
        
        Args:
            data: OHLCV data
            
        Returns:
            Percentage distance from VWAP
        """
        # TODO: Implement VWAP distance
        # 1. Calculate VWAP
        # 2. Get current price
        # 3. Calculate percentage distance
        # 4. Return distance
        pass
    
    def _calculate_options_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate options-specific features
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of options features
        """
        # TODO: Implement options feature calculation
        # 1. Get ATM options
        # 2. Calculate IV rank
        # 3. Calculate IV percentile
        # 4. Get put/call ratio
        # 5. Calculate gamma exposure
        # 6. Find delta neutral price
        # 7. Calculate max pain distance
        # 8. Get option volumes
        # 9. Handle missing data
        # 10. Return features dict
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
        # 3. Get VIX level
        # 4. Calculate term structure
        # 5. Determine market regime
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
    
    def _determine_market_regime(self, vix: float, trend: float) -> float:
        """
        Determine market regime (trending, ranging, volatile)
        
        Args:
            vix: VIX level
            trend: Trend strength
            
        Returns:
            Regime indicator (0-1)
        """
        # TODO: Implement regime detection
        # 1. Classify VIX level
        # 2. Assess trend strength
        # 3. Combine into regime score
        # 4. Return normalized score
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
        # 4. Check value ranges
        # 5. Return validation result
        pass
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from model
        
        Returns:
            Series with feature importance scores
        """
        # TODO: Implement importance retrieval
        # 1. Load model if available
        # 2. Extract feature importance
        # 3. Create Series with names
        # 4. Sort by importance
        # 5. Return Series
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
        # TODO: Implement category mapping
        # 1. Define categories
        # 2. Map features to categories
        # 3. Return organized dict
        pass
    
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