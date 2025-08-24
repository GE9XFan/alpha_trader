#!/usr/bin/env python3
"""
Data Validation Module
Validates data quality from all sources including Alpha Vantage, IBKR, and calculated values.
Critical for ensuring data integrity before trading decisions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from enum import Enum

from src.data.av_client import OptionData, TechnicalIndicator, SentimentData
from src.data.market_data import MarketSnapshot
from src.analytics.features import FeatureVector
from src.core.config import get_config

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types"""
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


@dataclass
class ValidationError:
    """Validation error details"""
    field: str
    value: Any
    error_type: str
    message: str
    severity: ValidationResult
    
    def __str__(self) -> str:
        return f"{self.severity.value}: {self.field} = {self.value} - {self.message}"


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    timestamp: datetime
    data_source: str
    total_checks: int
    passed_checks: int
    warnings: List[ValidationError]
    errors: List[ValidationError]
    quality_score: float  # 0-100
    
    @property
    def is_valid(self) -> bool:
        """Check if data passes validation"""
        return len(self.errors) == 0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate"""
        return self.passed_checks / max(1, self.total_checks)
    
    def get_summary(self) -> str:
        """Get summary string"""
        return (f"Quality Score: {self.quality_score:.1f}% | "
                f"Passed: {self.passed_checks}/{self.total_checks} | "
                f"Warnings: {len(self.warnings)} | Errors: {len(self.errors)}")


class DataValidator:
    """
    Validates data quality from all sources
    Ensures data integrity before use in trading decisions
    """
    
    def __init__(self):
        """Initialize DataValidator"""
        self.config = get_config()
        
        # Validation statistics
        self.total_validations = 0
        self.validation_failures = 0
        self.common_errors: Dict[str, int] = {}
        
        # Validation thresholds
        self.thresholds = {
            'max_spread_pct': 0.05,      # 5% max bid-ask spread
            'max_iv': 3.0,                # 300% max IV
            'min_iv': 0.01,               # 1% min IV
            'max_price_change_pct': 0.20, # 20% max price change
            'stale_data_seconds': 60,     # Data older than 60s is stale
        }
        
        logger.info("DataValidator initialized")
    
    def validate_av_greeks(self, option_data: OptionData) -> DataQualityReport:
        """
        Sanity check Alpha Vantage Greeks
        
        Args:
            option_data: Option data from Alpha Vantage
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement Greeks validation
        # 1. Check delta bounds:
        #    - Calls: 0 <= delta <= 1
        #    - Puts: -1 <= delta <= 0
        # 2. Check gamma: Always positive
        # 3. Check theta: Usually negative
        # 4. Check vega: Always positive
        # 5. Check rho bounds
        # 6. Cross-validate Greeks relationships
        # 7. Check for NaN or infinite values
        # 8. Create DataQualityReport
        # 9. Return report
        pass
    
    def validate_option_data(self, option: OptionData) -> DataQualityReport:
        """
        Comprehensive option data validation
        
        Args:
            option: Option data to validate
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement option validation
        # 1. Validate Greeks (call validate_av_greeks)
        # 2. Check bid <= ask
        # 3. Check spread reasonableness
        # 4. Check IV bounds
        # 5. Check volume >= 0
        # 6. Check open interest >= 0
        # 7. Check strike > 0
        # 8. Check days to expiry
        # 9. Validate option type
        # 10. Create report
        pass
    
    def validate_market_snapshot(self, snapshot: MarketSnapshot) -> DataQualityReport:
        """
        Validate market data snapshot
        
        Args:
            snapshot: Market snapshot to validate
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement market data validation
        # 1. Check bid <= ask
        # 2. Check prices > 0
        # 3. Check volume >= 0
        # 4. Check high >= low
        # 5. Check close within high/low
        # 6. Check timestamp freshness
        # 7. Check for stuck prices
        # 8. Create report
        pass
    
    def validate_technical_indicators(self, 
                                    indicators: Dict[str, float]) -> DataQualityReport:
        """
        Validate technical indicator ranges
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement indicator validation
        # 1. RSI: 0-100
        # 2. MACD: Reasonable bounds
        # 3. Bollinger Bands: Upper > Lower
        # 4. ATR: Positive
        # 5. ADX: 0-100
        # 6. Stochastic: 0-100
        # 7. Check for NaN values
        # 8. Create report
        pass
    
    def validate_sentiment_data(self, sentiment: SentimentData) -> DataQualityReport:
        """
        Validate sentiment data
        
        Args:
            sentiment: Sentiment data to validate
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement sentiment validation
        # 1. Check sentiment score: -1 to 1
        # 2. Check relevance score: 0 to 1
        # 3. Check article count >= 0
        # 4. Validate sentiment label
        # 5. Check timestamp validity
        # 6. Create report
        pass
    
    def validate_feature_vector(self, features: FeatureVector) -> DataQualityReport:
        """
        Validate ML feature vector
        
        Args:
            features: Feature vector to validate
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement feature validation
        # 1. Check feature count matches expected
        # 2. Check for NaN values
        # 3. Check for infinite values
        # 4. Validate feature ranges
        # 5. Check feature names match
        # 6. Validate timestamp
        # 7. Create report
        pass
    
    def validate_price_data(self, 
                          current_price: float,
                          previous_price: float,
                          max_change_pct: float = 0.20) -> Tuple[bool, Optional[str]]:
        """
        Validate price movement reasonableness
        
        Args:
            current_price: Current price
            previous_price: Previous price
            max_change_pct: Maximum allowed change percentage
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # TODO: Implement price validation
        # 1. Check prices are positive
        # 2. Calculate percentage change
        # 3. Check against threshold
        # 4. Return validation result
        pass
    
    def validate_spread(self, bid: float, ask: float, 
                       last: float) -> Tuple[bool, Optional[str]]:
        """
        Validate bid-ask spread
        
        Args:
            bid: Bid price
            ask: Ask price
            last: Last traded price
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # TODO: Implement spread validation
        # 1. Check bid <= ask
        # 2. Calculate spread percentage
        # 3. Check spread reasonableness
        # 4. Check last price position
        # 5. Return validation result
        pass
    
    def validate_volume(self, volume: int, 
                       avg_volume: float) -> Tuple[bool, Optional[str]]:
        """
        Validate volume data
        
        Args:
            volume: Current volume
            avg_volume: Average volume
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # TODO: Implement volume validation
        # 1. Check volume >= 0
        # 2. Check for unusual spikes
        # 3. Check for suspiciously low volume
        # 4. Return validation result
        pass
    
    def validate_timestamp(self, timestamp: datetime,
                          max_age_seconds: int = 60) -> Tuple[bool, Optional[str]]:
        """
        Validate data timestamp
        
        Args:
            timestamp: Data timestamp
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # TODO: Implement timestamp validation
        # 1. Check timestamp not in future
        # 2. Check age against threshold
        # 3. Check for suspicious patterns
        # 4. Return validation result
        pass
    
    def detect_data_anomalies(self, data: pd.DataFrame) -> List[str]:
        """
        Detect anomalies in data series
        
        Args:
            data: DataFrame to check
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # TODO: Implement anomaly detection
        # 1. Check for gaps in data
        # 2. Check for duplicate timestamps
        # 3. Check for stuck values
        # 4. Check for outliers (3+ std dev)
        # 5. Check for pattern breaks
        # 6. Return anomalies list
        pass
    
    def validate_options_chain(self, chain: List[OptionData]) -> DataQualityReport:
        """
        Validate entire options chain
        
        Args:
            chain: List of options in chain
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement chain validation
        # 1. Validate each option
        # 2. Check strike spacing consistency
        # 3. Check for missing strikes
        # 4. Validate put-call parity
        # 5. Check IV smile/skew
        # 6. Create comprehensive report
        pass
    
    def validate_fundamental_data(self, 
                                 fundamentals: Dict[str, Any]) -> DataQualityReport:
        """
        Validate fundamental data
        
        Args:
            fundamentals: Fundamental data dictionary
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement fundamental validation
        # 1. Check for required fields
        # 2. Validate ratio bounds
        # 3. Check for negative values where inappropriate
        # 4. Cross-validate related metrics
        # 5. Check data age
        # 6. Create report
        pass
    
    def cross_validate_data_sources(self,
                                  av_data: Dict[str, Any],
                                  ibkr_data: Dict[str, Any]) -> DataQualityReport:
        """
        Cross-validate data between Alpha Vantage and IBKR
        
        Args:
            av_data: Data from Alpha Vantage
            ibkr_data: Data from IBKR
            
        Returns:
            DataQualityReport
        """
        errors = []
        warnings = []
        checks_passed = 0
        total_checks = 0
        
        # TODO: Implement cross-validation
        # 1. Compare prices (should be close)
        # 2. Compare volumes
        # 3. Compare option strikes
        # 4. Check timestamp alignment
        # 5. Flag significant discrepancies
        # 6. Create report
        pass
    
    def calculate_data_quality_score(self, reports: List[DataQualityReport]) -> float:
        """
        Calculate overall data quality score
        
        Args:
            reports: List of quality reports
            
        Returns:
            Quality score (0-100)
        """
        # TODO: Implement scoring
        # 1. Weight each report by importance
        # 2. Calculate weighted average
        # 3. Apply penalties for errors
        # 4. Return score
        pass
    
    def should_halt_trading(self, report: DataQualityReport) -> bool:
        """
        Determine if trading should halt due to data issues
        
        Args:
            report: Data quality report
            
        Returns:
            True if trading should halt
        """
        # Halt if critical errors or quality score too low
        return not report.is_valid or report.quality_score < 70
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_validations': self.total_validations,
            'failure_rate': self.validation_failures / max(1, self.total_validations),
            'common_errors': dict(sorted(self.common_errors.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:10])
        }
    
    def log_validation_error(self, error: ValidationError) -> None:
        """
        Log validation error for tracking
        
        Args:
            error: Validation error
        """
        error_key = f"{error.field}:{error.error_type}"
        self.common_errors[error_key] = self.common_errors.get(error_key, 0) + 1
        
        if error.severity == ValidationResult.INVALID:
            logger.error(str(error))
        else:
            logger.warning(str(error))