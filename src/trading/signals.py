#!/usr/bin/env python3
"""
Signal Generation Module - UPDATED VERSION
Generates trading signals by combining ML predictions with trading rules.
Now includes sentiment signals and regime adjustments.
Brain of the system - reused by both paper and live trading.
"""

from datetime import datetime, time, timedelta
from typing import Optional, Dict, List, Any, Tuple
import asyncio
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
from collections import deque
from enum import Enum

from src.analytics.ml_model import MLPredictor, Prediction
from src.analytics.features import FeatureEngine
from src.data.market_data import MarketDataManager
from src.data.options_data import OptionsDataManager, OptionContract
from src.data.av_client import AlphaVantageClient, SentimentData
from src.data.fundamental_data import FundamentalDataManager
from src.data.market_regime import MarketRegimeDetector, MarketRegime
from src.core.config import get_config, TradingConfig

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal type enumeration"""
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    CLOSE_ALL = "CLOSE_ALL"  # Emergency close
    # New signal types
    SENTIMENT_BUY = "SENTIMENT_BUY"
    SENTIMENT_SELL = "SENTIMENT_SELL"
    REGIME_ADJUST = "REGIME_ADJUST"


@dataclass
class TradingSignal:
    """Complete trading signal with all required information"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float
    
    # Signal source
    source: str = "ML"  # 'ML', 'SENTIMENT', 'TECHNICAL', 'REGIME', 'COMPOSITE'
    
    # Option details
    option: Optional[OptionContract] = None
    contracts: int = 5  # Default contract size
    
    # ML prediction details
    prediction: Optional[Prediction] = None
    features: Optional[np.ndarray] = None
    
    # Sentiment data
    sentiment_score: Optional[float] = None
    news_count: Optional[int] = None
    
    # Market regime
    market_regime: Optional[MarketRegime] = None
    regime_adjustment: float = 1.0  # Position size adjustment
    
    # Risk checks
    risk_approved: bool = False
    risk_notes: List[str] = field(default_factory=list)
    
    # Execution details
    urgency: str = "NORMAL"  # NORMAL, HIGH, URGENT
    expiry_time: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'source': self.source,
            'option': self.option.to_dict() if self.option else None,
            'contracts': self.contracts,
            'features': self.features.tolist() if self.features is not None else None,
            'sentiment_score': self.sentiment_score,
            'market_regime': self.market_regime.value if self.market_regime else None,
            'regime_adjustment': self.regime_adjustment,
            'risk_approved': self.risk_approved,
            'risk_notes': self.risk_notes,
            'urgency': self.urgency,
            'metadata': self.metadata
        }
    
    def is_valid(self) -> bool:
        """Check if signal is valid for execution"""
        return (
            self.risk_approved and 
            self.confidence >= 0.6 and 
            self.option is not None and
            self.signal_type in [SignalType.BUY_CALL, SignalType.BUY_PUT, SignalType.CLOSE]
        )


class SignalGenerator:
    """
    Generates trading signals - BRAIN OF THE SYSTEM
    Combines ML predictions, sentiment analysis, and regime detection.
    Reused by paper and live trading.
    """
    
    def __init__(self,
                 ml_model: MLPredictor,
                 feature_engine: FeatureEngine,
                 market_data: MarketDataManager,
                 options_data: OptionsDataManager,
                 av_client: AlphaVantageClient,
                 fundamental_manager: Optional[FundamentalDataManager] = None,
                 market_regime: Optional[MarketRegimeDetector] = None):
        """
        Initialize SignalGenerator with multi-source capabilities
        
        Args:
            ml_model: ML predictor for signals
            feature_engine: Feature calculator
            market_data: Market data manager
            options_data: Options data manager
            av_client: Alpha Vantage client for sentiment
            fundamental_manager: Optional fundamental data
            market_regime: Optional regime detector
        """
        self.ml = ml_model
        self.features = feature_engine
        self.market = market_data
        self.options = options_data
        self.av_client = av_client  # NEW: For sentiment
        self.fundamentals = fundamental_manager  # NEW: For earnings checks
        self.regime = market_regime  # NEW: For regime adjustments
        self.config = get_config()
        
        # Signal tracking
        self.signals_today: List[TradingSignal] = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.signal_history: deque = deque(maxlen=1000)
        
        # Timing constraints
        self.min_time_between_signals = self.config.risk.min_time_between_signals  # 300 seconds
        self.market_open_delay = 60  # Wait 1 minute after open
        self.market_close_cutoff = 60  # Stop 1 minute before close
        
        # Signal filters
        self.enable_ml_signals = True
        self.enable_technical_signals = False  
        self.enable_sentiment_signals = self.config.ml.use_sentiment_features  # NEW
        self.enable_fundamental_filters = self.config.ml.use_fundamental_features  # NEW
        self.enable_regime_adjustments = True  # NEW
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_executed = 0
        self.signals_by_source: Dict[str, int] = {
            'ML': 0, 'SENTIMENT': 0, 'TECHNICAL': 0, 'REGIME': 0, 'COMPOSITE': 0
        }
        
        logger.info("SignalGenerator initialized with multi-source capabilities")
    
    async def generate_signals(self, 
                              symbols: List[str]) -> List[TradingSignal]:
        """
        Generate signals for symbols - CORE LOGIC
        Now combines ML, sentiment, and regime signals.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            List of trading signals
        """
        # TODO: Implement multi-source signal generation
        # 1. Check if market is open
        # 2. Get current market regime
        # 3. Filter symbols by timing and fundamentals
        # 4. For each symbol:
        #    a. Generate ML signals
        #    b. Generate sentiment signals
        #    c. Apply regime adjustments
        #    d. Combine signals
        # 5. Filter and prioritize signals
        # 6. Return final signal list
        pass
    
    async def generate_ml_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """
        Generate ML-based signals
        
        Args:
            symbols: List of symbols
            
        Returns:
            List of ML signals
        """
        # TODO: Implement ML signal generation
        # 1. For each symbol:
        #    a. Check timing constraints
        #    b. Get historical data
        #    c. Calculate features (uses AV indicators)
        #    d. Get ML prediction
        #    e. Apply trading rules
        #    f. Select option contract
        #    g. Create signal
        # 2. Return ML signals
        pass
    
    async def generate_sentiment_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """
        Generate signals from news sentiment using Alpha Vantage
        
        Args:
            symbols: List of symbols
            
        Returns:
            List of sentiment-based signals
        """
        if not self.enable_sentiment_signals:
            return []
        
        # TODO: Implement sentiment signal generation
        # 1. Get news sentiment from av_client
        # 2. For each symbol with strong sentiment:
        #    a. Check sentiment threshold
        #    b. Check volume of news
        #    c. Get insider transactions
        #    d. Generate signal if criteria met
        # 3. Return sentiment signals
        pass
    
    async def apply_regime_adjustments(self, 
                                      signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Adjust signals based on market regime
        
        Args:
            signals: Original signals
            
        Returns:
            Regime-adjusted signals
        """
        if not self.regime or not self.enable_regime_adjustments:
            return signals
        
        # TODO: Implement regime adjustments
        # 1. Get current market regime
        # 2. For each signal:
        #    a. Apply position size adjustment
        #    b. Adjust confidence thresholds
        #    c. Filter inappropriate signals
        #    d. Add regime metadata
        # 3. Return adjusted signals
        pass
    
    async def check_fundamental_filters(self, symbol: str) -> Tuple[bool, str]:
        """
        Check fundamental filters including earnings
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (passes_filter, reason)
        """
        if not self.fundamentals or not self.enable_fundamental_filters:
            return True, "No fundamental filters"
        
        # TODO: Implement fundamental filtering
        # 1. Check earnings date proximity
        # 2. Check fundamental health score
        # 3. Check debt levels
        # 4. Check market cap
        # 5. Return filter result
        pass
    
    async def generate_signal_for_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate signal for a single symbol combining all sources
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Trading signal or None
        """
        # TODO: Implement composite signal generation
        # 1. Check timing constraints
        # 2. Check fundamental filters
        # 3. Get ML signal
        # 4. Get sentiment signal
        # 5. Combine signals
        # 6. Apply regime adjustment
        # 7. Select best option
        # 8. Create composite signal
        # 9. Return signal
        pass
    
    def _combine_signals(self, 
                        ml_signal: Optional[TradingSignal],
                        sentiment_signal: Optional[TradingSignal]) -> Optional[TradingSignal]:
        """
        Combine ML and sentiment signals into composite
        
        Args:
            ml_signal: ML-based signal
            sentiment_signal: Sentiment-based signal
            
        Returns:
            Combined signal or None
        """
        # TODO: Implement signal combination
        # 1. If both agree, increase confidence
        # 2. If disagree, use higher confidence
        # 3. Weight by source reliability
        # 4. Create composite signal
        # 5. Return combined signal
        pass
    
    def _check_timing_constraints(self, symbol: str) -> bool:
        """
        Check if enough time has passed since last signal
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if timing is valid
        """
        # TODO: Implement timing check
        # 1. Check last signal time
        # 2. Calculate time elapsed
        # 3. Check minimum time
        # 4. Check market hours
        # 5. Return validity
        pass
    
    def _apply_trading_rules(self, 
                            prediction: Prediction,
                            market_conditions: Dict[str, Any]) -> bool:
        """
        Apply additional trading rules beyond ML
        
        Args:
            prediction: ML prediction
            market_conditions: Current market state
            
        Returns:
            True if rules pass
        """
        # TODO: Implement trading rules
        # 1. Check confidence threshold
        # 2. Check market regime compatibility
        # 3. Check volatility levels
        # 4. Check correlation limits
        # 5. Check time of day rules
        # 6. Check sentiment alignment
        # 7. Return combined result
        pass
    
    def _select_option_contract(self, 
                               symbol: str,
                               signal_type: SignalType) -> Optional[OptionContract]:
        """
        Select best option contract to trade
        
        Args:
            symbol: Stock symbol
            signal_type: Type of signal
            
        Returns:
            Selected option contract or None
        """
        # TODO: Implement option selection
        # 1. Get ATM options from options manager
        # 2. Filter by DTE (0-7 days)
        # 3. Check liquidity (volume, OI)
        # 4. Check spread reasonableness
        # 5. Consider regime (longer DTE in volatile)
        # 6. Return best option
        pass
    
    def _calculate_position_size(self, 
                                signal: TradingSignal,
                                account_value: float) -> int:
        """
        Calculate position size with regime adjustment
        
        Args:
            signal: Trading signal
            account_value: Current account value
            
        Returns:
            Number of contracts
        """
        # TODO: Implement position sizing
        # 1. Get base size from config
        # 2. Apply Kelly criterion if enabled
        # 3. Apply regime adjustment
        # 4. Apply confidence scaling
        # 5. Round to contracts
        # 6. Return size
        pass
    
    async def generate_close_signals(self, 
                                    positions: Dict[str, Any]) -> List[TradingSignal]:
        """
        Generate signals to close existing positions
        
        Args:
            positions: Current positions
            
        Returns:
            List of close signals
        """
        # TODO: Implement close signal generation
        # 1. Check each position for:
        #    a. Stop loss conditions
        #    b. Profit targets
        #    c. 0DTE expiry
        #    d. ML exit signals
        #    e. Sentiment reversal
        #    f. Regime change
        # 2. Create close signals
        # 3. Return signals
        pass
    
    def _check_sentiment_reversal(self, 
                                 position: Dict[str, Any],
                                 current_sentiment: float) -> bool:
        """
        Check if sentiment has reversed against position
        
        Args:
            position: Position details
            current_sentiment: Current sentiment score
            
        Returns:
            True if sentiment reversed
        """
        # TODO: Implement sentiment reversal check
        # 1. Get position direction
        # 2. Check current sentiment
        # 3. Define reversal threshold
        # 4. Return reversal status
        pass
    
    def _check_regime_exit(self, 
                          position: Dict[str, Any],
                          current_regime: MarketRegime) -> bool:
        """
        Check if regime change warrants exit
        
        Args:
            position: Position details
            current_regime: Current market regime
            
        Returns:
            True if should exit due to regime
        """
        # TODO: Implement regime exit check
        # 1. Check if regime changed
        # 2. Check if new regime incompatible
        # 3. Return exit decision
        pass
    
    def validate_signal(self, signal: TradingSignal) -> Tuple[bool, List[str]]:
        """
        Validate signal before execution
        
        Args:
            signal: Trading signal
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        # TODO: Implement signal validation
        # 1. Check signal completeness
        # 2. Verify option exists
        # 3. Check market hours
        # 4. Verify confidence
        # 5. Check risk approval
        # 6. Check source validity
        # 7. Return validation result
        pass
    
    def filter_signals(self, 
                      signals: List[TradingSignal],
                      max_signals: int = 5) -> List[TradingSignal]:
        """
        Filter and prioritize signals
        
        Args:
            signals: List of signals
            max_signals: Maximum signals to return
            
        Returns:
            Filtered signal list
        """
        # TODO: Implement signal filtering
        # 1. Remove invalid signals
        # 2. Apply fundamental filters
        # 3. Sort by confidence and source
        # 4. Remove duplicates
        # 5. Apply max limit
        # 6. Return filtered list
        pass
    
    def get_signal_metrics(self) -> Dict[str, Any]:
        """
        Get signal generation metrics
        
        Returns:
            Metrics dictionary
        """
        # TODO: Implement metrics calculation
        # 1. Count signals by type
        # 2. Count signals by source
        # 3. Calculate average confidence
        # 4. Track execution rate
        # 5. Calculate timing stats
        # 6. Return metrics
        pass
    
    def emergency_close_all(self) -> List[TradingSignal]:
        """
        Generate emergency close signals for all positions
        
        Returns:
            List of close signals
        """
        # TODO: Implement emergency close
        # 1. Create CLOSE_ALL signal
        # 2. Set high urgency
        # 3. Skip normal checks
        # 4. Return signals
        pass
    
    def is_market_open(self) -> bool:
        """
        Check if market is open for trading
        
        Returns:
            True if market is open
        """
        # TODO: Implement market hours check
        # 1. Get current time in ET
        # 2. Check weekday
        # 3. Check time range
        # 4. Check holidays
        # 5. Apply delays/cutoffs
        # 6. Return result
        pass
    
    async def backtest_signals(self, 
                              historical_data: pd.DataFrame,
                              start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """
        Backtest signal generation
        
        Args:
            historical_data: Historical market data
            start_date: Backtest start
            end_date: Backtest end
            
        Returns:
            DataFrame with backtest results
        """
        # TODO: Implement backtesting
        # 1. Iterate through historical data
        # 2. Generate signals at each point
        # 3. Track hypothetical trades
        # 4. Calculate performance
        # 5. Return results
        pass