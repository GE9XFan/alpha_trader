#!/usr/bin/env python3
"""
Signal Generation Module
Generates trading signals by combining ML predictions with trading rules.
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
from src.core.config import get_config, TradingConfig

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal type enumeration"""
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    CLOSE_ALL = "CLOSE_ALL"  # Emergency close


@dataclass
class TradingSignal:
    """Complete trading signal with all required information"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float
    
    # Option details
    option: Optional[OptionContract] = None
    contracts: int = 5  # Default contract size
    
    # ML prediction details
    prediction: Optional[Prediction] = None
    features: Optional[np.ndarray] = None
    
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
            'option': self.option.to_dict() if self.option else None,
            'contracts': self.contracts,
            'features': self.features.tolist() if self.features is not None else None,
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
    Combines ML predictions with trading rules and market conditions.
    Reused by paper and live trading.
    """
    
    def __init__(self,
                 ml_model: MLPredictor,
                 feature_engine: FeatureEngine,
                 market_data: MarketDataManager,
                 options_data: OptionsDataManager):
        """
        Initialize SignalGenerator
        
        Args:
            ml_model: ML predictor for signals
            feature_engine: Feature calculator
            market_data: Market data manager
            options_data: Options data manager
        """
        self.ml = ml_model
        self.features = feature_engine
        self.market = market_data
        self.options = options_data
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
        self.enable_technical_signals = False  # Can add rule-based signals
        self.enable_options_flow_signals = False  # Can add flow-based signals
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_executed = 0
        
        logger.info("SignalGenerator initialized")
    
    async def generate_signals(self, 
                              symbols: List[str]) -> List[TradingSignal]:
        """
        Generate signals for symbols - CORE LOGIC
        Called by both paper and live traders.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            List of trading signals
        """
        # TODO: Implement signal generation
        # 1. Check if market is open
        # 2. Filter symbols by timing
        # 3. For each symbol:
        #    a. Get historical data
        #    b. Calculate features
        #    c. Get ML prediction
        #    d. Apply trading rules
        #    e. Select best option
        #    f. Create signal
        # 4. Filter signals
        # 5. Return signals
        pass
    
    async def generate_signal_for_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate signal for a single symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Trading signal or None
        """
        # TODO: Implement single symbol signal generation
        # 1. Check timing constraints
        # 2. Get market data
        # 3. Calculate features
        # 4. Get ML prediction
        # 5. Validate signal
        # 6. Find option contract
        # 7. Create TradingSignal
        # 8. Return signal
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
        # 2. Check market regime
        # 3. Check volatility levels
        # 4. Check correlation limits
        # 5. Check time of day rules
        # 6. Return combined result
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
        # 1. Get ATM options
        # 2. Filter by DTE (0-7 days)
        # 3. Select shortest DTE for gamma
        # 4. Verify liquidity
        # 5. Check spread
        # 6. Return best option
        pass
    
    def _calculate_position_size(self, 
                                signal: TradingSignal,
                                account_value: float) -> int:
        """
        Calculate position size in contracts
        
        Args:
            signal: Trading signal
            account_value: Current account value
            
        Returns:
            Number of contracts
        """
        # TODO: Implement position sizing
        # 1. Get max position size from config
        # 2. Calculate Kelly criterion
        # 3. Apply risk limits
        # 4. Round to contracts
        # 5. Return size
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
        # 1. Check each position
        # 2. Check stop loss conditions
        # 3. Check profit targets
        # 4. Check 0DTE expiry
        # 5. Check ML exit signals
        # 6. Create close signals
        # 7. Return signals
        pass
    
    def _check_stop_loss(self, 
                        position: Dict[str, Any]) -> bool:
        """
        Check if position hit stop loss
        
        Args:
            position: Position details
            
        Returns:
            True if stop loss hit
        """
        # TODO: Implement stop loss check
        # 1. Calculate current P&L
        # 2. Check against stop loss %
        # 3. Check trailing stop
        # 4. Return result
        pass
    
    def _check_profit_target(self, 
                            position: Dict[str, Any]) -> bool:
        """
        Check if position hit profit target
        
        Args:
            position: Position details
            
        Returns:
            True if profit target hit
        """
        # TODO: Implement profit target check
        # 1. Calculate current P&L
        # 2. Check against target
        # 3. Consider time in position
        # 4. Return result
        pass
    
    def _check_expiry(self, position: Dict[str, Any]) -> bool:
        """
        Check if position is expiring today
        
        Args:
            position: Position details
            
        Returns:
            True if expiring today
        """
        # TODO: Implement expiry check
        # 1. Get position expiry
        # 2. Check if today
        # 3. Check time to close
        # 4. Return result
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
        # 6. Return validation result
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
        # 2. Sort by confidence
        # 3. Remove duplicates
        # 4. Apply max limit
        # 5. Return filtered list
        pass
    
    def get_signal_metrics(self) -> Dict[str, Any]:
        """
        Get signal generation metrics
        
        Returns:
            Metrics dictionary
        """
        # TODO: Implement metrics calculation
        # 1. Count signals by type
        # 2. Calculate average confidence
        # 3. Track execution rate
        # 4. Calculate timing stats
        # 5. Return metrics
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
    
    def add_technical_signals(self, enabled: bool = True) -> None:
        """
        Enable/disable technical analysis signals
        
        Args:
            enabled: Whether to enable
        """
        # TODO: Implement technical signal toggle
        # 1. Set flag
        # 2. Initialize if needed
        # 3. Log change
        pass
    
    def add_options_flow_signals(self, enabled: bool = True) -> None:
        """
        Enable/disable options flow signals
        
        Args:
            enabled: Whether to enable
        """
        # TODO: Implement flow signal toggle
        # 1. Set flag
        # 2. Initialize if needed
        # 3. Log change
        pass
    
    def save_signals(self, filepath: str) -> bool:
        """
        Save signals to file for analysis
        
        Args:
            filepath: Output file path
            
        Returns:
            True if saved successfully
        """
        # TODO: Implement signal saving
        # 1. Convert signals to DataFrame
        # 2. Add metadata
        # 3. Save to file
        # 4. Return success
        pass
    
    def load_signals(self, filepath: str) -> List[TradingSignal]:
        """
        Load signals from file
        
        Args:
            filepath: Input file path
            
        Returns:
            List of signals
        """
        # TODO: Implement signal loading
        # 1. Load from file
        # 2. Parse into signals
        # 3. Validate signals
        # 4. Return list
        pass