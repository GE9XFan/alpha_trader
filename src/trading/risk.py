#!/usr/bin/env python3
"""
Risk Management Module - UPDATED VERSION
Enforces all risk limits and portfolio constraints.
Now includes fundamental checks and earnings risk management.
Critical component - NEVER bypassed. Same rules for paper and live trading.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
from collections import defaultdict

from src.data.options_data import OptionsDataManager, OptionContract
from src.data.database import DatabaseManager, Position
from src.data.fundamental_data import FundamentalDataManager, EarningsData
from src.data.market_regime import MarketRegimeDetector, MarketRegime
from src.data.av_client import AlphaVantageClient
from src.trading.signals import TradingSignal, SignalType
from src.core.config import get_config, TradingConfig

logger = logging.getLogger(__name__)


class RiskCheckResult(Enum):
    """Risk check result types"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WARNING = "WARNING"


@dataclass
class RiskCheck:
    """Individual risk check result"""
    name: str
    result: RiskCheckResult
    message: str
    value: Optional[float] = None
    limit: Optional[float] = None
    
    def passed(self) -> bool:
        """Check if risk check passed"""
        return self.result in [RiskCheckResult.APPROVED, RiskCheckResult.WARNING]


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_value: float
    total_exposure: float
    position_count: int
    
    # Greeks (from Alpha Vantage)
    delta: float
    gamma: float
    theta: float
    vega: float
    
    # P&L
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk metrics
    var_95: float  # Value at Risk (95%)
    max_drawdown: float
    sharpe_ratio: float
    
    # Concentration
    largest_position_pct: float
    concentration_risk: Dict[str, float]
    
    # Fundamental risk
    earnings_exposure: int  # Number of positions with upcoming earnings
    avg_fundamental_score: float
    
    # Market regime
    regime_risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_value': self.total_value,
            'exposure': self.total_exposure,
            'positions': self.position_count,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'daily_pnl': self.daily_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'var_95': self.var_95,
            'max_drawdown': self.max_drawdown,
            'earnings_exposure': self.earnings_exposure,
            'fundamental_score': self.avg_fundamental_score,
            'regime_risk': self.regime_risk_score
        }


class RiskManager:
    """
    Risk management - NEVER BYPASSED
    Enforces all risk limits including fundamental and earnings checks.
    Every trade must pass all risk checks.
    """
    
    def __init__(self,
                 config: TradingConfig,
                 options_data: OptionsDataManager,
                 db: DatabaseManager,
                 fundamental_data: Optional[FundamentalDataManager] = None,
                 market_regime: Optional[MarketRegimeDetector] = None,
                 av_client: Optional[AlphaVantageClient] = None):
        """
        Initialize RiskManager with enhanced capabilities
        
        Args:
            config: Trading configuration
            options_data: Options data manager for Greeks (from AV)
            db: Database manager for position tracking
            fundamental_data: Optional fundamental data manager
            market_regime: Optional market regime detector
            av_client: Optional Alpha Vantage client for sentiment
        """
        self.config = config
        self.options = options_data
        self.db = db
        self.fundamentals = fundamental_data  # NEW: For earnings/fundamental checks
        self.regime = market_regime  # NEW: For regime-based risk
        self.av_client = av_client  # NEW: For sentiment risk
        
        # Risk limits from config
        self.max_positions = config.risk.max_positions
        self.max_position_size = config.risk.max_position_size
        self.daily_loss_limit = config.risk.daily_loss_limit
        self.greeks_limits = config.risk.greeks_limits
        
        # Fundamental limits
        self.block_earnings_days = config.risk.block_trades_before_earnings_days
        self.min_market_cap = config.risk.min_company_market_cap
        self.max_debt_equity = config.risk.max_debt_to_equity
        self.min_sentiment = config.risk.min_sentiment_score
        
        # Current state
        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.portfolio_greeks: Dict[str, float] = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }
        
        # Risk tracking
        self.risk_checks_performed = 0
        self.risk_rejections = 0
        self.risk_warnings = 0
        self.breach_history: List[Dict[str, Any]] = []
        self.rejection_reasons: Dict[str, int] = defaultdict(int)
        
        # Load current positions
        self._load_positions()
        
        logger.info("RiskManager initialized with enhanced checks:")
        logger.info(f"  Max positions: {self.max_positions}")
        logger.info(f"  Max position size: ${self.max_position_size}")
        logger.info(f"  Daily loss limit: ${self.daily_loss_limit}")
        logger.info(f"  Earnings block: {self.block_earnings_days} days")
        logger.info(f"  Greeks limits: {self.greeks_limits}")
    
    def _load_positions(self) -> None:
        """
        Load current positions from database
        """
        # TODO: Implement position loading
        # 1. Get positions from database
        # 2. Store in positions dict
        # 3. Calculate portfolio Greeks using AV data
        # 4. Calculate daily P&L
        # 5. Log loaded positions
        pass
    
    async def can_trade(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Check if trade is allowed - CRITICAL GATE
        Now includes fundamental and earnings checks.
        
        Args:
            signal: Trading signal to check
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # TODO: Implement comprehensive risk check
        # 1. Run all risk checks
        # 2. Include fundamental checks
        # 3. Include earnings checks
        # 4. Include sentiment checks
        # 5. Include regime checks
        # 6. Collect results
        # 7. Determine overall result
        # 8. Update signal with risk notes
        # 9. Log decision
        # 10. Return result
        pass
    
    def run_all_checks(self, signal: TradingSignal) -> List[RiskCheck]:
        """
        Run all risk checks on signal including new checks
        
        Args:
            signal: Trading signal
            
        Returns:
            List of risk check results
        """
        checks = []
        
        # Traditional risk checks
        checks.append(self._check_position_count())
        checks.append(self._check_position_size(signal))
        checks.append(self._check_daily_loss())
        checks.extend(self._check_greeks_limits(signal))
        checks.append(self._check_concentration(signal))
        checks.append(self._check_correlation(signal))
        
        # NEW: Fundamental checks
        if self.fundamentals:
            checks.append(await self._check_earnings_risk(signal))
            checks.append(await self._check_fundamental_health(signal))
            checks.append(await self._check_debt_levels(signal))
        
        # NEW: Sentiment checks
        if self.av_client:
            checks.append(await self._check_sentiment_risk(signal))
            checks.append(await self._check_news_volume(signal))
        
        # NEW: Regime checks
        if self.regime:
            checks.append(await self._check_regime_compatibility(signal))
            checks.append(await self._check_volatility_regime(signal))
        
        # Time-based checks
        checks.append(self._check_time_of_day(signal))
        checks.append(self._check_market_conditions(signal))
        
        self.risk_checks_performed += 1
        return checks
    
    async def _check_earnings_risk(self, signal: TradingSignal) -> RiskCheck:
        """
        Check if earnings announcement is too close
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        if not self.fundamentals:
            return RiskCheck(
                name="earnings_risk",
                result=RiskCheckResult.WARNING,
                message="Earnings check not available"
            )
        
        # TODO: Implement earnings risk check
        # 1. Get earnings date for symbol
        # 2. Calculate days until earnings
        # 3. Check against threshold
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    async def _check_fundamental_health(self, signal: TradingSignal) -> RiskCheck:
        """
        Check company fundamental health
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        if not self.fundamentals:
            return RiskCheck(
                name="fundamental_health",
                result=RiskCheckResult.WARNING,
                message="Fundamental check not available"
            )
        
        # TODO: Implement fundamental health check
        # 1. Get company metrics
        # 2. Check market cap
        # 3. Check profitability
        # 4. Get fundamental score
        # 5. Create RiskCheck result
        # 6. Return result
        pass
    
    async def _check_debt_levels(self, signal: TradingSignal) -> RiskCheck:
        """
        Check company debt levels
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        if not self.fundamentals:
            return RiskCheck(
                name="debt_levels",
                result=RiskCheckResult.WARNING,
                message="Debt check not available"
            )
        
        # TODO: Implement debt check
        # 1. Get debt/equity ratio
        # 2. Compare to threshold
        # 3. Check current ratio
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    async def _check_sentiment_risk(self, signal: TradingSignal) -> RiskCheck:
        """
        Check sentiment risk from news
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        if not self.av_client:
            return RiskCheck(
                name="sentiment_risk",
                result=RiskCheckResult.WARNING,
                message="Sentiment check not available"
            )
        
        # TODO: Implement sentiment risk check
        # 1. Get current sentiment
        # 2. Check against threshold
        # 3. Check sentiment trend
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    async def _check_news_volume(self, signal: TradingSignal) -> RiskCheck:
        """
        Check if news volume indicates unusual activity
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        # TODO: Implement news volume check
        # 1. Get news article count
        # 2. Check if abnormal
        # 3. Check negative news count
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    async def _check_regime_compatibility(self, signal: TradingSignal) -> RiskCheck:
        """
        Check if signal compatible with market regime
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        if not self.regime:
            return RiskCheck(
                name="regime_compatibility",
                result=RiskCheckResult.WARNING,
                message="Regime check not available"
            )
        
        # TODO: Implement regime compatibility check
        # 1. Get current regime
        # 2. Check signal compatibility
        # 3. Apply regime adjustments
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    async def _check_volatility_regime(self, signal: TradingSignal) -> RiskCheck:
        """
        Check volatility regime risk
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        # TODO: Implement volatility regime check
        # 1. Get volatility regime
        # 2. Check if too high
        # 3. Adjust position limits
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    def _check_position_count(self) -> RiskCheck:
        """
        Check if position count limit exceeded
        
        Returns:
            Risk check result
        """
        # TODO: Implement position count check
        # 1. Count active positions
        # 2. Compare to limit
        # 3. Create RiskCheck result
        # 4. Return result
        pass
    
    def _check_position_size(self, signal: TradingSignal) -> RiskCheck:
        """
        Check if position size within limits
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        # TODO: Implement position size check
        # 1. Calculate position value
        # 2. Apply regime adjustment
        # 3. Compare to limit
        # 4. Check as % of portfolio
        # 5. Create RiskCheck result
        # 6. Return result
        pass
    
    def _check_daily_loss(self) -> RiskCheck:
        """
        Check if daily loss limit exceeded
        
        Returns:
            Risk check result
        """
        # TODO: Implement daily loss check
        # 1. Get current daily P&L
        # 2. Compare to limit
        # 3. Calculate remaining room
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    def _check_greeks_limits(self, signal: TradingSignal) -> List[RiskCheck]:
        """
        Check if Greeks would exceed limits using AV data
        
        Args:
            signal: Trading signal
            
        Returns:
            List of Greeks check results
        """
        # TODO: Implement Greeks limit checks using AV data
        # 1. Get Greeks from option (already from AV)
        # 2. Calculate projected portfolio Greeks
        # 3. Check each Greek against limits
        # 4. Create RiskCheck for each
        # 5. Return results
        pass
    
    def _project_greeks_impact(self, signal: TradingSignal) -> Dict[str, float]:
        """
        Project Greeks impact using Alpha Vantage data
        
        Args:
            signal: Trading signal
            
        Returns:
            Dictionary of projected Greeks changes (from AV!)
        """
        # TODO: Implement Greeks projection using AV data
        # 1. Get option from signal
        # 2. Greeks are already in option from AV
        # 3. Multiply by position size
        # 4. Return Greeks dict
        pass
    
    def _check_concentration(self, signal: TradingSignal) -> RiskCheck:
        """
        Check concentration risk
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        # TODO: Implement concentration check
        # 1. Calculate position % of portfolio
        # 2. Check symbol concentration
        # 3. Check sector concentration
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    def _check_correlation(self, signal: TradingSignal) -> RiskCheck:
        """
        Check correlation risk
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        # TODO: Implement correlation check
        # 1. Check correlation with existing
        # 2. Calculate portfolio correlation
        # 3. Check if too correlated
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    def _check_time_of_day(self, signal: TradingSignal) -> RiskCheck:
        """
        Check time of day risk
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        # TODO: Implement time check
        # 1. Get current time
        # 2. Check market hours
        # 3. Check cutoff times
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    def _check_market_conditions(self, signal: TradingSignal) -> RiskCheck:
        """
        Check overall market conditions
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        # TODO: Implement market conditions check
        # 1. Check VIX level
        # 2. Check market breadth
        # 3. Check volume patterns
        # 4. Create RiskCheck result
        # 5. Return result
        pass
    
    def update_position(self, symbol: str, position: Position) -> None:
        """
        Update position tracking
        
        Args:
            symbol: Stock symbol
            position: Position details
        """
        # TODO: Implement position update
        # 1. Update positions dict
        # 2. Recalculate portfolio Greeks (using AV data)
        # 3. Update database
        # 4. Log update
        pass
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """
        Close position and calculate P&L
        
        Args:
            symbol: Stock symbol
            exit_price: Exit price per contract
            
        Returns:
            Realized P&L
        """
        # TODO: Implement position closing
        # 1. Get position details
        # 2. Calculate P&L
        # 3. Update daily P&L
        # 4. Remove from positions
        # 5. Recalculate Greeks
        # 6. Update database
        # 7. Return P&L
        pass
    
    def _recalculate_portfolio_greeks(self) -> None:
        """
        Recalculate total portfolio Greeks using Alpha Vantage data
        """
        # TODO: Implement Greeks recalculation using AV
        # 1. Initialize totals
        # 2. For each position:
        #    a. Get option from cache
        #    b. Greeks already in option from AV
        #    c. Multiply by position quantity
        #    d. Add to totals
        # 3. Store results
        # 4. Check for breaches
        pass
    
    def calculate_portfolio_risk(self) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics
        
        Returns:
            PortfolioRisk object
        """
        # TODO: Implement portfolio risk calculation
        # 1. Calculate total value
        # 2. Calculate exposure
        # 3. Aggregate Greeks (from AV)
        # 4. Calculate VaR
        # 5. Calculate drawdown
        # 6. Calculate Sharpe
        # 7. Check concentration
        # 8. Get earnings exposure
        # 9. Get fundamental scores
        # 10. Get regime risk
        # 11. Return PortfolioRisk
        pass
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        Args:
            confidence: Confidence level (default 95%)
            
        Returns:
            VaR amount
        """
        # TODO: Implement VaR calculation
        # 1. Get historical returns
        # 2. Calculate portfolio returns
        # 3. Find percentile
        # 4. Adjust for regime
        # 5. Return VaR
        pass
    
    def check_stop_loss(self, position: Position) -> bool:
        """
        Check if position hit stop loss
        
        Args:
            position: Position to check
            
        Returns:
            True if stop loss hit
        """
        # TODO: Implement stop loss check
        # 1. Calculate current P&L %
        # 2. Check against stop loss
        # 3. Check trailing stop
        # 4. Check fundamental deterioration
        # 5. Return result
        pass
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Get comprehensive risk report
        
        Returns:
            Risk report dictionary
        """
        # TODO: Implement risk reporting
        # 1. Get portfolio risk metrics
        # 2. List all positions
        # 3. Show Greeks (from AV)
        # 4. Show P&L
        # 5. Show limit usage
        # 6. Show breach history
        # 7. Show rejection reasons
        # 8. Show fundamental risks
        # 9. Show regime risks
        # 10. Return report
        pass
    
    def emergency_close_all(self) -> List[str]:
        """
        Emergency close all positions
        
        Returns:
            List of closed symbols
        """
        # TODO: Implement emergency close
        # 1. Get all positions
        # 2. Mark for closing
        # 3. Skip normal checks
        # 4. Update state
        # 5. Return closed list
        pass
    
    def reset_daily_metrics(self) -> None:
        """
        Reset daily metrics (called at market open)
        """
        # TODO: Implement daily reset
        # 1. Reset daily P&L
        # 2. Clear daily breach history
        # 3. Update database
        # 4. Log reset
        pass
    
    def log_risk_breach(self, 
                       check_name: str,
                       value: float,
                       limit: float) -> None:
        """
        Log risk limit breach
        
        Args:
            check_name: Name of breached check
            value: Current value
            limit: Limit that was breached
        """
        # TODO: Implement breach logging
        # 1. Create breach record
        # 2. Add to history
        # 3. Update rejection reasons
        # 4. Store in database
        # 5. Send alert if critical
        # 6. Log breach
        pass
    
    def get_remaining_capacity(self) -> Dict[str, Any]:
        """
        Get remaining risk capacity
        
        Returns:
            Dictionary with remaining capacity
        """
        # TODO: Implement capacity calculation
        # 1. Calculate position slots
        # 2. Calculate loss capacity
        # 3. Calculate Greeks room
        # 4. Calculate earnings capacity
        # 5. Return capacity dict
        pass
    
    def should_reduce_risk(self) -> bool:
        """
        Check if risk should be reduced based on multiple factors
        
        Returns:
            True if risk reduction recommended
        """
        # TODO: Implement risk reduction check
        # 1. Check P&L trend
        # 2. Check Greeks usage
        # 3. Check market conditions
        # 4. Check regime
        # 5. Check fundamental trends
        # 6. Return recommendation
        pass
    
    def get_rejection_analysis(self) -> Dict[str, Any]:
        """
        Analyze rejection patterns
        
        Returns:
            Analysis of rejection reasons
        """
        return {
            'total_rejections': self.risk_rejections,
            'rejection_rate': self.risk_rejections / max(1, self.risk_checks_performed),
            'top_rejection_reasons': dict(sorted(
                self.rejection_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }