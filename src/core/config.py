#!/usr/bin/env python3
"""
Trading Configuration Module - UPDATED VERSION
Single source of truth for all configuration across the system.
This module is imported by EVERY component and NEVER rewritten.
Updated with 600 calls/min rate limit and differentiated cache TTLs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import yaml
import os
from datetime import datetime, time
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class IBKRConfig:
    """Interactive Brokers connection configuration"""
    host: str = '127.0.0.1'
    paper_port: int = 7497  # Paper trading port
    live_port: int = 7496   # Live trading port
    client_id: int = 1
    timeout: int = 30  # Connection timeout in seconds
    
    def get_port(self, mode: str) -> int:
        """Get the appropriate port based on trading mode"""
        return self.paper_port if mode == 'paper' else self.live_port


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_positions: int = 5
    max_position_size: float = 10000.0  # USD per position
    daily_loss_limit: float = 1000.0    # USD daily stop loss
    
    # Portfolio Greeks limits
    greeks_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (-0.3, 0.3),    # Stay delta neutral
        'gamma': (-0.5, 0.5),     # Control gamma risk
        'vega': (-500, 500),      # Limit vega exposure
        'theta': (-200, float('inf'))  # Max theta burn
    })
    
    # Position-specific limits
    max_contracts_per_trade: int = 5
    min_time_between_signals: int = 300  # 5 minutes in seconds
    
    # Stop loss settings
    position_stop_loss_pct: float = 0.20  # 20% position stop loss
    trailing_stop_pct: float = 0.15       # 15% trailing stop
    
    # Earnings risk
    block_trades_before_earnings_days: int = 2  # Block trades 2 days before earnings
    
    # Fundamental thresholds
    min_company_market_cap: float = 1e9  # $1B minimum market cap
    max_debt_to_equity: float = 2.0      # Maximum D/E ratio
    
    # Sentiment thresholds
    min_sentiment_score: float = -0.5    # Block if sentiment below this
    max_negative_news_count: int = 5     # Block if too much negative news


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    model_path: str = 'models/xgboost_v1.pkl'
    scaler_path: str = 'models/xgboost_v1_scaler.pkl'
    min_confidence: float = 0.6  # Minimum confidence to trade
    
    # Training settings
    retrain_interval_days: int = 7
    lookback_days: int = 30
    feature_count: int = 28  # Expected number of features
    
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    n_classes: int = 4  # BUY_CALL, BUY_PUT, HOLD, CLOSE
    
    # Feature sources
    use_av_technical_indicators: bool = True  # Use AV instead of TA-Lib
    use_sentiment_features: bool = True       # Include news sentiment
    use_fundamental_features: bool = True     # Include fundamental data


@dataclass
class DatabaseConfig:
    """Database configuration"""
    # PostgreSQL settings
    postgres_host: str = 'localhost'
    postgres_port: int = 5432
    postgres_db: str = 'alphatrader'
    postgres_user: str = 'postgres'
    postgres_password: str = field(default='', repr=False)  # Load from env
    
    # Redis settings
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    
    # Connection pool settings
    min_connections: int = 1
    max_connections: int = 20
    
    # Data retention
    trade_retention_days: int = 365     # Keep 1 year of trades
    tick_data_retention_days: int = 30  # Keep 30 days of tick data
    
    def __post_init__(self):
        """Load sensitive data from environment variables"""
        if not self.postgres_password:
            self.postgres_password = os.getenv('POSTGRES_PASSWORD', 'postgres')


@dataclass
class ExecutionConfig:
    """Trade execution configuration"""
    slippage_ticks: int = 1  # Expected slippage in ticks
    market_order_timeout: int = 10  # Seconds to wait for fill
    use_adaptive_orders: bool = False  # Use IBKR adaptive orders
    
    # Order timing
    market_open_delay_seconds: int = 60  # Wait after market open
    market_close_cutoff_seconds: int = 60  # Stop before market close
    
    # 0DTE handling
    auto_close_0dte: bool = True
    close_0dte_time: str = '15:59:00'  # 3:59 PM
    
    # Position sizing
    use_kelly_criterion: bool = False  # Use Kelly for position sizing
    max_kelly_fraction: float = 0.25   # Maximum Kelly fraction


@dataclass
class AlphaVantageConfig:
    """Alpha Vantage API configuration - UPDATED"""
    api_key: str = ''  # Load from environment
    rate_limit: int = 600  # UPDATED: 600/min for premium (was 75)
    
    # Differentiated cache TTLs by API type
    cache_ttls: Dict[str, int] = field(default_factory=lambda: {
        'options': 60,          # 1 minute for options
        'technical': 300,       # 5 minutes for indicators
        'sentiment': 900,       # 15 minutes for news
        'fundamentals': 86400,  # 1 day for fundamentals
        'economic': 604800,     # 1 week for economic data
        'analytics': 3600,      # 1 hour for analytics
    })
    
    timeout: int = 30  # Request timeout in seconds
    retry_count: int = 3  # Number of retries on failure
    use_cache: bool = True  # Enable caching
    
    # API usage limits (daily)
    daily_limit: int = 864000  # 600/min * 60 * 24
    reserve_calls: int = 10000  # Keep 10k calls in reserve
    
    # Feature flags for different APIs
    use_realtime_options: bool = True
    use_technical_indicators: bool = True
    use_sentiment_data: bool = True
    use_fundamental_data: bool = True
    use_economic_data: bool = True
    use_analytics: bool = True
    
    def __post_init__(self):
        """Load API key from environment"""
        if not self.api_key:
            self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        if not self.api_key:
            logger.warning("Alpha Vantage API key not set!")
    
    def get_cache_ttl(self, api_type: str) -> int:
        """Get cache TTL for specific API type"""
        return self.cache_ttls.get(api_type, 300)  # Default 5 minutes


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_metrics: bool = True
    metrics_port: int = 8080
    
    # Performance thresholds
    max_latency_ms: int = 200        # Alert if latency > 200ms
    max_error_rate: float = 0.01     # Alert if error rate > 1%
    min_cache_hit_rate: float = 0.80 # Alert if cache hit < 80%
    
    # Health check intervals
    health_check_interval: int = 60  # seconds
    api_health_check_interval: int = 300  # 5 minutes
    
    # Alerting
    enable_discord_alerts: bool = True
    alert_webhook_url: str = ''  # Discord webhook for alerts
    
    # Dashboard
    enable_dashboard: bool = True
    dashboard_refresh_rate: int = 5  # seconds


@dataclass
class MarketRegimeConfig:
    """Market regime detection configuration"""
    # Regime thresholds
    vix_high_threshold: float = 30.0     # High volatility above this
    vix_low_threshold: float = 15.0      # Low volatility below this
    
    # Sentiment thresholds
    bullish_sentiment_threshold: float = 0.6
    bearish_sentiment_threshold: float = -0.6
    
    # Trend detection
    trend_lookback_days: int = 20
    strong_trend_threshold: float = 0.02  # 2% daily move
    
    # Regime-based adjustments
    volatility_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'HIGH_VOL': 0.5,     # Reduce position size by 50%
        'NORMAL': 1.0,       # Normal position size
        'LOW_VOL': 1.2,      # Increase by 20%
    })


@dataclass
class TradingConfig:
    """
    Master configuration class for the entire trading system.
    This is the single source of truth for all configuration.
    UPDATED with all enhancements for 38 Alpha Vantage APIs.
    """
    # Core settings
    mode: str = 'paper'  # 'paper' or 'live'
    symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM'])
    
    # Sub-configurations
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    alpha_vantage: AlphaVantageConfig = field(default_factory=AlphaVantageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    
    # Logging settings
    log_level: str = 'INFO'
    log_file: str = 'logs/alphatrader.log'
    log_rotation: str = 'daily'  # daily, weekly, size
    log_retention_days: int = 30
    
    # System settings
    timezone: str = 'America/New_York'
    market_open: str = '09:30:00'
    market_close: str = '16:00:00'
    
    # Market holidays (load from file or API)
    market_holidays: List[str] = field(default_factory=list)
    
    # Feature flags
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    enable_backtesting: bool = True
    enable_discord_bot: bool = False
    enable_web_dashboard: bool = False
    
    # Performance settings
    use_multiprocessing: bool = True
    max_workers: int = 4
    
    @classmethod
    def load(cls, config_path: str = 'config.yaml') -> 'TradingConfig':
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            TradingConfig instance
        """
        # TODO: Implement YAML loading
        # 1. Check if config file exists
        # 2. Load YAML content
        # 3. Parse into nested configuration objects
        # 4. Handle environment variable substitution
        # 5. Validate all required fields
        # 6. Apply environment variable overrides
        # 7. Load market holidays
        # 8. Return configured instance
        pass
    
    def save(self, config_path: str = 'config.yaml') -> None:
        """
        Save current configuration to YAML file
        
        Args:
            config_path: Path to save configuration
        """
        # TODO: Implement YAML saving
        # 1. Convert dataclasses to dictionaries
        # 2. Format for YAML output
        # 3. Add comments for documentation
        # 4. Write to file
        pass
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # TODO: Implement comprehensive validation
        # 1. Check mode is 'paper' or 'live'
        # 2. Verify symbols are valid
        # 3. Check risk limits are positive
        # 4. Verify Greeks limits are tuples
        # 5. Check file paths exist for models
        # 6. Verify database connectivity
        # 7. Check time formats are valid
        # 8. Validate API keys are set
        # 9. Check cache TTLs are reasonable
        # 10. Verify rate limits
        
        return errors
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open
        
        Returns:
            True if market is open, False otherwise
        """
        # TODO: Implement market hours check
        # 1. Get current time in market timezone
        # 2. Check if weekday (Monday-Friday)
        # 3. Check if between market_open and market_close
        # 4. Check for market holidays
        # 5. Apply delay/cutoff times
        pass
    
    def should_trade_symbol(self, symbol: str) -> bool:
        """
        Check if symbol should be traded based on config
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if symbol should be traded
        """
        # TODO: Implement symbol trading check
        # 1. Check if in symbols list
        # 2. Check if not in blacklist
        # 3. Check market cap requirements
        # 4. Return decision
        pass
    
    def get_trading_mode_suffix(self) -> str:
        """Get suffix for mode-specific resources (e.g., database tables)"""
        return f"_{self.mode}" if self.mode == 'paper' else ""
    
    def get_api_capacity_remaining(self) -> Dict[str, int]:
        """
        Get remaining API capacity for the day
        
        Returns:
            Dictionary with remaining calls by API
        """
        # TODO: Implement capacity tracking
        # 1. Calculate calls used today
        # 2. Compare to daily limits
        # 3. Return remaining capacity
        pass
    
    def adjust_for_market_regime(self, base_value: float, 
                                regime: str) -> float:
        """
        Adjust value based on market regime
        
        Args:
            base_value: Base value to adjust
            regime: Current market regime
            
        Returns:
            Adjusted value
        """
        adjustment = self.market_regime.volatility_adjustments.get(regime, 1.0)
        return base_value * adjustment
    
    def __post_init__(self):
        """Initialize configuration after creation"""
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Create necessary directories
        Path('logs').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        Path('data').mkdir(exist_ok=True)
        Path('reports').mkdir(exist_ok=True)
        
        # Log configuration loaded
        logger.info(f"Configuration loaded for {self.mode} mode")
        logger.info(f"Trading symbols: {self.symbols}")
        logger.info(f"Risk limits: {self.risk.max_positions} positions, "
                   f"${self.risk.max_position_size} per position")
        logger.info(f"Alpha Vantage rate limit: {self.alpha_vantage.rate_limit}/min")


# Global configuration instance
# This will be imported by all modules
_config: Optional[TradingConfig] = None


def initialize_config(config_path: str = 'config.yaml') -> TradingConfig:
    """
    Initialize global configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized TradingConfig instance
    """
    global _config
    _config = TradingConfig.load(config_path)
    
    # Validate configuration
    errors = _config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")
    
    # Test critical connections
    logger.info("Testing critical connections...")
    # TODO: Test database connectivity
    # TODO: Test Alpha Vantage API key
    # TODO: Test IBKR connection
    
    return _config


def get_config() -> TradingConfig:
    """
    Get global configuration instance
    
    Returns:
        TradingConfig instance
        
    Raises:
        RuntimeError: If configuration not initialized
    """
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return _config


def reload_config(config_path: str = 'config.yaml') -> TradingConfig:
    """
    Reload configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Reloaded TradingConfig instance
    """
    logger.info("Reloading configuration...")
    return initialize_config(config_path)