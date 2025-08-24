#!/usr/bin/env python3
"""
Trading Configuration Module
Single source of truth for all configuration across the system.
This module is imported by EVERY component and NEVER rewritten.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import yaml
import os
from datetime import datetime
import logging

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
    greeks_limits: Dict[str, tuple] = field(default_factory=lambda: {
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


@dataclass
class TradingConfig:
    """
    Master configuration class for the entire trading system.
    This is the single source of truth for all configuration.
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
    
    # Logging settings
    log_level: str = 'INFO'
    log_file: str = 'logs/alphatrader.log'
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_port: int = 8080
    
    # System settings
    timezone: str = 'America/New_York'
    market_open: str = '09:30:00'
    market_close: str = '16:00:00'
    
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
        # 4. Validate all required fields
        # 5. Apply environment variable overrides
        # 6. Return configured instance
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
        # 3. Write to file with comments
        pass
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # TODO: Implement validation
        # 1. Check mode is 'paper' or 'live'
        # 2. Verify symbols are valid
        # 3. Check risk limits are positive
        # 4. Verify Greeks limits are tuples
        # 5. Check file paths exist for models
        # 6. Verify database connectivity
        # 7. Check time formats are valid
        
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
        pass
    
    def get_trading_mode_suffix(self) -> str:
        """Get suffix for mode-specific resources (e.g., database tables)"""
        return f"_{self.mode}" if self.mode == 'paper' else ""
    
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
        
        # Log configuration loaded
        logger.info(f"Configuration loaded for {self.mode} mode")
        logger.info(f"Trading symbols: {self.symbols}")
        logger.info(f"Risk limits: {self.risk.max_positions} positions, "
                   f"${self.risk.max_position_size} per position")


# Global configuration instance
# This will be imported by all modules
config: Optional[TradingConfig] = None


def initialize_config(config_path: str = 'config.yaml') -> TradingConfig:
    """
    Initialize global configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized TradingConfig instance
    """
    global config
    config = TradingConfig.load(config_path)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")
    
    return config


def get_config() -> TradingConfig:
    """
    Get global configuration instance
    
    Returns:
        TradingConfig instance
        
    Raises:
        RuntimeError: If configuration not initialized
    """
    if config is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return config