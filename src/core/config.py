"""
Configuration management for AlphaTrader system.

Production-grade configuration with validation, type safety, and security.
Handles all system parameters including risk limits, API keys, and Greeks thresholds.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum
import yaml
from dotenv import load_dotenv

from src.core.exceptions import ConfigurationError

load_dotenv()

class TradingMode(Enum):
    """Trading system operation modes."""
    PRODUCTION = "production"
    PAPER = "paper"
    DEVELOPMENT = "development"
    TESTING = "testing"


@dataclass(frozen=True)
class GreeksLimits:
    """Portfolio Greeks risk limits. Immutable for safety."""
    delta_min: float = -0.3
    delta_max: float = 0.3
    gamma_min: float = -0.75
    gamma_max: float = 0.75
    vega_min: float = -1000.0
    vega_max: float = 1000.0
    theta_min: float = -500.0
    rho_min: float = -1000.0
    rho_max: float = 1000.0

    def __post_init__(self):
        """Validate Greeks limits on initialization."""
        if self.delta_min >= self.delta_max:
            raise ConfigurationError(f"Invalid delta range: [{self.delta_min}, {self.delta_max}]")
        if self.gamma_min >= self.gamma_max:
            raise ConfigurationError(f"Invalid gamma range: [{self.gamma_min}, {self.gamma_max}]")
        if self.vega_min >= self.vega_max:
            raise ConfigurationError(f"Invalid vega range: [{self.vega_min}, {self.vega_max}]")
        if self.theta_min > 0:
            raise ConfigurationError(f"Theta minimum must be negative: {self.theta_min}")


@dataclass(frozen=True)
class RiskLimits:
    """Risk management limits. Immutable for safety."""
    max_positions: int = 20
    max_position_size: float = 50000.0
    daily_loss_limit: float = 10000.0
    vpin_threshold: float = 0.7
    max_leverage: float = 2.0
    stop_loss_percentage: float = 0.02  # 2% default stop loss
    
    def __post_init__(self):
        """Validate risk limits."""
        if self.max_positions <= 0 or self.max_positions > 100:
            raise ConfigurationError(f"Invalid max_positions: {self.max_positions}")
        if self.max_position_size <= 0:
            raise ConfigurationError(f"Invalid max_position_size: {self.max_position_size}")
        if self.daily_loss_limit <= 0:
            raise ConfigurationError(f"Invalid daily_loss_limit: {self.daily_loss_limit}")
        if not 0 < self.vpin_threshold <= 1:
            raise ConfigurationError(f"VPIN threshold must be between 0 and 1: {self.vpin_threshold}")


@dataclass(frozen=True)
class IBKRConfig:
    """Interactive Brokers connection configuration."""
    account: str
    host: str = "127.0.0.1"
    live_port: int = 7496
    paper_port: int = 7497
    client_id: int = 1
    timeout: int = 30
    heartbeat_interval: int = 30  # seconds
    max_reconnect_attempts: int = 5
    
    def __post_init__(self):
        """Validate IBKR configuration."""
        if not self.account:
            raise ConfigurationError("IBKR account ID is required")
        if self.live_port == self.paper_port:
            raise ConfigurationError("Live and paper ports must be different")
        if self.heartbeat_interval <= 0:
            raise ConfigurationError(f"Invalid heartbeat interval: {self.heartbeat_interval}")


@dataclass(frozen=True)
class AlphaVantageConfig:
    """Alpha Vantage API configuration."""
    api_key: str
    rate_limit: int = 500  # calls per minute
    timeout: int = 30
    retry_count: int = 3
    critical_update_interval: int = 30  # seconds for options data
    standard_update_interval: int = 300  # 5 minutes for other data
    
    def __post_init__(self):
        """Validate Alpha Vantage configuration."""
        if not self.api_key:
            raise ConfigurationError("Alpha Vantage API key is required")
        if self.rate_limit <= 0 or self.rate_limit > 500:
            raise ConfigurationError(f"Invalid rate limit: {self.rate_limit}")


@dataclass(frozen=True)
class CommunityConfig:
    """Community platform configuration."""
    discord_token: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    whop_api_key: Optional[str] = None
    broadcast_delay_free: int = 300  # 5 minutes
    broadcast_delay_premium: int = 30  # 30 seconds
    broadcast_delay_vip: int = 0  # instant
    max_daily_signals_free: int = 5
    max_daily_signals_premium: int = 20
    max_daily_signals_vip: int = -1  # unlimited
    
    @property
    def is_enabled(self) -> bool:
        """Check if community features are enabled."""
        return bool(self.discord_token or self.whop_api_key)


@dataclass
class TradingConfig:
    """
    Master configuration for the AlphaTrader system.
    
    This is the central configuration object that all components use.
    It aggregates all sub-configurations and provides validation.
    """
    
    # Core settings
    mode: TradingMode
    environment: str  # production, staging, development
    
    # Component configurations
    ibkr: IBKRConfig
    alpha_vantage: AlphaVantageConfig
    greeks_limits: GreeksLimits
    risk_limits: RiskLimits
    community: CommunityConfig
    
    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM',  # ETFs
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # MAG7
        'PLTR', 'DIS', 'HMNS'  # Additional
    ])
    
    # Database configuration
    database_url: str = "postgresql://localhost/alphatrader"
    redis_url: str = "redis://localhost:6379/0"
    
    # Logging configuration
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Performance settings
    enable_profiling: bool = False
    enable_metrics: bool = True
    metrics_port: int = 8080
    
    def __post_init__(self):
        """Validate the complete configuration."""
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate symbols
        if not self.symbols:
            raise ConfigurationError("At least one trading symbol is required")
        
        # Validate mode consistency
        if self.mode == TradingMode.PRODUCTION and self.environment != "production":
            raise ConfigurationError(
                f"Mode {self.mode} inconsistent with environment {self.environment}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "environment": self.environment,
            "ibkr": asdict(self.ibkr),
            "alpha_vantage": asdict(self.alpha_vantage),
            "greeks_limits": asdict(self.greeks_limits),
            "risk_limits": asdict(self.risk_limits),
            "community": asdict(self.community),
            "symbols": self.symbols,
            "database_url": self.database_url,
            "redis_url": self.redis_url,
            "log_level": self.log_level,
            "log_dir": str(self.log_dir),
            "enable_profiling": self.enable_profiling,
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
        }
    
    def validate_for_production(self) -> List[str]:
        """
        Perform production readiness validation.
        Returns list of warnings/issues.
        """
        issues = []
        
        # Check for test/demo accounts
        if "demo" in self.ibkr.account.lower() or "test" in self.ibkr.account.lower():
            issues.append("IBKR account appears to be a demo account")
        
        # Ensure community is configured for production
        if self.community.is_enabled and not self.community.discord_token:
            issues.append("Community enabled but Discord token missing")
        
        # Check database URLs
        if "localhost" in self.database_url and self.environment == "production":
            issues.append("Using localhost database in production")
        
        # Verify risk limits are reasonable
        if self.risk_limits.daily_loss_limit > 50000:
            issues.append(f"Daily loss limit unusually high: ${self.risk_limits.daily_loss_limit}")
        
        return issues


class ConfigManager:
    """
    Manages configuration loading and validation.
    
    Supports loading from:
    - Environment variables (.env files)
    - YAML configuration files
    - Command-line overrides
    """
    
    def __init__(self, config_dir: Path = Path("config")):
        """Initialize configuration manager."""
        self.config_dir = config_dir
        self._config: Optional[TradingConfig] = None
    
    def load_from_env(self, env_file: Optional[Path] = None) -> TradingConfig:
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Optional .env file path
            
        Returns:
            TradingConfig object
            
        Raises:
            ConfigurationError: If required configuration is missing
        """
        # Load .env file if provided
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from default .env
        
        try:
            # Determine mode
            mode_str = os.getenv("TRADING_MODE", "development").lower()
            mode = TradingMode(mode_str)
            
            # Load IBKR configuration
            ibkr = IBKRConfig(
                account=self._get_required_env("IBKR_ACCOUNT"),
                host=os.getenv("IBKR_HOST", "127.0.0.1"),
                live_port=int(os.getenv("IBKR_LIVE_PORT", "7496")),
                paper_port=int(os.getenv("IBKR_PAPER_PORT", "7497")),
                client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
                timeout=int(os.getenv("IBKR_TIMEOUT", "30")),
                heartbeat_interval=int(os.getenv("IBKR_HEARTBEAT", "30")),
            )
            
            # Load Alpha Vantage configuration
            alpha_vantage = AlphaVantageConfig(
                api_key=self._get_required_env("ALPHA_VANTAGE_KEY"),
                rate_limit=int(os.getenv("AV_RATE_LIMIT", "500")),
                timeout=int(os.getenv("AV_TIMEOUT", "30")),
                retry_count=int(os.getenv("AV_RETRY_COUNT", "3")),
                critical_update_interval=int(os.getenv("AV_CRITICAL_UPDATE", "30")),
                standard_update_interval=int(os.getenv("AV_STANDARD_UPDATE", "300")),
            )
            
            # Load Greeks limits
            greeks = GreeksLimits(
                delta_min=float(os.getenv("PORTFOLIO_DELTA_MIN", "-0.3")),
                delta_max=float(os.getenv("PORTFOLIO_DELTA_MAX", "0.3")),
                gamma_min=float(os.getenv("PORTFOLIO_GAMMA_MIN", "-0.75")),
                gamma_max=float(os.getenv("PORTFOLIO_GAMMA_MAX", "0.75")),
                vega_min=float(os.getenv("PORTFOLIO_VEGA_MIN", "-1000")),
                vega_max=float(os.getenv("PORTFOLIO_VEGA_MAX", "1000")),
                theta_min=float(os.getenv("PORTFOLIO_THETA_MIN", "-500")),
                rho_min=float(os.getenv("PORTFOLIO_RHO_MIN", "-1000")),
                rho_max=float(os.getenv("PORTFOLIO_RHO_MAX", "1000")),
            )
            
            # Load risk limits
            risk = RiskLimits(
                max_positions=int(os.getenv("MAX_POSITIONS", "20")),
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "50000")),
                daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", "10000")),
                vpin_threshold=float(os.getenv("VPIN_THRESHOLD", "0.7")),
                max_leverage=float(os.getenv("MAX_LEVERAGE", "2.0")),
                stop_loss_percentage=float(os.getenv("STOP_LOSS_PCT", "0.02")),
            )
            
            # Load community configuration
            community = CommunityConfig(
                discord_token=os.getenv("DISCORD_TOKEN"),
                discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
                whop_api_key=os.getenv("WHOP_API_KEY"),
                broadcast_delay_free=int(os.getenv("BROADCAST_DELAY_FREE", "300")),
                broadcast_delay_premium=int(os.getenv("BROADCAST_DELAY_PREMIUM", "30")),
                broadcast_delay_vip=int(os.getenv("BROADCAST_DELAY_VIP", "0")),
            )
            
            # Load symbols
            symbols_str = os.getenv("TRADING_SYMBOLS", "")
            if symbols_str:
                symbols = [s.strip() for s in symbols_str.split(",")]
            else:
                # Use default symbols
                symbols = [
                    'SPY', 'QQQ', 'IWM',  # ETFs
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # MAG7
                    'PLTR', 'DIS', 'HMNS'  # Additional
                ]
            
            # Create configuration
            config = TradingConfig(
                mode=mode,
                environment=os.getenv("ENVIRONMENT", "development"),
                ibkr=ibkr,
                alpha_vantage=alpha_vantage,
                greeks_limits=greeks,
                risk_limits=risk,
                community=community,
                symbols=symbols,
                database_url=os.getenv("DATABASE_URL", "postgresql://localhost/alphatrader"),
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                log_dir=Path(os.getenv("LOG_DIR", "logs")),
                enable_profiling=os.getenv("ENABLE_PROFILING", "false").lower() == "true",
                enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
                metrics_port=int(os.getenv("METRICS_PORT", "8080")),
            )
            
            self._config = config
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def load_from_yaml(self, config_file: Path) -> TradingConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            TradingConfig object
        """
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse nested configurations
            mode = TradingMode(data["mode"])
            ibkr = IBKRConfig(**data["ibkr"])
            alpha_vantage = AlphaVantageConfig(**data["alpha_vantage"])
            greeks = GreeksLimits(**data["greeks_limits"])
            risk = RiskLimits(**data["risk_limits"])
            community = CommunityConfig(**data.get("community", {}))
            
            config = TradingConfig(
                mode=mode,
                environment=data["environment"],
                ibkr=ibkr,
                alpha_vantage=alpha_vantage,
                greeks_limits=greeks,
                risk_limits=risk,
                community=community,
                symbols=data.get("symbols", [
                    'SPY', 'QQQ', 'IWM',  # ETFs
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # MAG7
                    'PLTR', 'DIS', 'HMNS'  # Additional
                ]),
                database_url=data.get("database_url", "postgresql://localhost/alphatrader"),
                redis_url=data.get("redis_url", "redis://localhost:6379/0"),
                log_level=data.get("log_level", "INFO"),
                log_dir=Path(data.get("log_dir", "logs")),
                enable_profiling=data.get("enable_profiling", False),
                enable_metrics=data.get("enable_metrics", True),
                metrics_port=data.get("metrics_port", 8080),
            )
            
            self._config = config
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
    
    def get_config(self) -> TradingConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load_from_env()
        return self._config
    
    def save_to_yaml(self, config: TradingConfig, output_file: Path):
        """Save configuration to YAML file."""
        try:
            with open(output_file, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    @staticmethod
    def _get_required_env(key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(f"Required environment variable not set: {key}")
        return value


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> TradingConfig:
    """Get the global configuration instance."""
    return _config_manager.get_config()


def load_config(env_file: Optional[Path] = None) -> TradingConfig:
    """Load configuration from environment."""
    return _config_manager.load_from_env(env_file)