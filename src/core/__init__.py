"""
Core module for AlphaTrader system.

Contains configuration management, logging setup, exceptions, and system constants.
"""

from src.core.config import TradingConfig, ConfigManager
from src.core.exceptions import (
    AlphaTraderException,
    ConfigurationError,
    DataSourceError,
    RiskLimitExceeded,
    ConnectionError,
    ModelError,
)
from src.core.constants import (
    TradingHours,
    ErrorCodes,
    SystemLimits,
    LatencyTargets,
)

__all__ = [
    "TradingConfig",
    "ConfigManager",
    "AlphaTraderException",
    "ConfigurationError",
    "DataSourceError", 
    "RiskLimitExceeded",
    "ConnectionError",
    "ModelError",
    "TradingHours",
    "ErrorCodes",
    "SystemLimits",
    "LatencyTargets",
]