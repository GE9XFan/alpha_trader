"""
AlphaTrader - High-Frequency Options Trading System

Production-grade algorithmic trading system with sub-50ms execution latency,
real-time Greeks calculation, and integrated community platform.
"""

__version__ = "1.0.0"
__author__ = "AlphaTrader Team"

from src.core.config import TradingConfig
from src.core.exceptions import AlphaTraderException

__all__ = [
    "TradingConfig",
    "AlphaTraderException",
    "__version__",
]