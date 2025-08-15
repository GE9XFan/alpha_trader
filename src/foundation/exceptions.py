"""
Custom Exceptions for Trading System
"""


class TradingSystemException(Exception):
    """Base exception for trading system"""
    pass


class ConfigurationError(TradingSystemException):
    """Configuration related errors"""
    pass


class ConnectionError(TradingSystemException):
    """Connection related errors"""
    pass


class DataError(TradingSystemException):
    """Data related errors"""
    pass


class ValidationError(TradingSystemException):
    """Validation related errors"""
    pass


class RiskLimitError(TradingSystemException):
    """Risk limit violations"""
    pass


class ExecutionError(TradingSystemException):
    """Order execution errors"""
    pass


class RateLimitError(TradingSystemException):
    """API rate limit errors"""
    pass
