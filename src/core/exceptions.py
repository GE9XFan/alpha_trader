"""
Custom exceptions for AlphaTrader
"""

class AlphaTraderException(Exception):
    """Base exception for AlphaTrader"""
    pass


class DataSourceException(AlphaTraderException):
    """Data source related exceptions"""
    pass


class AlphaVantageException(DataSourceException):
    """Alpha Vantage API exceptions"""
    pass


class IBKRException(DataSourceException):
    """IBKR connection exceptions"""
    pass


class RateLimitException(AlphaVantageException):
    """Rate limit exceeded for Alpha Vantage"""
    pass


class GreeksUnavailableException(AlphaVantageException):
    """Greeks not available from Alpha Vantage"""
    pass


class RiskLimitException(AlphaTraderException):
    """Risk limit breached"""
    pass


class PositionLimitException(RiskLimitException):
    """Position limit exceeded"""
    pass


class GreeksLimitException(RiskLimitException):
    """Portfolio Greeks limit breached"""
    pass


class TradingException(AlphaTraderException):
    """Trading related exceptions"""
    pass


class OrderExecutionException(TradingException):
    """Order execution failed"""
    pass


class SignalException(TradingException):
    """Signal generation exception"""
    pass


class ConfigurationException(AlphaTraderException):
    """Configuration error"""
    pass


class DatabaseException(AlphaTraderException):
    """Database operation failed"""
    pass


class CacheException(AlphaTraderException):
    """Cache operation failed"""
    pass


class MonitoringException(AlphaTraderException):
    """Monitoring/alerting exception"""
    pass


class CircuitBreakerOpen(AlphaTraderException):
    """Circuit breaker is open"""
    pass
