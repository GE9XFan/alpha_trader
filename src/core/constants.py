"""
System-wide constants for AlphaTrader.

Defines trading hours, error codes, system limits, and performance targets.
All constants are immutable and used throughout the system.
"""

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Dict, Tuple


class ErrorCodes(Enum):
    """System error codes as defined in specification."""
    
    IBKR_CONNECTION_LOST = "E001"
    ALPHA_VANTAGE_RATE_LIMIT = "E002"
    MODEL_CONFIDENCE_LOW = "E003"
    GREEKS_LIMIT_BREACHED = "E004"
    DATABASE_CONNECTION_FAILED = "E005"
    DISCORD_API_ERROR = "E006"
    INSUFFICIENT_FUNDS = "E007"
    SYMBOL_HALTED = "E008"
    VPIN_THRESHOLD_EXCEEDED = "E009"
    ZERO_DTE_NOT_CLOSED = "E010"
    
    @property
    def description(self) -> str:
        """Get human-readable description of error code."""
        descriptions = {
            "E001": "IBKR connection lost - Reconnect with backoff",
            "E002": "Alpha Vantage rate limit - Throttle to critical APIs",
            "E003": "Model confidence low - Use fallback model",
            "E004": "Greeks limit breached - Halt trading immediately",
            "E005": "Database connection failed - Cache-only mode",
            "E006": "Discord API error - Retry with backoff",
            "E007": "Insufficient funds - Reduce position size",
            "E008": "Symbol halted - Cancel pending orders",
            "E009": "VPIN > 0.7 - Close all positions",
            "E010": "0DTE not closed - Force closure at 3:59 PM",
        }
        return descriptions.get(self.value, "Unknown error")
    
    @property
    def severity(self) -> str:
        """Get severity level of error."""
        critical = ["E004", "E009", "E010"]
        high = ["E001", "E003", "E005", "E007", "E008"]
        medium = ["E002", "E006"]
        
        if self.value in critical:
            return "CRITICAL"
        elif self.value in high:
            return "HIGH"
        elif self.value in medium:
            return "MEDIUM"
        else:
            # Unknown error codes default to MEDIUM
            return "MEDIUM"


@dataclass(frozen=True)
class TradingHours:
    """Market trading hours (all times in ET)."""
    
    # Regular market hours
    MARKET_OPEN: time = time(9, 30)
    MARKET_CLOSE: time = time(16, 0)
    
    # Extended hours
    PRE_MARKET_START: time = time(4, 0)
    PRE_MARKET_END: time = time(9, 30)
    AFTER_HOURS_START: time = time(16, 0)
    AFTER_HOURS_END: time = time(20, 0)
    
    # Critical times
    MOC_WINDOW_START: time = time(15, 40)  # 3:40 PM
    MOC_WINDOW_END: time = time(16, 0)     # 4:00 PM
    ZERO_DTE_CLOSURE: time = time(15, 59)  # 3:59 PM
    
    # Analysis times
    MORNING_ANALYSIS: time = time(8, 30)
    DAILY_RECAP: time = time(16, 30)
    
    @classmethod
    def is_market_open(cls, current_time: time) -> bool:
        """Check if market is currently open."""
        return cls.MARKET_OPEN <= current_time < cls.MARKET_CLOSE
    
    @classmethod
    def is_moc_window(cls, current_time: time) -> bool:
        """Check if in MOC order window."""
        return cls.MOC_WINDOW_START <= current_time < cls.MOC_WINDOW_END
    
    @classmethod
    def is_extended_hours(cls, current_time: time) -> bool:
        """Check if in extended trading hours."""
        pre_market = cls.PRE_MARKET_START <= current_time < cls.PRE_MARKET_END
        after_hours = cls.AFTER_HOURS_START <= current_time < cls.AFTER_HOURS_END
        return pre_market or after_hours


@dataclass(frozen=True)
class SystemLimits:
    """System-wide operational limits."""
    
    # Position limits
    MAX_POSITIONS: int = 20
    MAX_POSITION_SIZE: float = 50000.0
    MIN_POSITION_SIZE: float = 100.0
    
    # Risk limits
    DAILY_LOSS_LIMIT: float = 10000.0
    VPIN_THRESHOLD: float = 0.7
    MAX_LEVERAGE: float = 2.0
    
    # Greeks limits (portfolio-wide)
    DELTA_RANGE: Tuple[float, float] = (-0.3, 0.3)
    GAMMA_RANGE: Tuple[float, float] = (-0.75, 0.75)
    VEGA_RANGE: Tuple[float, float] = (-1000.0, 1000.0)
    THETA_MIN: float = -500.0
    RHO_RANGE: Tuple[float, float] = (-1000.0, 1000.0)
    
    # API limits
    ALPHA_VANTAGE_RATE_LIMIT: int = 500  # calls per minute
    IBKR_MESSAGE_RATE_LIMIT: int = 50   # messages per second
    
    # Data limits
    MAX_BARS_IN_MEMORY: int = 1000      # Per symbol
    MAX_OPTION_CONTRACTS: int = 1000    # Per symbol chain
    
    # Community limits
    MAX_DAILY_SIGNALS_FREE: int = 5
    MAX_DAILY_SIGNALS_PREMIUM: int = 20
    MAX_DAILY_SIGNALS_VIP: int = -1     # Unlimited
    
    # Performance limits
    MAX_RECONNECT_ATTEMPTS: int = 5
    MAX_API_RETRIES: int = 3
    
    @classmethod
    def validate_position_size(cls, size: float) -> bool:
        """Validate if position size is within limits."""
        return cls.MIN_POSITION_SIZE <= size <= cls.MAX_POSITION_SIZE


@dataclass(frozen=True)
class LatencyTargets:
    """Performance latency targets in milliseconds."""
    
    # Critical path components (total must be <50ms)
    IBKR_DATA_RECEIPT: int = 5
    FEATURE_CALCULATION: int = 15
    MODEL_INFERENCE: int = 10
    RISK_VALIDATION: int = 5
    ORDER_EXECUTION: int = 15
    CRITICAL_PATH_TOTAL: int = 50
    
    # Non-critical operations
    DATABASE_WRITE: int = 100
    CACHE_READ: int = 1
    CACHE_WRITE: int = 5
    
    # Community operations
    DISCORD_BROADCAST: int = 2000  # 2 seconds
    WEBHOOK_DELIVERY: int = 500
    
    # Data operations
    ALPHA_VANTAGE_FETCH: int = 500
    OPTIONS_GREEKS_CALC: int = 5
    VPIN_CALCULATION: int = 3
    
    @classmethod
    def validate_critical_path(cls, component_times: Dict[str, int]) -> bool:
        """Validate if component times meet critical path requirement."""
        total = sum(component_times.values())
        return total <= cls.CRITICAL_PATH_TOTAL


class DataPriority(Enum):
    """Data update priority levels for Alpha Vantage APIs."""
    
    CRITICAL = 1  # 30-second updates (options, key indicators)
    HIGH = 2      # 30-second updates (secondary indicators)
    MEDIUM = 3    # 5-minute updates (analytics, sentiment)
    LOW = 4       # 15-minute updates (fundamentals, economic)
    
    @property
    def update_interval(self) -> int:
        """Get update interval in seconds."""
        intervals = {
            DataPriority.CRITICAL: 30,
            DataPriority.HIGH: 30,
            DataPriority.MEDIUM: 300,
            DataPriority.LOW: 900,
        }
        return intervals[self]


class SubscriptionTier(Enum):
    """Community subscription tiers."""
    
    FREE = "FREE"
    PREMIUM = "PREMIUM"
    VIP = "VIP"
    
    @property
    def price(self) -> int:
        """Get tier price in USD."""
        prices = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.PREMIUM: 99,
            SubscriptionTier.VIP: 499,
        }
        return prices[self]
    
    @property
    def signal_delay(self) -> int:
        """Get signal delay in seconds."""
        delays = {
            SubscriptionTier.FREE: 300,    # 5 minutes
            SubscriptionTier.PREMIUM: 30,  # 30 seconds
            SubscriptionTier.VIP: 0,       # Instant
        }
        return delays[self]
    
    @property
    def max_daily_signals(self) -> int:
        """Get maximum daily signals (-1 for unlimited)."""
        limits = {
            SubscriptionTier.FREE: 5,
            SubscriptionTier.PREMIUM: 20,
            SubscriptionTier.VIP: -1,  # Unlimited
        }
        return limits[self]


class OrderType(Enum):
    """Supported order types."""
    
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"
    MOC = "MOC"  # Market on Close
    TRAIL = "TRAIL"
    
    @property
    def requires_price(self) -> bool:
        """Check if order type requires price specification."""
        return self in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]


class OptionType(Enum):
    """Option contract types."""
    
    CALL = "CALL"
    PUT = "PUT"
    
    @property
    def multiplier(self) -> int:
        """Get option contract multiplier."""
        return 100  # Standard equity options


# API Endpoints
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Critical Alpha Vantage APIs (30-second updates)
CRITICAL_AV_APIS = [
    "REALTIME_OPTIONS",
    "HISTORICAL_OPTIONS",
    "RSI",
    "MACD",
    "VWAP",
]

# Market holidays (2025)
MARKET_HOLIDAYS_2025 = [
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
]

# Supported symbols (default list)
DEFAULT_SYMBOLS = [
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # MAG7
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Additional high-volume
    "AMD", "INTC", "NFLX", "BABA", "JNJ", "V", "MA",
    # User specified
    "PLTR", "DIS", "HMNS",
]