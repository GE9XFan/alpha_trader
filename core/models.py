#!/usr/bin/env python3
"""
Data Models for Options Trading System
Pydantic models for type safety and validation
"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class OrderSide(str, Enum):
    BID = "BID"
    ASK = "ASK"


class OrderBookLevel(BaseModel):
    """Single level in order book"""
    price: float
    size: int
    market_maker: Optional[str] = None


class OrderBook(BaseModel):
    """Level 2 Order Book Data"""
    symbol: str
    timestamp: int  # Unix timestamp in milliseconds
    bids: List[OrderBookLevel]  # Sorted by price descending
    asks: List[OrderBookLevel]  # Sorted by price ascending

    @validator('bids', 'asks')
    def validate_levels(cls, v):
        """Ensure we have up to 10 levels"""
        if len(v) > 10:
            return v[:10]
        return v

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self.bids and self.asks:
            return (self.asks[0].price + self.bids[0].price) / 2
        return 0.0


class Trade(BaseModel):
    """Individual trade/print"""
    symbol: str
    timestamp: int
    price: float
    size: int
    is_buyer: Optional[bool] = None  # True if buyer initiated


class Bar(BaseModel):
    """OHLCV bar data"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    bar_count: Optional[int] = None  # Number of trades in bar


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class OptionContract(BaseModel):
    """Options contract with Greeks (PROVIDED by Alpha Vantage)"""
    symbol: str
    contract_id: str
    strike: float
    expiration: str  # YYYY-MM-DD format
    type: OptionType

    # Market data
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0

    # Greeks - PROVIDED by Alpha Vantage, not calculated!
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    @property
    def mid_price(self) -> float:
        """Mid price of option"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last or 0.0

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price"""
        if self.bid and self.ask and self.mid_price > 0:
            return ((self.ask - self.bid) / self.mid_price) * 100
        return 0.0

    @property
    def days_to_expiry(self) -> int:
        """Calculate days until expiration"""
        exp_date = datetime.strptime(self.expiration, '%Y-%m-%d').date()
        return (exp_date - datetime.now().date()).days


class OptionsChain(BaseModel):
    """Complete options chain for a symbol"""
    symbol: str
    spot_price: float
    timestamp: int
    options: List[OptionContract]

    def filter_by_expiry(self, expiry: str) -> List[OptionContract]:
        """Get all options for specific expiration"""
        return [opt for opt in self.options if opt.expiration == expiry]

    def filter_by_dte(self, min_dte: int, max_dte: int) -> List[OptionContract]:
        """Filter options by days to expiry"""
        return [
            opt for opt in self.options
            if min_dte <= opt.days_to_expiry <= max_dte
        ]

    def get_atm_strike(self) -> float:
        """Find at-the-money strike"""
        strikes = sorted(set(opt.strike for opt in self.options))
        if not strikes:
            return self.spot_price

        # Find closest strike to spot
        return min(strikes, key=lambda x: abs(x - self.spot_price))


class SignalStrategy(str, Enum):
    ZDTE = "0DTE"    # Zero days to expiry
    ONEDTE = "1DTE"  # One day to expiry
    SWING = "14DTE"  # Swing trades
    MOC = "MOC"      # Market on close


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal(BaseModel):
    """Trading signal with complete trade details"""
    signal_id: str
    timestamp: int
    symbol: str

    # Signal details
    action: SignalAction
    strategy: SignalStrategy
    confidence: float = Field(ge=0, le=100)  # 0-100

    # Trade parameters
    entry: float
    stop_loss: float
    targets: List[float] = Field(default_factory=list)  # 1-3 targets validated separately

    # Position sizing
    position_size: float  # Dollar amount
    max_risk: float      # Dollar risk
    risk_reward: float   # Ratio

    # Options specific (if applicable)
    contract: Optional[OptionContract] = None

    # Supporting data
    metrics: Dict[str, float] = {}
    reason: str = ""

    @validator('targets')
    def validate_targets(cls, v, values):
        """Ensure targets are beyond entry in right direction"""
        if not v or len(v) < 1 or len(v) > 3:
            raise ValueError("Must have 1-3 targets")
        if 'action' in values and 'entry' in values:
            entry = values['entry']
            if values['action'] == SignalAction.BUY:
                assert all(t > entry for t in v), "Buy targets must be above entry"
            elif values['action'] == SignalAction.SELL:
                assert all(t < entry for t in v), "Sell targets must be below entry"
        return v


class PositionStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class Position(BaseModel):
    """Active or closed position tracking"""
    position_id: str
    signal_id: str
    symbol: str
    strategy: SignalStrategy

    # Entry details
    entry_time: int
    entry_price: float
    size: int  # Number of shares/contracts
    direction: Literal["LONG", "SHORT"]

    # Current state
    status: PositionStatus
    current_price: float = 0.0
    stop_loss: float
    targets: List[float]
    current_target: int = 0  # Index of current target

    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees: float = 0.0

    # Exit details (if closed)
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def net_pnl(self) -> float:
        """Total P&L including fees"""
        return self.unrealized_pnl + self.realized_pnl - self.fees

    @property
    def pnl_percent(self) -> float:
        """P&L as percentage"""
        if self.entry_price and self.size:
            total_cost = self.entry_price * self.size
            return (self.net_pnl / total_cost) * 100
        return 0.0


class MarketMetrics(BaseModel):
    """Calculated market microstructure metrics"""
    symbol: str
    timestamp: int

    # Microstructure
    vpin: float = Field(ge=0, le=1)  # 0-1, >0.4 is toxic
    order_book_imbalance: float = Field(ge=-1, le=1)  # -1 to +1
    bid_ask_spread: float = Field(ge=0)
    book_pressure: float = Field(ge=-1, le=1)

    # Options metrics
    gamma_exposure: float = 0.0  # In millions
    pin_strike: float = 0.0
    put_call_ratio: float = 0.0
    iv_rank: float = Field(ge=0, le=100)  # 0-100 percentile

    # Flow detection
    sweep_detected: bool = False
    hidden_orders: bool = False
    unusual_activity: bool = False

    # Technical indicators
    rsi: Optional[float] = None
    macd_signal: Optional[float] = None
    atr: Optional[float] = None


class AccountStatus(BaseModel):
    """Account information and limits"""
    account_id: str
    timestamp: int

    # Capital
    buying_power: float
    net_liquidation: float
    cash_balance: float

    # Positions
    open_positions: int
    max_positions: int = 5

    # Daily metrics
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_fees: float = 0.0

    # Risk metrics
    current_risk: float = 0.0  # Total dollars at risk
    max_daily_loss: float = 2000.0
    consecutive_losses: int = 0

    @property
    def can_trade(self) -> bool:
        """Check if we can take new positions"""
        return (
            self.open_positions < self.max_positions and
            abs(self.daily_pnl) < self.max_daily_loss and
            self.consecutive_losses < 3 and
            self.buying_power > 1000
        )

    @property
    def risk_utilization(self) -> float:
        """Percentage of max daily loss used"""
        return abs(self.daily_pnl / self.max_daily_loss) * 100


class SystemHealth(BaseModel):
    """System health and monitoring"""
    timestamp: int

    # Connections
    ibkr_connected: bool = False
    redis_connected: bool = False
    av_api_healthy: bool = False

    # Performance
    latency_ms: Dict[str, float] = {}  # Component latencies
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0

    # API usage
    av_calls_used: int = 0
    av_calls_remaining: int = 600

    # System resources
    memory_used_gb: float = 0.0
    cpu_percent: float = 0.0

    @property
    def is_healthy(self) -> bool:
        """Overall system health check"""
        return (
            self.ibkr_connected and
            self.redis_connected and
            self.av_api_healthy and
            self.error_rate < 0.05  # Less than 5% errors
        )
