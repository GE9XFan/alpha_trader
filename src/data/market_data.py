#!/usr/bin/env python3
"""
Market Data Manager Module
Core market data component that handles IBKR connection and real-time data.
This component is NEVER REWRITTEN and used by ML, Trading, Risk, Paper, Live, and Community.
"""

from ib_insync import IB, Stock, Contract, BarData, Ticker
import asyncio
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import logging
from dataclasses import dataclass
import threading
from queue import Queue

from src.core.config import get_config, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Point-in-time market data snapshot"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: int
    ask_size: int
    open: float
    high: float
    low: float
    close: float
    
    @property
    def mid(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid if self.bid and self.ask else 0.0


class MarketDataManager:
    """
    Core market data component - NEVER REWRITTEN
    Manages IBKR connection, market data subscriptions, and data caching.
    Used by: ML, Trading, Risk, Paper, Live, Community
    """
    
    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize MarketDataManager
        
        Args:
            config: Trading configuration (uses global if not provided)
        """
        self.config = config or get_config()
        self.ib: Optional[IB] = None
        self.connected: bool = False
        
        # Data storage
        self.subscriptions: Dict[str, Contract] = {}
        self.tickers: Dict[str, Ticker] = {}
        self.latest_prices: Dict[str, float] = {}
        self.bars_5sec: Dict[str, deque] = {}  # Rolling window of 5-second bars
        self.bars_cache: Dict[str, pd.DataFrame] = {}  # Historical bars cache
        
        # Callbacks for real-time updates
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Connection management
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10
        self._stop_event = threading.Event()
        self._data_queue = Queue()
        
        # Performance tracking
        self.last_update_time: Dict[str, datetime] = {}
        self.update_latency: deque = deque(maxlen=1000)  # Track last 1000 updates
        
        logger.info("MarketDataManager initialized")
    
    async def connect(self) -> bool:
        """
        Connect to IBKR (paper or live based on config)
        
        Returns:
            True if connection successful, False otherwise
        """
        # TODO: Implement IBKR connection
        # 1. Create IB() instance
        # 2. Determine port based on config.mode
        # 3. Attempt connection with timeout
        # 4. Set up error handlers
        # 5. Verify connection is active
        # 6. Start keep-alive mechanism
        pass
    
    async def disconnect(self) -> None:
        """
        Disconnect from IBKR and cleanup
        """
        # TODO: Implement disconnection
        # 1. Cancel all subscriptions
        # 2. Clear callbacks
        # 3. Disconnect from IBKR
        # 4. Clear data caches
        # 5. Set connected flag to False
        pass
    
    async def reconnect(self) -> bool:
        """
        Reconnect to IBKR with exponential backoff
        
        Returns:
            True if reconnection successful, False otherwise
        """
        # TODO: Implement reconnection logic
        # 1. Disconnect if currently connected
        # 2. Implement exponential backoff
        # 3. Attempt reconnection up to max_attempts
        # 4. Restore subscriptions on success
        # 5. Log all attempts and results
        pass
    
    async def subscribe_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Subscribe to market data for symbols
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            Dictionary mapping symbols to subscription success
        """
        # TODO: Implement market data subscription
        # 1. Create Stock contracts for each symbol
        # 2. Qualify contracts with IBKR
        # 3. Request market data (Level 1)
        # 4. Request 5-second bars
        # 5. Set up update callbacks
        # 6. Initialize data storage
        # 7. Handle subscription failures gracefully
        pass
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """
        Unsubscribe from market data for a symbol
        
        Args:
            symbol: Symbol to unsubscribe from
            
        Returns:
            True if unsubscribed successfully
        """
        # TODO: Implement unsubscription
        # 1. Cancel market data subscription
        # 2. Cancel real-time bars
        # 3. Remove from subscriptions dict
        # 4. Clear associated data
        # 5. Remove callbacks
        pass
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price from cache (instant access)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest price or 0.0 if not available
        """
        # TODO: Implement price retrieval
        # 1. Check if symbol in latest_prices
        # 2. Return cached price
        # 3. If not available, check ticker
        # 4. Return 0.0 as fallback
        pass
    
    def get_market_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """
        Get complete market snapshot for symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            MarketSnapshot or None if not available
        """
        # TODO: Implement snapshot creation
        # 1. Get ticker for symbol
        # 2. Extract all price/volume data
        # 3. Create MarketSnapshot object
        # 4. Handle missing data gracefully
        pass
    
    async def get_historical_bars(self, 
                                 symbol: str, 
                                 days: int = 5,
                                 bar_size: str = '5 secs') -> pd.DataFrame:
        """
        Get historical bars for ML training and analysis
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
            bar_size: Bar size (e.g., '5 secs', '1 min')
            
        Returns:
            DataFrame with OHLCV data
        """
        # TODO: Implement historical data retrieval
        # 1. Check cache first (with TTL)
        # 2. Create contract for symbol
        # 3. Request historical data from IBKR
        # 4. Convert to pandas DataFrame
        # 5. Add technical columns (returns, etc.)
        # 6. Cache the result
        # 7. Handle errors and retries
        pass
    
    def get_rolling_bars(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """
        Get rolling window of recent 5-second bars
        
        Args:
            symbol: Stock symbol
            count: Number of bars to return
            
        Returns:
            DataFrame with recent bars
        """
        # TODO: Implement rolling bars retrieval
        # 1. Get bars from bars_5sec deque
        # 2. Convert to DataFrame
        # 3. Ensure we have enough bars
        # 4. Add calculated fields
        pass
    
    def register_callback(self, symbol: str, callback: Callable) -> None:
        """
        Register callback for real-time updates
        
        Args:
            symbol: Stock symbol
            callback: Function to call on updates
        """
        # TODO: Implement callback registration
        # 1. Add callback to callbacks dict
        # 2. Ensure symbol is subscribed
        # 3. Test callback with current data
        pass
    
    def unregister_callback(self, symbol: str, callback: Callable) -> None:
        """
        Unregister callback for symbol
        
        Args:
            symbol: Stock symbol
            callback: Callback to remove
        """
        # TODO: Implement callback removal
        # 1. Remove callback from list
        # 2. Clean up if no callbacks remain
        pass
    
    def _on_bar_update(self, bars: BarData, hasNewBar: bool) -> None:
        """
        Handle real-time bar updates
        
        Args:
            bars: Bar data from IBKR
            hasNewBar: Whether a new bar was added
        """
        # TODO: Implement bar update handler
        # 1. Extract symbol from bars
        # 2. Update bars_5sec deque
        # 3. Update latest_prices
        # 4. Calculate latency
        # 5. Trigger callbacks
        # 6. Log if latency exceeds threshold
        pass
    
    def _on_ticker_update(self, ticker: Ticker) -> None:
        """
        Handle ticker updates
        
        Args:
            ticker: Updated ticker from IBKR
        """
        # TODO: Implement ticker update handler
        # 1. Extract symbol
        # 2. Update latest_prices
        # 3. Update last_update_time
        # 4. Trigger callbacks
        # 5. Track update frequency
        pass
    
    def _keep_alive(self) -> None:
        """
        Keep connection alive with periodic requests
        """
        # TODO: Implement keep-alive mechanism
        # 1. Periodic connection check
        # 2. Request server time
        # 3. Reconnect if needed
        # 4. Log connection health
        pass
    
    async def _monitor_data_quality(self) -> None:
        """
        Monitor data quality and alert on issues
        """
        # TODO: Implement data quality monitoring
        # 1. Check update frequencies
        # 2. Detect stale data
        # 3. Monitor spreads
        # 4. Check for anomalies
        # 5. Alert on quality issues
        pass
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics
        
        Returns:
            Dictionary with latency metrics
        """
        # TODO: Calculate latency statistics
        # 1. Calculate mean latency
        # 2. Calculate p50, p95, p99
        # 3. Find max latency
        # 4. Count delayed updates
        pass
    
    def is_market_hours(self) -> bool:
        """
        Check if currently in market hours
        
        Returns:
            True if market is open
        """
        # TODO: Implement market hours check
        # 1. Get current time in ET
        # 2. Check weekday
        # 3. Check time range
        # 4. Check for holidays
        pass
    
    async def warmup(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Warmup market data before trading
        
        Args:
            symbols: Symbols to warmup (uses config if None)
            
        Returns:
            True if warmup successful
        """
        # TODO: Implement warmup procedure
        # 1. Connect to IBKR
        # 2. Subscribe to symbols
        # 3. Wait for initial data
        # 4. Verify data quality
        # 5. Load historical data
        # 6. Return success status
        pass
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.connected:
            asyncio.create_task(self.disconnect())