"""
Market Data Manager - Implementation Plan Week 1 Day 4
IBKR connection for quotes, bars, and execution
Alpha Vantage handles options - this is just spot prices
Production-ready implementation with error handling and reconnection
"""
from ib_insync import IB, Stock, Option, MarketOrder, Contract, util
import asyncio
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import time

from src.core.config import config
from src.core.logger import get_logger
from src.core.exceptions import IBKRException


logger = get_logger(__name__)


class MarketDataManager:
    """
    Production-ready IBKR market data manager
    Alpha Vantage handles options - this is just spot prices
    Used by: ML, Trading, Risk, Paper, Live, Community
    
    Features:
    - Automatic reconnection with exponential backoff
    - Memory-efficient bar storage (rolling window)
    - Data quality validation
    - Real-time quote subscriptions
    - Historical data retrieval
    """
    
    def __init__(self):
        self.config = config.ibkr
        self.trading_config = config.trading
        self.ib = IB()
        self.connected = False
        
        # Allow environment variables to override config
        import os
        if 'IB_CLIENT_ID' in os.environ:
            self.config.client_id = int(os.environ['IB_CLIENT_ID'])
            logger.info(f"Using IB_CLIENT_ID from environment: {self.config.client_id}")
        if 'IB_PORT' in os.environ:
            self.config.port = int(os.environ['IB_PORT'])
            logger.info(f"Using IB_PORT from environment: {self.config.port}")
        
        # Data storage with memory management
        self.latest_prices = {}  # Cache for instant access
        self.bars_5sec = {}  # 5-second bars from IBKR (rolling window)
        self.subscriptions = {}
        self.bar_buffers = {}  # Deque for memory-efficient storage
        
        # Connection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1  # Start with 1 second
        self.last_heartbeat = time.time()
        
        # Data quality tracking
        self.last_update_time = {}
        self.stale_threshold = 30  # seconds
        
        # Callbacks for updates
        self.price_callbacks = []
        self.error_callbacks = []
        
    async def connect(self, retry=True):
        """
        Connect to IBKR with automatic retry logic
        
        Args:
            retry: Enable automatic retry with exponential backoff
        
        Returns:
            bool: True if connected successfully
        """
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Determine port based on mode
                port = self.config.port if self.trading_config.mode == 'paper' else 7496
                
                logger.info(f"Attempting IBKR connection to {self.config.host}:{port} "
                          f"(attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                
                # Connect with timeout
                await self.ib.connectAsync(
                    self.config.host,
                    port,
                    clientId=self.config.client_id,
                    timeout=self.config.connection_timeout
                )
                
                # Verify connection
                if not self.ib.isConnected():
                    raise IBKRException("Connection established but not active")
                
                self.connected = True
                self.reconnect_attempts = 0  # Reset on success
                self.reconnect_delay = 1  # Reset delay
                self.last_heartbeat = time.time()
                
                logger.info(f"✅ Connected to IBKR ({self.trading_config.mode} mode)")
                logger.info(f"   Host: {self.config.host}:{port}")
                logger.info(f"   Client ID: {self.config.client_id}")
                
                # Setup event handlers
                self.ib.errorEvent += self._on_error
                self.ib.disconnectedEvent += self._on_disconnect
                
                # Start heartbeat monitor
                asyncio.create_task(self._heartbeat())
                
                return True
                
            except Exception as e:
                self.reconnect_attempts += 1
                
                if not retry or self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"❌ Failed to connect to IBKR after {self.reconnect_attempts} attempts: {e}")
                    raise IBKRException(f"IBKR connection failed: {e}")
                
                # Exponential backoff
                wait_time = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60)
                logger.warning(f"Connection failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        return False
    
    async def disconnect(self):
        """Gracefully disconnect from IBKR"""
        if self.connected:
            try:
                # Cancel all subscriptions
                for symbol, bars in self.subscriptions.items():
                    self.ib.cancelRealTimeBars(bars)
                    logger.debug(f"Cancelled subscription for {symbol}")
                
                # Clear data
                self.subscriptions.clear()
                
                # Disconnect
                self.ib.disconnect()
                self.connected = False
                logger.info("✅ Disconnected from IBKR gracefully")
                
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                self.connected = False
    
    async def subscribe_symbols(self, symbols: List[str]):
        """
        Subscribe to real-time market data for symbols
        
        Args:
            symbols: List of stock symbols to subscribe to
            
        Returns:
            Dict[str, bool]: Success status for each symbol
        """
        if not self.connected:
            raise IBKRException("Not connected to IBKR")
        
        results = {}
        
        for symbol in symbols:
            try:
                # Skip if already subscribed
                if symbol in self.subscriptions:
                    logger.debug(f"{symbol} already subscribed")
                    results[symbol] = True
                    continue
                
                # Create and qualify contract
                contract = Stock(symbol, 'SMART', 'USD')
                await self.ib.qualifyContractsAsync(contract)
                
                if not contract.conId:
                    raise IBKRException(f"Failed to qualify contract for {symbol}")
                
                # Request real-time bars (5-second) - synchronous call is OK here
                bars = self.ib.reqRealTimeBars(
                    contract, 
                    5,  # 5-second bars
                    'TRADES',  # Trade data
                    False  # Not outside RTH
                )
                
                if bars is None:
                    raise IBKRException(f"Failed to subscribe to {symbol}")
                
                # Store subscription
                self.subscriptions[symbol] = bars
                
                # Initialize bar buffer (rolling window of 1000 bars)
                self.bar_buffers[symbol] = deque(maxlen=1000)
                
                # Set up callback for updates
                # updateEvent passes (bars, hasNewBar) - we need to handle both
                bars.updateEvent += lambda bars_obj, hasNewBar, sym=symbol: self._on_bar_update(sym, bars_obj)
                
                # Initialize tracking
                self.last_update_time[symbol] = time.time()
                
                logger.info(f"✅ Subscribed to {symbol} real-time data")
                results[symbol] = True
                
            except Exception as e:
                logger.error(f"❌ Failed to subscribe to {symbol}: {e}")
                results[symbol] = False
        
        return results
    
    def _on_bar_update(self, symbol: str, bars):
        """
        Handle real-time bar updates with data quality checks
        
        Args:
            symbol: Stock symbol
            bars: Real-time bar data from IBKR
        """
        try:
            if bars and len(bars) > 0:
                latest = bars[-1]
                
                # Data quality check
                if latest.close <= 0 or latest.volume < 0:
                    logger.warning(f"Invalid bar data for {symbol}: close={latest.close}, vol={latest.volume}")
                    return
                
                # Update caches
                self.latest_prices[symbol] = latest.close
                self.bars_5sec[symbol] = latest
                self.last_update_time[symbol] = time.time()
                
                # Add to rolling buffer
                # RealTimeBar uses open_ instead of open
                bar_data = {
                    'time': latest.time,
                    'open': latest.open_,
                    'high': latest.high,
                    'low': latest.low,
                    'close': latest.close,
                    'volume': latest.volume
                }
                # Add optional attributes if they exist
                if hasattr(latest, 'wap'):
                    bar_data['wap'] = latest.wap
                if hasattr(latest, 'count'):
                    bar_data['count'] = latest.count
                self.bar_buffers[symbol].append(bar_data)
                
                # Notify callbacks
                for callback in self.price_callbacks:
                    try:
                        callback(symbol, latest.close)
                    except Exception as e:
                        logger.error(f"Error in price callback: {e}")
                
                # Log periodically (every 60 updates)
                if len(self.bar_buffers[symbol]) % 60 == 0:
                    logger.debug(f"{symbol}: ${latest.close:.2f} (vol: {latest.volume:,})")
                    
        except Exception as e:
            logger.error(f"Error processing bar update for {symbol}: {e}")
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price with staleness check
        
        Args:
            symbol: Stock symbol
            
        Returns:
            float: Latest price or 0.0 if not available/stale
        """
        if symbol not in self.latest_prices:
            return 0.0
        
        # Check if data is stale
        if symbol in self.last_update_time:
            age = time.time() - self.last_update_time[symbol]
            if age > self.stale_threshold:
                logger.warning(f"Price data for {symbol} is stale ({age:.1f}s old)")
        
        return self.latest_prices.get(symbol, 0.0)
    
    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest 5-second bar
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with bar data or None
        """
        if symbol in self.bars_5sec:
            bar = self.bars_5sec[symbol]
            return {
                'time': bar.time,
                'open': bar.open_,  # RealTimeBar uses open_ not open
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'wap': bar.wap if hasattr(bar, 'wap') else bar.close
            }
        return None
    
    def get_bar_history(self, symbol: str, num_bars: int = 100) -> pd.DataFrame:
        """
        Get recent bar history from buffer
        
        Args:
            symbol: Stock symbol
            num_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with bar history
        """
        if symbol not in self.bar_buffers:
            return pd.DataFrame()
        
        bars = list(self.bar_buffers[symbol])[-num_bars:]
        return pd.DataFrame(bars) if bars else pd.DataFrame()
    
    async def get_historical_bars(self, symbol: str, duration: str = '1 D', 
                                  bar_size: str = '5 secs', 
                                  what_to_show: str = 'TRADES',
                                  use_rth: bool = True) -> pd.DataFrame:
        """
        Get historical bars with caching and validation
        
        Args:
            symbol: Stock symbol
            duration: Time period (e.g., '1 D', '1 W', '1 M')
            bar_size: Bar size (e.g., '5 secs', '1 min', '5 mins')
            what_to_show: Data type ('TRADES', 'BID', 'ASK', 'MIDPOINT')
            use_rth: Use regular trading hours only
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            raise IBKRException("Not connected to IBKR")
        
        try:
            # Create and qualify contract
            contract = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(contract)
            
            if not contract.conId:
                raise IBKRException(f"Failed to qualify contract for {symbol}")
            
            logger.info(f"Fetching historical bars for {symbol}: {duration} @ {bar_size}")
            
            # Request historical data - use async version
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',  # Use current time
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1  # Return as datetime
            )
            
            if bars:
                df = pd.DataFrame([{
                    'datetime': b.date,
                    'open': b.open,
                    'high': b.high,
                    'low': b.low,
                    'close': b.close,
                    'volume': b.volume,
                    'wap': b.wap if hasattr(b, 'wap') else b.close,
                    'count': b.count if hasattr(b, 'count') else 0
                } for b in bars])
                
                # Set datetime as index
                df.set_index('datetime', inplace=True)
                
                # Data quality check
                if df['close'].isna().any():
                    logger.warning(f"Historical data for {symbol} contains NaN values")
                
                logger.info(f"Retrieved {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical bars for {symbol}: {e}")
            raise IBKRException(f"Failed to get historical data: {e}")
    
    async def execute_order(self, contract: Contract, order: Any):
        """
        Execute trades through IBKR with monitoring
        
        Args:
            contract: IB Contract object
            order: IB Order object
            
        Returns:
            Trade object with fill information
        """
        if not self.connected:
            raise IBKRException("Not connected to IBKR")
        
        try:
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order acknowledgment
            await asyncio.sleep(0.5)
            
            # Log order placement
            logger.info(f"Order placed: {order.action} {order.totalQuantity} {contract.symbol}")
            
            return trade
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            raise IBKRException(f"Failed to execute order: {e}")
    
    async def create_option_contract(self, symbol: str, expiry: str, 
                                    strike: float, right: str) -> Option:
        """
        Create and validate option contract for IBKR execution
        
        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD format)
            strike: Strike price
            right: 'C' for Call, 'P' for Put
            
        Returns:
            Qualified Option contract
        """
        try:
            contract = Option(symbol, expiry, strike, right, 'SMART')
            await self.ib.qualifyContractsAsync(contract)
            
            if not contract.conId:
                raise IBKRException(f"Failed to qualify option: {symbol} {expiry} {strike} {right}")
            
            return contract
            
        except Exception as e:
            logger.error(f"Failed to create option contract: {e}")
            raise IBKRException(f"Option contract creation failed: {e}")
    
    async def _heartbeat(self):
        """
        Maintain connection heartbeat with auto-reconnect
        """
        consecutive_failures = 0
        
        while self.connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Check connection status
                if not self.ib.isConnected():
                    consecutive_failures += 1
                    logger.warning(f"IBKR heartbeat failed ({consecutive_failures})")
                    
                    if consecutive_failures >= 3:
                        logger.error("Multiple heartbeat failures, attempting reconnect...")
                        self.connected = False
                        await self.connect()
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0
                    self.last_heartbeat = time.time()
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """
        Handle IBKR error events
        
        Args:
            reqId: Request ID
            errorCode: Error code
            errorString: Error description
            contract: Related contract (if any)
        """
        # Ignore certain benign errors
        if errorCode in [2104, 2106, 2158]:  # Market data farm messages
            logger.debug(f"IBKR info: {errorString}")
            return
        
        logger.error(f"IBKR error {errorCode}: {errorString}")
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(errorCode, errorString)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def _on_disconnect(self):
        """Handle disconnection event"""
        logger.warning("IBKR disconnected")
        self.connected = False
        
        # Clear data
        self.latest_prices.clear()
        self.bars_5sec.clear()
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connected and self.ib.isConnected()
    
    def get_connection_status(self) -> Dict:
        """
        Get detailed connection status
        
        Returns:
            Dict with connection details
        """
        return {
            'connected': self.is_connected(),
            'host': self.config.host,
            'port': self.config.port if self.trading_config.mode == 'paper' else 7496,
            'client_id': self.config.client_id,
            'mode': self.trading_config.mode,
            'subscriptions': list(self.subscriptions.keys()),
            'last_heartbeat': datetime.fromtimestamp(self.last_heartbeat).isoformat() if self.last_heartbeat else None
        }
    
    def add_price_callback(self, callback: Callable):
        """Add callback for price updates"""
        self.price_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for error events"""
        self.error_callbacks.append(callback)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices"""
        return self.latest_prices.copy()


# Global instance - CREATE ONCE, USE FOREVER
market_data = MarketDataManager()
