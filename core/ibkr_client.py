#!/usr/bin/env python3
"""
Interactive Brokers Client for Level 2 Data and Paper Trading
Handles market depth, trades, bars, and order execution
"""

import asyncio
import time
import os
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import defaultdict
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from ib_insync import (
    IB, Stock, Option, Contract, MarketOrder, LimitOrder, StopOrder,
    BarDataList, Ticker, Trade, Order, OrderStatus, util
)

from .models import (
    OrderBook, OrderBookLevel, Trade as TradeModel,
    Bar as BarModel, Position, SignalAction
)
from .cache import CacheManager


class IBKRClient:
    """IBKR client for Level 2 data and paper trading"""

    def __init__(self, cache_manager: CacheManager, config: Dict):
        """Initialize IBKR client with configuration"""
        self.cache = cache_manager
        # Process environment variables in config
        self.config = self._substitute_env_vars(config['ibkr'])
        self.ib = IB()

        # Ensure integer types for port and client_id
        if isinstance(self.config.get('port'), str) and not self.config.get('port', '').startswith('${'):
            try:
                self.config['port'] = int(self.config['port'])
            except ValueError:
                self.config['port'] = 7497  # Default to paper trading port
        elif isinstance(self.config.get('port'), str):
            self.config['port'] = 7497  # Default if env var not processed
            
        if isinstance(self.config.get('client_id'), str) and not self.config.get('client_id', '').startswith('${'):
            try:
                self.config['client_id'] = int(self.config['client_id'])
            except ValueError:
                self.config['client_id'] = 1  # Default client ID
        elif isinstance(self.config.get('client_id'), str):
            self.config['client_id'] = 1  # Default if env var not processed
        
        # Support dynamic client ID to avoid conflicts
        self._base_client_id = self.config.get('client_id', 1)
        self._client_id_offset = 0  # Will increment on reconnect attempts

        # Data storage
        self.order_books: Dict[str, OrderBook] = {}
        self.recent_trades: Dict[str, List[TradeModel]] = defaultdict(list)
        self.bars: Dict[str, BarModel] = {}

        # Active subscriptions
        self.market_depth_subs = {}
        self.ticker_subs = {}
        self.bar_subs = {}

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Connection state
        self.connected = False
        self.reconnect_count = 0
        self.max_reconnects = 5

        logger.info(f"IBKR client initialized for {self.config['host']}:{self.config['port']}")

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Parse ${VAR:default} format
            var_expr = config[2:-1]
            if ':' in var_expr:
                var_name, default = var_expr.split(':', 1)
                # Handle empty default case properly
                if default == '':
                    # Empty default means use empty string if env var not set
                    value = os.getenv(var_name, '')
                else:
                    value = os.getenv(var_name, default)
            else:
                # No default specified, use empty string if not found
                value = os.getenv(var_expr, '')
            
            # CRITICAL FIX: Don't return the template if env var not found
            # If value is still the original ${...} format, use empty string
            if value == config:
                value = ''
            
            # Try to convert to appropriate type
            if value and value != '':
                if value.isdigit():
                    return int(value)
                elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    return float(value)
                elif value.lower() in ('true', 'yes', '1'):
                    return True
                elif value.lower() in ('false', 'no', '0'):
                    return False
            
            return value
        return config

    async def connect(self, retry_count: int = 3) -> bool:
        """Connect to TWS/IB Gateway with retry logic"""
        for attempt in range(retry_count):
            try:
                # Clean up any existing connection first
                if self.ib.isConnected():
                    logger.info("Disconnecting existing connection...")
                    self.ib.disconnect()
                    await asyncio.sleep(1)  # Give time for clean disconnect
                
                # Calculate client ID with offset to avoid conflicts
                client_id = self._base_client_id + self._client_id_offset + attempt
                logger.info(f"Attempting connection (attempt {attempt + 1}/{retry_count}) with client ID {client_id}...")
                
                # Increase timeout for initial connection (30 seconds)
                await self.ib.connectAsync(
                    host=self.config['host'],
                    port=self.config['port'],
                    clientId=client_id,
                    timeout=30,  # Increased from 10 to 30 seconds
                    readonly=self.config.get('readonly', False)
                )

                # Verify connection is actually established
                if not self.ib.isConnected():
                    raise ConnectionError("Connection reported success but is not connected")
                
                self.connected = True
                self.reconnect_count = 0

                # Set up event handlers
                self._setup_event_handlers()

                # Wait a moment for connection to stabilize
                await asyncio.sleep(1)
                
                # Test connection with account request
                try:
                    # Use reqAccountSummary with timeout
                    account_future = asyncio.create_task(self._get_account_summary_async())
                    account_summary = await asyncio.wait_for(account_future, timeout=5)
                    
                    if account_summary:
                        for item in account_summary:
                            if item.tag == 'NetLiquidation':
                                logger.info(f"Account connected, Net Liquidation: ${float(item.value):,.2f}")
                                break
                except asyncio.TimeoutError:
                    logger.warning("Account summary timed out but connection appears stable")

                logger.success(f"Connected to IBKR at {self.config['host']}:{self.config['port']} with client ID {client_id}")
                return True

            except asyncio.TimeoutError:
                logger.warning(f"Connection attempt {attempt + 1} timed out")
                self._client_id_offset += 1  # Try different client ID next time
                if attempt < retry_count - 1:
                    await asyncio.sleep(2)  # Wait before retry
                    
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                self._client_id_offset += 1  # Try different client ID next time
                if attempt < retry_count - 1:
                    await asyncio.sleep(2)  # Wait before retry
        
        self.connected = False
        return False
    
    async def _get_account_summary_async(self):
        """Get account summary asynchronously"""
        return self.ib.accountSummary()

    def _setup_event_handlers(self):
        """Set up IB event handlers"""
        # Connection events - Use lambda to preserve self binding
        self.ib.disconnectedEvent += lambda: self._on_disconnect()
        self.ib.errorEvent += lambda reqId, errorCode, errorString, contract: self._on_error(reqId, errorCode, errorString, contract)

        # Market data events
        self.ib.updateEvent += lambda: self._on_update()

    def _on_disconnect(self):
        """Handle disconnection"""
        logger.warning("Disconnected from IBKR")
        self.connected = False

        # Only auto-reconnect if we haven't exceeded max attempts
        # and there's a running event loop
        if self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            logger.info(f"Attempting reconnect {self.reconnect_count}/{self.max_reconnects}")
            try:
                loop = asyncio.get_running_loop()
                # Schedule reconnect with delay
                loop.create_task(self._reconnect())
            except RuntimeError:
                # No running event loop - can't auto-reconnect
                logger.warning("No running event loop - cannot auto-reconnect")
    
    async def disconnect(self):
        """Cleanly disconnect from IBKR"""
        try:
            if self.ib.isConnected():
                # Cancel all subscriptions
                for symbol in list(self.market_depth_subs.keys()):
                    await self.unsubscribe_market_depth(symbol)
                
                for symbol in list(self.bar_subs.keys()):
                    self.unsubscribe_bars(symbol)
                
                # Disconnect
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def _reconnect(self):
        """Attempt to reconnect"""
        await asyncio.sleep(5)  # Wait before reconnecting
        self._client_id_offset += 1  # Use new client ID for reconnect

        if await self.connect():
            logger.info("Successfully reconnected")
            # Resubscribe to market data
            await self._resubscribe_all()
        else:
            logger.error("Reconnection failed")

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB errors"""
        # Informational messages - not errors
        if errorCode in [2104, 2106, 2107, 2108, 2158]:  # Market data farm connections
            logger.debug(f"Market data farm message: {errorString}")
        # Market data permissions - not critical for paper trading
        elif errorCode == 2152:  # Need additional market data permissions
            logger.debug(f"Market data permissions: {errorString}")
            return  # Don't log as error
        # Warning level errors
        elif errorCode in [200, 202, 203]:  # Security/contract related
            logger.warning(f"Contract issue {errorCode}: {errorString}")
        # Connection errors - trigger reconnect
        elif errorCode in [502, 503, 504, 1100, 1101, 1102]:  
            logger.error(f"Connection error {errorCode}: {errorString}")
            if errorCode == 1100:  # Connection lost
                self.connected = False
        # Client ID already in use
        elif errorCode == 326:
            logger.error(f"Client ID already in use - incrementing offset")
            self._client_id_offset += 10  # Jump by 10 to find free ID
        # Other errors
        else:
            logger.error(f"IB Error {errorCode}: {errorString}")

    def _on_update(self):
        """Handle general updates"""
        pass  # Process specific updates in their handlers

    async def subscribe_market_depth(self, symbol: str, num_rows: int = 10, exchanges: Optional[List[str]] = None) -> bool:
        """Subscribe to Level 2 order book - SMART routing with aggregated data"""
        try:
            # Use SMART routing with Smart Depth for aggregated data from all exchanges
            # API v974+ required - isSmartDepth=True enables SMART exchange support
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market depth with Smart Depth enabled for SMART exchange
            logger.debug(f"Requesting SMART market depth for {symbol} with {num_rows} rows")
            # Note: reqMktDepth doesn't return a request ID in ib_insync
            self.ib.reqMktDepth(
                contract,
                numRows=num_rows,
                isSmartDepth=True,  # CRITICAL: Must be True for SMART exchange
                mktDepthOptions=[]
            )
            
            # Store subscription info for proper cleanup
            self.market_depth_subs[symbol] = {
                'contract': contract,
                'isSmartDepth': True,  # Store this for proper cancellation
                'ticker': None  # Will be set below
            }
            
            # CRITICAL: Get ticker and connect update handler
            # This is what was missing - we need to properly connect the event
            ticker = self.ib.ticker(contract)
            
            if ticker:
                # Store ticker reference
                self.market_depth_subs[symbol]['ticker'] = ticker
                
                # Create handler for this specific symbol
                # Using closure to capture symbol correctly
                def create_handler(sym):
                    def on_depth_update(ticker_obj):
                        # Only process if we have actual depth data
                        if ticker_obj.domBids or ticker_obj.domAsks:
                            self._process_order_book(sym, ticker_obj)
                    return on_depth_update
                
                # Connect the handler
                handler = create_handler(symbol)
                ticker.updateEvent += handler  # type: ignore
                
                # Store handler reference for cleanup
                self.market_depth_subs[symbol]['handler'] = handler
                
                logger.info(f"✓ Subscribed to Level 2 for {symbol} using SMART routing")
                
                # Give it a moment to start receiving data
                await asyncio.sleep(0.5)
                
                # Check if we're getting data
                if ticker.domBids or ticker.domAsks:
                    logger.success(f"✓ Level 2 data flowing for {symbol}")
                else:
                    logger.warning(f"⚠ No Level 2 data yet for {symbol}, may take a moment...")
                
                return True
            else:
                logger.error(f"Failed to get ticker for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Failed to subscribe to market depth for {symbol}: {e}")
            return False

    def _process_order_book(self, symbol: str, ticker, exchange: Optional[str] = None):
        """Process order book update with better validation and logging"""
        try:
            # Validate we have actual depth data
            if not ticker.domBids and not ticker.domAsks:
                # No depth data yet - this is normal at startup
                return
            
            # Build order book from ticker
            bids = []
            asks = []

            if ticker.domBids:
                for level in ticker.domBids[:10]:  # Max 10 levels
                    # Validate price is positive
                    if level.price and level.price > 0:
                        bids.append(OrderBookLevel(
                            price=level.price,
                            size=level.size or 0,
                            market_maker=level.marketMaker or ""
                        ))

            if ticker.domAsks:
                for level in ticker.domAsks[:10]:  # Max 10 levels
                    # Validate price is positive
                    if level.price and level.price > 0:
                        asks.append(OrderBookLevel(
                            price=level.price,
                            size=level.size or 0,
                            market_maker=level.marketMaker or ""
                        ))

            # Only create order book if we have at least one side
            if bids or asks:
                order_book = OrderBook(
                    symbol=symbol,
                    timestamp=int(time.time() * 1000),
                    bids=bids,
                    asks=asks
                )

                # Store locally
                self.order_books[symbol] = order_book

                # Cache it with verification
                if self.cache.set_order_book(symbol, order_book.dict()):
                    # Log periodically to avoid spam
                    if not hasattr(self, '_last_ob_log') or time.time() - self._last_ob_log > 10:
                        logger.debug(f"✓ L2 cached for {symbol}: {len(bids)} bids, {len(asks)} asks, spread: {order_book.spread:.4f}")
                        self._last_ob_log = time.time()
                else:
                    logger.warning(f"Failed to cache order book for {symbol}")

                # Trigger callbacks
                self._trigger_callbacks(f"orderbook_{symbol}", order_book)
            else:
                logger.debug(f"No valid price levels for {symbol} order book")

        except Exception as e:
            logger.error(f"Error processing order book for {symbol}: {e}")

    async def subscribe_trades(self, symbol: str) -> bool:
        """Subscribe to trade tape with improved event handling"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')

            # Request tick data with extended hours support
            logger.debug(f"Requesting market data for {symbol}")
            self.ib.reqMktData(
                contract,
                genericTickList='233',  # RTVolume for trade tape
                snapshot=False,
                regulatorySnapshot=False
            )
            
            # Enable extended hours market data
            self.ib.reqMarketDataType(4)  # 4 = Delayed frozen (works in extended hours)

            # Get ticker and set up handler
            ticker = self.ib.ticker(contract)
            if ticker:
                # Create specific handler for this symbol
                def create_trade_handler(sym):
                    def on_trade_update(ticker_obj):
                        # Check for actual trade data (last price and size)
                        if ticker_obj.last and ticker_obj.last > 0 and ticker_obj.lastSize and ticker_obj.lastSize > 0:
                            self._process_trade(sym, ticker_obj)
                    return on_trade_update
                
                # Connect the handler
                handler = create_trade_handler(symbol)
                if hasattr(ticker, 'updateEvent'):
                    ticker.updateEvent += handler  # type: ignore  # type: ignore
                
                # Store ticker and handler
                self.ticker_subs[symbol] = {
                    'ticker': ticker,
                    'handler': handler,
                    'contract': contract
                }
                
                logger.info(f"✓ Subscribed to trades for {symbol}")
                
                # Give it a moment to start receiving data
                await asyncio.sleep(0.5)
                
                # Check if we're getting data
                if ticker.last and ticker.last > 0:
                    logger.success(f"✓ Trade data flowing for {symbol}: Last ${ticker.last:.2f}")
                else:
                    logger.warning(f"⚠ No trade data yet for {symbol}, may take a moment...")
                
                return True
            else:
                logger.error(f"Failed to get ticker for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Failed to subscribe to trades for {symbol}: {e}")
            return False

    def _process_trade(self, symbol: str, ticker):
        """Process trade update with better validation and caching"""
        try:
            # Validate we have valid trade data
            if ticker.last and ticker.last > 0 and ticker.lastSize and ticker.lastSize > 0:
                trade = TradeModel(
                    symbol=symbol,
                    timestamp=int(time.time() * 1000),
                    price=float(ticker.last),
                    size=int(ticker.lastSize),
                    is_buyer=None  # Can't determine from tick data
                )

                # Store recent trades locally
                self.recent_trades[symbol].append(trade)
                if len(self.recent_trades[symbol]) > 1000:
                    self.recent_trades[symbol] = self.recent_trades[symbol][-1000:]

                # Cache trade with verification
                if self.cache.append_trade(symbol, trade.dict()):
                    # Log periodically to avoid spam
                    if not hasattr(self, '_last_trade_log') or time.time() - self._last_trade_log > 10:
                        logger.debug(f"✓ Trade cached: {symbol} @ ${trade.price:.2f} x {trade.size}")
                        self._last_trade_log = time.time()
                else:
                    logger.warning(f"Failed to cache trade for {symbol}")

                # Trigger callbacks
                self._trigger_callbacks(f"trade_{symbol}", trade)

        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")

    async def subscribe_bars(self, symbol: str, bar_size: str = "5 secs") -> bool:
        """Subscribe to real-time bars"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')

            # Request real-time bars with extended hours support
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=5,  # Only 5 second bars supported for real-time
                whatToShow='TRADES',  # Must be TRADES for extended hours
                useRTH=False  # FALSE for extended hours trading!
            )

            # Set up bar handler - ib_insync passes TWO arguments: (bars, hasNewBar)
            def on_bar_update(bars_obj, has_new_bar):
                if has_new_bar:  # Only process when there's actually a new bar
                    self._process_bar(symbol, bars_obj)
            bars.updateEvent += on_bar_update

            self.bar_subs[symbol] = bars
            logger.info(f"Subscribed to {bar_size} bars for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to bars for {symbol}: {e}")
            return False

    def _process_bar(self, symbol: str, bars: BarDataList):
        """Process bar update - handles both RealTimeBar and BarData correctly"""
        try:
            if bars:
                latest_bar = bars[-1]

                # CRITICAL FIX: Handle RealTimeBar vs BarData
                # RealTimeBar has 'time' attribute (Unix timestamp)
                # BarData has 'date' attribute (datetime object)
                
                import datetime as dt
                
                # Check for 'date' first (most common - BarData)
                if hasattr(latest_bar, 'date'):
                    # Historical BarData - has 'date' as datetime
                    if hasattr(latest_bar.date, 'timestamp'):
                        timestamp = int(latest_bar.date.timestamp() * 1000)  # type: ignore
                    elif isinstance(latest_bar.date, dt.datetime):
                        timestamp = int(latest_bar.date.timestamp() * 1000)
                    elif isinstance(latest_bar.date, dt.date):
                        # It's a date object, convert to datetime
                        dt_obj = dt.datetime.combine(latest_bar.date, dt.time.min)
                        timestamp = int(dt_obj.timestamp() * 1000)
                    else:
                        # Fallback to current time
                        timestamp = int(time.time() * 1000)
                        logger.warning(f"Unexpected date format for {symbol}: {type(latest_bar.date)}")
                    logger.debug(f"Processing BarData for {symbol}, timestamp: {timestamp}")
                    
                elif hasattr(latest_bar, 'time'):
                    # RealTimeBar - has 'time' which could be Unix timestamp or datetime
                    # Use getattr to avoid IDE errors while maintaining same logic
                    bar_time = getattr(latest_bar, 'time', None)
                    
                    if bar_time is not None:
                        if isinstance(bar_time, dt.datetime):
                            timestamp = int(bar_time.timestamp() * 1000)
                        elif isinstance(bar_time, (int, float)):
                            # Unix timestamp in seconds - convert to milliseconds
                            timestamp = int(bar_time * 1000)
                        else:
                            # Unexpected type - fallback to current time
                            timestamp = int(time.time() * 1000)
                            logger.warning(f"Unexpected time type for {symbol}: {type(bar_time)}")
                    else:
                        # time attribute exists but is None - shouldn't happen
                        timestamp = int(time.time() * 1000)
                        logger.warning(f"Bar for {symbol} has time=None, using current time")
                    
                    logger.debug(f"Processing RealTimeBar for {symbol}, timestamp: {timestamp}")
                    
                else:
                    # Neither 'time' nor 'date' attribute found - shouldn't happen
                    timestamp = int(time.time() * 1000)
                    logger.warning(f"Bar for {symbol} has neither 'time' nor 'date' attribute, using current time")

                # Handle different attribute names for RealTimeBar vs BarData
                # RealTimeBar uses 'open_' while BarData uses 'open'
                open_price = getattr(latest_bar, 'open_', None) or getattr(latest_bar, 'open', 0.0)
                
                # Ensure all OHLC values are valid floats (not None)
                bar = BarModel(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(open_price) if open_price else 0.0,
                    high=float(latest_bar.high) if latest_bar.high else 0.0,
                    low=float(latest_bar.low) if latest_bar.low else 0.0,
                    close=float(latest_bar.close) if latest_bar.close else 0.0,
                    volume=int(latest_bar.volume) if latest_bar.volume else 0,  # Ensure int type
                    vwap=getattr(latest_bar, 'wap', None) or getattr(latest_bar, 'average', None),  # RealTimeBar has 'wap', BarData has 'average'
                    bar_count=getattr(latest_bar, 'count', None) or getattr(latest_bar, 'barCount', None)  # RealTimeBar has 'count', BarData has 'barCount'
                )

                # Store latest bar
                self.bars[symbol] = bar

                # Cache bar
                self.cache.set(f"bar:{symbol}", bar.dict(), ttl=10)

                # Trigger callbacks
                self._trigger_callbacks(f"bar_{symbol}", bar)

        except Exception as e:
            logger.error(f"Error processing bar for {symbol}: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: str = "1 min"
    ) -> Optional[List[BarModel]]:
        """Get historical bars"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,  # FALSE for extended hours data!
                formatDate=1
            )

            if bars:
                result = []
                for bar in bars:
                    # Handle datetime properly
                    import datetime as dt
                    if hasattr(bar.date, 'timestamp'):
                        timestamp = int(bar.date.timestamp() * 1000)  # type: ignore
                    elif isinstance(bar.date, dt.date) and not isinstance(bar.date, dt.datetime):
                        # It's a date object, convert to datetime
                        dt_obj = dt.datetime.combine(bar.date, dt.time.min)
                        timestamp = int(dt_obj.timestamp() * 1000)
                    elif isinstance(bar.date, dt.datetime):
                        timestamp = int(bar.date.timestamp() * 1000)
                    else:
                        timestamp = int(time.time() * 1000)

                    result.append(
                        BarModel(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=bar.open,
                            high=bar.high,
                            low=bar.low,
                            close=bar.close,
                            volume=int(bar.volume)  # Ensure int type
                        )
                    )
                return result

            return None

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    # Order Execution Methods (Paper Trading)
    async def place_order(
        self,
        symbol: str,
        action: SignalAction,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Trade]:
        """Place order (paper trading)"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')

            # Create order based on type
            if order_type == "MARKET":
                order = MarketOrder(
                    action=action.value,
                    totalQuantity=quantity
                )
            elif order_type == "LIMIT":
                if limit_price is None:
                    logger.error("Limit price required for LIMIT order")
                    return None
                order = LimitOrder(
                    action=action.value,
                    totalQuantity=quantity,
                    lmtPrice=limit_price
                )
            elif order_type == "STOP":
                if stop_price is None:
                    logger.error("Stop price required for STOP order")
                    return None
                order = StopOrder(
                    action=action.value,
                    totalQuantity=quantity,
                    stopPrice=stop_price
                )
            else:
                logger.error(f"Unknown order type: {order_type}")
                return None

            # Add adaptive algo for better fills
            order.algoStrategy = self.config['order_defaults'].get('algo_strategy', 'Adaptive')
            order.tif = self.config['order_defaults'].get('tif', 'DAY')

            # Place the order
            trade = self.ib.placeOrder(contract, order)

            logger.info(f"Placed {order_type} order: {action.value} {quantity} {symbol}")

            return trade

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def cancel_order(self, order: Order) -> bool:
        """Cancel an order"""
        try:
            self.ib.cancelOrder(order)
            logger.info(f"Cancelled order {order.orderId}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = []
            for position in self.ib.positions():
                positions.append({
                    'symbol': position.contract.symbol,
                    'position': position.position,
                    'avg_cost': position.avgCost,
                    'market_value': getattr(position, 'marketValue', 0),  # Safe access
                    'unrealized_pnl': getattr(position, 'unrealizedPnL', 0),  # Safe access
                    'realized_pnl': getattr(position, 'realizedPnL', 0)  # Safe access
                })

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_account_summary(self) -> Dict:
        """Get account summary"""
        try:
            summary = {}
            for item in self.ib.accountSummary():
                summary[item.tag] = item.value

            return {
                'buying_power': float(summary.get('BuyingPower', 0)),
                'net_liquidation': float(summary.get('NetLiquidation', 0)),
                'cash': float(summary.get('TotalCashValue', 0)),
                'unrealized_pnl': float(summary.get('UnrealizedPnL', 0)),
                'realized_pnl': float(summary.get('RealizedPnL', 0))
            }

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}

    # Callback Management
    def register_callback(self, event: str, callback: Callable):
        """Register callback for events"""
        self.callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered callbacks"""
        for callback in self.callbacks[event]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    # Subscription Management
    async def _resubscribe_all(self):
        """Resubscribe to all market data after reconnection"""
        # Resubscribe market depth
        for symbol in list(self.market_depth_subs.keys()):
            await self.subscribe_market_depth(symbol)

        # Resubscribe trades
        for symbol in list(self.ticker_subs.keys()):
            await self.subscribe_trades(symbol)

        # Resubscribe bars
        for symbol in list(self.bar_subs.keys()):
            await self.subscribe_bars(symbol)

    async def unsubscribe_all(self):
        """Unsubscribe from all market data with proper cleanup"""
        # Unsubscribe market depth
        for symbol in list(self.market_depth_subs.keys()):
            await self.unsubscribe_market_depth(symbol)
        
        # Unsubscribe trades
        for symbol, sub_info in self.ticker_subs.items():
            try:
                if isinstance(sub_info, dict):
                    if 'ticker' in sub_info:
                        ticker = sub_info['ticker']
                        if 'contract' in sub_info:
                            self.ib.cancelMktData(sub_info['contract'])
                        elif hasattr(ticker, 'contract'):
                            self.ib.cancelMktData(ticker.contract)
                        # Clear event handler
                        if 'handler' in sub_info and hasattr(ticker, 'updateEvent'):
                            try:
                                ticker.updateEvent -= sub_info['handler']
                            except:
                                pass
                else:
                    # Legacy format - sub_info is the ticker itself
                    if hasattr(sub_info, 'contract'):
                        self.ib.cancelMktData(sub_info.contract)
            except Exception as e:
                logger.warning(f"Error unsubscribing trades for {symbol}: {e}")
        
        # Unsubscribe bars
        for bars in self.bar_subs.values():
            try:
                self.ib.cancelRealTimeBars(bars)
            except Exception as e:
                logger.warning(f"Error unsubscribing bars: {e}")

        self.market_depth_subs.clear()
        self.ticker_subs.clear()
        self.bar_subs.clear()

        logger.info("✓ Unsubscribed from all market data")

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected and self.ib.isConnected()
    
    async def health_check(self) -> bool:
        """Perform health check on connection"""
        if not self.is_connected():
            return False
        
        try:
            # Try a simple request with timeout
            future = asyncio.create_task(self._get_account_summary_async())
            await asyncio.wait_for(future, timeout=2)
            return True
        except (asyncio.TimeoutError, Exception):
            return False
    
    async def unsubscribe_market_depth(self, symbol: str) -> bool:
        """Unsubscribe from market depth with proper cleanup"""
        try:
            if symbol in self.market_depth_subs:
                sub_info = self.market_depth_subs[symbol]
                
                # Handle new dictionary format
                if isinstance(sub_info, dict):
                    # Cancel market depth subscription using contract with same isSmartDepth flag
                    if 'contract' in sub_info:
                        # Use the same isSmartDepth flag as when subscribing
                        is_smart_depth = sub_info.get('isSmartDepth', True)
                        self.ib.cancelMktDepth(
                            sub_info['contract'],
                            isSmartDepth=is_smart_depth
                        )
                    
                    # Disconnect event handler if ticker exists
                    if 'ticker' in sub_info and 'handler' in sub_info:
                        ticker = sub_info['ticker']
                        handler = sub_info['handler']
                        # Remove specific handler
                        try:
                            ticker.updateEvent -= handler
                        except:
                            # Handler might already be removed
                            pass
                    
                    # Legacy multiple contracts format
                    elif 'contracts' in sub_info:
                        for contract in sub_info['contracts']:
                            self.ib.cancelMktDepth(contract, isSmartDepth=True)
                else:
                    # Legacy: sub_info might be a Ticker object or contract
                    if hasattr(sub_info, 'contract'):
                        # It's a Ticker object, use its contract
                        self.ib.cancelMktDepth(sub_info.contract, isSmartDepth=True)
                    else:
                        # Assume it's a contract
                        self.ib.cancelMktDepth(sub_info, isSmartDepth=True)  # type: ignore
                
                # Remove from tracking
                del self.market_depth_subs[symbol]
                
                logger.info(f"✓ Unsubscribed from Level 2 for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unsubscribe from market depth for {symbol}: {e}")
        return False
    
    def unsubscribe_bars(self, symbol: str) -> bool:
        """Unsubscribe from real-time bars"""
        try:
            if symbol in self.bar_subs:
                bars = self.bar_subs[symbol]
                self.ib.cancelRealTimeBars(bars)
                del self.bar_subs[symbol]
                logger.info(f"Unsubscribed from real-time bars for {symbol}")
                return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from bars for {symbol}: {e}")
        return False
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last traded price for symbol"""
        try:
            if symbol in self.ticker_subs:
                ticker = self.ticker_subs[symbol]
                if ticker and hasattr(ticker, 'last') and ticker.last:
                    return float(ticker.last)
        except Exception as e:
            logger.error(f"Error getting last price for {symbol}: {e}")
        return None
    
    async def get_spot_price(self, symbol: str, timeout: int = 5) -> Optional[float]:
        """Get spot price with multiple fallback methods"""
        # Method 1: Try existing subscription
        price = self.get_last_price(symbol)
        if price and price > 0:
            return price
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Method 2: Request snapshot with extended hours
            self.ib.reqMarketDataType(4)  # Enable extended hours
            ticker = self.ib.reqMktData(contract, '', True, False)  # Snapshot mode
            await asyncio.sleep(2)  # Wait for snapshot
            
            if ticker.last and ticker.last > 0:
                # No need to cancel - snapshot mode auto-terminates
                return float(ticker.last)
            elif ticker.close and ticker.close > 0:
                # No need to cancel - snapshot mode auto-terminates
                return float(ticker.close)
            
            # Method 3: Get from historical data (most recent bar)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,  # Include extended hours
                formatDate=1
            )
            
            if bars and len(bars) > 0:
                # No need to cancel - snapshot mode auto-terminates
                return float(bars[-1].close)
            
            # Method 4: Use bid/ask midpoint
            if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                midpoint = (ticker.bid + ticker.ask) / 2
                # No need to cancel - snapshot mode auto-terminates
                return float(midpoint)
            
            # No need to cancel - snapshot mode auto-terminates
            logger.error(f"Cannot get spot price for {symbol} - all methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get spot price for {symbol}: {e}")
            return None
    
    # =================== MONITORING & DEBUGGING METHODS ===================
    
    def get_market_data_status(self) -> Dict:
        """Get comprehensive status of all market data subscriptions"""
        status = {
            'connected': self.is_connected(),
            'market_depth': {},
            'trades': {},
            'bars': {},
            'data_flowing': {
                'order_books': 0,
                'trades': 0,
                'bars': 0
            }
        }
        
        # Check market depth subscriptions
        for symbol, sub_info in self.market_depth_subs.items():
            if isinstance(sub_info, dict) and 'ticker' in sub_info:
                ticker = sub_info['ticker']
                # Handle both dict and Ticker object types
                if isinstance(ticker, dict):
                    dom_bids = ticker.get('domBids', [])
                    dom_asks = ticker.get('domAsks', [])
                else:
                    dom_bids = getattr(ticker, 'domBids', []) if ticker else []
                    dom_asks = getattr(ticker, 'domAsks', []) if ticker else []
                
                has_bids = bool(dom_bids)
                has_asks = bool(dom_asks)
                status['market_depth'][symbol] = {
                    'subscribed': True,
                    'has_bids': has_bids,
                    'has_asks': has_asks,
                    'bid_levels': len(dom_bids),
                    'ask_levels': len(dom_asks)
                }
                if has_bids or has_asks:
                    status['data_flowing']['order_books'] += 1
        
        # Check trade subscriptions
        for symbol, sub_info in self.ticker_subs.items():
            if isinstance(sub_info, dict) and 'ticker' in sub_info:
                ticker = sub_info['ticker']
            else:
                ticker = sub_info
            
            # Handle both dict and Ticker object types
            if isinstance(ticker, dict):
                last_price = ticker.get('last', 0)
                last_size = ticker.get('lastSize', None)
            else:
                last_price = getattr(ticker, 'last', 0) if ticker else 0
                last_size = getattr(ticker, 'lastSize', None) if ticker else None
            
            has_last = bool(last_price and last_price > 0)
            status['trades'][symbol] = {
                'subscribed': True,
                'has_last_price': has_last,
                'last_price': last_price if has_last else None,
                'last_size': last_size
            }
            if has_last:
                status['data_flowing']['trades'] += 1
        
        # Check bar subscriptions
        for symbol, bars in self.bar_subs.items():
            has_bars = bool(bars and len(bars) > 0)
            status['bars'][symbol] = {
                'subscribed': True,
                'has_bars': has_bars,
                'bar_count': len(bars) if bars else 0
            }
            if has_bars:
                status['data_flowing']['bars'] += 1
        
        # Check cached data
        status['cached_data'] = {
            'order_books': len(self.order_books),
            'recent_trades': sum(len(trades) for trades in self.recent_trades.values()),
            'bars': len(self.bars)
        }
        
        return status
    
    def log_market_data_status(self):
        """Log current market data status for debugging"""
        status = self.get_market_data_status()
        
        logger.info("=" * 60)
        logger.info("MARKET DATA STATUS REPORT")
        logger.info("=" * 60)
        logger.info(f"Connected: {status['connected']}")
        logger.info(f"Data Flowing - Order Books: {status['data_flowing']['order_books']}, "
                   f"Trades: {status['data_flowing']['trades']}, "
                   f"Bars: {status['data_flowing']['bars']}")
        
        # Market depth details
        if status['market_depth']:
            logger.info("\nMARKET DEPTH:")
            for symbol, depth_status in status['market_depth'].items():
                logger.info(f"  {symbol}: Bids={depth_status['bid_levels']}, Asks={depth_status['ask_levels']}")
        
        # Trades details
        if status['trades']:
            logger.info("\nTRADES:")
            for symbol, trade_status in status['trades'].items():
                if trade_status['has_last_price']:
                    logger.info(f"  {symbol}: ${trade_status['last_price']:.2f} x {trade_status['last_size']}")
                else:
                    logger.info(f"  {symbol}: No data")
        
        # Bars details
        if status['bars']:
            logger.info("\nBARS:")
            for symbol, bar_status in status['bars'].items():
                logger.info(f"  {symbol}: {bar_status['bar_count']} bars")
        
        logger.info("=" * 60)
    
    async def monitor_data_flow(self, interval: int = 30):
        """Monitor and log data flow periodically"""
        while self.connected:
            try:
                self.log_market_data_status()
                
                # Check cache performance
                cache_stats = self.cache.get_stats()
                logger.info(f"Cache Performance - Hit Rate: {cache_stats['hit_rate']}, "
                           f"Keys: {cache_stats['keys']}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitor_data_flow: {e}")
                await asyncio.sleep(interval)
