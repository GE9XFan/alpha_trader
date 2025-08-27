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
                value = os.getenv(var_name, default)
            else:
                value = os.getenv(var_expr, config)
            
            # Try to convert to appropriate type
            if value.isdigit():
                return int(value)
            elif value.replace('.', '', 1).isdigit():
                return float(value)
            elif value.lower() in ('true', 'yes'):
                return True
            elif value.lower() in ('false', 'no'):
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
        """Subscribe to Level 2 order book from specific exchanges"""
        try:
            # Use exchanges with L2 subscriptions - SPY specific optimization
            if exchanges is None:
                # For SPY, use ARCA (primary) and optionally BATS
                # Don't use ISLAND (causes 10089), EDGX (10092), or BZX (use BATS instead)
                if symbol == 'SPY':
                    exchanges = ['ARCA', 'BATS']  # SPY primary is ARCA
                else:
                    exchanges = ['ARCA', 'NYSE', 'BATS', 'IEX']  # General stocks
            
            successful_subs = []
            
            for exchange in exchanges:
                try:
                    contract = Stock(symbol, exchange, 'USD')
                    contract.primaryExchange = exchange  # Force specific exchange

                    # Request market depth with isSmartDepth=False for real L2
                    self.ib.reqMktDepth(
                        contract,
                        numRows=num_rows,
                        isSmartDepth=False  # CRITICAL: Must be False for real Level 2
                    )
                    
                    successful_subs.append(exchange)
                    logger.debug(f"Subscribed to L2 for {symbol} on {exchange}")
                    
                except Exception as e:
                    logger.warning(f"L2 subscription failed for {symbol} on {exchange}: {e}")
                    continue
            
            if not successful_subs:
                logger.error(f"Failed to subscribe to any L2 exchange for {symbol}")
                return False

            # Set up order book handler for aggregated data
            # Store all exchange contracts
            self.market_depth_subs[symbol] = {'exchanges': successful_subs, 'contracts': []}
            
            # Aggregate order books from all exchanges
            for exchange in successful_subs:
                contract = Stock(symbol, exchange, 'USD')
                contract.primaryExchange = exchange
                ticker = self.ib.ticker(contract)
                if ticker and hasattr(ticker, 'updateEvent'):
                    # Use proper callback binding for ticker updates
                    def make_handler(exch):
                        def on_ticker_update(ticker_obj):
                            self._process_order_book(symbol, ticker_obj, exchange=exch)
                        return on_ticker_update
                    ticker.updateEvent += make_handler(exchange)  # type: ignore
                self.market_depth_subs[symbol]['contracts'].append(contract)

            logger.info(f"Subscribed to Level 2 for {symbol} on exchanges: {successful_subs}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to market depth for {symbol}: {e}")
            return False

    def _process_order_book(self, symbol: str, ticker, exchange: Optional[str] = None):
        """Process order book update, aggregating from multiple exchanges"""
        try:
            # Build order book from ticker
            bids = []
            asks = []

            if ticker.domBids:
                for level in ticker.domBids[:10]:  # Max 10 levels
                    bids.append(OrderBookLevel(
                        price=level.price,
                        size=level.size,
                        market_maker=level.marketMaker
                    ))

            if ticker.domAsks:
                for level in ticker.domAsks[:10]:  # Max 10 levels
                    asks.append(OrderBookLevel(
                        price=level.price,
                        size=level.size,
                        market_maker=level.marketMaker
                    ))

            # Create order book object
            order_book = OrderBook(
                symbol=symbol,
                timestamp=int(time.time() * 1000),
                bids=bids,
                asks=asks
            )

            # Store locally
            self.order_books[symbol] = order_book

            # Cache it
            self.cache.set_order_book(symbol, order_book.dict())

            # Trigger callbacks
            self._trigger_callbacks(f"orderbook_{symbol}", order_book)

        except Exception as e:
            logger.error(f"Error processing order book for {symbol}: {e}")

    async def subscribe_trades(self, symbol: str) -> bool:
        """Subscribe to trade tape"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')

            # Request tick data with extended hours support
            self.ib.reqMktData(
                contract,
                genericTickList='233',  # Add RTVolume for trade tape
                snapshot=False,
                regulatorySnapshot=False
            )
            
            # Enable extended hours market data
            self.ib.reqMarketDataType(4)  # 4 = Delayed frozen (works in extended hours)

            ticker = self.ib.ticker(contract)
            if ticker and hasattr(ticker, 'updateEvent'):
                # Use proper callback binding for ticker updates
                def on_trade_update(ticker_obj):
                    self._process_trade(symbol, ticker_obj)
                ticker.updateEvent += on_trade_update  # type: ignore

            self.ticker_subs[symbol] = ticker
            logger.info(f"Subscribed to trades for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to trades for {symbol}: {e}")
            return False

    def _process_trade(self, symbol: str, ticker):
        """Process trade update"""
        try:
            if ticker.last and ticker.lastSize:
                trade = TradeModel(
                    symbol=symbol,
                    timestamp=int(time.time() * 1000),
                    price=ticker.last,
                    size=ticker.lastSize
                )

                # Store recent trades
                self.recent_trades[symbol].append(trade)
                if len(self.recent_trades[symbol]) > 1000:
                    self.recent_trades[symbol] = self.recent_trades[symbol][-1000:]

                # Cache trade
                self.cache.append_trade(symbol, trade.dict())

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
        """Process bar update"""
        try:
            if bars:
                latest_bar = bars[-1]

                # Handle datetime properly
                import datetime as dt
                if hasattr(latest_bar.date, 'timestamp'):
                    timestamp = int(latest_bar.date.timestamp() * 1000)  # type: ignore
                elif isinstance(latest_bar.date, dt.date) and not isinstance(latest_bar.date, dt.datetime):
                    # It's a date object, convert to datetime
                    dt_obj = dt.datetime.combine(latest_bar.date, dt.time.min)
                    timestamp = int(dt_obj.timestamp() * 1000)
                elif isinstance(latest_bar.date, dt.datetime):
                    timestamp = int(latest_bar.date.timestamp() * 1000)
                else:
                    timestamp = int(time.time() * 1000)

                bar = BarModel(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=latest_bar.open,
                    high=latest_bar.high,
                    low=latest_bar.low,
                    close=latest_bar.close,
                    volume=int(latest_bar.volume),  # Ensure int type
                    vwap=getattr(latest_bar, 'wap', None),  # Safe attribute access
                    bar_count=getattr(latest_bar, 'count', None)  # Safe attribute access
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
        """Unsubscribe from all market data"""
        for contract in self.market_depth_subs.values():
            self.ib.cancelMktDepth(contract)

        for ticker in self.ticker_subs.values():
            self.ib.cancelMktData(ticker.contract)

        for bars in self.bar_subs.values():
            self.ib.cancelRealTimeBars(bars)

        self.market_depth_subs.clear()
        self.ticker_subs.clear()
        self.bar_subs.clear()

        logger.info("Unsubscribed from all market data")

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
        """Unsubscribe from market depth for all exchanges"""
        try:
            if symbol in self.market_depth_subs:
                sub_info = self.market_depth_subs[symbol]
                if isinstance(sub_info, dict) and 'contracts' in sub_info:
                    # New format with multiple exchanges
                    for contract in sub_info['contracts']:
                        self.ib.cancelMktDepth(contract)
                else:
                    # Legacy single contract format
                    self.ib.cancelMktDepth(sub_info)  # type: ignore
                del self.market_depth_subs[symbol]
                logger.info(f"Unsubscribed from Level 2 for {symbol}")
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
                self.ib.cancelMktData(contract)
                return float(ticker.last)
            elif ticker.close and ticker.close > 0:
                self.ib.cancelMktData(contract)
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
                self.ib.cancelMktData(contract)
                return float(bars[-1].close)
            
            # Method 4: Use bid/ask midpoint
            if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                midpoint = (ticker.bid + ticker.ask) / 2
                self.ib.cancelMktData(contract)
                return float(midpoint)
            
            self.ib.cancelMktData(contract)
            logger.error(f"Cannot get spot price for {symbol} - all methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get spot price for {symbol}: {e}")
            return None
