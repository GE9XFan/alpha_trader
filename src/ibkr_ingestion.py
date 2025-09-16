#!/usr/bin/env python3
"""
IBKR Ingestion - WebSocket Connection and Main Ingestion Class
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- ibkr:connected: Connection status
- market:{symbol}:book: Order book data
- market:{symbol}:ticker: Market ticker data
- market:{symbol}:trades: Trade data
- market:{symbol}:bars:1min: Bar data
- monitoring:ibkr:metrics: Performance metrics
- heartbeat:ibkr_ingestion: Heartbeat status
"""

import asyncio
import json
import time
import redis.asyncio as aioredis
import math
import random
import pytz
import holidays
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util, Ticker
import logging
import redis_keys as rkeys
from ibkr_processor import IBKRDataProcessor
import traceback
from decimal import Decimal, ROUND_HALF_UP

class IBKRIngestion:
    """
    IBKR WebSocket data ingestion with exchange-specific Level 2 depth.
    Uses async Redis throughout with proper sync wrappers for event handlers.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize IBKR ingestion with async Redis."""
        self.config = config
        self.redis = redis_conn  # This is async Redis (redis.asyncio.Redis)
        self.ib = IB()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.ibkr_config = config.get('ibkr', {})
        self.host = self.ibkr_config.get('host', '127.0.0.1')
        self.port = self.ibkr_config.get('port', 7497)
        self.client_id = self.ibkr_config.get('client_id', 1)
        self.timeout = self.ibkr_config.get('timeout', 10)
        
        # Build symbol lists from config
        self.level2_symbols = config.get('symbols', {}).get('level2', ['SPY', 'QQQ', 'IWM'])
        self.standard_symbols = config.get('symbols', {}).get('standard', [])
        self.all_symbols = self.level2_symbols + self.standard_symbols
        
        # Thread safety - use async lock
        self.lock = asyncio.Lock()
        
        # Store depth tickers to prevent garbage collection
        self.depth_tickers = {}

        # L2 fallback tracking
        self.l2_fallback_exchanges = set()

        # Processing helpers encapsulate normalization/stateful buffers
        self.processor = IBKRDataProcessor(config)

        # Background task tracking for clean shutdowns
        self._background_tasks: List[asyncio.Task] = []
        
        # TTL configuration
        self.ttls = config['modules']['data_ingestion']['store_ttls']
        
        # Market hours
        self.market_tz = pytz.timezone('US/Eastern')
        current_year = datetime.now().year
        # Use country_holidays for type compatibility
        self.us_holidays = holidays.country_holidays('US', years=range(current_year, current_year + 2))
        
        # Staleness configuration
        self.check_staleness_after_hours = config.get('market', {}).get('check_staleness_after_hours', False)
        self.staleness_threshold = config['modules']['data_ingestion'].get('data_freshness_threshold', 5)
        
        # Connection state
        self.connected = False
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.ibkr_config.get('max_reconnect_attempts', 10)
        self.reconnect_delay_base = self.ibkr_config.get('reconnect_delay_base', 2)
        
        # Performance tracking
        self.last_update_times = {}
        self.last_tob_ts = {}  # Track last TOB update timestamp to prevent stale overwrites
        self.update_counts = {'depth': 0, 'trades': 0, 'bars': 0}
        self.subscriptions = {}  # Track all subscriptions
        
        self.logger.info(f"IBKRIngestion initialized - L2: {self.level2_symbols}, Standard: {self.standard_symbols}")
    
    def _make_depth_handler(self, exchange: str) -> Callable[[Ticker], None]:
        """Create a strongly typed depth update handler."""
        def handler(ticker: Ticker) -> None:
            asyncio.create_task(self._on_depth_update_async(ticker, exchange))
        return handler
    
    def _make_tob_handler(self) -> Callable[[Ticker], None]:
        """Create a strongly typed TOB update handler."""
        def handler(ticker: Ticker) -> None:
            asyncio.create_task(self._on_tob_update_async(ticker))
        return handler
    
    async def start(self):
        """Start IBKR data ingestion."""
        self.logger.info("Starting IBKR data ingestion...")
        self.running = True
        
        try:
            # Connect to IBKR Gateway/TWS
            await self._connect_with_retry()
            
            if not self.connected:
                self.logger.error("Failed to establish IBKR connection")
                return
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Subscribe to market data
            await self._setup_subscriptions()
            
            # Start metrics loop
            metrics_task = asyncio.create_task(self._metrics_loop())

            # Start freshness check loop
            freshness_task = asyncio.create_task(self._freshness_check_loop())

            # Start status logger loop
            status_task = asyncio.create_task(self._status_logger_loop())

            self._background_tasks = [metrics_task, freshness_task, status_task]

            # Keep running
            while self.running and self.connected:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error in IBKR ingestion: {e}")
            traceback.print_exc()
        finally:
            await self._cleanup()
    
    async def _connect_with_retry(self):
        """Connect to IBKR with exponential backoff retry."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                await self.ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=self.timeout
                )
                
                if self.ib.isConnected():
                    self.connected = True
                    self.reconnect_attempts = 0
                    
                    # Update Redis connection status
                    await self.redis.set('ibkr:connected', '1')
                    
                    # Log account info
                    accounts = self.ib.managedAccounts()
                    if accounts:
                        self.logger.info(f"Connected to IBKR Gateway - Account: {accounts[0]}")
                    else:
                        self.logger.info("Connected to IBKR Gateway")
                    
                    return True
                    
            except Exception as e:
                self.logger.warning(f"IBKR connection attempt {self.reconnect_attempts + 1} failed: {e}")
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    delay = self.reconnect_delay_base ** self.reconnect_attempts
                    self.logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        self.logger.error("Max reconnection attempts reached")
        return False
    
    def _setup_event_handlers(self):
        """Set up all IBKR event handlers with sync wrappers for async handlers."""
        # Ticker handlers with sync wrapper
        self.ib.pendingTickersEvent += lambda tickers: asyncio.create_task(
            self._on_ticker_update_async(tickers)
        )
        
        # Bar handlers with sync wrapper
        self.ib.barUpdateEvent += lambda bars, hasNewBar: asyncio.create_task(
            self._on_bar_update_async(bars, hasNewBar)
        )
        
        # Error handlers (can stay sync)
        self.ib.errorEvent += self._on_error
        
        # Connection handlers (can stay sync)
        self.ib.disconnectedEvent += self._on_disconnect
        
        self.logger.info("Event handlers configured")
    
    async def _setup_subscriptions(self):
        """Set up all market data subscriptions."""
        tasks = []
        
        # Subscribe Level 2 symbols
        for symbol in self.level2_symbols:
            tasks.append(self._subscribe_level2_symbol(symbol))
        
        # Subscribe standard symbols
        for symbol in self.standard_symbols:
            tasks.append(self._subscribe_standard_symbol(symbol))
        
        # Execute all subscriptions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any subscription failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                symbol = self.all_symbols[i] if i < len(self.all_symbols) else "Unknown"
                self.logger.error(f"Failed to subscribe {symbol}: {result}")
    
    async def _subscribe_level2_symbol(self, symbol: str):
        """Subscribe to SMART Level 2 market depth to get venue codes in marketMaker field."""
        # Track how many active L2 depth subscriptions we have (max 3 allowed by IBKR)
        active_depth_count = len([k for k in self.depth_tickers.keys() 
                                 if k not in self.l2_fallback_exchanges])
        
        if active_depth_count >= 3:
            self.logger.warning(f"Already at max L2 depth limit (3), using TOB for {symbol}")
            # Just subscribe to standard market data
            await self._subscribe_standard_symbol(symbol)
            return
        
        # Use SMART aggregation to get venue codes (NSDQ, ARCA, EDGX, etc.)
        exchange = 'SMART'
        try:
            # Create SMART contract for aggregated depth with venue codes
            contract = Stock(symbol, 'SMART', 'USD')
            contracts = await self.ib.qualifyContractsAsync(contract)
            if contracts:
                contract = contracts[0]
                
            # Set market data type to live
            self.ib.reqMarketDataType(1)  # 1=Live
            
            # Request SMART L2 depth
            depth_ticker = await self._request_exchange_depth(symbol, exchange, contract)
            
            if depth_ticker:
                # CRITICAL: Store reference to prevent GC
                key = f"{symbol}:SMART"
                self.depth_tickers[key] = depth_ticker
                
                # Also request market data with RTVolume for trades
                trade_ticker = self.ib.reqMktData(
                    contract,
                    genericTickList='233',
                    snapshot=False,
                    regulatorySnapshot=False
                )
                
                # CRITICAL: Also request 5-second bars for L2 symbols
                # Analytics needs bar data from ALL symbols including L2
                bars = self.ib.reqRealTimeBars(
                    contract,
                    barSize=5,
                    whatToShow='TRADES',
                    useRTH=False
                )
                
                # Track subscription with depth ticker, trade ticker and bars
                self.subscriptions[symbol] = {
                    'type': 'LEVEL2', 
                    'depth': depth_ticker,
                    'ticker': trade_ticker, 
                    'exchange': 'SMART',
                    'bars': bars  # Add bars reference
                }
                
                self.logger.info(f"L2 subscription successful: {symbol} on SMART (with venue codes)")
                return
                
        except Exception as e:
            self.logger.warning(f"L2 subscription failed for {symbol}: {e}")
            # Fall back to standard market data
            await self._subscribe_standard_symbol(symbol)
    
    async def _request_exchange_depth(self, symbol: str, exchange: str, contract):
        """Request depth with automatic fallback to TOB on failure."""
        try:
            # Get configured depth rows
            num_rows = self.config['ibkr'].get('l2_num_rows', 10)
            
            # Use SMART aggregation to get venue codes in marketMaker field
            is_smart = (exchange == 'SMART')
            depth_ticker = self.ib.reqMktDepth(
                contract,
                numRows=num_rows,
                isSmartDepth=is_smart  # True for SMART to get venue codes
            )
            
            return depth_ticker
            
        except Exception as e:
            error_code = getattr(e, 'errorCode', None)
            error_msg = str(e)
            
            if error_code == 309 or '309' in error_msg:
                # Max depth requests reached
                self.logger.warning(f"Max L2 depth limit reached for {symbol} on {exchange}")
                return None  # Signal to try next exchange or fall back to TOB
                
            elif error_code == 2152 or '2152' in error_msg:
                # No L2 permission - fallback to TOB
                self.logger.info(f"No L2 entitlement for {symbol} on {exchange}, using TOB only")
                self.l2_fallback_exchanges.add(f"{symbol}:{exchange}")
                
                # Just request regular market data (TOB)
                ticker = self.ib.reqMktData(
                    contract,
                    genericTickList='233',
                    snapshot=False,
                    regulatorySnapshot=False
                )
                
                # Note: TOB updates are handled by pendingTickersEvent globally
                # Bars will be requested by the caller when this ticker is returned
                
                return ticker
            
            # Other errors - let them propagate
            raise
    
    async def _subscribe_standard_symbol(self, symbol: str):
        """Subscribe to standard market data for non-L2 symbols."""
        try:
            # Use SMART routing for standard symbols
            contract = Stock(symbol, 'SMART', 'USD')
            contracts = await self.ib.qualifyContractsAsync(contract)
            if contracts:
                contract = contracts[0]
            
            # Request market data with RTVolume for trades
            ticker = self.ib.reqMktData(
                contract,
                genericTickList='233',
                snapshot=False,
                regulatorySnapshot=False
            )
            
            # Note: TOB updates are handled by pendingTickersEvent globally
            # No need for per-ticker handlers
            
            # Request 5-second bars
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow='TRADES',
                useRTH=False
            )
            
            # Track subscription
            self.subscriptions[symbol] = {'type': 'STANDARD', 'ticker': ticker, 'bars': bars}
            
            self.logger.info(f"Standard subscription successful: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe standard symbol {symbol}: {e}")
    
    async def _on_depth_update_async(self, ticker, exchange: str):
        """Async handler for Level 2 market depth updates per exchange."""
        start_time = time.perf_counter()

        try:
            if not ticker or not ticker.contract:
                return
            
            symbol = ticker.contract.symbol
            book_key = f"{symbol}:{exchange}"
            
            # Skip if this exchange fell back to TOB
            if book_key in self.l2_fallback_exchanges:
                return
                
            # Only process Level 2 symbols
            if symbol not in self.level2_symbols:
                return
            
            async with self.lock:
                book = self.processor.update_depth_book(symbol, exchange, ticker)
                if not book:
                    return

                venues_seen = {
                    level.get('venue')
                    for level in [*book.get('bids', []), *book.get('asks', [])]
                    if level.get('venue')
                }
                if venues_seen and venues_seen != {'UNKNOWN'} and random.random() < 0.01:
                    self.logger.debug(f"Venues in {symbol} book: {sorted(venues_seen)}")

                async with self.redis.pipeline(transaction=False) as pipe:
                    book_ttl = self.ttls['order_book']
                    await pipe.setex(rkeys.market_book_key(symbol, exchange), book_ttl, json.dumps(book))
                    await pipe.setex(rkeys.market_book_key(symbol), book_ttl, json.dumps(book))

                    aggregated = self.processor.compute_aggregated_tob(symbol)
                    if aggregated:
                        await pipe.setex(
                            rkeys.market_ticker_key(symbol),
                            self.ttls['market_data'],
                            json.dumps(aggregated)
                        )

                    await pipe.execute()

                # Update monitoring
                self.update_counts['depth'] += 1
                self.last_update_times[symbol] = time.time()

        except Exception as e:
            self.logger.error(f"Error processing depth for {symbol}:{exchange}: {e}")
    
    async def _on_tob_update_async(self, ticker):
        """Update :ticker from top-of-book for non-L2 symbols, or for L2 symbols when *no* L2 venue is active."""
        if not ticker or not ticker.contract:
            return
            
        symbol = ticker.contract.symbol
        
        # If this is an L2 symbol, only allow TOB writes when there is no working L2 venue
        if symbol in self.level2_symbols:
            # Working L2 exists if we have any depth ticker for this symbol that is NOT in the fallback set
            has_working_l2 = any(
                key.startswith(f"{symbol}:") and key not in self.l2_fallback_exchanges
                for key in self.depth_tickers.keys()
            )
            if has_working_l2:
                return  # L2 aggregation will maintain ticker data
        
        ticker_data = self.processor.build_quote_ticker(symbol, ticker)
        if not ticker_data:
            return

        async with self.redis.pipeline(transaction=False) as pipe:
            await pipe.setex(
                rkeys.market_ticker_key(symbol),
                self.ttls['market_data'],
                json.dumps(ticker_data)
            )
            await pipe.execute()

        self.last_update_times[symbol] = time.time()
    
    async def _on_ticker_update_async(self, tickers):
        """Process ticker updates for trades and depth."""
        for ticker in tickers:
            if not ticker.contract:
                continue

            symbol = ticker.contract.symbol

            # Check if this is a depth ticker
            for key, depth_ticker in list(self.depth_tickers.items()):
                if depth_ticker is ticker:
                    # This is a depth ticker update
                    exchange = key.split(':')[1]
                    await self._on_depth_update_async(ticker, exchange)
                    continue
            
            # Process trades - check for valid values (not NaN)
            if (ticker.last and not math.isnan(ticker.last) and 
                ticker.lastSize and not math.isnan(ticker.lastSize)):
                # Extract venue from ticker's exchange field if available
                venue = getattr(ticker, 'exchange', 'UNKNOWN')
                if not venue or venue == 'SMART':
                    # Try to get venue from lastExchange
                    venue = getattr(ticker, 'lastExchange', 'UNKNOWN')
                
                trade = {
                    'symbol': symbol,
                    'price': float(ticker.last),
                    'size': int(ticker.lastSize),
                    'time': int(datetime.now().timestamp() * 1000),
                    'venue': venue  # Add venue to trade data
                }
                
                # Update Redis
                await self._update_trade_data_redis(symbol, trade, ticker)
                
                # Update monitoring
                self.update_counts['trades'] += 1
                self.last_update_times[symbol] = time.time()
    
    async def _update_trade_data_redis(self, symbol: str, trade: Dict, ticker):
        """Update Redis with trade data using :ticker key consistently."""
        last_price_data, ticker_data = self.processor.prepare_trade_storage(symbol, trade, ticker)

        async with self.redis.pipeline(transaction=False) as pipe:
            ttl = self.ttls['market_data']
            await pipe.setex(rkeys.market_last_key(symbol), ttl, json.dumps(last_price_data))
            await pipe.setex(rkeys.market_ticker_key(symbol), ttl, json.dumps(ticker_data))

            # Store trades list - APPEND without deleting!
            trades_key = rkeys.market_trades_key(symbol)

            # Only push new trades from buffer
            if self.processor.trades_buffer[symbol]:
                # Push the latest trade
                pipe.rpush(trades_key, json.dumps(trade))
                # Keep only last 1000 trades
                pipe.ltrim(trades_key, -1000, -1)
                pipe.expire(trades_key, self.ttls.get('trades', 3600))

            await pipe.execute()
    
    async def _on_bar_update_async(self, bars, hasNewBar):
        """Process 5-second bar updates."""
        if not hasNewBar or not bars:
            return
            
        # bars is a single RealTimeBarList object, not a list
        # Find which symbol this bars object belongs to
        symbol = None
        for sym, sub in self.subscriptions.items():
            if 'bars' in sub and sub['bars'] == bars:
                symbol = sym
                break
                
        if not symbol:
            return  # Can't identify which symbol these bars belong to
            
        # Get the latest bar from the RealTimeBarList
        if len(bars) > 0:
            bar = bars[-1]  # Get latest bar
            
            bar_data = {
                'time': int(bar.time.timestamp() * 1000),
                'open': float(bar.open_),  # Note: ib_insync uses open_ with underscore
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'wap': float(bar.wap) if hasattr(bar, 'wap') else None,  # weighted average price
                'count': int(bar.count) if hasattr(bar, 'count') else None
            }
            
            # Add to buffer
            self.processor.record_bar(symbol, bar_data)
            
            # Update Redis - APPEND without deleting!
            async with self.redis.pipeline(transaction=False) as pipe:
                bars_key = rkeys.market_bars_key(symbol, '1min')
                
                # Push new bar without deleting existing ones
                pipe.rpush(bars_key, json.dumps(bar_data))
                # Keep only last 500 bars for analysis
                pipe.ltrim(bars_key, -500, -1)
                pipe.expire(bars_key, self.ttls['bars'])
                
                await pipe.execute()
            
            # Update monitoring
            self.update_counts['bars'] += 1
            self.last_update_times[symbol] = time.time()
    
    def is_market_hours(self, extended: bool = False) -> bool:
        """Check if currently in market hours."""
        now = datetime.now(self.market_tz)
        
        # Check weekend
        if now.weekday() >= 5:
            return False
        
        # Check holiday
        if now.date() in self.us_holidays:
            return False
        
        # Check time
        from datetime import time
        current_time = now.time()
        if extended:
            return time(4, 0) <= current_time <= time(20, 0)
        else:
            return time(9, 30) <= current_time <= time(16, 0)
    
    async def _check_data_freshness(self):
        """Check if data is fresh for all symbols - only during market hours."""
        # Skip staleness check outside market hours
        if not self.check_staleness_after_hours and not self.is_market_hours():
            # Clear staleness tracking outside market hours
            await self.redis.delete('monitoring:data:stale')
            return
        
        current_time = time.time()
        stale_threshold = self.staleness_threshold
        
        stale_symbols = {}
        
        for symbol in self.all_symbols:
            last_update = self.last_update_times.get(symbol, 0)
            if last_update == 0:
                continue  # Never updated yet
                
            age = current_time - last_update
            if age > stale_threshold:
                self.logger.warning(f"Data stale for {symbol}: {age:.1f}s (threshold: {stale_threshold}s)")
                stale_symbols[symbol] = age
        
        # Update Redis with stale symbols
        status_key = rkeys.get_system_key('status', module='ibkr_ingestion')
        ttl = self.ttls.get('monitoring', 60)

        async with self.redis.pipeline(transaction=False) as pipe:
            status_payload = {
                'state': 'OK',
                'ts': int(time.time() * 1000)
            }

            if stale_symbols:
                mapping_data = {k: str(v) for k, v in stale_symbols.items()}
                await pipe.hset('monitoring:data:stale', mapping=mapping_data)
                status_payload.update({
                    'state': 'STALE',
                    'symbols': ','.join(sorted(stale_symbols.keys()))
                })
            else:
                await pipe.delete('monitoring:data:stale')

            await pipe.setex(status_key, ttl, json.dumps(status_payload))
            await pipe.execute()
    
    async def _freshness_check_loop(self):
        """Loop to check data freshness."""
        while self.running:
            try:
                await self._check_data_freshness()
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Freshness check error: {e}")
                await asyncio.sleep(5)
    
    async def _status_logger_loop(self):
        """Periodically log IBKR data status - similar to Alpha Vantage logging."""
        await asyncio.sleep(5)  # Initial delay
        
        while self.running:
            try:
                # Collect current market data status
                level2_status = []
                standard_status = []
                
                for symbol, sub in self.subscriptions.items():
                    # Get current data from Redis (stored as JSON string, not hash)
                    ticker_json = await self.redis.get(rkeys.market_ticker_key(symbol))
                    
                    if ticker_json:
                        try:
                            ticker_data = json.loads(ticker_json)
                            # Handle None values properly - JSON might have null values
                            bid = float(ticker_data.get('bid') or 0)
                            ask = float(ticker_data.get('ask') or 0) 
                            last = float(ticker_data.get('last') or 0)
                            volume = int(ticker_data.get('volume') or 0)
                            spread = float(ticker_data.get('spread') or 0)
                        except (TypeError, ValueError) as e:
                            # Skip this symbol if data is malformed
                            continue
                        
                        if sub.get('type') == 'LEVEL2':
                            # Check order book depth
                            book_data = await self.redis.get(rkeys.market_book_key(symbol))
                            depth_count = 0
                            if book_data:
                                book = json.loads(book_data)
                                depth_count = len(book.get('bids', [])) + len(book.get('asks', []))
                            
                            level2_status.append(f"{symbol}: bid={bid:.2f}, ask={ask:.2f}, spread={spread:.4f}, depth={depth_count}")
                        else:
                            standard_status.append(f"{symbol}: last={last:.2f}, vol={volume:,}")
                
                # Log summary every 10 seconds
                if level2_status:
                    self.logger.info(f"✓ IBKR L2 Depth: {', '.join(level2_status[:3])}")
                
                if standard_status:
                    self.logger.info(f"✓ IBKR Quotes: {', '.join(standard_status[:3])}")
                
                # Check for recent trades/sweeps
                sweep_alerts = await self.redis.lrange('alerts:sweeps', 0, 0)
                if sweep_alerts:
                    latest_sweep = json.loads(sweep_alerts[0])
                    self.logger.info(f"⚡ Sweep detected: {latest_sweep['symbol']} {latest_sweep['size']} @ ${latest_sweep['price']}")
                
                # Log bar update counts
                bars_count = self.update_counts.get('bars', 0)
                if bars_count > 0:
                    self.logger.info(f"✓ IBKR Bars: {bars_count} updates in last 10s")
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Status logger error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_loop(self):
        """Publish performance metrics to Redis."""
        while self.running:
            try:
                # Calculate metrics
                now = time.time()
                ages = [now - ts for ts in self.last_update_times.values() if ts]
                max_age = max(ages) if ages else 0
                stale_symbols = [
                    symbol for symbol, ts in self.last_update_times.items()
                    if ts and (now - ts) > self.staleness_threshold
                ]

                metrics = {
                    'connected': '1' if self.connected else '0',
                    'symbols_subscribed': len(self.subscriptions),
                    'level2_symbols': len([s for s in self.subscriptions if self.subscriptions[s].get('type') == 'LEVEL2']),
                    'standard_symbols': len([s for s in self.subscriptions if self.subscriptions[s].get('type') == 'STANDARD']),
                    'depth_updates': self.update_counts.get('depth', 0),
                    'trade_updates': self.update_counts.get('trades', 0),
                    'bar_updates': self.update_counts.get('bars', 0),
                    'max_lag_s': round(max_age, 3),
                    'stale_count': len(stale_symbols),
                    'reconnect_attempts': self.reconnect_attempts,
                }

                if stale_symbols:
                    metrics['stale_symbols'] = ','.join(sorted(stale_symbols))

                # Store metrics
                metrics_key = 'monitoring:ibkr:metrics'
                async with self.redis.pipeline(transaction=False) as pipe:
                    await pipe.hset(metrics_key, mapping={k: str(v) for k, v in metrics.items()})
                    await pipe.expire(metrics_key, self.ttls.get('metrics', 60))

                    # Update heartbeat alongside metrics
                    heartbeat = {
                        'ts': int(datetime.now().timestamp() * 1000),
                        'connected': self.connected,
                        'symbols': len(self.all_symbols)
                    }

                    ttl = self.ttls.get('heartbeat', 15)
                    await pipe.setex(rkeys.heartbeat_key('ibkr_ingestion'), ttl, json.dumps(heartbeat))

                    await pipe.execute()

                # Reset counters
                self.update_counts = {'depth': 0, 'trades': 0, 'bars': 0}

                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Metrics error: {e}")
                await asyncio.sleep(10)
    
    def _on_error(self, reqId, errorCode, errorString, contract=None):
        """Handle IBKR errors."""
        if errorCode == 309:
            # Max market depth requests reached (limit is 3)
            self.logger.warning(f"IBKR error 309: Max number (3) of market depth requests has been reached")
            # This is handled in subscription logic - we check depth count before subscribing
        elif errorCode == 2152:
            # Market data farm connection broken
            self.logger.warning(f"Market data farm connection issue: {errorString}")
        elif errorCode == 504:
            # Not connected
            self.connected = False
            self.logger.error("Lost connection to IBKR Gateway")
        elif errorCode in [2104, 2106, 2158]:
            # Info messages - data farm connections
            self.logger.debug(f"IBKR info: {errorString}")
        else:
            self.logger.warning(f"IBKR error {errorCode}: {errorString}")
    
    def _on_disconnect(self):
        """Handle disconnection."""
        self.connected = False
        
        # Only log and reconnect if we're still running (not shutting down)
        if self.running:
            self.logger.warning("Disconnected from IBKR Gateway")
            asyncio.create_task(self._reconnect())
        else:
            # Intentional disconnection during shutdown
            self.logger.info("IBKR disconnected (shutdown)")
    
    async def _reconnect(self):
        """Attempt to reconnect after disconnection."""
        if not self.running:
            return

        self.logger.info("Attempting to reconnect to IBKR...")
        await self._connect_with_retry()

        if self.connected:
            # Re-establish subscriptions
            await self._setup_subscriptions()

    async def _cancel_background_tasks(self):
        """Stop any background loops started in :meth:`start`."""
        tasks, self._background_tasks = self._background_tasks, []
        for task in tasks:
            task.cancel()

        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                self.logger.debug("Background task %s exited with %s", getattr(task, 'get_name', lambda: task)(), exc)

    async def _cleanup(self):
        """Clean up resources on shutdown."""
        self.logger.info("Cleaning up IBKR connections...")

        try:
            await self._cancel_background_tasks()

            # Only try to cancel if we're connected
            if self.ib.isConnected():
                # Cancel all depth subscriptions safely
                for key, depth_ticker in list(self.depth_tickers.items()):
                    try:
                        # Check if this is actually a depth ticker (not fallen back to TOB)
                        if key not in self.l2_fallback_exchanges:
                            self.ib.cancelMktDepth(depth_ticker)
                        else:
                            # This is a TOB ticker, not depth
                            self.ib.cancelMktData(depth_ticker)
                    except Exception as e:
                        # Silently ignore - subscription may already be cancelled
                        self.logger.debug(f"Depth/TOB cancellation skipped for {key}: {e}")
                
                # Cancel other subscriptions
                for symbol, sub in list(self.subscriptions.items()):
                    try:
                        if 'ticker' in sub and sub['ticker']:
                            # Skip if already cancelled above
                            if sub.get('type') != 'LEVEL2':
                                self.ib.cancelMktData(sub['ticker'])
                    except Exception:
                        pass  # Silently ignore
                    
                    try:
                        if 'bars' in sub and sub['bars']:
                            self.ib.cancelRealTimeBars(sub['bars'])
                    except Exception:
                        pass  # Silently ignore
                
                # Disconnect from IBKR
                self.ib.disconnect()

            # Clear all data structures
            self.depth_tickers.clear()
            self.subscriptions.clear()
            self.processor.reset()

            # Update Redis
            await self.redis.set('ibkr:connected', '0')
            
            self.logger.info("IBKR cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def stop(self):
        """Stop IBKR ingestion."""
        self.running = False
        await self._cleanup()