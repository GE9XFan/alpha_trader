#!/usr/bin/env python3
"""
Data Ingestion Module - IBKR and Alpha Vantage Data Collection
Day 4 Implementation with L2 exchange-specific subscriptions and async Redis
"""

import asyncio
import json
import time
import redis.asyncio as aioredis
import aiohttp
import math
import random
import pytz
import holidays
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util, Ticker
import logging
import traceback
from decimal import Decimal, ROUND_HALF_UP

# Venue normalization mapping for IBKR marketMaker field
VENUE_MAP = {
    "NSDQ": "NSDQ", "NASDAQ": "NSDQ",  # Normalize NASDAQ variants to NSDQ
    "BATS": "BATS", "EDGX": "EDGX", "EDGEA": "EDGEA", 
    "DRCTEDGE": "DRCTEDGE", "BYX": "BYX",
    "ARCA": "ARCA", "NYSE": "NYSE", "AMEX": "AMEX", 
    "NYSENAT": "NYSENAT", "PSX": "PSX", "IEX": "IEX", 
    "LTSE": "LTSE", "CHX": "CHX", "ISE": "ISE",
    "PEARL": "PEARL", "MEMX": "MEMX", "BEX": "BEX",
    "IBEOS": "IBEOS", "IBKRATS": "IBKRATS",
    "OVERNIGHT": "OVERNIGHT", "ISLAND": "NSDQ",  # ISLAND is NASDAQ
    "SMART": "SMART", "UNKNOWN": "UNKNOWN", "": "UNKNOWN"
}


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
        
        # Aggregated TOB across exchanges
        self.aggregated_tob = {}
        
        # L2 fallback tracking
        self.l2_fallback_exchanges = set()
        
        # Order books per exchange
        self.order_books = {}
        
        # Trade and bar buffers - increased sizes for proper accumulation
        self.trades_buffer = defaultdict(lambda: deque(maxlen=5000))
        self.bars_buffer = defaultdict(lambda: deque(maxlen=500))
        self.last_trade = {}  # Cache last trade prices
        
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
            
            # Initialize per-exchange order book
            self.order_books[f"{symbol}:{exchange}"] = {
                'bids': [],
                'asks': [],
                'timestamp': 0,
                'exchange': exchange
            }
            
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
                # Initialize book if needed
                if book_key not in self.order_books:
                    self.order_books[book_key] = {
                        'bids': [], 'asks': [], 
                        'timestamp': 0, 'exchange': exchange
                    }
                
                book = self.order_books[book_key]
                
                # Update order book from ticker's DOM data with venue normalization
                venues_seen = set()
                
                # Extract and normalize venue codes from SMART depth
                book['bids'] = []
                for level in (ticker.domBids or []):
                    # When using SMART depth, marketMaker contains venue code
                    raw_venue = level.marketMaker if hasattr(level, 'marketMaker') else ''
                    venue = VENUE_MAP.get(raw_venue, 'UNKNOWN') if raw_venue else 'UNKNOWN'
                    venues_seen.add(venue)
                    
                    book['bids'].append({
                        'price': float(level.price),
                        'size': int(level.size),
                        'venue': venue,  # Store normalized venue code
                        'mm': venue  # Keep for backward compatibility
                    })
                
                book['asks'] = []
                for level in (ticker.domAsks or []):
                    raw_venue = level.marketMaker if hasattr(level, 'marketMaker') else ''
                    venue = VENUE_MAP.get(raw_venue, 'UNKNOWN') if raw_venue else 'UNKNOWN'
                    venues_seen.add(venue)
                    
                    book['asks'].append({
                        'price': float(level.price),
                        'size': int(level.size),
                        'venue': venue,
                        'mm': venue
                    })
                
                # Log unique venues seen (only occasionally to avoid spam)
                if venues_seen and venues_seen != {'UNKNOWN'} and random.random() < 0.01:  # 1% sample
                    self.logger.debug(f"Venues in {symbol} book: {sorted(venues_seen)}")
                
                book['timestamp'] = int(datetime.now().timestamp() * 1000)
                
                # Update Redis with async pipeline
                async with self.redis.pipeline(transaction=False) as pipe:
                    # Store exchange-specific book
                    book_ttl = self.ttls['order_book']
                    await pipe.setex(f'market:{symbol}:{exchange}:book', book_ttl, json.dumps(book))
                    
                    # CRITICAL: Also store at the aggregated location for analytics
                    # Analytics and status logger expect market:{symbol}:book
                    await pipe.setex(f'market:{symbol}:book', book_ttl, json.dumps(book))
                    
                    # CRITICAL: Always update aggregated TOB and :ticker
                    await self._update_aggregated_tob(symbol, book, pipe)
                    
                    await pipe.execute()
                
                # Update monitoring
                self.update_counts['depth'] += 1
                self.last_update_times[symbol] = time.time()
                
        except Exception as e:
            self.logger.error(f"Error processing depth for {symbol}:{exchange}: {e}")
    
    async def _update_aggregated_tob(self, symbol: str, exchange_book: dict, pipe):
        """
        Update aggregated best bid/ask across all exchanges.
        NOTE: pipe is passed in from caller's async with block.
        """
        if symbol not in self.aggregated_tob:
            self.aggregated_tob[symbol] = {
                'best_bid': 0,
                'best_ask': float('inf'),
                'bid_exchange': '',
                'ask_exchange': ''
            }
        
        agg = self.aggregated_tob[symbol]
        exchange = exchange_book.get('exchange', 'UNKNOWN')
        
        # Update if this exchange has better prices
        if exchange_book['bids']:
            best_bid = float(exchange_book['bids'][0]['price'])
            if best_bid > agg['best_bid']:
                agg['best_bid'] = best_bid
                agg['bid_exchange'] = exchange
        
        if exchange_book['asks']:
            best_ask = float(exchange_book['asks'][0]['price'])
            if best_ask < agg['best_ask']:
                agg['best_ask'] = best_ask
                agg['ask_exchange'] = exchange
        
        # Write aggregated TOB to Redis as :ticker
        if agg['best_bid'] > 0 and agg['best_ask'] < float('inf'):
            # Generate timestamp once
            current_ts = int(time.time() * 1000)
            
            # Staleness guard: only write if this update is newer
            last_ts = self.last_tob_ts.get(symbol, 0)
            if current_ts <= last_ts:
                return  # Skip stale update
            
            # Get cached last trade and ensure it's float
            last_price = self.last_trade.get(symbol)
            if last_price is not None:
                last_price = float(last_price)
            
            # Calculate derived values with guards
            bid = round(float(agg['best_bid']), 6)
            ask = round(float(agg['best_ask']), 6)
            mid = round((bid + ask) / 2, 6)
            spread = round(ask - bid, 6)
            
            # Calculate spread_bps only if bid > 0
            spread_bps = None
            if bid > 0:
                spread_bps = round((spread / bid) * 10000, 2)
            
            ticker_data = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': last_price,
                'mid': mid,
                'spread': spread,
                'spread_bps': spread_bps,
                'bid_exchange': agg['bid_exchange'],
                'ask_exchange': agg['ask_exchange'],
                'timestamp': current_ts
            }
            
            # Numeric sanity check: skip if any value is non-finite
            import math
            numeric_values = [v for v in [bid, ask, mid, spread, spread_bps, last_price] if v is not None]
            if any(not math.isfinite(v) for v in numeric_values):
                self.logger.warning(f"Non-finite value detected for {symbol}, skipping update")
                return
            
            # Atomic write with TTL from config
            ttl = self.ttls['market_data']
            await pipe.setex(f'market:{symbol}:ticker', ttl, json.dumps(ticker_data))
            
            # Update last timestamp after successful write
            self.last_tob_ts[symbol] = current_ts
    
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
                return  # L2 aggregation will maintain market:{symbol}:ticker
        
        # Update :ticker with TOB data
        if ticker.bid and ticker.ask:
            # Cache last trade if available
            if ticker.last:
                self.last_trade[symbol] = float(ticker.last)
            
            # Calculate values with proper types and rounding
            bid = round(float(ticker.bid), 6)
            ask = round(float(ticker.ask), 6)
            last_price = float(ticker.last) if ticker.last else self.last_trade.get(symbol)
            mid = round((bid + ask) / 2, 6)
            spread = round(ask - bid, 6)
            
            # Calculate spread_bps only if bid > 0
            spread_bps = None
            if bid > 0:
                spread_bps = round((spread / bid) * 10000, 2)
            
            ticker_data = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': last_price,
                'mid': mid,
                'spread': spread,
                'spread_bps': spread_bps,
                'timestamp': int(time.time() * 1000)
            }
            
            # Numeric sanity check
            import math
            numeric_values = [v for v in [bid, ask, mid, spread, spread_bps, last_price] if v is not None]
            if any(not math.isfinite(v) for v in numeric_values):
                self.logger.warning(f"Non-finite value detected for {symbol} in TOB update, skipping")
                return
            
            async with self.redis.pipeline(transaction=False) as pipe:
                await pipe.setex(
                    f'market:{symbol}:ticker', 
                    self.ttls['market_data'], 
                    json.dumps(ticker_data)
                )
                await pipe.execute()
            
            # Update monitoring
            self.last_update_times[symbol] = time.time()
    
    async def _on_ticker_update_async(self, tickers):
        """Process ticker updates for trades and depth."""
        for ticker in tickers:
            if not ticker.contract:
                continue
                
            symbol = ticker.contract.symbol
            
            # Check if this is a depth ticker
            for key, depth_ticker in self.depth_tickers.items():
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
                
                # Cache last trade
                self.last_trade[symbol] = float(ticker.last)
                
                # Add to buffer
                self.trades_buffer[symbol].append(trade)
                
                # Update Redis
                await self._update_trade_data_redis(symbol, trade, ticker)
                
                # Update monitoring
                self.update_counts['trades'] += 1
                self.last_update_times[symbol] = time.time()
    
    async def _update_trade_data_redis(self, symbol: str, trade: Dict, ticker):
        """Update Redis with trade data using :ticker key consistently."""
        async with self.redis.pipeline(transaction=False) as pipe:
            # Get TTLs from config
            ttls = self.ttls
            
            # Update last price as JSON with timestamp
            last_price_data = {
                'price': trade['price'],
                'ts': trade['time']  # Fixed: use 'time' key, not 'timestamp'
            }
            await pipe.set(f'market:{symbol}:last', json.dumps(last_price_data))
            
            # CRITICAL: Write to :ticker key for analytics to read
            # Round values properly - check for NaN
            last_price = round(float(trade['price']), 6)
            bid = None
            if ticker.bid and not math.isnan(ticker.bid):
                bid = round(float(ticker.bid), 6)
            ask = None
            if ticker.ask and not math.isnan(ticker.ask):
                ask = round(float(ticker.ask), 6)
            
            # Handle volume - check for NaN
            volume = 0
            if ticker.volume and not math.isnan(ticker.volume):
                volume = int(ticker.volume)
            
            # Handle vwap - check for NaN
            vwap = None
            if ticker.vwap and not math.isnan(ticker.vwap):
                vwap = round(float(ticker.vwap), 6)
            
            ticker_data = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': last_price,
                'volume': volume,
                'vwap': vwap,
                'timestamp': trade['time']
            }
            
            # Calculate derived fields if we have bid/ask
            if bid is not None and ask is not None:
                ticker_data['mid'] = round((bid + ask) / 2, 6)
                ticker_data['spread'] = round(ask - bid, 6)
                # Calculate spread_bps only if bid > 0
                if bid > 0:
                    ticker_data['spread_bps'] = round((ask - bid) / bid * 10000, 2)
                else:
                    ticker_data['spread_bps'] = None
            
            await pipe.setex(f'market:{symbol}:ticker', ttls['market_data'], json.dumps(ticker_data))
            
            # Store trades list - APPEND without deleting!
            trades_key = f'market:{symbol}:trades'
            
            # Only push new trades from buffer
            if self.trades_buffer[symbol]:
                # Push the latest trade
                pipe.rpush(trades_key, json.dumps(trade))
                # Keep only last 1000 trades
                pipe.ltrim(trades_key, -1000, -1)
                pipe.expire(trades_key, ttls.get('trades', 3600))
            
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
            self.bars_buffer[symbol].append(bar_data)
            
            # Update Redis - APPEND without deleting!
            async with self.redis.pipeline(transaction=False) as pipe:
                bars_key = f'market:{symbol}:bars'
                
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
        if stale_symbols:
            # Store stale symbol info
            mapping_data = {k: str(v) for k, v in stale_symbols.items()}
            await self.redis.hset('monitoring:data:stale', mapping=mapping_data)
        else:
            await self.redis.delete('monitoring:data:stale')
    
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
                    ticker_json = await self.redis.get(f'market:{symbol}:ticker')
                    
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
                            book_data = await self.redis.get(f'market:{symbol}:book')
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
                metrics = {
                    'connected': '1' if self.connected else '0',
                    'symbols_subscribed': len(self.subscriptions),
                    'level2_symbols': len([s for s in self.subscriptions if self.subscriptions[s].get('type') == 'LEVEL2']),
                    'standard_symbols': len([s for s in self.subscriptions if self.subscriptions[s].get('type') == 'STANDARD']),
                    'depth_updates': self.update_counts.get('depth', 0),
                    'trade_updates': self.update_counts.get('trades', 0),
                    'bar_updates': self.update_counts.get('bars', 0)
                }
                
                # Store metrics
                # Explicitly store metrics to Redis
                await self.redis.hset('monitoring:ibkr:metrics', mapping=metrics)
                
                # Update heartbeat
                heartbeat = {
                    'ts': int(datetime.now().timestamp() * 1000),
                    'connected': self.connected,
                    'symbols': len(self.all_symbols)
                }
                
                ttl = self.ttls.get('heartbeat', 15)
                await self.redis.setex('hb:ibkr', ttl, json.dumps(heartbeat))
                
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
    
    async def _cleanup(self):
        """Clean up resources on shutdown."""
        self.logger.info("Cleaning up IBKR connections...")
        
        try:
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
            self.order_books.clear()
            self.trades_buffer.clear()
            self.aggregated_tob.clear()
            self.last_tob_ts.clear()
            self.last_trade.clear()
            
            # Update Redis
            await self.redis.set('ibkr:connected', '0')
            
            self.logger.info("IBKR cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def stop(self):
        """Stop IBKR ingestion."""
        self.running = False
        await self._cleanup()


class AlphaVantageIngestion:
    """
    Alpha Vantage REST API ingestion with semaphore-based rate limiting.
    
    Rate Limiting Approach:
    - Uses asyncio.Semaphore for concurrency control (not true rate limiting)
    - Sized to (calls_per_minute - safety_buffer) concurrent requests
    - Works well with small symbol sets and staggered update intervals
    - For larger scale: consider implementing a true rolling-window rate limiter
    
    Current implementation is sufficient because:
    1. Small symbol set (12 symbols)
    2. Staggered update intervals prevent bursts
    3. Retry logic with exponential backoff handles 429s
    4. Soft limit detection ("Note"/"Information") triggers backoff
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize with async Redis and unified symbol list."""
        self.config = config
        self.redis = redis_conn  # Async Redis
        self.logger = logging.getLogger(__name__)
        
        # Alpha Vantage configuration
        self.av_config = config.get('alpha_vantage', {})
        self.api_key = self.av_config.get('api_key')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        self.base_url = self.av_config.get('base_url', 'https://www.alphavantage.co/query')
        self.calls_per_minute = self.av_config.get('calls_per_minute', 600)
        self.safety_buffer = self.av_config.get('safety_buffer', 10)
        
        # Single rate limiting via semaphore
        max_concurrent = self.calls_per_minute - self.safety_buffer
        self.request_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Retry configuration
        self.retry_attempts = self.av_config.get('retry_attempts', 3)
        self.retry_delay_base = self.av_config.get('retry_delay', 2)
        
        # Build complete symbol list
        level2_symbols = config.get('symbols', {}).get('level2', [])
        standard_symbols = config.get('symbols', {}).get('standard', [])
        self.symbols = level2_symbols + standard_symbols
        
        # TTLs from data_ingestion config
        self.ttls = config['modules']['data_ingestion']['store_ttls']
        
        # Update intervals
        self.options_interval = self.av_config.get('options_update_interval', 10)
        self.sentiment_interval = self.av_config.get('sentiment_update_interval', 300)
        self.technicals_interval = self.av_config.get('technicals_update_interval', 60)
        
        # Initialize staggered update times
        now = time.time()
        self.last_options_update = {}
        self.last_sentiment_update = {}
        self.last_technicals_update = {}
        
        for i, symbol in enumerate(self.symbols):
            if symbol in level2_symbols:
                offset = level2_symbols.index(symbol) * 0.5
            else:
                offset = 2.0 + ((i - len(level2_symbols)) * 1.0)
            
            self.last_options_update[symbol] = now - self.options_interval + offset
            self.last_sentiment_update[symbol] = now - self.sentiment_interval + (offset * 5)
            self.last_technicals_update[symbol] = now - self.technicals_interval + (offset * 2)
        
        self.running = False
        self.logger.info(f"AlphaVantageIngestion initialized for {len(self.symbols)} symbols")
    
    async def start(self):
        """Start Alpha Vantage data ingestion."""
        # Log startup with API key status (masked)
        api_key_status = "configured" if self.api_key and not self.api_key.startswith("${") else "NOT SET"
        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 8 else "INVALID"
        
        self.logger.info(f"Starting Alpha Vantage data ingestion...")
        self.logger.info(f"  API Key: {api_key_status} ({masked_key})")
        self.logger.info(f"  Symbols: {self.symbols}")
        self.logger.info(f"  Update intervals: options={self.options_interval}s, sentiment={self.sentiment_interval}s, technicals={self.technicals_interval}s")
        
        self.running = True
        
        try:
            # Start update loop
            update_task = asyncio.create_task(self._update_loop())
            
            # Start metrics loop
            metrics_task = asyncio.create_task(self._metrics_loop())
            
            # Wait for tasks
            await asyncio.gather(update_task, metrics_task)
            
        except Exception as e:
            self.logger.error(f"Error in Alpha Vantage ingestion: {e}")
            traceback.print_exc()
    
    async def _update_loop(self):
        """Main update loop for fetching data."""
        # Configure timeout and connection limits for robustness
        timeout = aiohttp.ClientTimeout(total=10, connect=3, sock_read=7)
        connector = aiohttp.TCPConnector(limit=200, ttl_dns_cache=300)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            while self.running:
                try:
                    current_time = time.time()
                    tasks = []
                    
                    for symbol in self.symbols:
                        # Add jitter to prevent all symbols from aligning at minute boundaries
                        options_jitter = random.uniform(-0.25, 0.25) * self.options_interval
                        sentiment_jitter = random.uniform(-0.25, 0.25) * self.sentiment_interval
                        technicals_jitter = random.uniform(-0.25, 0.25) * self.technicals_interval
                        
                        # Check if options need update (with jitter)
                        if current_time - self.last_options_update.get(symbol, 0) >= (self.options_interval + options_jitter):
                            tasks.append(self._fetch_with_retry(
                                self.fetch_options_chain, session, symbol
                            ))
                            self.last_options_update[symbol] = current_time
                        
                        # Check if sentiment needs update (with jitter)
                        if current_time - self.last_sentiment_update.get(symbol, 0) >= (self.sentiment_interval + sentiment_jitter):
                            tasks.append(self._fetch_with_retry(
                                self.fetch_sentiment, session, symbol
                            ))
                            self.last_sentiment_update[symbol] = current_time
                        
                        # Check if technicals need update (with jitter)
                        if current_time - self.last_technicals_update.get(symbol, 0) >= (self.technicals_interval + technicals_jitter):
                            tasks.append(self._fetch_with_retry(
                                self.fetch_technicals, session, symbol
                            ))
                            self.last_technicals_update[symbol] = current_time
                    
                    if tasks:
                        self.logger.info(f"Fetching {len(tasks)} updates from Alpha Vantage...")
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Log errors
                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.error(f"Fetch error: {result}")
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Update loop error: {e}")
                    await asyncio.sleep(1)
    
    async def _fetch_with_retry(self, func, *args, **kwargs):
        """Retry wrapper with exponential backoff and jitter for robust network handling."""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)
                
            except aiohttp.ClientResponseError as e:
                last_error = e
                if e.status == 429:
                    wait_time = 60
                    self.logger.warning(f"Rate limited (429), waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                elif attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    # Add ±20% jitter
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.debug(f"HTTP {e.status}, retry in {actual_delay:.1f}s")
                    await asyncio.sleep(actual_delay)
                    
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.debug(f"Timeout, retry in {actual_delay:.1f}s")
                    await asyncio.sleep(actual_delay)
                    
            except (aiohttp.ClientError, aiohttp.ClientConnectionError, 
                    aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError) as e:
                # Handle specific network errors (DNS, connection reset, TLS issues)
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.warning(f"Network error ({type(e).__name__}), retry {attempt+1}/{self.retry_attempts} in {actual_delay:.1f}s")
                    await asyncio.sleep(actual_delay)
                    
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.warning(f"Unexpected error ({type(e).__name__}), retry in {actual_delay:.1f}s")
                    await asyncio.sleep(actual_delay)
        
        raise last_error or Exception(f"All {self.retry_attempts} retry attempts failed")
    
    async def fetch_options_chain(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch options chain with semaphore rate limiting."""
        try:
            # Single rate control via semaphore
            async with self.request_semaphore:
                params = {
                    'function': 'REALTIME_OPTIONS',
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'datatype': 'json',
                    'require_greeks': 'true'
                }
                
                async with session.get(self.base_url, params=params) as response:
                    # CRITICAL: Raise for HTTP errors
                    response.raise_for_status()
                    
                    data = await response.json()
                    
                    # Check for API errors FIRST
                    if 'Error Message' in data:
                        self.logger.error(f"AV error for {symbol}: {data['Error Message']}")
                        return None
                    
                    # Check for soft rate limits
                    if 'Note' in data:
                        self.logger.warning(f"AV soft limit: {data['Note']}")
                        await asyncio.sleep(30)
                        return None
                    
                    if 'Information' in data:
                        self.logger.warning(f"AV info: {data['Information']}")
                        await asyncio.sleep(60)
                        return None
                    
                    # Process options chain - Alpha Vantage uses 'data' not 'contracts'
                    if 'data' in data and isinstance(data['data'], list):
                        # Store to Redis
                        await self._store_options_data(symbol, data)
                        self.logger.info(f"✓ Fetched options for {symbol}: {len(data['data'])} contracts")
                    else:
                        self.logger.warning(f"AV options missing 'data' for {symbol}; keys={list(data.keys())[:6]}")
                        return None
                    
                    return data
                    
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self.logger.warning(f"Rate limited (429) for {symbol}")
                await asyncio.sleep(60)
            else:
                self.logger.error(f"HTTP {e.status} for {symbol}: {e}")
            return None
    
    async def fetch_sentiment(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch sentiment data."""
        # Skip sentiment for ETFs as Alpha Vantage doesn't support them
        if symbol in ['SPY', 'QQQ', 'IWM','VXX']:
            self.logger.info(f"Skipping sentiment for ETF {symbol} (not supported by Alpha Vantage)")
            return None
        
        try:
            async with self.request_semaphore:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.api_key,
                    'sort': 'LATEST',
                    'limit': 50
                }
                
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Check for API errors
                    if 'Error Message' in data:
                        self.logger.error(f"AV sentiment error for {symbol}: {data['Error Message']}")
                        return None
                    
                    # Check for rate limits
                    if 'Note' in data:
                        self.logger.warning(f"AV sentiment soft limit: {data['Note']}")
                        await asyncio.sleep(30)
                        return None
                    
                    if 'Information' in data:
                        self.logger.warning(f"AV sentiment info: {data['Information']}")
                        await asyncio.sleep(60)
                        return None
                    
                    if 'feed' in data:
                        await self._store_sentiment_data(symbol, data)
                        self.logger.info(f"✓ Fetched sentiment for {symbol}: {len(data.get('feed', []))} articles")
                    else:
                        self.logger.warning(f"AV sentiment missing 'feed' for {symbol}; keys={list(data.keys())[:6]}")
                        return None
                    
                    return data
                    
        except Exception as e:
            self.logger.error(f"Sentiment fetch error for {symbol}: {e}")
            return None
    
    async def fetch_technicals(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch all technical indicators: RSI, MACD, BBANDS, ATR."""
        try:
            # Define all indicators to fetch
            indicators = [
                {
                    'function': 'RSI',
                    'params': {
                        'function': 'RSI',
                        'symbol': symbol,
                        'interval': '5min',
                        'time_period': 14,
                        'series_type': 'close',
                        'apikey': self.api_key
                    },
                    'key': 'Technical Analysis: RSI',
                    'value_key': 'RSI'
                },
                {
                    'function': 'MACD',
                    'params': {
                        'function': 'MACD',
                        'symbol': symbol,
                        'interval': '5min',
                        'series_type': 'close',
                        'apikey': self.api_key
                    },
                    'key': 'Technical Analysis: MACD',
                    'value_keys': ['MACD', 'MACD_Signal', 'MACD_Hist']
                },
                {
                    'function': 'BBANDS',
                    'params': {
                        'function': 'BBANDS',
                        'symbol': symbol,
                        'interval': '5min',
                        'time_period': 20,
                        'series_type': 'close',
                        'nbdevup': 2,
                        'nbdevdn': 2,
                        'apikey': self.api_key
                    },
                    'key': 'Technical Analysis: BBANDS',
                    'value_keys': ['Real Upper Band', 'Real Middle Band', 'Real Lower Band']
                },
                {
                    'function': 'ATR',
                    'params': {
                        'function': 'ATR',
                        'symbol': symbol,
                        'interval': '5min',
                        'time_period': 14,
                        'apikey': self.api_key
                    },
                    'key': 'Technical Analysis: ATR',
                    'value_key': 'ATR'
                }
            ]
            
            # Fetch all indicators in parallel
            tasks = []
            for indicator in indicators:
                tasks.append(self._fetch_single_indicator(session, symbol, indicator))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful fetches
            success_count = sum(1 for r in results if r and not isinstance(r, Exception))
            
            if success_count > 0:
                self.logger.info(f"✓ Fetched {success_count}/{len(indicators)} technicals for {symbol}")
                return {'success': success_count, 'total': len(indicators)}
            else:
                self.logger.warning(f"Failed to fetch any technicals for {symbol}")
                return None
                    
        except Exception as e:
            self.logger.error(f"Technicals fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_single_indicator(self, session: aiohttp.ClientSession, symbol: str, indicator: Dict):
        """Fetch a single technical indicator."""
        try:
            async with self.request_semaphore:
                async with session.get(self.base_url, params=indicator['params']) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Check for API errors
                    if 'Error Message' in data:
                        self.logger.error(f"AV {indicator['function']} error for {symbol}: {data['Error Message']}")
                        return None
                    
                    # Check for rate limits
                    if 'Note' in data:
                        self.logger.warning(f"AV {indicator['function']} soft limit: {data['Note']}")
                        await asyncio.sleep(30)
                        return None
                    
                    if 'Information' in data:
                        self.logger.warning(f"AV {indicator['function']} info: {data['Information']}")
                        await asyncio.sleep(60)
                        return None
                    
                    # Process and store the indicator data
                    if indicator['key'] in data:
                        await self._store_technical_data(symbol, indicator['function'], data, indicator)
                        return data
                    else:
                        self.logger.warning(f"AV {indicator['function']} missing '{indicator['key']}' for {symbol}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error fetching {indicator['function']} for {symbol}: {e}")
            return None
    
    async def _store_options_data(self, symbol: str, data: Dict):
        """Store options chain data to Redis."""
        try:
            # Process and store options - Alpha Vantage uses 'data' array
            contracts = data.get('data', [])
            
            # Separate calls and puts - AV uses 'type' not 'option_type'
            calls = [c for c in contracts if c.get('type') == 'call']
            puts = [c for c in contracts if c.get('type') == 'put']
            
            # Store to Redis
            async with self.redis.pipeline(transaction=False) as pipe:
                if calls:
                    await pipe.setex(
                        f'options:{symbol}:calls',
                        self.ttls['options_chain'],
                        json.dumps(calls)
                    )
                
                if puts:
                    await pipe.setex(
                        f'options:{symbol}:puts',
                        self.ttls['options_chain'],
                        json.dumps(puts)
                    )
                
                # Store Greeks separately - AV has inline greeks, not nested
                for contract in contracts:
                    # Extract greeks from contract (they're inline in AV response)
                    greeks = {
                        'delta': contract.get('delta'),
                        'gamma': contract.get('gamma'),
                        'theta': contract.get('theta'),
                        'vega': contract.get('vega'),
                        'rho': contract.get('rho'),
                        'implied_volatility': contract.get('implied_volatility'),
                        'open_interest': contract.get('open_interest')
                    }
                    
                    # AV uses 'contractID' not 'contract_id'
                    contract_id = contract.get('contractID')
                    if contract_id and any(greeks.values()):
                        key = f"options:{symbol}:{contract_id}:greeks"
                        await pipe.setex(
                            key,
                            self.ttls['greeks'],
                            json.dumps(greeks)
                        )
                
                await pipe.execute()
                
                # Log success with sample contract details
                log_msg = f"Stored options for {symbol}: calls={len(calls)}, puts={len(puts)}"
                
                # Sample contract details commented out for cleaner logs
                # try:
                #     # Add sample call contract with Greeks
                #     if calls:
                #         sample_call = calls[0]  # Get first call as sample
                #         log_msg += f"\n  Sample CALL: {sample_call.get('contractID', 'N/A')}"
                #         log_msg += f" Strike=${float(sample_call.get('strike', 0) or 0):.2f}"
                #         log_msg += f" Exp={sample_call.get('expiration', 'N/A')}"
                #         log_msg += f" Bid=${float(sample_call.get('bid', 0) or 0):.2f}"
                #         log_msg += f" Ask=${float(sample_call.get('ask', 0) or 0):.2f}"
                #         log_msg += f" Vol={int(float(sample_call.get('volume', 0) or 0))}"
                #         log_msg += f" OI={int(float(sample_call.get('open_interest', 0) or 0))}"
                #         log_msg += f"\n    Greeks: Δ={float(sample_call.get('delta', 0) or 0):.4f}"
                #         log_msg += f" Γ={float(sample_call.get('gamma', 0) or 0):.4f}"
                #         log_msg += f" Θ={float(sample_call.get('theta', 0) or 0):.4f}"
                #         log_msg += f" Vega={float(sample_call.get('vega', 0) or 0):.4f}"
                #         log_msg += f" IV={float(sample_call.get('implied_volatility', 0) or 0):.2%}"
                #     
                #     # Add sample put contract with Greeks
                #     if puts:
                #         sample_put = puts[0]  # Get first put as sample
                #         log_msg += f"\n  Sample PUT: {sample_put.get('contractID', 'N/A')}"
                #         log_msg += f" Strike=${float(sample_put.get('strike', 0) or 0):.2f}"
                #         log_msg += f" Exp={sample_put.get('expiration', 'N/A')}"
                #         log_msg += f" Bid=${float(sample_put.get('bid', 0) or 0):.2f}"
                #         log_msg += f" Ask=${float(sample_put.get('ask', 0) or 0):.2f}"
                #         log_msg += f" Vol={int(float(sample_put.get('volume', 0) or 0))}"
                #         log_msg += f" OI={int(float(sample_put.get('open_interest', 0) or 0))}"
                #         log_msg += f"\n    Greeks: Δ={float(sample_put.get('delta', 0) or 0):.4f}"
                #         log_msg += f" Γ={float(sample_put.get('gamma', 0) or 0):.4f}"
                #         log_msg += f" Θ={float(sample_put.get('theta', 0) or 0):.4f}"
                #         log_msg += f" Vega={float(sample_put.get('vega', 0) or 0):.4f}"
                #         log_msg += f" IV={float(sample_put.get('implied_volatility', 0) or 0):.2%}"
                # except Exception as e:
                #     # If detailed logging fails, just log the basic message
                #     pass
                
                self.logger.info(log_msg)
                
        except Exception as e:
            self.logger.error(f"Error storing options for {symbol}: {e}")
    
    async def _store_sentiment_data(self, symbol: str, data: Dict):
        """Store complete sentiment data to Redis including all article details."""
        try:
            feed = data.get('feed', [])
            
            if feed:
                # Process and store complete feed data
                articles = []
                ticker_sentiments = []
                
                for article in feed:
                    # Store complete article data
                    article_data = {
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'time_published': article.get('time_published', ''),
                        'authors': article.get('authors', []),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'topics': article.get('topics', []),  # Includes topic and relevance_score
                        'overall_sentiment_score': article.get('overall_sentiment_score', 0),
                        'overall_sentiment_label': article.get('overall_sentiment_label', 'Neutral'),
                        'ticker_sentiment': []
                    }
                    
                    # Process ticker-specific sentiment for this article
                    if 'ticker_sentiment' in article:
                        for ticker_data in article['ticker_sentiment']:
                            if ticker_data.get('ticker') == symbol:
                                ticker_sent = {
                                    'ticker': ticker_data.get('ticker'),
                                    'relevance_score': float(ticker_data.get('relevance_score', 0)),
                                    'ticker_sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                                    'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label', 'Neutral')
                                }
                                article_data['ticker_sentiment'].append(ticker_sent)
                                ticker_sentiments.append(ticker_sent)
                    
                    articles.append(article_data)
                
                # Calculate aggregate metrics
                avg_sentiment = 0
                avg_relevance = 0
                sentiment_distribution = {'Bullish': 0, 'Somewhat-Bullish': 0, 'Neutral': 0, 'Somewhat-Bearish': 0, 'Bearish': 0}
                
                if ticker_sentiments:
                    sentiment_scores = [ts['ticker_sentiment_score'] for ts in ticker_sentiments]
                    relevance_scores = [ts['relevance_score'] for ts in ticker_sentiments]
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    avg_relevance = sum(relevance_scores) / len(relevance_scores)
                    
                    # Count sentiment distribution
                    for ts in ticker_sentiments:
                        label = ts.get('ticker_sentiment_label', 'Neutral')
                        if label in sentiment_distribution:
                            sentiment_distribution[label] += 1
                
                # Store complete sentiment data
                sentiment_data = {
                    'symbol': symbol,
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'article_count': len(feed),
                    'articles': articles,  # Complete article data
                    'aggregate': {
                        'avg_sentiment_score': avg_sentiment,
                        'avg_relevance_score': avg_relevance,
                        'sentiment_distribution': sentiment_distribution,
                        'total_mentions': len(ticker_sentiments)
                    },
                    'sentiment_definitions': data.get('sentiment_score_definition', ''),
                    'relevance_definition': data.get('relevance_score_definition', '')
                }
                
                await self.redis.setex(
                    f'sentiment:{symbol}',
                    self.ttls['sentiment'],
                    json.dumps(sentiment_data)
                )
                
                # Determine overall label based on average sentiment
                overall_label = 'Neutral'
                if avg_sentiment <= -0.35:
                    overall_label = 'Bearish'
                elif -0.35 < avg_sentiment <= -0.15:
                    overall_label = 'Somewhat-Bearish'
                elif 0.15 <= avg_sentiment < 0.35:
                    overall_label = 'Somewhat-Bullish'
                elif avg_sentiment >= 0.35:
                    overall_label = 'Bullish'
                
                self.logger.info(
                    f"Stored sentiment for {symbol}: score={avg_sentiment:.3f} ({overall_label}), "
                    f"relevance={avg_relevance:.3f}, articles={len(feed)}, mentions={len(ticker_sentiments)}"
                )
                
        except Exception as e:
            self.logger.error(f"Error storing sentiment for {symbol}: {e}")
    
    async def _store_technical_data(self, symbol: str, indicator_name: str, data: Dict, indicator_config: Dict):
        """Store technical indicator data to Redis."""
        try:
            tech_key = f'Technical Analysis: {indicator_name}'
            if tech_key in data:
                values = data[tech_key]
                
                # Get latest date
                latest_date = sorted(values.keys())[0]
                latest_values = values[latest_date]
                
                # Handle multi-value indicators (MACD, BBANDS)
                if 'value_keys' in indicator_config:
                    # Multi-value indicator like MACD or BBANDS
                    tech_data = {
                        'symbol': symbol,
                        'indicator': indicator_name,
                        'timestamp': int(datetime.now().timestamp() * 1000)
                    }
                    
                    # Add each value component
                    for value_key in indicator_config['value_keys']:
                        if value_key in latest_values:
                            # Store with cleaner keys
                            clean_key = value_key.replace(' ', '_').replace('Real_', '').lower()
                            tech_data[clean_key] = float(latest_values[value_key])
                    
                    await self.redis.setex(
                        f'technicals:{symbol}:{indicator_name.lower()}',
                        self.ttls['technicals'],
                        json.dumps(tech_data)
                    )
                    
                    # Log based on indicator type
                    if indicator_name == 'MACD':
                        self.logger.info(f"Stored MACD for {symbol}: MACD={tech_data.get('macd', 0):.4f}, Signal={tech_data.get('macd_signal', 0):.4f}, Hist={tech_data.get('macd_hist', 0):.4f}")
                    elif indicator_name == 'BBANDS':
                        self.logger.info(f"Stored BBANDS for {symbol}: Upper={tech_data.get('upper_band', 0):.2f}, Middle={tech_data.get('middle_band', 0):.2f}, Lower={tech_data.get('lower_band', 0):.2f}")
                    else:
                        self.logger.info(f"Stored {indicator_name} for {symbol}: {tech_data}")
                
                else:
                    # Single-value indicator like RSI or ATR
                    value_key = indicator_config.get('value_key', indicator_name)
                    latest_value = float(latest_values[value_key])
                    
                    tech_data = {
                        'symbol': symbol,
                        'indicator': indicator_name,
                        'value': latest_value,
                        'timestamp': int(datetime.now().timestamp() * 1000)
                    }
                    
                    await self.redis.setex(
                        f'technicals:{symbol}:{indicator_name.lower()}',
                        self.ttls['technicals'],
                        json.dumps(tech_data)
                    )
                    
                    self.logger.info(f"Stored {indicator_name} for {symbol}: {latest_value:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error storing {indicator_name} for {symbol}: {e}")
    
    async def _metrics_loop(self):
        """Publish metrics to Redis."""
        while self.running:
            try:
                # Update heartbeat
                heartbeat = {
                    'ts': int(datetime.now().timestamp() * 1000),
                    'symbols': len(self.symbols)
                }
                
                ttl = self.ttls.get('heartbeat', 15)
                await self.redis.setex('hb:av', ttl, json.dumps(heartbeat))
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Metrics error: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """Stop Alpha Vantage ingestion."""
        self.running = False
        self.logger.info("Stopping Alpha Vantage ingestion")


async def fetch_symbol_data(symbol: str, redis_conn: aioredis.Redis, av_session, av_config):
    """Fetch all data for a symbol (used for testing)."""
    # Implementation for testing
    pass