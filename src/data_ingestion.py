#!/usr/bin/env python3
"""
Data Ingestion Module - IBKR and Alpha Vantage Data Collection
Handles all data ingestion from external sources and writes to Redis

IBKR: Level 2 order book (SPY/QQQ/IWM only), trades, 5-second bars, execution status
Alpha Vantage: Options chains with Greeks, sentiment, technicals, fundamentals
"""

import asyncio
import json
import time
import redis
import aiohttp
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import logging
import traceback
from decimal import Decimal, ROUND_HALF_UP
import threading


class IBKRIngestion:
    """
    IBKR WebSocket data ingestion with differentiated data requirements:
    - Level 2 market depth for SPY, QQQ, IWM (0DTE/1DTE/MOC strategies)
    - Trades and bars only for other symbols (14+ DTE strategies)
    Writes all data to Redis with appropriate TTLs.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize IBKR ingestion with configuration.
        Production-ready with all config-driven parameters.
        """
        self.config = config
        self.redis = redis_conn
        self.ib = IB()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration parameters
        self.ibkr_config = config.get('ibkr', {})
        self.host = self.ibkr_config.get('host', '127.0.0.1')
        self.port = self.ibkr_config.get('port', 7497)  # 7497 for paper
        self.client_id = self.ibkr_config.get('client_id', 1)
        self.timeout = self.ibkr_config.get('timeout', 10)
        
        # Load all symbols from config
        self.all_symbols = config.get('symbols', [])
        
        # Classify symbols based on strategy requirements
        # Level 2 symbols from config or default to SPY, QQQ, IWM
        self.level2_symbols = self.ibkr_config.get('level2_symbols', ['SPY', 'QQQ', 'IWM'])
        self.standard_symbols = [s for s in self.all_symbols if s not in self.level2_symbols]
        
        # Market depth configuration
        self.market_depth_rows = self.ibkr_config.get('market_depth_rows', 20)
        
        # Data structures
        self.contracts = {}  # Symbol -> Contract mapping
        self.order_books = {}  # Order books for Level 2 symbols only
        self.trades_buffer = defaultdict(lambda: deque(maxlen=1000))  # Last 1000 trades per symbol
        self.bars_buffer = defaultdict(lambda: deque(maxlen=100))  # Last 100 bars per symbol
        
        # Subscription tracking
        self.subscriptions = {}
        self.market_data_req_ids = {}
        self.depth_req_ids = {}
        self.bar_req_ids = {}
        
        # Connection management
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.ibkr_config.get('max_reconnect_attempts', 10)
        self.reconnect_delay_base = self.ibkr_config.get('reconnect_delay_base', 2)  # seconds
        
        # Performance monitoring
        self.last_update_times = {}
        self.update_counts = defaultdict(int)
        self.processing_times = deque(maxlen=1000)
        
        # Thread safety for concurrent updates
        self.lock = threading.Lock()
        
        # Initialize order books for Level 2 symbols only
        for symbol in self.level2_symbols:
            self.order_books[symbol] = {
                'bids': [],
                'asks': [],
                'timestamp': 0,
                'market_makers': set()
            }
        
        self.logger.info(f"IBKRIngestion initialized - Level 2: {self.level2_symbols}, Standard: {self.standard_symbols}")
    
    async def start(self):
        """
        Start IBKR data ingestion with production-level reliability.
        Implements connection management, subscriptions, and monitoring.
        """
        self.logger.info("Starting IBKR data ingestion...")
        
        # Connect with retry logic
        await self._connect_with_retry()
        
        if not self.connected:
            self.logger.error("Failed to establish IBKR connection after all retries")
            return
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Subscribe to market data
        await self._subscribe_all_symbols()
        
        # Start monitoring tasks
        monitor_task = asyncio.create_task(self._monitor_connection())
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        metrics_task = asyncio.create_task(self._metrics_loop())
        
        try:
            # Keep running until shutdown
            while self.redis.get('system:halt') != '1':
                # Process events with regular asyncio.sleep since we're in async context
                # IB events are processed automatically when connected with connectAsync()
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in IBKR main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            # Clean shutdown
            await self._cleanup()
            monitor_task.cancel()
            heartbeat_task.cancel()
            metrics_task.cancel()
    
    async def _connect_with_retry(self):
        """
        Connect to IBKR with exponential backoff retry logic.
        Critical for production reliability.
        """
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.logger.info(f"Attempting IBKR connection to {self.host}:{self.port} (attempt {self.reconnect_attempts + 1})")
                
                # Connect to IBKR Gateway/TWS
                await self.ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=self.timeout
                )
                
                # Verify connection
                if self.ib.isConnected():
                    self.connected = True
                    self.reconnect_attempts = 0
                    self.redis.set('ibkr:connected', '1')
                    self.redis.set('ibkr:connection_time', datetime.now().isoformat())
                    
                    # Get account info
                    account = self.ib.managedAccounts()[0] if self.ib.managedAccounts() else 'unknown'
                    self.redis.set('ibkr:account', account)
                    
                    self.logger.info(f"Successfully connected to IBKR - Account: {account}")
                    return
                    
            except Exception as e:
                self.logger.error(f"IBKR connection failed: {e}")
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    delay = self.reconnect_delay_base ** self.reconnect_attempts
                    self.logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error("Max reconnection attempts reached")
                    self.redis.set('ibkr:connected', '0')
                    self.connected = False
                    break
    
    def _setup_event_handlers(self):
        """
        Set up all IBKR event handlers for data processing.
        Note: Market depth is handled via ticker objects, not global events.
        """
        # Ticker handlers (trades and quotes)
        self.ib.pendingTickersEvent += self.on_ticker_update
        
        # Bar handlers (5-second bars)
        self.ib.barUpdateEvent += self.on_bar_update
        
        # Error handlers
        self.ib.errorEvent += self._on_error
        
        # Connection handlers
        self.ib.disconnectedEvent += self._on_disconnect
        
        self.logger.info("Event handlers configured")
    
    async def _subscribe_all_symbols(self):
        """
        Subscribe to market data with differentiated requirements.
        Level 2 for SPY/QQQ/IWM, standard data for others.
        """
        self.logger.info("Subscribing to market data...")
        
        # Set market data type: 1=Live, 2=Frozen, 3=Delayed, 4=Delayed frozen
        market_data_type = 3 if self.config.get('use_delayed_data', False) else 1
        self.ib.reqMarketDataType(market_data_type)
        
        # Subscribe Level 2 symbols
        for symbol in self.level2_symbols:
            await self._subscribe_level2_symbol(symbol)
        
        # Subscribe standard symbols
        for symbol in self.standard_symbols:
            await self._subscribe_standard_symbol(symbol)
        
        self.logger.info(f"Subscribed to {len(self.all_symbols)} symbols")
    
    async def _subscribe_level2_symbol(self, symbol: str):
        """
        Subscribe to Level 2 market depth plus trades and bars.
        For 0DTE/1DTE/MOC strategies on SPY/QQQ/IWM.
        """
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.contracts[symbol] = contract
            
            # Qualify contract
            await self.ib.qualifyContractsAsync(contract)
            
            # Request Level 2 market depth
            # Returns a Ticker object that will hold the market depth data
            depth_ticker = self.ib.reqMktDepth(
                contract,
                numRows=self.market_depth_rows,
                isSmartDepth=True
            )
            self.depth_req_ids[symbol] = depth_ticker
            
            # Set up ticker update handler for market depth
            depth_ticker.updateEvent += lambda ticker: self.on_depth_update(ticker)
            
            # Request market data (trades and quotes)
            ticker = self.ib.reqMktData(
                contract,
                genericTickList='',
                snapshot=False,
                regulatorySnapshot=False
            )
            self.market_data_req_ids[symbol] = ticker
            
            # Request 5-second bars
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow='TRADES',
                useRTH=False,
                realTimeBarsOptions=[]
            )
            self.bar_req_ids[symbol] = bars
            
            self.subscriptions[symbol] = {
                'type': 'LEVEL2',
                'contract': contract,
                'depth': depth_ticker,
                'ticker': ticker,
                'bars': bars,
                'subscribed_at': datetime.now()
            }
            
            self.logger.info(f"Level 2 subscription complete for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe Level 2 for {symbol}: {e}")
            self.redis.hset('monitoring:subscription:errors', symbol, str(e))
    
    async def _subscribe_standard_symbol(self, symbol: str):
        """
        Subscribe to trades and bars only (no Level 2).
        For 14+ DTE strategies on other symbols.
        """
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.contracts[symbol] = contract
            
            # Qualify contract
            await self.ib.qualifyContractsAsync(contract)
            
            # Request market data (trades and quotes) - NO DEPTH
            ticker = self.ib.reqMktData(
                contract,
                genericTickList='',
                snapshot=False,
                regulatorySnapshot=False
            )
            self.market_data_req_ids[symbol] = ticker
            
            # Request 5-second bars
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow='TRADES',
                useRTH=False,
                realTimeBarsOptions=[]
            )
            self.bar_req_ids[symbol] = bars
            
            self.subscriptions[symbol] = {
                'type': 'STANDARD',
                'contract': contract,
                'ticker': ticker,
                'bars': bars,
                'subscribed_at': datetime.now()
            }
            
            self.logger.info(f"Standard subscription complete for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe standard data for {symbol}: {e}")
            self.redis.hset('monitoring:subscription:errors', symbol, str(e))
    
    def on_depth_update(self, ticker):
        """
        Handle Level 2 market depth updates for SPY/QQQ/IWM only.
        Critical for 0DTE/1DTE/MOC strategies.
        Processes domBids and domAsks from the ticker object.
        """
        start_time = time.perf_counter()
        
        try:
            # Extract symbol from ticker
            if not ticker or not ticker.contract:
                return
            symbol = ticker.contract.symbol
            
            # Only process Level 2 symbols
            if symbol not in self.level2_symbols:
                return
            
            with self.lock:
                # Get order book reference
                book = self.order_books[symbol]
                
                # Update order book from ticker's DOM data
                book['bids'] = [
                    {
                        'price': float(level.price),
                        'size': int(level.size),
                        'mm': level.marketMaker if hasattr(level, 'marketMaker') else 'UNKNOWN',
                        'cumSize': 0  # Will calculate if needed
                    }
                    for level in (ticker.domBids or [])
                ]
                
                book['asks'] = [
                    {
                        'price': float(level.price),
                        'size': int(level.size),
                        'mm': level.marketMaker if hasattr(level, 'marketMaker') else 'UNKNOWN',
                        'cumSize': 0  # Will calculate if needed
                    }
                    for level in (ticker.domAsks or [])
                ]
                
                # Track market makers from DOM data
                book['market_makers'].clear()
                for level in ticker.domBids or []:
                    if hasattr(level, 'marketMaker') and level.marketMaker:
                        book['market_makers'].add(level.marketMaker)
                for level in ticker.domAsks or []:
                    if hasattr(level, 'marketMaker') and level.marketMaker:
                        book['market_makers'].add(level.marketMaker)
                
                # Update timestamp
                book['timestamp'] = int(datetime.now().timestamp() * 1000)
                
                # Calculate metrics
                metrics = self._calculate_book_metrics(symbol)
                
                # Write to Redis with pipeline for atomicity
                pipe = self.redis.pipeline()
                
                # Store full order book (1 second TTL)
                book_data = {
                    'symbol': symbol,
                    'timestamp': book['timestamp'],
                    'bids': book['bids'][:self.market_depth_rows],  # Limit to configured depth
                    'asks': book['asks'][:self.market_depth_rows],
                    'market_makers': list(book['market_makers'])
                }
                pipe.setex(f'market:{symbol}:book', 30, json.dumps(book_data))  # 30s TTL for testing
                
                # Store metrics
                pipe.setex(f'market:{symbol}:imbalance', 30, metrics['imbalance'])  # 30s TTL
                pipe.setex(f'market:{symbol}:spread', 30, metrics['spread'])  # 30s TTL
                pipe.setex(f'market:{symbol}:mid', 30, metrics['mid'])  # 30s TTL
                pipe.set(f'market:{symbol}:timestamp', book['timestamp'])
                
                # Execute pipeline
                pipe.execute()
                
                # Update monitoring
                self.update_counts['depth'] += 1
                
        except Exception as e:
            self.logger.error(f"Error processing depth update for {symbol}: {e}")
            self.redis.hincrby('monitoring:errors:depth', symbol, 1)
        
        finally:
            # Track processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)
    
    # Note: _update_book_level method removed as we now process entire DOM from ticker object
    
    def _calculate_book_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Calculate order book metrics for Level 2 symbols.
        Critical for 0DTE/1DTE/MOC signal generation.
        """
        book = self.order_books[symbol]
        metrics = {'imbalance': 0, 'spread': 0, 'mid': 0, 'bid_liquidity': 0, 'ask_liquidity': 0}
        
        if book['bids'] and book['asks']:
            # Best bid/ask
            best_bid = book['bids'][0]['price'] if book['bids'] else 0
            best_ask = book['asks'][0]['price'] if book['asks'] else 0
            
            # Spread and mid
            metrics['spread'] = round(best_ask - best_bid, 4)
            metrics['mid'] = round((best_bid + best_ask) / 2, 4)
            
            # Calculate liquidity (top 5 levels)
            bid_liquidity = sum(level['size'] for level in book['bids'][:5])
            ask_liquidity = sum(level['size'] for level in book['asks'][:5])
            
            metrics['bid_liquidity'] = bid_liquidity
            metrics['ask_liquidity'] = ask_liquidity
            
            # Order book imbalance (-1 to 1, positive = more bids)
            total_liquidity = bid_liquidity + ask_liquidity
            if total_liquidity > 0:
                metrics['imbalance'] = round((bid_liquidity - ask_liquidity) / total_liquidity, 4)
        
        # Return all metrics with proper types
        return {
            'imbalance': metrics.get('imbalance', 0),
            'spread': metrics.get('spread', 0),
            'mid': metrics.get('mid', 0),
            'bid_liquidity': metrics.get('bid_liquidity', 0),
            'ask_liquidity': metrics.get('ask_liquidity', 0)
        }
    
    def on_ticker_update(self, tickers):
        """
        Handle ticker updates for all symbols.
        Processes trades and quotes with different logic for Level 2 vs standard symbols.
        """
        start_time = time.perf_counter()
        
        try:
            for ticker in tickers:
                if not ticker.contract:
                    continue
                    
                symbol = ticker.contract.symbol
                
                # Skip if no valid last price
                if ticker.last is None or ticker.last <= 0:
                    continue
                
                # Build trade object
                trade = {
                    'price': float(ticker.last),
                    'size': int(ticker.lastSize) if ticker.lastSize and not math.isnan(ticker.lastSize) else 0,
                    'time': int(datetime.now().timestamp() * 1000),
                    'bid': float(ticker.bid) if ticker.bid and ticker.bid > 0 and not math.isnan(ticker.bid) else None,
                    'ask': float(ticker.ask) if ticker.ask and ticker.ask > 0 and not math.isnan(ticker.ask) else None,
                    'volume': int(ticker.volume) if ticker.volume and not math.isnan(ticker.volume) else 0,
                    'vwap': float(ticker.vwap) if ticker.vwap and ticker.vwap > 0 and not math.isnan(ticker.vwap) else None
                }
                
                # Add to trades buffer
                self.trades_buffer[symbol].append(trade)
                
                # Enhanced processing for Level 2 symbols
                if symbol in self.level2_symbols:
                    self._process_level2_trade(symbol, trade, ticker)
                else:
                    self._process_standard_trade(symbol, trade, ticker)
                
                # Update Redis
                self._update_trade_data_redis(symbol, trade, ticker)
                
                # Update monitoring
                self.update_counts['trades'] += 1
                self.last_update_times[symbol] = time.time()
                
        except Exception as e:
            self.logger.error(f"Error processing ticker update: {e}")
            self.redis.hincrby('monitoring:errors:ticker', 'count', 1)
        
        finally:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)
    
    def _process_level2_trade(self, symbol: str, trade: Dict, ticker):
        """
        Enhanced trade processing for Level 2 symbols.
        Includes sweep detection and order book context.
        """
        # Detect potential sweep (large trade relative to book)
        if symbol in self.order_books:
            book = self.order_books[symbol]
            if book['bids'] and book['asks']:
                # Check if trade size exceeds top level liquidity
                top_bid_size = book['bids'][0]['size'] if book['bids'] else 0
                top_ask_size = book['asks'][0]['size'] if book['asks'] else 0
                
                if trade['size'] > max(top_bid_size, top_ask_size) * 2:
                    # Potential sweep detected
                    self.redis.setex(f'market:{symbol}:sweep', 5, json.dumps({
                        'time': trade['time'],
                        'size': trade['size'],
                        'price': trade['price']
                    }))
    
    def _process_standard_trade(self, symbol: str, trade: Dict, ticker):
        """
        Standard trade processing for non-Level 2 symbols.
        Focus on volume analysis for 14+ DTE strategies.
        """
        # Calculate rolling volume metrics
        recent_trades = list(self.trades_buffer[symbol])[-100:]  # Last 100 trades
        if len(recent_trades) >= 10:
            volumes = [t['size'] for t in recent_trades]
            avg_volume = sum(volumes) / len(volumes)
            
            # Detect unusual volume
            if trade['size'] > avg_volume * 3:
                self.redis.setex(f'market:{symbol}:unusual_volume', 30, json.dumps({
                    'time': trade['time'],
                    'size': trade['size'],
                    'price': trade['price'],
                    'ratio': round(trade['size'] / avg_volume, 2)
                }))
    
    def _update_trade_data_redis(self, symbol: str, trade: Dict, ticker):
        """
        Update Redis with trade data.
        Uses pipeline for atomic updates.
        """
        pipe = self.redis.pipeline()
        
        # Store last price
        pipe.set(f'market:{symbol}:last', trade['price'])
        
        # Store trades list (keep last 1000)
        trades_list = list(self.trades_buffer[symbol])[-1000:]
        pipe.setex(f'market:{symbol}:trades', 30, json.dumps(trades_list))  # 30s TTL
        
        # Store additional ticker data
        ticker_data = {
            'bid': trade['bid'],
            'ask': trade['ask'],
            'volume': trade['volume'],
            'vwap': trade['vwap'],
            'timestamp': trade['time']
        }
        pipe.setex(f'market:{symbol}:ticker', 30, json.dumps(ticker_data))  # 30s TTL
        
        # Calculate and store spread
        if trade['bid'] and trade['ask']:
            spread = round(trade['ask'] - trade['bid'], 4)
            pipe.setex(f'market:{symbol}:spread', 30, spread)  # 30s TTL
        
        pipe.execute()
    
    def on_bar_update(self, bars, hasNewBar):
        """
        Handle 5-second bar updates for all symbols.
        Critical for technical analysis and VPIN calculations.
        """
        start_time = time.perf_counter()
        
        try:
            # Only process if we have a new complete bar
            if not hasNewBar:
                return
            
            # Get the latest bar
            bar = bars[-1] if bars else None
            if not bar:
                return
            
            # Extract symbol from contract
            symbol = bars.contract.symbol
            
            # Build bar object (RealTimeBar uses open_ instead of open)
            bar_data = {
                'time': int(bar.time.timestamp()),
                'open': float(bar.open_),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume) if bar.volume and not math.isnan(bar.volume) else 0,
                'wap': float(bar.wap) if hasattr(bar, 'wap') and bar.wap and not math.isnan(bar.wap) else None,
                'count': int(bar.count) if hasattr(bar, 'count') and bar.count and not math.isnan(bar.count) else None
            }
            
            # Add to buffer
            self.bars_buffer[symbol].append(bar_data)
            
            # Calculate bar metrics
            metrics = self._calculate_bar_metrics(symbol)
            
            # Update Redis
            pipe = self.redis.pipeline()
            
            # Store bars (keep last 100)
            bars_list = list(self.bars_buffer[symbol])[-100:]
            pipe.setex(f'market:{symbol}:bars', 30, json.dumps(bars_list))  # 30s TTL
            
            # Store latest bar separately for quick access
            pipe.setex(f'market:{symbol}:latest_bar', 30, json.dumps(bar_data))  # 30s TTL
            
            # Store bar metrics
            pipe.setex(f'market:{symbol}:bar_metrics', 30, json.dumps(metrics))  # 30s TTL
            
            # Update timestamp
            pipe.set(f'market:{symbol}:bars_timestamp', int(datetime.now().timestamp() * 1000))
            
            pipe.execute()
            
            # Update monitoring
            self.update_counts['bars'] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing bar update: {e}")
            self.redis.hincrby('monitoring:errors:bars', 'count', 1)
        
        finally:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)
    
    def _calculate_bar_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate metrics from recent bars.
        Used for technical analysis and signal generation.
        """
        bars = list(self.bars_buffer[symbol])
        metrics = {}
        
        if len(bars) >= 2:
            # Price momentum (last 10 bars)
            recent_bars = bars[-10:] if len(bars) >= 10 else bars
            price_changes = [b['close'] - b['open'] for b in recent_bars]
            metrics['momentum'] = round(sum(price_changes), 4)
            
            # Volume profile
            volumes = [b['volume'] for b in recent_bars]
            metrics['avg_volume'] = round(sum(volumes) / len(volumes), 2)
            metrics['volume_trend'] = 'increasing' if volumes[-1] > metrics['avg_volume'] else 'decreasing'
            
            # Volatility (high-low range)
            ranges = [b['high'] - b['low'] for b in recent_bars]
            metrics['avg_range'] = round(sum(ranges) / len(ranges), 4)
            
            # VWAP calculation prep (for VPIN later)
            if bars[-1].get('wap'):
                metrics['vwap'] = bars[-1]['wap']
        
        return metrics
    
    async def _monitor_connection(self):
        """
        Monitor IBKR connection health and trigger reconnection if needed.
        Production-critical for maintaining data flow.
        """
        while True:
            try:
                if not self.ib.isConnected():
                    self.logger.warning("IBKR connection lost, attempting reconnection...")
                    self.connected = False
                    self.redis.set('ibkr:connected', '0')
                    
                    # Clear old subscriptions
                    self.subscriptions.clear()
                    self.market_data_req_ids.clear()
                    self.depth_req_ids.clear()
                    self.bar_req_ids.clear()
                    
                    # Reconnect
                    await self._connect_with_retry()
                    
                    if self.connected:
                        # Re-subscribe to all symbols
                        await self._subscribe_all_symbols()
                        self.logger.info("Reconnection successful, subscriptions restored")
                    else:
                        self.logger.error("Reconnection failed")
                
                # Check data freshness
                await self._check_data_freshness()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(5)
    
    async def _check_data_freshness(self):
        """
        Check if data is fresh for all symbols.
        Alert if data is stale.
        """
        current_time = time.time()
        stale_threshold = 10  # seconds
        
        for symbol in self.all_symbols:
            last_update = self.last_update_times.get(symbol, 0)
            if current_time - last_update > stale_threshold:
                self.logger.warning(f"Data stale for {symbol}: {current_time - last_update:.1f}s")
                self.redis.hset('monitoring:data:stale', symbol, current_time - last_update)
    
    async def _heartbeat_loop(self):
        """
        Update heartbeat for health monitoring.
        """
        while True:
            try:
                self.redis.set('module:heartbeat:ibkr_ingestion', str(datetime.now().timestamp()))
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_loop(self):
        """
        Publish performance metrics to Redis.
        """
        while True:
            try:
                # Calculate metrics (Redis requires string values, not booleans)
                metrics = {
                    'connected': '1' if self.connected else '0',
                    'symbols_subscribed': len(self.subscriptions),
                    'level2_symbols': len([s for s in self.subscriptions if self.subscriptions[s].get('type') == 'LEVEL2']),
                    'standard_symbols': len([s for s in self.subscriptions if self.subscriptions[s].get('type') == 'STANDARD']),
                    'depth_updates': self.update_counts['depth'],
                    'trade_updates': self.update_counts['trades'],
                    'bar_updates': self.update_counts['bars'],
                    'avg_processing_ms': round(sum(self.processing_times) / len(self.processing_times), 2) if self.processing_times else 0,
                    'max_processing_ms': round(max(self.processing_times), 2) if self.processing_times else 0
                }
                
                # Store metrics
                self.redis.hset('monitoring:ibkr:metrics', mapping=metrics)
                
                # Reset counters
                self.update_counts.clear()
                self.processing_times.clear()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics error: {e}")
                await asyncio.sleep(10)
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """
        Handle IBKR errors.
        """
        # Ignore common non-critical errors
        if errorCode in [2104, 2106, 2158]:  # Market data farm messages
            return
        
        self.logger.error(f"IBKR Error - ReqId: {reqId}, Code: {errorCode}, Message: {errorString}")
        
        # Store error in Redis for monitoring
        error_data = {
            'time': datetime.now().isoformat(),
            'reqId': reqId,
            'code': errorCode,
            'message': errorString,
            'contract': str(contract) if contract else None
        }
        self.redis.lpush('monitoring:ibkr:errors', json.dumps(error_data))
        self.redis.ltrim('monitoring:ibkr:errors', 0, 99)  # Keep last 100 errors
    
    def _on_disconnect(self):
        """
        Handle disconnection event.
        """
        self.logger.warning("IBKR disconnected")
        self.connected = False
        self.redis.set('ibkr:connected', '0')
        self.redis.set('ibkr:disconnect_time', datetime.now().isoformat())
    
    async def _cleanup(self):
        """
        Clean up resources on shutdown.
        """
        self.logger.info("Cleaning up IBKR connections...")
        
        try:
            # Cancel all subscriptions
            for symbol, sub in self.subscriptions.items():
                try:
                    if 'ticker' in sub and sub['ticker']:
                        self.ib.cancelMktData(sub['ticker'])
                except Exception as e:
                    self.logger.debug(f"Error canceling market data for {symbol}: {e}")
                
                try:
                    if 'bars' in sub and sub['bars']:
                        self.ib.cancelRealTimeBars(sub['bars'])
                except Exception as e:
                    self.logger.debug(f"Error canceling bars for {symbol}: {e}")
                
                # Note: Market depth is automatically canceled when ticker is canceled
            
            # Disconnect from IBKR
            if self.ib.isConnected():
                self.ib.disconnect()
            
            # Update Redis
            self.redis.set('ibkr:connected', '0')
            
            self.logger.info("IBKR cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def stop(self):
        """
        Stop the IBKR ingestion module.
        """
        await self._cleanup()


class AlphaVantageIngestion:
    """
    Alpha Vantage REST API ingestion for options chains, Greeks, sentiment, and technicals.
    Manages rate limiting (600 calls/minute) and writes to Redis.
    Production-ready with comprehensive error handling and monitoring.
    
    CRITICAL API REQUIREMENTS:
    - Options Greeks require: require_greeks=true parameter
    - IV is returned as percentage (4.66 = 466% volatility)
    - Deep ITM/OTM options can have very high IVs (>400%)
    - Premium API key recommended for 600 calls/minute
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize Alpha Vantage ingestion with full configuration.
        Production-ready initialization with all config-driven parameters.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        
        # Alpha Vantage configuration
        self.av_config = config.get('alpha_vantage', {})
        self.api_key = self.av_config.get('api_key')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")
            
        self.base_url = self.av_config.get('base_url', 'https://www.alphavantage.co/query')
        self.calls_per_minute = self.av_config.get('calls_per_minute', 600)
        self.safety_buffer = self.av_config.get('safety_buffer', 10)
        self.max_calls = self.calls_per_minute - self.safety_buffer  # 590 safe limit
        
        # Retry configuration
        self.retry_attempts = self.av_config.get('retry_attempts', 3)
        self.retry_delay_base = self.av_config.get('retry_delay', 2)
        
        # Update intervals from config
        self.options_interval = self.av_config.get('options_update_interval', 10)
        self.sentiment_interval = self.av_config.get('sentiment_update_interval', 300)
        self.technicals_interval = self.av_config.get('technicals_update_interval', 60)
        
        # Load symbols from config
        self.symbols = config.get('symbols', [])
        if not self.symbols:
            self.logger.warning("No symbols configured for Alpha Vantage ingestion")
        
        # Rate limiting: track API calls with timestamps
        self.call_times = deque(maxlen=self.calls_per_minute)
        self.rate_limit_lock = threading.Lock()
        
        # Track last update times for each data type with staggered initialization
        # CRITICAL: Stagger initial fetches to prevent thundering herd and rate limiting
        now = time.time()
        level2_symbols = ['SPY', 'QQQ', 'IWM']  # Priority symbols for 0DTE/1DTE/MOC strategies
        
        self.last_options_update = {}
        self.last_sentiment_update = {}
        self.last_technicals_update = {}
        
        for i, symbol in enumerate(self.symbols):
            if symbol in level2_symbols:
                # Priority symbols: minimal stagger (0.5 seconds apart)
                # These need rapid updates for 0DTE/1DTE/MOC strategies
                offset = level2_symbols.index(symbol) * 0.5
            else:
                # Other symbols (14+ DTE strategies): larger stagger (1 second apart)
                # Start after priority symbols with more spacing
                offset = 2.0 + ((i - len(level2_symbols)) * 1.0)
            
            # Initialize timestamps in the past to trigger staggered fetches
            # This prevents all symbols from fetching simultaneously at startup
            self.last_options_update[symbol] = now - self.options_interval + offset
            self.last_sentiment_update[symbol] = now - self.sentiment_interval + (offset * 5)  # Spread sentiment more
            self.last_technicals_update[symbol] = now - self.technicals_interval + (offset * 2)  # Moderate spread for technicals
            
            self.logger.debug(f"Initialized {symbol} with offset {offset:.1f}s")
        
        # Performance tracking
        self.api_call_count = 0
        self.api_error_count = 0
        self.last_error_time = 0
        
        # Session will be created in start()
        self.session = None
        self.running = False
        
        self.logger.info(f"AlphaVantageIngestion initialized for {len(self.symbols)} symbols")
        self.logger.info(f"Rate limit: {self.max_calls} calls/min (safety buffer: {self.safety_buffer})")
    
    async def start(self):
        """
        Start Alpha Vantage data collection loop.
        Production-ready with error handling, rate limiting, and monitoring.
        """
        self.logger.info("Starting Alpha Vantage data ingestion...")
        self.running = True
        
        # Initialize session with timeout and connection pooling
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            self.session = session
            
            # Update Redis status
            self.redis.set('av:connected', '1')
            self.redis.set('av:start_time', datetime.now().isoformat())
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_api_usage())
            
            try:
                while self.running:
                    cycle_start = time.time()
                    
                    # Create tasks for parallel data fetching
                    tasks = []
                    
                    for symbol in self.symbols:
                        now = time.time()
                        
                        # CRITICAL: Check what needs updating WITHOUT modifying timestamps
                        # Timestamps will be updated ONLY after successful fetch and storage
                        needs_options = now - self.last_options_update[symbol] >= self.options_interval
                        needs_sentiment = now - self.last_sentiment_update[symbol] >= self.sentiment_interval
                        needs_technicals = now - self.last_technicals_update[symbol] >= self.technicals_interval
                        
                        # Only create task if something needs updating
                        # Pass flags to fetch ONLY what's needed (selective fetching)
                        if needs_options or needs_sentiment or needs_technicals:
                            tasks.append(self._fetch_with_retry(
                                self.fetch_symbol_data, session, symbol,
                                needs_options, needs_sentiment, needs_technicals
                            ))
                            
                            self.logger.debug(f"{symbol} needs update - Options: {needs_options}, "
                                            f"Sentiment: {needs_sentiment}, Technicals: {needs_technicals}")
                    
                    # Execute all tasks concurrently
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Log any exceptions
                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.error(f"Task failed: {result}")
                                self.api_error_count += 1
                    
                    # Calculate sleep time to maintain cycle interval
                    cycle_time = time.time() - cycle_start
                    sleep_time = max(0, self.options_interval - cycle_time)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                self.logger.info("Alpha Vantage ingestion cancelled")
            except Exception as e:
                self.logger.error(f"Fatal error in AV ingestion loop: {e}")
                self.redis.hincrby('monitoring:errors:av', 'fatal', 1)
            finally:
                monitor_task.cancel()
                self.running = False
                self.redis.set('av:connected', '0')
    
    async def rate_limit_check(self):
        """
        Enforce Alpha Vantage rate limiting (600 calls/minute).
        Production-critical to avoid API ban.
        """
        with self.rate_limit_lock:
            now = time.time()
            
            # Remove calls older than 60 seconds
            while self.call_times and self.call_times[0] < now - 60:
                self.call_times.popleft()
            
            # Count recent calls
            recent_calls = len(self.call_times)
            
            # Update Redis monitoring
            self.redis.set('monitoring:api:av:calls', recent_calls)
            self.redis.set('monitoring:api:av:remaining', self.max_calls - recent_calls)
            
            # If approaching limit, wait
            if recent_calls >= self.max_calls:
                # Calculate wait time until oldest call expires
                wait_time = 60 - (now - self.call_times[0]) + 0.1  # Add 100ms buffer
                
                self.logger.warning(f"Rate limit reached ({recent_calls}/{self.max_calls}), waiting {wait_time:.1f}s")
                self.redis.hincrby('monitoring:api:av:rate_limits', 'count', 1)
                
                await asyncio.sleep(wait_time)
                
                # Remove expired calls after waiting
                while self.call_times and self.call_times[0] < time.time() - 60:
                    self.call_times.popleft()
            
            # Record this call
            self.call_times.append(now)
            self.api_call_count += 1
    
    async def fetch_symbol_data(self, session: aiohttp.ClientSession, symbol: str,
                               needs_options: bool = True, needs_sentiment: bool = True, 
                               needs_technicals: bool = True):
        """
        Fetch Alpha Vantage data for a symbol - ONLY the types that need updating.
        CRITICAL: Updates timestamps ONLY after successful fetch and storage.
        
        Args:
            session: aiohttp session for API calls
            symbol: Stock symbol to fetch
            needs_options: Whether to fetch options data
            needs_sentiment: Whether to fetch sentiment data  
            needs_technicals: Whether to fetch technical indicators
        """
        now = time.time()
        results = {'options': False, 'sentiment': False, 'technicals': False}
        
        try:
            # FETCH OPTIONS CHAIN if needed
            if needs_options:
                try:
                    options_data = await self.fetch_options_chain(session, symbol)
                    
                    if options_data:
                        # Detect unusual activity
                        unusual = self.detect_unusual_activity(options_data.get('contracts', []))
                        
                        # Calculate GEX and DEX
                        gex, dex = self._calculate_greek_exposures(options_data.get('contracts', []))
                        
                        # Store in Redis with appropriate TTLs
                        pipe = self.redis.pipeline()
                        
                        # Options chain (10 second TTL)
                        pipe.setex(f'options:{symbol}:chain', 10, json.dumps(options_data))
                        
                        # Greeks by strike/expiry (10 second TTL)
                        greeks = self._extract_greeks(options_data.get('contracts', []))
                        pipe.setex(f'options:{symbol}:greeks', 10, json.dumps(greeks))
                        
                        # Gamma and Delta exposure (10 second TTL)
                        pipe.setex(f'options:{symbol}:gex', 10, json.dumps(gex))
                        pipe.setex(f'options:{symbol}:dex', 10, json.dumps(dex))
                        
                        # Unusual activity (10 second TTL)
                        pipe.setex(f'options:{symbol}:unusual', 10, json.dumps(unusual))
                        
                        # Options flow (10 second TTL)
                        flow = self._calculate_options_flow(options_data.get('contracts', []))
                        pipe.setex(f'options:{symbol}:flow', 10, json.dumps(flow))
                        
                        # Update timestamp
                        pipe.set(f'options:{symbol}:timestamp', int(datetime.now().timestamp() * 1000))
                        
                        pipe.execute()
                        
                        # CRITICAL: Update timestamp ONLY after successful storage
                        self.last_options_update[symbol] = now
                        results['options'] = True
                        self.logger.debug(f"Updated options data for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error fetching options for {symbol}: {e}")
                    self.redis.hincrby('monitoring:errors:av:options', symbol, 1)
                    # DO NOT update timestamp on failure - will retry next cycle
            
            # FETCH SENTIMENT DATA if needed
            if needs_sentiment:
                try:
                    sentiment_data = await self.fetch_sentiment(session, symbol)
                    if sentiment_data:
                        # Sentiment already stores itself in fetch_sentiment
                        # CRITICAL: Update timestamp ONLY after successful fetch
                        self.last_sentiment_update[symbol] = now
                        results['sentiment'] = True
                        self.logger.debug(f"Updated sentiment for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error fetching sentiment for {symbol}: {e}")
                    self.redis.hincrby('monitoring:errors:av:sentiment', symbol, 1)
                    # DO NOT update timestamp on failure
            
            # FETCH TECHNICAL INDICATORS if needed
            if needs_technicals:
                try:
                    technical_data = await self.fetch_technical_indicators(session, symbol)
                    if technical_data:
                        # Technicals already store themselves in fetch_technical_indicators
                        # CRITICAL: Update timestamp ONLY after successful fetch
                        self.last_technicals_update[symbol] = now
                        results['technicals'] = True
                        self.logger.debug(f"Updated technical indicators for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error fetching technicals for {symbol}: {e}")
                    self.redis.hincrby('monitoring:errors:av:technicals', symbol, 1)
                    # DO NOT update timestamp on failure
            
            # Log results for monitoring
            successful = sum(1 for v in results.values() if v)
            requested = sum([needs_options, needs_sentiment, needs_technicals])
            
            if successful < requested:
                self.logger.warning(f"{symbol}: Only {successful}/{requested} data types fetched successfully")
                self.redis.hincrby('monitoring:av:partial_fetches', symbol, 1)
            elif successful > 0:
                self.logger.debug(f"{symbol}: Successfully fetched {successful} data type(s)")
                
        except Exception as e:
            self.logger.error(f"Critical error fetching data for {symbol}: {e}")
            self.redis.hincrby('monitoring:errors:av:critical', symbol, 1)
    
    async def fetch_options_chain(self, session: aiohttp.ClientSession, symbol: str):
        """
        Fetch options chain with Greeks from Alpha Vantage.
        
        CRITICAL: Must use require_greeks=true to get Greeks data!
        Without this parameter, only basic pricing data is returned.
        Greeks are PROVIDED by Alpha Vantage, ensuring institutional-grade accuracy.
        """
        try:
            # Check rate limit before making request
            await self.rate_limit_check()
            
            # Build API URL with REALTIME_OPTIONS function
            # CRITICAL: Must include require_greeks=true to get Greeks data!
            params = {
                'function': 'REALTIME_OPTIONS',
                'symbol': symbol,
                'apikey': self.api_key,
                'datatype': 'json',
                'require_greeks': 'true'  # Essential for Greeks data
            }
            
            async with session.get(self.base_url, params=params) as response:
                if response.status == 429:  # Rate limited
                    self.logger.warning(f"Rate limited by Alpha Vantage for {symbol}")
                    self.redis.hincrby('monitoring:api:av:rate_limits', 'external', 1)
                    await asyncio.sleep(60)  # Wait before retry
                    return None
                
                if response.status != 200:
                    self.logger.error(f"API error {response.status} for {symbol}")
                    return None
                
                data = await response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    self.logger.error(f"API error for {symbol}: {data['Error Message']}")
                    return None
                
                if 'Note' in data:  # API call frequency limit
                    self.logger.warning(f"API limit warning: {data['Note']}")
                    await asyncio.sleep(60)
                    return None
                
                # Parse options chain
                contracts = []
                option_chain = data.get('data', [])
                
                for contract in option_chain:
                    # Validate and extract contract data
                    try:
                        # Alpha Vantage returns IV as a percentage (e.g., 4.66 = 466%)
                        # Convert to decimal form for consistency
                        iv_raw = float(contract.get('implied_volatility', 0))
                        
                        # Convert from percentage to decimal (4.66 -> 0.0466)
                        # BUT if IV seems already in decimal form (< 10), keep as is
                        if iv_raw > 10:
                            # Likely in percentage form (e.g., 466%)
                            iv = iv_raw / 100.0  # Convert to decimal
                        else:
                            # Either already decimal or reasonable percentage
                            iv = iv_raw
                        
                        # Skip if IV is still invalid (0 or negative)
                        if iv <= 0:
                            self.logger.warning(f"Invalid IV {iv_raw} for {symbol}")
                            continue
                        
                        contract_data = {
                            'symbol': symbol,
                            'contractID': contract.get('contractID', ''),  # Correct field name
                            'strike': float(contract.get('strike', 0)),
                            'expiration': contract.get('expiration', ''),
                            'type': contract.get('type', '').lower(),  # 'call' or 'put'
                            'last': float(contract.get('last', 0)),
                            'mark': float(contract.get('mark', 0)),  # Mark price from API
                            'bid': float(contract.get('bid', 0)),
                            'ask': float(contract.get('ask', 0)),
                            'bid_size': int(contract.get('bid_size', 0)),
                            'ask_size': int(contract.get('ask_size', 0)),
                            'volume': int(contract.get('volume', 0)),
                            'openInterest': int(contract.get('open_interest', 0)),
                            'impliedVolatility': iv,  # Now in decimal form
                            # Greeks provided by Alpha Vantage (with require_greeks=true)
                            'delta': float(contract.get('delta', 0)),
                            'gamma': float(contract.get('gamma', 0)),
                            'theta': float(contract.get('theta', 0)),
                            'vega': float(contract.get('vega', 0)),
                            'rho': float(contract.get('rho', 0)),
                            'timestamp': int(datetime.now().timestamp() * 1000)
                        }
                        
                        # Additional calculated metrics
                        contract_data['midPrice'] = (contract_data['bid'] + contract_data['ask']) / 2
                        contract_data['spread'] = contract_data['ask'] - contract_data['bid']
                        contract_data['volumeOIRatio'] = (
                            contract_data['volume'] / contract_data['openInterest'] 
                            if contract_data['openInterest'] > 0 else 0
                        )
                        
                        contracts.append(contract_data)
                        
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f"Skipping invalid contract for {symbol}: {e}")
                        continue
                
                # Sort by expiration and strike
                contracts.sort(key=lambda x: (x['expiration'], x['strike']))
                
                return {
                    'symbol': symbol,
                    'contracts': contracts,
                    'count': len(contracts),
                    'timestamp': int(datetime.now().timestamp() * 1000)
                }
                
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching options for {symbol}")
            self.redis.hincrby('monitoring:errors:av', 'timeouts', 1)
            return None
        except Exception as e:
            self.logger.error(f"Error fetching options chain for {symbol}: {e}")
            self.api_error_count += 1
            return None
    
    async def fetch_sentiment(self, session: aiohttp.ClientSession, symbol: str):
        """
        Fetch news sentiment for a symbol.
        Aggregates sentiment from multiple news articles for market sentiment analysis.
        """
        try:
            # Check rate limit
            await self.rate_limit_check()
            
            # Build API URL with NEWS_SENTIMENT function
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_key,
                'limit': 50  # Get last 50 articles
            }
            
            async with session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"Sentiment API error {response.status} for {symbol}")
                    return None
                
                data = await response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    self.logger.error(f"Sentiment API error: {data['Error Message']}")
                    return None
                
                # Parse sentiment from articles
                articles = data.get('feed', [])
                if not articles:
                    return None
                
                # Calculate aggregate sentiment
                total_score = 0
                weighted_score = 0
                article_count = 0
                bullish_count = 0
                bearish_count = 0
                neutral_count = 0
                
                recent_articles = []
                
                for article in articles[:20]:  # Focus on 20 most recent
                    # Extract ticker-specific sentiment
                    ticker_sentiment = None
                    for ts in article.get('ticker_sentiment', []):
                        if ts.get('ticker') == symbol:
                            ticker_sentiment = ts
                            break
                    
                    if ticker_sentiment:
                        score = float(ticker_sentiment.get('ticker_sentiment_score', 0))
                        relevance = float(ticker_sentiment.get('relevance_score', 0))
                        
                        # Weight by relevance
                        weighted_score += score * relevance
                        total_score += score
                        article_count += 1
                        
                        # Classify sentiment
                        if score > 0.15:
                            bullish_count += 1
                        elif score < -0.15:
                            bearish_count += 1
                        else:
                            neutral_count += 1
                        
                        # Store article info
                        recent_articles.append({
                            'title': article.get('title', ''),
                            'source': article.get('source', ''),
                            'sentiment_score': score,
                            'relevance': relevance,
                            'time_published': article.get('time_published', '')
                        })
                
                if article_count > 0:
                    avg_sentiment = total_score / article_count
                    weighted_avg = weighted_score / article_count
                    
                    sentiment_data = {
                        'symbol': symbol,
                        'sentiment_score': round(avg_sentiment, 4),
                        'weighted_score': round(weighted_avg, 4),
                        'bullish_count': bullish_count,
                        'bearish_count': bearish_count,
                        'neutral_count': neutral_count,
                        'article_count': article_count,
                        'sentiment_label': self._classify_sentiment(weighted_avg),
                        'recent_articles': recent_articles[:5],  # Keep top 5
                        'timestamp': int(datetime.now().timestamp() * 1000)
                    }
                    
                    # Store in Redis with 5-minute TTL
                    pipe = self.redis.pipeline()
                    pipe.setex(f'sentiment:{symbol}:score', 300, json.dumps(sentiment_data))
                    pipe.setex(f'sentiment:{symbol}:articles', 300, json.dumps(recent_articles))
                    pipe.execute()
                    
                    self.logger.debug(f"Updated sentiment for {symbol}: {sentiment_data['sentiment_label']}")
                    
                    return sentiment_data
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching sentiment for {symbol}: {e}")
            self.api_error_count += 1
            return None
    
    def detect_unusual_activity(self, options_chain: list) -> dict:
        """
        Detect unusual options activity from chain data.
        Critical for identifying institutional positioning and smart money flow.
        """
        try:
            unusual_contracts = []
            
            for contract in options_chain:
                volume = contract.get('volume', 0)
                open_interest = contract.get('openInterest', 0)
                
                # Skip contracts with no activity
                if volume == 0 or open_interest == 0:
                    continue
                
                # Calculate volume/OI ratio
                ratio = volume / open_interest if open_interest > 0 else 0
                
                # Flag unusual activity (volume > 2x OI)
                if ratio > 2.0:
                    unusual_contracts.append({
                        'strike': contract['strike'],
                        'expiration': contract['expiration'],
                        'type': contract['type'],
                        'volume': volume,
                        'openInterest': open_interest,
                        'ratio': round(ratio, 2),
                        'delta': contract.get('delta', 0),
                        'gamma': contract.get('gamma', 0),
                        'impliedVolatility': contract.get('impliedVolatility', 0),
                        'last': contract.get('last', 0),
                        'notional': volume * contract.get('last', 0) * 100  # Notional value
                    })
            
            # Sort by ratio (highest first)
            unusual_contracts.sort(key=lambda x: x['ratio'], reverse=True)
            
            # Return top 10 most unusual
            top_unusual = unusual_contracts[:10]
            
            return {
                'detected': len(unusual_contracts) > 0,
                'count': len(unusual_contracts),
                'contracts': top_unusual,
                'total_unusual_volume': sum(c['volume'] for c in unusual_contracts),
                'total_unusual_notional': sum(c['notional'] for c in unusual_contracts),
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting unusual activity: {e}")
            return {'detected': False, 'contracts': [], 'error': str(e)}
    
    async def fetch_technical_indicators(self, session: aiohttp.ClientSession, symbol: str):
        """
        Fetch technical indicators from Alpha Vantage.
        RSI, MACD, and Bollinger Bands for comprehensive technical analysis.
        """
        try:
            indicators = {}
            
            # Fetch RSI (14-period)
            await self.rate_limit_check()
            rsi_params = {
                'function': 'RSI',
                'symbol': symbol,
                'interval': '5min',
                'time_period': 14,
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            async with session.get(self.base_url, params=rsi_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Technical Analysis: RSI' in data:
                        rsi_data = data['Technical Analysis: RSI']
                        latest_time = list(rsi_data.keys())[0] if rsi_data else None
                        if latest_time:
                            indicators['rsi'] = {
                                'value': float(rsi_data[latest_time]['RSI']),
                                'timestamp': latest_time,
                                'overbought': float(rsi_data[latest_time]['RSI']) > 70,
                                'oversold': float(rsi_data[latest_time]['RSI']) < 30
                            }
            
            # Fetch MACD
            await self.rate_limit_check()
            macd_params = {
                'function': 'MACD',
                'symbol': symbol,
                'interval': '5min',
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            async with session.get(self.base_url, params=macd_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Technical Analysis: MACD' in data:
                        macd_data = data['Technical Analysis: MACD']
                        latest_time = list(macd_data.keys())[0] if macd_data else None
                        if latest_time:
                            macd_val = float(macd_data[latest_time].get('MACD', 0))
                            signal_val = float(macd_data[latest_time].get('MACD_Signal', 0))
                            hist_val = float(macd_data[latest_time].get('MACD_Hist', 0))
                            
                            indicators['macd'] = {
                                'macd': macd_val,
                                'signal': signal_val,
                                'histogram': hist_val,
                                'timestamp': latest_time,
                                'bullish_crossover': macd_val > signal_val and hist_val > 0,
                                'bearish_crossover': macd_val < signal_val and hist_val < 0
                            }
            
            # Fetch Bollinger Bands
            await self.rate_limit_check()
            bb_params = {
                'function': 'BBANDS',
                'symbol': symbol,
                'interval': '5min',
                'time_period': 20,
                'series_type': 'close',
                'nbdevup': 2,
                'nbdevdn': 2,
                'apikey': self.api_key
            }
            
            async with session.get(self.base_url, params=bb_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Technical Analysis: BBANDS' in data:
                        bb_data = data['Technical Analysis: BBANDS']
                        latest_time = list(bb_data.keys())[0] if bb_data else None
                        if latest_time:
                            upper = float(bb_data[latest_time].get('Real Upper Band', 0))
                            middle = float(bb_data[latest_time].get('Real Middle Band', 0))
                            lower = float(bb_data[latest_time].get('Real Lower Band', 0))
                            
                            # Get current price from Redis
                            last_price_str = self.redis.get(f'market:{symbol}:last')
                            last_price = float(last_price_str) if last_price_str else middle
                            
                            indicators['bbands'] = {
                                'upper': upper,
                                'middle': middle,
                                'lower': lower,
                                'bandwidth': upper - lower,
                                'timestamp': latest_time,
                                'position': 'above' if last_price > upper else 'below' if last_price < lower else 'within',
                                'squeeze': (upper - lower) / middle < 0.1  # Volatility squeeze detection
                            }
            
            # Store in Redis if we have data
            if indicators:
                pipe = self.redis.pipeline()
                
                # Store each indicator separately (60 second TTL)
                if 'rsi' in indicators:
                    pipe.setex(f'technicals:{symbol}:rsi', 60, json.dumps(indicators['rsi']))
                if 'macd' in indicators:
                    pipe.setex(f'technicals:{symbol}:macd', 60, json.dumps(indicators['macd']))
                if 'bbands' in indicators:
                    pipe.setex(f'technicals:{symbol}:bbands', 60, json.dumps(indicators['bbands']))
                
                # Store combined technicals
                pipe.setex(f'technicals:{symbol}:all', 60, json.dumps(indicators))
                pipe.set(f'technicals:{symbol}:timestamp', int(datetime.now().timestamp() * 1000))
                
                pipe.execute()
                
                self.logger.debug(f"Updated technical indicators for {symbol}")
                
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error fetching technical indicators for {symbol}: {e}")
            self.api_error_count += 1
            return {}
    
    async def handle_api_error(self, response: aiohttp.ClientResponse, symbol: str):
        """
        Handle Alpha Vantage API errors and implement retry logic.
        Production-critical for maintaining data flow despite API issues.
        """
        try:
            status = response.status
            
            if status == 429:  # Rate limited
                self.logger.warning(f"Rate limited (429) for {symbol}")
                self.redis.hincrby('monitoring:api:av:errors', 'rate_limit', 1)
                await asyncio.sleep(60)  # Wait 1 minute
                return 'retry'
                
            elif status == 401:  # Invalid API key
                self.logger.error("Invalid Alpha Vantage API key")
                self.redis.hincrby('monitoring:api:av:errors', 'auth', 1)
                return 'fatal'
                
            elif status == 404:  # Symbol not found
                self.logger.error(f"Symbol {symbol} not found")
                self.redis.hincrby('monitoring:api:av:errors', 'not_found', 1)
                return 'skip'
                
            elif status >= 500:  # Server errors
                self.logger.error(f"Server error {status} for {symbol}")
                self.redis.hincrby('monitoring:api:av:errors', 'server', 1)
                await asyncio.sleep(10)  # Brief wait
                return 'retry'
                
            else:
                self.logger.error(f"Unexpected status {status} for {symbol}")
                self.redis.hincrby('monitoring:api:av:errors', 'other', 1)
                return 'skip'
                
        except Exception as e:
            self.logger.error(f"Error handling API error: {e}")
            return 'skip'
    
    async def _fetch_with_retry(self, fetch_func, session, symbol, *args):
        """
        Wrapper for API calls with retry logic and exponential backoff.
        Supports additional arguments for selective fetching.
        
        Args:
            fetch_func: The function to call with retries
            session: aiohttp session
            symbol: Stock symbol
            *args: Additional arguments to pass to fetch_func (e.g., needs_options, needs_sentiment, needs_technicals)
        """
        for attempt in range(self.retry_attempts):
            try:
                # Call with all provided arguments
                if args:
                    result = await fetch_func(session, symbol, *args)
                else:
                    result = await fetch_func(session, symbol)
                    
                if result is not None:
                    return result
                    
                # If None, might be temporary issue, retry with backoff
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay_base ** attempt
                    self.logger.debug(f"Retrying {fetch_func.__name__} for {symbol} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                self.logger.error(f"Error in {fetch_func.__name__} for {symbol}: {e}")
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay_base ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    self.redis.hincrby('monitoring:errors:av', 'max_retries', 1)
                    
        return None
    
    def _calculate_greek_exposures(self, contracts: list) -> tuple:
        """
        Calculate Gamma Exposure (GEX) and Delta Exposure (DEX) from options chain.
        Critical for understanding market maker positioning.
        """
        try:
            total_gex = 0
            total_dex = 0
            
            for contract in contracts:
                # Extract necessary values
                gamma = contract.get('gamma', 0)
                delta = contract.get('delta', 0)
                open_interest = contract.get('openInterest', 0)
                strike = contract.get('strike', 0)
                contract_type = contract.get('type', '')
                
                # Contract multiplier (100 shares per contract)
                multiplier = 100
                
                # Calculate exposures
                # GEX = OI * Gamma * Strike^2 * 0.01 * Multiplier
                gex_value = open_interest * gamma * (strike ** 2) * 0.01 * multiplier
                
                # DEX = OI * Delta * Strike * Multiplier
                dex_value = open_interest * delta * strike * multiplier
                
                # Adjust for put/call
                if contract_type == 'put':
                    gex_value = -gex_value
                    dex_value = -dex_value
                
                total_gex += gex_value
                total_dex += dex_value
            
            return (
                {
                    'total': round(total_gex, 2),
                    'normalized': round(total_gex / 1e9, 4),  # In billions
                    'timestamp': int(datetime.now().timestamp() * 1000)
                },
                {
                    'total': round(total_dex, 2),
                    'normalized': round(total_dex / 1e9, 4),  # In billions
                    'timestamp': int(datetime.now().timestamp() * 1000)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Greek exposures: {e}")
            return ({'total': 0, 'normalized': 0}, {'total': 0, 'normalized': 0})
    
    def _extract_greeks(self, contracts: list) -> dict:
        """
        Extract and organize Greeks by strike and expiration.
        """
        try:
            greeks_map = {}
            
            for contract in contracts:
                key = f"{contract['strike']}_{contract['expiration']}_{contract['type']}"
                greeks_map[key] = {
                    'delta': contract.get('delta', 0),
                    'gamma': contract.get('gamma', 0),
                    'theta': contract.get('theta', 0),
                    'vega': contract.get('vega', 0),
                    'rho': contract.get('rho', 0),
                    'iv': contract.get('impliedVolatility', 0)
                }
            
            return greeks_map
            
        except Exception as e:
            self.logger.error(f"Error extracting Greeks: {e}")
            return {}
    
    def _calculate_options_flow(self, contracts: list) -> dict:
        """
        Calculate options flow metrics for signal generation.
        """
        try:
            call_volume = 0
            put_volume = 0
            call_oi = 0
            put_oi = 0
            call_premium = 0
            put_premium = 0
            
            for contract in contracts:
                volume = contract.get('volume', 0)
                oi = contract.get('openInterest', 0)
                last = contract.get('last', 0)
                
                if contract.get('type') == 'call':
                    call_volume += volume
                    call_oi += oi
                    call_premium += volume * last * 100
                else:
                    put_volume += volume
                    put_oi += oi
                    put_premium += volume * last * 100
            
            total_volume = call_volume + put_volume
            
            return {
                'call_volume': call_volume,
                'put_volume': put_volume,
                'total_volume': total_volume,
                'put_call_ratio': put_volume / call_volume if call_volume > 0 else 0,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'oi_put_call_ratio': put_oi / call_oi if call_oi > 0 else 0,
                'call_premium': round(call_premium, 2),
                'put_premium': round(put_premium, 2),
                'net_premium': round(call_premium - put_premium, 2),
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating options flow: {e}")
            return {}
    
    def _classify_sentiment(self, score: float) -> str:
        """
        Classify sentiment score into categories.
        """
        if score > 0.35:
            return 'Very Bullish'
        elif score > 0.15:
            return 'Bullish'
        elif score > -0.15:
            return 'Neutral'
        elif score > -0.35:
            return 'Bearish'
        else:
            return 'Very Bearish'
    
    async def _monitor_api_usage(self):
        """
        Monitor API usage and update metrics.
        """
        while self.running:
            try:
                # Update metrics
                self.redis.set('monitoring:api:av:total_calls', self.api_call_count)
                self.redis.set('monitoring:api:av:error_count', self.api_error_count)
                
                # Calculate calls per minute
                with self.rate_limit_lock:
                    now = time.time()
                    recent_calls = sum(1 for t in self.call_times if t > now - 60)
                    self.redis.set('monitoring:api:av:calls_per_minute', recent_calls)
                
                # Check if we're healthy
                if self.api_error_count > 100:
                    self.logger.warning(f"High error count: {self.api_error_count}")
                    self.redis.set('monitoring:api:av:health', 'degraded')
                else:
                    self.redis.set('monitoring:api:av:health', 'healthy')
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in API monitoring: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """
        Stop the Alpha Vantage ingestion module.
        """
        self.logger.info("Stopping Alpha Vantage ingestion...")
        self.running = False
        self.redis.set('av:connected', '0')
        self.redis.set('av:stop_time', datetime.now().isoformat())


class DataQualityMonitor:
    """
    Monitor data quality and freshness from all sources.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize data quality monitoring.
        Production-ready implementation for monitoring data quality and freshness.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        
        # Define freshness thresholds (in seconds)
        self.freshness_thresholds = {
            'ibkr': 1.0,  # IBKR data should be < 1 second old
            'av_options': 10.0,  # Alpha Vantage options < 10 seconds old
            'av_sentiment': 300.0,  # Sentiment < 5 minutes old
            'av_technicals': 60.0  # Technicals < 1 minute old
        }
        
        # Track data gaps
        self.last_update_times = {}
        self.gap_alerts = {}
        
        # Metrics tracking
        self.validation_failures = deque(maxlen=1000)
        self.freshness_violations = deque(maxlen=1000)
        
        # Reasonable spread thresholds (% of mid price)
        self.max_spread_pct = {
            'SPY': 0.01,  # 1 basis point for SPY
            'QQQ': 0.01,  # 1 basis point for QQQ
            'IWM': 0.02,  # 2 basis points for IWM
            'default': 0.05  # 5 basis points for other symbols
        }
    
    async def monitor_data_freshness(self):
        """
        Check data freshness for all symbols and sources.
        Production implementation that alerts on stale data and tracks gaps.
        """
        while True:
            try:
                now = time.time()
                symbols = self.config.get('symbols', [])
                
                for symbol in symbols:
                    # Check IBKR data freshness
                    ibkr_timestamp = self.redis.get(f'market:{symbol}:timestamp')
                    if ibkr_timestamp:
                        age = now - (int(ibkr_timestamp) / 1000.0)
                        
                        if age > self.freshness_thresholds['ibkr']:
                            # Data is stale
                            self.logger.warning(f"IBKR data stale for {symbol}: {age:.2f}s old")
                            self.redis.setex(f'monitoring:data:freshness:ibkr:{symbol}', 60, json.dumps({
                                'status': 'stale',
                                'age': age,
                                'threshold': self.freshness_thresholds['ibkr'],
                                'timestamp': now
                            }))
                            self.freshness_violations.append((symbol, 'ibkr', age))
                            
                            # Track gap if significant
                            last_update = self.last_update_times.get(f'ibkr:{symbol}', 0)
                            if last_update > 0 and age - last_update > 5:
                                self.redis.hincrby('monitoring:data:gaps:ibkr', symbol, 1)
                        else:
                            # Data is fresh
                            self.redis.setex(f'monitoring:data:freshness:ibkr:{symbol}', 60, json.dumps({
                                'status': 'fresh',
                                'age': age,
                                'timestamp': now
                            }))
                        
                        self.last_update_times[f'ibkr:{symbol}'] = age
                    
                    # Check Alpha Vantage options freshness
                    av_options_timestamp = self.redis.get(f'options:{symbol}:timestamp')
                    if av_options_timestamp:
                        age = now - (int(av_options_timestamp) / 1000.0)
                        
                        if age > self.freshness_thresholds['av_options']:
                            self.logger.warning(f"AV options stale for {symbol}: {age:.2f}s old")
                            self.redis.setex(f'monitoring:data:freshness:av:{symbol}', 60, json.dumps({
                                'status': 'stale',
                                'age': age,
                                'threshold': self.freshness_thresholds['av_options'],
                                'timestamp': now
                            }))
                            self.freshness_violations.append((symbol, 'av_options', age))
                        else:
                            self.redis.setex(f'monitoring:data:freshness:av:{symbol}', 60, json.dumps({
                                'status': 'fresh',
                                'age': age,
                                'timestamp': now
                            }))
                    
                    # Check technical indicators freshness
                    tech_timestamp = self.redis.get(f'technicals:{symbol}:timestamp')
                    if tech_timestamp:
                        age = now - (int(tech_timestamp) / 1000.0)
                        if age > self.freshness_thresholds['av_technicals']:
                            self.logger.warning(f"Technicals stale for {symbol}: {age:.2f}s old")
                            self.freshness_violations.append((symbol, 'av_technicals', age))
                
                # Store summary metrics
                self.redis.setex('monitoring:data:freshness:summary', 60, json.dumps({
                    'total_violations': len(self.freshness_violations),
                    'recent_violations': list(self.freshness_violations)[-10:],
                    'timestamp': now
                }))
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in data freshness monitoring: {e}")
                await asyncio.sleep(5)
    
    def validate_market_data(self, symbol: str, data: dict) -> bool:
        """
        Validate market data quality.
        Production implementation with comprehensive validation.
        
        Returns:
            True if data passes validation
        """
        try:
            # Check for required fields
            required_fields = ['bid', 'ask', 'last', 'volume', 'timestamp']
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Missing field {field} in {symbol} data")
                    self.validation_failures.append((symbol, f'missing_{field}', time.time()))
                    return False
            
            # Check for zero/null values
            if not data.get('bid') or not data.get('ask'):
                self.logger.error(f"Zero bid/ask for {symbol}: bid={data.get('bid')}, ask={data.get('ask')}")
                self.validation_failures.append((symbol, 'zero_price', time.time()))
                return False
            
            # Validate bid < ask
            if data['bid'] >= data['ask']:
                self.logger.error(f"Invalid spread for {symbol}: bid={data['bid']} >= ask={data['ask']}")
                self.validation_failures.append((symbol, 'inverted_spread', time.time()))
                return False
            
            # Check spread reasonableness
            spread = data['ask'] - data['bid']
            mid = (data['bid'] + data['ask']) / 2
            spread_pct = spread / mid if mid > 0 else 999
            
            max_spread = self.max_spread_pct.get(symbol, self.max_spread_pct['default'])
            if spread_pct > max_spread:
                self.logger.warning(f"Wide spread for {symbol}: {spread_pct:.4%} > {max_spread:.4%}")
                self.redis.setex(f'monitoring:data:quality:spread:{symbol}', 60, json.dumps({
                    'spread_pct': spread_pct,
                    'threshold': max_spread,
                    'status': 'wide'
                }))
            
            # Verify price continuity (if we have previous price)
            last_price_str = self.redis.get(f'market:{symbol}:last')
            if last_price_str:
                last_price = float(last_price_str)
                price_change = abs(data['last'] - last_price) / last_price if last_price > 0 else 0
                
                # Alert on > 10% price jump (likely data error)
                if price_change > 0.10:
                    self.logger.error(f"Price discontinuity for {symbol}: {price_change:.2%} change")
                    self.validation_failures.append((symbol, 'price_jump', time.time()))
                    return False
            
            # Validate volume is non-negative
            if data.get('volume', 0) < 0:
                self.logger.error(f"Negative volume for {symbol}: {data['volume']}")
                self.validation_failures.append((symbol, 'negative_volume', time.time()))
                return False
            
            # Check timestamp is recent (< 5 seconds old)
            now = time.time() * 1000
            if abs(now - data['timestamp']) > 5000:
                self.logger.warning(f"Timestamp mismatch for {symbol}: {abs(now - data['timestamp'])/1000:.1f}s difference")
                self.validation_failures.append((symbol, 'timestamp_drift', time.time()))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating market data for {symbol}: {e}")
            return False
    
    def validate_options_data(self, symbol: str, chain: list) -> bool:
        """
        Validate options chain data quality.
        Production implementation with comprehensive Greek and contract validation.
        
        Returns:
            True if data passes validation
        """
        try:
            if not chain:
                self.logger.warning(f"Empty options chain for {symbol}")
                return False
            
            validation_errors = []
            now = datetime.now()
            
            for contract in chain:
                # Validate Greeks ranges
                delta = contract.get('delta', 0)
                if not (-1 <= delta <= 1):
                    validation_errors.append(f"Invalid delta: {delta}")
                
                gamma = contract.get('gamma', 0)
                if gamma < 0 or gamma > 1:
                    validation_errors.append(f"Invalid gamma: {gamma}")
                
                # Theta is usually negative (time decay) but can be positive for deep ITM puts
                # or near expiration. Alpha Vantage real data shows this is normal.
                theta = contract.get('theta', 0)
                if theta > 1.0:  # Only flag if extremely positive (>$1/day)
                    validation_errors.append(f"Extremely positive theta: {theta}")
                
                # Vega should be positive
                vega = contract.get('vega', 0)
                if vega < 0:
                    validation_errors.append(f"Negative vega: {vega}")
                
                # Check IV is reasonable (0 < IV < 5)
                iv = contract.get('impliedVolatility', 0)
                if iv <= 0:
                    validation_errors.append(f"Non-positive IV: {iv}")
                elif iv > 5:  # 500% annualized vol
                    validation_errors.append(f"Excessive IV: {iv}")
                
                # Verify strike is positive
                strike = contract.get('strike', 0)
                if strike <= 0:
                    validation_errors.append(f"Invalid strike: {strike}")
                
                # Check expiration is in the future
                exp_str = contract.get('expiration', '')
                if exp_str:
                    try:
                        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                        if exp_date.date() < now.date():
                            validation_errors.append(f"Past expiration: {exp_str}")
                    except ValueError:
                        validation_errors.append(f"Invalid expiration format: {exp_str}")
                
                # Validate open interest >= 0
                oi = contract.get('openInterest', 0)
                if oi < 0:
                    validation_errors.append(f"Negative OI: {oi}")
                
                # Validate volume >= 0
                volume = contract.get('volume', 0)
                if volume < 0:
                    validation_errors.append(f"Negative volume: {volume}")
            
            # Log validation results
            if validation_errors:
                self.logger.error(f"Options validation failed for {symbol}: {validation_errors[:5]}")
                self.redis.setex(f'monitoring:data:quality:options:{symbol}', 60, json.dumps({
                    'status': 'invalid',
                    'errors': validation_errors[:10],  # Store first 10 errors
                    'total_errors': len(validation_errors),
                    'timestamp': time.time()
                }))
                
                # Track validation failures
                for error in validation_errors[:5]:
                    self.validation_failures.append((symbol, error, time.time()))
                
                return False
            
            # Verify strike spacing is reasonable
            strikes = sorted(set(c['strike'] for c in chain if c.get('strike')))
            if len(strikes) > 1:
                spacings = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
                min_spacing = min(spacings)
                max_spacing = max(spacings)
                
                # Check for irregular spacing (> 10x difference)
                if max_spacing > min_spacing * 10:
                    self.logger.warning(f"Irregular strike spacing for {symbol}: min={min_spacing:.2f}, max={max_spacing:.2f}")
            
            self.redis.setex(f'monitoring:data:quality:options:{symbol}', 60, json.dumps({
                'status': 'valid',
                'contracts': len(chain),
                'timestamp': time.time()
            }))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating options data for {symbol}: {e}")
            return False