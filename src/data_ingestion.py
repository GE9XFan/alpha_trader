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
                # Just use asyncio.sleep for the main loop
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
                pipe.setex(f'market:{symbol}:book', 1, json.dumps(book_data))
                
                # Store metrics
                pipe.setex(f'market:{symbol}:imbalance', 1, metrics['imbalance'])
                pipe.setex(f'market:{symbol}:spread', 1, metrics['spread'])
                pipe.setex(f'market:{symbol}:mid', 1, metrics['mid'])
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
        pipe.setex(f'market:{symbol}:trades', 1, json.dumps(trades_list))
        
        # Store additional ticker data
        ticker_data = {
            'bid': trade['bid'],
            'ask': trade['ask'],
            'volume': trade['volume'],
            'vwap': trade['vwap'],
            'timestamp': trade['time']
        }
        pipe.setex(f'market:{symbol}:ticker', 1, json.dumps(ticker_data))
        
        # Calculate and store spread
        if trade['bid'] and trade['ask']:
            spread = round(trade['ask'] - trade['bid'], 4)
            pipe.setex(f'market:{symbol}:spread', 1, spread)
        
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
            
            # Build bar object
            bar_data = {
                'time': int(bar.time.timestamp()),
                'open': float(bar.open),
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
            pipe.setex(f'market:{symbol}:bars', 10, json.dumps(bars_list))
            
            # Store latest bar separately for quick access
            pipe.setex(f'market:{symbol}:latest_bar', 10, json.dumps(bar_data))
            
            # Store bar metrics
            pipe.setex(f'market:{symbol}:bar_metrics', 10, json.dumps(metrics))
            
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
                # Calculate metrics
                metrics = {
                    'connected': self.connected,
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
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize Alpha Vantage ingestion with configuration.
        
        TODO: Load API key from config
        TODO: Initialize Redis connection
        TODO: Set up rate limiting tracker (600 calls/min)
        TODO: Load symbols from config
        """
        self.config = config  # Loaded from config.yaml
        self.redis = redis_conn
        self.api_key = config['alpha_vantage']['api_key']  # Must be in config
        self.base_url = 'https://www.alphavantage.co/query'
        
        # Symbols from config
        self.symbols = config.get('symbols', ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'PLTR', 'VXX'])
        
        # Rate limiting: track last 600 API calls
        self.call_times = deque(maxlen=600)
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """
        Start Alpha Vantage data collection loop.
        
        TODO: Create aiohttp session for async requests
        TODO: Loop through symbols continuously
        TODO: Fetch options chain with Greeks for each symbol
        TODO: Fetch sentiment data
        TODO: Implement rate limiting (600 calls/minute)
        TODO: Handle API errors and retries
        TODO: Update cycle every 10 seconds
        """
        self.logger.info("Starting Alpha Vantage data ingestion...")
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Fetch data for each symbol
                # TODO: Call fetch_symbol_data for each symbol
                # TODO: Check rate limits before each call
                
                await asyncio.sleep(10)  # Full cycle every 10 seconds
    
    async def rate_limit_check(self):
        """
        Enforce Alpha Vantage rate limiting (600 calls/minute).
        
        TODO: Track call timestamps in deque
        TODO: Count calls in last 60 seconds
        TODO: Sleep if approaching limit (590 calls safety buffer)
        TODO: Log rate limit status to Redis for monitoring
        
        Redis keys to update:
        - monitoring:api:av:calls - Current API usage count
        """
        pass
    
    async def fetch_symbol_data(self, session: aiohttp.ClientSession, symbol: str):
        """
        Fetch all Alpha Vantage data for a symbol.
        
        TODO: Fetch REALTIME_OPTIONS endpoint for options chain
        TODO: Parse options chain and extract Greeks (provided by AV)
        TODO: Calculate gamma exposure (GEX) from Greeks
        TODO: Detect unusual options activity (volume > 2x OI)
        TODO: Fetch NEWS_SENTIMENT for sentiment scores
        TODO: Store all data in Redis with appropriate TTLs
        
        Redis keys to update:
        - options:{symbol}:chain - Full options chain (10 sec TTL)
        - options:{symbol}:greeks - Greeks by strike/expiry (10 sec TTL)
        - options:{symbol}:unusual - Unusual activity detected (10 sec TTL)
        - sentiment:{symbol}:score - Sentiment analysis (300 sec TTL)
        """
        pass
    
    async def fetch_options_chain(self, session: aiohttp.ClientSession, symbol: str):
        """
        Fetch options chain with Greeks from Alpha Vantage.
        
        TODO: Build API URL with REALTIME_OPTIONS function
        TODO: Make async HTTP request
        TODO: Parse JSON response
        TODO: Extract contracts with strike, expiry, type (call/put)
        TODO: Extract Greeks: delta, gamma, theta, vega, rho, IV
        TODO: Extract open interest and volume
        TODO: Return structured options data
        
        Note: Greeks are PROVIDED by Alpha Vantage, not calculated
        """
        pass
    
    async def fetch_sentiment(self, session: aiohttp.ClientSession, symbol: str):
        """
        Fetch news sentiment for a symbol.
        
        TODO: Build API URL with NEWS_SENTIMENT function
        TODO: Make async HTTP request
        TODO: Parse sentiment scores from articles
        TODO: Calculate aggregate sentiment score
        TODO: Store in Redis with 5-minute TTL
        
        Redis keys to update:
        - sentiment:{symbol}:score - JSON with score, count, timestamp
        """
        pass
    
    def detect_unusual_activity(self, options_chain: list) -> dict:
        """
        Detect unusual options activity from chain data.
        
        TODO: Calculate volume/open_interest ratio for each contract
        TODO: Flag contracts where volume > 2x open interest
        TODO: Sort by ratio (highest first)
        TODO: Return top 10 unusual contracts
        TODO: Include strike, expiry, type, volume, OI, ratio
        
        Returns:
            Dictionary with 'detected' boolean and 'contracts' list
        """
        pass
    
    async def fetch_technical_indicators(self, session: aiohttp.ClientSession, symbol: str):
        """
        Fetch technical indicators from Alpha Vantage.
        
        TODO: Fetch RSI indicator
        TODO: Fetch MACD indicator
        TODO: Fetch Bollinger Bands
        TODO: Store in Redis: technicals:{symbol}:rsi, etc.
        
        Redis keys to update:
        - technicals:{symbol}:rsi
        - technicals:{symbol}:macd
        - technicals:{symbol}:bbands
        """
        pass
    
    async def handle_api_error(self, response: aiohttp.ClientResponse, symbol: str):
        """
        Handle Alpha Vantage API errors and implement retry logic.
        
        TODO: Check response status code
        TODO: Handle rate limit errors (wait and retry)
        TODO: Handle invalid API key
        TODO: Handle symbol not found
        TODO: Log errors to Redis for monitoring
        TODO: Implement exponential backoff for retries
        """
        pass


class DataQualityMonitor:
    """
    Monitor data quality and freshness from all sources.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize data quality monitoring.
        
        TODO: Set up Redis connection
        TODO: Define freshness thresholds from config
        TODO: Initialize monitoring metrics
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
    
    async def monitor_data_freshness(self):
        """
        Check data freshness for all symbols and sources.
        
        TODO: Check timestamps for IBKR data (should be < 1 second old)
        TODO: Check timestamps for Alpha Vantage data (should be < 10 seconds old)
        TODO: Alert if data is stale
        TODO: Track data gaps
        TODO: Store metrics in Redis
        
        Redis keys to update:
        - monitoring:data:freshness:{source}:{symbol}
        - monitoring:data:gaps:{source}
        """
        pass
    
    def validate_market_data(self, symbol: str, data: dict) -> bool:
        """
        Validate market data quality.
        
        TODO: Check bid/ask spread reasonableness
        TODO: Verify price continuity
        TODO: Check for zero/null values
        TODO: Validate volume is positive
        TODO: Check timestamp is recent
        
        Returns:
            True if data passes validation
        """
        pass
    
    def validate_options_data(self, symbol: str, chain: list) -> bool:
        """
        Validate options chain data quality.
        
        TODO: Verify Greeks are within reasonable ranges
        TODO: Check IV is positive and < 500%
        TODO: Verify strikes are properly spaced
        TODO: Check expiration dates are future
        TODO: Validate open interest >= 0
        
        Returns:
            True if data passes validation
        """
        pass