#!/usr/bin/env python3
"""
IBKR Ingestion Module - Institutional Grade
Handles Level 2 market depth, trades, bars with complete market microstructure
Based on complete_tech_spec.md Section 3.1 with institutional enhancements
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

import redis
import nest_asyncio
from ib_insync import IB, Stock, Future, MarketOrder, LimitOrder, util
import orjson
import numpy as np
from sortedcontainers import SortedDict

# Enable nested asyncio for Jupyter/interactive environments
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IBKRIngestion:
    """Institutional-grade IBKR data ingestion with complete market microstructure"""
    
    def __init__(self, config: Dict[str, Any], redis_client: redis.Redis):
        """Initialize IBKR ingestion with configuration from config.yaml"""
        
        self.config = config
        self.redis = redis_client
        self.ib = IB()
        
        # Connection settings from config
        self.primary_gateway = {
            'host': config['ibkr']['host'],
            'port': config['ibkr']['port'],
            'client_id': config['ibkr']['client_id']
        }
        
        # Backup gateway (port + 1 for redundancy)
        self.backup_gateway = {
            'host': config['ibkr']['host'],
            'port': config['ibkr']['port'] + 1,
            'client_id': config['ibkr']['client_id'] + 1
        }
        
        # Trading symbols from config
        self.symbols = config['trading']['symbols']
        self.contracts = {}
        
        # Futures contracts for correlation tracking
        self.futures_symbols = ['ES', 'NQ', 'RTY', 'VX', 'DX', 'GC', 'CL', 'ZB']
        self.futures_contracts = {}
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.audit_sequence = 0
        self.message_log = deque(maxlen=100000)  # Circular buffer for audit trail
        
        # Connection state
        self.connected = False
        self.using_backup = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1  # Exponential backoff
        
        # Initialize subcomponents (imported separately for modularity)
        self._initialize_components()
        
        # Performance monitoring
        self.metrics = {
            'messages_processed': 0,
            'messages_per_second': 0,
            'last_message_time': time.time_ns(),
            'latency_percentiles': {'p50': 0, 'p95': 0, 'p99': 0}
        }
        
        # Message rate tracking
        self.message_timestamps = deque(maxlen=1000)
        
    def _initialize_components(self):
        """Initialize all market microstructure tracking components"""
        
        # Import components
        from .market_microstructure import MarketMicrostructure
        from .exchange_handler import ExchangeHandler
        from .auction_processor import AuctionProcessor
        from .halt_manager import HaltManager
        from .timestamp_tracker import TimestampTracker
        from .mm_detector import OptionsMMDetector
        from .hidden_order_detector import HiddenOrderDetector
        from .trade_classifier import TradeClassifier
        
        # Initialize each component with config
        self.microstructure = MarketMicrostructure(self.config, self.redis)
        self.exchange_handler = ExchangeHandler(self.config, self.redis)
        self.auction_processor = AuctionProcessor(self.config, self.redis)
        self.halt_manager = HaltManager(self.config, self.redis)
        self.timestamp_tracker = TimestampTracker(self.config, self.redis)
        self.mm_detector = OptionsMMDetector(self.config, self.redis)
        self.hidden_order_detector = HiddenOrderDetector(self.config, self.redis)
        self.trade_classifier = TradeClassifier(self.config, self.redis)
        
        logger.info(f"Initialized all market microstructure components for session {self.session_id}")
        
    async def start(self):
        """Start IBKR ingestion with failover support"""
        
        logger.info(f"Starting IBKR Ingestion - Session: {self.session_id}")
        
        # Connect with failover
        await self._connect_with_failover()
        
        # Subscribe to all symbols
        await self._subscribe_all_symbols()
        
        # Start monitoring tasks
        monitoring_task = asyncio.create_task(self._monitor_connection())
        metrics_task = asyncio.create_task(self._update_metrics())
        
        # Main event loop
        try:
            while True:
                await asyncio.sleep(0.01)  # 10ms processing interval
                
                # Process any pending messages
                self.ib.sleep(0)
                
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        finally:
            await self._cleanup()
            
    async def _connect_with_failover(self):
        """Connect to IBKR with automatic failover to backup gateway"""
        
        # Try primary gateway first
        try:
            gateway = self.primary_gateway
            logger.info(f"Connecting to primary gateway: {gateway['host']}:{gateway['port']}")
            
            await self.ib.connectAsync(
                gateway['host'],
                gateway['port'],
                clientId=gateway['client_id'],
                timeout=20
            )
            
            self.connected = True
            self.using_backup = False
            logger.info("Connected to primary IBKR gateway")
            
            # Store gateway info for audit
            self.redis.hset('audit:session', self.session_id, json.dumps({
                'gateway': 'primary',
                'host': gateway['host'],
                'port': gateway['port'],
                'start_time': datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.warning(f"Primary gateway connection failed: {e}")
            
            # Try backup gateway
            try:
                gateway = self.backup_gateway
                logger.info(f"Connecting to backup gateway: {gateway['host']}:{gateway['port']}")
                
                await self.ib.connectAsync(
                    gateway['host'],
                    gateway['port'],
                    clientId=gateway['client_id'],
                    timeout=20
                )
                
                self.connected = True
                self.using_backup = True
                logger.info("Connected to backup IBKR gateway")
                
                # Store gateway info
                self.redis.hset('audit:session', self.session_id, json.dumps({
                    'gateway': 'backup',
                    'host': gateway['host'],
                    'port': gateway['port'],
                    'start_time': datetime.now().isoformat()
                }))
                
            except Exception as backup_error:
                logger.error(f"Both gateways failed: Primary: {e}, Backup: {backup_error}")
                raise ConnectionError("Unable to connect to any IBKR gateway")
                
        # Set up event handlers
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Set up all IBKR event handlers"""
        
        # Market depth handlers
        self.ib.updateMktDepthEvent += self._on_depth_update
        self.ib.updateMktDepthL2Event += self._on_depth_l2_update
        
        # Trade handlers
        self.ib.pendingTickersEvent += self._on_ticker_update
        self.ib.execDetailsEvent += self._on_execution
        
        # Bar data handlers
        self.ib.barUpdateEvent += self._on_bar_update
        
        # Error and system handlers
        self.ib.errorEvent += self._on_error
        self.ib.disconnectedEvent += self._on_disconnect
        
        # Time and sales
        self.ib.tickByTickAllLastEvent += self._on_tick_by_tick
        
        logger.info("Event handlers configured")
        
    async def _subscribe_all_symbols(self):
        """Subscribe to all symbols for Level 2, trades, and bars"""
        
        # Subscribe to equity symbols
        for symbol in self.symbols:
            await self._subscribe_symbol(symbol, is_future=False)
            
        # Subscribe to futures for correlation
        for symbol in self.futures_symbols:
            await self._subscribe_symbol(symbol, is_future=True)
            
        logger.info(f"Subscribed to {len(self.symbols)} equities and {len(self.futures_symbols)} futures")
        
    async def _create_front_month_future(self, symbol: str):
        """Create front month futures contract dynamically"""
        
        from datetime import datetime, timedelta
        import calendar
        
        # Determine exchange based on symbol
        exchange_map = {
            'ES': 'CME',   # E-mini S&P 500
            'NQ': 'CME',   # E-mini Nasdaq
            'RTY': 'CME',  # E-mini Russell
            'VX': 'CFE',   # VIX futures
            'DX': 'NYBOT', # Dollar Index
            'GC': 'COMEX', # Gold
            'CL': 'NYMEX', # Crude Oil
            'ZB': 'CBOT'   # 30-year Treasury Bond
        }
        
        exchange = exchange_map.get(symbol, 'CME')
        
        # Calculate front month expiry
        now = datetime.now()
        
        # Futures expiry rules (simplified - would need full calendar for production)
        if symbol in ['ES', 'NQ', 'RTY']:  # Equity index futures
            # Quarterly contracts (Mar, Jun, Sep, Dec)
            # Third Friday of expiry month
            quarter_months = [3, 6, 9, 12]
            
            # Find next quarterly month
            current_month = now.month
            expiry_month = next((m for m in quarter_months if m >= current_month), quarter_months[0])
            expiry_year = now.year if expiry_month >= current_month else now.year + 1
            
        elif symbol == 'VX':  # VIX futures
            # Monthly contracts, expire on Wednesday
            # 30 days before the third Friday of the following month
            expiry_month = now.month + 1 if now.day < 15 else now.month + 2
            if expiry_month > 12:
                expiry_month -= 12
                expiry_year = now.year + 1
            else:
                expiry_year = now.year
                
        elif symbol in ['GC', 'CL']:  # Commodities
            # Monthly contracts
            expiry_month = now.month + 1 if now.day < 25 else now.month + 2
            if expiry_month > 12:
                expiry_month -= 12
                expiry_year = now.year + 1
            else:
                expiry_year = now.year
                
        else:
            # Default to next month
            expiry_month = now.month + 1
            expiry_year = now.year
            if expiry_month > 12:
                expiry_month = 1
                expiry_year += 1
        
        # Format expiry as YYYYMM
        expiry = f"{expiry_year}{expiry_month:02d}"
        
        # Create future contract
        contract = Future(symbol, exchange, expiry)
        
        logger.info(f"Created {symbol} futures contract for {expiry} on {exchange}")
        
        return contract
        
    async def _subscribe_symbol(self, symbol: str, is_future: bool = False):
        """Subscribe to a single symbol with all data types"""
        
        try:
            if is_future:
                # Create futures contract with dynamic front month
                contract = await self._create_front_month_future(symbol)
                self.futures_contracts[symbol] = contract
            else:
                # Create stock contract
                contract = Stock(symbol, 'SMART', 'USD')
                self.contracts[symbol] = contract
                
            # Qualify the contract
            await self.ib.qualifyContractsAsync(contract)
            
            # Request Level 2 market depth (10 levels)
            self.ib.reqMktDepthExchanges(contract)
            self.ib.reqMktDepth(
                contract,
                numRows=10,
                isSmartDepth=True,
                mktDepthOptions=[]
            )
            
            # Request market data with all generic ticks
            # Generic tick list includes auction data (233), LULD bands (232), etc.
            self.ib.reqMktData(
                contract,
                genericTickList='232,233,236,258,293,294,295,318',
                snapshot=False,
                regulatorySnapshot=False,
                mktDataOptions=[]
            )
            
            # Request 5-second bars
            self.ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow='TRADES',
                useRTH=False,
                realTimeBarsOptions=[]
            )
            
            # Request tick-by-tick data for precise trade tracking
            self.ib.reqTickByTickData(
                contract,
                tickType='AllLast',
                numberOfTicks=0,
                ignoreSize=False
            )
            
            logger.info(f"Subscribed to {symbol} ({'Future' if is_future else 'Stock'})")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            
    def _on_depth_update(self, depth):
        """Process Level 2 depth updates"""
        
        try:
            # Track message for audit
            self._track_message('depth', depth)
            
            # Get symbol
            symbol = depth.contract.symbol
            
            # Track timestamps
            exchange_time = depth.time if hasattr(depth, 'time') else None
            gateway_time = time.time_ns()
            
            # Process through exchange handler for multi-venue tracking
            self.exchange_handler.process_depth_update(symbol, depth, exchange_time, gateway_time)
            
            # Update metrics
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing depth update: {e}")
            
    def _on_depth_l2_update(self, depth):
        """Process Level 2 depth updates with exchange information"""
        
        try:
            # Track message
            self._track_message('depth_l2', depth)
            
            symbol = depth.contract.symbol
            
            # Process with exchange-specific handling
            self.exchange_handler.process_l2_update(symbol, depth)
            
            # Check for hidden orders
            self.hidden_order_detector.analyze_depth_update(symbol, depth)
            
            # Detect MM patterns
            self.mm_detector.analyze_market_maker(symbol, depth)
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing L2 depth: {e}")
            
    def _on_ticker_update(self, tickers):
        """Process ticker updates (trades and quotes)"""
        
        for ticker in tickers:
            try:
                if not ticker.contract:
                    continue
                    
                symbol = ticker.contract.symbol
                
                # Track timestamps
                exchange_time = ticker.time if hasattr(ticker, 'time') else None
                gateway_time = time.time_ns()
                
                # Process last trade
                if ticker.last and ticker.lastSize:
                    trade_data = {
                        'price': ticker.last,
                        'size': ticker.lastSize,
                        'time': exchange_time,
                        'bid': ticker.bid,
                        'ask': ticker.ask,
                        'bidSize': ticker.bidSize,
                        'askSize': ticker.askSize
                    }
                    
                    # Classify trade
                    classified_trade = self.trade_classifier.classify_trade(symbol, trade_data)
                    
                    # Store in Redis with classification
                    self._store_trade(symbol, classified_trade)
                    
                # Process auction data if available
                if hasattr(ticker, 'auctionVolume') and ticker.auctionVolume:
                    self.auction_processor.process_auction_update(symbol, ticker)
                    
                # Check for halt conditions
                if hasattr(ticker, 'halted') and ticker.halted:
                    self.halt_manager.process_halt(symbol, ticker)
                    
                self.metrics['messages_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing ticker: {e}")
                
    def _on_bar_update(self, bars, hasNewBar):
        """Process 5-second bar updates"""
        
        if not hasNewBar:
            return
            
        for bar in bars:
            try:
                symbol = bar.contract.symbol
                
                bar_data = {
                    'time': bar.time.timestamp() if hasattr(bar.time, 'timestamp') else time.time(),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'wap': bar.wap if hasattr(bar, 'wap') else None,
                    'count': bar.count if hasattr(bar, 'count') else None
                }
                
                # Calculate microstructure metrics from bar
                self.microstructure.process_bar(symbol, bar_data)
                
                # Store bar data
                self._store_bar(symbol, bar_data)
                
                self.metrics['messages_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing bar: {e}")
                
    def _on_tick_by_tick(self, ticker, tickType, time_, price, size, tickAttribLast, exchange, specialConditions):
        """Process tick-by-tick trade data with full detail"""
        
        try:
            symbol = ticker.contract.symbol if ticker else 'UNKNOWN'
            
            # Create detailed trade record
            trade_detail = {
                'symbol': symbol,
                'time': time_,
                'price': price,
                'size': size,
                'exchange': exchange,
                'conditions': specialConditions,
                'past_limit': tickAttribLast.pastLimit if tickAttribLast else False,
                'unreported': tickAttribLast.unreported if tickAttribLast else False
            }
            
            # Process through trade classifier for condition codes
            classified = self.trade_classifier.classify_tick_trade(trade_detail)
            
            # Check for sweeps
            if classified.get('is_sweep'):
                self._process_sweep(symbol, classified)
                
            # Store classified trade
            self._store_tick_trade(symbol, classified)
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing tick trade: {e}")
            
    def _on_execution(self, trade, fill):
        """Process execution details for trades"""
        
        try:
            # This would handle our own executions if we were trading
            # For now, just track for monitoring
            
            execution_data = {
                'symbol': trade.contract.symbol,
                'side': trade.order.action,
                'qty': fill.execution.shares,
                'price': fill.execution.price,
                'time': fill.execution.time,
                'exchange': fill.execution.exchange,
                'exec_id': fill.execution.execId
            }
            
            logger.info(f"Execution: {execution_data}")
            
        except Exception as e:
            logger.error(f"Error processing execution: {e}")
            
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IBKR errors with specific recovery actions"""
        
        # Critical errors that need immediate action
        critical_errors = {
            504: "Gateway timeout",
            1100: "Connection lost",
            1300: "Socket port reset",
            2110: "Connection broken"
        }
        
        # Data errors that affect quality
        data_errors = {
            2103: "Market data farm broken",
            2104: "Market data farm OK",
            2106: "Historical data farm broken",
            2107: "Historical data farm OK"
        }
        
        if errorCode in critical_errors:
            logger.critical(f"Critical error {errorCode}: {errorString}")
            asyncio.create_task(self._handle_critical_error(errorCode))
            
        elif errorCode in data_errors:
            logger.warning(f"Data error {errorCode}: {errorString}")
            self._handle_data_error(errorCode, contract)
            
        else:
            logger.info(f"Error {errorCode}: {errorString}")
            
    async def _handle_critical_error(self, error_code: int):
        """Handle critical errors with reconnection"""
        
        if error_code in [504, 1100, 2110]:  # Connection issues
            logger.info("Attempting reconnection due to critical error")
            await self._reconnect()
            
    def _handle_data_error(self, error_code: int, contract):
        """Handle data quality errors"""
        
        if error_code == 2103:  # Market data farm broken
            # Mark data as potentially stale
            if contract:
                symbol = contract.symbol
                self.redis.hset(f'market:{symbol}:health', 'data_quality', 'degraded')
                
        elif error_code == 2104:  # Market data farm OK
            # Mark data as healthy
            if contract:
                symbol = contract.symbol
                self.redis.hset(f'market:{symbol}:health', 'data_quality', 'healthy')
                
    def _on_disconnect(self):
        """Handle disconnection from IBKR"""
        
        logger.warning("Disconnected from IBKR gateway")
        self.connected = False
        
        # Set system status
        self.redis.hset('system:status', 'ibkr_connected', 'false')
        
        # Attempt reconnection
        asyncio.create_task(self._reconnect())
        
    async def _reconnect(self):
        """Reconnect with exponential backoff"""
        
        while not self.connected and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            
            logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            
            try:
                # Disconnect if still partially connected
                if self.ib.isConnected():
                    self.ib.disconnect()
                    
                # Wait with exponential backoff
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 30)  # Max 30 seconds
                
                # Try to reconnect
                await self._connect_with_failover()
                
                if self.connected:
                    # Resubscribe to all symbols
                    await self._subscribe_all_symbols()
                    
                    # Reset reconnection parameters
                    self.reconnect_attempts = 0
                    self.reconnect_delay = 1
                    
                    logger.info("Successfully reconnected")
                    break
                    
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
                
        if not self.connected:
            logger.critical("Failed to reconnect after maximum attempts")
            self.redis.hset('system:status', 'ibkr_connected', 'failed')
            
    def _store_trade(self, symbol: str, trade_data: Dict):
        """Store classified trade in Redis"""
        
        pipe = self.redis.pipeline()
        
        # Store in classified trades list
        trade_json = orjson.dumps(trade_data).decode('utf-8')
        pipe.lpush(f'market:{symbol}:trades:classified', trade_json)
        pipe.ltrim(f'market:{symbol}:trades:classified', 0, 9999)  # Keep last 10000
        pipe.expire(f'market:{symbol}:trades:classified', 300)  # 5 minute TTL for audit
        
        # Store last price
        pipe.setex(f'market:{symbol}:last', 5, trade_data['price'])  # 5 second TTL
        
        # Store by condition type if special
        if trade_data.get('is_sweep'):
            pipe.lpush(f'market:{symbol}:trades:sweeps', trade_json)
            pipe.ltrim(f'market:{symbol}:trades:sweeps', 0, 999)  # Keep last 1000
            pipe.expire(f'market:{symbol}:trades:sweeps', 300)  # 5 minute TTL
            
        if trade_data.get('is_block'):
            pipe.lpush(f'market:{symbol}:trades:blocks', trade_json)
            pipe.ltrim(f'market:{symbol}:trades:blocks', 0, 999)  # Keep last 1000
            pipe.expire(f'market:{symbol}:trades:blocks', 300)  # 5 minute TTL
            
        if trade_data.get('is_dark'):
            pipe.lpush(f'market:{symbol}:trades:dark', trade_json)
            pipe.ltrim(f'market:{symbol}:trades:dark', 0, 999)  # Keep last 1000
            pipe.expire(f'market:{symbol}:trades:dark', 300)  # 5 minute TTL
            
        pipe.execute()
        
    def _store_tick_trade(self, symbol: str, trade_data: Dict):
        """Store tick-by-tick trade data"""
        
        # Store with nanosecond precision
        trade_data['local_time_ns'] = time.time_ns()
        
        pipe = self.redis.pipeline()
        
        trade_json = orjson.dumps(trade_data).decode('utf-8')
        pipe.lpush(f'market:{symbol}:trades:tick', trade_json)
        pipe.ltrim(f'market:{symbol}:trades:tick', 0, 4999)  # Keep last 5000 ticks
        pipe.expire(f'market:{symbol}:trades:tick', 5)
        
        pipe.execute()
        
    def _store_bar(self, symbol: str, bar_data: Dict):
        """Store bar data in Redis"""
        
        # Get existing bars
        bars_json = self.redis.get(f'market:{symbol}:bars')
        if bars_json:
            bars = orjson.loads(bars_json)
        else:
            bars = []
            
        # Add new bar and keep last 100
        bars.append(bar_data)
        bars = bars[-100:]
        
        # Store updated bars
        self.redis.setex(
            f'market:{symbol}:bars',
            10,
            orjson.dumps(bars).decode('utf-8')
        )
        
    def _process_sweep(self, symbol: str, trade_data: Dict):
        """Process detected sweep order"""
        
        sweep_data = {
            'symbol': symbol,
            'price': trade_data['price'],
            'size': trade_data['size'],
            'time': trade_data['time'],
            'exchanges': trade_data.get('exchanges', []),
            'aggressor': 'buy' if trade_data['price'] >= trade_data.get('ask', trade_data['price']) else 'sell'
        }
        
        # Store sweep alert
        self.redis.setex(
            f'market:{symbol}:sweep:latest',
            30,
            orjson.dumps(sweep_data).decode('utf-8')
        )
        
        logger.info(f"Sweep detected on {symbol}: {sweep_data}")
        
    def _track_message(self, msg_type: str, message: Any):
        """Track message for audit trail"""
        
        self.audit_sequence += 1
        
        audit_entry = {
            'sequence': self.audit_sequence,
            'type': msg_type,
            'time': time.time_ns(),
            'session': self.session_id
        }
        
        # Add to circular buffer
        self.message_log.append(audit_entry)
        
        # Track message rate
        self.message_timestamps.append(time.time())
        
        # Periodically store audit info
        if self.audit_sequence % 1000 == 0:
            self.redis.hset('audit:sequence', self.session_id, self.audit_sequence)
            
    async def _monitor_connection(self):
        """Monitor connection health"""
        
        while True:
            try:
                await asyncio.sleep(5)
                
                if self.connected:
                    # Check connection is alive
                    if not self.ib.isConnected():
                        logger.warning("Connection lost, triggering reconnect")
                        self.connected = False
                        await self._reconnect()
                        
                    # Update health status
                    health_data = {
                        'connected': self.connected,
                        'using_backup': self.using_backup,
                        'messages_per_second': self._calculate_message_rate(),
                        'last_message': time.time() - (self.metrics['last_message_time'] / 1e9)
                    }
                    
                    self.redis.setex(
                        'system:health:ibkr',
                        10,
                        json.dumps(health_data)
                    )
                    
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                
    async def _update_metrics(self):
        """Update performance metrics"""
        
        while True:
            try:
                await asyncio.sleep(1)
                
                # Calculate message rate
                self.metrics['messages_per_second'] = self._calculate_message_rate()
                
                # Store metrics in Redis
                self.redis.setex(
                    'metrics:ibkr:performance',
                    5,
                    json.dumps(self.metrics)
                )
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                
    def _calculate_message_rate(self) -> float:
        """Calculate messages per second"""
        
        if len(self.message_timestamps) < 2:
            return 0.0
            
        now = time.time()
        one_second_ago = now - 1.0
        
        recent_messages = sum(1 for ts in self.message_timestamps if ts > one_second_ago)
        
        return float(recent_messages)
        
    async def _cleanup(self):
        """Clean up resources on shutdown"""
        
        logger.info("Cleaning up IBKR connection")
        
        try:
            # Cancel all subscriptions
            for contract in list(self.contracts.values()) + list(self.futures_contracts.values()):
                self.ib.cancelMktDepth(contract)
                self.ib.cancelMktData(contract)
                self.ib.cancelRealTimeBars(contract)
                self.ib.cancelTickByTickData(contract)
                
            # Disconnect
            if self.ib.isConnected():
                self.ib.disconnect()
                
            # Update status
            self.redis.hset('system:status', 'ibkr_connected', 'shutdown')
            
            # Store final audit info
            self.redis.hset('audit:sessions', self.session_id, json.dumps({
                'end_time': datetime.now().isoformat(),
                'total_messages': self.audit_sequence,
                'clean_shutdown': True
            }))
            
            logger.info("Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")