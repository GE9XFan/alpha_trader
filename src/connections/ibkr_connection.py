"""IBKR TWS connection manager - Phase 3.1"""

import sys
from pathlib import Path
from threading import Thread, Event
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from src.data.ingestion import DataIngestion
from datetime import datetime


sys.path.append(str(Path(__file__).parent.parent.parent))
from src.foundation.config_manager import ConfigManager


class IBKRConnectionManager(EWrapper, EClient):
    """
    Manages connection to Interactive Brokers TWS/Gateway
    Phase 3.1: Basic connection only
    """
    
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        
        # Load configuration
        self.config = ConfigManager()
        
        # Connection parameters from .env
        self.host = self.config.ibkr_host
        self.port = int(self.config.ibkr_port)
        self.client_id = int(self.config.ibkr_client_id)
        
        # Connection state
        self.connected = False
        self.connection_event = Event()

        # Data ingestion handler
        self.ingestion = DataIngestion()
        
        # Track active subscriptions
        self.active_bars = {}  # req_id -> symbol
        self.active_quotes = {}  # req_id -> symbol

        print(f"IBKR Connection Manager initialized")
        print(f"  Host: {self.host}")
        print(f"  Port: {self.port} ({'Paper' if self.port == 7497 else 'Live'})")
        print(f"  Client ID: {self.client_id}")
    
    def error(self, reqId, errorCode, errorString):
        """Handle errors from TWS"""
        print(f"Error {errorCode}: {errorString} (Request {reqId})")
    
    def nextValidId(self, orderId):
        """Called when connection is established"""
        self.connected = True
        self.connection_event.set()
        print(f"✓ Connected to IBKR TWS! Next valid order ID: {orderId}")
    
    def connect_tws(self):
        """Connect to TWS/Gateway"""
        print(f"\nConnecting to TWS at {self.host}:{self.port}...")
        
        # Connect
        self.connect(self.host, self.port, self.client_id)
        
        # Start message processing thread
        api_thread = Thread(target=self.run, daemon=True)
        api_thread.start()
        
        # Wait for connection
        self.connection_event.wait(timeout=10)
        
        if self.connected:
            return True
        else:
            print("✗ Failed to connect to TWS")
            return False
    
    def disconnect_tws(self):
        """Disconnect from TWS"""
        if self.connected:
            self.disconnect()
            self.connected = False
            print("Disconnected from TWS")
    
    def subscribe_bars(self, symbol, bar_size='5 secs'):
        """
        Subscribe to real-time bars for a symbol
        Note: IBKR reqRealTimeBars only supports 5-second bars
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            bar_size: Ignored - IBKR only provides 5-second real-time bars
        """
        if not self.connected:
            print("Not connected to TWS")
            return None
        
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        # Request ID (use symbol hash for consistency)
        req_id = abs(hash(f"{symbol}_{bar_size}")) % 10000
        
        print(f"Subscribing to {bar_size} bars for {symbol} (req_id: {req_id})")
        
        # NOTE: reqRealTimeBars only supports 5-second bars
        # We collect these and aggregate them to larger timeframes
        self.reqRealTimeBars(
            reqId=req_id,
            contract=contract,
            barSize=5,  # Always 5 seconds - IBKR limitation
            whatToShow="TRADES",
            useRTH=False,
            realTimeBarsOptions=[]
        )
        
        self.active_bars[req_id] = symbol
        return req_id
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        """
        Callback for real-time bar data
        """
        timestamp = datetime.fromtimestamp(time)
        symbol = self.active_bars.get(reqId, f"Unknown_{reqId}")
        
        print(f"[{timestamp.strftime('%H:%M:%S')}] {symbol}: "
            f"O={open_:.2f} H={high:.2f} L={low:.2f} C={close:.2f} "
            f"V={volume} VWAP={wap:.2f}")
        
        # Always store as 5-second bars since that's what IBKR provides
        # The aggregator will create 1-min and 5-min bars
        self.ingestion.ingest_ibkr_bar(
            symbol, time, open_, high, low, close, volume, wap, count, '5sec'
        )
        
    def get_quotes(self, symbol):
            """
            Subscribe to real-time quotes (bid/ask/last)
            Phase 3.6: Track symbol for quotes
            """
            if not self.connected:
                print("Not connected to TWS")
                return None
            
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Request ID for quotes
            req_id = abs(hash(f"{symbol}_quote")) % 10000
            
            # Track this subscription
            self.active_quotes[req_id] = {
                'symbol': symbol,
                'bid': 0,
                'bid_size': 0,
                'ask': 0,
                'ask_size': 0,
                'last': 0,
                'last_size': 0,
                'volume': 0
            }
            
            print(f"Subscribing to quotes for {symbol} (req_id: {req_id})")
            
            # Request market data
            self.reqMktData(
                reqId=req_id,
                contract=contract,
                genericTickList="",
                snapshot=False,
                regulatorySnapshot=False,
                mktDataOptions=[]
            )
            
            return req_id
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Handle price updates and store to database"""
        if reqId not in self.active_quotes:
            return
            
        quote_data = self.active_quotes[reqId]
        
        if tickType == 1:  # BID
            quote_data['bid'] = price
        elif tickType == 2:  # ASK
            quote_data['ask'] = price
        elif tickType == 4:  # LAST
            quote_data['last'] = price
            # Store quote when we get a last price update
            if price > 0:
                self.ingestion.ingest_ibkr_quote(
                    quote_data['symbol'],
                    quote_data['bid'],
                    quote_data['bid_size'],
                    quote_data['ask'],
                    quote_data['ask_size'],
                    quote_data['last'],
                    quote_data['last_size'],
                    quote_data['volume']
                )
        
        tick_types = {1: "BID", 2: "ASK", 4: "LAST", 6: "HIGH", 7: "LOW", 9: "CLOSE"}
        if tickType in tick_types:
            print(f"{quote_data['symbol']}: {tick_types[tickType]}={price:.2f}")
    
    def tickSize(self, reqId, tickType, size):
        """Handle size updates"""
        if reqId not in self.active_quotes:
            return
            
        quote_data = self.active_quotes[reqId]
        
        if tickType == 0:  # BID_SIZE
            quote_data['bid_size'] = size
        elif tickType == 3:  # ASK_SIZE
            quote_data['ask_size'] = size
        elif tickType == 5:  # LAST_SIZE
            quote_data['last_size'] = size
        elif tickType == 8:  # VOLUME
            quote_data['volume'] = size
        
        size_types = {0: "BID_SIZE", 3: "ASK_SIZE", 5: "LAST_SIZE", 8: "VOLUME"}
        if tickType in size_types:
            print(f"{quote_data['symbol']}: {size_types[tickType]}={size}")
        
        if tickType in size_types:
            print(f"Quote {reqId}: {size_types[tickType]}={size}")