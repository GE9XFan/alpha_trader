"""
IBKR TWS Connection Manager
Handles all IBKR data feeds and order execution
Uses configuration from config/apis/ibkr.yaml
"""

from ib_insync import *
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, time as datetime_time
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.foundation.base_module import BaseModule
from src.foundation.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class IBKRConnectionManager(BaseModule):
    """
    Manages IBKR TWS API connection
    Provides real-time data feeds and order execution
    All configuration from config/apis/ibkr.yaml
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize IBKR connection manager from YAML configuration
        
        Args:
            config_manager: ConfigManager instance (creates new if None)
        """
        # Get configuration manager
        self.config_manager = config_manager or ConfigManager()
        
        # Load IBKR configuration from YAML
        ibkr_config = self.config_manager.get('apis.ibkr.ibkr', {})
        
        # Initialize base module
        super().__init__(ibkr_config, "IBKRConnectionManager")
        
        # Extract configuration values from YAML
        self.username = self.config_manager.get('env.IBKR_USERNAME')
        self.password = self.config_manager.get('env.IBKR_PASSWORD')
        self.account = self.config_manager.get('env.IBKR_ACCOUNT')
        
        self.trading_mode = ibkr_config.get('trading_mode', 'paper')
        self.gateway_host = ibkr_config.get('gateway_host', '127.0.0.1')
        self.gateway_port = ibkr_config.get('gateway_port', 4001)
        self.client_id = ibkr_config.get('client_id', 1)
        self.connection_timeout = ibkr_config.get('connection_timeout', 30)
        self.readonly = ibkr_config.get('readonly', False)
        self.reconnect_attempts = ibkr_config.get('reconnect_attempts', 3)
        self.reconnect_delay = ibkr_config.get('reconnect_delay', 5)
        self.max_concurrent_subscriptions = ibkr_config.get('max_concurrent_subscriptions', 50)
        
        # Data feed configuration
        self.data_feeds = ibkr_config.get('data_feeds', {})
        
        # Quotes configuration
        self.quotes_config = self.data_feeds.get('quotes', {})
        self.quotes_enabled = self.quotes_config.get('enabled', True)
        self.quote_fields = self.quotes_config.get('fields', [])
        
        # Bars configuration
        self.bars_config = self.data_feeds.get('bars', {})
        self.bars_enabled = self.bars_config.get('enabled', True)
        self.bar_sizes = self.bars_config.get('sizes', [])
        
        # MOC configuration
        self.moc_config = self.data_feeds.get('moc_imbalance', {})
        self.moc_enabled = self.moc_config.get('enabled', True)
        self.moc_start = self._parse_time(self.moc_config.get('start_time', '15:40'))
        self.moc_end = self._parse_time(self.moc_config.get('end_time', '15:55'))
        
        # IB-insync client
        self.ib = IB()
        
        # Subscription tracking
        self.bar_subscriptions = {}      # symbol -> {bar_size -> subscription}
        self.quote_subscriptions = {}    # symbol -> ticker
        self.moc_subscription = None
        
        # Callbacks
        self.bar_callbacks = []          # List of callbacks for bar updates
        self.quote_callbacks = []        # List of callbacks for quote updates
        self.moc_callbacks = []          # List of callbacks for MOC updates
        
        # Statistics
        self.stats = {
            'connection_attempts': 0,
            'bars_received': 0,
            'quotes_received': 0,
            'moc_updates': 0,
            'errors': 0,
            'last_error': None,
            'connection_time': None
        }
        
        self.logger.info(
            f"IBKRConnectionManager initialized: "
            f"Mode={self.trading_mode}, "
            f"Gateway={self.gateway_host}:{self.gateway_port}, "
            f"Max subscriptions={self.max_concurrent_subscriptions}"
        )
    
    def _parse_time(self, time_str: str) -> datetime_time:
        """Parse time string from YAML"""
        try:
            parts = time_str.split(':')
            return datetime_time(int(parts[0]), int(parts[1]))
        except:
            return datetime_time(0, 0)
    
    def initialize(self) -> bool:
        """
        Initialize IBKR connection
        Required by BaseModule
        """
        try:
            # Connect to TWS/Gateway
            connected = self.connect()
            
            if connected:
                self.logger.info("IBKR connection initialized successfully")
                
                # Set up event handlers
                self._setup_event_handlers()
                
                return True
            else:
                self.logger.error("Failed to initialize IBKR connection")
                return False
                
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            self.last_error = str(e)
            return False
    
    def connect(self) -> bool:
        """Connect to TWS/Gateway with retry logic"""
        for attempt in range(self.reconnect_attempts):
            try:
                self.stats['connection_attempts'] += 1
                
                self.logger.info(
                    f"Connecting to IBKR (attempt {attempt + 1}/{self.reconnect_attempts}): "
                    f"{self.gateway_host}:{self.gateway_port}"
                )
                
                # Connect with timeout
                self.ib.connect(
                    host=self.gateway_host,
                    port=self.gateway_port,
                    clientId=self.client_id,
                    timeout=self.connection_timeout,
                    readonly=self.readonly
                )
                
                # Verify connection
                if self.ib.isConnected():
                    self.is_connected = True
                    self.stats['connection_time'] = datetime.now()
                    
                    # Log account info
                    if self.account:
                        self.logger.info(f"Connected to account: {self.account}")
                    
                    self.logger.info(
                        f"✅ IBKR connection established: {self.trading_mode} mode"
                    )
                    
                    return True
                    
            except Exception as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < self.reconnect_attempts - 1:
                    self.logger.info(f"Waiting {self.reconnect_delay} seconds before retry...")
                    self.ib.sleep(self.reconnect_delay)
                else:
                    self.logger.error("All connection attempts failed")
                    self.stats['errors'] += 1
                    self.stats['last_error'] = str(e)
        
        self.is_connected = False
        return False
    
    def _setup_event_handlers(self) -> None:
        """Set up IB event handlers"""
        # Connection events
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        
        # Data events are set up per subscription
    
    def _on_connected(self) -> None:
        """Handle connection event"""
        self.logger.info("IBKR connected event received")
        self.is_connected = True
    
    def _on_disconnected(self) -> None:
        """Handle disconnection event"""
        self.logger.warning("IBKR disconnected event received")
        self.is_connected = False
        
        # TODO: Implement auto-reconnect logic if needed
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Handle error events"""
        self.logger.error(f"IBKR Error: {errorCode} - {errorString}")
        self.stats['errors'] += 1
        self.stats['last_error'] = f"{errorCode}: {errorString}"
    
    def subscribe_bars(self, symbol: str, bar_size: str = '1 min') -> bool:
        """
        Subscribe to real-time bars
        
        Args:
            symbol: Trading symbol
            bar_size: From config: '1 secs', '5 secs', '1 min', '5 mins', etc.
        """
        if not self.bars_enabled:
            self.logger.warning("Bars data feed is disabled in configuration")
            return False
        
        if bar_size not in self.bar_sizes:
            self.logger.error(f"Bar size '{bar_size}' not in configured sizes: {self.bar_sizes}")
            return False
        
        if len(self.bar_subscriptions) >= self.max_concurrent_subscriptions:
            self.logger.error(f"Max subscriptions reached: {self.max_concurrent_subscriptions}")
            return False
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Determine bar size for IB API
            if '1 min' in bar_size:
                ib_bar_size = 60
            elif '5 min' in bar_size:
                ib_bar_size = 300
            elif '5 sec' in bar_size:
                ib_bar_size = 5
            elif '1 sec' in bar_size:
                ib_bar_size = 1
            else:
                ib_bar_size = 60  # Default to 1 minute
            
            # Request real-time bars
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=ib_bar_size,
                whatToShow='TRADES',
                useRTH=False,  # Include extended hours
                realTimeBarsOptions=[]
            )
            
            # Store subscription
            if symbol not in self.bar_subscriptions:
                self.bar_subscriptions[symbol] = {}
            self.bar_subscriptions[symbol][bar_size] = bars
            
            # Set up callback
            bars.updateEvent += lambda bars, hasNewBar: self._on_bar_update(
                symbol, bar_size, bars, hasNewBar
            )
            
            self.logger.info(f"Subscribed to {bar_size} bars for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to bars for {symbol}: {e}")
            self.stats['errors'] += 1
            return False
    
    def _on_bar_update(self, symbol: str, bar_size: str, bars, hasNewBar: bool) -> None:
        """Handle new bar data"""
        if hasNewBar:
            latest_bar = bars[-1]
            self.stats['bars_received'] += 1
            
            self.logger.debug(
                f"New {bar_size} bar for {symbol}: "
                f"O={latest_bar.open} H={latest_bar.high} "
                f"L={latest_bar.low} C={latest_bar.close} V={latest_bar.volume}"
            )
            
            # Notify callbacks
            for callback in self.bar_callbacks:
                try:
                    callback(symbol, bar_size, latest_bar)
                except Exception as e:
                    self.logger.error(f"Bar callback error: {e}")
    
    def subscribe_quotes(self, symbol: str) -> bool:
        """Subscribe to real-time quotes"""
        if not self.quotes_enabled:
            self.logger.warning("Quotes data feed is disabled in configuration")
            return False
        
        if len(self.quote_subscriptions) >= self.max_concurrent_subscriptions:
            self.logger.error(f"Max subscriptions reached: {self.max_concurrent_subscriptions}")
            return False
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market data
            ticker = self.ib.reqMktData(
                contract,
                genericTickList='',
                snapshot=False,
                regulatorySnapshot=False
            )
            
            # Store subscription
            self.quote_subscriptions[symbol] = ticker
            
            # Set up callback
            ticker.updateEvent += lambda ticker: self._on_quote_update(symbol, ticker)
            
            self.logger.info(f"Subscribed to quotes for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to quotes for {symbol}: {e}")
            self.stats['errors'] += 1
            return False
    
    def _on_quote_update(self, symbol: str, ticker) -> None:
        """Handle quote updates"""
        self.stats['quotes_received'] += 1
        
        # Extract quote data based on configured fields
        quote_data = {}
        for field in self.quote_fields:
            if field == 'bid':
                quote_data['bid'] = ticker.bid
            elif field == 'ask':
                quote_data['ask'] = ticker.ask
            elif field == 'last':
                quote_data['last'] = ticker.last
            elif field == 'bid_size':
                quote_data['bid_size'] = ticker.bidSize
            elif field == 'ask_size':
                quote_data['ask_size'] = ticker.askSize
            elif field == 'volume':
                quote_data['volume'] = ticker.volume
        
        self.logger.debug(f"Quote update for {symbol}: {quote_data}")
        
        # Notify callbacks
        for callback in self.quote_callbacks:
            try:
                callback(symbol, quote_data)
            except Exception as e:
                self.logger.error(f"Quote callback error: {e}")
    
    def subscribe_moc_imbalance(self) -> bool:
        """Subscribe to MOC imbalance feed during configured window"""
        if not self.moc_enabled:
            self.logger.warning("MOC imbalance feed is disabled in configuration")
            return False
        
        # Check if in MOC window
        now = datetime.now().time()
        if not (self.moc_start <= now <= self.moc_end):
            self.logger.info(
                f"Outside MOC window ({self.moc_start}-{self.moc_end}), "
                f"current time: {now}"
            )
            return False
        
        # TODO: Implement MOC imbalance subscription
        # This requires specific market data permissions
        self.logger.info("MOC imbalance subscription would be activated here")
        return True
    
    def add_bar_callback(self, callback: Callable) -> None:
        """Add callback for bar updates"""
        self.bar_callbacks.append(callback)
    
    def add_quote_callback(self, callback: Callable) -> None:
        """Add callback for quote updates"""
        self.quote_callbacks.append(callback)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        positions = []
        
        if not self.is_connected:
            self.logger.warning("Not connected to IBKR")
            return positions
        
        try:
            for position in self.ib.positions():
                positions.append({
                    'account': position.account,
                    'symbol': position.contract.symbol,
                    'position': position.position,
                    'avg_cost': position.avgCost
                })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        if not self.is_connected:
            return {}
        
        try:
            summary = {}
            
            for value in self.ib.accountSummary():
                if value.account == self.account:
                    summary[value.tag] = value.value
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get account summary: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check connection health
        Required by BaseModule
        """
        health = {
            'connected': self.is_connected,
            'trading_mode': self.trading_mode,
            'gateway': f"{self.gateway_host}:{self.gateway_port}",
            'active_bar_subscriptions': len(self.bar_subscriptions),
            'active_quote_subscriptions': len(self.quote_subscriptions),
            'max_subscriptions': self.max_concurrent_subscriptions,
            'stats': self.stats,
            'checks': {
                'connection': self.is_connected,
                'subscriptions_available': len(self.bar_subscriptions) + len(self.quote_subscriptions) < self.max_concurrent_subscriptions,
                'bars_enabled': self.bars_enabled,
                'quotes_enabled': self.quotes_enabled,
                'moc_enabled': self.moc_enabled
            }
        }
        
        # Overall health
        health['healthy'] = all([
            health['connected'],
            health['checks']['subscriptions_available']
        ])
        
        return health
    
    def shutdown(self) -> bool:
        """
        Shutdown connection
        Required by BaseModule
        """
        try:
            self.logger.info("Shutting down IBKR connection...")
            
            # Cancel all subscriptions
            for symbol, bars_dict in self.bar_subscriptions.items():
                for bar_size, bars in bars_dict.items():
                    self.ib.cancelRealTimeBars(bars)
                    self.logger.debug(f"Cancelled {bar_size} bars for {symbol}")
            
            for symbol, ticker in self.quote_subscriptions.items():
                self.ib.cancelMktData(ticker)
                self.logger.debug(f"Cancelled quotes for {symbol}")
            
            # Disconnect
            if self.ib.isConnected():
                self.ib.disconnect()
                self.logger.info("Disconnected from IBKR")
            
            self.is_connected = False
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from TWS/Gateway"""
        return self.shutdown()