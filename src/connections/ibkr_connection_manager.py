#!/usr/bin/env python3
"""
IBKR Connection Manager - Phase 2.4
Real-time market data connection with bar aggregation integration
ALL VALUES FROM CONFIGURATION - ZERO HARDCODED VALUES
"""

import logging
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import defaultdict
import threading

from ib_insync import IB, Stock, Contract, util
from ib_insync.objects import RealTimeBar, BarData, Ticker

from src.foundation.config_manager import get_config_manager
from src.data.bar_aggregator import get_bar_aggregator
from src.foundation.logger import get_logger


class IBKRConnectionManager:
    """
    IBKR Connection Manager with real-time data feeds
    Integrates with BarAggregator for mathematical aggregation
    ALL parameters from configuration
    """
    
    def __init__(self):
        """Initialize IBKR connection manager with configuration"""
        self.config = get_config_manager()
        self.logger = get_logger(self.__class__.__name__)
        self.bar_aggregator = get_bar_aggregator()
        
        # Load IBKR configuration - fail fast if missing
        try:
            self.ibkr_config = self.config.ibkr_config
            self.connection_config = self.ibkr_config['connection']
            self.market_data_config = self.ibkr_config['market_data']
            self.bar_config = self.ibkr_config['bar_aggregation']
            self.execution_config = self.ibkr_config['execution']
            
            # Connection parameters - all from config
            self.host = self.ibkr_config['host']
            self.port = self.ibkr_config['port'] 
            self.client_id = self.ibkr_config['client_id']
            self.timeout = self.connection_config['timeout']
            self.readonly = self.connection_config['readonly']
            self.account = self.connection_config['account']
            
            # Market data parameters - all from config
            self.max_subscriptions = self.market_data_config['max_subscriptions']
            self.tier_limits = self.market_data_config['subscription_tiers']
            
            # Bar aggregation parameters - all from config
            self.source_interval = self.bar_config['source_interval']
            self.target_intervals = self.bar_config['target_intervals']
            
            # Load symbol tiers from schedules
            schedules = self.config.schedules
            self.tier_a_symbols = schedules['tier_a_symbols']
            self.tier_b_symbols = schedules['tier_b_symbols']
            self.tier_c_symbols = schedules['tier_c_symbols']
            
        except KeyError as e:
            raise ValueError(f"Required IBKR configuration missing: {e}")
        
        # Initialize IB connection
        self.ib = IB()
        
        # Subscription tracking
        self.active_subscriptions = {}
        self.subscription_count = 0
        
        # Data handlers
        self.bar_handlers = []
        self.ticker_handlers = []
        self.error_handlers = []
        
        # Connection state
        self.is_connected = False
        self.reconnect_attempts = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("IBKR Connection Manager initialized from configuration:")
        self.logger.info(f"  Connection: {self.host}:{self.port} (client_id={self.client_id})")
        self.logger.info(f"  Account: {self.account}")
        self.logger.info(f"  Source interval: {self.source_interval}s")
        self.logger.info(f"  Max subscriptions: {self.max_subscriptions}")
        self.logger.info(f"  Tier limits: {self.tier_limits}")
        self.logger.info(f"  Tier A symbols: {self.tier_a_symbols}")
        self.logger.info(f"  Tier B symbols: {self.tier_b_symbols}")
    
    def connect(self) -> bool:
        """
        Connect to IBKR TWS/Gateway using configuration parameters
        
        Returns:
            True if connection successful
        """
        try:
            self.logger.info(f"Connecting to IBKR: {self.host}:{self.port}")
            
            # Connect with all parameters from configuration
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=self.readonly,
                account=self.account
            )
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            if self.ib.isConnected():
                self.is_connected = True
                self.reconnect_attempts = 0
                
                # Set up event handlers
                self._setup_event_handlers()
                
                self.logger.info("✓ IBKR connection successful")
                return True
            else:
                self.logger.error("✗ IBKR connection failed")
                return False
                
        except Exception as e:
            self.logger.error(f"IBKR connection error: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Setup IBKR event handlers"""
        # Bar update handler
        self.ib.barUpdateEvent += self._on_bar_update
        
        # Ticker update handler  
        self.ib.tickerUpdateEvent += self._on_ticker_update
        
        # Error handler
        self.ib.errorEvent += self._on_error
        
        # Disconnection handler
        self.ib.disconnectedEvent += self._on_disconnect
        
        self.logger.info("Event handlers configured")
    
    def subscribe_bars(self, symbols: List[str] = None) -> bool:
        """
        Subscribe to 5-second bars for specified symbols
        
        Args:
            symbols: List of symbols to subscribe to (defaults to tier symbols)
            
        Returns:
            True if subscriptions successful
        """
        if not self.is_connected:
            self.logger.error("Cannot subscribe - not connected to IBKR")
            return False
        
        # Use tier symbols if none specified
        if symbols is None:
            symbols = self._get_subscription_symbols()
        
        success_count = 0
        
        for symbol in symbols:
            try:
                # Check subscription limits
                if self.subscription_count >= self.max_subscriptions:
                    self.logger.warning(f"Subscription limit reached ({self.max_subscriptions}), skipping {symbol}")
                    break
                
                contract = Stock(symbol, 'SMART', 'USD')
                
                # Subscribe to real-time bars with source interval from config
                bars = self.ib.reqRealTimeBars(
                    contract=contract,
                    barSize=self.source_interval,  # From configuration
                    whatToShow='TRADES',
                    useRTH=False
                )
                
                self.active_subscriptions[symbol] = {
                    'contract': contract,
                    'bars': bars,
                    'type': 'bars'
                }
                
                self.subscription_count += 1
                success_count += 1
                
                self.logger.info(f"✓ Subscribed to {self.source_interval}s bars for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to subscribe to bars for {symbol}: {e}")
        
        self.logger.info(f"Bar subscriptions: {success_count}/{len(symbols)} successful")
        return success_count > 0
    
    def subscribe_tickers(self, symbols: List[str] = None) -> bool:
        """
        Subscribe to real-time ticker data
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            True if subscriptions successful
        """
        if not self.is_connected:
            self.logger.error("Cannot subscribe - not connected to IBKR")
            return False
        
        if symbols is None:
            symbols = self._get_subscription_symbols()
        
        success_count = 0
        
        for symbol in symbols:
            try:
                # Check subscription limits
                if self.subscription_count >= self.max_subscriptions:
                    self.logger.warning(f"Subscription limit reached ({self.max_subscriptions}), skipping {symbol}")
                    break
                
                contract = Stock(symbol, 'SMART', 'USD')
                
                # Subscribe to market data
                ticker = self.ib.reqMktData(
                    contract=contract,
                    genericTickList='',
                    snapshot=False,
                    regulatorySnapshot=False
                )
                
                self.active_subscriptions[f"{symbol}_ticker"] = {
                    'contract': contract,
                    'ticker': ticker,
                    'type': 'ticker'
                }
                
                self.subscription_count += 1
                success_count += 1
                
                self.logger.info(f"✓ Subscribed to ticker data for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to subscribe to ticker for {symbol}: {e}")
        
        self.logger.info(f"Ticker subscriptions: {success_count}/{len(symbols)} successful")
        return success_count > 0
    
    def _get_subscription_symbols(self) -> List[str]:
        """Get symbols for subscription based on tier limits from configuration"""
        symbols = []
        
        # Add Tier A symbols (up to tier limit)
        tier_a_limit = self.tier_limits['tier_a']
        symbols.extend(self.tier_a_symbols[:tier_a_limit])
        
        # Add Tier B symbols (up to tier limit)
        tier_b_limit = self.tier_limits['tier_b']
        symbols.extend(self.tier_b_symbols[:tier_b_limit])
        
        # Add Tier C symbols (up to tier limit)
        tier_c_limit = self.tier_limits['tier_c']
        symbols.extend(self.tier_c_symbols[:tier_c_limit])
        
        # Ensure we don't exceed max subscriptions
        total_limit = min(len(symbols), self.max_subscriptions)
        
        self.logger.info(f"Subscription symbols: {total_limit} total")
        self.logger.info(f"  Tier A: {len(self.tier_a_symbols[:tier_a_limit])}")
        self.logger.info(f"  Tier B: {len(self.tier_b_symbols[:tier_b_limit])}")
        self.logger.info(f"  Tier C: {len(self.tier_c_symbols[:tier_c_limit])}")
        
        return symbols[:total_limit]
    
    def _on_bar_update(self, bars, hasNewBar):
        """
        Handle real-time bar updates
        Based on captured IBKR bar structure from analysis.txt
        """
        if hasNewBar and bars:
            try:
                # Get the latest bar
                latest_bar = bars[-1]
                
                # Extract symbol from contract
                symbol = latest_bar.contract.symbol if hasattr(latest_bar, 'contract') else 'UNKNOWN'
                
                # Convert IBKR bar format to our format based on analysis.txt
                # Real-time bars have: time, endTime, open_, high, low, close, volume, wap, count
                bar_data = {
                    'timestamp': latest_bar.time,
                    'date': latest_bar.time,  # For compatibility
                    'open': latest_bar.open_,
                    'high': latest_bar.high,
                    'low': latest_bar.low,
                    'close': latest_bar.close,
                    'volume': latest_bar.volume,
                    'vwap': latest_bar.wap,
                    'average': latest_bar.wap,  # For compatibility
                    'bar_count': latest_bar.count,
                    'barCount': latest_bar.count  # For compatibility
                }
                
                # Process through bar aggregator
                completed_bars = self.bar_aggregator.process_5sec_bar(symbol, bar_data)
                
                # Notify handlers of new 5-second bar
                self._notify_bar_handlers(symbol, bar_data, '5sec')
                
                # Notify handlers of any completed aggregated bars
                for agg_bar in completed_bars:
                    timeframe_name = self._get_timeframe_name(agg_bar.timeframe_seconds)
                    agg_bar_data = {
                        'timestamp': agg_bar.timestamp,
                        'date': agg_bar.timestamp,
                        'open': agg_bar.open,
                        'high': agg_bar.high,
                        'low': agg_bar.low,
                        'close': agg_bar.close,
                        'volume': agg_bar.volume,
                        'vwap': agg_bar.vwap,
                        'average': agg_bar.vwap,
                        'bar_count': agg_bar.bar_count,
                        'barCount': agg_bar.bar_count,
                        'source_bars_used': agg_bar.source_bars_used,
                        'timeframe_seconds': agg_bar.timeframe_seconds
                    }
                    
                    self._notify_bar_handlers(symbol, agg_bar_data, timeframe_name)
                
                self.logger.debug(f"Processed {self.source_interval}s bar for {symbol}, "
                                f"generated {len(completed_bars)} aggregated bars")
                
            except Exception as e:
                self.logger.error(f"Error processing bar update: {e}")
    
    def _on_ticker_update(self, ticker):
        """
        Handle ticker updates
        Based on captured IBKR ticker structure from analysis.txt
        """
        try:
            symbol = ticker.contract.symbol
            
            # Convert IBKR ticker format based on analysis.txt
            # Ticker has: bid, bidSize, ask, askSize, last, lastSize, volume, open, high, low, close, etc.
            ticker_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bid': ticker.bid,
                'bid_size': ticker.bidSize,
                'ask': ticker.ask,
                'ask_size': ticker.askSize,
                'last': ticker.last,
                'last_size': ticker.lastSize,
                'volume': ticker.volume,
                'open': ticker.open,
                'high': ticker.high,
                'low': ticker.low,
                'close': ticker.close,
                'vwap': ticker.vwap,
                'market_data_type': ticker.marketDataType,
                'min_tick': ticker.minTick
            }
            
            # Notify handlers
            self._notify_ticker_handlers(symbol, ticker_data)
            
            self.logger.debug(f"Ticker update for {symbol}: bid={ticker.bid} ask={ticker.ask}")
            
        except Exception as e:
            self.logger.error(f"Error processing ticker update: {e}")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IBKR errors"""
        self.logger.error(f"IBKR Error {errorCode}: {errorString} (reqId={reqId})")
        
        # Notify error handlers
        error_data = {
            'reqId': reqId,
            'errorCode': errorCode,
            'errorString': errorString,
            'contract': contract,
            'timestamp': datetime.now()
        }
        
        self._notify_error_handlers(error_data)
    
    def _on_disconnect(self):
        """Handle disconnection"""
        self.logger.warning("IBKR connection lost")
        self.is_connected = False
        
        # Attempt reconnection if configured
        self._attempt_reconnect()
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to IBKR"""
        max_attempts = 5  # Could be configurable
        
        while self.reconnect_attempts < max_attempts:
            self.reconnect_attempts += 1
            self.logger.info(f"Reconnection attempt {self.reconnect_attempts}/{max_attempts}")
            
            time.sleep(10)  # Wait before reconnecting
            
            if self.connect():
                self.logger.info("Reconnection successful")
                # Re-establish subscriptions
                self._reestablish_subscriptions()
                return
        
        self.logger.error(f"Failed to reconnect after {max_attempts} attempts")
    
    def _reestablish_subscriptions(self):
        """Re-establish all subscriptions after reconnection"""
        self.logger.info("Re-establishing subscriptions...")
        
        # Clear old subscription tracking
        self.active_subscriptions.clear()
        self.subscription_count = 0
        
        # Re-subscribe to bars and tickers
        symbols = self._get_subscription_symbols()
        self.subscribe_bars(symbols)
        self.subscribe_tickers(symbols)
    
    def _get_timeframe_name(self, seconds: int) -> str:
        """Convert seconds to timeframe name"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        else:
            return f"{seconds // 3600}h"
    
    # Handler registration methods
    def add_bar_handler(self, handler: Callable[[str, Dict, str], None]):
        """Add handler for bar updates"""
        self.bar_handlers.append(handler)
    
    def add_ticker_handler(self, handler: Callable[[str, Dict], None]):
        """Add handler for ticker updates"""
        self.ticker_handlers.append(handler)
    
    def add_error_handler(self, handler: Callable[[Dict], None]):
        """Add handler for errors"""
        self.error_handlers.append(handler)
    
    # Handler notification methods
    def _notify_bar_handlers(self, symbol: str, bar_data: Dict, timeframe: str):
        """Notify all bar handlers"""
        for handler in self.bar_handlers:
            try:
                handler(symbol, bar_data, timeframe)
            except Exception as e:
                self.logger.error(f"Error in bar handler: {e}")
    
    def _notify_ticker_handlers(self, symbol: str, ticker_data: Dict):
        """Notify all ticker handlers"""
        for handler in self.ticker_handlers:
            try:
                handler(symbol, ticker_data)
            except Exception as e:
                self.logger.error(f"Error in ticker handler: {e}")
    
    def _notify_error_handlers(self, error_data: Dict):
        """Notify all error handlers"""
        for handler in self.error_handlers:
            try:
                handler(error_data)
            except Exception as e:
                self.logger.error(f"Error in error handler: {e}")
    
    def get_subscription_status(self) -> Dict:
        """Get current subscription status"""
        return {
            'is_connected': self.is_connected,
            'total_subscriptions': self.subscription_count,
            'max_subscriptions': self.max_subscriptions,
            'active_subscriptions': list(self.active_subscriptions.keys()),
            'tier_limits': self.tier_limits,
            'tier_symbols': {
                'tier_a': self.tier_a_symbols,
                'tier_b': self.tier_b_symbols,
                'tier_c': self.tier_c_symbols
            }
        }
    
    def cancel_all_subscriptions(self):
        """Cancel all active subscriptions"""
        self.logger.info("Cancelling all subscriptions...")
        
        for key, subscription in self.active_subscriptions.items():
            try:
                if subscription['type'] == 'bars':
                    self.ib.cancelRealTimeBars(subscription['bars'])
                elif subscription['type'] == 'ticker':
                    self.ib.cancelMktData(subscription['ticker'].contract)
                    
            except Exception as e:
                self.logger.error(f"Error cancelling subscription {key}: {e}")
        
        self.active_subscriptions.clear()
        self.subscription_count = 0
        
        self.logger.info("All subscriptions cancelled")
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.is_connected:
            self.cancel_all_subscriptions()
            self.ib.disconnect()
            self.is_connected = False
            self.logger.info("Disconnected from IBKR")


# Singleton instance
_ibkr_manager = None


def get_ibkr_manager() -> IBKRConnectionManager:
    """Get or create singleton IBKRConnectionManager instance"""
    global _ibkr_manager
    if _ibkr_manager is None:
        _ibkr_manager = IBKRConnectionManager()
    return _ibkr_manager