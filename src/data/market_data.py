"""
Market Data Manager - Implementation Plan Week 1 Day 1-2
IBKR connection for quotes, bars, and execution
Alpha Vantage handles options - this is just spot prices
"""
from ib_insync import IB, Stock, Option, MarketOrder, Contract
import asyncio
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from src.core.config import config
from src.core.logger import get_logger
from src.core.exceptions import IBKRException


logger = get_logger(__name__)


class MarketDataManager:
    """
    IBKR market data for quotes, bars, and execution
    Alpha Vantage handles options - this is just spot prices
    Used by: ML, Trading, Risk, Paper, Live, Community
    Implementation Plan Week 1 Day 1-2
    """
    
    def __init__(self):
        self.config = config.ibkr
        self.trading_config = config.trading
        self.ib = IB()
        self.connected = False
        
        # Data storage
        self.latest_prices = {}  # Cache for instant access
        self.bars_5sec = {}  # 5-second bars from IBKR
        self.subscriptions = {}
        
    async def connect(self):
        """Connect to IBKR - reused for paper and live"""
        try:
            port = self.config.port if self.trading_config.mode == 'paper' else 7496
            
            await self.ib.connectAsync(
                self.config.host,
                port,
                clientId=self.config.client_id,
                timeout=self.config.connection_timeout
            )
            
            self.connected = True
            logger.info(f"Connected to IBKR ({self.trading_config.mode} mode) for quotes/execution")
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat())
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise IBKRException(f"IBKR connection failed: {e}")
    
    async def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to market data - spot prices for options pricing"""
        for symbol in symbols:
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                
                # Subscribe to real-time bars
                bars = self.ib.reqRealTimeBars(
                    contract, 5, 'TRADES', False
                )
                self.subscriptions[symbol] = bars
                
                # Set up callback for updates
                bars.updateEvent += lambda bars, symbol=symbol: self._on_bar_update(symbol, bars)
                
                logger.info(f"Subscribed to {symbol} market data")
                
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
    
    def _on_bar_update(self, symbol: str, bars):
        """Handle bar updates - feeds everything"""
        if bars:
            latest = bars[-1]
            self.latest_prices[symbol] = latest.close
            self.bars_5sec[symbol] = latest
            # This spot price is used by Alpha Vantage options queries
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price - instant from cache"""
        return self.latest_prices.get(symbol, 0.0)
    
    async def get_bars(self, symbol: str, duration: str = '1 D', 
                      bar_size: str = '5 secs') -> pd.DataFrame:
        """Get historical bars for price action analysis"""
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        
        if bars:
            return pd.DataFrame([{
                'date': b.date,
                'open': b.open,
                'high': b.high,
                'low': b.low,
                'close': b.close,
                'volume': b.volume
            } for b in bars])
        
        return pd.DataFrame()
    
    async def execute_order(self, contract: Contract, order: Any):
        """Execute trades through IBKR"""
        if not self.connected:
            raise IBKRException("Not connected to IBKR")
        
        trade = self.ib.placeOrder(contract, order)
        
        # Wait for order to be placed
        await asyncio.sleep(0.1)
        
        return trade
    
    async def create_option_contract(self, symbol: str, expiry: str, 
                                    strike: float, right: str) -> Option:
        """Create option contract for IBKR execution"""
        contract = Option(symbol, expiry, strike, right, 'SMART')
        self.ib.qualifyContracts(contract)
        return contract
    
    async def _heartbeat(self):
        """Maintain connection heartbeat"""
        while self.connected:
            await asyncio.sleep(self.config.heartbeat_interval)
            if not self.ib.isConnected():
                logger.warning("IBKR connection lost, attempting reconnect...")
                await self.connect()


# Global instance - CREATE ONCE, USE FOREVER
market_data = MarketDataManager()
