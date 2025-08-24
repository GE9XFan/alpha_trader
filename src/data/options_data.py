#!/usr/bin/env python3
"""
Options Data Manager Module
Handles option chains and options-specific data using Alpha Vantage.
Greeks are now provided by Alpha Vantage - no local calculation needed.
Built on top of MarketDataManager and AlphaVantageClient.
"""

import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import asyncio
import logging
from collections import defaultdict

from src.data.market_data import MarketDataManager
from src.data.av_client import AlphaVantageClient, OptionData
from src.core.config import get_config, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Represents a single option contract"""
    symbol: str
    strike: float
    expiry: date
    option_type: str  # 'CALL' or 'PUT'
    contract: Optional[Contract] = None
    
    # Market data
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last
    
    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiry"""
        return (self.expiry - datetime.now().date()).days
    
    @property
    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years"""
        return max(0, self.days_to_expiry / 365.0)
    
    @property
    def is_itm(self) -> bool:
        """Check if option is in the money"""
        # Requires spot price - will be set by OptionsDataManager
        return False
    
    @property
    def moneyness(self) -> float:
        """Calculate moneyness (spot/strike for calls, strike/spot for puts)"""
        # Requires spot price - will be set by OptionsDataManager
        return 1.0


@dataclass
class OptionChainSnapshot:
    """Snapshot of entire option chain for a symbol"""
    symbol: str
    timestamp: datetime
    spot_price: float
    contracts: List[OptionContract]
    
    def get_strikes(self) -> List[float]:
        """Get unique strikes in the chain"""
        return sorted(list(set(c.strike for c in self.contracts)))
    
    def get_expirations(self) -> List[date]:
        """Get unique expiration dates"""
        return sorted(list(set(c.expiry for c in self.contracts)))
    
    def filter_by_expiry(self, expiry: date) -> List[OptionContract]:
        """Get contracts for specific expiry"""
        return [c for c in self.contracts if c.expiry == expiry]
    
    def filter_by_dte(self, min_dte: int, max_dte: int) -> List[OptionContract]:
        """Get contracts within DTE range"""
        return [c for c in self.contracts 
                if min_dte <= c.days_to_expiry <= max_dte]


class OptionsDataManager:
    """
    Options data manager using Alpha Vantage - REUSED BY ALL TRADING
    Greeks come pre-calculated from Alpha Vantage - no local calculation needed!
    Built on top of MarketDataManager for spot prices and AlphaVantageClient for options.
    """
    
    def __init__(self, 
                 market_data_manager: MarketDataManager,
                 av_client: Optional[AlphaVantageClient] = None):
        """
        Initialize OptionsDataManager
        
        Args:
            market_data_manager: Market data manager for spot prices
            av_client: Alpha Vantage client (creates new if None)
        """
        self.market = market_data_manager  # For spot prices
        self.av_client = av_client or AlphaVantageClient()  # For options data
        self.config = get_config()
        
        # Data storage
        self.chains: Dict[str, OptionChainSnapshot] = {}
        self.latest_options: Dict[str, List[OptionData]] = {}  # AV option data
        
        # Performance tracking
        self.data_fetches: int = 0
        self.fetch_errors: int = 0
        
        logger.info("OptionsDataManager initialized with Alpha Vantage")
    
    async def fetch_option_chain(self, 
                                symbol: str,
                                min_dte: int = 0,
                                max_dte: int = 45) -> OptionChainSnapshot:
        """
        Fetch option chain from Alpha Vantage
        
        Args:
            symbol: Underlying symbol
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            
        Returns:
            OptionChainSnapshot with all contracts (Greeks included from AV!)
        """
        # TODO: Implement option chain fetching from Alpha Vantage
        # 1. Get spot price from market data manager
        # 2. Call av_client.get_option_chain()
        # 3. Filter by DTE range
        # 4. Convert OptionData to OptionContract format
        # 5. Create OptionChainSnapshot
        # 6. Cache the chain
        # 7. Handle errors gracefully
        pass
    
    def get_option_greeks(self, option: OptionData) -> Dict[str, float]:
        """
        Get Greeks for option - DIRECTLY FROM ALPHA VANTAGE DATA
        No calculation needed!
        
        Args:
            option: OptionData from Alpha Vantage
            
        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        # Greeks are already in the OptionData from Alpha Vantage!
        return {
            'delta': option.delta,
            'gamma': option.gamma,
            'theta': option.theta,
            'vega': option.vega,
            'rho': option.rho
        }
    
    def find_atm_options(self,
                        symbol: str,
                        dte_min: int = 0,
                        dte_max: int = 7) -> List[OptionContract]:
        """
        Find at-the-money options for trading (still needed for option selection)
        
        Args:
            symbol: Underlying symbol
            dte_min: Minimum days to expiry
            dte_max: Maximum days to expiry
            
        Returns:
            List of ATM option contracts
        """
        # TODO: Implement ATM option finding using AV data
        # 1. Get current spot price from market manager
        # 2. Get cached option chain or fetch from AV
        # 3. Filter by DTE range
        # 4. For each expiry, find closest strike to spot
        # 5. Return both calls and puts
        # 6. Sort by DTE
        pass
    
    def get_iv_from_option(self, option: OptionData) -> float:
        """
        Get implied volatility - DIRECTLY FROM ALPHA VANTAGE
        
        Args:
            option: OptionData from Alpha Vantage
            
        Returns:
            Implied volatility (already calculated by AV!)
        """
        return option.implied_volatility
    
    def get_put_call_ratio(self, symbol: str, expiry: Optional[date] = None) -> float:
        """
        Calculate put/call ratio using AV data
        
        Args:
            symbol: Underlying symbol
            expiry: Specific expiry (all if None)
            
        Returns:
            Put/call volume ratio
        """
        # TODO: Implement put/call ratio using AV data
        # 1. Get latest options from cache
        # 2. Filter by expiry if provided
        # 3. Sum put volume from AV data
        # 4. Sum call volume from AV data
        # 5. Calculate ratio
        # 6. Handle edge cases
        pass
    
    def calculate_portfolio_greeks(self, 
                                  positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for portfolio using AV data
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Aggregate Greeks dictionary (from AV, not calculated!)
        """
        # TODO: Implement portfolio Greeks aggregation
        # 1. Initialize total Greeks
        # 2. For each position:
        #    a. Get option data from AV cache
        #    b. Get Greeks directly from AV data
        #    c. Multiply by position size
        #    d. Add to totals
        # 3. Return aggregate Greeks
        pass
    
    def estimate_slippage(self, 
                         contract: OptionContract,
                         quantity: int,
                         side: str) -> float:
        """
        Estimate slippage for option order
        
        Args:
            contract: Option contract
            quantity: Number of contracts
            side: 'BUY' or 'SELL'
            
        Returns:
            Estimated slippage in dollars per contract
        """
        # TODO: Implement slippage estimation
        # 1. Get bid-ask spread
        # 2. Estimate market impact
        # 3. Consider contract liquidity
        # 4. Apply size adjustment
        # 5. Return estimated slippage
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Cache statistics dictionary
        """
        # TODO: Calculate cache statistics
        # 1. Calculate hit rate
        # 2. Get cache size
        # 3. Memory usage estimate
        # 4. Return stats dictionary
        pass
    
    async def warmup(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Warmup options data
        
        Args:
            symbols: Symbols to warmup (uses config if None)
            
        Returns:
            True if warmup successful
        """
        # TODO: Implement warmup procedure
        # 1. Get symbols from config if not provided
        # 2. Fetch chains for each symbol
        # 3. Calculate initial Greeks
        # 4. Build IV surfaces
        # 5. Verify data quality
        # 6. Return success status
        pass