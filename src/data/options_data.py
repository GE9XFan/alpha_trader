#!/usr/bin/env python3
"""
Options Data Manager Module - UPDATED VERSION
Handles option chains and options-specific data using Alpha Vantage.
Greeks are PROVIDED by Alpha Vantage - NO local calculation needed!
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
    """Represents a single option contract with AV data"""
    symbol: str
    strike: float
    expiry: date
    option_type: str  # 'CALL' or 'PUT'
    
    # Market data from Alpha Vantage
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    
    # Greeks from Alpha Vantage (NOT calculated!)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Additional data
    theoretical_value: float = 0.0
    time_value: float = 0.0
    intrinsic_value: float = 0.0
    
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
    
    def is_itm(self, spot_price: float) -> bool:
        """Check if option is in the money"""
        if self.option_type == 'CALL':
            return spot_price > self.strike
        else:  # PUT
            return spot_price < self.strike
    
    def moneyness(self, spot_price: float) -> float:
        """Calculate moneyness"""
        if self.option_type == 'CALL':
            return spot_price / self.strike
        else:  # PUT
            return self.strike / spot_price
    
    @classmethod
    def from_av_option(cls, av_option: OptionData, symbol: str) -> 'OptionContract':
        """Create OptionContract from Alpha Vantage OptionData"""
        return cls(
            symbol=symbol,
            strike=av_option.strike,
            expiry=av_option.expiry,
            option_type=av_option.option_type,
            bid=av_option.bid,
            ask=av_option.ask,
            last=av_option.last,
            volume=av_option.volume,
            open_interest=av_option.open_interest,
            implied_volatility=av_option.implied_volatility,
            delta=av_option.delta,
            gamma=av_option.gamma,
            theta=av_option.theta,
            vega=av_option.vega,
            rho=av_option.rho,
            theoretical_value=av_option.theoretical_value,
            time_value=av_option.time_value,
            intrinsic_value=av_option.intrinsic_value
        )


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
    
    def get_atm_strike(self) -> float:
        """Get at-the-money strike"""
        strikes = self.get_strikes()
        if not strikes:
            return self.spot_price
        return min(strikes, key=lambda x: abs(x - self.spot_price))


class OptionsDataManager:
    """
    Options data manager using Alpha Vantage - SIMPLIFIED VERSION
    Greeks come pre-calculated from Alpha Vantage - NO local calculation!
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
        self.latest_options: Dict[str, List[OptionData]] = {}  # Raw AV data
        
        # Performance tracking
        self.data_fetches: int = 0
        self.cache_hits: int = 0
        self.fetch_errors: int = 0
        
        logger.info("OptionsDataManager initialized with Alpha Vantage (Greeks provided)")
    
    async def fetch_option_chain(self, 
                                symbol: str,
                                min_dte: int = 0,
                                max_dte: int = 45) -> OptionChainSnapshot:
        """
        Fetch option chain from Alpha Vantage with Greeks included
        
        Args:
            symbol: Underlying symbol
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            
        Returns:
            OptionChainSnapshot with all contracts (Greeks from AV!)
        """
        # TODO: Implement option chain fetching from Alpha Vantage
        # 1. Get spot price from market data manager
        # 2. Call av_client.get_realtime_options()
        # 3. Store raw AV data in latest_options
        # 4. Filter by DTE range
        # 5. Convert OptionData to OptionContract
        # 6. Create OptionChainSnapshot
        # 7. Cache the chain
        # 8. Handle errors gracefully
        # 9. Return snapshot
        pass
    
    def get_option_greeks(self, option: OptionContract) -> Dict[str, float]:
        """
        Get Greeks for option - DIRECTLY FROM ALPHA VANTAGE DATA
        No calculation needed - Greeks are already in the OptionContract!
        
        Args:
            option: OptionContract with AV data
            
        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        # Greeks are already in the OptionContract from Alpha Vantage!
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
        Find at-the-money options for trading
        
        Args:
            symbol: Underlying symbol
            dte_min: Minimum days to expiry
            dte_max: Maximum days to expiry
            
        Returns:
            List of ATM option contracts
        """
        # TODO: Implement ATM option finding using cached chain
        # 1. Get current spot price from market manager
        # 2. Get cached option chain or fetch if needed
        # 3. Filter by DTE range
        # 4. For each expiry, find closest strike to spot
        # 5. Return both calls and puts
        # 6. Sort by DTE
        pass
    
    def get_iv_from_option(self, option: OptionContract) -> float:
        """
        Get implied volatility - DIRECTLY FROM ALPHA VANTAGE
        
        Args:
            option: OptionContract with AV data
            
        Returns:
            Implied volatility (already in the option data!)
        """
        return option.implied_volatility
    
    def get_put_call_ratio(self, symbol: str, expiry: Optional[date] = None) -> float:
        """
        Calculate put/call ratio using cached AV data
        
        Args:
            symbol: Underlying symbol
            expiry: Specific expiry (all if None)
            
        Returns:
            Put/call volume ratio
        """
        # TODO: Implement put/call ratio using cached data
        # 1. Get latest options from cache
        # 2. Filter by expiry if provided
        # 3. Sum put volume from option data
        # 4. Sum call volume from option data
        # 5. Calculate ratio (handle division by zero)
        # 6. Return ratio
        pass
    
    def calculate_portfolio_greeks(self, 
                                  positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for portfolio using AV Greeks
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Aggregate Greeks dictionary (from AV, not calculated!)
        """
        # TODO: Implement portfolio Greeks aggregation
        # 1. Initialize total Greeks
        # 2. For each position:
        #    a. Find matching option in cache
        #    b. Get Greeks directly from option data (AV provided)
        #    c. Multiply by position size
        #    d. Add to totals
        # 3. Return aggregate Greeks
        pass
    
    def get_iv_rank(self, symbol: str, lookback_days: int = 365) -> float:
        """
        Calculate IV rank using historical IV data
        
        Args:
            symbol: Underlying symbol
            lookback_days: Days to look back
            
        Returns:
            IV rank (0-100)
        """
        # TODO: Implement IV rank calculation
        # 1. Get current IV from ATM options
        # 2. Get historical IV data (may need to store)
        # 3. Calculate percentile rank
        # 4. Return IV rank
        pass
    
    def get_iv_percentile(self, symbol: str, lookback_days: int = 365) -> float:
        """
        Calculate IV percentile
        
        Args:
            symbol: Underlying symbol
            lookback_days: Days to look back
            
        Returns:
            IV percentile (0-100)
        """
        # TODO: Implement IV percentile
        # 1. Get current IV
        # 2. Get historical IV data
        # 3. Calculate percentage of days below current
        # 4. Return percentile
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
        # 2. Estimate market impact based on volume
        # 3. Consider contract liquidity (open interest)
        # 4. Apply size adjustment
        # 5. Return estimated slippage
        pass
    
    def get_max_pain(self, symbol: str, expiry: date) -> float:
        """
        Calculate max pain strike
        
        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            
        Returns:
            Max pain strike price
        """
        # TODO: Implement max pain calculation
        # 1. Get all options for expiry
        # 2. For each strike, calculate total pain
        # 3. Find strike with maximum pain
        # 4. Return max pain strike
        pass
    
    def get_gamma_exposure(self, symbol: str) -> float:
        """
        Calculate net gamma exposure
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            Net gamma exposure
        """
        # TODO: Implement gamma exposure
        # 1. Get all options from cache
        # 2. Calculate gamma exposure for each
        # 3. Weight by open interest
        # 4. Sum net exposure
        # 5. Return total
        pass
    
    async def refresh_chain(self, symbol: str) -> bool:
        """
        Force refresh of option chain
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            True if refresh successful
        """
        # TODO: Implement chain refresh
        # 1. Clear cache for symbol
        # 2. Fetch new chain from AV
        # 3. Update cache
        # 4. Return success status
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Cache statistics dictionary
        """
        hit_rate = self.cache_hits / max(1, self.data_fetches)
        
        return {
            'chains_cached': len(self.chains),
            'total_fetches': self.data_fetches,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': hit_rate,
            'fetch_errors': self.fetch_errors
        }
    
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
        # 2. For each symbol:
        #    a. Fetch option chain from AV
        #    b. Cache the data
        # 3. Log warmup status
        # 4. Return success status
        pass
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data
        
        Args:
            symbol: Specific symbol to clear (all if None)
        """
        if symbol:
            self.chains.pop(symbol, None)
            self.latest_options.pop(symbol, None)
            logger.info(f"Cleared cache for {symbol}")
        else:
            self.chains.clear()
            self.latest_options.clear()
            logger.info("Cleared all options cache")