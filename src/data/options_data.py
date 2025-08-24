#!/usr/bin/env python3
"""
Options Data Manager Module
Handles option chains, Greeks calculations, and options-specific data.
Built on top of MarketDataManager and reused by all trading components.
"""

import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from ib_insync import Option, OptionChain, Contract
import asyncio
import logging
from functools import lru_cache
from collections import defaultdict

from src.data.market_data import MarketDataManager
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
    Options data and Greeks calculator - REUSED BY ALL TRADING
    Built on top of MarketDataManager for spot prices
    """
    
    def __init__(self, market_data_manager: MarketDataManager):
        """
        Initialize OptionsDataManager
        
        Args:
            market_data_manager: Market data manager for spot prices
        """
        self.market = market_data_manager  # Reuse market data!
        self.config = get_config()
        
        # Data storage
        self.chains: Dict[str, OptionChainSnapshot] = {}
        self.greeks_cache: Dict[str, Dict[str, float]] = {}
        self.iv_surface: Dict[str, pd.DataFrame] = {}
        
        # Risk-free rate (update periodically)
        self.risk_free_rate: float = 0.05  # 5% default
        
        # Greeks calculation parameters
        self.iv_iterations: int = 100
        self.iv_tolerance: float = 0.0001
        
        # Performance tracking
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        logger.info("OptionsDataManager initialized")
    
    async def fetch_option_chain(self, 
                                symbol: str,
                                min_dte: int = 0,
                                max_dte: int = 45) -> OptionChainSnapshot:
        """
        Fetch option chain from IBKR
        
        Args:
            symbol: Underlying symbol
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            
        Returns:
            OptionChainSnapshot with all contracts
        """
        # TODO: Implement option chain fetching
        # 1. Create Stock contract for underlying
        # 2. Request option chain from IBKR
        # 3. Filter by DTE range
        # 4. Get market data for each contract
        # 5. Calculate Greeks for each contract
        # 6. Create OptionChainSnapshot
        # 7. Cache the chain
        # 8. Handle errors gracefully
        pass
    
    def calculate_greeks(self,
                        spot: float,
                        strike: float,
                        time_to_expiry: float,
                        volatility: float,
                        option_type: str,
                        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks - CRITICAL CALCULATION
        Reused by: Risk, ML features, position management
        
        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (annualized)
            option_type: 'CALL' or 'PUT'
            risk_free_rate: Risk-free rate (uses default if None)
            
        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        # TODO: Implement Black-Scholes Greeks calculation
        # 1. Create cache key from parameters
        # 2. Check cache first
        # 3. Calculate d1 and d2
        # 4. Calculate delta based on option type
        # 5. Calculate gamma (same for calls and puts)
        # 6. Calculate theta based on option type
        # 7. Calculate vega (same for calls and puts)
        # 8. Calculate rho based on option type
        # 9. Cache results
        # 10. Return Greeks dictionary
        pass
    
    @lru_cache(maxsize=1000)
    def _black_scholes_price(self,
                            spot: float,
                            strike: float,
                            time_to_expiry: float,
                            volatility: float,
                            option_type: str,
                            risk_free_rate: float) -> float:
        """
        Calculate Black-Scholes option price
        
        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            option_type: 'CALL' or 'PUT'
            risk_free_rate: Risk-free rate
            
        Returns:
            Theoretical option price
        """
        # TODO: Implement Black-Scholes pricing
        # 1. Calculate d1 and d2
        # 2. Calculate call price
        # 3. If put, use put-call parity
        # 4. Return theoretical price
        pass
    
    def calculate_implied_volatility(self,
                                    option_price: float,
                                    spot: float,
                                    strike: float,
                                    time_to_expiry: float,
                                    option_type: str,
                                    risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            option_price: Market price of option
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            option_type: 'CALL' or 'PUT'
            risk_free_rate: Risk-free rate
            
        Returns:
            Implied volatility
        """
        # TODO: Implement IV calculation
        # 1. Set initial guess (e.g., 0.25)
        # 2. Iterate using Newton-Raphson
        # 3. Calculate vega for adjustment
        # 4. Update volatility estimate
        # 5. Check convergence
        # 6. Return final IV
        pass
    
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
        # TODO: Implement ATM option finding
        # 1. Get current spot price
        # 2. Get option chain for symbol
        # 3. Filter by DTE range
        # 4. For each expiry, find closest strike
        # 5. Return both calls and puts
        # 6. Sort by DTE
        pass
    
    def find_options_by_delta(self,
                             symbol: str,
                             target_delta: float,
                             option_type: str,
                             dte_min: int = 0,
                             dte_max: int = 45) -> List[OptionContract]:
        """
        Find options with specific delta
        
        Args:
            symbol: Underlying symbol
            target_delta: Target delta (e.g., 0.30)
            option_type: 'CALL' or 'PUT'
            dte_min: Minimum days to expiry
            dte_max: Maximum days to expiry
            
        Returns:
            List of options near target delta
        """
        # TODO: Implement delta-based option finding
        # 1. Get option chain
        # 2. Filter by DTE and type
        # 3. Calculate deltas if not cached
        # 4. Find options closest to target delta
        # 5. Return sorted by delta difference
        pass
    
    def calculate_max_pain(self, symbol: str, expiry: date) -> float:
        """
        Calculate max pain strike for expiry
        
        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            
        Returns:
            Max pain strike price
        """
        # TODO: Implement max pain calculation
        # 1. Get option chain for expiry
        # 2. Get open interest for all strikes
        # 3. Calculate total value at each strike
        # 4. Find strike with minimum total value
        # 5. Return max pain strike
        pass
    
    def calculate_gamma_exposure(self, 
                                symbol: str,
                                spot_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Calculate gamma exposure (GEX) profile
        
        Args:
            symbol: Underlying symbol
            spot_range: Range of spot prices to calculate
            
        Returns:
            DataFrame with gamma exposure by strike
        """
        # TODO: Implement GEX calculation
        # 1. Get option chain
        # 2. Calculate gamma for each contract
        # 3. Multiply by open interest
        # 4. Aggregate by strike
        # 5. Create GEX profile
        # 6. Return as DataFrame
        pass
    
    def get_put_call_ratio(self, symbol: str, expiry: Optional[date] = None) -> float:
        """
        Calculate put/call ratio
        
        Args:
            symbol: Underlying symbol
            expiry: Specific expiry (all if None)
            
        Returns:
            Put/call volume ratio
        """
        # TODO: Implement put/call ratio
        # 1. Get option chain
        # 2. Filter by expiry if provided
        # 3. Sum put volume
        # 4. Sum call volume
        # 5. Calculate ratio
        # 6. Handle edge cases
        pass
    
    def calculate_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Calculate IV rank (0-100)
        
        Args:
            symbol: Underlying symbol
            lookback_days: Historical period for comparison
            
        Returns:
            IV rank percentage
        """
        # TODO: Implement IV rank calculation
        # 1. Get current IV (ATM)
        # 2. Get historical IV data
        # 3. Calculate min and max over period
        # 4. Calculate rank: (current - min) / (max - min)
        # 5. Convert to percentage
        # 6. Handle edge cases
        pass
    
    def calculate_iv_percentile(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Calculate IV percentile (0-100)
        
        Args:
            symbol: Underlying symbol
            lookback_days: Historical period for comparison
            
        Returns:
            IV percentile
        """
        # TODO: Implement IV percentile calculation
        # 1. Get current IV
        # 2. Get historical IV data
        # 3. Count days with lower IV
        # 4. Calculate percentile
        # 5. Return as percentage
        pass
    
    def build_iv_surface(self, symbol: str) -> pd.DataFrame:
        """
        Build implied volatility surface
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            DataFrame with IV surface (strikes x expirations)
        """
        # TODO: Implement IV surface construction
        # 1. Get complete option chain
        # 2. Calculate IV for each contract
        # 3. Pivot by strike and expiry
        # 4. Interpolate missing values
        # 5. Smooth surface if needed
        # 6. Cache and return
        pass
    
    def get_term_structure(self, symbol: str, strike: Optional[float] = None) -> pd.Series:
        """
        Get volatility term structure
        
        Args:
            symbol: Underlying symbol
            strike: Specific strike (ATM if None)
            
        Returns:
            Series with IV by expiration
        """
        # TODO: Implement term structure extraction
        # 1. Get option chain
        # 2. Use ATM strike if not specified
        # 3. Get IV for each expiry
        # 4. Create term structure series
        # 5. Return sorted by expiry
        pass
    
    def calculate_portfolio_greeks(self, 
                                  positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for portfolio
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Aggregate Greeks dictionary
        """
        # TODO: Implement portfolio Greeks aggregation
        # 1. Initialize total Greeks
        # 2. For each position:
        #    a. Get current spot price
        #    b. Calculate position Greeks
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