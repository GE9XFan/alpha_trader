#!/usr/bin/env python3
"""
Alpha Vantage Client Module
Handles all interactions with Alpha Vantage API for options data.
Manages rate limiting, caching, and fallback strategies.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
import logging
from collections import deque
from functools import lru_cache
import hashlib

from src.core.config import get_config, AlphaVantageConfig

logger = logging.getLogger(__name__)


@dataclass
class OptionData:
    """Structure for Alpha Vantage option data response"""
    symbol: str
    strike: float
    expiry: date
    option_type: str  # 'CALL' or 'PUT'
    
    # Market data from AV
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    
    # Greeks from AV (no calculation needed!)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Additional AV data
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'symbol': self.symbol,
            'strike': self.strike,
            'expiry': self.expiry.isoformat(),
            'option_type': self.option_type,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho
        }


class RateLimiter:
    """Rate limiter for Alpha Vantage API calls"""
    
    def __init__(self, calls_per_minute: int):
        """
        Initialize rate limiter
        
        Args:
            calls_per_minute: Maximum API calls per minute
        """
        self.calls_per_minute = calls_per_minute
        self.call_times: deque = deque()
        self.lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """
        Acquire permission to make an API call
        Blocks if rate limit would be exceeded
        """
        async with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            while self.call_times and self.call_times[0] < now - 60:
                self.call_times.popleft()
            
            # If at limit, wait until oldest call expires
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                    # Recurse to clean up and check again
                    await self.acquire()
                    return
            
            # Record this call
            self.call_times.append(now)
    
    def reset(self) -> None:
        """Reset rate limiter"""
        self.call_times.clear()


class AlphaVantageClient:
    """
    Client for Alpha Vantage API
    Handles options data retrieval with caching and rate limiting
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, config: Optional[AlphaVantageConfig] = None):
        """
        Initialize Alpha Vantage client
        
        Args:
            config: Alpha Vantage configuration
        """
        self.config = config or get_config().alpha_vantage
        
        # Rate limiting
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        
        # Caching
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (data, expiry_time)
        self.cache_enabled = self.config.use_cache
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.api_calls_made = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        
        logger.info(f"AlphaVantageClient initialized (rate limit: {self.config.rate_limit}/min)")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, function: str, **params) -> str:
        """
        Generate cache key for request
        
        Args:
            function: API function name
            params: API parameters
            
        Returns:
            Cache key string
        """
        # Create deterministic key from function and params
        key_data = f"{function}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get data from cache if not expired
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None
        """
        if not self.cache_enabled:
            return None
            
        if cache_key in self.cache:
            data, expiry = self.cache[cache_key]
            if time.time() < expiry:
                self.cache_hits += 1
                logger.debug(f"Cache hit for {cache_key}")
                return data
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def _add_to_cache(self, cache_key: str, data: Any) -> None:
        """
        Add data to cache
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        if self.cache_enabled:
            expiry = time.time() + self.config.cache_ttl
            self.cache[cache_key] = (data, expiry)
            logger.debug(f"Cached {cache_key} until {datetime.fromtimestamp(expiry)}")
    
    async def _make_request(self, function: str, **params) -> Dict[str, Any]:
        """
        Make API request with rate limiting and retries
        
        Args:
            function: API function name
            params: API parameters
            
        Returns:
            API response as dictionary
        """
        # Check cache first
        cache_key = self._get_cache_key(function, **params)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Build request
        params = {
            'function': function,
            'apikey': self.config.api_key,
            **params
        }
        
        # Retry logic
        for attempt in range(self.config.retry_count):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    self.api_calls_made += 1
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if "Error Message" in data:
                            raise ValueError(f"API Error: {data['Error Message']}")
                        if "Note" in data:
                            logger.warning(f"API Note: {data['Note']}")
                            # Rate limit hit
                            await asyncio.sleep(60)
                            continue
                        
                        # Cache successful response
                        self._add_to_cache(cache_key, data)
                        return data
                    
                    elif response.status == 429:
                        # Rate limited
                        logger.warning("Rate limited by Alpha Vantage")
                        await asyncio.sleep(60)
                        continue
                    
                    else:
                        logger.error(f"API request failed: {response.status}")
                        
            except asyncio.TimeoutError:
                logger.error(f"Request timeout (attempt {attempt + 1}/{self.config.retry_count})")
            except Exception as e:
                logger.error(f"Request error: {e}")
                self.errors += 1
            
            if attempt < self.config.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"Failed to get data after {self.config.retry_count} attempts")
    
    async def get_realtime_options(self, symbol: str) -> List[OptionData]:
        """
        Get real-time options data with Greeks from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of OptionData objects with Greeks included
        """
        # TODO: Implement real-time options fetching
        # 1. Call REALTIME_OPTIONS function
        # 2. Parse response
        # 3. Extract Greeks from response (already calculated!)
        # 4. Create OptionData objects
        # 5. Handle missing data
        # 6. Return list of options
        pass
    
    async def get_historical_options(self, 
                                    symbol: str,
                                    date: Optional[str] = None) -> List[OptionData]:
        """
        Get historical options data with Greeks
        
        Args:
            symbol: Stock symbol
            date: Optional date (YYYY-MM-DD format)
            
        Returns:
            List of OptionData objects
        """
        # TODO: Implement historical options fetching
        # 1. Call HISTORICAL_OPTIONS function
        # 2. Parse response
        # 3. Extract Greeks from response
        # 4. Create OptionData objects
        # 5. Return list
        pass
    
    def parse_option_response(self, response: Dict[str, Any]) -> List[OptionData]:
        """
        Parse Alpha Vantage option response into OptionData objects
        
        Args:
            response: Raw API response
            
        Returns:
            List of parsed OptionData objects
        """
        # TODO: Implement response parsing
        # 1. Extract option chain from response
        # 2. For each contract:
        #    a. Extract strike, expiry, type
        #    b. Extract market data
        #    c. Extract Greeks (delta, gamma, theta, vega, rho)
        #    d. Extract IV
        #    e. Create OptionData object
        # 3. Handle missing fields
        # 4. Return list
        pass
    
    async def get_option_chain(self, 
                              symbol: str,
                              min_dte: int = 0,
                              max_dte: int = 45) -> List[OptionData]:
        """
        Get complete option chain for symbol
        
        Args:
            symbol: Stock symbol
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            
        Returns:
            Filtered list of OptionData objects
        """
        # TODO: Implement option chain fetching
        # 1. Get real-time options
        # 2. Filter by DTE range
        # 3. Sort by expiry and strike
        # 4. Return filtered list
        pass
    
    async def get_atm_options(self,
                            symbol: str,
                            spot_price: float,
                            dte_range: Tuple[int, int] = (0, 7)) -> List[OptionData]:
        """
        Get at-the-money options
        
        Args:
            symbol: Stock symbol
            spot_price: Current spot price
            dte_range: DTE range tuple
            
        Returns:
            List of ATM options
        """
        # TODO: Implement ATM option fetching
        # 1. Get option chain
        # 2. Filter by DTE
        # 3. Find closest strikes to spot
        # 4. Return ATM options
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics
        
        Returns:
            Statistics dictionary
        """
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )
        
        return {
            'api_calls': self.api_calls_made,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'errors': self.errors,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired_cache(self) -> int:
        """
        Remove expired entries from cache
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if expiry < now
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    async def warmup(self, symbols: List[str]) -> bool:
        """
        Warmup cache with option data for symbols
        
        Args:
            symbols: List of symbols to warmup
            
        Returns:
            True if warmup successful
        """
        # TODO: Implement cache warmup
        # 1. For each symbol:
        #    a. Get option chain
        #    b. Cache the data
        # 2. Log warmup status
        # 3. Return success
        pass
    
    async def test_connection(self) -> bool:
        """
        Test Alpha Vantage connection
        
        Returns:
            True if connection successful
        """
        try:
            # Try a simple API call
            response = await self._make_request('SYMBOL_SEARCH', keywords='SPY')
            return 'bestMatches' in response
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False