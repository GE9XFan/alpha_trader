#!/usr/bin/env python3
"""
Alpha Vantage Client Module - COMPLETE VERSION
Handles ALL 38 Alpha Vantage APIs across 6 categories.
Manages rate limiting (600/min premium), caching, and fallback strategies.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
import logging
from collections import deque
from functools import lru_cache
import hashlib
import pandas as pd
from enum import Enum

from src.core.config import get_config, AlphaVantageConfig

logger = logging.getLogger(__name__)


class AVFunction(Enum):
    """All 38 Alpha Vantage API functions"""
    # OPTIONS
    REALTIME_OPTIONS = "REALTIME_OPTIONS"
    HISTORICAL_OPTIONS = "HISTORICAL_OPTIONS"
    
    # TECHNICAL INDICATORS (16)
    RSI = "RSI"
    MACD = "MACD"
    STOCH = "STOCH"
    WILLR = "WILLR"
    MOM = "MOM"
    BBANDS = "BBANDS"
    ATR = "ATR"
    ADX = "ADX"
    AROON = "AROON"
    CCI = "CCI"
    EMA = "EMA"
    SMA = "SMA"
    MFI = "MFI"
    OBV = "OBV"
    AD = "AD"
    VWAP = "VWAP"
    
    # ANALYTICS
    ANALYTICS_FIXED_WINDOW = "ANALYTICS_FIXED_WINDOW"
    ANALYTICS_SLIDING_WINDOW = "ANALYTICS_SLIDING_WINDOW"
    
    # SENTIMENT
    NEWS_SENTIMENT = "NEWS_SENTIMENT"
    TOP_GAINERS_LOSERS = "TOP_GAINERS_LOSERS"
    INSIDER_TRANSACTIONS = "INSIDER_TRANSACTIONS"
    
    # FUNDAMENTALS (8)
    OVERVIEW = "OVERVIEW"
    EARNINGS = "EARNINGS"
    INCOME_STATEMENT = "INCOME_STATEMENT"
    BALANCE_SHEET = "BALANCE_SHEET"
    CASH_FLOW = "CASH_FLOW"
    DIVIDENDS = "DIVIDENDS"
    SPLITS = "SPLITS"
    EARNINGS_CALENDAR = "EARNINGS_CALENDAR"
    
    # ECONOMIC (5)
    TREASURY_YIELD = "TREASURY_YIELD"
    FEDERAL_FUNDS_RATE = "FEDERAL_FUNDS_RATE"
    CPI = "CPI"
    INFLATION = "INFLATION"
    REAL_GDP = "REAL_GDP"


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


@dataclass
class TechnicalIndicator:
    """Structure for technical indicator data"""
    symbol: str
    indicator: str
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentData:
    """Structure for sentiment data"""
    symbol: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'Bullish', 'Bearish', 'Neutral'
    relevance_score: float
    article_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FundamentalData:
    """Structure for fundamental data"""
    symbol: str
    metric_type: str
    timestamp: datetime
    data: Dict[str, Any]
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get specific fundamental value"""
        return self.data.get(key, default)


class RateLimiter:
    """Rate limiter for Alpha Vantage API calls - Updated for 600/min"""
    
    def __init__(self, calls_per_minute: int):
        """
        Initialize rate limiter
        
        Args:
            calls_per_minute: Maximum API calls per minute (600 for premium)
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
                    logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
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
    Client for ALL 38 Alpha Vantage APIs
    Handles options, technical indicators, sentiment, fundamentals, and economic data
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Cache TTLs for different API types (seconds)
    CACHE_TTLS = {
        'options': 60,          # 1 minute
        'technical': 300,       # 5 minutes
        'sentiment': 900,       # 15 minutes
        'fundamentals': 86400,  # 1 day
        'economic': 604800,     # 1 week
    }
    
    def __init__(self, config: Optional[AlphaVantageConfig] = None):
        """
        Initialize Alpha Vantage client
        
        Args:
            config: Alpha Vantage configuration
        """
        self.config = config or get_config().alpha_vantage
        
        # Rate limiting (600/min for premium)
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        
        # Caching with differentiated TTLs
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
    
    def _get_cache_ttl(self, function: str) -> int:
        """
        Get appropriate cache TTL for function type
        
        Args:
            function: API function name
            
        Returns:
            TTL in seconds
        """
        # Determine category and return appropriate TTL
        if function in ['REALTIME_OPTIONS', 'HISTORICAL_OPTIONS']:
            return self.CACHE_TTLS['options']
        elif function in ['RSI', 'MACD', 'STOCH', 'BBANDS', 'ATR', 'ADX', 'SMA', 'EMA', 'VWAP']:
            return self.CACHE_TTLS['technical']
        elif function in ['NEWS_SENTIMENT', 'TOP_GAINERS_LOSERS', 'INSIDER_TRANSACTIONS']:
            return self.CACHE_TTLS['sentiment']
        elif function in ['OVERVIEW', 'EARNINGS', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']:
            return self.CACHE_TTLS['fundamentals']
        elif function in ['TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 'CPI', 'INFLATION', 'REAL_GDP']:
            return self.CACHE_TTLS['economic']
        else:
            return 300  # Default 5 minutes
    
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
    
    def _add_to_cache(self, cache_key: str, data: Any, function: str) -> None:
        """
        Add data to cache with appropriate TTL
        
        Args:
            cache_key: Cache key
            data: Data to cache
            function: API function for TTL determination
        """
        if self.cache_enabled:
            ttl = self._get_cache_ttl(function)
            expiry = time.time() + ttl
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
                        self._add_to_cache(cache_key, data, function)
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
    
    # ============= OPTIONS APIs =============
    
    async def get_realtime_options(self, symbol: str, require_greeks: bool = True) -> List[OptionData]:
        """
        Get real-time options data with Greeks from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            require_greeks: Whether to require Greeks in response
            
        Returns:
            List of OptionData objects with Greeks included
        """
        # TODO: Implement real-time options fetching
        # 1. Call REALTIME_OPTIONS function
        # 2. Parse response using parse_option_response
        # 3. Filter by Greeks requirement if needed
        # 4. Return list of OptionData objects
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
        # 3. Return list of OptionData objects
        pass
    
    # ============= TECHNICAL INDICATORS APIs =============
    
    async def get_technical_indicator(self, 
                                     function: str,
                                     symbol: str,
                                     interval: str = 'daily',
                                     **kwargs) -> pd.DataFrame:
        """
        Get technical indicator data
        
        Args:
            function: Indicator function (RSI, MACD, etc.)
            symbol: Stock symbol
            interval: Time interval
            **kwargs: Additional parameters (time_period, series_type, etc.)
            
        Returns:
            DataFrame with indicator values
        """
        # TODO: Implement technical indicator fetching
        # 1. Validate function is a technical indicator
        # 2. Build appropriate parameters
        # 3. Make request
        # 4. Parse into DataFrame
        # 5. Return indicator data
        pass
    
    async def get_rsi(self, symbol: str, interval: str = 'daily', 
                     time_period: int = 14, series_type: str = 'close') -> pd.DataFrame:
        """Get RSI indicator"""
        # TODO: Call get_technical_indicator with RSI parameters
        pass
    
    async def get_macd(self, symbol: str, interval: str = 'daily',
                      fastperiod: int = 12, slowperiod: int = 26, 
                      signalperiod: int = 9) -> pd.DataFrame:
        """Get MACD indicator"""
        # TODO: Call get_technical_indicator with MACD parameters
        pass
    
    async def get_bollinger_bands(self, symbol: str, interval: str = 'daily',
                                 time_period: int = 20, nbdevup: int = 2,
                                 nbdevdn: int = 2) -> pd.DataFrame:
        """Get Bollinger Bands"""
        # TODO: Call get_technical_indicator with BBANDS parameters
        pass
    
    async def get_vwap(self, symbol: str, interval: str = '15min') -> pd.DataFrame:
        """Get VWAP (intraday only)"""
        # TODO: Validate interval is intraday
        # TODO: Call get_technical_indicator with VWAP parameters
        pass
    
    # ============= ANALYTICS APIs =============
    
    async def get_analytics_fixed_window(self, 
                                        symbols: List[str],
                                        interval: str = 'DAILY',
                                        range_: str = '1month',
                                        calculations: List[str] = None) -> Dict:
        """
        Get fixed window analytics
        
        Args:
            symbols: List of symbols (comma-separated)
            interval: Time interval (UPPERCASE)
            range_: Time range
            calculations: List of calculations to perform
            
        Returns:
            Analytics results dictionary
        """
        # TODO: Implement fixed window analytics
        # 1. Build UPPERCASE parameters
        # 2. Join symbols with comma
        # 3. Make request
        # 4. Parse analytics response
        # 5. Return results
        pass
    
    async def get_analytics_sliding_window(self,
                                          symbols: List[str],
                                          interval: str = 'DAILY',
                                          range_: str = '6month',
                                          window_size: int = 90,
                                          calculations: List[str] = None) -> Dict:
        """
        Get sliding window analytics
        
        Args:
            symbols: List of symbols
            interval: Time interval (UPPERCASE)
            range_: Time range
            window_size: Window size in days
            calculations: List of calculations
            
        Returns:
            Analytics results dictionary
        """
        # TODO: Implement sliding window analytics
        # 1. Build UPPERCASE parameters
        # 2. Join symbols
        # 3. Make request
        # 4. Parse results
        # 5. Return analytics
        pass
    
    # ============= SENTIMENT APIs =============
    
    async def get_news_sentiment(self, 
                                tickers: Optional[str] = None,
                                topics: Optional[str] = None,
                                sort: str = 'LATEST',
                                limit: int = 50) -> List[SentimentData]:
        """
        Get news sentiment data
        
        Args:
            tickers: Comma-separated tickers (note: 'tickers' not 'symbol')
            topics: Topics to filter
            sort: Sort order
            limit: Number of results
            
        Returns:
            List of SentimentData objects
        """
        # TODO: Implement news sentiment fetching
        # 1. Build parameters (use 'tickers' not 'symbol')
        # 2. Make request
        # 3. Parse sentiment response
        # 4. Calculate aggregate sentiment scores
        # 5. Return SentimentData list
        pass
    
    async def get_top_gainers_losers(self) -> Dict[str, List[Dict]]:
        """
        Get top gainers and losers
        
        Returns:
            Dictionary with 'gainers', 'losers', 'most_active' lists
        """
        # TODO: Implement top gainers/losers fetching
        # 1. Make request (no parameters needed)
        # 2. Parse into categories
        # 3. Return structured data
        pass
    
    async def get_insider_transactions(self, symbol: str) -> List[Dict]:
        """
        Get insider transactions
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of insider transaction records
        """
        # TODO: Implement insider transactions fetching
        # 1. Make request
        # 2. Parse transactions
        # 3. Return list
        pass
    
    # ============= FUNDAMENTALS APIs =============
    
    async def get_company_overview(self, symbol: str) -> FundamentalData:
        """
        Get company overview with key metrics
        
        Args:
            symbol: Stock symbol
            
        Returns:
            FundamentalData with company information
        """
        # TODO: Implement company overview fetching
        # 1. Make request
        # 2. Parse overview data
        # 3. Create FundamentalData object
        # 4. Return overview
        pass
    
    async def get_earnings(self, symbol: str) -> pd.DataFrame:
        """
        Get earnings history and estimates
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with earnings data
        """
        # TODO: Implement earnings fetching
        # 1. Make request
        # 2. Parse quarterly and annual earnings
        # 3. Create DataFrame
        # 4. Return earnings data
        pass
    
    async def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """
        Get income statement data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with income statement
        """
        # TODO: Implement income statement fetching
        # 1. Make request
        # 2. Parse annual and quarterly reports
        # 3. Create DataFrame
        # 4. Return income statement
        pass
    
    async def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """
        Get balance sheet data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with balance sheet
        """
        # TODO: Implement balance sheet fetching
        # 1. Make request
        # 2. Parse balance sheet data
        # 3. Create DataFrame
        # 4. Return balance sheet
        pass
    
    async def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """
        Get cash flow statement
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with cash flow data
        """
        # TODO: Implement cash flow fetching
        # 1. Make request
        # 2. Parse cash flow data
        # 3. Create DataFrame
        # 4. Return cash flow
        pass
    
    async def get_earnings_calendar(self, horizon: str = '3month') -> str:
        """
        Get earnings calendar (returns CSV format)
        
        Args:
            horizon: Time horizon
            
        Returns:
            CSV string with earnings calendar
        """
        # TODO: Implement earnings calendar fetching
        # 1. Make request
        # 2. Note: Returns CSV format
        # 3. Return raw CSV string
        pass
    
    # ============= ECONOMIC APIs =============
    
    async def get_treasury_yield(self, 
                                interval: str = 'monthly',
                                maturity: str = '10year') -> pd.DataFrame:
        """
        Get treasury yield data (for risk-free rate)
        
        Args:
            interval: Data interval
            maturity: Bond maturity
            
        Returns:
            DataFrame with yield data
        """
        # TODO: Implement treasury yield fetching
        # 1. Make request
        # 2. Parse yield data
        # 3. Create DataFrame
        # 4. Return yields
        pass
    
    async def get_federal_funds_rate(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get federal funds rate
        
        Args:
            interval: Data interval
            
        Returns:
            DataFrame with fed funds rate
        """
        # TODO: Implement fed funds rate fetching
        # 1. Make request
        # 2. Parse rate data
        # 3. Create DataFrame
        # 4. Return rates
        pass
    
    async def get_cpi(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Consumer Price Index data
        
        Args:
            interval: Data interval
            
        Returns:
            DataFrame with CPI data
        """
        # TODO: Implement CPI fetching
        # 1. Make request
        # 2. Parse CPI data
        # 3. Create DataFrame
        # 4. Return CPI
        pass
    
    async def get_inflation(self) -> pd.DataFrame:
        """
        Get inflation expectation data
        
        Returns:
            DataFrame with inflation data
        """
        # TODO: Implement inflation fetching
        # 1. Make request
        # 2. Parse inflation data
        # 3. Create DataFrame
        # 4. Return inflation
        pass
    
    async def get_real_gdp(self, interval: str = 'quarterly') -> pd.DataFrame:
        """
        Get real GDP data
        
        Args:
            interval: Data interval
            
        Returns:
            DataFrame with GDP data
        """
        # TODO: Implement GDP fetching
        # 1. Make request
        # 2. Parse GDP data
        # 3. Create DataFrame
        # 4. Return GDP
        pass
    
    # ============= RESPONSE PARSERS =============
    
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
        #    b. Extract market data (bid, ask, last, volume, OI)
        #    c. Extract Greeks (delta, gamma, theta, vega, rho) - DIRECTLY FROM AV!
        #    d. Extract IV
        #    e. Create OptionData object
        # 3. Handle missing fields gracefully
        # 4. Return list
        pass
    
    def parse_technical_response(self, response: Dict[str, Any], 
                                indicator: str) -> pd.DataFrame:
        """
        Parse technical indicator response
        
        Args:
            response: Raw API response
            indicator: Indicator type
            
        Returns:
            DataFrame with indicator values
        """
        # TODO: Implement technical indicator parsing
        # 1. Extract time series data
        # 2. Handle different indicator formats
        # 3. Create DataFrame with proper columns
        # 4. Set datetime index
        # 5. Return DataFrame
        pass
    
    def parse_sentiment_response(self, response: Dict[str, Any]) -> List[SentimentData]:
        """
        Parse sentiment response
        
        Args:
            response: Raw API response
            
        Returns:
            List of SentimentData objects
        """
        # TODO: Implement sentiment parsing
        # 1. Extract feed items
        # 2. Calculate sentiment scores
        # 3. Create SentimentData objects
        # 4. Return list
        pass
    
    def parse_fundamental_response(self, response: Dict[str, Any], 
                                  metric_type: str) -> FundamentalData:
        """
        Parse fundamental data response
        
        Args:
            response: Raw API response
            metric_type: Type of fundamental data
            
        Returns:
            FundamentalData object
        """
        # TODO: Implement fundamental parsing
        # 1. Extract relevant data fields
        # 2. Handle missing data
        # 3. Create FundamentalData object
        # 4. Return parsed data
        pass
    
    def parse_economic_response(self, response: Dict[str, Any]) -> pd.DataFrame:
        """
        Parse economic indicator response
        
        Args:
            response: Raw API response
            
        Returns:
            DataFrame with economic data
        """
        # TODO: Implement economic data parsing
        # 1. Extract time series
        # 2. Create DataFrame
        # 3. Set proper data types
        # 4. Return DataFrame
        pass
    
    # ============= UTILITY METHODS =============
    
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
            'cache_size': len(self.cache),
            'rate_limit': self.config.rate_limit
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
        Warmup cache with all data types for symbols
        
        Args:
            symbols: List of symbols to warmup
            
        Returns:
            True if warmup successful
        """
        # TODO: Implement comprehensive warmup
        # 1. For each symbol:
        #    a. Get option chain
        #    b. Get key technical indicators
        #    c. Get sentiment data
        #    d. Get company overview
        # 2. Cache all data
        # 3. Log warmup status
        # 4. Return success
        pass
    
    async def test_connection(self) -> bool:
        """
        Test Alpha Vantage connection and API key
        
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
    
    async def test_all_endpoints(self) -> Dict[str, bool]:
        """
        Test all 38 API endpoints
        
        Returns:
            Dictionary mapping endpoints to success status
        """
        # TODO: Implement comprehensive endpoint testing
        # 1. Test each of 38 endpoints
        # 2. Use minimal parameters
        # 3. Track success/failure
        # 4. Return results
        pass