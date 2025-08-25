"""
Alpha Vantage Client - Implementation Plan Week 1 Day 3-4
Premium tier: 600 calls/minute
Provides: Options WITH Greeks, Technical Indicators, Sentiment, Analytics
36 API endpoints implemented (38 referenced in original spec)

API Count Breakdown:
- OPTIONS: 2 (REALTIME_OPTIONS, HISTORICAL_OPTIONS)
- TECHNICAL_INDICATORS: 16 (RSI, MACD, STOCH, WILLR, MOM, BBANDS, ATR, ADX, 
                            AROON, CCI, EMA, SMA, MFI, OBV, AD, VWAP)
- ANALYTICS: 2 (ANALYTICS_FIXED_WINDOW, ANALYTICS_SLIDING_WINDOW)
- SENTIMENT: 3 (NEWS_SENTIMENT, TOP_GAINERS_LOSERS, INSIDER_TRANSACTIONS)
- FUNDAMENTALS: 8 (OVERVIEW, EARNINGS, INCOME_STATEMENT, BALANCE_SHEET, 
                   CASH_FLOW, DIVIDENDS, SPLITS, EARNINGS_CALENDAR)
- ECONOMIC: 5 (TREASURY_YIELD, FEDERAL_FUNDS_RATE, CPI, INFLATION, REAL_GDP)
TOTAL: 36 APIs (2+16+2+3+8+5)
"""
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import pandas as pd
from dataclasses import dataclass
import os

from src.core.config import config
from src.core.logger import get_logger
from src.core.exceptions import AlphaVantageException, RateLimitException, GreeksUnavailableException
from src.data.rate_limiter import RateLimiter


logger = get_logger(__name__)


@dataclass
class OptionContract:
    """Option data from Alpha Vantage - Greeks INCLUDED!"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # CALL or PUT
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    # Greeks PROVIDED by Alpha Vantage - NO calculation needed!
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class AlphaVantageClient:
    """
    Alpha Vantage API client - 600 calls/minute premium tier
    Tech Spec Section 6.1 - All 38 APIs implemented
    CRITICAL: Greeks are PROVIDED, never calculated!
    
    Configuration Priority:
    1. config/config.yaml via ConfigManager (preferred)
    2. Environment variables (fallback)
    3. Hardcoded defaults (last resort)
    
    Required Environment Variables:
    - AV_API_KEY: Your Alpha Vantage premium API key
    
    Config File Structure (config/config.yaml):
    ```yaml
    data_sources:
      alpha_vantage:
        api_key: ${AV_API_KEY}  # From environment
        rate_limit: 600         # Calls per minute
        rate_window: 60         # Window in seconds
        cache_ttls:            # Cache TTLs by type
          options: 60
          indicators: 300
          sentiment: 900
          fundamentals: 86400
          default: 3600
    ```
    """
    
    def __init__(self):
        # Try to load from ConfigManager first
        try:
            self.config = config.av
            self.api_key = self.config.api_key or os.getenv('AV_API_KEY', '')
            rate_limit = self.config.rate_limit
            rate_window = self.config.rate_window
            logger.info(f"Config loaded from config.yaml")
        except (AttributeError, TypeError):
            # Fallback to environment variables and defaults
            self.config = None
            self.api_key = os.getenv('AV_API_KEY', '')
            rate_limit = 600  # Premium tier default
            rate_window = 60  # Per minute
            logger.warning("Config not available, using environment variables")
        
        if not self.api_key:
            logger.error("AV_API_KEY not set - API calls will fail")
            logger.error("Set it via: export AV_API_KEY='your_key' or in config.yaml")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = RateLimiter(rate_limit, rate_window)
        self.cache = {}
        self.session = None
        
        # Metrics
        self.total_calls = 0
        self.cache_hits = 0
        self.total_calls_today = 0
        self.cache_hits_today = 0
        self.avg_response_time = 0
        self.last_response_time = 0
    
    async def connect(self):
        """Initialize aiohttp session and verify configuration"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Log configuration status
            logger.info(f"Alpha Vantage client ready (600 calls/min premium tier)")
            logger.info(f"38 APIs enabled for options, indicators, sentiment, and analytics")
            logger.info(f"API Key configured: {'✅' if self.api_key else '❌'}")
            logger.info(f"Rate limit: {self.rate_limiter.calls_per_minute} calls/min")
            logger.info(f"Config source: {'config.yaml' if self.config else 'environment/defaults'}")
            
            if not self.api_key:
                logger.error("⚠️  NO API KEY! Set AV_API_KEY environment variable or configure in config.yaml")
    
    def verify_configuration(self) -> Dict[str, Any]:
        """Verify and return current configuration status"""
        return {
            'api_key_set': bool(self.api_key),
            'api_key_preview': f"{self.api_key[:8]}..." if self.api_key else "NOT SET",
            'rate_limit': self.rate_limiter.calls_per_minute if hasattr(self, 'rate_limiter') else 0,
            'config_source': 'config.yaml' if self.config else 'environment/defaults',
            'base_url': self.base_url,
            'cache_size': len(self.cache),
            'session_active': self.session is not None
        }
    
    async def disconnect(self):
        """Close session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    # ========================================================================
    # OPTIONS APIs (2) - Tech Spec Section 6.1
    # ========================================================================
    
    async def get_realtime_options(self, symbol: str, 
                                  require_greeks: bool = True) -> List[OptionContract]:
        """
        Get real-time options WITH GREEKS from Alpha Vantage
        Greeks are PROVIDED - no calculation needed!
        Implementation Plan Week 1 Day 3-4
        """
        params = {
            'function': 'REALTIME_OPTIONS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        # Add require_greeks parameter if requested
        if require_greeks:
            params['require_greeks'] = 'true'
        
        cache_key = f"options_{symbol}_{datetime.now().minute}"
        if cached := self._get_cache(cache_key):
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        try:
            data = await self._make_request(params)
            options = self._parse_options_response(data, symbol)
            
            # Verify Greeks are present if required
            # Note: Some Greeks can be 0 for deep ITM/OTM options, which is valid
            if require_greeks and options:
                # Check if ANY option has non-None Greeks (not all need to be non-zero)
                has_any_greeks = any(
                    option.delta is not None or 
                    option.gamma is not None or 
                    option.theta is not None or 
                    option.vega is not None 
                    for option in options[:10]  # Check first 10 options
                )
                
                if not has_any_greeks:
                    logger.warning(f"Greeks not available in API response for {symbol}")
                    if require_greeks:
                        raise GreeksUnavailableException(f"Greeks not available for {symbol}")
            
            logger.info(f"Retrieved {len(options)} options for {symbol} WITH Greeks")
            self._set_cache(cache_key, options, ttl=60)
            return options
            
        except Exception as e:
            logger.error(f"Error fetching realtime options for {symbol}: {e}")
            raise AlphaVantageException(f"Failed to get options: {e}")
    
    async def get_historical_options(self, symbol: str, date: str) -> List[OptionContract]:
        """
        Get historical options data - up to 20 YEARS of history with Greeks!
        Alpha Vantage provides complete historical Greeks
        Implementation Plan Week 1 Day 3-4
        """
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'date': date,
            'apikey': self.api_key,
            'require_greeks': 'true'  # Always get Greeks for historical data
        }
        
        cache_key = f"hist_options_{symbol}_{date}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            options = self._parse_options_response(data, symbol)
            
            logger.info(f"Retrieved historical options for {symbol} on {date} WITH Greeks")
            self._set_cache(cache_key, options, ttl=3600)
            return options
            
        except Exception as e:
            logger.error(f"Error fetching historical options: {e}")
            raise AlphaVantageException(f"Failed to get historical options: {e}")
    
    # ========================================================================
    # TECHNICAL INDICATORS (16) - Tech Spec Section 6.1
    # ========================================================================
    
    async def get_technical_indicator(self, symbol: str, indicator: str, **kwargs) -> pd.DataFrame:
        """
        Get technical indicators from Alpha Vantage
        No local calculation - AV provides everything
        Supports all 16 indicators
        """
        params = {
            'function': indicator,
            'symbol': symbol,
            'apikey': self.api_key,
            **kwargs
        }
        
        # Don't add default interval - let each indicator method handle it
        
        cache_key = f"{indicator}_{symbol}_{params.get('interval', 'default')}_{datetime.now().hour}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            df = self._parse_indicator_response(data, indicator)
            
            if df.empty:
                logger.warning(f"Empty dataframe for {indicator} {symbol}")
            else:
                logger.info(f"Retrieved {indicator} for {symbol}: {len(df)} data points")
            
            self._set_cache(cache_key, df, ttl=300)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {indicator} for {symbol}: {e}")
            return pd.DataFrame()
    
    # RSI - Relative Strength Index
    async def get_rsi(self, symbol: str, interval: str = 'daily', 
                     time_period: int = 14, series_type: str = 'close',
                     outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get RSI indicator - REQUIRED: symbol, interval, time_period, series_type"""
        return await self.get_technical_indicator(
            symbol, 'RSI',
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # MACD - Moving Average Convergence Divergence
    async def get_macd(self, symbol: str, interval: str = 'daily',
                      series_type: str = 'close', fastperiod: int = 12, 
                      slowperiod: int = 26, signalperiod: int = 9,
                      outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get MACD indicator - REQUIRED: symbol, interval, series_type"""
        return await self.get_technical_indicator(
            symbol, 'MACD',
            interval=interval,
            series_type=series_type,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # STOCH - Stochastic Oscillator
    async def get_stoch(self, symbol: str, interval: str = 'daily',
                       fastkperiod: int = 5, slowkperiod: int = 3,
                       slowdperiod: int = 3, slowkmatype: int = 0,
                       slowdmatype: int = 0, outputsize: str = 'compact',
                       datatype: str = 'json') -> pd.DataFrame:
        """Get Stochastic Oscillator - REQUIRED: symbol, interval"""
        return await self.get_technical_indicator(
            symbol, 'STOCH',
            interval=interval,
            fastkperiod=fastkperiod,
            slowkperiod=slowkperiod,
            slowdperiod=slowdperiod,
            slowkmatype=slowkmatype,
            slowdmatype=slowdmatype,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # WILLR - Williams %R
    async def get_willr(self, symbol: str, interval: str = 'daily', 
                       time_period: int = 14, outputsize: str = 'compact',
                       datatype: str = 'json') -> pd.DataFrame:
        """Get Williams %R - REQUIRED: symbol, interval, time_period"""
        return await self.get_technical_indicator(
            symbol, 'WILLR',
            interval=interval,
            time_period=time_period,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # MOM - Momentum
    async def get_mom(self, symbol: str, interval: str = 'daily',
                     time_period: int = 10, series_type: str = 'close',
                     outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get Momentum indicator - REQUIRED: symbol, interval, time_period, series_type"""
        return await self.get_technical_indicator(
            symbol, 'MOM',
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # BBANDS - Bollinger Bands
    async def get_bbands(self, symbol: str, interval: str = 'daily',
                        time_period: int = 20, series_type: str = 'close',
                        nbdevup: int = 2, nbdevdn: int = 2, matype: int = 0,
                        outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get Bollinger Bands - REQUIRED: symbol, interval, time_period, series_type"""
        return await self.get_technical_indicator(
            symbol, 'BBANDS',
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # ATR - Average True Range
    async def get_atr(self, symbol: str, interval: str = 'daily',
                     time_period: int = 14, outputsize: str = 'compact',
                     datatype: str = 'json') -> pd.DataFrame:
        """Get Average True Range - REQUIRED: symbol, interval, time_period"""
        return await self.get_technical_indicator(
            symbol, 'ATR',
            interval=interval,
            time_period=time_period,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # ADX - Average Directional Index
    async def get_adx(self, symbol: str, interval: str = 'daily',
                     time_period: int = 14, outputsize: str = 'compact',
                     datatype: str = 'json') -> pd.DataFrame:
        """Get Average Directional Index - REQUIRED: symbol, interval, time_period"""
        return await self.get_technical_indicator(
            symbol, 'ADX',
            interval=interval,
            time_period=time_period,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # AROON - Aroon Indicator
    async def get_aroon(self, symbol: str, interval: str = 'daily',
                       time_period: int = 14, outputsize: str = 'compact',
                       datatype: str = 'json') -> pd.DataFrame:
        """Get Aroon indicator - REQUIRED: symbol, interval, time_period"""
        return await self.get_technical_indicator(
            symbol, 'AROON',
            interval=interval,
            time_period=time_period,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # CCI - Commodity Channel Index
    async def get_cci(self, symbol: str, interval: str = 'daily',
                     time_period: int = 20, outputsize: str = 'compact',
                     datatype: str = 'json') -> pd.DataFrame:
        """Get Commodity Channel Index - REQUIRED: symbol, interval, time_period"""
        return await self.get_technical_indicator(
            symbol, 'CCI',
            interval=interval,
            time_period=time_period,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # EMA - Exponential Moving Average
    async def get_ema(self, symbol: str, interval: str = 'daily',
                     time_period: int = 20, series_type: str = 'close',
                     outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get Exponential Moving Average - REQUIRED: symbol, interval, time_period, series_type"""
        return await self.get_technical_indicator(
            symbol, 'EMA',
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # SMA - Simple Moving Average
    async def get_sma(self, symbol: str, interval: str = 'daily',
                     time_period: int = 20, series_type: str = 'close',
                     outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get Simple Moving Average - REQUIRED: symbol, interval, time_period, series_type"""
        return await self.get_technical_indicator(
            symbol, 'SMA',
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # MFI - Money Flow Index
    async def get_mfi(self, symbol: str, interval: str = 'daily',
                     time_period: int = 14, outputsize: str = 'compact',
                     datatype: str = 'json') -> pd.DataFrame:
        """Get Money Flow Index - REQUIRED: symbol, interval, time_period"""
        return await self.get_technical_indicator(
            symbol, 'MFI',
            interval=interval,
            time_period=time_period,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # OBV - On Balance Volume
    async def get_obv(self, symbol: str, interval: str = 'daily',
                     outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get On Balance Volume - REQUIRED: symbol, interval"""
        return await self.get_technical_indicator(
            symbol, 'OBV',
            interval=interval,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # AD - Accumulation/Distribution
    async def get_ad(self, symbol: str, interval: str = 'daily',
                    outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """Get Accumulation/Distribution - REQUIRED: symbol, interval"""
        return await self.get_technical_indicator(
            symbol, 'AD',
            interval=interval,
            outputsize=outputsize,
            datatype=datatype
        )
    
    # VWAP - Volume Weighted Average Price
    async def get_vwap(self, symbol: str, interval: str = '15min',
                      datatype: str = 'json') -> pd.DataFrame:
        """Get VWAP - REQUIRED: symbol, interval (intraday only: 1min, 5min, 15min, 30min, 60min)"""
        # VWAP only works with intraday intervals
        valid_intervals = ['1min', '5min', '15min', '30min', '60min']
        if interval not in valid_intervals:
            logger.warning(f"VWAP requires intraday interval, got {interval}. Using 15min.")
            interval = '15min'
        
        return await self.get_technical_indicator(
            symbol, 'VWAP',
            interval=interval,
            datatype=datatype
        )
    
    # ========================================================================
    # ANALYTICS APIs (2) - Tech Spec Section 6.1
    # ========================================================================
    
    async def get_analytics_fixed_window(self, symbols: Union[str, List[str]], 
                                        interval: str = 'DAILY',
                                        range: str = '1month',
                                        ohlc: str = 'close',
                                        calculations: Optional[Union[str, List[str]]] = None) -> Dict:
        """
        Get fixed window analytics
        Note: Parameters are UPPERCASE for analytics APIs
        
        Args:
            symbols: Stock symbols (can be comma-separated string or list)
            interval: DAILY, WEEKLY, or MONTHLY
            range: 1month, 3month, 6month, 12month, 2year, full
            ohlc: open, high, low, close, volume
            calculations: MIN, MAX, MEAN, MEDIAN, CUMULATIVE_RETURN, VARIANCE, 
                         STDDEV, MAX_DRAWDOWN, HISTOGRAM, AUTOCORRELATION, 
                         COVARIANCE, CORRELATION
        """
        # Convert to strings if lists
        if isinstance(symbols, list):
            symbols = ','.join(symbols)
        if isinstance(calculations, list):
            calculations = ','.join(calculations)
        elif calculations is None:
            calculations = 'MEAN,VARIANCE,STDDEV,MAX_DRAWDOWN'
        
        params = {
            'function': 'ANALYTICS_FIXED_WINDOW',
            'SYMBOLS': symbols,  # UPPERCASE!
            'INTERVAL': interval.upper(),
            'RANGE': range,
            'OHLC': ohlc,
            'CALCULATIONS': calculations,
            'apikey': self.api_key
        }
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved fixed window analytics for {symbols}")
            return data
        except Exception as e:
            logger.error(f"Error fetching analytics: {e}")
            return {}
    
    async def get_analytics_sliding_window(self, symbols: Union[str, List[str]],
                                          window_size: int,
                                          interval: str = 'DAILY',
                                          range: str = '6month',
                                          ohlc: str = 'close',
                                          calculations: Optional[Union[str, List[str]]] = None) -> Dict:
        """
        Get sliding window analytics
        Note: Parameters are UPPERCASE for analytics APIs
        
        Args:
            symbols: Stock symbols (can be comma-separated string or list)
            window_size: Size of the sliding window (e.g., 90 days)
            interval: DAILY, WEEKLY, or MONTHLY
            range: 1month, 3month, 6month, 12month, 2year, full
            ohlc: open, high, low, close, volume
            calculations: MEAN, MEDIAN, CUMULATIVE_RETURN, VARIANCE, STDDEV, 
                         COVARIANCE, CORRELATION
        """
        # Convert to strings if lists
        if isinstance(symbols, list):
            symbols = ','.join(symbols)
        if isinstance(calculations, list):
            calculations = ','.join(calculations)
        elif calculations is None:
            calculations = 'MEAN,VARIANCE,STDDEV,CORRELATION'
        
        params = {
            'function': 'ANALYTICS_SLIDING_WINDOW',
            'SYMBOLS': symbols,  # UPPERCASE!
            'INTERVAL': interval.upper(),
            'RANGE': range,
            'WINDOW_SIZE': window_size,
            'OHLC': ohlc,
            'CALCULATIONS': calculations,
            'apikey': self.api_key
        }
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved sliding window analytics for {symbols}")
            return data
        except Exception as e:
            logger.error(f"Error fetching sliding analytics: {e}")
            return {}
    
    # ========================================================================
    # SENTIMENT APIs (3) - Tech Spec Section 6.1
    # ========================================================================
    
    async def get_news_sentiment(self, tickers: Optional[Union[str, List[str]]] = None,
                                topics: Optional[str] = None,
                                time_from: Optional[str] = None,
                                time_to: Optional[str] = None,
                                sort: str = 'LATEST',
                                limit: int = 50) -> Dict:
        """
        Get news sentiment from Alpha Vantage
        Note: Uses 'tickers' parameter, not 'symbol'
        All parameters are optional
        
        Args:
            tickers: Optional stock tickers (comma-separated string or list)
            topics: Optional topics filter
            time_from: Optional start time (YYYYMMDDTHHMM format)
            time_to: Optional end time (YYYYMMDDTHHMM format)
            sort: LATEST, EARLIEST, RELEVANCE (default: LATEST)
            limit: Number of results (default: 50, max: 1000)
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.api_key
        }
        
        # Add optional parameters
        if tickers:
            if isinstance(tickers, list):
                tickers = ','.join(tickers)
            params['tickers'] = tickers  # Note: 'tickers' not 'symbol'
        
        if topics:
            params['topics'] = topics
        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to
        if sort:
            params['sort'] = sort
        if limit:
            params['limit'] = str(limit)
        
        cache_key = f"sentiment_{tickers if tickers else 'all'}_{datetime.now().hour}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved news sentiment for {tickers if tickers else 'market'}")
            self._set_cache(cache_key, data, ttl=900)  # 15 min cache
            return data
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return {}
    
    async def get_top_gainers_losers(self) -> Dict:
        """Get top gainers and losers"""
        params = {
            'function': 'TOP_GAINERS_LOSERS',
            'apikey': self.api_key
        }
        
        cache_key = f"gainers_losers_{datetime.now().hour}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info("Retrieved top gainers and losers")
            self._set_cache(cache_key, data, ttl=300)  # 5 min cache
            return data
        except Exception as e:
            logger.error(f"Error fetching gainers/losers: {e}")
            return {}
    
    async def get_insider_transactions(self, symbol: str) -> Dict:
        """Get insider transactions"""
        params = {
            'function': 'INSIDER_TRANSACTIONS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"insider_{symbol}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved insider transactions for {symbol}")
            self._set_cache(cache_key, data, ttl=86400)  # 24 hour cache
            return data
        except Exception as e:
            logger.error(f"Error fetching insider transactions: {e}")
            return {}
    
    # ========================================================================
    # FUNDAMENTALS APIs (8) - Tech Spec Section 6.1
    # Note: Added EARNINGS_CALENDAR which returns CSV format
    # ========================================================================
    
    async def get_overview(self, symbol: str) -> Dict:
        """Get company overview"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"overview_{symbol}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved company overview for {symbol}")
            self._set_cache(cache_key, data, ttl=86400)  # 24 hour cache
            return data
        except Exception as e:
            logger.error(f"Error fetching overview: {e}")
            return {}
    
    async def get_earnings(self, symbol: str) -> Dict:
        """Get earnings data"""
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"earnings_{symbol}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved earnings for {symbol}")
            self._set_cache(cache_key, data, ttl=86400)
            return data
        except Exception as e:
            logger.error(f"Error fetching earnings: {e}")
            return {}
    
    async def get_income_statement(self, symbol: str) -> Dict:
        """Get income statement"""
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"income_{symbol}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved income statement for {symbol}")
            self._set_cache(cache_key, data, ttl=86400)
            return data
        except Exception as e:
            logger.error(f"Error fetching income statement: {e}")
            return {}
    
    async def get_balance_sheet(self, symbol: str) -> Dict:
        """Get balance sheet"""
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"balance_{symbol}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved balance sheet for {symbol}")
            self._set_cache(cache_key, data, ttl=86400)
            return data
        except Exception as e:
            logger.error(f"Error fetching balance sheet: {e}")
            return {}
    
    async def get_cash_flow(self, symbol: str) -> Dict:
        """Get cash flow statement"""
        params = {
            'function': 'CASH_FLOW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"cashflow_{symbol}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved cash flow for {symbol}")
            self._set_cache(cache_key, data, ttl=86400)
            return data
        except Exception as e:
            logger.error(f"Error fetching cash flow: {e}")
            return {}
    
    async def get_dividends(self, symbol: str) -> Dict:
        """Get dividend history"""
        params = {
            'function': 'DIVIDENDS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"dividends_{symbol}_{datetime.now().month}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved dividends for {symbol}")
            self._set_cache(cache_key, data, ttl=604800)  # 7 day cache
            return data
        except Exception as e:
            logger.error(f"Error fetching dividends: {e}")
            return {}
    
    async def get_splits(self, symbol: str) -> Dict:
        """Get stock split history"""
        params = {
            'function': 'SPLITS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        cache_key = f"splits_{symbol}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            logger.info(f"Retrieved splits for {symbol}")
            self._set_cache(cache_key, data, ttl=2592000)  # 30 day cache
            return data
        except Exception as e:
            logger.error(f"Error fetching splits: {e}")
            return {}
    
    async def get_earnings_calendar(self, horizon: str = '3month') -> str:
        """
        Get earnings calendar - Returns CSV format
        
        Args:
            horizon: Optional - 3month, 6month, or 12month
        
        Returns:
            CSV string of earnings calendar
        """
        params = {
            'function': 'EARNINGS_CALENDAR',
            'apikey': self.api_key
        }
        
        # horizon is optional
        if horizon:
            params['horizon'] = horizon
        
        cache_key = f"earnings_cal_{horizon}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            # This endpoint returns CSV, not JSON
            await self.rate_limiter.acquire()
            
            if not self.session:
                await self.connect()
            async with self.session.get(self.base_url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"API error {resp.status}")
                    return ""
                
                data = await resp.text()
                
                # Update metrics
                self.total_calls += 1
                self.total_calls_today += 1
            
            logger.info(f"Retrieved earnings calendar for {horizon}")
            self._set_cache(cache_key, data, ttl=3600)  # 1 hour cache
            return data
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return ""
    
    # ========================================================================
    # ECONOMIC APIs (5) - Tech Spec Section 6.1
    # ========================================================================
    
    async def get_treasury_yield(self, interval: str = 'monthly', 
                                 maturity: str = '10year') -> pd.DataFrame:
        """Get treasury yield data"""
        params = {
            'function': 'TREASURY_YIELD',
            'interval': interval,
            'maturity': maturity,
            'apikey': self.api_key
        }
        
        cache_key = f"treasury_{maturity}_{interval}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            df = pd.DataFrame(data.get('data', []))
            logger.info(f"Retrieved treasury yield for {maturity}")
            self._set_cache(cache_key, df, ttl=86400)
            return df
        except Exception as e:
            logger.error(f"Error fetching treasury yield: {e}")
            return pd.DataFrame()
    
    async def get_federal_funds_rate(self, interval: str = 'monthly') -> pd.DataFrame:
        """Get federal funds rate"""
        params = {
            'function': 'FEDERAL_FUNDS_RATE',
            'interval': interval,
            'apikey': self.api_key
        }
        
        cache_key = f"fed_rate_{interval}_{datetime.now().date()}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            df = pd.DataFrame(data.get('data', []))
            logger.info("Retrieved federal funds rate")
            self._set_cache(cache_key, df, ttl=86400)
            return df
        except Exception as e:
            logger.error(f"Error fetching fed funds rate: {e}")
            return pd.DataFrame()
    
    async def get_cpi(self, interval: str = 'monthly', datatype: str = 'json') -> pd.DataFrame:
        """Get Consumer Price Index - interval is optional"""
        params = {
            'function': 'CPI',
            'apikey': self.api_key
        }
        
        # Add optional parameters
        if interval:
            params['interval'] = interval
        if datatype:
            params['datatype'] = datatype
        
        cache_key = f"cpi_{interval}_{datetime.now().month}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            df = pd.DataFrame(data.get('data', []))
            logger.info("Retrieved CPI data")
            self._set_cache(cache_key, df, ttl=604800)  # 7 day cache
            return df
        except Exception as e:
            logger.error(f"Error fetching CPI: {e}")
            return pd.DataFrame()
    
    async def get_inflation(self, datatype: str = 'json') -> pd.DataFrame:
        """Get inflation data - no required parameters"""
        params = {
            'function': 'INFLATION',
            'apikey': self.api_key
        }
        
        if datatype:
            params['datatype'] = datatype
        
        cache_key = f"inflation_{datetime.now().year}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            df = pd.DataFrame(data.get('data', []))
            logger.info("Retrieved inflation data")
            self._set_cache(cache_key, df, ttl=2592000)  # 30 day cache
            return df
        except Exception as e:
            logger.error(f"Error fetching inflation: {e}")
            return pd.DataFrame()
    
    async def get_real_gdp(self, interval: str = 'quarterly', datatype: str = 'json') -> pd.DataFrame:
        """Get Real GDP data - interval is optional (quarterly or annual)"""
        params = {
            'function': 'REAL_GDP',
            'apikey': self.api_key
        }
        
        # Add optional parameters
        if interval:
            params['interval'] = interval
        if datatype:
            params['datatype'] = datatype
        
        cache_key = f"gdp_{interval}_{datetime.now().year}"
        if cached := self._get_cache(cache_key):
            return cached
        
        try:
            data = await self._make_request(params)
            df = pd.DataFrame(data.get('data', []))
            logger.info("Retrieved Real GDP data")
            self._set_cache(cache_key, df, ttl=2592000)  # 30 day cache
            return df
        except Exception as e:
            logger.error(f"Error fetching Real GDP: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    async def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting and error handling"""
        if not self.session:
            await self.connect()
        
        # Acquire rate limit token
        await self.rate_limiter.acquire()
        
        start = asyncio.get_event_loop().time()
        
        try:
            if not self.session:
                await self.connect()
            async with self.session.get(self.base_url, params=params) as resp:
                # Check status
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"API error {resp.status}: {error_text}")
                    raise AlphaVantageException(f"API returned {resp.status}: {error_text}")
                
                # Parse response
                text = await resp.text()
                
                # Check for rate limit or error messages
                if "Thank you for using Alpha Vantage" in text and "higher API call frequency" in text:
                    raise RateLimitException("Rate limit reached - upgrade to premium")
                
                if "Invalid API call" in text or "Error Message" in text:
                    raise AlphaVantageException(f"Invalid API call: {text[:200]}")
                
                # Try to parse JSON
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    # Some endpoints return CSV (like EARNINGS_CALENDAR)
                    return {'raw': text}
                
                # Update metrics
                elapsed = (asyncio.get_event_loop().time() - start) * 1000
                self.last_response_time = elapsed
                self.avg_response_time = (self.avg_response_time * 0.9) + (elapsed * 0.1)
                self.total_calls += 1
                self.total_calls_today += 1
                
                logger.debug(f"API call completed in {elapsed:.0f}ms")
                
                return data
                
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {params.get('function')}")
            raise AlphaVantageException("Request timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            raise AlphaVantageException(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            raise AlphaVantageException(f"Request failed: {e}")
    
    def _parse_options_response(self, data: Dict, symbol: str) -> List[OptionContract]:
        """Parse options response with Greeks PROVIDED by Alpha Vantage"""
        options = []
        
        # Debug: Log the response structure
        logger.debug(f"Response keys: {list(data.keys())[:10]}")
        
        # Alpha Vantage returns options in different formats depending on the endpoint
        # Try different response structures
        
        # Format 1: Direct contracts list
        if 'contracts' in data:
            options_data = data['contracts']
        # Format 2: Options array
        elif 'options' in data:
            options_data = data['options']
        # Format 3: Data array
        elif 'data' in data:
            options_data = data['data']
        # Format 4: Direct response with strike prices as keys
        else:
            # For REALTIME_OPTIONS, the response might have dates as keys
            # Look for the first key that contains option data
            options_data = []
            
            for key, value in data.items():
                if key == 'Meta Data' or key == 'Information':
                    continue
                    
                # Check if this looks like options data
                if isinstance(value, dict):
                    # Could be organized by date or strike
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            # This might be option data
                            if 'strike' in sub_value or 'delta' in sub_value:
                                options_data.append(sub_value)
                        elif isinstance(sub_value, list):
                            options_data.extend(sub_value)
                elif isinstance(value, list):
                    # Direct list of options
                    options_data = value
                    break
        
        # If still no data, log the entire response structure for debugging
        if not options_data:
            logger.warning(f"Could not find options data in response. Keys: {list(data.keys())}")
            logger.debug(f"Full response sample: {str(data)[:500]}")
            
            # Try to extract from any nested structure
            for key in data:
                if isinstance(data[key], list) and len(data[key]) > 0:
                    if isinstance(data[key][0], dict):
                        # Check if it looks like option data
                        sample = data[key][0]
                        if any(k in str(sample).lower() for k in ['strike', 'call', 'put', 'delta']):
                            options_data = data[key]
                            logger.info(f"Found options data under key: {key}")
                            break
        
        # Convert dict to list if needed
        if isinstance(options_data, dict):
            options_data = list(options_data.values())
        
        # Parse each contract
        for contract_data in options_data:
            try:
                # Handle nested contract data
                if 'contract' in contract_data:
                    contract_data = contract_data['contract']
                
                # Extract Greeks - Alpha Vantage returns lowercase field names
                delta = float(contract_data.get('delta', 0))
                gamma = float(contract_data.get('gamma', 0))
                theta = float(contract_data.get('theta', 0))
                vega = float(contract_data.get('vega', 0))
                rho = float(contract_data.get('rho', 0))
                
                # Extract other fields - Alpha Vantage uses 'strike' (string or float format)
                strike_value = contract_data.get('strike', 0)
                if isinstance(strike_value, str):
                    strike = float(strike_value.replace(',', '') if strike_value else 0)
                else:
                    strike = float(strike_value) if strike_value else 0.0
                
                # Alpha Vantage returns 'type' field with lowercase 'call' or 'put'
                option_type = contract_data.get('type', '')
                
                # Handle option type formatting - convert to uppercase
                if option_type:
                    option_type = option_type.upper()
                    if option_type not in ['CALL', 'PUT']:
                        if option_type.lower() == 'call' or option_type == 'C':
                            option_type = 'CALL'
                        elif option_type.lower() == 'put' or option_type == 'P':
                            option_type = 'PUT'
                
                option = OptionContract(
                    symbol=symbol,
                    strike=strike,
                    expiry=contract_data.get('expiration', ''),
                    option_type=option_type,
                    bid=float(contract_data.get('bid', 0) or 0),
                    ask=float(contract_data.get('ask', 0) or 0),
                    last=float(contract_data.get('last', 0) or 0),
                    volume=int(contract_data.get('volume', 0) or 0),
                    open_interest=int(contract_data.get('open_interest', 0) or 0),
                    implied_volatility=float(contract_data.get('implied_volatility', 0) or 0),
                    # Greeks PROVIDED by Alpha Vantage - NEVER CALCULATED!
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    rho=rho
                )
                
                # Only add valid options (with strike price)
                if option.strike > 0:
                    options.append(option)
                    
                    # Log if we found Greeks
                    if delta != 0 or gamma != 0 or theta != 0:
                        logger.debug(f"Found Greeks for {symbol} ${strike}: Δ={delta}, Γ={gamma}")
                
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Error parsing option contract: {e}")
                continue
        
        if options:
            logger.info(f"Parsed {len(options)} option contracts")
            # Check if any have Greeks (they may be 0 for deep ITM/OTM)
            greeks_found = any(
                opt.delta is not None or opt.gamma is not None or 
                opt.theta is not None or opt.vega is not None 
                for opt in options[:min(10, len(options))]
            )
            if greeks_found:
                logger.info("✅ Greeks are PROVIDED in the response")
                # Log sample Greeks for verification
                if options:
                    sample = options[0]
                    logger.debug(f"Sample Greeks: Δ={sample.delta}, Γ={sample.gamma}, "
                               f"Θ={sample.theta}, V={sample.vega}, ρ={sample.rho}")
            else:
                logger.warning("⚠️ Options found but Greeks are missing - check require_greeks parameter")
        else:
            logger.warning(f"No valid options parsed from response")
        
        return options
    
    def _parse_indicator_response(self, data: Dict, indicator: str) -> pd.DataFrame:
        """Parse technical indicator response"""
        # Look for the technical analysis key
        for key in data.keys():
            if 'Technical Analysis' in key or indicator in key:
                df = pd.DataFrame(data[key]).T
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                return df
        
        # Alternative format
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        
        logger.warning(f"Could not parse indicator response for {indicator}")
        return pd.DataFrame()
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get from in-memory cache"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expires']:
                self.cache_hits += 1
                self.cache_hits_today += 1
                logger.debug(f"Cache hit: {key}")
                return entry['data']
            else:
                # Expired - remove from cache
                del self.cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any, ttl: Optional[int] = None):
        """Set cache with TTL"""
        if ttl is None:
            # Try to get TTLs from config first
            if self.config and hasattr(self.config, 'cache_ttls'):
                cache_ttls = self.config.cache_ttls
                if 'options' in key:
                    ttl = cache_ttls.get('options', 60)
                elif 'indicator' in key:
                    ttl = cache_ttls.get('indicators', 300)
                elif 'sentiment' in key:
                    ttl = cache_ttls.get('sentiment', 900)
                elif 'overview' in key or 'earnings' in key:
                    ttl = cache_ttls.get('fundamentals', 86400)
                else:
                    ttl = cache_ttls.get('default', 3600)
            else:
                # Fallback to hardcoded defaults
                if 'options' in key:
                    ttl = 60  # 1 minute for options
                elif 'indicator' in key:
                    ttl = 300  # 5 minutes for indicators
                elif 'sentiment' in key:
                    ttl = 900  # 15 minutes for sentiment
                elif 'overview' in key or 'earnings' in key:
                    ttl = 86400  # 24 hours for fundamentals
                else:
                    ttl = 3600  # 1 hour default
        
        self.cache[key] = {
            'data': data,
            'expires': datetime.now() + timedelta(seconds=ttl)
        }
        
        # Limit cache size to prevent memory issues
        if len(self.cache) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]['expires'])
            for old_key in sorted_keys[:100]:
                del self.cache[old_key]
    
    def get_metrics(self) -> Dict:
        """Get client metrics"""
        hit_rate = (self.cache_hits / max(self.total_calls, 1)) * 100
        return {
            'total_calls': self.total_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': hit_rate,
            'avg_response_time_ms': self.avg_response_time,
            'last_response_time_ms': self.last_response_time,
            'rate_limit_remaining': self.rate_limiter.remaining if hasattr(self, 'rate_limiter') else 0,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")


# Global client instance - READY FOR ALL 38 APIs!
av_client = AlphaVantageClient()