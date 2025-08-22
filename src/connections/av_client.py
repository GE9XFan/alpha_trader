#!/usr/bin/env python3
"""
Alpha Vantage Client - Complete implementation of all 35 APIs
Phase 1: Batch implementation with ZERO hardcoded values
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import requests
from requests.exceptions import RequestException, Timeout, HTTPError

from src.foundation.config_manager import get_config_manager
from src.data.cache_manager import get_cache_manager
from src.data.rate_limiter import RateLimiter


class AlphaVantageClient:
    """Complete Alpha Vantage API client for all 35 endpoints - Zero hardcoded values"""
    
    def __init__(self):
        """Initialize Alpha Vantage client with configuration"""
        self.config = get_config_manager()
        self.cache = get_cache_manager()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load Alpha Vantage configuration - fail fast if missing
        try:
            self.av_config = self.config.av_config
            self.base_url = self.av_config['base_url']
            self.api_key = self.av_config['api_key']
            
            if not self.api_key:
                raise ValueError("Alpha Vantage API key not configured")
            
            # Initialize rate limiter - all values from config, no defaults
            rate_config = self.av_config['rate_limit']
            self.rate_limiter = RateLimiter(
                calls_per_minute=rate_config['calls_per_minute'],
                burst_capacity=rate_config['burst_capacity'],
                refill_rate=rate_config['refill_rate'],
                time_window=rate_config['time_window'],
                check_interval=rate_config['check_interval'],
                initial_tokens=rate_config['initial_tokens'],
                initial_total_calls=rate_config['initial_total_calls'],
                initial_rejected_calls=rate_config['initial_rejected_calls'],
                initial_window_calls=rate_config['initial_window_calls']
            )
            
            # Load configurations - fail fast if missing
            self.endpoints = self.av_config['endpoints']
            self.retry_config = self.av_config['retry']
            self.timeout_config = self.av_config['timeout']
            
        except KeyError as e:
            raise ValueError(f"Required Alpha Vantage configuration missing: {e}")
        
        self.logger.info("AlphaVantageClient initialized with 35 API endpoints")
    
    def _make_av_request(self, endpoint_name: str, params: Dict[str, Any], 
                        use_cache: bool = True) -> Optional[Dict]:
        """
        Make Alpha Vantage API request with rate limiting, caching, and retry logic
        
        Args:
            endpoint_name: Name of the endpoint configuration
            params: Request parameters
            use_cache: Whether to use caching
            
        Returns:
            API response data or None if failed
        """
        try:
            # Get endpoint configuration - fail fast if missing
            if endpoint_name not in self.endpoints:
                raise ValueError(f"Endpoint configuration not found: {endpoint_name}")
            
            endpoint_config = self.endpoints[endpoint_name]
            
            # Build cache key
            cache_key = f"av:{endpoint_name}:{params.get('symbol', 'global')}:{hash(str(sorted(params.items())))}"
            
            # Check cache first
            if use_cache:
                cached_data = self.cache.get(cache_key, prefix='av')
                if cached_data:
                    self.logger.debug(f"Cache hit for {endpoint_name}")
                    return cached_data
            
            # Rate limiting
            self.rate_limiter.acquire()
            
            # Add API key to parameters
            request_params = params.copy()
            request_params['apikey'] = self.api_key
            
            # Make request with retry logic
            response_data = self._make_request_with_retry(request_params, endpoint_name)
            
            if response_data:
                # Cache successful response - TTL from config only
                if use_cache:
                    cache_ttl = endpoint_config['cache_ttl']
                    self.cache.set(cache_key, response_data, ttl=cache_ttl, prefix='av')
                
                self.logger.debug(f"Successful API call: {endpoint_name}")
                return response_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in {endpoint_name} request: {e}")
            return None
    
    def _make_request_with_retry(self, params: Dict[str, Any], 
                                endpoint_name: str) -> Optional[Dict]:
        """
        Make HTTP request with retry logic - all values from config
        
        Args:
            params: Request parameters
            endpoint_name: Endpoint name for logging
            
        Returns:
            Response data or None
        """
        # All retry values from config - no fallbacks
        max_attempts = self.retry_config['max_attempts']
        initial_delay = self.retry_config['initial_delay']
        max_delay = self.retry_config['max_delay']
        exponential_base = self.retry_config['exponential_base']
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=(
                        self.timeout_config['connect'],
                        self.timeout_config['read']
                    )
                )
                
                response.raise_for_status()
                
                # Handle CSV responses (earnings_calendar returns CSV by default)
                if endpoint_name == 'earnings_calendar' and 'datatype' not in params:
                    # Return CSV content as text in a dict wrapper
                    return {'csv_data': response.text}
                
                # Handle JSON responses
                try:
                    data = response.json()
                except ValueError as json_error:
                    self.logger.error(f"JSON parsing failed for {endpoint_name}: {json_error}")
                    self.logger.debug(f"Response content: {response.text[:200]}...")
                    return None
                
                # Check for API-specific errors in JSON responses
                if 'Error Message' in data:
                    self.logger.error(f"API Error in {endpoint_name}: {data['Error Message']}")
                    return None
                
                if 'Note' in data:
                    self.logger.warning(f"API Note in {endpoint_name}: {data['Note']}")
                    # Continue with the data as Note is often just rate limit warning
                
                return data
                
            except (RequestException, HTTPError, Timeout) as e:
                delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                
                if attempt < max_attempts - 1:
                    self.logger.warning(f"Request failed for {endpoint_name}, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All retry attempts failed for {endpoint_name}: {e}")
                    return None
            
            except Exception as e:
                self.logger.error(f"Unexpected error in {endpoint_name} request: {e}")
                return None
        
        return None
    
    # ==========================================
    # OPTIONS & GREEKS METHODS (2 APIs)
    # ==========================================
    
    def get_realtime_options(self, symbol: str, require_greeks: Optional[str] = None,
                           datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """
        Get real-time options data with Greeks
        
        Args:
            symbol: Stock symbol
            require_greeks: Whether to require Greeks calculation
            datatype: Response format
            use_cache: Whether to use caching
            
        Returns:
            Options data with Greeks
        """
        endpoint_config = self.endpoints['realtime_options']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'require_greeks': require_greeks or default_params['require_greeks'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('realtime_options', params, use_cache)
    
    def get_historical_options(self, symbol: str, date: Optional[str] = None,
                             datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """
        Get historical options data
        
        Args:
            symbol: Stock symbol
            date: Date for historical data (YYYY-MM-DD format)
            datatype: Response format
            use_cache: Whether to use caching
            
        Returns:
            Historical options data
        """
        endpoint_config = self.endpoints['historical_options']
        default_params = endpoint_config['default_params']
        
        # Default to one month ago if no date specified
        if not date:
            date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'date': date,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('historical_options', params, use_cache)
    
    # ==========================================
    # TECHNICAL INDICATORS METHODS (16 APIs)
    # ==========================================
    
    def get_rsi(self, symbol: str, interval: Optional[str] = None, 
                time_period: Optional[int] = None, series_type: Optional[str] = None,
                datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get RSI (Relative Strength Index) data"""
        endpoint_config = self.endpoints['rsi']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'series_type': series_type or default_params['series_type'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('rsi', params, use_cache)
    
    def get_macd(self, symbol: str, interval: Optional[str] = None,
                 series_type: Optional[str] = None, fastperiod: Optional[int] = None,
                 slowperiod: Optional[int] = None, signalperiod: Optional[int] = None,
                 datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get MACD (Moving Average Convergence Divergence) data"""
        endpoint_config = self.endpoints['macd']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'series_type': series_type or default_params['series_type'],
            'fastperiod': fastperiod or default_params['fastperiod'],
            'slowperiod': slowperiod or default_params['slowperiod'],
            'signalperiod': signalperiod or default_params['signalperiod'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('macd', params, use_cache)
    
    def get_stoch(self, symbol: str, interval: Optional[str] = None,
                  fastkperiod: Optional[int] = None, slowkperiod: Optional[int] = None,
                  slowdperiod: Optional[int] = None, slowkmatype: Optional[int] = None,
                  slowdmatype: Optional[int] = None, datatype: Optional[str] = None,
                  use_cache: bool = True) -> Optional[Dict]:
        """Get Stochastic Oscillator data"""
        endpoint_config = self.endpoints['stoch']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'fastkperiod': fastkperiod or default_params['fastkperiod'],
            'slowkperiod': slowkperiod or default_params['slowkperiod'],
            'slowdperiod': slowdperiod or default_params['slowdperiod'],
            'slowkmatype': slowkmatype or default_params['slowkmatype'],
            'slowdmatype': slowdmatype or default_params['slowdmatype'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('stoch', params, use_cache)
    
    def get_willr(self, symbol: str, interval: Optional[str] = None,
                  time_period: Optional[int] = None, datatype: Optional[str] = None,
                  use_cache: bool = True) -> Optional[Dict]:
        """Get Williams %R data"""
        endpoint_config = self.endpoints['willr']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('willr', params, use_cache)
    
    def get_mom(self, symbol: str, interval: Optional[str] = None,
                time_period: Optional[int] = None, series_type: Optional[str] = None,
                datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get Momentum data"""
        endpoint_config = self.endpoints['mom']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'series_type': series_type or default_params['series_type'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('mom', params, use_cache)
    
    def get_bbands(self, symbol: str, interval: Optional[str] = None,
                   time_period: Optional[int] = None, series_type: Optional[str] = None,
                   nbdevup: Optional[int] = None, nbdevdn: Optional[int] = None,
                   matype: Optional[int] = None, datatype: Optional[str] = None,
                   use_cache: bool = True) -> Optional[Dict]:
        """Get Bollinger Bands data"""
        endpoint_config = self.endpoints['bbands']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'series_type': series_type or default_params['series_type'],
            'nbdevup': nbdevup or default_params['nbdevup'],
            'nbdevdn': nbdevdn or default_params['nbdevdn'],
            'matype': matype or default_params['matype'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('bbands', params, use_cache)
    
    def get_atr(self, symbol: str, interval: Optional[str] = None,
                time_period: Optional[int] = None, datatype: Optional[str] = None,
                use_cache: bool = True) -> Optional[Dict]:
        """Get Average True Range data"""
        endpoint_config = self.endpoints['atr']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('atr', params, use_cache)
    
    def get_adx(self, symbol: str, interval: Optional[str] = None,
                time_period: Optional[int] = None, datatype: Optional[str] = None,
                use_cache: bool = True) -> Optional[Dict]:
        """Get Average Directional Index data"""
        endpoint_config = self.endpoints['adx']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('adx', params, use_cache)
    
    def get_aroon(self, symbol: str, interval: Optional[str] = None,
                  time_period: Optional[int] = None, datatype: Optional[str] = None,
                  use_cache: bool = True) -> Optional[Dict]:
        """Get Aroon data"""
        endpoint_config = self.endpoints['aroon']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('aroon', params, use_cache)
    
    def get_cci(self, symbol: str, interval: Optional[str] = None,
                time_period: Optional[int] = None, datatype: Optional[str] = None,
                use_cache: bool = True) -> Optional[Dict]:
        """Get Commodity Channel Index data"""
        endpoint_config = self.endpoints['cci']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('cci', params, use_cache)
    
    def get_ema(self, symbol: str, interval: Optional[str] = None,
                time_period: Optional[int] = None, series_type: Optional[str] = None,
                datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get Exponential Moving Average data"""
        endpoint_config = self.endpoints['ema']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'series_type': series_type or default_params['series_type'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('ema', params, use_cache)
    
    def get_sma(self, symbol: str, interval: Optional[str] = None,
                time_period: Optional[int] = None, series_type: Optional[str] = None,
                datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get Simple Moving Average data"""
        endpoint_config = self.endpoints['sma']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'series_type': series_type or default_params['series_type'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('sma', params, use_cache)
    
    def get_mfi(self, symbol: str, interval: Optional[str] = None,
                time_period: Optional[int] = None, datatype: Optional[str] = None,
                use_cache: bool = True) -> Optional[Dict]:
        """Get Money Flow Index data"""
        endpoint_config = self.endpoints['mfi']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'time_period': time_period or default_params['time_period'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('mfi', params, use_cache)
    
    def get_obv(self, symbol: str, interval: Optional[str] = None,
                datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get On Balance Volume data"""
        endpoint_config = self.endpoints['obv']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('obv', params, use_cache)
    
    def get_ad(self, symbol: str, interval: Optional[str] = None,
               datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get Accumulation/Distribution data"""
        endpoint_config = self.endpoints['ad']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('ad', params, use_cache)
    
    def get_vwap(self, symbol: str, interval: Optional[str] = None,
                 datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get Volume Weighted Average Price data"""
        endpoint_config = self.endpoints['vwap']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'interval': interval or default_params['interval'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('vwap', params, use_cache)
    
    # ==========================================
    # ANALYTICS METHODS (2 APIs)
    # ==========================================
    
    def get_analytics_fixed_window(self, symbols: Optional[str] = None, 
                                  interval: Optional[str] = None, ohlc: Optional[str] = None,
                                  range_param: Optional[str] = None, calculations: Optional[str] = None,
                                  datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get analytics with fixed window calculations"""
        endpoint_config = self.endpoints['analytics_fixed_window']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'SYMBOLS': symbols or default_params['SYMBOLS'],
            'INTERVAL': interval or default_params['INTERVAL'],
            'OHLC': ohlc or default_params['OHLC'],
            'RANGE': range_param or default_params['RANGE'],
            'CALCULATIONS': calculations or default_params['CALCULATIONS'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('analytics_fixed_window', params, use_cache)
    
    def get_analytics_sliding_window(self, symbols: Optional[str] = None,
                                   interval: Optional[str] = None, ohlc: Optional[str] = None,
                                   range_param: Optional[str] = None, window_size: Optional[int] = None,
                                   calculations: Optional[str] = None, datatype: Optional[str] = None,
                                   use_cache: bool = True) -> Optional[Dict]:
        """Get analytics with sliding window calculations"""
        endpoint_config = self.endpoints['analytics_sliding_window']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'SYMBOLS': symbols or default_params['SYMBOLS'],
            'INTERVAL': interval or default_params['INTERVAL'],
            'OHLC': ohlc or default_params['OHLC'],
            'RANGE': range_param or default_params['RANGE'],
            'WINDOW_SIZE': window_size or default_params['WINDOW_SIZE'],
            'CALCULATIONS': calculations or default_params['CALCULATIONS'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('analytics_sliding_window', params, use_cache)
    
    # ==========================================
    # SENTIMENT & NEWS METHODS (3 APIs)
    # ==========================================
    
    def get_news_sentiment(self, tickers: Optional[str] = None, sort: Optional[str] = None,
                          limit: Optional[int] = None, datatype: Optional[str] = None,
                          use_cache: bool = True) -> Optional[Dict]:
        """Get news sentiment data"""
        endpoint_config = self.endpoints['news_sentiment']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'tickers': tickers or default_params['tickers'],
            'sort': sort or default_params['sort'],
            'limit': limit or default_params['limit'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('news_sentiment', params, use_cache)
    
    def get_top_gainers_losers(self, datatype: Optional[str] = None,
                              use_cache: bool = True) -> Optional[Dict]:
        """Get top gainers, losers, and most actively traded stocks"""
        endpoint_config = self.endpoints['top_gainers_losers']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('top_gainers_losers', params, use_cache)
    
    def get_insider_transactions(self, symbol: str, datatype: Optional[str] = None,
                               use_cache: bool = True) -> Optional[Dict]:
        """Get insider transaction data"""
        endpoint_config = self.endpoints['insider_transactions']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('insider_transactions', params, use_cache)
    
    # ==========================================
    # FUNDAMENTALS METHODS (7 APIs)
    # ==========================================
    
    def get_overview(self, symbol: str, datatype: Optional[str] = None,
                    use_cache: bool = True) -> Optional[Dict]:
        """Get company overview"""
        endpoint_config = self.endpoints['overview']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('overview', params, use_cache)
    
    def get_earnings_calendar(self, use_cache: bool = True) -> Optional[Dict]:
        """Get earnings calendar"""
        endpoint_config = self.endpoints['earnings_calendar']
        
        params = {
            'function': endpoint_config['function']
            # Returns CSV by default, no datatype parameter needed per API docs
        }
        
        return self._make_av_request('earnings_calendar', params, use_cache)
    
    def get_income_statement(self, symbol: str, datatype: Optional[str] = None,
                           use_cache: bool = True) -> Optional[Dict]:
        """Get income statement"""
        endpoint_config = self.endpoints['income_statement']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('income_statement', params, use_cache)
    
    def get_balance_sheet(self, symbol: str, datatype: Optional[str] = None,
                         use_cache: bool = True) -> Optional[Dict]:
        """Get balance sheet"""
        endpoint_config = self.endpoints['balance_sheet']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('balance_sheet', params, use_cache)
    
    def get_cash_flow(self, symbol: str, datatype: Optional[str] = None,
                     use_cache: bool = True) -> Optional[Dict]:
        """Get cash flow statement"""
        endpoint_config = self.endpoints['cash_flow']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('cash_flow', params, use_cache)
    
    def get_dividends(self, symbol: str, datatype: Optional[str] = None,
                     use_cache: bool = True) -> Optional[Dict]:
        """Get dividend data"""
        endpoint_config = self.endpoints['dividends']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('dividends', params, use_cache)
    
    def get_splits(self, symbol: str, datatype: Optional[str] = None,
                  use_cache: bool = True) -> Optional[Dict]:
        """Get stock split data"""
        endpoint_config = self.endpoints['splits']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'symbol': symbol,
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('splits', params, use_cache)
    
    # ==========================================
    # ECONOMIC INDICATORS METHODS (5 APIs)
    # ==========================================
    
    def get_treasury_yield(self, interval: Optional[str] = None, 
                          maturity: Optional[str] = None, datatype: Optional[str] = None,
                          use_cache: bool = True) -> Optional[Dict]:
        """Get treasury yield data"""
        endpoint_config = self.endpoints['treasury_yield']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'interval': interval or default_params['interval'],
            'maturity': maturity or default_params['maturity'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('treasury_yield', params, use_cache)
    
    def get_federal_funds_rate(self, interval: Optional[str] = None,
                              datatype: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """Get federal funds rate data"""
        endpoint_config = self.endpoints['federal_funds_rate']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'interval': interval or default_params['interval'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('federal_funds_rate', params, use_cache)
    
    def get_cpi(self, interval: Optional[str] = None, datatype: Optional[str] = None,
                use_cache: bool = True) -> Optional[Dict]:
        """Get Consumer Price Index data"""
        endpoint_config = self.endpoints['cpi']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'interval': interval or default_params['interval'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('cpi', params, use_cache)
    
    def get_inflation(self, datatype: Optional[str] = None,
                     use_cache: bool = True) -> Optional[Dict]:
        """Get inflation data"""
        endpoint_config = self.endpoints['inflation']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('inflation', params, use_cache)
    
    def get_real_gdp(self, interval: Optional[str] = None, datatype: Optional[str] = None,
                    use_cache: bool = True) -> Optional[Dict]:
        """Get Real GDP data"""
        endpoint_config = self.endpoints['real_gdp']
        default_params = endpoint_config['default_params']
        
        params = {
            'function': endpoint_config['function'],
            'interval': interval or default_params['interval'],
            'datatype': datatype or default_params['datatype']
        }
        
        return self._make_av_request('real_gdp', params, use_cache)


# Singleton instance
_av_client = None


def get_av_client() -> AlphaVantageClient:
    """Get or create singleton AlphaVantageClient instance"""
    global _av_client
    if _av_client is None:
        _av_client = AlphaVantageClient()
    return _av_client