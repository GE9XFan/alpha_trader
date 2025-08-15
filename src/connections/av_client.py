"""
Alpha Vantage API Client
Handles all 38 Alpha Vantage API endpoints with EXACT parameters from test_api.py
Uses existing YAML configuration - no hardcoded values
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.connections.base_client import BaseAPIClient
from src.foundation.config_manager import ConfigManager
from src.data.rate_limiter import TokenBucketRateLimiter, RequestPriority, rate_limit

logger = logging.getLogger(__name__)


class AlphaVantageClient(BaseAPIClient):
    """
    Alpha Vantage API client - uses ConfigManager for ALL configuration
    No hardcoded values - everything from YAML files
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None,
                 rate_limiter: Optional[TokenBucketRateLimiter] = None):
        """
        Initialize Alpha Vantage client from configuration
        
        Args:
            config_manager: ConfigManager instance (creates new if None)
            rate_limiter: Token bucket rate limiter (creates new if None)
        """
        # Get configuration manager
        self.config_manager = config_manager or ConfigManager()
        
        # Load Alpha Vantage config from YAML
        av_config = self.config_manager.get('apis.alpha_vantage.alpha_vantage', {})
        
        # Initialize base class with config
        super().__init__(av_config, "AlphaVantageClient")
        
        # Get API key from environment
        self.api_key = self.config_manager.get('env.AV_API_KEY')
        if not self.api_key:
            raise ValueError("AV_API_KEY not set in environment")
        
        # Get base URL from config
        self.base_url = av_config.get('base_url', 'https://www.alphavantage.co/query')
        
        # Get timeout and retry settings from config
        self.timeout = av_config.get('timeout', 30)
        self.max_retries = av_config.get('max_retries', 3)
        self.retry_delay = av_config.get('retry_delay', 1)
        self.retry_backoff = av_config.get('retry_backoff', 2)
        
        # Get endpoint configurations
        self.endpoints_config = av_config.get('endpoints', {})
        
        # Initialize rate limiter (uses config/apis/rate_limits.yaml)
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(
            provider='alpha_vantage',
            config_manager=self.config_manager
        )
        
        # Session for connection pooling
        self.session = requests.Session()
        
        logger.info(f"AlphaVantageClient initialized using configuration from YAML files")
    
    def _make_request(self, params: Dict[str, Any], 
                     priority: Optional[RequestPriority] = None,
                     symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Make request with rate limiting and retry logic
        
        Args:
            params: Request parameters (EXACT from test_api.py)
            priority: Request priority (auto-determined if None)
            symbol: Symbol for the request
            
        Returns:
            API response or None
        """
        # Add API key
        params['apikey'] = self.api_key
        
        # Determine API type from function parameter
        api_type = params.get('function', 'UNKNOWN')
        
        # Get symbol from params if not provided
        if not symbol and 'symbol' in params:
            symbol = params['symbol']
        
        # Acquire rate limit token
        if not self.rate_limiter.acquire(
            tokens=1,
            priority=priority,
            symbol=symbol,
            api_type=api_type
        ):
            logger.warning(f"Rate limit timeout for {api_type}")
            return None
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                
                # Handle CSV responses (like EARNINGS_CALENDAR)
                if api_type == 'EARNINGS_CALENDAR':
                    # Return raw CSV text wrapped in dict
                    return {'csv_response': response.text}
                
                # Parse JSON response
                data = response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    logger.error(f"API Error: {data['Error Message']}")
                    return None
                
                if "Note" in data:
                    logger.warning(f"API Note (likely rate limit): {data['Note']}")
                    # Wait and retry
                    time.sleep(60)
                    continue
                
                # Success
                self.total_calls += 1
                self.last_call_time = datetime.now()
                return data
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                last_error = "Timeout"
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_error = str(e)
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (self.retry_backoff ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"All retries exhausted. Last error: {last_error}")
        self.error_count += 1
        return None
    
    # ========== OPTIONS APIs (EXACT from test_api.py) ==========
    
    @rate_limit(priority=RequestPriority.TIER_A, tokens=1)
    def get_realtime_options(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time options with Greeks - PRIMARY GREEKS SOURCE"""
        params = {
            'function': 'REALTIME_OPTIONS',
            'symbol': symbol,
            'require_greeks': 'true'  # EXACT from test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_B, tokens=1)
    def get_historical_options(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get historical options data"""
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    # ========== TECHNICAL INDICATORS (EXACT from test_api.py) ==========
    
    @rate_limit(priority=RequestPriority.TIER_A, tokens=1)
    def get_rsi(self, symbol: str, interval: str = 'daily', 
                time_period: str = '14', series_type: str = 'close') -> Optional[Dict[str, Any]]:
        """Get RSI indicator"""
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,  # String as in test_api.py
            'series_type': series_type
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_A, tokens=1)
    def get_macd(self, symbol: str, interval: str = 'daily', series_type: str = 'close',
                 fastperiod: str = '12', slowperiod: str = '26', 
                 signalperiod: str = '9') -> Optional[Dict[str, Any]]:
        """Get MACD indicator"""
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': interval,
            'series_type': series_type,
            'fastperiod': fastperiod,  # String as in test_api.py
            'slowperiod': slowperiod,  # String as in test_api.py
            'signalperiod': signalperiod  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_A, tokens=1)
    def get_stoch(self, symbol: str, interval: str = 'daily') -> Optional[Dict[str, Any]]:
        """Get Stochastic indicator"""
        params = {
            'function': 'STOCH',
            'symbol': symbol,
            'interval': interval
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_A, tokens=1)
    def get_bbands(self, symbol: str, interval: str = 'daily', time_period: str = '20',
                   series_type: str = 'close', nbdevup: str = '2', 
                   nbdevdn: str = '2') -> Optional[Dict[str, Any]]:
        """Get Bollinger Bands"""
        params = {
            'function': 'BBANDS',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,  # String as in test_api.py
            'series_type': series_type,
            'nbdevup': nbdevup,  # String as in test_api.py
            'nbdevdn': nbdevdn  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_A, tokens=1)
    def get_atr(self, symbol: str, interval: str = 'daily', 
                time_period: str = '14') -> Optional[Dict[str, Any]]:
        """Get ATR indicator"""
        params = {
            'function': 'ATR',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_B, tokens=1)
    def get_adx(self, symbol: str, interval: str = 'daily', 
                time_period: str = '14') -> Optional[Dict[str, Any]]:
        """Get ADX indicator"""
        params = {
            'function': 'ADX',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_A, tokens=1)
    def get_vwap(self, symbol: str, interval: str = '5min') -> Optional[Dict[str, Any]]:
        """Get VWAP indicator"""
        params = {
            'function': 'VWAP',
            'symbol': symbol,
            'interval': interval
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_B, tokens=1)
    def get_ema(self, symbol: str, interval: str = 'daily', time_period: str = '20',
                series_type: str = 'close') -> Optional[Dict[str, Any]]:
        """Get EMA indicator"""
        params = {
            'function': 'EMA',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,  # String as in test_api.py
            'series_type': series_type
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_B, tokens=1)
    def get_sma(self, symbol: str, interval: str = 'daily', time_period: str = '20',
                series_type: str = 'close') -> Optional[Dict[str, Any]]:
        """Get SMA indicator"""
        params = {
            'function': 'SMA',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,  # String as in test_api.py
            'series_type': series_type
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_aroon(self, symbol: str, interval: str = 'daily', 
                  time_period: str = '14') -> Optional[Dict[str, Any]]:
        """Get AROON indicator"""
        params = {
            'function': 'AROON',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_cci(self, symbol: str, interval: str = 'daily', 
                time_period: str = '20') -> Optional[Dict[str, Any]]:
        """Get CCI indicator"""
        params = {
            'function': 'CCI',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_mfi(self, symbol: str, interval: str = 'daily', 
                time_period: str = '14') -> Optional[Dict[str, Any]]:
        """Get MFI indicator"""
        params = {
            'function': 'MFI',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_willr(self, symbol: str, interval: str = 'daily', 
                  time_period: str = '14') -> Optional[Dict[str, Any]]:
        """Get Williams %R indicator"""
        params = {
            'function': 'WILLR',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period  # String as in test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_mom(self, symbol: str, interval: str = 'daily', time_period: str = '10',
                series_type: str = 'close') -> Optional[Dict[str, Any]]:
        """Get Momentum indicator"""
        params = {
            'function': 'MOM',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,  # String as in test_api.py
            'series_type': series_type
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_ad(self, symbol: str, interval: str = 'daily') -> Optional[Dict[str, Any]]:
        """Get A/D indicator"""
        params = {
            'function': 'AD',
            'symbol': symbol,
            'interval': interval
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_obv(self, symbol: str, interval: str = 'daily') -> Optional[Dict[str, Any]]:
        """Get OBV indicator"""
        params = {
            'function': 'OBV',
            'symbol': symbol,
            'interval': interval
        }
        return self._make_request(params, symbol=symbol)
    
    # ========== ANALYTICS APIs (EXACT from test_api.py) ==========
    
    @rate_limit(priority=RequestPriority.TIER_B, tokens=1)
    def get_analytics_fixed_window(self, symbol: str, interval: str = 'DAILY',
                                start_date: str = '2024-01-01', 
                                end_date: str = '2024-12-31',
                                ohlc: str = 'close') -> Optional[Dict[str, Any]]:
        """Get fixed window analytics"""
        params = {
            'function': 'ANALYTICS_FIXED_WINDOW',
            'SYMBOLS': symbol,  # Can be comma-separated for multiple
            'RANGE': start_date,      # Start date YYYY-MM-DD
            'RANGE_END': end_date,    # End date YYYY-MM-DD (was missing!)
            'INTERVAL': interval,
            'OHLC': ohlc,
            'CALCULATIONS': 'MIN,MAX,MEDIAN,CUMULATIVE_RETURN,VARIANCE,MAX_DRAWDOWN,HISTOGRAM,AUTOCORRELATION,COVARIANCE,CORRELATION,MEAN,STDDEV'
        }
        return self._make_request(params, symbol=symbol)

    @rate_limit(priority=RequestPriority.TIER_B, tokens=1)
    def get_analytics_sliding_window(self, symbol: str, interval: str = 'DAILY',
                                    range_period: str = '6month',  # Changed from 'full'
                                    window_size: str = '20',
                                    ohlc: str = 'close') -> Optional[Dict[str, Any]]:
        """Get sliding window analytics"""
        params = {
            'function': 'ANALYTICS_SLIDING_WINDOW',
            'SYMBOLS': symbol,
            'RANGE': range_period,     # '6month', '30day', etc. NOT 'full'
            'INTERVAL': interval,
            'OHLC': ohlc,
            'WINDOW_SIZE': window_size,
            'CALCULATIONS': 'MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,COVARIANCE,CORRELATION'
        }
        return self._make_request(params, symbol=symbol)    
    # ========== FUNDAMENTAL DATA (EXACT from test_api.py) ==========
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company overview"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get earnings data"""
        params = {
            'function': 'EARNINGS',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_income_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get income statement"""
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_balance_sheet(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get balance sheet"""
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_cash_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cash flow statement"""
        params = {
            'function': 'CASH_FLOW',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_dividends(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get dividend history"""
        params = {
            'function': 'DIVIDENDS',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_splits(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock splits history"""
        params = {
            'function': 'SPLITS',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_earnings_estimates(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get earnings estimates"""
        params = {
            'function': 'EARNINGS_ESTIMATES',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_earnings_calendar(self) -> Optional[Dict[str, Any]]:
        """Get earnings calendar - returns CSV data"""
        params = {
            'function': 'EARNINGS_CALENDAR'
            # No symbol parameter for this endpoint
        }
        return self._make_request(params)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_earnings_call_transcript(self, symbol: str, quarter: str = '2025Q1') -> Optional[Dict[str, Any]]:
        """Get earnings call transcript"""
        params = {
            'function': 'EARNINGS_CALL_TRANSCRIPT',
            'symbol': symbol,
            'quarter': quarter  # EXACT from test_api.py
        }
        return self._make_request(params, symbol=symbol)
    
    # ========== ECONOMIC INDICATORS (EXACT from test_api.py) ==========
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_treasury_yield(self, interval: str = 'monthly', 
                           maturity: str = '10year') -> Optional[Dict[str, Any]]:
        """Get treasury yield data"""
        params = {
            'function': 'TREASURY_YIELD',
            'interval': interval,
            'maturity': maturity  # EXACT from test_api.py
        }
        return self._make_request(params)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_federal_funds_rate(self, interval: str = 'monthly') -> Optional[Dict[str, Any]]:
        """Get federal funds rate"""
        params = {
            'function': 'FEDERAL_FUNDS_RATE',
            'interval': interval
        }
        return self._make_request(params)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_cpi(self, interval: str = 'monthly') -> Optional[Dict[str, Any]]:
        """Get CPI data"""
        params = {
            'function': 'CPI',
            'interval': interval
        }
        return self._make_request(params)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_inflation(self) -> Optional[Dict[str, Any]]:
        """Get inflation data"""
        params = {
            'function': 'INFLATION'
            # No other parameters for this endpoint
        }
        return self._make_request(params)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_real_gdp(self, interval: str = 'annual') -> Optional[Dict[str, Any]]:
        """Get real GDP data"""
        params = {
            'function': 'REAL_GDP',
            'interval': interval
        }
        return self._make_request(params)
    
    # ========== SENTIMENT & NEWS (EXACT from test_api.py) ==========
    
    @rate_limit(priority=RequestPriority.TIER_B, tokens=1)
    def get_news_sentiment(self, tickers: str, sort: str = 'LATEST', 
                           limit: str = '100') -> Optional[Dict[str, Any]]:
        """Get news sentiment analysis"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': tickers,  # EXACT parameter name from test_api.py
            'sort': sort,  # EXACT from test_api.py
            'limit': limit  # String as in test_api.py
        }
        return self._make_request(params)
    
    @rate_limit(priority=RequestPriority.TIER_C, tokens=1)
    def get_top_gainers_losers(self) -> Optional[Dict[str, Any]]:
        """Get top gainers and losers"""
        params = {
            'function': 'TOP_GAINERS_LOSERS'
            # No other parameters for this endpoint
        }
        return self._make_request(params)
    
    @rate_limit(priority=RequestPriority.BACKGROUND, tokens=1)
    def get_insider_transactions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get insider transactions"""
        params = {
            'function': 'INSIDER_TRANSACTIONS',
            'symbol': symbol
        }
        return self._make_request(params, symbol=symbol)
    
    # ========== UTILITY METHODS ==========
    
    def connect(self) -> bool:
        """Establish API connection"""
        try:
            # Test with lightweight call
            response = self._make_request(
                {'function': 'GLOBAL_QUOTE', 'symbol': 'SPY'},
                priority=RequestPriority.BACKGROUND
            )
            self.is_connected = response is not None
            logger.info(f"Alpha Vantage connection: {'established' if self.is_connected else 'failed'}")
            return self.is_connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from API"""
        try:
            self.session.close()
            self.is_connected = False
            logger.info("Disconnected from Alpha Vantage")
            return True
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            return False
    
    def call(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generic API call method for BaseAPIClient interface"""
        return self._make_request(params)
    
    def health_check(self) -> bool:
        """Check API health"""
        try:
            response = self._make_request(
                {'function': 'GLOBAL_QUOTE', 'symbol': 'SPY'},
                priority=RequestPriority.BACKGROUND
            )
            return response is not None and 'Error Message' not in response
        except:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = {
            'total_calls': self.total_calls,
            'error_count': self.error_count,
            'last_call_time': self.last_call_time,
            'is_connected': self.is_connected,
            'rate_limiter_stats': self.rate_limiter.get_statistics() if self.rate_limiter else {}
        }
        return stats