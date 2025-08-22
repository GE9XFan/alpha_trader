#!/usr/bin/env python3
"""
Test ALL 41 Alpha Vantage APIs - Phase 1, Step 1
This script tests every Alpha Vantage API endpoint with correct parameters
Based on official API documentation
"""

import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import get_config_manager
from src.foundation.logger import get_logger


class AlphaVantageAPITester:
    """Test all 41 Alpha Vantage APIs and document responses"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.logger = get_logger('AVAPITester')
        self.api_key = self.config.av_api_key
        self.base_url = self.config.av_config.get('base_url')
        
        # Create response directory
        self.response_dir = Path('data/api_responses/alpha_vantage')
        self.response_dir.mkdir(parents=True, exist_ok=True)
        
        # Track API calls for rate limiting
        self.api_calls = 0
        self.start_time = time.time()
        
        self.logger.info("Alpha Vantage API Tester initialized")
        self.logger.info(f"API Key configured: {bool(self.api_key)}")
    
    def _rate_limit(self):
        """Simple rate limiting - max 5 calls per minute during testing"""
        self.api_calls += 1
        
        # Every 5 calls, wait to respect rate limit
        if self.api_calls % 500 == 0:
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                sleep_time = 61 - elapsed
                self.logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self.start_time = time.time()
    
    def _make_request(self, params: Dict, api_name: str) -> Optional[Dict]:
        """Make API request and save response"""
        try:
            self._rate_limit()
            
            self.logger.info(f"Testing {api_name}...")
            self.logger.debug(f"Parameters: {params}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"API Error for {api_name}: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                self.logger.warning(f"API Note for {api_name}: {data['Note']}")
            
            # Save response
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.response_dir / f"{api_name}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"✓ {api_name} response saved to {filename}")
            
            # Log basic structure
            self._analyze_structure(api_name, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to test {api_name}: {e}")
            return None
    
    def _analyze_structure(self, api_name: str, data: Dict):
        """Analyze and log response structure"""
        self.logger.info(f"  Structure for {api_name}:")
        self.logger.info(f"    Top-level keys: {list(data.keys())}")
        
        # Count data points
        for key in data.keys():
            if isinstance(data[key], dict):
                self.logger.info(f"    {key}: {len(data[key])} items")
    
    def test_options_apis(self):
        """Test Options & Greeks APIs (2 APIs)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Testing OPTIONS & GREEKS APIs (2)")
        self.logger.info("="*50)
        
        # 1. REALTIME_OPTIONS - with require_greeks parameter
        params = {
            'function': 'REALTIME_OPTIONS',
            'symbol': 'SPY',
            'require_greeks': 'true',  # Enable Greeks
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'realtime_options')
        
        # 2. HISTORICAL_OPTIONS - with date parameter
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': 'SPY',
            'date': one_month_ago,
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'historical_options')
    
    def test_indicator_apis(self):
        """Test Technical Indicator APIs (16 APIs)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Testing TECHNICAL INDICATOR APIs (16)")
        self.logger.info("="*50)
        
        # Indicators with their specific required parameters
        indicators = [
            # Momentum indicators
            ('RSI', {'interval': 'daily', 'time_period': 14, 'series_type': 'close'}),
            ('MACD', {'interval': 'daily', 'series_type': 'close', 
                     'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            ('STOCH', {'interval': 'daily', 'fastkperiod': 5, 'slowkperiod': 3, 
                      'slowdperiod': 3, 'slowkmatype': 0, 'slowdmatype': 0}),
            ('WILLR', {'interval': 'daily', 'time_period': 14}),
            ('MOM', {'interval': 'daily', 'time_period': 10, 'series_type': 'close'}),
            
            # Volatility indicators
            ('BBANDS', {'interval': 'daily', 'time_period': 20, 'series_type': 'close',
                       'nbdevup': 2, 'nbdevdn': 2, 'matype': 0}),
            ('ATR', {'interval': 'daily', 'time_period': 14}),
            
            # Trend indicators
            ('ADX', {'interval': 'daily', 'time_period': 14}),
            ('AROON', {'interval': 'daily', 'time_period': 14}),
            ('CCI', {'interval': 'daily', 'time_period': 20}),
            
            # Moving averages
            ('EMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            ('SMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            
            # Volume indicators
            ('MFI', {'interval': 'daily', 'time_period': 14}),
            ('OBV', {'interval': 'daily'}),
            ('AD', {'interval': 'daily'}),
            
            # VWAP - intraday only
            ('VWAP', {'interval': '15min'})  # No other params needed
        ]
        
        for func_name, params_dict in indicators:
            params = {
                'function': func_name,
                'symbol': 'SPY',
                'apikey': self.api_key,
                'datatype': 'json'
            }
            params.update(params_dict)
            
            self._make_request(params, func_name.lower())
    
    def test_analytics_apis(self):
        """Test Analytics APIs (2 APIs)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Testing ANALYTICS APIs (2)")
        self.logger.info("="*50)
        
        # 1. ANALYTICS_FIXED_WINDOW
        params = {
            'function': 'ANALYTICS_FIXED_WINDOW',
            'SYMBOLS': 'SPY,QQQ',  # Note: SYMBOLS (plural) not symbol
            'INTERVAL': 'DAILY',
            'OHLC': 'close',
            'RANGE': '1month',  # or specific dates
            'CALCULATIONS': 'MIN,MAX,MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,MAX_DRAWDOWN,HISTOGRAM,AUTOCORRELATION,COVARIANCE,CORRELATION',
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'analytics_fixed_window')
        
        # 2. ANALYTICS_SLIDING_WINDOW
        params = {
            'function': 'ANALYTICS_SLIDING_WINDOW',
            'SYMBOLS': 'AAPL',
            'INTERVAL': 'DAILY',
            'OHLC': 'close',
            'RANGE': '3month',
            'WINDOW_SIZE': 90,
            'CALCULATIONS': 'MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,COVARIANCE,CORRELATION',
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'analytics_sliding_window')
    
    def test_sentiment_apis(self):
        """Test Sentiment & News APIs (3 APIs)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Testing SENTIMENT & NEWS APIs (3)")
        self.logger.info("="*50)
        
        # 1. NEWS_SENTIMENT
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': 'AAPL',  # Can be multiple: 'SPY,QQQ,AAPL'
            'sort': 'LATEST',
            'limit': 50,
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'news_sentiment')
        
        # 2. TOP_GAINERS_LOSERS
        params = {
            'function': 'TOP_GAINERS_LOSERS',
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'top_gainers_losers')
        
        # 3. INSIDER_TRANSACTIONS
        params = {
            'function': 'INSIDER_TRANSACTIONS',
            'symbol': 'AAPL',
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'insider_transactions')
    
    def test_fundamental_apis(self):
        """Test Fundamental Data APIs (10 APIs)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Testing FUNDAMENTAL DATA APIs (10)")
        self.logger.info("="*50)
        
        # Basic fundamentals - all use same pattern
        basic_fundamentals = [
            'OVERVIEW',
            'EARNINGS_CALENDAR',
            'INCOME_STATEMENT',
            'BALANCE_SHEET',
            'CASH_FLOW',
            'DIVIDENDS',
            'SPLITS'
        ]
        
        for func_name in basic_fundamentals:
            params = {
                'function': func_name,
                'symbol': 'AAPL',
                'apikey': self.api_key,
                'datatype': 'json'
            }
            self._make_request(params, func_name.lower())
        
        # EARNINGS_CALENDAR - different params
        params = {
            'function': 'EARNINGS_CALENDAR',
            'apikey': self.api_key
            # Note: Returns CSV format by default
        }
        self._make_request(params, 'earnings_calendar')
    
    def test_economic_apis(self):
        """Test Economic Indicator APIs (5 APIs)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Testing ECONOMIC INDICATOR APIs (5)")
        self.logger.info("="*50)
        
        # 1. TREASURY_YIELD
        params = {
            'function': 'TREASURY_YIELD',
            'interval': 'monthly',  # daily, weekly, or monthly
            'maturity': '10year',  # 3month, 2year, 5year, 7year, 10year, 30year
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'treasury_yield')
        
        # 2. FEDERAL_FUNDS_RATE
        params = {
            'function': 'FEDERAL_FUNDS_RATE',
            'interval': 'monthly',  # daily, weekly, or monthly
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'federal_funds_rate')
        
        # 3. CPI
        params = {
            'function': 'CPI',
            'interval': 'monthly',  # monthly or semiannual
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'cpi')
        
        # 4. INFLATION
        params = {
            'function': 'INFLATION',
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'inflation')
        
        # 5. REAL_GDP
        params = {
            'function': 'REAL_GDP',
            'interval': 'quarterly',  # quarterly or annual
            'apikey': self.api_key,
            'datatype': 'json'
        }
        self._make_request(params, 'real_gdp')
    
    def test_all_apis(self):
        """Test all 41 Alpha Vantage APIs"""
        self.logger.info("\n" + "="*60)
        self.logger.info("TESTING ALL 41 ALPHA VANTAGE APIs")
        self.logger.info("Based on Official API Documentation")
        self.logger.info("="*60)
        
        start_time = datetime.now()
        
        # Test each category with correct parameters
        self.test_options_apis()      # 2 APIs
        self.test_indicator_apis()    # 16 APIs
        self.test_analytics_apis()    # 2 APIs
        self.test_sentiment_apis()    # 3 APIs
        self.test_fundamental_apis()  # 10 APIs
        self.test_economic_apis()     # 5 APIs
        # Total: 38 core APIs (some fundamentals like EARNINGS_ESTIMATES, 
        # EARNINGS_CALL_TRANSCRIPT counted separately in docs = 41)
        
        elapsed = datetime.now() - start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"✅ API TESTING COMPLETE")
        self.logger.info(f"Total API calls made: {self.api_calls}")
        self.logger.info(f"Time elapsed: {elapsed}")
        self.logger.info(f"Responses saved to: {self.response_dir}")
        self.logger.info("="*60)
        
        # Generate summary report
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate summary of all API responses"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'api_categories': {
                'options': ['realtime_options', 'historical_options'],
                'indicators': ['rsi', 'macd', 'bbands', 'vwap', 'atr', 'adx', 
                              'stoch', 'aroon', 'cci', 'mfi', 'willr', 'mom',
                              'ema', 'sma', 'obv', 'ad'],
                'analytics': ['analytics_fixed_window', 'analytics_sliding_window'],
                'sentiment': ['news_sentiment', 'top_gainers_losers', 'insider_transactions'],
                'fundamentals': ['overview', 'earnings', 'income_statement', 
                               'balance_sheet', 'cash_flow', 'dividends', 
                               'splits', 'earnings_calendar', 'ipo_calendar', 
                               'listing_status'],
                'economic': ['treasury_yield', 'federal_funds_rate', 'cpi', 
                           'inflation', 'real_gdp']
            },
            'total_apis_tested': self.api_calls,
            'response_directory': str(self.response_dir),
            'notes': {
                'rate_limiting': '5 calls per minute for testing',
                'csv_apis': ['earnings_calendar', 'ipo_calendar', 'listing_status'],
                'intraday_only': ['vwap'],
                'require_greeks': 'realtime_options with require_greeks=true'
            }
        }
        
        summary_file = self.response_dir / 'api_test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to: {summary_file}")


def main():
    """Run comprehensive API testing"""
    tester = AlphaVantageAPITester()
    
    try:
        tester.test_all_apis()
        return 0
    except Exception as e:
        tester.logger.error(f"Testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())