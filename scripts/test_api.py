#!/usr/bin/env python3
"""
COMPREHENSIVE ALPHA VANTAGE API TESTER - CORRECT VERSION
Based on SSOT-Ops.md specifications
NO TIME SERIES - that comes from IBKR!
Premium account: 600 calls/minute
"""

import sys
import json
import time
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ALPHA VANTAGE APIs FROM PROJECT SPECIFICATION
# Based on SSOT-Ops.md - Alpha Vantage provides 43 APIs
# ============================================================================

ALPHA_VANTAGE_APIS = [
    # ========== OPTIONS WITH GREEKS (PRIMARY SOURCE) ==========
    {
        'name': 'realtime_options',
        'params': {
            'function': 'REALTIME_OPTIONS',
            'symbol': 'AAPL',
            'require_greeks': 'true' # Include greeks in response
        }
    },
    {
        'name': 'historical_options',
        'params': {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': 'AAPL',
        }
    },
    
    # ========== TECHNICAL INDICATORS (16 total) ==========
    {
        'name': 'rsi',
        'params': {
            'function': 'RSI',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '14',
            'series_type': 'close'
        }
    },
    {
        'name': 'macd',
        'params': {
            'function': 'MACD',
            'symbol': 'AAPL',
            'interval': 'daily',
            'series_type': 'close',
            'fastperiod': '12',
            'slowperiod': '26',
            'signalperiod': '9'
        }
    },
    {
        'name': 'stoch',
        'params': {
            'function': 'STOCH',
            'symbol': 'AAPL',
            'interval': 'daily',
            'fastkperiod': '5',
            'slowkperiod': '3',
            'slowdperiod': '3',
            'slowkmatype': '0',
            'slowdmatype': '0'
        }
    },
    {
        'name': 'bbands',
        'params': {
            'function': 'BBANDS',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '20',
            'series_type': 'close',
            'nbdevup': '2',
            'nbdevdn': '2',
            'matype': '0'  # SMA
        }
    },
    {
        'name': 'atr',
        'params': {
            'function': 'ATR',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '14'
        }
    },
    {
        'name': 'adx',
        'params': {
            'function': 'ADX',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '14'
        }
    },
    {
        'name': 'vwap',
        'params': {
            'function': 'VWAP',
            'symbol': 'AAPL',
            'interval': '15min'  # VWAP is intraday only
        }
    },
    {
        'name': 'ema',
        'params': {
            'function': 'EMA',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '20',
            'series_type': 'close'
        }
    },
    {
        'name': 'sma',
        'params': {
            'function': 'SMA',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '20',
            'series_type': 'close'
        }
    },
    {
        'name': 'aroon',
        'params': {
            'function': 'AROON',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '14'
        }
    },
    {
        'name': 'cci',
        'params': {
            'function': 'CCI',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '20'
        }
    },
    {
        'name': 'mfi',
        'params': {
            'function': 'MFI',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '14'
        }
    },
    {
        'name': 'willr',
        'params': {
            'function': 'WILLR',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '14'
        }
    },
    {
        'name': 'mom',
        'params': {
            'function': 'MOM',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': '10',
            'series_type': 'close'
        }
    },
    {
        'name': 'ad',
        'params': {
            'function': 'AD',
            'symbol': 'AAPL',
            'interval': 'daily',
        }
    },
    {
        'name': 'obv',
        'params': {
            'function': 'OBV',
            'symbol': 'AAPL',
            'interval': 'daily'
        }
    },
    
    # ========== ANALYTICS ==========
    {
        'name': 'analytics_fixed_window',
        'params': {
            'function': 'ANALYTICS_FIXED_WINDOW',
            'SYMBOLS': 'AAPL,MSFT,GOOGL',  # Multiple symbols
            'RANGE': '2024-01-01',          # Start date YYYY-MM-DD
            'RANGE_END': '2024-12-31',      # End date YYYY-MM-DD  
            'INTERVAL': 'DAILY',            # or WEEKLY, MONTHLY
            'OHLC': 'close',
            'CALCULATIONS': 'MIN,MAX,MEDIAN,CUMULATIVE_RETURN,VARIANCE,MAX_DRAWDOWN,HISTOGRAM,AUTOCORRELATION,COVARIANCE,CORRELATION,MEAN,STDDEV',

        }
    },
    {
        'name': 'analytics_sliding_window',
        'params': {
            'function': 'ANALYTICS_SLIDING_WINDOW',
            'SYMBOLS': 'AAPL,MSFT,GOOGL',
            'RANGE': '6month',              # Relative period!
            'INTERVAL': 'DAILY',
            'OHLC': 'close',
            'WINDOW_SIZE': '50',
            'CALCULATIONS': 'MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,COVARIANCE,CORRELATION',
        }
    },
    
    # ========== FUNDAMENTALS (11 APIs) ==========
    {
        'name': 'overview',
        'params': {
            'function': 'OVERVIEW',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'earnings',
        'params': {
            'function': 'EARNINGS',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'income_statement',
        'params': {
            'function': 'INCOME_STATEMENT',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'balance_sheet',
        'params': {
            'function': 'BALANCE_SHEET',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'cash_flow',
        'params': {
            'function': 'CASH_FLOW',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'dividends',
        'params': {
            'function': 'DIVIDENDS',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'splits',
        'params': {
            'function': 'SPLITS',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'earnings_estimates',
        'params': {
            'function': 'EARNINGS_ESTIMATES',
            'symbol': 'AAPL'
        }
    },
    {
        'name': 'earnings_calendar',
        'params': {
            'function': 'EARNINGS_CALENDAR',
        }
    },
    {
        'name': 'earnings_call_transcript',
        'params': {
            'function': 'EARNINGS_CALL_TRANSCRIPT',
            'symbol': 'AAPL',
            'quarter':'2025Q1',
        }
    },
    
    # ========== ECONOMIC INDICATORS (5 APIs) ==========
    {
        'name': 'treasury_yield',
        'params': {
            'function': 'TREASURY_YIELD',
            'interval': 'monthly',
            'maturity': '10year'
        }
    },
    {
        'name': 'federal_funds_rate',
        'params': {
            'function': 'FEDERAL_FUNDS_RATE',
            'interval': 'monthly'
        }
    },
    {
        'name': 'cpi',
        'params': {
            'function': 'CPI',
            'interval': 'monthly'
        }
    },
    {
        'name': 'inflation',
        'params': {
            'function': 'INFLATION'
        }
    },
    {
        'name': 'real_gdp',
        'params': {
            'function': 'REAL_GDP',
            'interval': 'annual'
        }
    },
    
    # ========== SENTIMENT & NEWS (3 APIs) ==========
    {
        'name': 'news_sentiment',
        'params': {
            'function': 'NEWS_SENTIMENT',
            'tickers': 'AAPL',
            'sort':'LATEST',
            'limit': '100'
        }
    },
    {
        'name': 'top_gainers_losers',
        'params': {
            'function': 'TOP_GAINERS_LOSERS'
        }
    },
    {
        'name': 'insider_transactions',
        'params': {
            'function': 'INSIDER_TRANSACTIONS',
            'symbol': 'AAPL'
        }
    }
]

class ComprehensiveAPITester:
    """Test ALL Alpha Vantage APIs with your existing analyzer"""
    
    def __init__(self):
        self.api_key = os.getenv('AV_API_KEY')
        if not self.api_key:
            raise ValueError("No API key! Set AV_API_KEY in .env")
        self.base_url = "https://www.alphavantage.co/query"
        
        # Create directories
        Path("data/api_responses").mkdir(parents=True, exist_ok=True)
        Path("data/schemas").mkdir(parents=True, exist_ok=True)
        
        # Track results
        self.results = {}
        
    def test_all_apis(self, delay_between_calls: float = 0.1):
        """
        Test ALL APIs in sequence
        With 600 calls/min limit, we can do 10 calls/second
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE API TESTING")
        logger.info(f"Testing {len(ALPHA_VANTAGE_APIS)} APIs")
        logger.info(f"Rate limit: 600 calls/min (using {delay_between_calls}s delay)")
        logger.info("=" * 80)
        
        for i, api_config in enumerate(ALPHA_VANTAGE_APIS, 1):
            api_name = api_config['name']
            params = api_config['params'].copy()
            
            logger.info(f"\n[{i}/{len(ALPHA_VANTAGE_APIS)}] Testing: {api_name}")
            logger.info(f"Function: {params.get('function', 'N/A')}")
            
            # Add API key
            params['apikey'] = self.api_key
            
            try:
                # Make API call
                response = requests.get(self.base_url, params=params, timeout=30)
                if api_name == 'earnings_calendar':
                    # Just save the raw text as-is
                    data = {'csv_response': response.text}
                else:
                    # Normal JSON parsing
                     data = response.json()
                
                # Check for errors
                if "Error Message" in data:
                    logger.error(f"API Error: {data['Error Message']}")
                    self.results[api_name] = 'ERROR'
                    continue
                    
                if "Note" in data:
                    logger.warning(f"Rate limit hit: {data['Note']}")
                    self.results[api_name] = 'RATE_LIMITED'
                    time.sleep(60)  # Wait a minute
                    continue
                
                # Save response
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{api_name}_AAPL_{timestamp}.json"
                filepath = Path(f"data/api_responses/{filename}")
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                logger.info(f"✓ Response saved: {filepath}")
                self.results[api_name] = 'SUCCESS'
                
                # Analyze and generate schema (using your existing methods)
                # You would call your analyze_response and generate_schema here
                
            except Exception as e:
                logger.error(f"Failed: {e}")
                self.results[api_name] = f'EXCEPTION: {str(e)}'
            
            # Rate limit delay
            if i < len(ALPHA_VANTAGE_APIS):
                time.sleep(delay_between_calls)
        
        # Print summary
        self.print_summary()
        
    def print_summary(self):
        """Print test results summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        success = [k for k, v in self.results.items() if v == 'SUCCESS']
        failed = [k for k, v in self.results.items() if v != 'SUCCESS']
        
        logger.info(f"✓ Successful: {len(success)}/{len(self.results)}")
        logger.info(f"✗ Failed: {len(failed)}/{len(self.results)}")
        
        if failed:
            logger.info("\nFailed APIs:")
            for api in failed:
                logger.info(f"  - {api}: {self.results[api]}")
        
        # Save results
        results_file = Path(f"data/schemas/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")


def main():
    """Run comprehensive API testing"""
    tester = ComprehensiveAPITester()
    
    # Test all APIs
    tester.test_all_apis(delay_between_calls=0.1)  # 10 calls/second = 600/min
    
    logger.info("\n✅ Comprehensive testing complete!")
    logger.info("Next steps:")
    logger.info("1. Review all responses in data/api_responses/")
    logger.info("2. Run analyze_response on each to generate schemas")
    logger.info("3. Create tables from generated schemas")
    logger.info("4. Implement ingestion for each API")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())