#!/usr/bin/env python3
"""
Alpha Vantage API Health Check Script
Tests all 38 Alpha Vantage API endpoints to ensure they're working.
Run this as part of pre-market checks to verify API availability.
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.av_client import AlphaVantageClient, AVFunction
from src.core.config import initialize_config, get_config


class AlphaVantageHealthChecker:
    """
    Tests all Alpha Vantage API endpoints for health and performance
    """
    
    def __init__(self):
        """Initialize health checker"""
        # Initialize configuration
        self.config = initialize_config()
        self.av_client = AlphaVantageClient(self.config.alpha_vantage)
        
        # Test results storage
        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time = None
        self.end_time = None
        
        # API test parameters
        self.test_params = {
            # OPTIONS
            'REALTIME_OPTIONS': {'symbol': 'SPY'},
            'HISTORICAL_OPTIONS': {'symbol': 'SPY', 'date': '2025-01-01'},
            
            # TECHNICAL INDICATORS
            'RSI': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 14, 'series_type': 'close'},
            'MACD': {'symbol': 'SPY', 'interval': 'daily', 'series_type': 'close'},
            'STOCH': {'symbol': 'SPY', 'interval': 'daily'},
            'WILLR': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 14},
            'MOM': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 10, 'series_type': 'close'},
            'BBANDS': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 20, 'series_type': 'close'},
            'ATR': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 14},
            'ADX': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 14},
            'AROON': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 14},
            'CCI': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 20},
            'EMA': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 20, 'series_type': 'close'},
            'SMA': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 20, 'series_type': 'close'},
            'MFI': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 14},
            'OBV': {'symbol': 'SPY', 'interval': 'daily'},
            'AD': {'symbol': 'SPY', 'interval': 'daily'},
            'VWAP': {'symbol': 'SPY', 'interval': '15min'},  # Intraday only
            
            # ANALYTICS
            'ANALYTICS_FIXED_WINDOW': {
                'SYMBOLS': 'SPY,QQQ',
                'INTERVAL': 'DAILY',
                'RANGE': '1month',
                'OHLC': 'close',
                'CALCULATIONS': 'MEAN,STDDEV'
            },
            'ANALYTICS_SLIDING_WINDOW': {
                'SYMBOLS': 'SPY,QQQ',
                'INTERVAL': 'DAILY',
                'RANGE': '3month',
                'WINDOW_SIZE': 30,
                'OHLC': 'close',
                'CALCULATIONS': 'MEAN,STDDEV'
            },
            
            # SENTIMENT
            'NEWS_SENTIMENT': {'tickers': 'SPY', 'limit': 10},
            'TOP_GAINERS_LOSERS': {},
            'INSIDER_TRANSACTIONS': {'symbol': 'AAPL'},
            
            # FUNDAMENTALS
            'OVERVIEW': {'symbol': 'AAPL'},
            'EARNINGS': {'symbol': 'AAPL'},
            'INCOME_STATEMENT': {'symbol': 'AAPL'},
            'BALANCE_SHEET': {'symbol': 'AAPL'},
            'CASH_FLOW': {'symbol': 'AAPL'},
            'DIVIDENDS': {'symbol': 'AAPL'},
            'SPLITS': {'symbol': 'AAPL'},
            'EARNINGS_CALENDAR': {},
            
            # ECONOMIC
            'TREASURY_YIELD': {'interval': 'monthly', 'maturity': '10year'},
            'FEDERAL_FUNDS_RATE': {'interval': 'monthly'},
            'CPI': {'interval': 'monthly'},
            'INFLATION': {},
            'REAL_GDP': {'interval': 'quarterly'}
        }
    
    async def test_endpoint(self, function: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a single API endpoint
        
        Args:
            function: API function name
            params: Parameters for the API call
            
        Returns:
            Test result dictionary
        """
        result = {
            'function': function,
            'status': 'UNKNOWN',
            'latency_ms': None,
            'error': None,
            'cached': False,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Time the request
            start = time.perf_counter()
            
            # Make the API call
            response = await self.av_client._make_request(function, **params)
            
            # Calculate latency
            latency = (time.perf_counter() - start) * 1000
            result['latency_ms'] = round(latency, 2)
            
            # Check if response is valid
            if response and not 'Error Message' in response:
                result['status'] = 'SUCCESS'
                
                # Check if it was cached
                cache_key = self.av_client._get_cache_key(function, **params)
                if cache_key in self.av_client.cache:
                    result['cached'] = True
            else:
                result['status'] = 'FAILED'
                result['error'] = response.get('Error Message', 'Unknown error')
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    async def test_all_endpoints(self) -> None:
        """
        Test all 38 Alpha Vantage API endpoints
        """
        print("\n" + "="*70)
        print("ALPHA VANTAGE API HEALTH CHECK")
        print(f"Testing all 38 endpoints with {self.config.alpha_vantage.rate_limit}/min rate limit")
        print("="*70 + "\n")
        
        self.start_time = datetime.now()
        
        # Group tests by category
        categories = {
            'OPTIONS': ['REALTIME_OPTIONS', 'HISTORICAL_OPTIONS'],
            'TECHNICAL': ['RSI', 'MACD', 'STOCH', 'WILLR', 'MOM', 'BBANDS', 
                         'ATR', 'ADX', 'AROON', 'CCI', 'EMA', 'SMA', 
                         'MFI', 'OBV', 'AD', 'VWAP'],
            'ANALYTICS': ['ANALYTICS_FIXED_WINDOW', 'ANALYTICS_SLIDING_WINDOW'],
            'SENTIMENT': ['NEWS_SENTIMENT', 'TOP_GAINERS_LOSERS', 'INSIDER_TRANSACTIONS'],
            'FUNDAMENTALS': ['OVERVIEW', 'EARNINGS', 'INCOME_STATEMENT', 
                           'BALANCE_SHEET', 'CASH_FLOW', 'DIVIDENDS', 
                           'SPLITS', 'EARNINGS_CALENDAR'],
            'ECONOMIC': ['TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 'CPI', 
                        'INFLATION', 'REAL_GDP']
        }
        
        # Test each category
        for category, functions in categories.items():
            print(f"\n📁 Testing {category} APIs...")
            print("-" * 40)
            
            for function in functions:
                params = self.test_params.get(function, {})
                result = await self.test_endpoint(function, params)
                self.results[function] = result
                
                # Print result
                status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
                cache_icon = "💾" if result['cached'] else "🌐"
                
                print(f"{status_icon} {function:30} {cache_icon} {result['latency_ms']:7.1f}ms")
                
                if result['error']:
                    print(f"   └─ Error: {result['error']}")
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
        
        self.end_time = datetime.now()
    
    def generate_report(self) -> None:
        """
        Generate comprehensive health check report
        """
        print("\n" + "="*70)
        print("HEALTH CHECK SUMMARY")
        print("="*70)
        
        # Calculate statistics
        total_tests = len(self.results)
        successful = sum(1 for r in self.results.values() if r['status'] == 'SUCCESS')
        failed = sum(1 for r in self.results.values() if r['status'] in ['FAILED', 'ERROR'])
        cached = sum(1 for r in self.results.values() if r.get('cached', False))
        
        # Calculate latency statistics
        latencies = [r['latency_ms'] for r in self.results.values() if r['latency_ms']]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
        else:
            avg_latency = max_latency = min_latency = 0
        
        # Print summary
        print(f"\n📊 Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful} ({successful/total_tests*100:.1f}%)")
        print(f"   Failed: {failed} ({failed/total_tests*100:.1f}%)")
        print(f"   Cached: {cached} ({cached/total_tests*100:.1f}%)")
        
        print(f"\n⏱️  Performance:")
        print(f"   Average Latency: {avg_latency:.1f}ms")
        print(f"   Min Latency: {min_latency:.1f}ms")
        print(f"   Max Latency: {max_latency:.1f}ms")
        
        print(f"\n🔧 Configuration:")
        print(f"   Rate Limit: {self.config.alpha_vantage.rate_limit} calls/minute")
        print(f"   Daily Limit: {self.config.alpha_vantage.daily_limit} calls/day")
        print(f"   Cache Enabled: {self.config.alpha_vantage.use_cache}")
        
        # List failed endpoints
        if failed > 0:
            print(f"\n❌ Failed Endpoints:")
            for func, result in self.results.items():
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"   - {func}: {result['error']}")
        
        # Test duration
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"\n⏰ Test Duration: {duration:.1f} seconds")
        
        # API statistics
        print(f"\n📈 API Statistics:")
        print(f"   Total API Calls Made: {self.av_client.api_calls_made}")
        print(f"   Cache Hits: {self.av_client.cache_hits}")
        print(f"   Cache Misses: {self.av_client.cache_misses}")
        print(f"   Cache Hit Rate: {self.av_client.get_stats()['cache_hit_rate']*100:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if failed > 0:
            print("   ⚠️  Some endpoints failed - check API key and parameters")
        if avg_latency > 500:
            print("   ⚠️  High average latency - consider using cache more effectively")
        if cached < total_tests * 0.3:
            print("   ⚠️  Low cache hit rate - consider pre-warming cache")
        if successful == total_tests:
            print("   ✅ All endpoints operational - system ready for trading!")
    
    def save_results(self, filepath: str = "reports/av_health_check.json") -> None:
        """
        Save test results to file
        
        Args:
            filepath: Path to save results
        """
        # Create reports directory if needed
        Path(filepath).parent.mkdir(exist_ok=True)
        
        # Prepare data for saving
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'rate_limit': self.config.alpha_vantage.rate_limit,
                'daily_limit': self.config.alpha_vantage.daily_limit,
                'cache_enabled': self.config.alpha_vantage.use_cache
            },
            'results': self.results,
            'statistics': {
                'total_tests': len(self.results),
                'successful': sum(1 for r in self.results.values() if r['status'] == 'SUCCESS'),
                'failed': sum(1 for r in self.results.values() if r['status'] in ['FAILED', 'ERROR']),
                'api_calls': self.av_client.api_calls_made,
                'cache_hits': self.av_client.cache_hits,
                'cache_misses': self.av_client.cache_misses
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
    
    async def test_critical_endpoints(self) -> bool:
        """
        Test only critical endpoints for quick health check
        
        Returns:
            True if all critical endpoints are working
        """
        print("\n🚀 Quick Health Check - Testing Critical Endpoints...")
        
        critical = {
            'REALTIME_OPTIONS': {'symbol': 'SPY'},
            'RSI': {'symbol': 'SPY', 'interval': 'daily', 'time_period': 14, 'series_type': 'close'},
            'NEWS_SENTIMENT': {'tickers': 'SPY', 'limit': 5},
            'OVERVIEW': {'symbol': 'SPY'}
        }
        
        all_working = True
        
        for function, params in critical.items():
            result = await self.test_endpoint(function, params)
            
            if result['status'] == 'SUCCESS':
                print(f"✅ {function}: OK ({result['latency_ms']:.1f}ms)")
            else:
                print(f"❌ {function}: FAILED - {result['error']}")
                all_working = False
        
        return all_working
    
    async def warmup_cache(self, symbols: List[str]) -> None:
        """
        Warmup cache with common requests
        
        Args:
            symbols: List of symbols to warmup
        """
        print(f"\n🔥 Warming up cache for {symbols}...")
        
        for symbol in symbols:
            # Options
            await self.av_client.get_realtime_options(symbol)
            
            # Key technical indicators
            await self.av_client.get_rsi(symbol)
            await self.av_client.get_macd(symbol)
            await self.av_client.get_bollinger_bands(symbol)
            
            # Sentiment
            await self.av_client.get_news_sentiment(tickers=symbol)
            
            # Fundamentals
            await self.av_client.get_company_overview(symbol)
            
            print(f"✅ Cache warmed for {symbol}")
        
        print(f"\n📊 Cache Statistics:")
        stats = self.av_client.get_stats()
        print(f"   Cache Size: {stats['cache_size']} entries")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']*100:.1f}%")


async def main():
    """
    Main function to run health checks
    """
    checker = AlphaVantageHealthChecker()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            # Quick health check
            success = await checker.test_critical_endpoints()
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == '--warmup':
            # Warmup cache
            symbols = sys.argv[2:] if len(sys.argv) > 2 else ['SPY', 'QQQ', 'IWM']
            await checker.warmup_cache(symbols)
            sys.exit(0)
    
    # Full health check
    await checker.test_all_endpoints()
    checker.generate_report()
    checker.save_results()
    
    # Exit with appropriate code
    failed = sum(1 for r in checker.results.values() if r['status'] in ['FAILED', 'ERROR'])
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Alpha Vantage API Health Check Tool              ║
    ║                                                          ║
    ║  Usage:                                                  ║
    ║    python av_api_health.py          # Full test         ║
    ║    python av_api_health.py --quick  # Critical only     ║
    ║    python av_api_health.py --warmup SPY QQQ  # Warmup   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())