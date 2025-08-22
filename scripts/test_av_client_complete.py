#!/usr/bin/env python3
"""
Test complete Alpha Vantage client implementation
Phase 1: Verify all 35 API methods work correctly
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import get_av_client
from src.foundation.logger import get_logger


class AlphaVantageClientTester:
    """Test all 35 Alpha Vantage client methods"""
    
    def __init__(self):
        self.client = get_av_client()
        self.logger = get_logger('AVClientTester')
        self.test_results = {}
        self.start_time = time.time()
        
        self.logger.info("Alpha Vantage Client Tester initialized")
        self.logger.info("Testing all 35 API methods...")
    
    def _test_method(self, method_name: str, method_func, *args, **kwargs):
        """Test a single API method"""
        try:
            self.logger.info(f"Testing {method_name}...")
            start_time = time.time()
            
            result = method_func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            
            if result is not None:
                self.test_results[method_name] = {
                    'status': 'SUCCESS',
                    'elapsed_time': elapsed,
                    'has_data': bool(result),
                    'data_keys': list(result.keys()) if isinstance(result, dict) else 'non-dict',
                    'data_size': len(str(result)) if result else 0
                }
                self.logger.info(f"✓ {method_name} succeeded in {elapsed:.2f}s")
            else:
                self.test_results[method_name] = {
                    'status': 'FAILED',
                    'elapsed_time': elapsed,
                    'error': 'No data returned'
                }
                self.logger.error(f"✗ {method_name} failed - no data returned")
                
        except Exception as e:
            self.test_results[method_name] = {
                'status': 'ERROR',
                'elapsed_time': time.time() - start_time,
                'error': str(e)
            }
            self.logger.error(f"✗ {method_name} error: {e}")
    
    def test_options_apis(self):
        """Test Options & Greeks APIs (2)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING OPTIONS & GREEKS APIS (2)")
        self.logger.info("="*50)
        
        # Test with default parameters (all from config)
        self._test_method('get_realtime_options', 
                         self.client.get_realtime_options, 'SPY')
        
        self._test_method('get_historical_options',
                         self.client.get_historical_options, 'SPY')
    
    def test_indicator_apis(self):
        """Test Technical Indicator APIs (16)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING TECHNICAL INDICATOR APIS (16)")
        self.logger.info("="*50)
        
        # Test all indicators with default parameters
        self._test_method('get_rsi', self.client.get_rsi, 'SPY')
        self._test_method('get_macd', self.client.get_macd, 'SPY')
        self._test_method('get_stoch', self.client.get_stoch, 'SPY')
        self._test_method('get_willr', self.client.get_willr, 'SPY')
        self._test_method('get_mom', self.client.get_mom, 'SPY')
        self._test_method('get_bbands', self.client.get_bbands, 'SPY')
        self._test_method('get_atr', self.client.get_atr, 'SPY')
        self._test_method('get_adx', self.client.get_adx, 'SPY')
        self._test_method('get_aroon', self.client.get_aroon, 'SPY')
        self._test_method('get_cci', self.client.get_cci, 'SPY')
        self._test_method('get_ema', self.client.get_ema, 'SPY')
        self._test_method('get_sma', self.client.get_sma, 'SPY')
        self._test_method('get_mfi', self.client.get_mfi, 'SPY')
        self._test_method('get_obv', self.client.get_obv, 'SPY')
        self._test_method('get_ad', self.client.get_ad, 'SPY')
        self._test_method('get_vwap', self.client.get_vwap, 'SPY')
    
    def test_analytics_apis(self):
        """Test Analytics APIs (2)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING ANALYTICS APIS (2)")
        self.logger.info("="*50)
        
        # Test with default parameters
        self._test_method('get_analytics_fixed_window',
                         self.client.get_analytics_fixed_window)
        
        self._test_method('get_analytics_sliding_window',
                         self.client.get_analytics_sliding_window)
    
    def test_sentiment_apis(self):
        """Test Sentiment & News APIs (3)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING SENTIMENT & NEWS APIS (3)")
        self.logger.info("="*50)
        
        # Test with default parameters
        self._test_method('get_news_sentiment',
                         self.client.get_news_sentiment)
        
        self._test_method('get_top_gainers_losers',
                         self.client.get_top_gainers_losers)
        
        self._test_method('get_insider_transactions',
                         self.client.get_insider_transactions, 'AAPL')
    
    def test_fundamental_apis(self):
        """Test Fundamental APIs (7)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING FUNDAMENTAL APIS (7)")
        self.logger.info("="*50)
        
        # Test with AAPL as in original test
        self._test_method('get_overview', self.client.get_overview, 'AAPL')
        self._test_method('get_earnings_calendar', self.client.get_earnings_calendar)
        self._test_method('get_income_statement', self.client.get_income_statement, 'AAPL')
        self._test_method('get_balance_sheet', self.client.get_balance_sheet, 'AAPL')
        self._test_method('get_cash_flow', self.client.get_cash_flow, 'AAPL')
        self._test_method('get_dividends', self.client.get_dividends, 'AAPL')
        self._test_method('get_splits', self.client.get_splits, 'AAPL')
    
    def test_economic_apis(self):
        """Test Economic Indicator APIs (5)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING ECONOMIC INDICATOR APIS (5)")
        self.logger.info("="*50)
        
        # Test with default parameters
        self._test_method('get_treasury_yield', self.client.get_treasury_yield)
        self._test_method('get_federal_funds_rate', self.client.get_federal_funds_rate)
        self._test_method('get_cpi', self.client.get_cpi)
        self._test_method('get_inflation', self.client.get_inflation)
        self._test_method('get_real_gdp', self.client.get_real_gdp)
    
    def test_caching(self):
        """Test caching functionality"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING CACHE FUNCTIONALITY")
        self.logger.info("="*50)
        
        # Test cache hit on second call
        self.logger.info("Testing cache hit...")
        
        # First call (should be cached)
        start_time = time.time()
        result1 = self.client.get_rsi('QQQ')
        first_call_time = time.time() - start_time
        
        # Second call (should hit cache)
        start_time = time.time()
        result2 = self.client.get_rsi('QQQ')
        second_call_time = time.time() - start_time
        
        if result1 and result2:
            cache_speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
            self.logger.info(f"First call: {first_call_time:.3f}s")
            self.logger.info(f"Second call: {second_call_time:.3f}s")
            self.logger.info(f"Cache speedup: {cache_speedup:.1f}x")
            
            if cache_speedup > 5:  # Should be much faster
                self.logger.info("✓ Cache working correctly")
            else:
                self.logger.warning("⚠ Cache may not be working optimally")
        else:
            self.logger.error("✗ Cache test failed - no data returned")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING RATE LIMITING")
        self.logger.info("="*50)
        
        # Get initial rate limiter stats
        stats_before = self.client.rate_limiter.get_stats()
        self.logger.info(f"Initial stats: {stats_before}")
        
        # Make several rapid calls
        rapid_calls = 5
        start_time = time.time()
        
        for i in range(rapid_calls):
            result = self.client.get_rsi('SPY', use_cache=False)  # Disable cache to force API calls
            if result:
                self.logger.info(f"Rapid call {i+1} succeeded")
        
        elapsed = time.time() - start_time
        stats_after = self.client.rate_limiter.get_stats()
        
        self.logger.info(f"Final stats: {stats_after}")
        self.logger.info(f"Time for {rapid_calls} calls: {elapsed:.2f}s")
        
        # Check if rate limiting is working
        if stats_after['total_calls'] > stats_before['total_calls']:
            self.logger.info("✓ Rate limiter tracking calls")
        else:
            self.logger.warning("⚠ Rate limiter may not be tracking correctly")
    
    def test_parameter_override(self):
        """Test parameter override functionality"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING PARAMETER OVERRIDE")
        self.logger.info("="*50)
        
        # Test with custom parameters vs defaults
        self.logger.info("Testing RSI with custom time_period...")
        result_custom = self.client.get_rsi('SPY', time_period=21)  # Override default 14
        
        if result_custom:
            self.logger.info("✓ Parameter override working")
        else:
            self.logger.error("✗ Parameter override failed")
        
        # Test MACD with custom periods
        self.logger.info("Testing MACD with custom periods...")
        result_macd = self.client.get_macd('SPY', fastperiod=8, slowperiod=21, signalperiod=5)
        
        if result_macd:
            self.logger.info("✓ Multiple parameter override working")
        else:
            self.logger.error("✗ Multiple parameter override failed")
    
    def run_all_tests(self):
        """Run all test suites"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING COMPREHENSIVE CLIENT TESTING")
        self.logger.info("Testing all 35 Alpha Vantage API methods")
        self.logger.info("="*60)
        
        # Run all test suites
        self.test_options_apis()      # 2 APIs
        self.test_indicator_apis()    # 16 APIs
        self.test_analytics_apis()    # 2 APIs
        self.test_sentiment_apis()    # 3 APIs
        self.test_fundamental_apis()  # 7 APIs
        self.test_economic_apis()     # 5 APIs
        
        # Test functionality
        self.test_caching()
        self.test_rate_limiting()
        self.test_parameter_override()
        
        # Generate summary report
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate test summary report"""
        total_time = time.time() - self.start_time
        
        successful = sum(1 for result in self.test_results.values() 
                        if result['status'] == 'SUCCESS')
        failed = sum(1 for result in self.test_results.values() 
                    if result['status'] in ['FAILED', 'ERROR'])
        
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST SUMMARY REPORT")
        self.logger.info("="*60)
        self.logger.info(f"Total APIs tested: {len(self.test_results)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Success rate: {(successful/len(self.test_results)*100):.1f}%")
        self.logger.info(f"Total test time: {total_time:.2f}s")
        
        # Rate limiter final stats
        final_stats = self.client.rate_limiter.get_stats()
        self.logger.info(f"\nFinal rate limiter stats:")
        self.logger.info(f"  Total calls: {final_stats['total_calls']}")
        self.logger.info(f"  Rejected calls: {final_stats['rejected_calls']}")
        self.logger.info(f"  Current tokens: {final_stats['current_tokens']}")
        self.logger.info(f"  Window calls: {final_stats['window_calls']}")
        
        # Failed tests details
        if failed > 0:
            self.logger.info(f"\nFailed tests:")
            for method, result in self.test_results.items():
                if result['status'] != 'SUCCESS':
                    self.logger.error(f"  {method}: {result.get('error', 'Unknown error')}")
        
        if successful == len(self.test_results):
            self.logger.info("\n✅ ALL TESTS PASSED!")
            self.logger.info("Alpha Vantage client implementation complete and working")
        else:
            self.logger.warning(f"\n⚠ {failed} tests failed")
            self.logger.info("Review failed tests and fix implementation")
        
        self.logger.info("="*60)


def main():
    """Run comprehensive Alpha Vantage client testing"""
    tester = AlphaVantageClientTester()
    
    try:
        tester.run_all_tests()
        return 0
    except Exception as e:
        tester.logger.error(f"Testing failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit(main())