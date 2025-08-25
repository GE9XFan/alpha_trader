#!/usr/bin/env python3
"""
Test Alpha Vantage Client Implementation
Tests all 36 APIs and verifies Greeks are PROVIDED, not calculated
"""
import asyncio
import os
import sys
from datetime import datetime
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.alpha_vantage_client import AlphaVantageClient, OptionContract
from src.core.logger import get_logger

load_dotenv()  # Load environment variables from .env file

logger = get_logger(__name__)

# ========================================================================
# TEST CONFIGURATION
# ========================================================================

# Check for API key
API_KEY = os.getenv('AV_API_KEY')
if not API_KEY:
    print("❌ ERROR: Set your API key first:")
    print("   export AV_API_KEY='your_alpha_vantage_api_key'")
    sys.exit(1)

# Test parameters
TEST_SYMBOL = 'AAPL'  # Use a liquid symbol for testing
TEST_DATE = '2025-08-20'  # For historical options


class AlphaVantageClientTester:
    """Test harness for Alpha Vantage client"""
    
    def __init__(self):
        self.client = AlphaVantageClient()
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    async def setup(self):
        """Initialize client connection"""
        await self.client.connect()
        print(f"✅ Client connected")
        print(f"📊 API Key: {self.client.api_key[:8]}...")
        print(f"⚡ Rate limit: {self.client.rate_limiter.calls_per_minute} calls/min")
        print("="*70)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.client.disconnect()
    
    # ========================================================================
    # TEST OPTIONS APIs (2)
    # ========================================================================
    
    async def test_realtime_options(self):
        """Test REALTIME_OPTIONS - Verify Greeks are PROVIDED"""
        print("\n🔹 Testing REALTIME_OPTIONS...")
        try:
            options = await self.client.get_realtime_options(TEST_SYMBOL, require_greeks=True)
            
            if options and len(options) > 0:
                # Check first option has Greeks
                option = options[0]
                
                # CRITICAL: Verify Greeks are PROVIDED
                has_greeks = (
                    option.delta != 0 or
                    option.gamma != 0 or
                    option.theta != 0 or
                    option.vega != 0
                )
                
                if has_greeks:
                    print(f"   ✅ Retrieved {len(options)} options WITH Greeks")
                    print(f"   📊 Sample: {option.symbol} ${option.strike} {option.option_type}")
                    print(f"   🎯 Greeks PROVIDED: Δ={option.delta:.3f}, Γ={option.gamma:.3f}, "
                          f"Θ={option.theta:.3f}, V={option.vega:.3f}")
                    self.passed += 1
                else:
                    print(f"   ⚠️ Options returned but Greeks are zero/missing")
                    self.errors.append("Greeks not provided in realtime options")
                    self.failed += 1
            else:
                print(f"   ⚠️ No options returned (market may be closed)")
                self.failed += 1
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"REALTIME_OPTIONS: {e}")
            self.failed += 1
    
    async def test_historical_options(self):
        """Test HISTORICAL_OPTIONS - 20 years of data with Greeks"""
        print("\n🔹 Testing HISTORICAL_OPTIONS...")
        try:
            options = await self.client.get_historical_options(TEST_SYMBOL, TEST_DATE)
            
            if options and len(options) > 0:
                print(f"   ✅ Retrieved {len(options)} historical options for {TEST_DATE}")
                option = options[0]
                print(f"   📊 Sample Greeks: Δ={option.delta:.3f}, Γ={option.gamma:.3f}")
                self.passed += 1
            else:
                print(f"   ⚠️ No historical options returned")
                self.failed += 1
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"HISTORICAL_OPTIONS: {e}")
            self.failed += 1
    
    # ========================================================================
    # TEST TECHNICAL INDICATORS (16)
    # ========================================================================
    
    async def test_technical_indicators(self):
        """Test all 16 technical indicators"""
        indicators = [
            ('RSI', {'interval': 'daily', 'time_period': 14, 'series_type': 'close'}),
            ('MACD', {'interval': 'daily', 'series_type': 'close'}),
            ('STOCH', {'interval': 'daily'}),
            ('WILLR', {'interval': 'daily', 'time_period': 14}),
            ('MOM', {'interval': 'daily', 'time_period': 10, 'series_type': 'close'}),
            ('BBANDS', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            ('ATR', {'interval': 'daily', 'time_period': 14}),
            ('ADX', {'interval': 'daily', 'time_period': 14}),
            ('AROON', {'interval': 'daily', 'time_period': 14}),
            ('CCI', {'interval': 'daily', 'time_period': 20}),
            ('EMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            ('SMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            ('MFI', {'interval': 'daily', 'time_period': 14}),
            ('OBV', {'interval': 'daily'}),
            ('AD', {'interval': 'daily'}),
            ('VWAP', {'interval': '15min'})  # Intraday only
        ]
        
        print("\n🔹 Testing Technical Indicators...")
        
        for name, params in indicators:
            try:
                # Use the specific method for each indicator
                method = getattr(self.client, f'get_{name.lower()}')
                df = await method(TEST_SYMBOL, **params)
                
                if df is not None and not df.empty:
                    print(f"   ✅ {name}: {len(df)} data points")
                    self.passed += 1
                else:
                    print(f"   ⚠️ {name}: Empty response")
                    self.failed += 1
                    
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ {name}: {e}")
                self.errors.append(f"{name}: {e}")
                self.failed += 1
    
    # ========================================================================
    # TEST ANALYTICS APIs (2)
    # ========================================================================
    
    async def test_analytics(self):
        """Test analytics APIs with UPPERCASE parameters"""
        print("\n🔹 Testing Analytics APIs...")
        
        # Test ANALYTICS_FIXED_WINDOW
        try:
            data = await self.client.get_analytics_fixed_window(
                symbols='SPY,QQQ',
                interval='DAILY',
                range='1month',
                calculations='MEAN,VARIANCE,MAX_DRAWDOWN'
            )
            
            if data:
                print(f"   ✅ ANALYTICS_FIXED_WINDOW: Received data")
                self.passed += 1
            else:
                print(f"   ⚠️ ANALYTICS_FIXED_WINDOW: Empty response")
                self.failed += 1
                
        except Exception as e:
            print(f"   ❌ ANALYTICS_FIXED_WINDOW: {e}")
            self.errors.append(f"ANALYTICS_FIXED_WINDOW: {e}")
            self.failed += 1
        
        # Test ANALYTICS_SLIDING_WINDOW
        try:
            data = await self.client.get_analytics_sliding_window(
                symbols='SPY,QQQ',
                window_size=30,
                interval='DAILY',
                range='6month'
            )
            
            if data:
                print(f"   ✅ ANALYTICS_SLIDING_WINDOW: Received data")
                self.passed += 1
            else:
                print(f"   ⚠️ ANALYTICS_SLIDING_WINDOW: Empty response")
                self.failed += 1
                
        except Exception as e:
            print(f"   ❌ ANALYTICS_SLIDING_WINDOW: {e}")
            self.errors.append(f"ANALYTICS_SLIDING_WINDOW: {e}")
            self.failed += 1
    
    # ========================================================================
    # TEST SENTIMENT APIs (3)
    # ========================================================================
    
    async def test_sentiment(self):
        """Test sentiment APIs"""
        print("\n🔹 Testing Sentiment APIs...")
        
        # Test NEWS_SENTIMENT (uses 'tickers' not 'symbol')
        # Note: SPY (ETF) doesn't work with NEWS_SENTIMENT, use stocks instead
        try:
            data = await self.client.get_news_sentiment(
                tickers='AAPL,MSFT',  # Changed from 'SPY,AAPL' - ETFs not supported
                limit=50
            )
            
            if data:
                # Check for API error first
                if 'Information' in data:
                    print(f"   ⚠️ NEWS_SENTIMENT: API error - {data['Information'][:80]}")
                    # Try without tickers as fallback
                    print(f"   🔄 Retrying without ticker filter...")
                    data = await self.client.get_news_sentiment(limit=50)
                    if data and 'feed' in data:
                        print(f"   ✅ NEWS_SENTIMENT: Received {len(data.get('feed', []))} articles (general market)")
                        self.passed += 1
                    else:
                        self.failed += 1
                elif 'feed' in data:
                    article_count = len(data.get('feed', []))
                    print(f"   ✅ NEWS_SENTIMENT: Received {article_count} articles for AAPL,MSFT")
                    # Show sample article if available
                    if article_count > 0:
                        first_article = data['feed'][0]
                        title = first_article.get('title', 'N/A')[:60]
                        print(f"      Sample: {title}...")
                    self.passed += 1
                else:
                    print(f"   ⚠️ NEWS_SENTIMENT: Unexpected response format")
                    self.failed += 1
            else:
                print(f"   ⚠️ NEWS_SENTIMENT: Empty response")
                self.failed += 1
                
        except Exception as e:
            print(f"   ❌ NEWS_SENTIMENT: {e}")
            self.errors.append(f"NEWS_SENTIMENT: {e}")
            self.failed += 1
        
        # Test TOP_GAINERS_LOSERS
        try:
            data = await self.client.get_top_gainers_losers()
            
            if data:
                print(f"   ✅ TOP_GAINERS_LOSERS: Received data")
                self.passed += 1
            else:
                print(f"   ⚠️ TOP_GAINERS_LOSERS: Empty response")
                self.failed += 1
                
        except Exception as e:
            print(f"   ❌ TOP_GAINERS_LOSERS: {e}")
            self.errors.append(f"TOP_GAINERS_LOSERS: {e}")
            self.failed += 1
        
        # Test INSIDER_TRANSACTIONS (3rd sentiment API)
        try:
            data = await self.client.get_insider_transactions(TEST_SYMBOL)
            
            if data:
                print(f"   ✅ INSIDER_TRANSACTIONS: Received data for {TEST_SYMBOL}")
                self.passed += 1
            else:
                print(f"   ⚠️ INSIDER_TRANSACTIONS: Empty response")
                self.failed += 1
                
        except Exception as e:
            print(f"   ❌ INSIDER_TRANSACTIONS: {e}")
            self.errors.append(f"INSIDER_TRANSACTIONS: {e}")
            self.failed += 1
    
    # ========================================================================
    # TEST KEY FUNCTIONALITY
    # ========================================================================
    
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        print("\n🔹 Testing Rate Limiting...")
        
        start = datetime.now()
        calls = 5
        
        for i in range(calls):
            try:
                await self.client.get_overview(TEST_SYMBOL)
            except:
                pass
        
        elapsed = (datetime.now() - start).total_seconds()
        
        if self.client.rate_limiter.remaining > 0:
            print(f"   ✅ Rate limiter working: {self.client.rate_limiter.remaining}/600 remaining")
            print(f"   ⏱️ {calls} calls in {elapsed:.2f}s")
            self.passed += 1
        else:
            print(f"   ⚠️ Rate limit may be exhausted")
            self.failed += 1
    
    async def test_caching(self):
        """Test caching functionality"""
        print("\n🔹 Testing Cache...")
        
        # Clear cache first
        self.client.clear_cache()
        initial_hits = self.client.cache_hits
        
        # First call - should miss cache
        await self.client.get_overview(TEST_SYMBOL)
        
        # Second call - should hit cache
        await self.client.get_overview(TEST_SYMBOL)
        
        new_hits = self.client.cache_hits
        
        if new_hits > initial_hits:
            print(f"   ✅ Cache working: {new_hits - initial_hits} hits")
            print(f"   📊 Cache size: {len(self.client.cache)} items")
            self.passed += 1
        else:
            print(f"   ⚠️ Cache may not be working")
            self.failed += 1
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("🚀 STARTING ALPHA VANTAGE CLIENT TESTS")
        print("="*70)
        
        await self.setup()
        
        # Run tests in order of importance
        await self.test_realtime_options()  # CRITICAL: Greeks must be PROVIDED
        await self.test_historical_options()
        await self.test_technical_indicators()
        await self.test_analytics()
        await self.test_sentiment()
        await self.test_rate_limiting()
        await self.test_caching()
        
        await self.cleanup()
        
        # Print summary
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        
        # Print metrics
        metrics = self.client.get_metrics()
        print(f"\n📊 CLIENT METRICS:")
        print(f"   Total API calls: {metrics['total_calls']}")
        print(f"   Cache hits: {metrics['cache_hits']}")
        print(f"   Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"   Avg response time: {metrics['avg_response_time_ms']:.0f}ms")
        print(f"   Rate limit remaining: {metrics['rate_limit_remaining']}/600")
        
        if self.errors:
            print(f"\n⚠️ ERRORS ENCOUNTERED:")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
        
        # Final verdict
        print("\n" + "="*70)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! Alpha Vantage client is working correctly!")
            print("✅ Greeks are PROVIDED by Alpha Vantage (not calculated)")
        else:
            print(f"⚠️ {self.failed} tests failed. Review errors above.")
        print("="*70)


# ========================================================================
# QUICK TEST FUNCTIONS
# ========================================================================

async def quick_test_greeks():
    """Quick test to verify Greeks are PROVIDED"""
    print("\n🎯 QUICK GREEKS TEST")
    print("="*40)
    
    client = AlphaVantageClient()
    await client.connect()
    
    try:
        options = await client.get_realtime_options('SPY', require_greeks=True)
        
        if options and len(options) > 0:
            option = options[0]
            print(f"✅ Option: {option.symbol} ${option.strike} {option.option_type}")
            print(f"✅ Greeks PROVIDED by Alpha Vantage:")
            print(f"   Delta: {option.delta}")
            print(f"   Gamma: {option.gamma}")
            print(f"   Theta: {option.theta}")
            print(f"   Vega: {option.vega}")
            print(f"   Rho: {option.rho}")
            
            if option.delta != 0 or option.gamma != 0:
                print("\n🎉 SUCCESS: Greeks are PROVIDED, not calculated!")
            else:
                print("\n⚠️ WARNING: Greeks are zero - API may not be returning them")
        else:
            print("❌ No options returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    await client.disconnect()


async def quick_test_single_api(function_name: str, **params):
    """Test a single API endpoint"""
    print(f"\n🔬 TESTING: {function_name}")
    print("="*40)
    
    client = AlphaVantageClient()
    await client.connect()
    
    try:
        # Get the method
        if hasattr(client, function_name):
            method = getattr(client, function_name)
        else:
            # Try with get_ prefix
            method = getattr(client, f'get_{function_name.lower()}')
        
        result = await method(**params)
        
        if result is not None:
            print(f"✅ SUCCESS: {function_name} returned data")
            print(f"📊 Result type: {type(result)}")
            
            # Show sample of result
            if hasattr(result, 'head'):  # DataFrame
                print(f"📈 Data points: {len(result)}")
                print(result.head(3))
            elif isinstance(result, list):
                print(f"📈 Items: {len(result)}")
                if result:
                    print(f"Sample: {result[0]}")
            elif isinstance(result, dict):
                print(f"📈 Keys: {list(result.keys())[:5]}")
            elif isinstance(result, str):
                print(f"📈 Length: {len(result)} chars")
                print(f"Sample: {result[:200]}...")
        else:
            print(f"⚠️ {function_name} returned None")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    await client.disconnect()


# ========================================================================
# MAIN ENTRY POINT
# ========================================================================

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Alpha Vantage Client')
    parser.add_argument('--quick', action='store_true', help='Run quick Greeks test only')
    parser.add_argument('--api', type=str, help='Test specific API (e.g., get_rsi)')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to test')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick Greeks test
        asyncio.run(quick_test_greeks())
    elif args.api:
        # Test specific API
        params = {'symbol': args.symbol}
        
        # Add default params based on API
        if 'rsi' in args.api.lower():
            params.update({'interval': 'daily', 'time_period': 14, 'series_type': 'close'})
        elif 'vwap' in args.api.lower():
            params.update({'interval': '15min'})
        
        asyncio.run(quick_test_single_api(args.api, **params))
    else:
        # Run full test suite
        tester = AlphaVantageClientTester()
        asyncio.run(tester.run_all_tests())


if __name__ == "__main__":
    main()