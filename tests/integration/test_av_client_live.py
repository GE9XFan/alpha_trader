#!/usr/bin/env python3
"""
Test AlphaVantageClient with live API calls
Tests a representative sample from each category
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.foundation.config_manager import ConfigManager
from src.connections.av_client import AlphaVantageClient
from src.data.rate_limiter import TokenBucketRateLimiter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlphaVantageClientTester:
    """Test the AlphaVantageClient with live calls"""
    
    def __init__(self):
        """Initialize tester with client"""
        logger.info("Initializing AlphaVantageClient tester...")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.rate_limiter = TokenBucketRateLimiter(
            provider='alpha_vantage',
            config_manager=self.config_manager
        )
        
        # Initialize client
        self.client = AlphaVantageClient(
            config_manager=self.config_manager,
            rate_limiter=self.rate_limiter
        )
        
        # Create output directory
        self.output_dir = Path("data/av_client_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track results
        self.results = {}
        self.responses = {}
    
    def save_response(self, api_name: str, response: Any) -> None:
        """Save API response to file"""
        if response:
            filename = self.output_dir / f"{api_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(response, f, indent=2, default=str)
            logger.info(f"  💾 Response saved to {filename}")
    
    def test_connection(self) -> bool:
        """Test basic connection"""
        logger.info("\n" + "="*60)
        logger.info("🔌 TESTING CONNECTION")
        logger.info("="*60)
        
        connected = self.client.connect()
        logger.info(f"Connection status: {'✅ Connected' if connected else '❌ Failed'}")
        return connected
    
    def test_options_apis(self) -> None:
        """Test options APIs with Greeks"""
        logger.info("\n" + "="*60)
        logger.info("📊 TESTING OPTIONS APIs (with Greeks)")
        logger.info("="*60)
        
        # Test realtime options
        logger.info("\n1. Testing REALTIME_OPTIONS (PRIMARY GREEKS SOURCE)...")
        try:
            response = self.client.get_realtime_options('SPY')
            if response:
                self.results['realtime_options'] = 'SUCCESS'
                self.save_response('realtime_options', response)
                
                # Check for Greeks in response
                if 'options' in response:
                    logger.info("  ✅ Options data received")
                    # Check first option for Greeks
                    first_option = next(iter(response.get('options', {}).values()), {})
                    if first_option:
                        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
                        has_greeks = all(g in str(first_option).lower() for g in greeks)
                        logger.info(f"  📈 Greeks present: {'✅ Yes' if has_greeks else '⚠️ No'}")
            else:
                self.results['realtime_options'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['realtime_options'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
        
        time.sleep(0.5)  # Small delay between calls
        
        # Test historical options
        logger.info("\n2. Testing HISTORICAL_OPTIONS...")
        try:
            response = self.client.get_historical_options('SPY')
            if response:
                self.results['historical_options'] = 'SUCCESS'
                self.save_response('historical_options', response)
                logger.info("  ✅ Historical options received")
            else:
                self.results['historical_options'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['historical_options'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
    
    def test_technical_indicators(self) -> None:
        """Test key technical indicators"""
        logger.info("\n" + "="*60)
        logger.info("📈 TESTING TECHNICAL INDICATORS")
        logger.info("="*60)
        
        indicators = [
            ('RSI', self.client.get_rsi, {'symbol': 'SPY'}),
            ('MACD', self.client.get_macd, {'symbol': 'SPY'}),
            ('BBANDS', self.client.get_bbands, {'symbol': 'SPY'}),
            ('VWAP', self.client.get_vwap, {'symbol': 'SPY', 'interval': '5min'}),
            ('ATR', self.client.get_atr, {'symbol': 'SPY'}),
        ]
        
        for i, (name, func, params) in enumerate(indicators, 1):
            logger.info(f"\n{i}. Testing {name}...")
            try:
                response = func(**params)
                if response:
                    self.results[name.lower()] = 'SUCCESS'
                    self.save_response(name.lower(), response)
                    
                    # Check for indicator data
                    if f'Technical Analysis: {name}' in str(response):
                        logger.info(f"  ✅ {name} data received")
                    else:
                        logger.info(f"  ⚠️ Response received but may not contain {name} data")
                else:
                    self.results[name.lower()] = 'FAILED'
                    logger.error("  ❌ No response received")
            except Exception as e:
                self.results[name.lower()] = f'ERROR: {str(e)}'
                logger.error(f"  ❌ Error: {e}")
            
            time.sleep(0.5)  # Rate limit respect
    
    def test_analytics_apis(self) -> None:
        """Test analytics APIs"""
        logger.info("\n" + "="*60)
        logger.info("🔬 TESTING ANALYTICS APIs")
        logger.info("="*60)
        
        # Test fixed window
        logger.info("\n1. Testing ANALYTICS_FIXED_WINDOW...")
        try:
            response = self.client.get_analytics_fixed_window('AAPL,META')
            if response:
                self.results['analytics_fixed_window'] = 'SUCCESS'
                self.save_response('analytics_fixed_window', response)
                logger.info("  ✅ Fixed window analytics received")
            else:
                self.results['analytics_fixed_window'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['analytics_fixed_window'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
        
        time.sleep(0.5)
        
        # Test sliding window
        logger.info("\n2. Testing ANALYTICS_SLIDING_WINDOW...")
        try:
            response = self.client.get_analytics_sliding_window('AAPL,META')
            if response:
                self.results['analytics_sliding_window'] = 'SUCCESS'
                self.save_response('analytics_sliding_window', response)
                logger.info("  ✅ Sliding window analytics received")
            else:
                self.results['analytics_sliding_window'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['analytics_sliding_window'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
    
    def test_fundamental_data(self) -> None:
        """Test fundamental data APIs"""
        logger.info("\n" + "="*60)
        logger.info("💼 TESTING FUNDAMENTAL DATA")
        logger.info("="*60)
        
        # Test company overview
        logger.info("\n1. Testing OVERVIEW...")
        try:
            response = self.client.get_overview('AAPL')
            if response:
                self.results['overview'] = 'SUCCESS'
                self.save_response('overview', response)
                
                # Check for key fields
                if 'Symbol' in response and 'MarketCapitalization' in response:
                    logger.info("  ✅ Company overview received")
                    logger.info(f"  📊 Company: {response.get('Name', 'N/A')}")
                    logger.info(f"  📊 Sector: {response.get('Sector', 'N/A')}")
            else:
                self.results['overview'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['overview'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
        
        time.sleep(0.5)
        
        # Test earnings
        logger.info("\n2. Testing EARNINGS...")
        try:
            response = self.client.get_earnings('AAPL')
            if response:
                self.results['earnings'] = 'SUCCESS'
                self.save_response('earnings', response)
                logger.info("  ✅ Earnings data received")
            else:
                self.results['earnings'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['earnings'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
    
    def test_economic_indicators(self) -> None:
        """Test economic indicator APIs"""
        logger.info("\n" + "="*60)
        logger.info("🏛️ TESTING ECONOMIC INDICATORS")
        logger.info("="*60)
        
        # Test treasury yield
        logger.info("\n1. Testing TREASURY_YIELD...")
        try:
            response = self.client.get_treasury_yield()
            if response:
                self.results['treasury_yield'] = 'SUCCESS'
                self.save_response('treasury_yield', response)
                logger.info("  ✅ Treasury yield data received")
            else:
                self.results['treasury_yield'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['treasury_yield'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
        
        time.sleep(0.5)
        
        # Test CPI
        logger.info("\n2. Testing CPI...")
        try:
            response = self.client.get_cpi()
            if response:
                self.results['cpi'] = 'SUCCESS'
                self.save_response('cpi', response)
                logger.info("  ✅ CPI data received")
            else:
                self.results['cpi'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['cpi'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
    
    def test_sentiment_apis(self) -> None:
        """Test sentiment and news APIs"""
        logger.info("\n" + "="*60)
        logger.info("📰 TESTING SENTIMENT & NEWS")
        logger.info("="*60)
        
        # Test news sentiment
        logger.info("\n1. Testing NEWS_SENTIMENT...")
        try:
            response = self.client.get_news_sentiment('AAPL')
            if response:
                self.results['news_sentiment'] = 'SUCCESS'
                self.save_response('news_sentiment', response)
                
                # Check for feed items
                if 'feed' in response:
                    logger.info(f"  ✅ News sentiment received ({len(response.get('feed', []))} articles)")
            else:
                self.results['news_sentiment'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['news_sentiment'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
        
        time.sleep(0.5)
        
        # Test top gainers/losers
        logger.info("\n2. Testing TOP_GAINERS_LOSERS...")
        try:
            response = self.client.get_top_gainers_losers()
            if response:
                self.results['top_gainers_losers'] = 'SUCCESS'
                self.save_response('top_gainers_losers', response)
                
                # Check for data
                if 'top_gainers' in response and 'top_losers' in response:
                    logger.info("  ✅ Top gainers/losers received")
                    logger.info(f"  📈 Top gainer: {response['top_gainers'][0]['ticker'] if response.get('top_gainers') else 'N/A'}")
                    logger.info(f"  📉 Top loser: {response['top_losers'][0]['ticker'] if response.get('top_losers') else 'N/A'}")
            else:
                self.results['top_gainers_losers'] = 'FAILED'
                logger.error("  ❌ No response received")
        except Exception as e:
            self.results['top_gainers_losers'] = f'ERROR: {str(e)}'
            logger.error(f"  ❌ Error: {e}")
    
    def test_rate_limiter_integration(self) -> None:
        """Test that rate limiter is working"""
        logger.info("\n" + "="*60)
        logger.info("⏱️ TESTING RATE LIMITER INTEGRATION")
        logger.info("="*60)
        
        # Get rate limiter stats
        stats = self.client.rate_limiter.get_statistics()
        
        logger.info(f"\n📊 Rate Limiter Statistics:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Successful acquisitions: {stats['successful_acquisitions']}")
        logger.info(f"  Current RPM: {stats['current_rpm']:.1f}")
        logger.info(f"  Available tokens: {stats['available_tokens']:.1f}/{stats['max_tokens']}")
        logger.info(f"  Market period: {stats['market_period']}")
        logger.info(f"  Is MOC window: {stats['is_moc_window']}")
        
        # Check health
        health = self.client.rate_limiter.health_check()
        logger.info(f"\n🏥 Rate Limiter Health:")
        logger.info(f"  Status: {'✅ Healthy' if health['healthy'] else '⚠️ Issues detected'}")
        if health['warnings']:
            for warning in health['warnings']:
                logger.warning(f"  ⚠️ {warning}")
    
    def print_summary(self) -> None:
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("📋 TEST SUMMARY")
        logger.info("="*60)
        
        # Count results
        success = [k for k, v in self.results.items() if v == 'SUCCESS']
        failed = [k for k, v in self.results.items() if v != 'SUCCESS']
        
        logger.info(f"\n✅ Successful: {len(success)}/{len(self.results)}")
        logger.info(f"❌ Failed: {len(failed)}/{len(self.results)}")
        
        if success:
            logger.info("\nSuccessful APIs:")
            for api in success:
                logger.info(f"  ✅ {api}")
        
        if failed:
            logger.info("\nFailed APIs:")
            for api in failed:
                logger.info(f"  ❌ {api}: {self.results[api]}")
        
        # Save results
        results_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n💾 Results saved to: {results_file}")
        
        # Client statistics
        client_stats = self.client.get_statistics()
        logger.info(f"\n📊 Client Statistics:")
        logger.info(f"  Total API calls: {client_stats['total_calls']}")
        logger.info(f"  Error count: {client_stats['error_count']}")
        logger.info(f"  Connected: {client_stats['is_connected']}")
    
    def run_all_tests(self) -> None:
        """Run all test categories"""
        logger.info("="*60)
        logger.info("🚀 ALPHA VANTAGE CLIENT LIVE TEST")
        logger.info("="*60)
        logger.info(f"Environment: {self.config_manager.environment}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Test connection first
        if not self.test_connection():
            logger.error("❌ Connection failed, aborting tests")
            return
        
        # Run test categories
        test_categories = [
            ("Options APIs", self.test_options_apis),
            ("Technical Indicators", self.test_technical_indicators),
            ("Analytics APIs", self.test_analytics_apis),
            ("Fundamental Data", self.test_fundamental_data),
            ("Economic Indicators", self.test_economic_indicators),
            ("Sentiment APIs", self.test_sentiment_apis),
        ]
        
        for category_name, test_func in test_categories:
            try:
                test_func()
            except Exception as e:
                logger.error(f"❌ Error in {category_name}: {e}")
            
            # Small delay between categories
            time.sleep(1)
        
        # Test rate limiter
        self.test_rate_limiter_integration()
        
        # Print summary
        self.print_summary()
        
        # Disconnect
        self.client.disconnect()
        logger.info("\n✅ All tests completed!")


def main():
    """Run the Alpha Vantage client tests"""
    tester = AlphaVantageClientTester()
    tester.run_all_tests()
    return 0


if __name__ == "__main__":
    sys.exit(main())