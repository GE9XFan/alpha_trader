#!/usr/bin/env python3
"""
Test Day 2 implementation - IBKR Data Ingestion
Tests Level 2 market depth for SPY/QQQ/IWM and standard data for other symbols
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import redis
from main import AlphaTrader
from data_ingestion import IBKRIngestion
from dotenv import load_dotenv
from ib_insync import IB, Stock


class TestDay2IBKR:
    """Comprehensive test suite for Day 2 IBKR implementation"""
    
    def __init__(self):
        self.redis = None
        self.trader = None
        self.ibkr = None
        self.results = {
            'connection': False,
            'symbol_classification': False,
            'level2_subscriptions': False,
            'standard_subscriptions': False,
            'data_flow': False,
            'data_quality': False,
            'performance': False,
            'redis_keys': False
        }
    
    def setup(self):
        """Initialize test environment"""
        print("\n=== Setting up Day 2 Test Environment ===")
        
        # Load environment
        load_dotenv()
        
        # Initialize Redis
        self.redis = redis.Redis(
            host='127.0.0.1', 
            port=6379, 
            decode_responses=True,
            socket_connect_timeout=5
        )
        
        # Verify Redis is running
        if not self.redis.ping():
            raise ConnectionError("Redis not running - start with: redis-server config/redis.conf")
        
        # Initialize trader with config
        self.trader = AlphaTrader()
        self.trader.setup_redis()
        
        print("✓ Test environment initialized")
        return True
    
    async def test_connection(self):
        """Test 1: IBKR Connection"""
        print("\n=== Test 1: IBKR Connection ===")
        
        try:
            # Create IBKR ingestion instance
            self.ibkr = IBKRIngestion(self.trader.config, self.redis)
            
            # Test connection
            await self.ibkr._connect_with_retry()
            
            if self.ibkr.connected:
                print("✓ Connected to IBKR Gateway/TWS")
                
                # Verify Redis flags
                connected = self.redis.get('ibkr:connected')
                assert connected == '1', "IBKR connection flag not set in Redis"
                print("✓ Connection status in Redis")
                
                # Check account info
                account = self.redis.get('ibkr:account')
                print(f"✓ Connected to account: {account}")
                
                self.results['connection'] = True
                return True
            else:
                print("✗ Failed to connect to IBKR")
                print("  Please ensure IBKR Gateway/TWS is running on port 7497")
                return False
                
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False
    
    def test_symbol_classification(self):
        """Test 2: Symbol Classification"""
        print("\n=== Test 2: Symbol Classification ===")
        
        try:
            # Check Level 2 symbols
            expected_level2 = ['SPY', 'QQQ', 'IWM']
            assert set(self.ibkr.level2_symbols) == set(expected_level2), \
                f"Level 2 symbols mismatch. Expected: {expected_level2}, Got: {self.ibkr.level2_symbols}"
            print(f"✓ Level 2 symbols: {self.ibkr.level2_symbols}")
            
            # Check standard symbols
            expected_standard = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'GOOGL', 'META', 'AMZN', 'MSFT', 'VXX']
            assert set(self.ibkr.standard_symbols) == set(expected_standard), \
                f"Standard symbols mismatch"
            print(f"✓ Standard symbols: {len(self.ibkr.standard_symbols)} symbols")
            
            # Verify order book initialization
            for symbol in expected_level2:
                assert symbol in self.ibkr.order_books, f"Order book not initialized for {symbol}"
                book = self.ibkr.order_books[symbol]
                assert 'bids' in book and 'asks' in book, f"Invalid book structure for {symbol}"
            print("✓ Order books initialized for Level 2 symbols only")
            
            self.results['symbol_classification'] = True
            return True
            
        except Exception as e:
            print(f"✗ Symbol classification failed: {e}")
            return False
    
    async def test_subscriptions(self):
        """Test 3: Market Data Subscriptions"""
        print("\n=== Test 3: Market Data Subscriptions ===")
        
        try:
            # Subscribe to all symbols
            await self.ibkr._subscribe_all_symbols()
            
            # Wait for subscriptions to establish
            await asyncio.sleep(2)
            
            # Check Level 2 subscriptions
            level2_count = 0
            for symbol in self.ibkr.level2_symbols:
                if symbol in self.ibkr.subscriptions:
                    sub = self.ibkr.subscriptions[symbol]
                    assert sub['type'] == 'LEVEL2', f"{symbol} should have LEVEL2 subscription"
                    assert 'depth' in sub, f"{symbol} missing depth subscription"
                    level2_count += 1
            
            print(f"✓ Level 2 subscriptions: {level2_count}/{len(self.ibkr.level2_symbols)}")
            self.results['level2_subscriptions'] = (level2_count == len(self.ibkr.level2_symbols))
            
            # Check standard subscriptions
            standard_count = 0
            for symbol in self.ibkr.standard_symbols:
                if symbol in self.ibkr.subscriptions:
                    sub = self.ibkr.subscriptions[symbol]
                    assert sub['type'] == 'STANDARD', f"{symbol} should have STANDARD subscription"
                    assert 'depth' not in sub, f"{symbol} should not have depth subscription"
                    standard_count += 1
            
            print(f"✓ Standard subscriptions: {standard_count}/{len(self.ibkr.standard_symbols)}")
            self.results['standard_subscriptions'] = (standard_count == len(self.ibkr.standard_symbols))
            
            return True
            
        except Exception as e:
            print(f"✗ Subscription test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_data_flow(self):
        """Test 4: Data Flow to Redis"""
        print("\n=== Test 4: Data Flow to Redis ===")
        
        try:
            # Start data collection for a short period
            print("Collecting data for 10 seconds...")
            
            # Create a task for IBKR data collection
            ibkr_task = asyncio.create_task(self._run_ibkr_briefly())
            
            # Wait for data to flow
            await asyncio.sleep(10)
            
            # Stop the task
            ibkr_task.cancel()
            try:
                await ibkr_task
            except asyncio.CancelledError:
                pass
            
            # Check Redis keys for Level 2 symbols
            level2_keys_found = 0
            for symbol in self.ibkr.level2_symbols:
                keys = [
                    f'market:{symbol}:book',
                    f'market:{symbol}:imbalance',
                    f'market:{symbol}:spread',
                    f'market:{symbol}:trades',
                    f'market:{symbol}:last'
                ]
                
                for key in keys:
                    if self.redis.exists(key):
                        level2_keys_found += 1
                
                # Check order book structure
                book_data = self.redis.get(f'market:{symbol}:book')
                if book_data:
                    book = json.loads(book_data)
                    if 'bids' in book and 'asks' in book:
                        print(f"✓ {symbol}: Order book with {len(book['bids'])} bids, {len(book['asks'])} asks")
            
            print(f"✓ Level 2 Redis keys: {level2_keys_found} keys found")
            
            # Check Redis keys for standard symbols
            standard_keys_found = 0
            for symbol in self.ibkr.standard_symbols[:3]:  # Check first 3 for speed
                keys = [
                    f'market:{symbol}:trades',
                    f'market:{symbol}:last',
                    f'market:{symbol}:ticker'
                ]
                
                for key in keys:
                    if self.redis.exists(key):
                        standard_keys_found += 1
                
                # Verify NO order book for standard symbols
                assert not self.redis.exists(f'market:{symbol}:book'), \
                    f"{symbol} should not have order book data"
            
            print(f"✓ Standard Redis keys: {standard_keys_found} keys found (no order books)")
            
            self.results['data_flow'] = True
            return True
            
        except Exception as e:
            print(f"✗ Data flow test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _run_ibkr_briefly(self):
        """Run IBKR ingestion briefly for testing"""
        try:
            # Set up event handlers
            self.ibkr._setup_event_handlers()
            
            # Process events for a short time
            while True:
                # Just use asyncio.sleep in async context
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
    
    def test_data_quality(self):
        """Test 5: Data Quality Validation"""
        print("\n=== Test 5: Data Quality Validation ===")
        
        try:
            issues = []
            
            # Check Level 2 data quality
            for symbol in self.ibkr.level2_symbols:
                # Check spread
                spread = self.redis.get(f'market:{symbol}:spread')
                if spread:
                    spread_val = float(spread)
                    if spread_val < 0:
                        issues.append(f"{symbol}: Negative spread {spread_val}")
                    elif spread_val > 10:
                        issues.append(f"{symbol}: Unusually wide spread {spread_val}")
                
                # Check imbalance
                imbalance = self.redis.get(f'market:{symbol}:imbalance')
                if imbalance:
                    imb_val = float(imbalance)
                    if not -1 <= imb_val <= 1:
                        issues.append(f"{symbol}: Imbalance out of range {imb_val}")
                
                # Check last price
                last = self.redis.get(f'market:{symbol}:last')
                if last:
                    last_val = float(last)
                    if last_val <= 0:
                        issues.append(f"{symbol}: Invalid price {last_val}")
            
            # Check standard symbol data quality
            for symbol in self.ibkr.standard_symbols[:3]:
                # Check trades
                trades_data = self.redis.get(f'market:{symbol}:trades')
                if trades_data:
                    trades = json.loads(trades_data)
                    if trades:
                        latest_trade = trades[-1]
                        if latest_trade['price'] <= 0:
                            issues.append(f"{symbol}: Invalid trade price")
            
            if issues:
                print("⚠️  Data quality issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✓ All data quality checks passed")
                self.results['data_quality'] = True
            
            return len(issues) == 0
            
        except Exception as e:
            print(f"✗ Data quality test failed: {e}")
            return False
    
    def test_performance(self):
        """Test 6: Performance Metrics"""
        print("\n=== Test 6: Performance Metrics ===")
        
        try:
            # Check processing times
            metrics = self.redis.hgetall('monitoring:ibkr:metrics')
            
            if metrics:
                # Check average processing time
                avg_ms = float(metrics.get('avg_processing_ms', 0))
                max_ms = float(metrics.get('max_processing_ms', 0))
                
                print(f"Processing times - Avg: {avg_ms}ms, Max: {max_ms}ms")
                
                # Performance thresholds
                if avg_ms > 0:
                    if avg_ms < 50:  # Target for Level 2
                        print("✓ Average processing time < 50ms")
                    else:
                        print(f"⚠️  Average processing time {avg_ms}ms > 50ms target")
                
                    if max_ms < 100:
                        print("✓ Max processing time < 100ms")
                    else:
                        print(f"⚠️  Max processing time {max_ms}ms > 100ms target")
                
                # Check update counts
                depth_updates = int(metrics.get('depth_updates', 0))
                trade_updates = int(metrics.get('trade_updates', 0))
                bar_updates = int(metrics.get('bar_updates', 0))
                
                print(f"Update counts - Depth: {depth_updates}, Trades: {trade_updates}, Bars: {bar_updates}")
                
                self.results['performance'] = avg_ms < 50 if avg_ms > 0 else False
            else:
                print("⚠️  No performance metrics available yet")
            
            return True
            
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
            return False
    
    def test_redis_keys(self):
        """Test 7: Redis Key Structure"""
        print("\n=== Test 7: Redis Key Structure ===")
        
        try:
            # Expected key patterns
            expected_patterns = {
                'market:*:book': "Order books (Level 2 only)",
                'market:*:trades': "Trade lists",
                'market:*:last': "Last prices",
                'market:*:bars': "5-second bars",
                'market:*:timestamp': "Update timestamps",
                'market:*:spread': "Bid-ask spreads",
                'market:*:imbalance': "Order book imbalances (Level 2)",
                'ibkr:connected': "Connection status",
                'module:heartbeat:ibkr_ingestion': "Module heartbeat",
                'monitoring:ibkr:metrics': "Performance metrics"
            }
            
            found_patterns = {}
            for pattern, description in expected_patterns.items():
                keys = self.redis.keys(pattern)
                if keys:
                    found_patterns[pattern] = len(keys)
                    print(f"✓ {pattern}: {len(keys)} keys - {description}")
                else:
                    print(f"⚠️  {pattern}: No keys found - {description}")
            
            # Verify Level 2 vs Standard separation
            book_keys = self.redis.keys('market:*:book')
            if book_keys:
                book_symbols = [key.split(':')[1] for key in book_keys]
                for symbol in book_symbols:
                    assert symbol in self.ibkr.level2_symbols, \
                        f"{symbol} has order book but is not a Level 2 symbol"
                print(f"✓ Order books only for Level 2 symbols: {book_symbols}")
            
            self.results['redis_keys'] = len(found_patterns) >= 5
            return True
            
        except Exception as e:
            print(f"✗ Redis keys test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test resources"""
        print("\n=== Cleaning up test resources ===")
        
        try:
            # Stop IBKR if running
            if self.ibkr:
                await self.ibkr.stop()
            
            # Clear test keys from Redis
            test_keys = self.redis.keys('market:*')
            if test_keys:
                self.redis.delete(*test_keys)
            
            # Close Redis connection
            if self.redis:
                self.redis.close()
            
            print("✓ Cleanup complete")
            
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("Day 2 IBKR Test Summary")
        print("=" * 60)
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for test, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test.ljust(25)}: {status}")
        
        print("-" * 60)
        print(f"Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All Day 2 tests passed! IBKR ingestion is working correctly.")
            print("\nNext steps:")
            print("1. Monitor data flow with: redis-cli monitor")
            print("2. Check order books: redis-cli get market:SPY:book | jq")
            print("3. Verify metrics: redis-cli hgetall monitoring:ibkr:metrics")
            print("4. Proceed to Day 3: Alpha Vantage Integration")
        else:
            print("\n⚠️  Some tests failed. Please review and fix issues before proceeding.")
            if not self.results['connection']:
                print("\n💡 Tip: Ensure IBKR Gateway/TWS is running on port 7497")
                print("   - Use paper trading account")
                print("   - Enable API connections in Gateway/TWS settings")
        
        return passed == total


async def main():
    """Main test runner for Day 2"""
    print("\n" + "🚀 " * 20)
    print("AlphaTrader Pro - Day 2 IBKR Data Ingestion Test Suite")
    print("🚀 " * 20)
    
    # Check prerequisites
    print("\n=== Checking Prerequisites ===")
    
    # Check Redis
    try:
        r = redis.Redis(host='127.0.0.1', port=6379, socket_connect_timeout=2)
        if not r.ping():
            raise ConnectionError()
        r.close()
        print("✓ Redis is running")
    except:
        print("✗ Redis not running - start with: redis-server config/redis.conf")
        return False
    
    # Check IBKR Gateway/TWS
    print("⚠️  Please ensure IBKR Gateway/TWS is running on port 7497 (paper trading)")
    print("   Press Enter to continue...")
    input()
    
    # Run tests
    tester = TestDay2IBKR()
    
    try:
        # Setup
        if not tester.setup():
            return False
        
        # Run tests sequentially
        await tester.test_connection()
        
        if tester.results['connection']:
            tester.test_symbol_classification()
            await tester.test_subscriptions()
            await tester.test_data_flow()
            tester.test_data_quality()
            tester.test_performance()
            tester.test_redis_keys()
        
        # Print summary
        success = tester.print_summary()
        
        # Cleanup
        await tester.cleanup()
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        await tester.cleanup()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)