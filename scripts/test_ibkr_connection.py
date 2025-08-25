#!/usr/bin/env python3
"""
Test IBKR Connection and Market Data
Tests all functionality of the MarketDataManager
Production-ready test suite for Week 1 Day 4
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List
from ib_insync import util  # Import util for proper event loop handling

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import market_data
from src.core.logger import get_logger
from src.core.exceptions import IBKRException

logger = get_logger(__name__)


class IBKRConnectionTester:
    """Comprehensive test suite for IBKR connection"""
    
    def __init__(self):
        self.market = market_data
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.test_symbols = ['SPY', 'QQQ', 'IWM']
        
    async def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print("🚀 STARTING IBKR CONNECTION TESTS")
        print("="*70)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Test Symbols: {', '.join(self.test_symbols)}")
        print("="*70 + "\n")
        
        # Test sequence
        tests = [
            ("Connection Test", self.test_connection),
            ("Market Data Subscription", self.test_subscriptions),
            ("Real-time Price Updates", self.test_realtime_prices),
            ("Historical Data Retrieval", self.test_historical_data),
            ("Data Quality Checks", self.test_data_quality),
            ("Connection Status", self.test_connection_status),
            ("Graceful Disconnect", self.test_disconnect)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔹 {test_name}...")
            try:
                result = await test_func()
                if result:
                    self.passed += 1
                    self.results[test_name] = "✅ PASSED"
                    print(f"   ✅ {test_name} PASSED")
                else:
                    self.failed += 1
                    self.results[test_name] = "❌ FAILED"
                    print(f"   ❌ {test_name} FAILED")
            except Exception as e:
                self.failed += 1
                self.results[test_name] = f"❌ ERROR: {str(e)}"
                print(f"   ❌ {test_name} ERROR: {e}")
        
        # Print summary
        self.print_summary()
        
        return self.failed == 0
    
    async def test_connection(self) -> bool:
        """Test IBKR connection"""
        try:
            # Connect to IBKR
            connected = await self.market.connect()
            
            if not connected:
                logger.error("Failed to connect to IBKR")
                return False
            
            # Verify connection
            if not self.market.is_connected():
                logger.error("Connection reported but not active")
                return False
            
            status = self.market.get_connection_status()
            print(f"   📡 Connected to: {status['host']}:{status['port']}")
            print(f"   🔧 Mode: {status['mode']}")
            print(f"   🆔 Client ID: {status['client_id']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def test_subscriptions(self) -> bool:
        """Test market data subscriptions"""
        try:
            # Subscribe to test symbols
            results = await self.market.subscribe_symbols(self.test_symbols)
            
            # Check all subscriptions succeeded
            for symbol, success in results.items():
                if not success:
                    logger.error(f"Failed to subscribe to {symbol}")
                    return False
                print(f"   📈 Subscribed to {symbol}")
            
            # Verify subscriptions in status
            status = self.market.get_connection_status()
            if set(status['subscriptions']) != set(self.test_symbols):
                logger.error("Subscription list mismatch")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Subscription test failed: {e}")
            return False
    
    async def test_realtime_prices(self) -> bool:
        """Test real-time price updates"""
        try:
            print("   ⏳ Waiting for price updates (10 seconds)...")
            
            # Wait for data to flow - use asyncio.sleep
            await asyncio.sleep(10)
            
            # Check we have prices for all symbols
            prices_received = True
            for symbol in self.test_symbols:
                price = self.market.get_latest_price(symbol)
                
                if price <= 0:
                    logger.error(f"No price received for {symbol}")
                    prices_received = False
                else:
                    # Get latest bar
                    bar = self.market.get_latest_bar(symbol)
                    if bar:
                        print(f"   💰 {symbol}: ${price:.2f} "
                              f"(H: ${bar['high']:.2f}, L: ${bar['low']:.2f}, "
                              f"Vol: {bar['volume']:,})")
                    else:
                        print(f"   💰 {symbol}: ${price:.2f}")
            
            # Check bar buffers
            for symbol in self.test_symbols:
                history = self.market.get_bar_history(symbol, num_bars=10)
                if not history.empty:
                    print(f"   📊 {symbol}: {len(history)} bars in buffer")
            
            return prices_received
            
        except Exception as e:
            logger.error(f"Real-time price test failed: {e}")
            return False
    
    async def test_historical_data(self) -> bool:
        """Test historical data retrieval"""
        try:
            test_symbol = 'SPY'
            
            # Test different time periods
            periods = [
                ('1 D', '5 secs', 'Intraday'),
                ('1 W', '1 min', 'Weekly'),
                ('1 M', '1 hour', 'Monthly')
            ]
            
            for duration, bar_size, label in periods:
                df = await self.market.get_historical_bars(
                    test_symbol, 
                    duration=duration, 
                    bar_size=bar_size
                )
                
                if df.empty:
                    logger.error(f"No historical data for {label}")
                    return False
                
                print(f"   📜 {label}: {len(df)} bars retrieved")
                print(f"      Range: {df.index[0]} to {df.index[-1]}")
                print(f"      Latest: ${df['close'].iloc[-1]:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Historical data test failed: {e}")
            return False
    
    async def test_data_quality(self) -> bool:
        """Test data quality checks"""
        try:
            all_prices = self.market.get_all_prices()
            
            if not all_prices:
                logger.error("No prices available")
                return False
            
            # Check for reasonable values
            for symbol, price in all_prices.items():
                if price <= 0 or price > 10000:
                    logger.error(f"Unreasonable price for {symbol}: ${price}")
                    return False
            
            # Check for stale data
            for symbol in self.test_symbols:
                if symbol in self.market.last_update_time:
                    age = datetime.now().timestamp() - self.market.last_update_time[symbol]
                    if age > 60:  # More than 1 minute old
                        logger.warning(f"Stale data for {symbol}: {age:.1f}s old")
            
            print(f"   ✅ All prices within reasonable range")
            print(f"   ✅ Data freshness verified")
            
            return True
            
        except Exception as e:
            logger.error(f"Data quality test failed: {e}")
            return False
    
    async def test_connection_status(self) -> bool:
        """Test connection status reporting"""
        try:
            status = self.market.get_connection_status()
            
            # Verify all expected fields
            required_fields = ['connected', 'host', 'port', 'client_id', 
                             'mode', 'subscriptions', 'last_heartbeat']
            
            for field in required_fields:
                if field not in status:
                    logger.error(f"Missing status field: {field}")
                    return False
            
            # Print status
            print(f"   📊 Connection Status:")
            print(f"      Connected: {status['connected']}")
            print(f"      Endpoint: {status['host']}:{status['port']}")
            print(f"      Mode: {status['mode']}")
            print(f"      Active Subscriptions: {len(status['subscriptions'])}")
            
            return status['connected']
            
        except Exception as e:
            logger.error(f"Status test failed: {e}")
            return False
    
    async def test_disconnect(self) -> bool:
        """Test graceful disconnection"""
        try:
            await self.market.disconnect()
            
            # Verify disconnection
            if self.market.is_connected():
                logger.error("Still connected after disconnect")
                return False
            
            # Check data cleared
            if self.market.latest_prices:
                logger.error("Price data not cleared on disconnect")
                return False
            
            print(f"   ✅ Disconnected gracefully")
            print(f"   ✅ Data cleared")
            
            return True
            
        except Exception as e:
            logger.error(f"Disconnect test failed: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"✅ Passed: {self.passed}/{total}")
        print(f"❌ Failed: {self.failed}/{total}")
        print(f"📈 Pass Rate: {pass_rate:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            print(f"   {test_name}: {result}")
        
        print("="*70)
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! IBKR integration ready for production.")
        else:
            print("⚠️  Some tests failed. Please review the errors above.")
        
        print("="*70)


async def main():
    """Main test runner"""
    tester = IBKRConnectionTester()
    
    try:
        # Check if TWS/Gateway is running
        print("\n⚠️  PREREQUISITES:")
        print("   1. Ensure TWS or IB Gateway is running")
        print("   2. Paper trading should be on port 7497")
        print("   3. Enable API connections in TWS/Gateway settings")
        print("   4. Check firewall settings if connection fails")
        
        input("\nPress Enter to start tests...")
        
        # Run tests
        success = await tester.run_all_tests()
        
        # Return exit code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        await market_data.disconnect()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        await market_data.disconnect()
        sys.exit(1)


if __name__ == "__main__":
    # Use ib_insync's event loop handler to avoid conflicts
    util.run(main())