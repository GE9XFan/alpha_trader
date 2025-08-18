#!/usr/bin/env python3
"""Comprehensive test to verify all methods work together properly"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

def test_av_client_methods():
    """Test all AlphaVantageClient methods"""
    print("=== Testing AlphaVantageClient Methods ===\n")
    
    from src.connections.av_client import AlphaVantageClient
    
    client = AlphaVantageClient()
    symbol = "SPY"
    
    tests_passed = 0
    tests_failed = 0
    
    # Test each method
    methods_to_test = [
        ("get_realtime_options", [symbol]),
        ("get_historical_options", [symbol]),
        ("get_rsi", [symbol]),
        ("get_macd", [symbol]),
        ("get_bbands", [symbol]),  # Should work with defaults now
        ("get_vwap", [symbol]),
        ("get_rate_limit_status", []),
        ("get_cache_status", [])
    ]
    
    for method_name, args in methods_to_test:
        try:
            print(f"Testing {method_name}...", end=" ")
            method = getattr(client, method_name)
            result = method(*args)
            
            if result is not None:
                print("✅ PASSED")
                tests_passed += 1
            else:
                print("⚠️ Returned None")
                tests_failed += 1
                
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:50]}")
            tests_failed += 1
    
    print(f"\nClient Results: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def test_ingestion_methods():
    """Test all DataIngestion methods"""
    print("\n=== Testing DataIngestion Methods ===\n")
    
    from src.data.ingestion import DataIngestion
    from src.connections.av_client import AlphaVantageClient
    
    ingestion = DataIngestion()
    client = AlphaVantageClient()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test ingestion for each data type
    test_cases = [
        ("ingest_options_data", lambda: client.get_realtime_options("SPY"), ["SPY"]),
        ("ingest_historical_options", lambda: client.get_historical_options("SPY"), ["SPY"]),
        ("ingest_rsi_data", lambda: client.get_rsi("SPY"), ["SPY", "1min", 14]),
        ("ingest_macd_data", lambda: client.get_macd("SPY"), ["SPY", "1min", 12, 26, 9, "close"]),
        ("ingest_bbands_data", lambda: client.get_bbands("SPY"), ["SPY", "5min", 20, 2, 2, 0, "close"]),
        ("ingest_vwap_data", lambda: client.get_vwap("SPY"), ["SPY", "5min"])
    ]
    
    for method_name, data_fetcher, extra_args in test_cases:
        try:
            print(f"Testing {method_name}...", end=" ")
            
            # Get data from API
            api_data = data_fetcher()
            
            if api_data:
                # Try to ingest
                method = getattr(ingestion, method_name)
                result = method(api_data, *extra_args)
                
                if result >= 0:
                    print(f"✅ PASSED ({result} records)")
                    tests_passed += 1
                else:
                    print("⚠️ No records processed")
                    tests_failed += 1
            else:
                print("⚠️ No API data")
                tests_failed += 1
                
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:50]}")
            tests_failed += 1
        
        time.sleep(1)  # Rate limit respect
    
    print(f"\nIngestion Results: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def test_scheduler_methods():
    """Test scheduler initialization and job creation"""
    print("\n=== Testing DataScheduler Methods ===\n")
    
    from src.data.scheduler import DataScheduler
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        print("Testing scheduler initialization...", end=" ")
        scheduler = DataScheduler(test_mode=True)
        print("✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
        return tests_passed, tests_failed
    
    # Test job creation methods
    job_methods = [
        "_schedule_realtime_options",
        "_schedule_historical_options", 
        "_schedule_rsi_indicators",
        "_schedule_macd_indicators",
        "_schedule_bbands_indicators",
        "_schedule_vwap_indicators"
    ]
    
    for method_name in job_methods:
        try:
            print(f"Testing {method_name}...", end=" ")
            method = getattr(scheduler, method_name)
            method()
            print("✅ PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:50]}")
            tests_failed += 1
    
    # Test fetch methods
    fetch_tests = [
        ("_fetch_realtime_options", ["SPY"]),
        ("_fetch_historical_options", ["SPY"]),
        ("_fetch_rsi", ["SPY", "1min", 14]),
        ("_fetch_macd", ["SPY", "1min", 12, 26, 9, "close"]),
        ("_fetch_bbands", ["SPY", "5min", 20, "close", 2, 2, 0]),
        ("_fetch_vwap", ["SPY", "5min"]),
        ("_fetch_atr", ["SPY"]),  # ← Just symbol (uses config defaults)
        ("_fetch_adx", ["SPY"])   # ← Just symbol (uses config defaults)
    ]
    
    for method_name, args in fetch_tests:
        try:
            print(f"Testing {method_name}...", end=" ")
            method = getattr(scheduler, method_name)
            method(*args)
            print("✅ PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:50]}")
            tests_failed += 1
        
        time.sleep(1)  # Rate limit
    
    # Cleanup
    scheduler.stop()
    
    print(f"\nScheduler Results: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def test_cache_integration():
    """Test cache manager integration"""
    print("\n=== Testing Cache Integration ===\n")
    
    from src.data.cache_manager import get_cache
    
    cache = get_cache()
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Test basic operations
        print("Testing cache set/get...", end=" ")
        cache.set("test:key", {"data": "test"}, ttl=10)
        result = cache.get("test:key")
        
        if result and result.get("data") == "test":
            print("✅ PASSED")
            tests_passed += 1
        else:
            print("❌ FAILED")
            tests_failed += 1
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    try:
        # Test pattern operations
        print("Testing cache patterns...", end=" ")
        cache.set("av:test:1", {"id": 1}, ttl=10)
        cache.set("av:test:2", {"id": 2}, ttl=10)
        
        keys = cache.redis_client.keys("av:test:*")
        if len(keys) >= 2:
            print("✅ PASSED")
            tests_passed += 1
        else:
            print("❌ FAILED")
            tests_failed += 1
            
        # Cleanup
        cache.flush_pattern("av:test:*")
        cache.delete("test:key")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    print(f"\nCache Results: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("COMPREHENSIVE INTEGRATION TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    test_suites = [
        test_av_client_methods,
        test_ingestion_methods,
        test_scheduler_methods,
        test_cache_integration
    ]
    
    for test_suite in test_suites:
        try:
            passed, failed = test_suite()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            total_failed += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("Your system is fully integrated and working!")
    else:
        print(f"\n⚠️ {total_failed} tests failed. Please review and fix the issues.")
        print("\nCommon fixes:")
        print("1. Run the fix script: python fix_integration_issues.py")
        print("2. Check Redis is running: redis-cli ping")
        print("3. Verify database connection in .env")
        print("4. Check API key is valid")
    
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())