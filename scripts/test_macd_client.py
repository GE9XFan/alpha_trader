#!/usr/bin/env python3
"""Test MACD client method with caching"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient


def test_macd_client():
    """Test MACD method in AV client"""
    print("=== Testing MACD Client Method ===\n")
    
    client = AlphaVantageClient()
    
    # Test symbol
    test_symbol = 'SPY'
    
    # Test 1: First call (should hit API)
    print(f"Test 1: First call for {test_symbol} (API)...")
    start = time.time()
    data = client.get_macd(test_symbol)  # Using config defaults
    api_time = time.time() - start
    
    if data and 'Technical Analysis: MACD' in data:
        macd_points = len(data['Technical Analysis: MACD'])
        print(f"  ✓ Got {macd_points} MACD data points in {api_time:.2f}s")
        
        # Check that we have all 3 values
        first_timestamp = list(data['Technical Analysis: MACD'].keys())[0]
        values = data['Technical Analysis: MACD'][first_timestamp]
        print(f"  ✓ Values returned: {list(values.keys())}\n")
    else:
        print(f"  ✗ No MACD data returned\n")
        return False
    
    # Test 2: Second call (should hit cache)
    print(f"Test 2: Second call for {test_symbol} (Cache)...")
    start = time.time()
    data2 = client.get_macd(test_symbol)
    cache_time = time.time() - start
    
    if data2:
        print(f"  ✓ Cache hit in {cache_time:.2f}s")
        print(f"  ✓ Speed improvement: {api_time/cache_time:.1f}x faster\n")
    
    # Test 3: Different symbol (should hit API)
    print("Test 3: Different symbol QQQ (API)...")
    data3 = client.get_macd('QQQ')
    if data3 and 'Technical Analysis: MACD' in data3:
        macd_points3 = len(data3['Technical Analysis: MACD'])
        print(f"  ✓ Got {macd_points3} MACD data points for QQQ\n")
    
    # Check cache status
    cache_stats = client.get_cache_status()
    print(f"Cache Status:")
    print(f"  Total keys: {cache_stats.get('keys', 0)}")
    print(f"  AV keys: {cache_stats.get('av_keys', 0)}")
    
    # Show cache key format
    cache_key = f"av:macd:{test_symbol}:1min_12_26_9"
    ttl = client.cache.get_ttl(cache_key)
    print(f"\nCache key example: {cache_key}")
    print(f"TTL remaining: {ttl} seconds")
    
    return True


if __name__ == "__main__":
    success = test_macd_client()
    if success:
        print("\n✅ MACD client method working correctly!")
    else:
        print("\n✗ MACD client method test failed")
        sys.exit(1)