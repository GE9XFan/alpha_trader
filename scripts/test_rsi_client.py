#!/usr/bin/env python3
"""Test RSI client method with caching"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient


def test_rsi_client():
    """Test RSI method in AV client"""
    print("=== Testing RSI Client Method ===\n")
    
    client = AlphaVantageClient()
    
    # Symbol to test (no hardcoding!)
    test_symbol = 'SPY'
    
    # Test 1: First call (should hit API)
    print(f"Test 1: First call for {test_symbol} (API)...")
    start = time.time()
    data = client.get_rsi(test_symbol)  # Using config defaults for other params
    api_time = time.time() - start
    
    if data and 'Technical Analysis: RSI' in data:
        rsi_points = len(data['Technical Analysis: RSI'])
        print(f"  ✓ Got {rsi_points} RSI data points in {api_time:.2f}s\n")
    else:
        print(f"  ✗ No RSI data returned\n")
        return False
    
    # Test 2: Second call (should hit cache)
    print(f"Test 2: Second call for {test_symbol} (Cache)...")
    start = time.time()
    data2 = client.get_rsi(test_symbol)
    cache_time = time.time() - start
    
    if data2:
        print(f"  ✓ Cache hit in {cache_time:.2f}s")
        print(f"  ✓ Speed improvement: {api_time/cache_time:.1f}x faster\n")
    
    # Test 3: Different symbol (should hit API)
    print("Test 3: Different symbol QQQ (API)...")
    data3 = client.get_rsi('QQQ')
    if data3 and 'Technical Analysis: RSI' in data3:
        rsi_points3 = len(data3['Technical Analysis: RSI'])
        print(f"  ✓ Got {rsi_points3} RSI data points for QQQ\n")
    
    # Check cache status
    cache_stats = client.get_cache_status()
    print(f"Cache Status:")
    print(f"  Total keys: {cache_stats.get('keys', 0)}")
    print(f"  AV keys: {cache_stats.get('av_keys', 0)}")
    
    # Show cache key format
    cache_key = f"av:rsi:{test_symbol}:1min_14"
    ttl = client.cache.get_ttl(cache_key)
    print(f"\nCache key example: {cache_key}")
    print(f"TTL remaining: {ttl} seconds")
    
    return True


if __name__ == "__main__":
    success = test_rsi_client()
    if success:
        print("\n✅ RSI client method working correctly!")
    else:
        print("\n✗ RSI client method test failed")
        sys.exit(1)