#!/usr/bin/env python3
"""Test Alpha Vantage client with caching"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient


def test_caching():
    """Test that caching reduces API calls"""
    print("=== Testing Alpha Vantage Client with Cache ===\n")
    
    client = AlphaVantageClient()
    
    # Get initial stats
    initial_stats = client.get_rate_limit_status()
    initial_calls = initial_stats['calls_made']
    
    print(f"Initial API calls made: {initial_calls}\n")
    
    # Test 1: First call should hit API
    print("Test 1: First call (should hit API)")
    start = time.time()
    data1 = client.get_realtime_options('SPY')
    time1 = time.time() - start
    print(f"  Time taken: {time1:.2f} seconds")
    print(f"  Got {len(data1.get('data', []))} contracts\n")
    
    # Test 2: Second call should hit cache (fast)
    print("Test 2: Second call (should hit cache)")
    start = time.time()
    data2 = client.get_realtime_options('SPY')
    time2 = time.time() - start
    print(f"  Time taken: {time2:.2f} seconds")
    print(f"  Got {len(data2.get('data', []))} contracts\n")
    
    # Verify data is the same
    assert len(data1.get('data', [])) == len(data2.get('data', [])), "Data mismatch!"
    
    # Check that only 1 API call was made
    final_stats = client.get_rate_limit_status()
    final_calls = final_stats['calls_made']
    api_calls_made = final_calls - initial_calls
    
    print(f"API calls made: {api_calls_made} (should be 1)")
    print(f"Cache saved {time1 - time2:.2f} seconds")
    
    # Test 3: Different symbol should hit API
    print("\nTest 3: Different symbol (should hit API)")
    data3 = client.get_realtime_options('QQQ')
    print(f"  Got {len(data3.get('data', []))} contracts for QQQ")
    
    # Check cache status
    cache_stats = client.get_cache_status()
    print(f"\nCache Status:")
    print(f"  Total keys: {cache_stats.get('keys', 0)}")
    print(f"  AV-specific keys: {cache_stats.get('av_keys', 0)}")
    print(f"  Memory used: {cache_stats.get('used_memory', 'Unknown')}")
    
    # Test 4: Wait for cache to expire
    print("\nTest 4: Testing cache expiration (waiting 31 seconds)...")
    print("  Waiting for 30-second TTL to expire...")
    time.sleep(31)
    
    print("  Calling SPY again (should hit API after expiration)")
    start = time.time()
    data4 = client.get_realtime_options('SPY')
    time4 = time.time() - start
    print(f"  Time taken: {time4:.2f} seconds (cache expired, hit API)")
    
    # Final API call count
    final_stats2 = client.get_rate_limit_status()
    total_api_calls = final_stats2['calls_made'] - initial_calls
    
    print(f"\n✅ Caching test complete!")
    print(f"  Total API calls made: {total_api_calls}")
    print(f"  Expected: 3 (SPY first, QQQ, SPY after expiry)")
    
    assert total_api_calls == 3, f"Expected 3 API calls, got {total_api_calls}"
    
    return True


if __name__ == "__main__":
    success = test_caching()
    sys.exit(0 if success else 1)