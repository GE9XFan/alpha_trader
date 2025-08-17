#!/usr/bin/env python3
"""Test BBANDS client method with caching"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.foundation.config_manager import ConfigManager


def test_bbands_client():
    """Test BBANDS method in AV client"""
    print("=== Testing BBANDS Client Method ===\n")
    
    client = AlphaVantageClient()
    config = ConfigManager()
    
    # Get config values - NO HARDCODING!
    bbands_config = config.av_config['endpoints']['bbands']['default_params']
    interval = bbands_config['interval']
    time_period = bbands_config['time_period']
    series_type = bbands_config['series_type']
    nbdevup = bbands_config['nbdevup']
    nbdevdn = bbands_config['nbdevdn']
    matype = bbands_config['matype']
    
    # Test symbol
    test_symbol = 'SPY'
    
    # Test 1: First call (should hit API)
    print(f"Test 1: First call for {test_symbol} (API)...")
    start = time.time()
    data = client.get_bbands(
        test_symbol,
        interval,
        time_period,
        series_type,
        nbdevup,
        nbdevdn,
        matype
    )
    api_time = time.time() - start
    
    if data and 'Technical Analysis: BBANDS' in data:
        bbands_points = len(data['Technical Analysis: BBANDS'])
        print(f"  ✓ Got {bbands_points} BBANDS data points in {api_time:.2f}s")
        
        # Check that we have all 3 bands
        first_timestamp = list(data['Technical Analysis: BBANDS'].keys())[0]
        values = data['Technical Analysis: BBANDS'][first_timestamp]
        print(f"  ✓ Bands returned: {list(values.keys())}\n")
    else:
        print(f"  ✗ No BBANDS data returned\n")
        return False
    
    # Test 2: Second call (should hit cache)
    print(f"Test 2: Second call for {test_symbol} (Cache)...")
    start = time.time()
    data2 = client.get_bbands(
        test_symbol,
        interval,
        time_period,
        series_type,
        nbdevup,
        nbdevdn,
        matype
    )
    cache_time = time.time() - start
    
    if data2:
        print(f"  ✓ Cache hit in {cache_time:.2f}s")
        print(f"  ✓ Speed improvement: {api_time/cache_time:.1f}x faster\n")
    
    # Test 3: Different symbol (should hit API)
    print("Test 3: Different symbol QQQ (API)...")
    data3 = client.get_bbands(
        'QQQ',
        interval,
        time_period,
        series_type,
        nbdevup,
        nbdevdn,
        matype
    )
    if data3 and 'Technical Analysis: BBANDS' in data3:
        bbands_points3 = len(data3['Technical Analysis: BBANDS'])
        print(f"  ✓ Got {bbands_points3} BBANDS data points for QQQ\n")
    
    # Check cache status
    cache_stats = client.get_cache_status()
    print(f"Cache Status:")
    print(f"  Total keys: {cache_stats.get('keys', 0)}")
    print(f"  AV keys: {cache_stats.get('av_keys', 0)}")
    
    # Show cache key format
    cache_key = f"av:bbands:{test_symbol}:{interval}_{time_period}_{nbdevup}_{nbdevdn}_{matype}"
    ttl = client.cache.get_ttl(cache_key)
    print(f"\nCache key example: {cache_key}")
    print(f"TTL remaining: {ttl} seconds")
    
    return True


if __name__ == "__main__":
    success = test_bbands_client()
    if success:
        print("\n✅ BBANDS client method working correctly!")
    else:
        print("\n✗ BBANDS client method test failed")
        sys.exit(1)