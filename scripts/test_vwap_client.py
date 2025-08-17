#!/usr/bin/env python3
"""Test VWAP client method - Phase 5.4"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.foundation.config_manager import ConfigManager


def test_vwap_client():
    """Test VWAP client method"""
    print("=== Testing VWAP Client Method ===\n")
    
    client = AlphaVantageClient()
    config = ConfigManager()
    
    # Check config
    vwap_config = config.av_config['endpoints']['vwap']
    print("1. Configuration:")
    print(f"   Default interval: {vwap_config['default_params']['interval']}")
    print(f"   Cache TTL: {vwap_config.get('cache_ttl')} seconds\n")
    
    # Test 1: Default interval
    print("2. Testing with default interval...")
    start = time.time()
    data = client.get_vwap('SPY')
    elapsed = time.time() - start
    
    if data and 'Technical Analysis: VWAP' in data:
        points = len(data['Technical Analysis: VWAP'])
        print(f"   ✓ Got {points} data points in {elapsed:.2f}s")
    else:
        print(f"   ✗ Failed to get data")
        return False
    
    # Test 2: Cache hit
    print("\n3. Testing cache...")
    start = time.time()
    cached = client.get_vwap('SPY')
    elapsed = time.time() - start
    
    if elapsed < 0.1:
        print(f"   ✓ Cache hit ({elapsed*1000:.1f}ms)")
    else:
        print(f"   ✗ Cache miss ({elapsed:.2f}s)")
    
    # Test 3: Different interval
    print("\n4. Testing 1min interval...")
    data_1min = client.get_vwap('SPY', '1min')
    if data_1min and 'Technical Analysis: VWAP' in data_1min:
        points = len(data_1min['Technical Analysis: VWAP'])
        interval = data_1min['Meta Data'].get('4: Interval')
        print(f"   ✓ Got {points} points for {interval}")
    
    print("\n✅ VWAP Client Test Complete!")
    return True


if __name__ == "__main__":
    success = test_vwap_client()
    sys.exit(0 if success else 1)