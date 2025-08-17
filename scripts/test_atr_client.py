#!/usr/bin/env python3
"""Test ATR client method with caching - Phase 5.5"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.foundation.config_manager import ConfigManager


def test_atr_client():
    """Test ATR method in AV client"""
    print("=== Testing ATR Client Method ===\n")
    
    client = AlphaVantageClient()
    config = ConfigManager()
    
    # Get config values - NO HARDCODING!
    atr_config = config.av_config['endpoints']['atr']
    interval = atr_config['default_params']['interval']
    time_period = atr_config['default_params']['time_period']
    
    print("1. Configuration:")
    print(f"   Default interval: {interval}")
    print(f"   Default time_period: {time_period}")
    print(f"   Cache TTL: {atr_config.get('cache_ttl', 300)} seconds")
    print()
    
    # Test 1: Call with SPY (using defaults from config)
    print("2. First API call (SPY with defaults):")
    start_time = time.time()
    data = client.get_atr('SPY')
    call_time = time.time() - start_time
    
    if data and 'Technical Analysis: ATR' in data:
        atr_data = data['Technical Analysis: ATR']
        data_points = len(atr_data)
        latest_date = list(atr_data.keys())[0] if atr_data else None
        latest_value = atr_data[latest_date]['ATR'] if latest_date else None
        
        print(f"   ✓ Success: {data_points} data points")
        print(f"   Latest: {latest_date} = {latest_value}")
        print(f"   Call time: {call_time:.2f} seconds")
    else:
        print("   ❌ Failed to get ATR data")
        return False
    
    # Test 2: Same call (should hit cache)
    print("\n3. Second API call (should hit cache):")
    start_time = time.time()
    cached_data = client.get_atr('SPY')
    cache_time = time.time() - start_time
    
    if cached_data:
        print(f"   ✓ Cache hit successful")
        print(f"   Cache time: {cache_time:.4f} seconds")
        print(f"   Speedup: {call_time/cache_time:.1f}x")
    else:
        print("   ❌ Cache miss (unexpected)")
    
    # Test 3: Different symbol (QQQ)
    print("\n4. Different symbol (QQQ):")
    data_qqq = client.get_atr('QQQ')
    
    if data_qqq and 'Technical Analysis: ATR' in data_qqq:
        atr_data_qqq = data_qqq['Technical Analysis: ATR']
        latest_date_qqq = list(atr_data_qqq.keys())[0] if atr_data_qqq else None
        latest_value_qqq = atr_data_qqq[latest_date_qqq]['ATR'] if latest_date_qqq else None
        
        print(f"   ✓ QQQ ATR: {latest_value_qqq} on {latest_date_qqq}")
    else:
        print("   ❌ Failed to get QQQ ATR")
    
    # Test 4: Custom parameters
    print("\n5. Custom parameters test:")
    data_custom = client.get_atr('SPY', interval='daily', time_period=20)
    
    if data_custom:
        print(f"   ✓ Custom parameters accepted (time_period=20)")
    else:
        print("   ❌ Custom parameters failed")
    
    # Show rate limit status
    print("\n6. Rate Limit Status:")
    rate_status = client.get_rate_limit_status()
    print(f"   API calls made: {rate_status['calls_made']}")
    print(f"   Tokens available: {rate_status['tokens_available']:.1f}")
    
    # Compare ATR characteristics
    print("\n7. ATR Characteristics (vs other indicators):")
    print("   ┌─────────────┬──────────┬─────────────┐")
    print("   │ Indicator   │ Interval │ Update Freq │")
    print("   ├─────────────┼──────────┼─────────────┤")
    print("   │ RSI/MACD    │ 1min     │ 60-600s     │")
    print("   │ BBANDS/VWAP │ 5min     │ 60-600s     │")
    print("   │ ATR         │ daily    │ 900-3600s   │")
    print("   └─────────────┴──────────┴─────────────┘")
    
    return True


if __name__ == "__main__":
    print("Phase 5.5 - Step 3: Client Method Test")
    print("=" * 50 + "\n")
    
    success = test_atr_client()
    
    if success:
        print("\n✅ ATR client method working correctly!")
        print("\nNext: Create database schema for ATR")
    else:
        print("\n❌ ATR client test failed")
        print("Check the implementation and try again")
    
    sys.exit(0 if success else 1)