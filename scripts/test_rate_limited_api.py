#!/usr/bin/env python3
"""Test rate-limited API calls"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient


def test_rapid_calls():
    """Test multiple rapid API calls with rate limiting"""
    print("=== Testing Rate-Limited API Calls ===\n")
    
    client = AlphaVantageClient()
    
    # Test making several calls rapidly
    symbols = ['SPY', 'QQQ', 'IWM']
    
    print("Making rapid API calls...\n")
    start_time = time.time()
    
    for i, symbol in enumerate(symbols * 2):  # 6 calls total
        try:
            call_start = time.time()
            
            # Alternate between realtime and historical
            if i % 2 == 0:
                data = client.get_realtime_options(symbol)
                api_type = "REALTIME"
            else:
                # For historical, we won't specify a date (gets latest)
                data = client.get_historical_options(symbol)
                api_type = "HISTORICAL"
            
            call_time = time.time() - call_start
            
            if data and 'data' in data:
                print(f"  Call {i+1}: {api_type} {symbol} - "
                      f"{len(data['data'])} contracts in {call_time:.2f}s")
            else:
                print(f"  Call {i+1}: {api_type} {symbol} - "
                      f"Response received in {call_time:.2f}s")
            
        except Exception as e:
            print(f"  Call {i+1} failed: {e}")
    
    total_time = time.time() - start_time
    
    # Show final statistics
    stats = client.get_rate_limit_status()
    
    print(f"\n=== Results ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per call: {total_time/6:.2f} seconds")
    print(f"\nRate Limiter Stats:")
    print(f"  - Calls made: {stats['calls_made']}")
    print(f"  - Calls blocked: {stats['calls_blocked']}")
    print(f"  - Tokens available: {stats['tokens_available']:.1f}")
    print(f"  - Success rate: {stats['success_rate']:.1f}%")


if __name__ == "__main__":
    test_rapid_calls()