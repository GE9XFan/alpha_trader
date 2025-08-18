#!/usr/bin/env python3
"""Test ADX client method - Phase 5.6"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient

def test_adx_client():
    client = AlphaVantageClient()
    
    # Test with SPY
    print("Testing get_adx() method...")
    data = client.get_adx('SPY')
    
    if data and 'Technical Analysis: ADX' in data:
        adx_data = data['Technical Analysis: ADX']
        print(f"✅ Success! Got {len(adx_data)} ADX data points")
        
        # Show sample
        first_timestamp = list(adx_data.keys())[0]
        print(f"Sample: {first_timestamp} -> {adx_data[first_timestamp]}")
        return True
    else:
        print("❌ Failed to get ADX data")
        return False

if __name__ == "__main__":
    success = test_adx_client()
    sys.exit(0 if success else 1)