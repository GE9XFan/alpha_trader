#!/usr/bin/env python3
"""Test ADX API - Phase 5.6 - Day 23"""

import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient

def test_adx_api():
    """Test Alpha Vantage ADX API directly"""
    print("=== Testing ADX API - Phase 5.6 ===\n")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    client = AlphaVantageClient()
    
    params = {
        'function': 'ADX',
        'symbol': 'SPY',
        'interval': '5min',
        'time_period': 14,
        'apikey': client.api_key
    }
    
    print("Testing ADX with parameters:")
    for k, v in params.items():
        if k != 'apikey':
            print(f"  {k}: {v}")
    
    client.rate_limiter.acquire()
    
    import requests
    response = requests.get(client.base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        Path('data/api_responses').mkdir(parents=True, exist_ok=True)
        output_file = f'data/api_responses/adx_SPY_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Response saved to: {output_file}")
        
        if 'Technical Analysis: ADX' in data:
            adx_data = data['Technical Analysis: ADX']
            timestamps = list(adx_data.keys())
            
            print(f"\n📊 ADX Response Analysis:")
            print(f"  Data points: {len(timestamps)}")
            
            if timestamps:
                first = adx_data[timestamps[0]]
                print(f"\n  Sample data point:")
                print(f"    Timestamp: {timestamps[0]}")
                print(f"    ADX: {first.get('ADX')}")
                
                print(f"\n  Recent ADX values:")
                for ts in timestamps[:5]:
                    print(f"    {ts}: {adx_data[ts]['ADX']}")
        
        return data

if __name__ == "__main__":
    test_adx_api()