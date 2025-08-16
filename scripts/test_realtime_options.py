#!/usr/bin/env python3
"""Test REALTIME_OPTIONS API and save response"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient


def test_realtime_options():
    """Test and document REALTIME_OPTIONS response"""
    
    client = AlphaVantageClient()
    
    # Test with SPY
    symbol = 'SPY'
    print(f"\n=== Testing REALTIME_OPTIONS for {symbol} ===\n")
    
    try:
        # Make API call
        response = client.get_realtime_options(symbol)
        
        # Save complete response for analysis
        output_dir = Path('data/api_responses')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'realtime_options_{symbol}_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(response, f, indent=2)
        
        print(f"\n✓ Response saved to: {filename}")
        
        # Basic analysis of response structure
        print("\n=== Response Structure ===")
        print(f"Top-level keys: {list(response.keys())}")
        
        if 'data' in response:
            print(f"Number of options contracts: {len(response['data'])}")
            if response['data']:
                # Show first contract structure
                first_contract = response['data'][0]
                print(f"\nFirst contract keys: {list(first_contract.keys())}")
                print(f"\nSample contract data:")
                for key, value in list(first_contract.items())[:5]:
                    print(f"  {key}: {value}")
        
        return response
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_realtime_options()