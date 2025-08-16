#!/usr/bin/env python3
"""Examine HISTORICAL_OPTIONS response structure"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient


def examine_historical_options():
    client = AlphaVantageClient()
    
    # Get historical options for SPY
    print("Fetching HISTORICAL_OPTIONS for SPY...")
    data = client.get_historical_options('SPY')
    
    # Save response
    output_dir = Path('data/api_responses')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f'historical_options_SPY_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Response saved to: {filename}")
    
    # Analyze structure
    print("\n=== Response Structure ===")
    print(f"Top-level keys: {list(data.keys())}")
    
    if 'data' in data:
        print(f"Number of records: {len(data['data'])}")
        if data['data']:
            # Show first record
            first = data['data'][0]
            print(f"\nFirst record keys: {list(first.keys())}")
            print(f"\nSample data:")
            for key, value in first.items():
                print(f"  {key}: {value}")
    
    return data


if __name__ == "__main__":
    examine_historical_options()