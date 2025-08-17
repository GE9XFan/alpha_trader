#!/usr/bin/env python3
"""Test VWAP API and document response structure - Phase 5.4"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager
import requests


def test_vwap_api():
    """Test and document VWAP API response"""
    
    config = ConfigManager()
    
    # VWAP parameters based on AV documentation
    params = {
        'function': 'VWAP',
        'symbol': 'SPY',
        'interval': '5min',  # 1min, 5min, 15min, 30min, 60min
        'apikey': config.av_api_key,
        'datatype': 'json'
    }
    
    print("=== Testing VWAP API ===\n")
    print(f"Parameters: {params}\n")
    
    # Make the API call
    response = requests.get(config.av_config['base_url'], params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Save complete response
    output_dir = Path('data/api_responses')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f'vwap_SPY_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Response saved to: {filename}\n")
    
    # Analyze structure
    print("=== Response Structure ===")
    print(f"Top-level keys: {list(data.keys())}\n")
    
    # Show metadata
    if 'Meta Data' in data:
        print("Metadata:")
        for key, value in data['Meta Data'].items():
            print(f"  {key}: {value}")
    
    # Analyze VWAP data
    if 'Technical Analysis: VWAP' in data:
        vwap_data = data['Technical Analysis: VWAP']
        timestamps = list(vwap_data.keys())
        
        print(f"\nVWAP Data Points: {len(timestamps)}")
        print(f"Latest timestamp: {timestamps[0] if timestamps else 'None'}")
        print(f"Oldest timestamp: {timestamps[-1] if timestamps else 'None'}")
        
        # Show sample data point
        if timestamps:
            sample_time = timestamps[0]
            sample_values = vwap_data[sample_time]
            print(f"\nSample data point:")
            print(f"  Timestamp: {sample_time}")
            print(f"  VWAP Value: {sample_values}")
    
    return data


if __name__ == "__main__":
    test_vwap_api()