#!/usr/bin/env python3
"""Test RSI API and document response structure - Phase 5.1"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager
import requests


def test_rsi_api():
    """Test and document RSI API response"""
    
    config = ConfigManager()
    
    # RSI parameters based on AV documentation
    params = {
        'function': 'RSI',
        'symbol': 'SPY',
        'interval': '1min',  # 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        'time_period': 14,   # Number of data points used to calculate each RSI value
        'series_type': 'close',  # close, open, high, low
        'apikey': config.av_api_key,
        'datatype': 'json'
    }
    
    print("=== Testing RSI API ===\n")
    print(f"Parameters: {params}\n")
    
    # Make the API call
    response = requests.get(config.av_config['base_url'], params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Save complete response
    output_dir = Path('data/api_responses')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f'rsi_SPY_{timestamp}.json'
    
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
    
    # Analyze technical analysis data
    if 'Technical Analysis: RSI' in data:
        rsi_data = data['Technical Analysis: RSI']
        timestamps = list(rsi_data.keys())
        
        print(f"\nRSI Data Points: {len(timestamps)}")
        print(f"Latest timestamp: {timestamps[0] if timestamps else 'None'}")
        print(f"Oldest timestamp: {timestamps[-1] if timestamps else 'None'}")
        
        # Show sample data point
        if timestamps:
            sample_time = timestamps[0]
            sample_value = rsi_data[sample_time]
            print(f"\nSample data point:")
            print(f"  Timestamp: {sample_time}")
            print(f"  RSI Value: {sample_value}")
    
    return data


if __name__ == "__main__":
    test_rsi_api()