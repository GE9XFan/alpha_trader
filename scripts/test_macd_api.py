#!/usr/bin/env python3
"""Test MACD API and document response structure - Phase 5.2"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager
import requests


def test_macd_api():
    """Test and document MACD API response"""
    
    config = ConfigManager()
    
    # MACD parameters based on AV documentation
    params = {
        'function': 'MACD',
        'symbol': 'SPY',
        'interval': '1min',  # 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        'series_type': 'close',  # close, open, high, low
        'fastperiod': 12,  # Default for MACD
        'slowperiod': 26,  # Default for MACD
        'signalperiod': 9,  # Default for signal line
        'apikey': config.av_api_key,
        'datatype': 'json'
    }
    
    print("=== Testing MACD API ===\n")
    print(f"Parameters: {params}\n")
    
    # Make the API call
    response = requests.get(config.av_config['base_url'], params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Save complete response
    output_dir = Path('data/api_responses')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f'macd_SPY_{timestamp}.json'
    
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
    
    # Analyze MACD data - expecting 3 values per timestamp
    if 'Technical Analysis: MACD' in data:
        macd_data = data['Technical Analysis: MACD']
        timestamps = list(macd_data.keys())
        
        print(f"\nMACD Data Points: {len(timestamps)}")
        print(f"Latest timestamp: {timestamps[0] if timestamps else 'None'}")
        print(f"Oldest timestamp: {timestamps[-1] if timestamps else 'None'}")
        
        # Show sample data point - expecting MACD, Signal, Histogram
        if timestamps:
            sample_time = timestamps[0]
            sample_values = macd_data[sample_time]
            print(f"\nSample data point:")
            print(f"  Timestamp: {sample_time}")
            for key, value in sample_values.items():
                print(f"  {key}: {value}")
    
    return data


if __name__ == "__main__":
    test_macd_api()