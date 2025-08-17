#!/usr/bin/env python3
"""Test BBANDS API and document response structure - Phase 5.3"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager
import requests


def test_bbands_api():
    """Test and document BBANDS API response"""
    
    config = ConfigManager()
    
    # BBANDS parameters based on AV documentation
    params = {
        'function': 'BBANDS',
        'symbol': 'SPY',
        'interval': '5min',  # 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        'time_period': 60,   # Number of data points for moving average
        'series_type': 'close',  # close, open, high, low
        'nbdevup': 2,        # Standard deviations for upper band
        'nbdevdn': 2,        # Standard deviations for lower band
        'matype': 0,         # 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
        'apikey': config.av_api_key,
        'datatype': 'json'
    }
    
    print("=== Testing BBANDS API ===\n")
    print(f"Parameters: {params}\n")
    
    # Make the API call
    response = requests.get(config.av_config['base_url'], params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Save complete response
    output_dir = Path('data/api_responses')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f'bbands_SPY_{timestamp}.json'
    
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
    
    # Analyze BBANDS data - expecting 3 bands per timestamp
    if 'Technical Analysis: BBANDS' in data:
        bbands_data = data['Technical Analysis: BBANDS']
        timestamps = list(bbands_data.keys())
        
        print(f"\nBBANDS Data Points: {len(timestamps)}")
        print(f"Latest timestamp: {timestamps[0] if timestamps else 'None'}")
        print(f"Oldest timestamp: {timestamps[-1] if timestamps else 'None'}")
        
        # Show sample data point - expecting Upper, Middle, Lower bands
        if timestamps:
            sample_time = timestamps[0]
            sample_values = bbands_data[sample_time]
            print(f"\nSample data point:")
            print(f"  Timestamp: {sample_time}")
            for key, value in sample_values.items():
                print(f"  {key}: {value}")
    
    return data


if __name__ == "__main__":
    test_bbands_api()