#!/usr/bin/env python3
"""Test ATR API and document response structure - Phase 5.5"""

import sys
import json
from pathlib import Path
from datetime import datetime
import requests

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager


def test_atr_api():
    """Test Alpha Vantage ATR API and save response"""
    print("=== Testing ATR (Average True Range) API ===\n")
    
    # Load configuration
    config = ConfigManager()
    
    # ATR parameters - testing with daily interval (different from others!)
    params = {
        'function': 'ATR',
        'symbol': 'SPY',
        'interval': 'daily',  # Note: ATR typically uses daily
        'time_period': 14,    # Standard ATR period
        'apikey': config.av_api_key,
        'datatype': 'json'
    }
    
    print("1. API Parameters:")
    for key, value in params.items():
        if key != 'apikey':
            print(f"   {key}: {value}")
    print()
    
    # Make API call
    print("2. Making API call to Alpha Vantage...")
    base_url = config.av_config.get('base_url', 'https://www.alphavantage.co/query')
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API error
        if 'Error Message' in data:
            print(f"   ❌ API Error: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            print(f"   ⚠️ API Note: {data['Note']}")
            return None
            
        print("   ✓ API call successful\n")
        
        # Analyze response structure
        print("3. Response Structure:")
        print(f"   Top-level keys: {list(data.keys())}")
        
        # Check metadata
        if 'Meta Data' in data:
            print("\n   Metadata:")
            for key, value in data['Meta Data'].items():
                print(f"     {key}: {value}")
        
        # Check ATR data
        atr_key = 'Technical Analysis: ATR'
        if atr_key in data:
            atr_data = data[atr_key]
            timestamps = list(atr_data.keys())
            
            print(f"\n   ATR Data Points: {len(timestamps)}")
            print(f"   Date Range: {timestamps[-1]} to {timestamps[0]}")
            
            # Show sample data point structure
            if timestamps:
                sample_date = timestamps[0]
                sample_data = atr_data[sample_date]
                print(f"\n   Sample data point ({sample_date}):")
                print(f"     Structure: {sample_data}")
                print(f"     Data type: {type(sample_data)}")
                
                # Check if it's a dict or direct value
                if isinstance(sample_data, dict):
                    for key, value in sample_data.items():
                        print(f"     {key}: {value} (type: {type(value).__name__})")
                else:
                    print(f"     Direct value: {sample_data}")
        
        # Save response for reference
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent.parent / 'data' / 'api_responses'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'atr_SPY_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n4. Response saved to: {output_file}")
        
        # Key observations for ATR
        print("\n5. Key Observations:")
        print("   - ATR is a volatility indicator")
        print("   - Measures average true range over time_period")
        print("   - Daily interval is standard (not minute)")
        print("   - Single value per timestamp (not multiple like BBANDS)")
        print("   - Used for position sizing and stop losses")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return None


def analyze_atr_characteristics(data):
    """Analyze ATR-specific characteristics"""
    if not data or 'Technical Analysis: ATR' not in data:
        return
    
    print("\n6. ATR-Specific Analysis:")
    
    atr_data = data['Technical Analysis: ATR']
    
    # Get all ATR values
    atr_values = []
    for timestamp, value_dict in atr_data.items():
        if isinstance(value_dict, dict) and 'ATR' in value_dict:
            try:
                atr_values.append(float(value_dict['ATR']))
            except:
                pass
    
    if atr_values:
        print(f"   Min ATR: {min(atr_values):.4f}")
        print(f"   Max ATR: {max(atr_values):.4f}")
        print(f"   Avg ATR: {sum(atr_values)/len(atr_values):.4f}")
        print(f"   Latest ATR: {atr_values[0]:.4f}")
        print("\n   Note: ATR measures volatility in price units")
        print("   Higher ATR = More volatile")
        print("   Lower ATR = Less volatile")


if __name__ == "__main__":
    print("Phase 5.5 - Day 22: ATR Implementation")
    print("=" * 50 + "\n")
    
    # Test the API
    api_data = test_atr_api()
    
    if api_data:
        # Analyze ATR characteristics
        analyze_atr_characteristics(api_data)
        
        print("\n✅ ATR API Discovery Complete!")
        print("\nNext: Update configuration files with ATR settings")
    else:
        print("\n❌ ATR API test failed")
        print("Check your API key and rate limits")
    
    sys.exit(0 if api_data else 1)