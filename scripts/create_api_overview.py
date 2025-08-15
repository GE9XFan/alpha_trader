#!/usr/bin/env python3
"""
Create consolidated overview with limited data from each API
"""

import json
from pathlib import Path

def truncate_data(data, max_items=10):
    """Truncate data structure to manageable size"""
    
    # Special handling for CSV responses
    if isinstance(data, dict) and 'csv_response' in data:
        csv_text = data['csv_response']
        csv_lines = csv_text.split('\r\n')
        truncated_csv = '\r\n'.join(csv_lines[:20])
        return {
            'csv_response': truncated_csv,
            'total_rows': len(csv_lines),
            'showing': 'first 20 rows'
        }
    
    # Handle time series or nested data
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "Meta Data" or key == "meta_data":
                # Keep metadata as-is
                result[key] = value
            elif isinstance(value, dict) and len(value) > max_items:
                # Truncate nested dicts (like time series data)
                items = dict(list(value.items())[:max_items])
                result[key] = items
                result[key + "_note"] = f"Showing {max_items} of {len(value)} items"
            elif isinstance(value, list) and len(value) > max_items:
                # Truncate lists
                result[key] = value[:max_items]
                result[key + "_note"] = f"Showing {max_items} of {len(value)} items"
            else:
                result[key] = value
        return result
    
    return data

def main():
    response_dir = Path("data/api_responses")
    output = {}
    
    # Read each JSON file
    for json_file in sorted(response_dir.glob("*.json")):
        # Extract clean API name
        filename = json_file.stem
        parts = filename.split('_')
        
        # Get API name (everything before symbol/timestamp)
        api_name = []
        for part in parts:
            if part in ['AAPL', 'SPY', 'MSFT'] or part.isdigit():
                break
            api_name.append(part)
        api_name = '_'.join(api_name)
        
        print(f"Processing {api_name}...")
        
        # Load the JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Truncate data to manageable size
        truncated = truncate_data(data, max_items=10)
        
        # Store truncated version
        output[api_name] = truncated
    
    # Save consolidated overview
    with open("data/api_responses_overview.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    # Check file size
    import os
    size_kb = os.path.getsize("data/api_responses_overview.json") / 1024
    print(f"✅ Created overview with {len(output)} APIs")
    print(f"📁 File size: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()