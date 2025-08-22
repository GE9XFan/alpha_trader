#!/usr/bin/env python3
"""
Show ALL FIELDS for each IBKR data type - NOTHING HIDDEN
Based on Alpha Vantage analysis template
"""

import json
from pathlib import Path
from collections import defaultdict

response_dir = Path('data/api_responses/ibkr')
files = sorted(response_dir.glob('*.json'))

# Group by data type
data_types = defaultdict(list)
for file in files:
    data_type = file.stem.rsplit('_', 2)[0]
    data_types[data_type].append(file)

print("COMPLETE FIELD ANALYSIS FOR EACH IBKR DATA TYPE\n")
print("="*80)

for data_type in sorted(data_types.keys()):
    latest_file = data_types[data_type][-1]
    
    print(f"\n{data_type.upper()}")
    print("-"*40)
    print(f"File: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Show EVERYTHING based on data structure
    
    if data_type == 'accounts':
        print(f"\nData location: root level")
        print(f"ALL FIELDS:")
        for field, value in data.items():
            if isinstance(value, list):
                print(f"  {field}: list with {len(value)} items")
                for item in value:
                    print(f"    {item}")
            else:
                print(f"  {field} = {value}")
    
    elif data_type == 'summary':
        if isinstance(data, list) and data:
            print(f"\nData location: list of account summary items")
            print(f"Number of items: {len(data)}")
            
            print(f"\nALL FIELDS in first summary item:")
            sample = data[0]
            for field, value in sample.items():
                print(f"  {field} = {value}")
        else:
            print(f"\nData structure: {type(data)}")
            print(f"Content: {data}")
    
    elif data_type == 'positions':
        if isinstance(data, list) and data:
            print(f"\nData location: list of positions")
            print(f"Number of positions: {len(data)}")
            
            print(f"\nALL FIELDS in first position:")
            sample = data[0]
            for field, value in sample.items():
                if isinstance(value, dict):
                    print(f"  {field}: dict with keys: {list(value.keys())}")
                    for k, v in value.items():
                        print(f"    {k} = {v}")
                else:
                    print(f"  {field} = {value}")
        else:
            print(f"\nNo positions or different structure")
            print(f"Data: {data}")
    
    elif data_type == 'contract_details':
        print(f"\nData location: contract details object")
        print(f"ALL FIELDS:")
        
        def show_nested_fields(obj, indent="  "):
            for field, value in obj.items():
                if isinstance(value, dict):
                    print(f"{indent}{field}: dict with keys: {list(value.keys())}")
                    show_nested_fields(value, indent + "  ")
                elif isinstance(value, list):
                    print(f"{indent}{field}: list with {len(value)} items")
                    if value and isinstance(value[0], dict):
                        print(f"{indent}  First item fields: {list(value[0].keys())}")
                else:
                    print(f"{indent}{field} = {value}")
        
        show_nested_fields(data)
    
    elif data_type.startswith('historical'):
        if isinstance(data, list) and data:
            print(f"\nData location: list of historical bars")
            print(f"Number of bars: {len(data)}")
            
            print(f"\nALL FIELDS in first bar:")
            sample = data[0]
            for field, value in sample.items():
                print(f"  {field} = {value}")
        else:
            print(f"\nNo bars or different structure")
            print(f"Data: {data}")
    
    elif data_type == 'ticker':
        print(f"\nData location: ticker object")
        print(f"ALL FIELDS:")
        for field, value in data.items():
            if isinstance(value, dict):
                print(f"  {field}: dict with keys: {list(value.keys())}")
                for k, v in value.items():
                    print(f"    {k} = {v}")
            elif isinstance(value, list):
                print(f"  {field}: list with {len(value)} items")
            else:
                print(f"  {field} = {value}")
    
    elif data_type == 'realtime_bars' or data_type.startswith('realtime'):
        if isinstance(data, list) and data:
            print(f"\nData location: list of real-time bars")
            print(f"Number of bars: {len(data)}")
            
            print(f"\nALL FIELDS in first bar:")
            sample = data[0]
            for field, value in sample.items():
                print(f"  {field} = {value}")
        else:
            print(f"\nNo real-time bars or different structure")
            print(f"Data: {data}")
    
    else:
        # Generic handler for any other data types
        print(f"\nData at root level:")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"ALL FIELDS:")
            for field, value in data.items():
                if isinstance(value, dict):
                    print(f"  {field}: dict with {len(value)} keys")
                    if len(value) < 10:  # Show small dicts completely
                        for k, v in value.items():
                            print(f"    {k} = {v}")
                elif isinstance(value, list):
                    print(f"  {field}: list with {len(value)} items")
                    if value and isinstance(value[0], dict):
                        print(f"    First item ALL FIELDS:")
                        for k, v in value[0].items():
                            print(f"      {k} = {v}")
                else:
                    print(f"  {field} = {value}")
        elif isinstance(data, list):
            print(f"List with {len(data)} items")
            if data and isinstance(data[0], dict):
                print(f"First item ALL FIELDS:")
                for k, v in data[0].items():
                    print(f"  {k} = {v}")
        else:
            print(f"Content: {data}")

print("\n" + "="*80)
print("COMPLETE IBKR FIELD LISTING DONE")