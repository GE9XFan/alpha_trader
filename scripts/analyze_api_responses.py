#!/usr/bin/env python3
"""
Show ALL FIELDS for each API - NOTHING HIDDEN
"""

import json
from pathlib import Path
from collections import defaultdict

response_dir = Path('data/api_responses/alpha_vantage')
files = sorted(response_dir.glob('*.json'))
files = [f for f in files if 'summary' not in f.stem]

# Group by API
api_groups = defaultdict(list)
for file in files:
    api_name = file.stem.rsplit('_', 2)[0]
    api_groups[api_name].append(file)

print("COMPLETE FIELD ANALYSIS FOR EACH API\n")
print("="*80)

for api_name in sorted(api_groups.keys()):
    latest_file = api_groups[api_name][-1]
    
    print(f"\n{api_name.upper()}")
    print("-"*40)
    print(f"File: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Show EVERYTHING
    
    # Technical indicators
    tech_indicator = None
    for key in data.keys():
        if key.startswith('Technical Analysis:'):
            tech_indicator = key
            break
    
    if tech_indicator:
        dates = list(data[tech_indicator].keys())
        print(f"\nData location: {tech_indicator}")
        print(f"Number of dates: {len(dates)}")
        if dates:
            # Show ALL fields from first data point
            print(f"\nALL FIELDS in {dates[0]}:")
            for field, value in data[tech_indicator][dates[0]].items():
                print(f"  {field} = {value}")
    
    elif 'data' in data and isinstance(data['data'], list) and data['data']:
        print(f"\nData location: data[]")
        print(f"Number of records: {len(data['data'])}")
        
        # Show ALL fields from first record
        print(f"\nALL FIELDS in first record:")
        sample = data['data'][0]
        for field, value in sample.items():
            if isinstance(value, dict):
                print(f"  {field}: dict with keys: {list(value.keys())}")
            elif isinstance(value, list):
                print(f"  {field}: list with {len(value)} items")
                if value and isinstance(value[0], dict):
                    print(f"    First item fields: {list(value[0].keys())}")
            else:
                print(f"  {field} = {value}")
    
    elif 'feed' in data and data['feed']:
        print(f"\nData location: feed[]")
        print(f"Number of records: {len(data['feed'])}")
        
        print(f"\nALL FIELDS in first feed item:")
        sample = data['feed'][0]
        for field, value in sample.items():
            if isinstance(value, list) and value:
                print(f"  {field}: list with {len(value)} items")
                if isinstance(value[0], dict):
                    print(f"    First item has these fields: {list(value[0].keys())}")
                    # Show the actual fields in nested objects
                    for k, v in value[0].items():
                        print(f"      {k} = {v}")
            else:
                print(f"  {field} = {str(value)[:100]}")
    
    elif 'top_gainers' in data:
        for list_name in ['top_gainers', 'top_losers', 'most_actively_traded']:
            if list_name in data and data[list_name]:
                print(f"\nData location: {list_name}[]")
                print(f"Number of records: {len(data[list_name])}")
                
                print(f"\nALL FIELDS in first {list_name} record:")
                for field, value in data[list_name][0].items():
                    print(f"  {field} = {value}")
                break
    
    elif 'annualReports' in data:
        for report_type in ['annualReports', 'quarterlyReports']:
            if report_type in data and data[report_type]:
                print(f"\nData location: {report_type}[]")
                print(f"Number of records: {len(data[report_type])}")
                
                print(f"\nALL FIELDS in first {report_type} record:")
                sample = data[report_type][0]
                for field, value in sample.items():
                    print(f"  {field} = {value}")
                break
    
    elif 'Symbol' in data and 'MarketCapitalization' in data:
        # Company overview - show ALL fields
        print(f"\nData location: root level")
        print(f"Number of fields: {len(data)}")
        
        print(f"\nALL FIELDS:")
        for field, value in data.items():
            print(f"  {field} = {value}")
    
    elif 'payload' in data:
        print(f"\nData location: payload")
        print(f"Payload contents:")
        for key, value in data['payload'].items():
            print(f"  {key}: {value}")
        
        if 'RETURNS_CALCULATIONS' in data['payload']:
            calc_data = data['payload']['RETURNS_CALCULATIONS']
            if isinstance(calc_data, dict):
                for calc_type, calc_value in calc_data.items():
                    print(f"\n  {calc_type}:")
                    if isinstance(calc_value, dict):
                        for k, v in calc_value.items():
                            print(f"    {k} = {v}")
                    else:
                        print(f"    {calc_value}")
    
    else:
        # Just show everything at root
        print(f"\nData at root level:")
        print(f"ALL FIELDS:")
        for field, value in data.items():
            if isinstance(value, dict):
                print(f"  {field}: dict with {len(value)} keys")
            elif isinstance(value, list):
                print(f"  {field}: list with {len(value)} items")
                if value and isinstance(value[0], dict):
                    print(f"    First item ALL FIELDS:")
                    for k, v in value[0].items():
                        print(f"      {k} = {v}")
            else:
                print(f"  {field} = {value}")

print("\n" + "="*80)
print("COMPLETE FIELD LISTING DONE")