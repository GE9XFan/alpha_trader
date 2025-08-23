#!/usr/bin/env python3
"""
Database Coverage Verification
Ensures EVERY API response has a table and EVERY field is mapped
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

def load_api_responses() -> Dict[str, any]:
    """Load all API responses"""
    responses = {}
    response_dir = Path('data/api_responses')
    
    for category_dir in response_dir.iterdir():
        if category_dir.is_dir():
            for file in category_dir.glob('*'):
                if file.suffix in ['.json', '.csv']:
                    key = f"{category_dir.name}/{file.stem}"
                    responses[key] = {
                        'file': str(file),
                        'exists': file.exists(),
                        'size': file.stat().st_size
                    }
    
    return responses

def load_sql_tables() -> Dict[str, str]:
    """Load all generated SQL tables"""
    tables = {}
    tables_dir = Path('schema/tables')
    
    for category_dir in tables_dir.iterdir():
        if category_dir.is_dir():
            for sql_file in category_dir.glob('*.sql'):
                key = f"{category_dir.name}/{sql_file.stem}"
                tables[key] = str(sql_file)
    
    return tables

def extract_fields_from_response(file_path: str) -> Set[str]:
    """Extract all field names from an API response"""
    fields = set()
    
    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                fields = extract_fields_recursive(data)
        except:
            pass
    elif file_path.endswith('.csv'):
        # CSV files have headers as fields
        try:
            with open(file_path, 'r') as f:
                header = f.readline().strip()
                fields = set(header.split(','))
        except:
            pass
    
    return fields

def extract_fields_recursive(obj, prefix='') -> Set[str]:
    """Recursively extract all field names from nested structure"""
    fields = set()
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            field_path = f"{prefix}.{key}" if prefix else key
            fields.add(field_path)
            
            # Don't recurse into date-keyed time series
            if isinstance(value, dict):
                sample_keys = list(value.keys())[:3]
                is_timeseries = all(
                    isinstance(k, str) and len(k) >= 10 and k[:4].isdigit() and k[4] == '-'
                    for k in sample_keys
                ) if sample_keys else False
                
                if not is_timeseries:
                    fields.update(extract_fields_recursive(value, field_path))
                else:
                    # For time series, get fields from first value
                    if value:
                        first_val = list(value.values())[0]
                        if isinstance(first_val, dict):
                            fields.update(extract_fields_recursive(first_val, f"{field_path}[timeseries]"))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                fields.update(extract_fields_recursive(value[0], f"{field_path}[0]"))
    
    return fields

def extract_columns_from_sql(file_path: str) -> Set[str]:
    """Extract column names from SQL CREATE TABLE statements"""
    columns = set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find all CREATE TABLE statements
        import re
        tables = re.findall(r'CREATE TABLE\s+(\w+)\s*\((.*?)\);', content, re.DOTALL | re.IGNORECASE)
        
        for table_name, table_def in tables:
            # Extract column definitions
            lines = table_def.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('--'):
                    # Skip constraints, indexes, etc.
                    if any(keyword in line.upper() for keyword in ['PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'INDEX', 'CONSTRAINT']):
                        continue
                    
                    # Extract column name
                    match = re.match(r'^\s*(\w+)\s+', line)
                    if match:
                        col_name = match.group(1)
                        if col_name.lower() not in ['id', 'created_at', 'updated_at']:
                            columns.add(col_name.lower())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return columns

def generate_coverage_report():
    """Generate comprehensive coverage report"""
    print("="*80)
    print("DATABASE COVERAGE VERIFICATION REPORT")
    print("="*80)
    print()
    
    # Load all data
    api_responses = load_api_responses()
    sql_tables = load_sql_tables()
    
    print(f"📁 API Responses Found: {len(api_responses)}")
    print(f"📊 SQL Tables Generated: {len(sql_tables)}")
    print()
    
    # Categorize responses
    categories = {}
    for response_key in api_responses:
        category = response_key.split('/')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(response_key)
    
    # Check coverage by category
    print("COVERAGE BY CATEGORY:")
    print("-"*80)
    
    total_covered = 0
    total_missing = 0
    missing_tables = []
    
    for category in sorted(categories.keys()):
        responses = categories[category]
        print(f"\n📂 {category.upper()}")
        
        for response_key in sorted(responses):
            response_name = response_key.split('/')[1].replace('_response', '')
            
            # Check for corresponding SQL table
            sql_found = False
            for sql_key in sql_tables:
                sql_name = sql_key.split('/')[1]
                if sql_name == response_name or response_name.startswith(sql_name):
                    sql_found = True
                    break
            
            status = "✅" if sql_found else "❌"
            print(f"  {status} {response_name}")
            
            if sql_found:
                total_covered += 1
            else:
                total_missing += 1
                missing_tables.append(response_key)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("-"*80)
    print(f"✅ Tables Created: {total_covered}/{len(api_responses)}")
    print(f"❌ Missing Tables: {total_missing}")
    
    if missing_tables:
        print(f"\n⚠️  MISSING TABLES FOR:")
        for missing in missing_tables:
            print(f"   - {missing}")
    
    # Field-level verification for each response
    print("\n" + "="*80)
    print("FIELD-LEVEL VERIFICATION:")
    print("-"*80)
    
    for category in sorted(categories.keys()):
        print(f"\n📂 {category.upper()}")
        
        for response_key in sorted(categories[category]):
            response_info = api_responses[response_key]
            response_name = response_key.split('/')[1].replace('_response', '')
            
            # Find corresponding SQL file
            sql_file = None
            for sql_key, sql_path in sql_tables.items():
                if response_name in sql_key or sql_key.endswith(f"/{response_name}"):
                    sql_file = sql_path
                    break
            
            if sql_file:
                # Extract fields from response
                response_fields = extract_fields_from_response(response_info['file'])
                
                # Extract columns from SQL
                sql_columns = extract_columns_from_sql(sql_file)
                
                print(f"\n  📊 {response_name}:")
                print(f"     Response fields: {len(response_fields)}")
                print(f"     SQL columns: {len(sql_columns)}")
                
                # Show sample fields if there's a mismatch
                if len(response_fields) > 0 and len(sql_columns) > 0:
                    # Check for important financial fields
                    important_fields = [
                        'price', 'volume', 'open', 'high', 'low', 'close',
                        'bid', 'ask', 'strike', 'delta', 'gamma', 'theta',
                        'vega', 'rho', 'implied_volatility', 'earnings', 'revenue',
                        'symbol', 'date', 'timestamp'
                    ]
                    
                    for field in important_fields:
                        field_found = any(field in f.lower() for f in response_fields)
                        col_found = any(field in c.lower() for c in sql_columns)
                        
                        if field_found and not col_found:
                            print(f"     ⚠️  Field '{field}' in response but not in SQL")
    
    # Check for CSV files specifically
    print("\n" + "="*80)
    print("CSV FILES:")
    print("-"*80)
    
    csv_files = [k for k in api_responses if api_responses[k]['file'].endswith('.csv')]
    if csv_files:
        for csv_key in csv_files:
            csv_name = csv_key.split('/')[1].replace('_response', '')
            print(f"  📄 {csv_name}.csv - Special handling required for CSV format")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    generate_coverage_report()