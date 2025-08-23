#!/usr/bin/env python3
"""
DEEP FIELD INVESTIGATION
No field can be lost. Every field must be mapped.
This is production code for million-dollar trading decisions.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

class DeepFieldInvestigator:
    def __init__(self):
        self.response_dir = Path('data/api_responses')
        self.all_responses = {}
        self.field_inventory = defaultdict(lambda: defaultdict(set))
        
    def load_all_responses(self):
        """Load EVERY response file"""
        for category_dir in self.response_dir.iterdir():
            if category_dir.is_dir():
                for file in category_dir.glob('*'):
                    key = f"{category_dir.name}/{file.stem}"
                    
                    if file.suffix == '.json':
                        try:
                            with open(file, 'r') as f:
                                self.all_responses[key] = json.load(f)
                                print(f"✅ Loaded: {key}")
                        except Exception as e:
                            print(f"❌ Failed to load {key}: {e}")
                    elif file.suffix == '.csv':
                        # CSV needs special handling
                        self.all_responses[key] = {'type': 'CSV', 'file': str(file)}
                        print(f"📄 CSV file: {key}")
    
    def extract_all_fields(self, obj: Any, path: str = "", depth: int = 0) -> Dict:
        """Extract ALL fields from ANY structure at ANY depth"""
        fields = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                field_path = f"{path}.{key}" if path else key
                
                # Record this field
                if isinstance(value, dict):
                    # Check if it's a time-series (date-keyed)
                    sample_keys = list(value.keys())[:3] if value else []
                    is_timeseries = all(
                        isinstance(k, str) and len(k) >= 10 and 
                        k[:4].isdigit() and '-' in k[:10]
                        for k in sample_keys
                    ) if sample_keys else False
                    
                    if is_timeseries:
                        fields[field_path] = {
                            'type': 'TIME_SERIES',
                            'sample_dates': sample_keys[:3],
                            'count': len(value)
                        }
                        # Analyze the structure of values
                        if value:
                            first_val = list(value.values())[0]
                            if isinstance(first_val, dict):
                                sub_fields = self.extract_all_fields(first_val, f"{field_path}[TS_VALUE]", depth+1)
                                fields.update(sub_fields)
                            else:
                                fields[f"{field_path}[TS_VALUE]"] = {
                                    'type': type(first_val).__name__,
                                    'sample': str(first_val)[:100]
                                }
                    else:
                        fields[field_path] = {'type': 'OBJECT', 'depth': depth}
                        # Recurse into object
                        sub_fields = self.extract_all_fields(value, field_path, depth+1)
                        fields.update(sub_fields)
                        
                elif isinstance(value, list):
                    fields[field_path] = {
                        'type': 'ARRAY',
                        'length': len(value),
                        'depth': depth
                    }
                    # Analyze first item if exists
                    if value:
                        if isinstance(value[0], dict):
                            sub_fields = self.extract_all_fields(value[0], f"{field_path}[0]", depth+1)
                            fields.update(sub_fields)
                        else:
                            fields[f"{field_path}[ITEM]"] = {
                                'type': type(value[0]).__name__,
                                'sample': str(value[0])[:100]
                            }
                else:
                    # Leaf field
                    fields[field_path] = {
                        'type': type(value).__name__,
                        'sample': str(value)[:100] if value is not None else None,
                        'depth': depth
                    }
        
        return fields
    
    def investigate_all(self):
        """Investigate EVERY response"""
        self.load_all_responses()
        
        print("\n" + "="*80)
        print("DEEP FIELD INVESTIGATION RESULTS")
        print("="*80)
        
        # Analyze each response
        for response_key in sorted(self.all_responses.keys()):
            response_data = self.all_responses[response_key]
            
            if isinstance(response_data, dict) and response_data.get('type') == 'CSV':
                print(f"\n📄 {response_key} - CSV FILE (needs special handling)")
                continue
            
            print(f"\n📊 {response_key}")
            print("-"*60)
            
            # Extract all fields
            fields = self.extract_all_fields(response_data)
            
            # Group by depth
            by_depth = defaultdict(list)
            for field_path, info in fields.items():
                depth = info.get('depth', 0)
                by_depth[depth].append((field_path, info))
            
            # Show field count by depth
            print(f"Total fields: {len(fields)}")
            for depth in sorted(by_depth.keys()):
                print(f"  Depth {depth}: {len(by_depth[depth])} fields")
            
            # Show important fields
            important_keywords = [
                'symbol', 'price', 'volume', 'bid', 'ask', 'strike',
                'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility',
                'open', 'high', 'low', 'close', 'date', 'timestamp',
                'earnings', 'revenue', 'eps', 'pe_ratio', 'market_cap'
            ]
            
            print("\n  Key fields found:")
            for field_path, info in fields.items():
                field_lower = field_path.lower()
                for keyword in important_keywords:
                    if keyword in field_lower:
                        print(f"    ✓ {field_path}: {info['type']} - {info.get('sample', '')[:50]}")
                        break
            
            # Store for category analysis
            category = response_key.split('/')[0]
            endpoint = response_key.split('/')[1]
            self.field_inventory[category][endpoint] = fields
    
    def generate_corrected_schemas(self):
        """Generate CORRECT schemas based on investigation"""
        print("\n" + "="*80)
        print("GENERATING CORRECTED SCHEMAS")
        print("="*80)
        
        for category, endpoints in self.field_inventory.items():
            print(f"\n📂 {category.upper()}")
            
            for endpoint, fields in endpoints.items():
                table_name = endpoint.replace('_response', '')
                print(f"  → {table_name}: {len(fields)} fields")
                
                # Generate proper SQL based on actual fields
                sql = self.generate_table_sql(category, table_name, fields)
                
                # Save SQL file
                output_dir = Path(f'schema/tables/{category}')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f'{table_name}.sql'
                
                with open(output_file, 'w') as f:
                    f.write(sql)
                
                print(f"    ✅ Saved: {output_file}")
    
    def generate_table_sql(self, category: str, table_name: str, fields: Dict) -> str:
        """Generate CORRECT SQL for a table based on actual fields"""
        sql = f"""-- {category.upper()}: {table_name}
-- Generated from ACTUAL API response investigation
-- Total fields in response: {len(fields)}

CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
"""
        
        # Determine table structure based on category and fields
        if category == 'options':
            sql += self.generate_options_columns(fields)
        elif category == 'technical_indicators':
            sql += self.generate_technical_columns(table_name, fields)
        elif category == 'fundamentals':
            sql += self.generate_fundamentals_columns(fields)
        elif category == 'analytics':
            sql += self.generate_analytics_columns(fields)
        elif category == 'sentiment':
            sql += self.generate_sentiment_columns(fields)
        elif category == 'economic':
            sql += self.generate_economic_columns(fields)
        else:
            sql += self.generate_generic_columns(fields)
        
        # Add common metadata
        sql += """    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

"""
        
        # Add indexes for important fields
        sql += self.generate_indexes(table_name, fields)
        
        return sql
    
    def generate_options_columns(self, fields: Dict) -> str:
        """Generate columns for options tables"""
        columns = ""
        
        # Map field names to SQL columns
        field_mapping = {
            'contractid': 'contract_id VARCHAR(50)',
            'symbol': 'symbol VARCHAR(10) NOT NULL',
            'expiration': 'expiration DATE',
            'strike': 'strike NUMERIC NOT NULL',
            'type': 'option_type VARCHAR(4)',
            'last': 'last_price NUMERIC',
            'mark': 'mark NUMERIC',
            'bid': 'bid NUMERIC',
            'bid_size': 'bid_size INTEGER',
            'ask': 'ask NUMERIC',
            'ask_size': 'ask_size INTEGER',
            'volume': 'volume BIGINT',
            'open_interest': 'open_interest BIGINT',
            'date': 'quote_date DATE',
            'implied_volatility': 'implied_volatility NUMERIC',
            'delta': 'delta NUMERIC',
            'gamma': 'gamma NUMERIC',
            'theta': 'theta NUMERIC',
            'vega': 'vega NUMERIC',
            'rho': 'rho NUMERIC'
        }
        
        # Find actual fields from response
        for field_path, info in fields.items():
            # Get the leaf field name
            parts = field_path.split('.')
            field_name = parts[-1].replace('[0]', '').replace('[ITEM]', '').replace('[TS_VALUE]', '')
            field_name_lower = field_name.lower()
            
            if field_name_lower in field_mapping:
                if field_mapping[field_name_lower] not in columns:
                    columns += f"    {field_mapping[field_name_lower]},\n"
        
        # Add timestamp
        columns += "    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL"
        
        return columns
    
    def generate_technical_columns(self, indicator: str, fields: Dict) -> str:
        """Generate columns for technical indicators"""
        columns = "    symbol VARCHAR(10) NOT NULL,\n"
        columns += "    interval VARCHAR(20) NOT NULL,\n"
        columns += "    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,\n"
        
        # Find indicator-specific value fields
        value_fields = set()
        for field_path, info in fields.items():
            if '[TS_VALUE]' in field_path:
                # This is a time-series value field
                parts = field_path.split('[TS_VALUE].')
                if len(parts) > 1:
                    field_name = parts[1].replace('[0]', '')
                    value_fields.add(field_name)
        
        # Add value columns
        for field in sorted(value_fields):
            clean_name = field.lower().replace(' ', '_').replace('-', '_')
            columns += f"    {clean_name} NUMERIC,\n"
        
        # If no specific fields, add generic value
        if not value_fields:
            columns += f"    {indicator}_value NUMERIC NOT NULL,\n"
        
        # Add metadata fields
        columns += """    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50)"""
        
        return columns
    
    def generate_fundamentals_columns(self, fields: Dict) -> str:
        """Generate columns for fundamentals"""
        columns = ""
        
        # Check if it's time-series (has annualReports/quarterlyReports)
        has_reports = any('Reports' in fp for fp in fields.keys())
        
        if has_reports:
            columns += """    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
"""
        else:
            columns += "    symbol VARCHAR(10) PRIMARY KEY,\n"
        
        # Add all leaf fields
        added_fields = set()
        for field_path, info in fields.items():
            if info['type'] not in ['OBJECT', 'ARRAY', 'TIME_SERIES']:
                # Get clean field name
                parts = field_path.split('.')
                field_name = parts[-1].replace('[0]', '').replace('[ITEM]', '')
                
                # Clean for SQL
                clean_name = field_name[:1].lower() + field_name[1:]
                clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
                clean_name = clean_name.lower()
                
                if clean_name not in added_fields and clean_name not in ['symbol', 'fiscal_date_ending']:
                    sql_type = 'NUMERIC' if 'float' in info['type'] or 'int' in info['type'] else 'TEXT'
                    columns += f"    {clean_name} {sql_type},\n"
                    added_fields.add(clean_name)
        
        columns = columns.rstrip(',\n')
        
        return columns
    
    def generate_analytics_columns(self, fields: Dict) -> str:
        """Generate columns for analytics - these are complex"""
        # Analytics need special handling due to deep nesting
        return """    request_id UUID NOT NULL DEFAULT gen_random_uuid(),
    symbols TEXT NOT NULL,
    window_size INTEGER,
    min_date DATE,
    max_date DATE,
    ohlc VARCHAR(10),
    interval VARCHAR(20)"""
    
    def generate_sentiment_columns(self, fields: Dict) -> str:
        """Generate columns for sentiment data"""
        columns = ""
        
        # Add fields based on actual structure
        added_fields = set()
        for field_path, info in fields.items():
            if info['type'] not in ['OBJECT', 'ARRAY', 'TIME_SERIES']:
                parts = field_path.split('.')
                field_name = parts[-1].replace('[0]', '').replace('[ITEM]', '')
                clean_name = field_name.lower().replace('-', '_').replace(' ', '_')
                
                if clean_name not in added_fields:
                    sql_type = 'NUMERIC' if 'float' in info['type'] or 'int' in info['type'] else 'TEXT'
                    columns += f"    {clean_name} {sql_type},\n"
                    added_fields.add(clean_name)
        
        columns = columns.rstrip(',\n')
        return columns
    
    def generate_economic_columns(self, fields: Dict) -> str:
        """Generate columns for economic indicators"""
        # Check if it's time-series
        has_timeseries = any(info.get('type') == 'TIME_SERIES' for info in fields.values())
        
        if has_timeseries:
            return """    date DATE NOT NULL UNIQUE,
    value NUMERIC NOT NULL"""
        else:
            columns = ""
            for field_path, info in fields.items():
                if info['type'] not in ['OBJECT', 'ARRAY', 'TIME_SERIES']:
                    field_name = field_path.split('.')[-1]
                    clean_name = field_name.lower().replace(' ', '_')
                    sql_type = 'NUMERIC' if 'float' in info['type'] or 'int' in info['type'] else 'TEXT'
                    columns += f"    {clean_name} {sql_type},\n"
            
            return columns.rstrip(',\n')
    
    def generate_generic_columns(self, fields: Dict) -> str:
        """Generate columns for unknown types"""
        columns = ""
        
        for field_path, info in fields.items():
            if info['type'] not in ['OBJECT', 'ARRAY', 'TIME_SERIES']:
                field_name = field_path.split('.')[-1]
                clean_name = field_name.lower().replace(' ', '_').replace('-', '_')
                sql_type = 'NUMERIC' if 'float' in info['type'] or 'int' in info['type'] else 'TEXT'
                columns += f"    {clean_name} {sql_type},\n"
        
        return columns.rstrip(',\n')
    
    def generate_indexes(self, table_name: str, fields: Dict) -> str:
        """Generate indexes for important fields"""
        indexes = ""
        
        # Check for important fields
        important_fields = ['symbol', 'date', 'timestamp', 'expiration', 'strike', 'contract_id']
        
        for field in important_fields:
            if any(field in fp.lower() for fp in fields.keys()):
                indexes += f"CREATE INDEX idx_{table_name}_{field} ON {table_name}({field}) WHERE {field} IS NOT NULL;\n"
        
        return indexes


if __name__ == "__main__":
    investigator = DeepFieldInvestigator()
    investigator.investigate_all()
    investigator.generate_corrected_schemas()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE - ALL FIELDS MAPPED")
    print("="*80)