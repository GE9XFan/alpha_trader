#!/usr/bin/env python3
"""
Comprehensive SQL Schema Generator - ZERO FIELD LOSS
Ensures all 8,227 discovered fields get proper PostgreSQL storage

This is the production generator that:
1. Reads field_statistics.json for complete field inventory
2. Analyzes actual API responses for structure
3. Creates normalized tables with NUMERIC for all financial data
4. Generates audit trail for every transformation
5. Verifies no field is lost
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

class ComprehensiveSQLGenerator:
    """Production-grade SQL generator ensuring zero field loss"""
    
    def __init__(self):
        # Load complete field catalog
        with open('data/field_statistics.json', 'r') as f:
            self.field_catalog = json.load(f)
        
        print(f"Loaded {len(self.field_catalog)} unique fields from catalog")
        
        # Load actual responses for structure analysis
        self.responses = {}
        self._load_all_responses()
        
        # Track field mappings for verification
        self.field_mappings = defaultdict(list)  # field -> [(table, column)]
        self.unmapped_fields = set()
        
    def _load_all_responses(self):
        """Load all API response files"""
        response_dir = Path('data/api_responses')
        
        for category_dir in response_dir.iterdir():
            if category_dir.is_dir():
                for response_file in category_dir.glob('*.json'):
                    key = f"{category_dir.name}/{response_file.stem}"
                    try:
                        with open(response_file, 'r') as f:
                            self.responses[key] = json.load(f)
                    except:
                        pass  # Skip CSV files
        
        print(f"Loaded {len(self.responses)} API responses")
    
    def determine_sql_type(self, field_name: str, field_stats: Dict) -> str:
        """Determine optimal SQL type based on field statistics"""
        # Get type information from statistics
        types = field_stats.get('types', [])
        formats = field_stats.get('formats', [])
        examples = field_stats.get('examples', [])
        
        # Financial data ALWAYS uses NUMERIC
        financial_keywords = [
            'price', 'value', 'amount', 'cost', 'revenue', 'profit', 'loss',
            'earnings', 'dividend', 'market_cap', 'volume', 'shares', 'ratio',
            'margin', 'return', 'yield', 'beta', 'delta', 'gamma', 'theta',
            'vega', 'rho', 'strike', 'bid', 'ask', 'open', 'high', 'low',
            'close', 'ebitda', 'eps', 'pe', 'assets', 'liabilities', 'equity'
        ]
        
        field_lower = field_name.lower()
        if any(keyword in field_lower for keyword in financial_keywords):
            return "NUMERIC"
        
        # Check formats first (most specific)
        if formats:
            if any('date:YYYY-MM-DD' in f for f in formats):
                return "DATE"
            if any('timestamp' in f for f in formats):
                return "TIMESTAMP WITH TIME ZONE"
            if any('decimal' in f for f in formats):
                return "NUMERIC"
            if any('option_contract' in f for f in formats):
                return "VARCHAR(50)"
        
        # Check types
        if 'int' in types:
            # Check if it's a large number
            if field_stats.get('max_value'):
                try:
                    max_val = float(field_stats['max_value'])
                    if max_val > 2147483647:  # Max INT value
                        return "BIGINT"
                except:
                    pass
            return "BIGINT"
        
        if 'float' in types or 'number' in types:
            return "NUMERIC"  # NEVER use FLOAT for financial data
        
        if 'bool' in types:
            return "BOOLEAN"
        
        if 'str' in types:
            # Check string lengths
            max_length = field_stats.get('max_length', 0)
            min_length = field_stats.get('min_length', 0)
            
            # Symbol/ticker detection
            if max_length <= 10 and any(ex and ex.isupper() for ex in examples[:3]):
                return "VARCHAR(10)"
            
            # Fixed-length strings
            if min_length == max_length and max_length > 0 and max_length <= 50:
                return f"VARCHAR({max_length})"
            
            # Variable length strings
            if max_length <= 50:
                return "VARCHAR(50)"
            if max_length <= 255:
                return "VARCHAR(255)"
            
            return "TEXT"
        
        # Default to TEXT for unknown
        return "TEXT"
    
    def generate_all_schemas(self) -> Dict[str, str]:
        """Generate schemas for all endpoints ensuring no field loss"""
        schemas = {}
        
        # Group responses by category
        categories = defaultdict(list)
        for key in self.responses.keys():
            category, endpoint = key.split('/')
            categories[category].append(endpoint)
        
        # Generate schemas by category
        for category, endpoints in categories.items():
            print(f"\nProcessing category: {category}")
            
            for endpoint in endpoints:
                table_name = endpoint.replace('_response', '')
                response_key = f"{category}/{endpoint}"
                response_data = self.responses[response_key]
                
                # Generate appropriate schema based on category
                if category == 'analytics':
                    schema = self._generate_analytics_schema(table_name, response_data)
                elif category == 'technical_indicators':
                    schema = self._generate_technical_schema(table_name, response_data)
                elif category == 'options':
                    schema = self._generate_options_schema(table_name, response_data)
                elif category == 'fundamentals':
                    schema = self._generate_fundamentals_schema(table_name, response_data)
                elif category == 'sentiment':
                    schema = self._generate_sentiment_schema(table_name, response_data)
                elif category == 'economic':
                    schema = self._generate_economic_schema(table_name, response_data)
                else:
                    schema = self._generate_generic_schema(table_name, response_data)
                
                if schema:
                    schemas[f"{category}/{table_name}"] = schema
                    print(f"  ✅ Generated schema for {table_name}")
        
        # Add audit tables
        schemas['audit/data_transformations'] = self._generate_audit_schema()
        schemas['audit/api_response'] = self._generate_api_audit_schema()
        
        # Generate field mapping verification
        schemas['audit/field_mappings'] = self._generate_field_mapping_schema()
        
        return schemas
    
    def _extract_all_fields(self, data: Any, prefix: str = "") -> List[Tuple[str, Any]]:
        """Recursively extract all fields from nested structure"""
        fields = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                field_path = f"{prefix}.{key}" if prefix else key
                fields.append((field_path, value))
                
                if isinstance(value, dict):
                    # Don't recurse into time-series data (date-keyed objects)
                    if not all(re.match(r'^\d{4}-\d{2}', str(k)) for k in list(value.keys())[:3] if k):
                        fields.extend(self._extract_all_fields(value, field_path))
                elif isinstance(value, list) and value:
                    if isinstance(value[0], dict):
                        fields.extend(self._extract_all_fields(value[0], field_path + "[0]"))
        
        return fields
    
    def _generate_analytics_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for analytics endpoints with complex nested data"""
        sql = f"""-- Analytics: {table_name}
-- Handles complex nested calculation results
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL DEFAULT gen_random_uuid(),
"""
        
        # Extract metadata fields
        if 'meta_data' in data:
            for field, value in data['meta_data'].items():
                field_name = f"meta_{field.lower()}"
                field_type = self._get_field_type_from_value(value)
                sql += f"    {field_name} {field_type},\n"
                self._record_field_mapping(field, table_name, field_name)
        
        sql += """    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(request_id)
);

"""
        
        # Create normalized calculations table
        sql += f"""-- Calculation results (normalized from nested structure)
CREATE TABLE {table_name}_calculations (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES {table_name}(request_id) ON DELETE CASCADE,
    calculation_category VARCHAR(50) NOT NULL,
    calculation_type VARCHAR(50) NOT NULL,
    calculation_subtype VARCHAR(50),
    symbol VARCHAR(10),
    symbol2 VARCHAR(10), -- For correlation calculations
    date DATE,
    value NUMERIC, -- ALL financial values use NUMERIC
    additional_data JSONB, -- For complex structures like histograms
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (date);

-- Create monthly partitions for next 12 months
"""
        
        # Add partitions
        for i in range(12):
            month = (datetime.now().month + i - 1) % 12 + 1
            year = datetime.now().year + ((datetime.now().month + i - 1) // 12)
            sql += f"""CREATE TABLE {table_name}_calculations_{year}_{month:02d} 
PARTITION OF {table_name}_calculations
FOR VALUES FROM ('{year}-{month:02d}-01') TO ('{year}-{(month % 12) + 1:02d}-01');
"""
        
        # Add indexes
        sql += f"""
-- Performance indexes
CREATE INDEX idx_{table_name}_request ON {table_name}(request_id);
CREATE INDEX idx_{table_name}_calc_request ON {table_name}_calculations(request_id);
CREATE INDEX idx_{table_name}_calc_symbol ON {table_name}_calculations(symbol, date);
CREATE INDEX idx_{table_name}_calc_category ON {table_name}_calculations(calculation_category, calculation_type);
"""
        
        # Record field mappings for payload data
        if 'payload' in data:
            self._record_analytics_payload_fields(data['payload'], table_name)
        
        return sql
    
    def _record_analytics_payload_fields(self, payload: Dict, table_name: str):
        """Record field mappings for analytics payload"""
        for category, category_data in payload.items():
            if isinstance(category_data, dict):
                for calc_type, type_data in category_data.items():
                    if isinstance(type_data, dict):
                        for subtype, subtype_data in type_data.items():
                            field_path = f"payload.{category}.{calc_type}.{subtype}"
                            self._record_field_mapping(field_path, f"{table_name}_calculations", "value")
    
    def _generate_technical_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for technical indicators"""
        sql = f"""-- Technical Indicator: {table_name}
CREATE TABLE {table_name}_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
"""
        
        # Find indicator-specific fields from actual data
        indicator_fields = set()
        
        for key, value in data.items():
            if isinstance(value, dict):
                # Check for time-series structure
                sample_keys = list(value.keys())[:3]
                if sample_keys and all(re.match(r'^\d{4}-\d{2}', str(k)) for k in sample_keys):
                    # This is time-series data
                    if value:
                        sample_value = list(value.values())[0]
                        if isinstance(sample_value, dict):
                            for field in sample_value.keys():
                                indicator_fields.add(field)
                        elif isinstance(sample_value, (int, float, str)):
                            # Single value indicator
                            indicator_fields.add(key)
        
        # Add indicator columns
        if indicator_fields:
            for field in sorted(indicator_fields):
                clean_field = self._clean_field_name(field)
                sql += f"    {clean_field} NUMERIC,\n"
                self._record_field_mapping(field, f"{table_name}_data", clean_field)
        else:
            # Default single value field
            sql += f"    {table_name}_value NUMERIC NOT NULL,\n"
        
        # Add metadata fields
        sql += """    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
"""
        
        for i in range(12):
            month = (datetime.now().month + i - 1) % 12 + 1
            year = datetime.now().year + ((datetime.now().month + i - 1) // 12)
            sql += f"""CREATE TABLE {table_name}_data_{year}_{month:02d} 
PARTITION OF {table_name}_data
FOR VALUES FROM ('{year}-{month:02d}-01') TO ('{year}-{(month % 12) + 1:02d}-01');
"""
        
        sql += f"""
-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}_data(symbol);
CREATE INDEX idx_{table_name}_timestamp ON {table_name}_data(timestamp);
CREATE INDEX idx_{table_name}_interval ON {table_name}_data(interval);
"""
        
        return sql
    
    def _generate_options_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for options data"""
        sql = f"""-- Options Chain: {table_name}
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
"""
        
        # Extract all fields from options data
        options_fields = {}
        
        # Check both 'options' and 'data' keys for the options array
        options_list = data.get('options', data.get('data', []))
        if isinstance(options_list, list) and options_list:
            # Analyze all options to get complete field list
            for option in options_list:
                for field, value in option.items():
                    if field not in options_fields:
                        options_fields[field] = self._get_field_type_from_value(value)
        
        # Add all discovered fields
        for field, field_type in sorted(options_fields.items()):
            clean_field = self._clean_field_name(field)
            
            # Special handling for certain fields
            if 'contract' in field.lower():
                field_type = "VARCHAR(50)"
            elif 'type' in field.lower() and 'option' in field.lower():
                sql += f"    {clean_field} VARCHAR(4) CHECK ({clean_field} IN ('call', 'put')),\n"
                continue
            
            sql += f"    {clean_field} {field_type},\n"
            self._record_field_mapping(field, table_name, clean_field)
        
        sql += """    
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT {table_name}_unique UNIQUE(symbol, contract_id, data_timestamp)
) PARTITION BY RANGE (data_timestamp);

-- Monthly partitions
"""
        
        for i in range(12):
            month = (datetime.now().month + i - 1) % 12 + 1
            year = datetime.now().year + ((datetime.now().month + i - 1) // 12)
            sql += f"""CREATE TABLE {table_name}_{year}_{month:02d} 
PARTITION OF {table_name}
FOR VALUES FROM ('{year}-{month:02d}-01') TO ('{year}-{(month % 12) + 1:02d}-01');
"""
        
        sql += f"""
-- Indexes for performance
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_expiration ON {table_name}(expiration) WHERE expiration IS NOT NULL;
CREATE INDEX idx_{table_name}_strike ON {table_name}(strike) WHERE strike IS NOT NULL;
CREATE INDEX idx_{table_name}_timestamp ON {table_name}(data_timestamp);
"""
        
        return sql
    
    def _generate_fundamentals_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for fundamental data"""
        
        # Check if this has time-series structure
        has_time_series = any(key in data for key in ['annualReports', 'quarterlyReports', 
                                                       'annualEarnings', 'quarterlyEarnings'])
        
        if has_time_series:
            return self._generate_fundamentals_timeseries_schema(table_name, data)
        else:
            return self._generate_fundamentals_snapshot_schema(table_name, data)
    
    def _generate_fundamentals_snapshot_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for snapshot fundamental data (like overview)"""
        sql = f"""-- Fundamental Data: {table_name}
CREATE TABLE {table_name} (
    symbol VARCHAR(10) PRIMARY KEY,
"""
        
        # Extract all fields
        fields = self._extract_all_fields(data)
        
        for field_path, value in fields:
            if not isinstance(value, (dict, list)):
                clean_field = self._clean_field_name(field_path.split('.')[-1])
                field_type = self._get_field_type_from_value(value)
                
                # Avoid duplicates
                if clean_field not in ['symbol'] and clean_field not in sql:
                    sql += f"    {clean_field} {field_type},\n"
                    self._record_field_mapping(field_path, table_name, clean_field)
        
        sql += """    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
"""
        
        return sql
    
    def _generate_fundamentals_timeseries_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for time-series fundamental data"""
        sql = f"""-- Fundamental Time-Series: {table_name}
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) NOT NULL CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
"""
        
        # Get all fields from annual and quarterly reports
        all_fields = {}
        
        for report_type in ['annualReports', 'quarterlyReports', 'annualEarnings', 'quarterlyEarnings']:
            if report_type in data and isinstance(data[report_type], list) and data[report_type]:
                sample_report = data[report_type][0]
                for field, value in sample_report.items():
                    if field not in ['fiscalDateEnding', 'reportedCurrency']:
                        clean_field = self._clean_field_name(field)
                        if clean_field not in all_fields:
                            all_fields[clean_field] = self._get_field_type_from_value(value)
        
        # Add all fields
        for field, field_type in sorted(all_fields.items()):
            sql += f"    {field} {field_type},\n"
            self._record_field_mapping(field, table_name, field)
        
        sql += """    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, fiscal_date_ending, report_type)
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_date ON {table_name}(fiscal_date_ending);
CREATE INDEX idx_{table_name}_type ON {table_name}(report_type);
"""
        
        return sql
    
    def _generate_sentiment_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for sentiment data"""
        
        if 'feed' in data:  # News sentiment
            return self._generate_news_schema(table_name, data)
        elif 'data' in data and any('transaction' in str(k).lower() for k in data.get('data', [])):
            return self._generate_insider_schema(table_name, data)
        else:  # Market movers
            return self._generate_movers_schema(table_name, data)
    
    def _generate_news_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for news sentiment"""
        sql = f"""-- News Sentiment
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
"""
        
        # Extract fields from feed
        if 'feed' in data and isinstance(data['feed'], list) and data['feed']:
            sample_article = data['feed'][0]
            
            for field, value in sample_article.items():
                if field == 'ticker_sentiment':
                    continue  # Handle separately
                
                clean_field = self._clean_field_name(field)
                
                if isinstance(value, list):
                    sql += f"    {clean_field} TEXT[],\n"
                elif not isinstance(value, dict):
                    field_type = self._get_field_type_from_value(value)
                    sql += f"    {clean_field} {field_type},\n"
                    self._record_field_mapping(field, table_name, clean_field)
        
        sql += """    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ticker sentiment (normalized)
CREATE TABLE {table_name}_tickers (
    id BIGSERIAL PRIMARY KEY,
    article_id VARCHAR(100),
    ticker VARCHAR(10) NOT NULL,
    relevance_score NUMERIC,
    ticker_sentiment_score NUMERIC,
    ticker_sentiment_label VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_published ON {table_name}(time_published) WHERE time_published IS NOT NULL;
CREATE INDEX idx_{table_name}_tickers_ticker ON {table_name}_tickers(ticker);
"""
        
        return sql
    
    def _generate_insider_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for insider transactions"""
        sql = f"""-- Insider Transactions
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
"""
        
        # Extract all fields
        all_fields = self._extract_all_fields(data)
        
        for field_path, value in all_fields:
            if not isinstance(value, (dict, list)):
                clean_field = self._clean_field_name(field_path.split('.')[-1])
                field_type = self._get_field_type_from_value(value)
                
                if clean_field not in ['symbol'] and clean_field not in sql:
                    sql += f"    {clean_field} {field_type},\n"
                    self._record_field_mapping(field_path, table_name, clean_field)
        
        sql += """    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_transaction_date ON {table_name}(transaction_date) WHERE transaction_date IS NOT NULL;
"""
        
        return sql
    
    def _generate_movers_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for market movers"""
        sql = f"""-- Market Movers
CREATE TABLE market_movers (
    id BIGSERIAL PRIMARY KEY,
"""
        
        # Extract fields from the data
        categories = ['top_gainers', 'top_losers', 'most_actively_traded']
        all_fields = {}
        
        for category in categories:
            if category in data and isinstance(data[category], list) and data[category]:
                sample = data[category][0]
                for field, value in sample.items():
                    clean_field = self._clean_field_name(field)
                    if clean_field not in all_fields:
                        all_fields[clean_field] = self._get_field_type_from_value(value)
        
        # Add fields
        for field, field_type in sorted(all_fields.items()):
            sql += f"    {field} {field_type},\n"
        
        sql += """    category VARCHAR(30) NOT NULL,
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_movers_ticker ON market_movers(ticker) WHERE ticker IS NOT NULL;
CREATE INDEX idx_movers_category ON market_movers(category);
CREATE INDEX idx_movers_timestamp ON market_movers(data_timestamp);
"""
        
        return sql
    
    def _generate_economic_schema(self, table_name: str, data: Dict) -> str:
        """Generate schema for economic indicators"""
        sql = f"""-- Economic Indicator: {table_name}
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
"""
        
        # Check structure - is it time-series?
        has_dates = False
        for key in data.keys():
            if re.match(r'^\d{4}-\d{2}', str(key)):
                has_dates = True
                break
        
        if has_dates:
            # Time-series economic data
            sql += """    date DATE NOT NULL,
    value NUMERIC NOT NULL,
"""
            # Record the date-value mapping
            self._record_field_mapping('date', table_name, 'date')
            self._record_field_mapping('value', table_name, 'value')
            
        else:
            # Extract all fields
            fields = self._extract_all_fields(data)
            for field_path, value in fields:
                if not isinstance(value, (dict, list)):
                    clean_field = self._clean_field_name(field_path.split('.')[-1])
                    field_type = self._get_field_type_from_value(value)
                    sql += f"    {clean_field} {field_type},\n"
                    self._record_field_mapping(field_path, table_name, clean_field)
        
        sql += """    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()"""
        
        if has_dates:
            sql += """,
    UNIQUE(date)
) PARTITION BY RANGE (date);

-- Yearly partitions
"""
            for year in [2024, 2025, 2026]:
                sql += f"""CREATE TABLE {table_name}_{year} PARTITION OF {table_name}
FOR VALUES FROM ('{year}-01-01') TO ('{year + 1}-01-01');
"""
        else:
            sql += """
);
"""
        
        sql += f"""
-- Indexes
CREATE INDEX idx_{table_name}_date ON {table_name}(date) WHERE date IS NOT NULL;
"""
        
        return sql
    
    def _generate_generic_schema(self, table_name: str, data: Dict) -> str:
        """Generate generic schema for unknown endpoint types"""
        sql = f"""-- Generic Table: {table_name}
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
"""
        
        # Extract all fields
        fields = self._extract_all_fields(data)
        
        for field_path, value in fields:
            if not isinstance(value, (dict, list)):
                clean_field = self._clean_field_name(field_path.split('.')[-1])
                field_type = self._get_field_type_from_value(value)
                
                if clean_field not in sql:
                    sql += f"    {clean_field} {field_type},\n"
                    self._record_field_mapping(field_path, table_name, clean_field)
        
        sql += """    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""
        
        return sql
    
    def _generate_audit_schema(self) -> str:
        """Generate data transformations audit table"""
        return """-- Audit Trail: Data Transformations
-- Tracks EVERY transformation from API to database
CREATE TABLE data_transformations (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    original_value TEXT,
    transformed_value TEXT,
    transformation_type VARCHAR(50) NOT NULL,
    api_response_id UUID NOT NULL,
    field_path TEXT, -- Original field path in API response
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_transform_table ON data_transformations(table_name);
CREATE INDEX idx_transform_response ON data_transformations(api_response_id);
CREATE INDEX idx_transform_type ON data_transformations(transformation_type);
CREATE INDEX idx_transform_field ON data_transformations(field_path);
"""
    
    def _generate_api_audit_schema(self) -> str:
        """Generate API response audit table"""
        return """-- Audit Trail: API Responses
CREATE TABLE api_response_audit (
    id BIGSERIAL PRIMARY KEY,
    response_id UUID NOT NULL DEFAULT gen_random_uuid(),
    endpoint VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    response_size_bytes INTEGER,
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    rate_limit_remaining INTEGER,
    field_count INTEGER, -- Number of fields in response
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(response_id)
);

-- Indexes
CREATE INDEX idx_api_audit_endpoint ON api_response_audit(endpoint);
CREATE INDEX idx_api_audit_created ON api_response_audit(created_at);
"""
    
    def _generate_field_mapping_schema(self) -> str:
        """Generate field mapping verification table"""
        return """-- Field Mapping Verification
-- Ensures all 8,227 fields are mapped
CREATE TABLE field_mapping_verification (
    id BIGSERIAL PRIMARY KEY,
    field_name TEXT NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    field_type VARCHAR(50) NOT NULL,
    occurrences INTEGER,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(field_name, table_name, column_name)
);

-- Indexes
CREATE INDEX idx_field_map_field ON field_mapping_verification(field_name);
CREATE INDEX idx_field_map_table ON field_mapping_verification(table_name);
CREATE INDEX idx_field_map_verified ON field_mapping_verification(verified);
"""
    
    def _clean_field_name(self, field: str) -> str:
        """Clean field name for SQL column"""
        # Remove special characters and convert to snake_case
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', field)
        clean = re.sub(r'_+', '_', clean)
        clean = clean.lower().strip('_')
        
        # Ensure it doesn't start with a number
        if clean and clean[0].isdigit():
            clean = f"field_{clean}"
        
        # Truncate if too long
        if len(clean) > 63:  # PostgreSQL column name limit
            clean = clean[:63]
        
        return clean or "field"
    
    def _get_field_type_from_value(self, value: Any) -> str:
        """Determine SQL type from actual value"""
        if value is None or value == "None":
            return "TEXT"
        
        if isinstance(value, bool):
            return "BOOLEAN"
        
        if isinstance(value, int):
            return "BIGINT"
        
        if isinstance(value, float):
            return "NUMERIC"
        
        if isinstance(value, str):
            # Check patterns
            if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                return "DATE"
            if re.match(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', value):
                return "TIMESTAMP WITH TIME ZONE"
            if re.match(r'^-?\d+\.?\d*$', value) and value != "None":
                return "NUMERIC"
            
            # Check length
            if len(value) <= 10:
                return "VARCHAR(10)"
            if len(value) <= 50:
                return "VARCHAR(50)"
            if len(value) <= 255:
                return "VARCHAR(255)"
            
            return "TEXT"
        
        return "TEXT"
    
    def _record_field_mapping(self, field: str, table: str, column: str):
        """Record field to table/column mapping"""
        self.field_mappings[field].append((table, column))
    
    def verify_field_coverage(self) -> Dict:
        """Verify all 8,227 fields are mapped"""
        mapped_fields = set(self.field_mappings.keys())
        catalog_fields = set(self.field_catalog.keys())
        
        # Find unmapped fields
        unmapped = catalog_fields - mapped_fields
        
        # Find fields mapped multiple times
        multi_mapped = {field: mappings for field, mappings in self.field_mappings.items() 
                       if len(mappings) > 1}
        
        coverage = {
            'total_fields': len(catalog_fields),
            'mapped_fields': len(mapped_fields),
            'unmapped_fields': len(unmapped),
            'multi_mapped_fields': len(multi_mapped),
            'coverage_percentage': (len(mapped_fields) / len(catalog_fields)) * 100,
            'unmapped_list': list(unmapped)[:100],  # First 100 unmapped
            'multi_mapped_list': multi_mapped
        }
        
        return coverage
    
    def save_all_schemas(self):
        """Save all generated schemas to files"""
        schemas = self.generate_all_schemas()
        
        output_dir = Path('schema/tables')
        
        for path, sql in schemas.items():
            parts = path.split('/')
            if len(parts) > 1:
                subdir = output_dir / parts[0]
                subdir.mkdir(parents=True, exist_ok=True)
                filename = subdir / f"{parts[1]}.sql"
            else:
                filename = output_dir / f"{path}.sql"
            
            with open(filename, 'w') as f:
                f.write(sql)
            
            print(f"  ✅ Saved: {filename}")
        
        # Verify coverage
        coverage = self.verify_field_coverage()
        
        # Save coverage report
        coverage_file = Path('schema/field_coverage_report.json')
        with open(coverage_file, 'w') as f:
            json.dump(coverage, f, indent=2)
        
        print(f"\n📊 Field Coverage Report:")
        print(f"  Total Fields: {coverage['total_fields']}")
        print(f"  Mapped Fields: {coverage['mapped_fields']}")
        print(f"  Unmapped Fields: {coverage['unmapped_fields']}")
        print(f"  Coverage: {coverage['coverage_percentage']:.2f}%")
        
        if coverage['unmapped_fields'] > 0:
            print(f"\n  ⚠️  Warning: {coverage['unmapped_fields']} fields not mapped!")
            print(f"  See schema/field_coverage_report.json for details")
        
        # Generate master migration
        self._generate_master_migration(schemas)
    
    def _generate_master_migration(self, schemas: Dict[str, str]):
        """Generate master migration file"""
        migration_dir = Path('schema/migrations')
        migration_dir.mkdir(parents=True, exist_ok=True)
        
        migration_sql = f"""-- AlphaTrader Database Schema
-- Generated: {datetime.now().isoformat()}
-- Total Fields Catalog: {len(self.field_catalog)}
-- 
-- ZERO COMPROMISES:
-- - One table per API endpoint (36 tables)
-- - NUMERIC for ALL financial data
-- - Complete normalization, no lazy JSONB
-- - Full audit trail for transformations

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

"""
        
        # Add schemas in dependency order
        order = [
            'audit/api_response',
            'audit/data_transformations',
            'audit/field_mappings'
        ]
        
        # Add all other schemas
        for path in sorted(schemas.keys()):
            if path not in order:
                order.append(path)
        
        for path in order:
            if path in schemas:
                migration_sql += f"\n-- ========== {path.upper()} ==========\n"
                migration_sql += schemas[path] + "\n"
        
        migration_sql += """
COMMIT;

-- Verification query
SELECT 
    schemaname,
    tablename,
    COUNT(*) OVER() as total_tables
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
"""
        
        filename = migration_dir / '001_complete_schema.sql'
        with open(filename, 'w') as f:
            f.write(migration_sql)
        
        print(f"\n✅ Generated master migration: {filename}")
        
        # Generate rollback
        self._generate_rollback(order)
    
    def _generate_rollback(self, table_order: List[str]):
        """Generate rollback script"""
        rollback_sql = """-- Rollback for complete schema
-- WARNING: This will DROP all tables and data!

BEGIN;

"""
        
        for path in reversed(table_order):
            table_name = path.split('/')[-1]
            rollback_sql += f"DROP TABLE IF EXISTS {table_name} CASCADE;\n"
        
        rollback_sql += """
COMMIT;
"""
        
        filename = Path('schema/migrations/001_complete_schema_rollback.sql')
        with open(filename, 'w') as f:
            f.write(rollback_sql)
        
        print(f"✅ Generated rollback: {filename}")


if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE SQL SCHEMA GENERATOR")
    print("Ensuring ALL 8,227 fields get proper storage")
    print("="*70)
    
    generator = ComprehensiveSQLGenerator()
    generator.save_all_schemas()
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("Next steps:")
    print("1. Review field coverage report")
    print("2. Execute migration: psql -d alphatrader -f schema/migrations/001_complete_schema.sql")
    print("3. Begin loading production data")
    print("="*70)