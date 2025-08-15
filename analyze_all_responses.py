#!/usr/bin/env python3
"""
BATCH ANALYZER AND SCHEMA GENERATOR
Processes all API responses and generates schemas
Run AFTER test_api.py completes
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchAnalyzerSchemaGenerator:
    """Analyze all API responses and generate schemas"""
    
    def __init__(self):
        self.response_dir = Path("data/api_responses")
        self.schema_dir = Path("data/schemas")
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def analyze_response(self, response: dict, api_name: str) -> dict:
        """GENERIC analyzer - works for ANY API response structure"""
        analysis = {
            'api_name': api_name,
            'timestamp': datetime.now().isoformat(),
            'top_level_keys': list(response.keys()),
            'field_analysis': {},
            'record_count': 0,
            'sample_data': None
        }
        
        # GENERIC LOGIC: Find where the actual data is
        sample_record = None
        
        # Try different common patterns
        for key, value in response.items():
            # Skip metadata keys
            if key in ['endpoint', 'message', 'Meta Data', 'Information', 'Note', 'Error Message']:
                continue
                
            # Found an array of records
            if isinstance(value, list) and len(value) > 0:
                analysis['record_count'] = len(value)
                sample_record = value[0]
                logger.info(f"Found array '{key}' with {len(value)} records")
                break
                
            # Found a dict of records (time series)
            if isinstance(value, dict) and len(value) > 0:
                # Check if it's time series data (keys are dates/times)
                first_key = list(value.keys())[0]
                if isinstance(value[first_key], dict):
                    analysis['record_count'] = len(value)
                    sample_record = value[first_key]
                    logger.info(f"Found time series '{key}' with {len(value)} records")
                    break
        
        # If no nested structure found, treat the whole response as the record
        if sample_record is None and response:
            # Filter out metadata
            sample_record = {k: v for k, v in response.items() 
                            if k not in ['endpoint', 'message', 'Meta Data', 'Information', 'Note', 'Error Message']}
            if sample_record:
                analysis['record_count'] = 1
                logger.info("Using top-level response as record")
        
        # Analyze the sample record
        if sample_record:
            analysis['sample_data'] = sample_record
            
            for field, value in sample_record.items():
                # Determine type from actual value
                value_type = 'unknown'
                is_numeric = False
                is_decimal = False
                
                if value is None or value == '':
                    value_type = 'nullable'
                elif isinstance(value, bool):
                    value_type = 'boolean'
                elif isinstance(value, (int, float)):
                    value_type = 'number'
                    is_numeric = True
                    is_decimal = isinstance(value, float) or '.' in str(value)
                elif isinstance(value, str):
                    # Check if string contains a number
                    try:
                        if '.' in value:
                            float(value)
                            value_type = 'decimal_string'
                            is_numeric = True
                            is_decimal = True
                        else:
                            int(value)
                            value_type = 'integer_string'
                            is_numeric = True
                            is_decimal = False
                    except:
                        value_type = 'string'
                elif isinstance(value, dict):
                    value_type = 'object'
                elif isinstance(value, list):
                    value_type = 'array'
                
                analysis['field_analysis'][field] = {
                    'type': value_type,
                    'is_numeric': is_numeric,
                    'is_decimal': is_decimal,
                    'sample': str(value)[:100] if value else None,
                    'nullable': value is None or value == ''
                }
        
        # Save analysis
        analysis_file = self.schema_dir / f"{api_name}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"✓ Analysis saved to {analysis_file}")
        logger.info(f"  Found {len(analysis['field_analysis'])} fields, {analysis['record_count']} records")
        
        return analysis
    
    def generate_schema(self, analysis: dict) -> str:
        """GENERIC schema generator - works for ANY analyzed data"""
        api_name = analysis['api_name']
        table_name = f"av_{api_name.lower()}"
        
        sql = f"-- Generated from {api_name} API\n"
        sql += f"-- Date: {analysis['timestamp']}\n"
        sql += f"-- Records analyzed: {analysis['record_count']}\n"
        sql += f"-- Fields found: {len(analysis['field_analysis'])}\n\n"
        
        sql += f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        sql += f"    id SERIAL PRIMARY KEY,\n"
        
        if not analysis['field_analysis']:
            # No fields found - store as JSONB
            logger.warning(f"No fields analyzed for {api_name} - using JSONB storage")
            sql += f"    data JSONB NOT NULL,\n"
        else:
            for field_name, field_info in analysis['field_analysis'].items():
                # Clean field name for PostgreSQL
                pg_field = field_name.lower().replace(' ', '_').replace('.', '_').replace('-', '_')
                
                # Start with safe default
                pg_type = 'TEXT'
                
                # FIRST: Check if it's numeric (this takes priority)
                if field_info['is_numeric']:
                    if field_info['is_decimal']:
                        # Decimal number - use field name hints for precision
                        if any(x in field_name.lower() for x in ['price', 'strike', 'last', 'bid', 'ask', 'mark']):
                            pg_type = 'DECIMAL(12,4)'
                        elif any(x in field_name.lower() for x in ['delta', 'gamma', 'theta', 'vega', 'rho']):
                            pg_type = 'DECIMAL(10,6)'
                        elif any(x in field_name.lower() for x in ['iv', 'implied_volatility', 'volatility']):
                            pg_type = 'DECIMAL(10,6)'
                        elif any(x in field_name.lower() for x in ['percent', 'rate', 'yield']):
                            pg_type = 'DECIMAL(8,4)'
                        elif any(x in field_name.lower() for x in ['eps', 'pe', 'dividend']):
                            pg_type = 'DECIMAL(12,4)'
                        elif any(x in field_name.lower() for x in ['revenue', 'income', 'cash', 'assets', 'liabilities']):
                            pg_type = 'DECIMAL(20,2)'
                        else:
                            pg_type = 'DECIMAL(15,6)'  # Generic decimal
                    else:
                        # Integer - use field name hints for size
                        if any(x in field_name.lower() for x in ['volume', 'open_interest', 'oi', 'shares']):
                            pg_type = 'BIGINT'
                        elif any(x in field_name.lower() for x in ['size', 'count', 'quantity']):
                            pg_type = 'BIGINT'
                        else:
                            pg_type = 'INTEGER'
                
                # SECOND: Override for special field types (only if not numeric)
                elif not field_info['is_numeric']:
                    if any(x in field_name.lower() for x in ['date', 'time', 'expir', 'timestamp']):
                        pg_type = 'TIMESTAMP'
                    elif 'contractid' in field_name.lower() or field_name.lower() == 'contractid':
                        pg_type = 'VARCHAR(50)'
                    elif field_name.lower() in ['symbol', 'ticker']:
                        pg_type = 'VARCHAR(10)'
                    elif field_name.lower() == 'type':
                        pg_type = 'VARCHAR(10)'  # call/put
                    elif field_name.lower() in ['sector', 'industry', 'exchange']:
                        pg_type = 'VARCHAR(100)'
                    elif field_name.lower() in ['name', 'description']:
                        pg_type = 'TEXT'
                    elif any(x in field_name.lower() for x in ['id', 'identifier', 'code', 'cusip', 'isin']):
                        pg_type = 'VARCHAR(50)'
                    elif field_info['type'] == 'boolean':
                        pg_type = 'BOOLEAN'
                    elif field_info['type'] in ['object', 'array']:
                        pg_type = 'JSONB'
                    else:
                        # Keep as TEXT for other strings
                        pg_type = 'TEXT'
                
                sql += f"    {pg_field} {pg_type},\n"
        
        sql += f"    api_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
        sql += f"    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
        sql += f");\n\n"
        
        # GENERIC INDEXES - based on common field names
        sql += f"-- Indexes for common query patterns\n"
        sql += f"CREATE INDEX idx_{table_name}_created ON {table_name}(created_at);\n"
        
        # Add indexes for commonly queried fields if they exist
        common_index_fields = ['symbol', 'contractid', 'date', 'timestamp', 'expiration', 'strike', 'type', 'ticker']
        for field in analysis['field_analysis'].keys():
            field_lower = field.lower()
            if any(common in field_lower for common in common_index_fields):
                pg_field = field_lower.replace(' ', '_').replace('.', '_').replace('-', '_')
                sql += f"CREATE INDEX idx_{table_name}_{pg_field} ON {table_name}({pg_field});\n"
        
        # Save schema
        schema_file = self.schema_dir / f"{table_name}_schema.sql"
        with open(schema_file, 'w') as f:
            f.write(sql)
        
        logger.info(f"✓ Schema saved to {schema_file}")
        
        return sql
    
    def process_all_responses(self):
        """Process all JSON response files"""
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING ALL API RESPONSES")
        logger.info("=" * 80)
        
        # Find all JSON files in response directory
        json_files = list(self.response_dir.glob("*.json"))
        
        if not json_files:
            logger.error(f"No JSON files found in {self.response_dir}")
            logger.error("Run test_api.py first to generate API responses!")
            return
        
        logger.info(f"Found {len(json_files)} response files to process\n")
        
        successful = 0
        failed = 0
        
        for json_file in sorted(json_files):
            # Extract API name from filename
            # Format: api_name_SYMBOL_timestamp.json
            filename = json_file.stem
            parts = filename.split('_')
            
            # Handle different naming patterns
            if len(parts) >= 2:
                # Try to reconstruct API name (might be multi-part like analytics_fixed_window)
                # Remove the symbol (AAPL) and timestamp parts
                api_name_parts = []
                for part in parts:
                    # Stop when we hit a known symbol or timestamp pattern
                    if part in ['AAPL', 'SPY', 'MSFT'] or part.isdigit():
                        break
                    api_name_parts.append(part)
                api_name = '_'.join(api_name_parts)
            else:
                api_name = parts[0]
            
            logger.info(f"\nProcessing: {json_file.name}")
            logger.info(f"API Name: {api_name}")
            
            try:
                # Load the response
                with open(json_file, 'r') as f:
                    response = json.load(f)
                
                # Check for empty response
                if not response or response == {}:
                    logger.warning(f"  ⚠ Empty response, skipping")
                    self.results[api_name] = 'EMPTY_RESPONSE'
                    continue
                
                # Check for error response
                if 'Error Message' in response:
                    logger.error(f"  ✗ API error: {response['Error Message']}")
                    self.results[api_name] = 'API_ERROR'
                    failed += 1
                    continue
                
                # Analyze the response
                analysis = self.analyze_response(response, api_name)
                
                # Generate schema
                schema = self.generate_schema(analysis)
                
                self.results[api_name] = 'SUCCESS'
                successful += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process: {e}")
                self.results[api_name] = f'ERROR: {str(e)}'
                failed += 1
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Successfully processed: {successful}")
        logger.info(f"✗ Failed: {failed}")
        logger.info(f"⚠ Empty responses: {sum(1 for v in self.results.values() if v == 'EMPTY_RESPONSE')}")
        
        # Save processing results
        results_file = self.schema_dir / f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")
        
        # List all generated schemas
        logger.info("\n" + "=" * 80)
        logger.info("GENERATED SCHEMAS")
        logger.info("=" * 80)
        
        sql_files = sorted(self.schema_dir.glob("av_*.sql"))
        for sql_file in sql_files:
            logger.info(f"  ✓ {sql_file.name}")
        
        logger.info(f"\nTotal schemas generated: {len(sql_files)}")
        
        # Next steps
        logger.info("\n" + "=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)
        logger.info("1. Review generated schemas in data/schemas/")
        logger.info("2. Create database tables:")
        logger.info("   for f in data/schemas/av_*.sql; do")
        logger.info("     psql -U your_user -d trading_system < $f")
        logger.info("   done")
        logger.info("3. Implement ingestion logic for each API")
        logger.info("4. Test data persistence")


def main():
    """Run batch processing"""
    processor = BatchAnalyzerSchemaGenerator()
    processor.process_all_responses()
    return 0


if __name__ == "__main__":
    sys.exit(main())