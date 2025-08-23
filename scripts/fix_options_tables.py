#!/usr/bin/env python3
"""
Fix Options Tables - Ensure ALL fields are included
"""

import json
from pathlib import Path

def generate_options_sql():
    """Generate proper SQL for options tables with ALL fields"""
    
    # Load actual options response to get all fields
    realtime_file = Path('data/api_responses/options/realtime_options_response.json')
    historical_file = Path('data/api_responses/options/historical_options_response.json')
    
    # Get fields from actual responses
    with open(realtime_file, 'r') as f:
        realtime_data = json.load(f)
    
    with open(historical_file, 'r') as f:
        historical_data = json.load(f)
    
    # Extract all fields from options array
    all_fields = set()
    
    if 'options' in realtime_data:
        for option in realtime_data['options']:
            all_fields.update(option.keys())
    
    if 'options' in historical_data:
        for option in historical_data['options']:
            all_fields.update(option.keys())
    
    print(f"Found {len(all_fields)} unique fields in options data")
    print(f"Fields: {sorted(all_fields)}")
    
    # Generate SQL for both tables
    for table_name in ['realtime_options', 'historical_options']:
        sql = f"""-- Options Chain: {table_name}
-- Complete schema with ALL fields from API response
CREATE TABLE {table_name} (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    contract_id VARCHAR(50) NOT NULL,
    
    -- Option Details
    option_type VARCHAR(4) CHECK (option_type IN ('call', 'put')),
    strike NUMERIC NOT NULL,
    expiration DATE NOT NULL,
    expiration_type VARCHAR(20),
    days_to_expiration INTEGER,
    
    -- Pricing Data (ALL NUMERIC for precision)
    last_price NUMERIC,
    bid NUMERIC,
    ask NUMERIC,
    bid_size INTEGER,
    ask_size INTEGER,
    close NUMERIC,
    open_price NUMERIC,
    high NUMERIC,
    low NUMERIC,
    mark NUMERIC,
    previous_close NUMERIC,
    change NUMERIC,
    change_percent NUMERIC,
    
    -- Volume and Interest
    volume BIGINT,
    open_interest BIGINT,
    
    -- Greeks (ALL NUMERIC for precision)
    implied_volatility NUMERIC,
    delta NUMERIC,
    gamma NUMERIC,
    theta NUMERIC,
    vega NUMERIC,
    rho NUMERIC,
    
    -- Additional Fields
    in_the_money BOOLEAN,
    near_the_money BOOLEAN,
    theoretical_value NUMERIC,
    time_value NUMERIC,
    intrinsic_value NUMERIC,
    
    -- Metadata
    last_trade_datetime TIMESTAMP WITH TIME ZONE,
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT {table_name}_unique UNIQUE(symbol, contract_id, data_timestamp)
) PARTITION BY RANGE (data_timestamp);

-- Monthly partitions for next 12 months
"""
        
        # Add partitions
        from datetime import datetime
        for i in range(12):
            month = (datetime.now().month + i - 1) % 12 + 1
            year = datetime.now().year + ((datetime.now().month + i - 1) // 12)
            next_month = (month % 12) + 1
            next_year = year if next_month > 1 else year + 1
            
            sql += f"""CREATE TABLE {table_name}_{year}_{month:02d} 
PARTITION OF {table_name}
FOR VALUES FROM ('{year}-{month:02d}-01') TO ('{next_year}-{next_month:02d}-01');
"""
        
        # Add indexes
        sql += f"""
-- Performance indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_expiration ON {table_name}(expiration);
CREATE INDEX idx_{table_name}_strike ON {table_name}(strike);
CREATE INDEX idx_{table_name}_contract ON {table_name}(contract_id);
CREATE INDEX idx_{table_name}_timestamp ON {table_name}(data_timestamp);
CREATE INDEX idx_{table_name}_type ON {table_name}(option_type);
CREATE INDEX idx_{table_name}_itm ON {table_name}(in_the_money) WHERE in_the_money IS NOT NULL;
"""
        
        # Save the fixed SQL
        output_file = Path(f'schema/tables/options/{table_name}.sql')
        with open(output_file, 'w') as f:
            f.write(sql)
        
        print(f"✅ Fixed: {output_file}")

if __name__ == "__main__":
    generate_options_sql()