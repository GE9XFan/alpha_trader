-- Options Chain: realtime_options
CREATE TABLE realtime_options (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT {table_name}_unique UNIQUE(symbol, contract_id, data_timestamp)
) PARTITION BY RANGE (data_timestamp);

-- Monthly partitions
CREATE TABLE realtime_options_2025_08 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE realtime_options_2025_09 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE realtime_options_2025_10 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE realtime_options_2025_11 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE realtime_options_2025_12 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE realtime_options_2026_01 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE realtime_options_2026_02 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE realtime_options_2026_03 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE realtime_options_2026_04 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE realtime_options_2026_05 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE realtime_options_2026_06 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE realtime_options_2026_07 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes for performance
CREATE INDEX idx_realtime_options_symbol ON realtime_options(symbol);
CREATE INDEX idx_realtime_options_expiration ON realtime_options(expiration) WHERE expiration IS NOT NULL;
CREATE INDEX idx_realtime_options_strike ON realtime_options(strike) WHERE strike IS NOT NULL;
CREATE INDEX idx_realtime_options_timestamp ON realtime_options(data_timestamp);
