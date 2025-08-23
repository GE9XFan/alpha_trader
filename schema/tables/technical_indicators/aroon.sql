-- Technical Indicator: aroon
CREATE TABLE aroon_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    aroon_down NUMERIC,
    aroon_up NUMERIC,
    
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
CREATE TABLE aroon_data_2025_08 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE aroon_data_2025_09 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE aroon_data_2025_10 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE aroon_data_2025_11 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE aroon_data_2025_12 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE aroon_data_2026_01 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE aroon_data_2026_02 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE aroon_data_2026_03 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE aroon_data_2026_04 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE aroon_data_2026_05 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE aroon_data_2026_06 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE aroon_data_2026_07 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_aroon_symbol ON aroon_data(symbol);
CREATE INDEX idx_aroon_timestamp ON aroon_data(timestamp);
CREATE INDEX idx_aroon_interval ON aroon_data(interval);
