-- Analytics: analytics_sliding_window
-- Handles complex nested calculation results
CREATE TABLE analytics_sliding_window (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL DEFAULT gen_random_uuid(),
    meta_symbols VARCHAR(10),
    meta_window_size BIGINT,
    meta_min_dt DATE,
    meta_max_dt DATE,
    meta_ohlc VARCHAR(10),
    meta_interval VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(request_id)
);

-- Calculation results (normalized from nested structure)
CREATE TABLE analytics_sliding_window_calculations (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES analytics_sliding_window(request_id) ON DELETE CASCADE,
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
CREATE TABLE analytics_sliding_window_calculations_2025_08 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE analytics_sliding_window_calculations_2025_09 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE analytics_sliding_window_calculations_2025_10 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE analytics_sliding_window_calculations_2025_11 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE analytics_sliding_window_calculations_2025_12 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE analytics_sliding_window_calculations_2026_01 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE analytics_sliding_window_calculations_2026_02 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE analytics_sliding_window_calculations_2026_03 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE analytics_sliding_window_calculations_2026_04 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE analytics_sliding_window_calculations_2026_05 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE analytics_sliding_window_calculations_2026_06 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE analytics_sliding_window_calculations_2026_07 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Performance indexes
CREATE INDEX idx_analytics_sliding_window_request ON analytics_sliding_window(request_id);
CREATE INDEX idx_analytics_sliding_window_calc_request ON analytics_sliding_window_calculations(request_id);
CREATE INDEX idx_analytics_sliding_window_calc_symbol ON analytics_sliding_window_calculations(symbol, date);
CREATE INDEX idx_analytics_sliding_window_calc_category ON analytics_sliding_window_calculations(calculation_category, calculation_type);
