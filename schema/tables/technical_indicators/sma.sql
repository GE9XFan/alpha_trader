-- TECHNICAL_INDICATORS: sma
-- Generated from ACTUAL API response investigation
-- Total fields in response: 10

CREATE TABLE sma (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    sma NUMERIC,
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50)    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sma_symbol ON sma(symbol) WHERE symbol IS NOT NULL;
