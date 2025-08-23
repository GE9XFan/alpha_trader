-- TECHNICAL_INDICATORS: macd
-- Generated from ACTUAL API response investigation
-- Total fields in response: 14

CREATE TABLE macd (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    macd NUMERIC,
    macd_hist NUMERIC,
    macd_signal NUMERIC,
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50)    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_macd_symbol ON macd(symbol) WHERE symbol IS NOT NULL;
