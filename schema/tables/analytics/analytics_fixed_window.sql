-- ANALYTICS: analytics_fixed_window
-- Generated from ACTUAL API response investigation
-- Total fields in response: 64

CREATE TABLE analytics_fixed_window (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL DEFAULT gen_random_uuid(),
    symbols TEXT NOT NULL,
    window_size INTEGER,
    min_date DATE,
    max_date DATE,
    ohlc VARCHAR(10),
    interval VARCHAR(20)    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_analytics_fixed_window_symbol ON analytics_fixed_window(symbol) WHERE symbol IS NOT NULL;
