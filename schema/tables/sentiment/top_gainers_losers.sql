-- SENTIMENT: top_gainers_losers
-- Generated from ACTUAL API response investigation
-- Total fields in response: 20

CREATE TABLE top_gainers_losers (
    id BIGSERIAL PRIMARY KEY,
    metadata TEXT,
    last_updated TEXT,
    ticker TEXT,
    price TEXT,
    change_amount TEXT,
    change_percentage TEXT,
    volume TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_top_gainers_losers_date ON top_gainers_losers(date) WHERE date IS NOT NULL;
