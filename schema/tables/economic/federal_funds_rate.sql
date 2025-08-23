-- ECONOMIC: federal_funds_rate
-- Generated from ACTUAL API response investigation
-- Total fields in response: 6

CREATE TABLE federal_funds_rate (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    interval TEXT,
    unit TEXT,
    date TEXT,
    value TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_federal_funds_rate_date ON federal_funds_rate(date) WHERE date IS NOT NULL;
