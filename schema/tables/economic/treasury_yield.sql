-- ECONOMIC: treasury_yield
-- Generated from ACTUAL API response investigation
-- Total fields in response: 6

CREATE TABLE treasury_yield (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    interval TEXT,
    unit TEXT,
    date TEXT,
    value TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_treasury_yield_date ON treasury_yield(date) WHERE date IS NOT NULL;
