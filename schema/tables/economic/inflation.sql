-- ECONOMIC: inflation
-- Generated from ACTUAL API response investigation
-- Total fields in response: 6

CREATE TABLE inflation (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    interval TEXT,
    unit TEXT,
    date TEXT,
    value TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_inflation_date ON inflation(date) WHERE date IS NOT NULL;
