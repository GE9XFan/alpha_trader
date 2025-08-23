-- ECONOMIC: real_gdp
-- Generated from ACTUAL API response investigation
-- Total fields in response: 6

CREATE TABLE real_gdp (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    interval TEXT,
    unit TEXT,
    date TEXT,
    value TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_real_gdp_date ON real_gdp(date) WHERE date IS NOT NULL;
