-- ECONOMIC: cpi
-- Generated from ACTUAL API response investigation
-- Total fields in response: 6

CREATE TABLE cpi (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    interval TEXT,
    unit TEXT,
    date TEXT,
    value TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_cpi_date ON cpi(date) WHERE date IS NOT NULL;
