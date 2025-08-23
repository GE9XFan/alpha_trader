-- Economic Indicator: federal_funds_rate
CREATE TABLE federal_funds_rate (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(10),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_federal_funds_rate_date ON federal_funds_rate(date) WHERE date IS NOT NULL;
