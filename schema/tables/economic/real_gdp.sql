-- Economic Indicator: real_gdp
CREATE TABLE real_gdp (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(50),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_real_gdp_date ON real_gdp(date) WHERE date IS NOT NULL;
