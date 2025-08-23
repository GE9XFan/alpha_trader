-- Economic Indicator: treasury_yield
CREATE TABLE treasury_yield (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(10),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_treasury_yield_date ON treasury_yield(date) WHERE date IS NOT NULL;
