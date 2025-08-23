-- Economic Indicator: inflation
CREATE TABLE inflation (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(10),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_inflation_date ON inflation(date) WHERE date IS NOT NULL;
