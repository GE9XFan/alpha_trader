-- Economic Indicator: cpi
CREATE TABLE cpi (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(50),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_cpi_date ON cpi(date) WHERE date IS NOT NULL;
