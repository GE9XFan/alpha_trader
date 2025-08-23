-- Fundamental Data: splits
CREATE TABLE splits (
    symbol VARCHAR(10) PRIMARY KEY,
    effective_date DATE,
    split_factor NUMERIC,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
