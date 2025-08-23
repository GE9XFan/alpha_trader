-- Fundamental Data: dividends
CREATE TABLE dividends (
    symbol VARCHAR(10) PRIMARY KEY,
    ex_dividend_date DATE,
    declaration_date DATE,
    record_date DATE,
    payment_date DATE,
    amount NUMERIC,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
