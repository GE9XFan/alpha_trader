-- Insider Transactions
CREATE TABLE insider_transactions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    transaction_date DATE,
    ticker VARCHAR(10),
    executive VARCHAR(50),
    executive_title VARCHAR(50),
    security_type VARCHAR(50),
    acquisition_or_disposal VARCHAR(10),
    shares NUMERIC,
    share_price NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_transaction_date ON {table_name}(transaction_date) WHERE transaction_date IS NOT NULL;
