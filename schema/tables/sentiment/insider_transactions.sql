-- SENTIMENT: insider_transactions
-- Generated from ACTUAL API response investigation
-- Total fields in response: 9

CREATE TABLE insider_transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_date TEXT,
    ticker TEXT,
    executive TEXT,
    executive_title TEXT,
    security_type TEXT,
    acquisition_or_disposal TEXT,
    shares TEXT,
    share_price TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_insider_transactions_date ON insider_transactions(date) WHERE date IS NOT NULL;
