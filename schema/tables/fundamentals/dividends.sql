-- FUNDAMENTALS: dividends
-- Generated from ACTUAL API response investigation
-- Total fields in response: 7

CREATE TABLE dividends (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) PRIMARY KEY,
    ex_dividend_date TEXT,
    declaration_date TEXT,
    record_date TEXT,
    payment_date TEXT,
    amount TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_dividends_symbol ON dividends(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_dividends_date ON dividends(date) WHERE date IS NOT NULL;
