-- FUNDAMENTALS: splits
-- Generated from ACTUAL API response investigation
-- Total fields in response: 4

CREATE TABLE splits (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) PRIMARY KEY,
    effective_date TEXT,
    split_factor TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_splits_symbol ON splits(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_splits_date ON splits(date) WHERE date IS NOT NULL;
