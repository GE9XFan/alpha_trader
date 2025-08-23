-- FUNDAMENTALS: earnings
-- Generated from ACTUAL API response investigation
-- Total fields in response: 12

CREATE TABLE earnings (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) PRIMARY KEY,
    fiscaldateending TEXT,
    reportedeps TEXT,
    reporteddate TEXT,
    estimatedeps TEXT,
    surprise TEXT,
    surprisepercentage TEXT,
    reporttime TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_earnings_symbol ON earnings(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_earnings_date ON earnings(date) WHERE date IS NOT NULL;
