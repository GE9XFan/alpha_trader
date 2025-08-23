-- Market Movers
CREATE TABLE market_movers (
    id BIGSERIAL PRIMARY KEY,
    change_amount NUMERIC,
    change_percentage VARCHAR(10),
    price NUMERIC,
    ticker VARCHAR(10),
    volume NUMERIC,
    category VARCHAR(30) NOT NULL,
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_movers_ticker ON market_movers(ticker) WHERE ticker IS NOT NULL;
CREATE INDEX idx_movers_category ON market_movers(category);
CREATE INDEX idx_movers_timestamp ON market_movers(data_timestamp);
