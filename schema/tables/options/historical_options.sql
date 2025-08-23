-- OPTIONS: historical_options
-- Generated from ACTUAL API response investigation
-- Total fields in response: 23

CREATE TABLE historical_options (
    id BIGSERIAL PRIMARY KEY,
    contract_id VARCHAR(50),
    symbol VARCHAR(10) NOT NULL,
    expiration DATE,
    strike NUMERIC NOT NULL,
    option_type VARCHAR(4),
    last_price NUMERIC,
    mark NUMERIC,
    bid NUMERIC,
    bid_size INTEGER,
    ask NUMERIC,
    ask_size INTEGER,
    volume BIGINT,
    open_interest BIGINT,
    quote_date DATE,
    implied_volatility NUMERIC,
    delta NUMERIC,
    gamma NUMERIC,
    theta NUMERIC,
    vega NUMERIC,
    rho NUMERIC,
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_historical_options_symbol ON historical_options(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_historical_options_date ON historical_options(date) WHERE date IS NOT NULL;
CREATE INDEX idx_historical_options_expiration ON historical_options(expiration) WHERE expiration IS NOT NULL;
CREATE INDEX idx_historical_options_strike ON historical_options(strike) WHERE strike IS NOT NULL;
