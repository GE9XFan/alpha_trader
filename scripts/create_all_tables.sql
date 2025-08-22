-- ============================================================================
-- COMPLETE DATABASE SCHEMA - ALL TABLES
-- Based on ACTUAL API responses from Alpha Vantage and IBKR
-- Critical: Every field mapped exactly as returned by APIs
-- ============================================================================

-- ============================================================================
-- IBKR TABLES
-- ============================================================================

-- IBKR Account Information
CREATE TABLE IF NOT EXISTS ibkr_accounts (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IBKR Account Summary (80+ different tags)
CREATE TABLE IF NOT EXISTS ibkr_account_summary (
    id SERIAL PRIMARY KEY,
    account VARCHAR(50) NOT NULL,
    tag VARCHAR(100) NOT NULL,
    value TEXT,
    currency VARCHAR(10),
    model_code VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account, tag, timestamp)
);

-- IBKR Contract Details (complete contract specifications)
CREATE TABLE IF NOT EXISTS ibkr_contract_details (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    sec_type VARCHAR(20),
    con_id BIGINT,
    exchange VARCHAR(20),
    primary_exchange VARCHAR(20),
    currency VARCHAR(10),
    local_symbol VARCHAR(50),
    trading_class VARCHAR(20),
    market_name VARCHAR(50),
    min_tick DECIMAL(10,8),
    price_magnifier INTEGER,
    under_con_id BIGINT,
    long_name TEXT,
    contract_month VARCHAR(10),
    industry VARCHAR(100),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    time_zone_id VARCHAR(50),
    trading_hours TEXT,
    liquid_hours TEXT,
    ev_rule VARCHAR(50),
    ev_multiplier INTEGER,
    md_size_multiplier INTEGER,
    agg_group INTEGER,
    under_symbol VARCHAR(20),
    under_sec_type VARCHAR(20),
    real_expiration_date VARCHAR(20),
    last_trade_time VARCHAR(50),
    stock_type VARCHAR(20),
    min_size DECIMAL(15,8),
    size_increment DECIMAL(15,8),
    suggested_size_increment DECIMAL(15,2),
    cusip VARCHAR(20),
    ratings VARCHAR(20),
    desc_append TEXT,
    bond_type VARCHAR(20),
    coupon_type VARCHAR(20),
    callable BOOLEAN,
    putable BOOLEAN,
    coupon DECIMAL(8,4),
    convertible BOOLEAN,
    maturity VARCHAR(20),
    issue_date VARCHAR(20),
    next_option_date VARCHAR(20),
    next_option_type VARCHAR(20),
    next_option_partial BOOLEAN,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, sec_type, exchange)
);

-- IBKR Contract Order Types (many-to-many: contracts support multiple order types)
CREATE TABLE IF NOT EXISTS ibkr_contract_order_types (
    id SERIAL PRIMARY KEY,
    contract_details_id INTEGER NOT NULL REFERENCES ibkr_contract_details(id),
    order_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_details_id, order_type)
);

-- IBKR Contract Valid Exchanges (many-to-many: contracts trade on multiple exchanges)
CREATE TABLE IF NOT EXISTS ibkr_contract_valid_exchanges (
    id SERIAL PRIMARY KEY,
    contract_details_id INTEGER NOT NULL REFERENCES ibkr_contract_details(id),
    exchange VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_details_id, exchange)
);

-- IBKR Contract Market Rule IDs (track market rules for each contract)
CREATE TABLE IF NOT EXISTS ibkr_contract_market_rules (
    id SERIAL PRIMARY KEY,
    contract_details_id INTEGER NOT NULL REFERENCES ibkr_contract_details(id),
    rule_id INTEGER NOT NULL,
    rule_position INTEGER NOT NULL,  -- Position in the list
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_details_id, rule_position)
);

-- IBKR Contract Security Identifiers (secIdList - critical for compliance)
CREATE TABLE IF NOT EXISTS ibkr_contract_sec_ids (
    id SERIAL PRIMARY KEY,
    contract_details_id INTEGER NOT NULL REFERENCES ibkr_contract_details(id),
    sec_id_type VARCHAR(20) NOT NULL,  -- e.g., 'ISIN', 'CUSIP', 'SEDOL'
    sec_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_details_id, sec_id_type)
);

-- IBKR Historical Bars (5-second source data)
CREATE TABLE IF NOT EXISTS ibkr_historical_5sec (
    symbol VARCHAR(20),
    date TIMESTAMP,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    average DECIMAL(12,6),
    bar_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
) PARTITION BY RANGE (date);

-- IBKR Aggregated Bars (1-minute)
CREATE TABLE IF NOT EXISTS ibkr_bars_1min (
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    vwap DECIMAL(12,6),
    bar_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- IBKR Aggregated Bars (5-minute)
CREATE TABLE IF NOT EXISTS ibkr_bars_5min (
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    vwap DECIMAL(12,6),
    bar_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- IBKR Aggregated Bars (10-minute)
CREATE TABLE IF NOT EXISTS ibkr_bars_10min (
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    vwap DECIMAL(12,6),
    bar_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- IBKR Aggregated Bars (15-minute)
CREATE TABLE IF NOT EXISTS ibkr_bars_15min (
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    vwap DECIMAL(12,6),
    bar_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- IBKR Aggregated Bars (30-minute)
CREATE TABLE IF NOT EXISTS ibkr_bars_30min (
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    vwap DECIMAL(12,6),
    bar_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- IBKR Aggregated Bars (1-hour)
CREATE TABLE IF NOT EXISTS ibkr_bars_1hour (
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    vwap DECIMAL(12,6),
    bar_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- IBKR Real-time Bars
CREATE TABLE IF NOT EXISTS ibkr_realtime_bars (
    symbol VARCHAR(20),
    time TIMESTAMP,
    end_time BIGINT,
    open_ DECIMAL(12,4),  -- Note: field name is 'open_' in IBKR
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    wap DECIMAL(12,6),
    count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, time)
);

-- IBKR Real-time Quotes/Tickers (main ticker data - no lists)
CREATE TABLE IF NOT EXISTS ibkr_tickers (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    time TIMESTAMP,
    market_data_type INTEGER,
    min_tick DECIMAL(10,8),
    bid DECIMAL(12,4),
    bid_size DECIMAL(15,2),
    bid_exchange VARCHAR(20),
    ask DECIMAL(12,4),
    ask_size DECIMAL(15,2),
    ask_exchange VARCHAR(20),
    last DECIMAL(12,4),
    last_size DECIMAL(15,2),
    last_exchange VARCHAR(20),
    prev_bid DECIMAL(12,4),
    prev_bid_size DECIMAL(15,2),
    prev_ask DECIMAL(12,4),
    prev_ask_size DECIMAL(15,2),
    prev_last DECIMAL(12,4),
    prev_last_size DECIMAL(15,2),
    volume DECIMAL(15,2),
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    vwap DECIMAL(12,6),
    low_13week DECIMAL(12,4),
    high_13week DECIMAL(12,4),
    low_26week DECIMAL(12,4),
    high_26week DECIMAL(12,4),
    low_52week DECIMAL(12,4),
    high_52week DECIMAL(12,4),
    bid_yield DECIMAL(8,4),
    ask_yield DECIMAL(8,4),
    last_yield DECIMAL(8,4),
    mark_price DECIMAL(12,4),
    halted DECIMAL(8,4),
    rt_hist_volatility DECIMAL(8,6),
    rt_volume DECIMAL(15,2),
    rt_trade_volume DECIMAL(15,2),
    av_volume DECIMAL(15,2),
    trade_count DECIMAL(12,0),
    trade_rate DECIMAL(12,4),
    volume_rate DECIMAL(12,4),
    shortable_shares DECIMAL(15,0),
    index_future_premium DECIMAL(12,4),
    futures_open_interest DECIMAL(15,0),
    put_open_interest DECIMAL(15,0),
    call_open_interest DECIMAL(15,0),
    put_volume DECIMAL(15,2),
    call_volume DECIMAL(15,2),
    av_option_volume DECIMAL(15,2),
    hist_volatility DECIMAL(8,6),
    implied_volatility DECIMAL(8,6),
    auction_volume DECIMAL(15,2),
    auction_price DECIMAL(12,4),
    auction_imbalance DECIMAL(15,2),
    regulatory_imbalance DECIMAL(15,2),
    bbo_exchange VARCHAR(20),
    snapshot_permissions INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IBKR Ticker Individual Ticks (from ticks list)
CREATE TABLE IF NOT EXISTS ibkr_ticker_ticks (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL REFERENCES ibkr_tickers(id),
    tick_type INTEGER,
    tick_value DECIMAL(12,6),
    tick_size DECIMAL(15,2),
    tick_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IBKR Tick-by-Tick Data (from tickByTicks list)
CREATE TABLE IF NOT EXISTS ibkr_tick_by_tick (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    tick_type VARCHAR(20),  -- 'Last', 'BidAsk', 'MidPoint'
    time TIMESTAMP,
    price DECIMAL(12,4),
    size DECIMAL(15,2),
    exchange VARCHAR(20),
    special_conditions VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IBKR Depth of Market - Bids (from domBids list)
CREATE TABLE IF NOT EXISTS ibkr_dom_bids (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP,
    price DECIMAL(12,4),
    size DECIMAL(15,2),
    market_maker VARCHAR(10),
    level INTEGER,  -- DOM level (0 = best bid, 1 = second best, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IBKR Depth of Market - Asks (from domAsks list)
CREATE TABLE IF NOT EXISTS ibkr_dom_asks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP,
    price DECIMAL(12,4),
    size DECIMAL(15,2),
    market_maker VARCHAR(10),
    level INTEGER,  -- DOM level (0 = best ask, 1 = second best, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IBKR DOM Ticks (from domTicks list - depth of market tick updates)
CREATE TABLE IF NOT EXISTS ibkr_dom_ticks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP,
    side VARCHAR(10),  -- 'BID' or 'ASK'
    level INTEGER,
    price DECIMAL(12,4),
    size DECIMAL(15,2),
    operation VARCHAR(20),  -- 'INSERT', 'UPDATE', 'DELETE'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IBKR Dividend Data (from dividends field when populated)
CREATE TABLE IF NOT EXISTS ibkr_dividends (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    ex_date DATE,
    pay_date DATE,
    amount DECIMAL(8,4),
    currency VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, ex_date)
);

-- IBKR Fundamental Ratios (from fundamentalRatios field when populated)
CREATE TABLE IF NOT EXISTS ibkr_fundamental_ratios (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP,
    pe_ratio DECIMAL(8,2),
    pb_ratio DECIMAL(8,2),
    price_to_sales DECIMAL(8,2),
    price_to_cash_flow DECIMAL(8,2),
    enterprise_value_to_ebitda DECIMAL(8,2),
    debt_to_equity DECIMAL(8,2),
    return_on_equity DECIMAL(8,4),
    return_on_assets DECIMAL(8,4),
    profit_margin DECIMAL(8,4),
    operating_margin DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

-- ============================================================================
-- ALPHA VANTAGE OPTIONS TABLES
-- ============================================================================

-- Alpha Vantage Real-time Options (PRIMARY GREEKS SOURCE)
CREATE TABLE IF NOT EXISTS av_realtime_options (
    id SERIAL PRIMARY KEY,
    contract_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    expiration DATE NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    type VARCHAR(10) NOT NULL,  -- 'call' or 'put'
    last DECIMAL(12,4),
    mark DECIMAL(12,4),
    bid DECIMAL(12,4),
    bid_size INTEGER,
    ask DECIMAL(12,4),
    ask_size INTEGER,
    volume BIGINT,
    open_interest BIGINT,
    date DATE,
    implied_volatility DECIMAL(8,6),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_id, date)
);

-- Alpha Vantage Historical Options
CREATE TABLE IF NOT EXISTS av_historical_options (
    id SERIAL PRIMARY KEY,
    contract_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    expiration DATE NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    type VARCHAR(10) NOT NULL,
    last DECIMAL(12,4),
    mark DECIMAL(12,4),
    bid DECIMAL(12,4),
    bid_size INTEGER,
    ask DECIMAL(12,4),
    ask_size INTEGER,
    volume BIGINT,
    open_interest BIGINT,
    date DATE,
    implied_volatility DECIMAL(8,6),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_id, date)
);

-- ============================================================================
-- ALPHA VANTAGE TECHNICAL INDICATORS (16 TABLES)
-- ============================================================================

-- RSI - Relative Strength Index
CREATE TABLE IF NOT EXISTS av_rsi (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    rsi DECIMAL(8,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    series_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- MACD - Moving Average Convergence Divergence
CREATE TABLE IF NOT EXISTS av_macd (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    macd DECIMAL(12,6),
    macd_signal DECIMAL(12,6),
    macd_hist DECIMAL(12,6),
    interval VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval)
);

-- BBANDS - Bollinger Bands
CREATE TABLE IF NOT EXISTS av_bbands (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    real_upper_band DECIMAL(12,4),
    real_middle_band DECIMAL(12,4),
    real_lower_band DECIMAL(12,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- ATR - Average True Range
CREATE TABLE IF NOT EXISTS av_atr (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    atr DECIMAL(12,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- ADX - Average Directional Index
CREATE TABLE IF NOT EXISTS av_adx (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    adx DECIMAL(8,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- AROON - Aroon Indicator
CREATE TABLE IF NOT EXISTS av_aroon (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    aroon_down DECIMAL(8,4),
    aroon_up DECIMAL(8,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- CCI - Commodity Channel Index
CREATE TABLE IF NOT EXISTS av_cci (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    cci DECIMAL(12,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- EMA - Exponential Moving Average
CREATE TABLE IF NOT EXISTS av_ema (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    ema DECIMAL(12,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    series_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- SMA - Simple Moving Average
CREATE TABLE IF NOT EXISTS av_sma (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    sma DECIMAL(12,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    series_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- MFI - Money Flow Index
CREATE TABLE IF NOT EXISTS av_mfi (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    mfi DECIMAL(8,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- MOM - Momentum
CREATE TABLE IF NOT EXISTS av_mom (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    mom DECIMAL(12,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    series_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- OBV - On Balance Volume
CREATE TABLE IF NOT EXISTS av_obv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    obv DECIMAL(20,4),  -- Large values like 17825831983.0000
    interval VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval)
);

-- AD - Chaikin A/D Line
CREATE TABLE IF NOT EXISTS av_ad (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    chaikin_a_d DECIMAL(20,4),  -- Large values like 43676276696.5820
    interval VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval)
);

-- VWAP - Volume Weighted Average Price
CREATE TABLE IF NOT EXISTS av_vwap (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    vwap DECIMAL(12,4),
    interval VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval)
);

-- WILLR - Williams %R
CREATE TABLE IF NOT EXISTS av_willr (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    willr DECIMAL(8,4),
    interval VARCHAR(20) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);

-- STOCH - Stochastic Oscillator
CREATE TABLE IF NOT EXISTS av_stoch (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    slowk DECIMAL(8,4),
    slowd DECIMAL(8,4),
    interval VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval)
);

-- ============================================================================
-- ALPHA VANTAGE ANALYTICS TABLES
-- ============================================================================

-- Analytics Fixed Window - Basic Statistics
CREATE TABLE IF NOT EXISTS av_analytics_basic_stats (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,  -- Link related calculations
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    ohlc VARCHAR(20) NOT NULL,
    range_param VARCHAR(50) NOT NULL,
    min_value DECIMAL(15,10),
    max_value DECIMAL(15,10),
    mean_value DECIMAL(15,10),
    median_value DECIMAL(15,10),
    cumulative_return DECIMAL(15,10),
    variance_value DECIMAL(15,10),
    stddev_value DECIMAL(15,10),
    autocorrelation DECIMAL(15,10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol)
);

-- Analytics Fixed Window - Max Drawdown (complex nested object)
CREATE TABLE IF NOT EXISTS av_analytics_max_drawdown (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    max_drawdown DECIMAL(15,10),
    start_drawdown DATE,
    end_drawdown DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol)
);

-- Analytics Fixed Window - Histogram (arrays)
CREATE TABLE IF NOT EXISTS av_analytics_histogram (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    bin_edges DECIMAL(8,3)[],  -- Array of bin edges like [-0.025, -0.02, ...]
    bin_count INTEGER[],       -- Array of counts like [0, 1, 1, 4, 6, ...]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol)
);

-- Analytics Fixed Window - Covariance Matrix
CREATE TABLE IF NOT EXISTS av_analytics_covariance (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    symbol_1 VARCHAR(20) NOT NULL,
    symbol_2 VARCHAR(20) NOT NULL,
    covariance_value DECIMAL(15,12),  -- Very precise for financial calculations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol_1, symbol_2)
);

-- Analytics Fixed Window - Correlation Matrix
CREATE TABLE IF NOT EXISTS av_analytics_correlation (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    symbol_1 VARCHAR(20) NOT NULL,
    symbol_2 VARCHAR(20) NOT NULL,
    correlation_value DECIMAL(10,8),  -- Range -1 to 1 with high precision
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol_1, symbol_2)
);

-- Analytics Sliding Window - Running Statistics
CREATE TABLE IF NOT EXISTS av_analytics_sliding_window (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    ohlc VARCHAR(20) NOT NULL,
    range_param VARCHAR(50) NOT NULL,
    window_size INTEGER NOT NULL,
    window_start_date DATE,
    running_mean DECIMAL(15,10),
    running_median DECIMAL(15,10),
    running_cumulative_return DECIMAL(15,10),
    running_variance DECIMAL(15,10),
    running_stddev DECIMAL(15,10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol, window_start_date)
);

-- Analytics Sliding Window - Running Covariance
CREATE TABLE IF NOT EXISTS av_analytics_sliding_covariance (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    symbol_1 VARCHAR(20) NOT NULL,
    symbol_2 VARCHAR(20) NOT NULL,
    window_start_date DATE,
    running_covariance DECIMAL(15,12),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol_1, symbol_2, window_start_date)
);

-- Analytics Sliding Window - Running Correlation
CREATE TABLE IF NOT EXISTS av_analytics_sliding_correlation (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    symbol_1 VARCHAR(20) NOT NULL,
    symbol_2 VARCHAR(20) NOT NULL,
    window_start_date DATE,
    running_correlation DECIMAL(10,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(request_id, symbol_1, symbol_2, window_start_date)
);

-- Analytics Request Metadata (track parameters for each request)
CREATE TABLE IF NOT EXISTS av_analytics_requests (
    request_id UUID PRIMARY KEY,
    symbols TEXT NOT NULL,  -- Original comma-separated symbols
    interval VARCHAR(20) NOT NULL,
    ohlc VARCHAR(20) NOT NULL,
    range_param VARCHAR(50) NOT NULL,
    calculations TEXT NOT NULL,  -- What was requested
    window_size INTEGER,  -- NULL for fixed window
    request_type VARCHAR(20) NOT NULL,  -- 'fixed_window' or 'sliding_window'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ALPHA VANTAGE SENTIMENT & NEWS TABLES
-- ============================================================================

-- News Articles (main article data)
CREATE TABLE IF NOT EXISTS av_news_articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    time_published VARCHAR(50) NOT NULL,
    summary TEXT,
    banner_image TEXT,
    source VARCHAR(100),
    category_within_source VARCHAR(100),
    source_domain VARCHAR(100),
    overall_sentiment_score DECIMAL(8,6),
    overall_sentiment_label VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- News Article Authors (many-to-many: articles can have multiple authors)
CREATE TABLE IF NOT EXISTS av_news_authors (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES av_news_articles(id),
    author_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_id, author_name)
);

-- News Article Topics (many-to-many: articles can have multiple topics)
CREATE TABLE IF NOT EXISTS av_news_topics (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES av_news_articles(id),
    topic VARCHAR(100) NOT NULL,
    relevance_score DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_id, topic)
);

-- News Ticker Sentiment (many-to-many: articles can mention multiple tickers)
CREATE TABLE IF NOT EXISTS av_news_ticker_sentiment (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES av_news_articles(id),
    ticker VARCHAR(20) NOT NULL,
    relevance_score DECIMAL(8,6),
    ticker_sentiment_score DECIMAL(8,6),
    ticker_sentiment_label VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_id, ticker)
);

-- Top Gainers/Losers/Most Active
CREATE TABLE IF NOT EXISTS av_top_gainers_losers (
    id SERIAL PRIMARY KEY,
    list_type VARCHAR(50) NOT NULL,  -- 'top_gainers', 'top_losers', 'most_actively_traded'
    ticker VARCHAR(20) NOT NULL,
    price DECIMAL(12,4),
    change_amount DECIMAL(12,4),
    change_percentage VARCHAR(20),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insider Transactions
CREATE TABLE IF NOT EXISTS av_insider_transactions (
    id SERIAL PRIMARY KEY,
    transaction_date DATE,
    ticker VARCHAR(20),
    executive TEXT,
    executive_title TEXT,
    security_type VARCHAR(100),
    acquisition_or_disposal VARCHAR(10),
    shares DECIMAL(15,2),
    share_price DECIMAL(12,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ALPHA VANTAGE FUNDAMENTAL DATA TABLES
-- ============================================================================

-- Company Overview (55 fields)
CREATE TABLE IF NOT EXISTS av_company_overview (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(50),
    name TEXT,
    description TEXT,
    cik VARCHAR(20),
    exchange VARCHAR(20),
    currency VARCHAR(10),
    country VARCHAR(10),
    sector VARCHAR(100),
    industry VARCHAR(100),
    address TEXT,
    official_site TEXT,
    fiscal_year_end VARCHAR(20),
    latest_quarter DATE,
    market_capitalization BIGINT,
    ebitda BIGINT,
    pe_ratio DECIMAL(8,2),
    peg_ratio DECIMAL(8,2),
    book_value DECIMAL(8,3),
    dividend_per_share DECIMAL(8,2),
    dividend_yield DECIMAL(8,4),
    eps DECIMAL(8,2),
    revenue_per_share_ttm DECIMAL(8,2),
    profit_margin DECIMAL(8,3),
    operating_margin_ttm DECIMAL(8,3),
    return_on_assets_ttm DECIMAL(8,3),
    return_on_equity_ttm DECIMAL(8,3),
    revenue_ttm BIGINT,
    gross_profit_ttm BIGINT,
    diluted_eps_ttm DECIMAL(8,2),
    quarterly_earnings_growth_yoy DECIMAL(8,3),
    quarterly_revenue_growth_yoy DECIMAL(8,3),
    analyst_target_price DECIMAL(8,2),
    analyst_rating_strong_buy INTEGER,
    analyst_rating_buy INTEGER,
    analyst_rating_hold INTEGER,
    analyst_rating_sell INTEGER,
    analyst_rating_strong_sell INTEGER,
    trailing_pe DECIMAL(8,2),
    forward_pe DECIMAL(8,2),
    price_to_sales_ratio_ttm DECIMAL(8,2),
    price_to_book_ratio DECIMAL(8,2),
    ev_to_revenue DECIMAL(8,2),
    ev_to_ebitda DECIMAL(8,2),
    beta DECIMAL(8,3),
    week_52_high DECIMAL(8,2),
    week_52_low DECIMAL(8,2),
    day_50_moving_average DECIMAL(8,2),
    day_200_moving_average DECIMAL(8,2),
    shares_outstanding BIGINT,
    shares_float BIGINT,
    percent_insiders DECIMAL(8,3),
    percent_institutions DECIMAL(8,3),
    dividend_date DATE,
    ex_dividend_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol)
);

-- Balance Sheet
CREATE TABLE IF NOT EXISTS av_balance_sheet (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    fiscal_date_ending DATE,
    reported_currency VARCHAR(10),
    total_assets BIGINT,
    total_current_assets BIGINT,
    cash_and_cash_equivalents_at_carrying_value BIGINT,
    cash_and_short_term_investments BIGINT,
    inventory BIGINT,
    current_net_receivables BIGINT,
    total_non_current_assets BIGINT,
    property_plant_equipment BIGINT,
    accumulated_depreciation_amortization_ppe BIGINT,
    intangible_assets BIGINT,
    intangible_assets_excluding_goodwill BIGINT,
    goodwill BIGINT,
    investments BIGINT,
    long_term_investments BIGINT,
    short_term_investments BIGINT,
    other_current_assets BIGINT,
    other_non_current_assets BIGINT,
    total_liabilities BIGINT,
    total_current_liabilities BIGINT,
    current_accounts_payable BIGINT,
    deferred_revenue BIGINT,
    current_debt BIGINT,
    short_term_debt BIGINT,
    total_non_current_liabilities BIGINT,
    capital_lease_obligations BIGINT,
    long_term_debt BIGINT,
    current_long_term_debt BIGINT,
    long_term_debt_noncurrent BIGINT,
    short_long_term_debt_total BIGINT,
    other_current_liabilities BIGINT,
    other_non_current_liabilities BIGINT,
    total_shareholder_equity BIGINT,
    treasury_stock BIGINT,
    retained_earnings BIGINT,
    common_stock BIGINT,
    common_stock_shares_outstanding BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, fiscal_date_ending)
);

-- Income Statement
CREATE TABLE IF NOT EXISTS av_income_statement (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    fiscal_date_ending DATE,
    reported_currency VARCHAR(10),
    gross_profit BIGINT,
    total_revenue BIGINT,
    cost_of_revenue BIGINT,
    cost_of_goods_and_services_sold BIGINT,
    operating_income BIGINT,
    selling_general_and_administrative BIGINT,
    research_and_development BIGINT,
    operating_expenses BIGINT,
    investment_income_net BIGINT,
    net_interest_income BIGINT,
    interest_income BIGINT,
    interest_expense BIGINT,
    non_interest_income BIGINT,
    other_non_operating_income BIGINT,
    depreciation BIGINT,
    depreciation_and_amortization BIGINT,
    income_before_tax BIGINT,
    income_tax_expense BIGINT,
    interest_and_debt_expense BIGINT,
    net_income_from_continuing_operations BIGINT,
    comprehensive_income_net_of_tax BIGINT,
    ebit BIGINT,
    ebitda BIGINT,
    net_income BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, fiscal_date_ending)
);

-- Cash Flow
CREATE TABLE IF NOT EXISTS av_cash_flow (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    fiscal_date_ending DATE,
    reported_currency VARCHAR(10),
    operating_cashflow BIGINT,
    payments_for_operating_activities BIGINT,
    proceeds_from_operating_activities BIGINT,
    change_in_operating_liabilities BIGINT,
    change_in_operating_assets BIGINT,
    depreciation_depletion_and_amortization BIGINT,
    capital_expenditures BIGINT,
    change_in_receivables BIGINT,
    change_in_inventory BIGINT,
    profit_loss BIGINT,
    cashflow_from_investment BIGINT,
    cashflow_from_financing BIGINT,
    proceeds_from_repayments_of_short_term_debt BIGINT,
    payments_for_repurchase_of_common_stock BIGINT,
    payments_for_repurchase_of_equity BIGINT,
    payments_for_repurchase_of_preferred_stock BIGINT,
    dividend_payout BIGINT,
    dividend_payout_common_stock BIGINT,
    dividend_payout_preferred_stock BIGINT,
    proceeds_from_issuance_of_common_stock BIGINT,
    proceeds_from_issuance_of_long_term_debt_and_capital_securities_net BIGINT,
    proceeds_from_issuance_of_preferred_stock BIGINT,
    proceeds_from_repurchase_of_equity BIGINT,
    proceeds_from_sale_of_treasury_stock BIGINT,
    change_in_cash_and_cash_equivalents BIGINT,
    change_in_exchange_rate BIGINT,
    net_income BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, fiscal_date_ending)
);

-- Dividends
CREATE TABLE IF NOT EXISTS av_dividends (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    ex_dividend_date DATE,
    declaration_date DATE,
    record_date DATE,
    payment_date DATE,
    amount DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, ex_dividend_date)
);

-- Stock Splits
CREATE TABLE IF NOT EXISTS av_splits (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    effective_date DATE,
    split_factor DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, effective_date)
);

-- ============================================================================
-- ALPHA VANTAGE ECONOMIC INDICATORS TABLES
-- ============================================================================

-- CPI - Consumer Price Index
CREATE TABLE IF NOT EXISTS av_cpi (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    value DECIMAL(12,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Federal Funds Rate
CREATE TABLE IF NOT EXISTS av_federal_funds_rate (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    value DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Inflation
CREATE TABLE IF NOT EXISTS av_inflation (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    value DECIMAL(12,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Real GDP
CREATE TABLE IF NOT EXISTS av_real_gdp (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    value DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Treasury Yield
CREATE TABLE IF NOT EXISTS av_treasury_yield (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    value DECIMAL(8,4),
    maturity VARCHAR(20) NOT NULL,  -- '3month', '2year', '10year', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, maturity)
);

-- ============================================================================
-- PERFORMANCE INDEXES
-- ============================================================================

-- IBKR Indexes
CREATE INDEX IF NOT EXISTS idx_ibkr_historical_5sec_symbol_date 
    ON ibkr_historical_5sec(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_bars_1min_symbol_timestamp 
    ON ibkr_bars_1min(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_bars_5min_symbol_timestamp 
    ON ibkr_bars_5min(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_bars_10min_symbol_timestamp 
    ON ibkr_bars_10min(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_bars_15min_symbol_timestamp 
    ON ibkr_bars_15min(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_bars_30min_symbol_timestamp 
    ON ibkr_bars_30min(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_bars_1hour_symbol_timestamp 
    ON ibkr_bars_1hour(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_realtime_bars_symbol_time 
    ON ibkr_realtime_bars(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_tickers_symbol_time 
    ON ibkr_tickers(symbol, time DESC);

-- IBKR Contract Details Indexes
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_details_symbol 
    ON ibkr_contract_details(symbol);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_details_con_id 
    ON ibkr_contract_details(con_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_order_types_contract_id 
    ON ibkr_contract_order_types(contract_details_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_order_types_order_type 
    ON ibkr_contract_order_types(order_type, contract_details_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_valid_exchanges_contract_id 
    ON ibkr_contract_valid_exchanges(contract_details_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_valid_exchanges_exchange 
    ON ibkr_contract_valid_exchanges(exchange, contract_details_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_market_rules_contract_id 
    ON ibkr_contract_market_rules(contract_details_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_sec_ids_contract_id 
    ON ibkr_contract_sec_ids(contract_details_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_sec_ids_type_id 
    ON ibkr_contract_sec_ids(sec_id_type, sec_id);

-- IBKR Market Microstructure Indexes
CREATE INDEX IF NOT EXISTS idx_ibkr_ticker_ticks_ticker_id 
    ON ibkr_ticker_ticks(ticker_id);
CREATE INDEX IF NOT EXISTS idx_ibkr_ticker_ticks_type_time 
    ON ibkr_ticker_ticks(tick_type, tick_time DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_tick_by_tick_symbol_time 
    ON ibkr_tick_by_tick(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_tick_by_tick_type_symbol 
    ON ibkr_tick_by_tick(tick_type, symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_dom_bids_symbol_timestamp 
    ON ibkr_dom_bids(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_dom_bids_symbol_level 
    ON ibkr_dom_bids(symbol, level, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_dom_asks_symbol_timestamp 
    ON ibkr_dom_asks(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_dom_asks_symbol_level 
    ON ibkr_dom_asks(symbol, level, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_dom_ticks_symbol_timestamp 
    ON ibkr_dom_ticks(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_dom_ticks_symbol_side_level 
    ON ibkr_dom_ticks(symbol, side, level, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_dividends_symbol_ex_date 
    ON ibkr_dividends(symbol, ex_date DESC);
CREATE INDEX IF NOT EXISTS idx_ibkr_fundamental_ratios_symbol_timestamp 
    ON ibkr_fundamental_ratios(symbol, timestamp DESC);

-- Alpha Vantage Options Indexes
CREATE INDEX IF NOT EXISTS idx_av_realtime_options_symbol_expiration 
    ON av_realtime_options(symbol, expiration, date DESC);
CREATE INDEX IF NOT EXISTS idx_av_realtime_options_contract_date 
    ON av_realtime_options(contract_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_av_historical_options_symbol_expiration 
    ON av_historical_options(symbol, expiration, date DESC);

-- Technical Indicators Indexes
CREATE INDEX IF NOT EXISTS idx_av_rsi_symbol_timestamp 
    ON av_rsi(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_macd_symbol_timestamp 
    ON av_macd(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_bbands_symbol_timestamp 
    ON av_bbands(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_atr_symbol_timestamp 
    ON av_atr(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_adx_symbol_timestamp 
    ON av_adx(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_aroon_symbol_timestamp 
    ON av_aroon(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_cci_symbol_timestamp 
    ON av_cci(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_ema_symbol_timestamp 
    ON av_ema(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_sma_symbol_timestamp 
    ON av_sma(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_mfi_symbol_timestamp 
    ON av_mfi(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_mom_symbol_timestamp 
    ON av_mom(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_obv_symbol_timestamp 
    ON av_obv(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_ad_symbol_timestamp 
    ON av_ad(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_vwap_symbol_timestamp 
    ON av_vwap(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_willr_symbol_timestamp 
    ON av_willr(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_av_stoch_symbol_timestamp 
    ON av_stoch(symbol, timestamp DESC);

-- Company Data Indexes
CREATE INDEX IF NOT EXISTS idx_av_company_overview_symbol 
    ON av_company_overview(symbol);
CREATE INDEX IF NOT EXISTS idx_av_balance_sheet_symbol_date 
    ON av_balance_sheet(symbol, fiscal_date_ending DESC);
CREATE INDEX IF NOT EXISTS idx_av_income_statement_symbol_date 
    ON av_income_statement(symbol, fiscal_date_ending DESC);
CREATE INDEX IF NOT EXISTS idx_av_cash_flow_symbol_date 
    ON av_cash_flow(symbol, fiscal_date_ending DESC);
CREATE INDEX IF NOT EXISTS idx_av_dividends_symbol_date 
    ON av_dividends(symbol, ex_dividend_date DESC);

-- Economic Data Indexes
CREATE INDEX IF NOT EXISTS idx_av_cpi_date 
    ON av_cpi(date DESC);
CREATE INDEX IF NOT EXISTS idx_av_federal_funds_rate_date 
    ON av_federal_funds_rate(date DESC);
CREATE INDEX IF NOT EXISTS idx_av_inflation_date 
    ON av_inflation(date DESC);
CREATE INDEX IF NOT EXISTS idx_av_real_gdp_date 
    ON av_real_gdp(date DESC);
CREATE INDEX IF NOT EXISTS idx_av_treasury_yield_maturity_date 
    ON av_treasury_yield(maturity, date DESC);

-- Analytics Indexes
CREATE INDEX IF NOT EXISTS idx_av_analytics_basic_stats_request_symbol 
    ON av_analytics_basic_stats(request_id, symbol);
CREATE INDEX IF NOT EXISTS idx_av_analytics_basic_stats_symbol_created 
    ON av_analytics_basic_stats(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_av_analytics_max_drawdown_request_symbol 
    ON av_analytics_max_drawdown(request_id, symbol);
CREATE INDEX IF NOT EXISTS idx_av_analytics_histogram_request_symbol 
    ON av_analytics_histogram(request_id, symbol);
CREATE INDEX IF NOT EXISTS idx_av_analytics_covariance_request_symbols 
    ON av_analytics_covariance(request_id, symbol_1, symbol_2);
CREATE INDEX IF NOT EXISTS idx_av_analytics_correlation_request_symbols 
    ON av_analytics_correlation(request_id, symbol_1, symbol_2);
CREATE INDEX IF NOT EXISTS idx_av_analytics_sliding_window_request_symbol 
    ON av_analytics_sliding_window(request_id, symbol, window_start_date);
CREATE INDEX IF NOT EXISTS idx_av_analytics_sliding_covariance_request_symbols 
    ON av_analytics_sliding_covariance(request_id, symbol_1, symbol_2, window_start_date);
CREATE INDEX IF NOT EXISTS idx_av_analytics_sliding_correlation_request_symbols 
    ON av_analytics_sliding_correlation(request_id, symbol_1, symbol_2, window_start_date);
CREATE INDEX IF NOT EXISTS idx_av_analytics_requests_created 
    ON av_analytics_requests(created_at DESC);

-- News and Sentiment Indexes
CREATE INDEX IF NOT EXISTS idx_av_news_articles_time_published 
    ON av_news_articles(time_published DESC);
CREATE INDEX IF NOT EXISTS idx_av_news_articles_source 
    ON av_news_articles(source, time_published DESC);
CREATE INDEX IF NOT EXISTS idx_av_news_articles_sentiment_score 
    ON av_news_articles(overall_sentiment_score DESC, time_published DESC);
CREATE INDEX IF NOT EXISTS idx_av_news_articles_url 
    ON av_news_articles(url);

CREATE INDEX IF NOT EXISTS idx_av_news_authors_article 
    ON av_news_authors(article_id);
CREATE INDEX IF NOT EXISTS idx_av_news_authors_name 
    ON av_news_authors(author_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_av_news_topics_article 
    ON av_news_topics(article_id);
CREATE INDEX IF NOT EXISTS idx_av_news_topics_topic_relevance 
    ON av_news_topics(topic, relevance_score DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_av_news_ticker_sentiment_article 
    ON av_news_ticker_sentiment(article_id);
CREATE INDEX IF NOT EXISTS idx_av_news_ticker_sentiment_ticker_score 
    ON av_news_ticker_sentiment(ticker, ticker_sentiment_score DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_av_news_ticker_sentiment_ticker_time 
    ON av_news_ticker_sentiment(ticker, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_av_news_ticker_sentiment_relevance 
    ON av_news_ticker_sentiment(ticker, relevance_score DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_av_top_gainers_losers_type_created 
    ON av_top_gainers_losers(list_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_av_insider_transactions_ticker_date 
    ON av_insider_transactions(ticker, transaction_date DESC);

-- ============================================================================
-- PARTITION MAINTENANCE (for time-series tables)
-- ============================================================================

-- Example partition creation for IBKR 5-second bars (monthly partitions)
-- This would typically be automated, but showing the pattern:

-- CREATE TABLE ibkr_historical_5sec_202508 PARTITION OF ibkr_historical_5sec
--     FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

-- CREATE TABLE ibkr_historical_5sec_202509 PARTITION OF ibkr_historical_5sec
--     FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- ============================================================================
-- COMMENTS FOR CRITICAL FIELDS
-- ============================================================================

COMMENT ON TABLE av_realtime_options IS 'PRIMARY GREEKS SOURCE - Real-time options with Greeks from Alpha Vantage';
COMMENT ON TABLE ibkr_historical_5sec IS 'ONLY timeframe from IBKR - all others aggregated from this';
COMMENT ON COLUMN ibkr_realtime_bars.open_ IS 'Field name is open_ (with underscore) in IBKR API';
COMMENT ON COLUMN av_obv.obv IS 'Large values like 17825831983.0000 - use DECIMAL(20,4)';
COMMENT ON COLUMN av_ad.chaikin_a_d IS 'Large values like 43676276696.5820 - use DECIMAL(20,4)';
COMMENT ON TABLE av_news_ticker_sentiment IS 'Critical for sentiment analysis - ticker-specific sentiment scores from news articles';
COMMENT ON TABLE av_analytics_correlation IS 'Correlation matrix data for portfolio optimization and risk management';
COMMENT ON TABLE ibkr_contract_order_types IS 'Order type capabilities per contract - critical for algorithm execution';
COMMENT ON TABLE ibkr_contract_valid_exchanges IS 'Exchange routing options - critical for best execution';
COMMENT ON TABLE ibkr_contract_sec_ids IS 'Security identifiers (ISIN, CUSIP, etc.) - critical for regulatory compliance';
COMMENT ON TABLE ibkr_dom_bids IS 'Depth of market bids - critical for microstructure analysis and optimal execution';
COMMENT ON TABLE ibkr_dom_asks IS 'Depth of market asks - critical for microstructure analysis and optimal execution';
COMMENT ON TABLE ibkr_tick_by_tick IS 'Tick-by-tick trade data - highest resolution market data for execution analysis';
COMMENT ON TABLE ibkr_dom_ticks IS 'Real-time depth of market updates - critical for order book analysis';

-- ============================================================================
-- END OF COMPLETE SCHEMA
-- All tables, indexes, and constraints based on actual API responses
-- Every field mapped exactly as returned by Alpha Vantage and IBKR APIs
-- 
-- CRITICAL TRADING SYSTEM QUERIES NOW POSSIBLE:
--
-- Order Execution Optimization:
-- SELECT order_type FROM ibkr_contract_order_types 
-- WHERE contract_details_id = (SELECT id FROM ibkr_contract_details WHERE symbol = 'SPY')
-- AND order_type IN ('ALGO', 'SMARTSTG', 'ADAPTIVE');
--
-- Best Execution Analysis:
-- SELECT exchange FROM ibkr_contract_valid_exchanges 
-- WHERE contract_details_id = (SELECT id FROM ibkr_contract_details WHERE symbol = 'AAPL')
-- ORDER BY exchange;
--
-- Market Microstructure Analysis:
-- SELECT level, price, size FROM ibkr_dom_bids 
-- WHERE symbol = 'SPY' AND timestamp >= NOW() - INTERVAL '1 minute'
-- ORDER BY level;
--
-- Sentiment-Driven Trading:
-- SELECT AVG(ticker_sentiment_score) as avg_sentiment
-- FROM av_news_ticker_sentiment 
-- WHERE ticker = 'TSLA' AND created_at >= NOW() - INTERVAL '24 hours';
--
-- Risk Management:
-- SELECT symbol_1, symbol_2, correlation_value 
-- FROM av_analytics_correlation 
-- WHERE symbol_1 = 'SPY' AND correlation_value > 0.8
-- ORDER BY correlation_value DESC;
--
-- Regulatory Compliance:
-- SELECT sec_id_type, sec_id FROM ibkr_contract_sec_ids 
-- WHERE contract_details_id = (SELECT id FROM ibkr_contract_details WHERE symbol = 'AAPL');
-- ============================================================================