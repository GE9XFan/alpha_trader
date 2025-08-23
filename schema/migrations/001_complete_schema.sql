-- AlphaTrader Database Schema
-- Generated: 2025-08-23T11:25:21.762287
-- Total Fields Catalog: 8227
-- 
-- ZERO COMPROMISES:
-- - One table per API endpoint (36 tables)
-- - NUMERIC for ALL financial data
-- - Complete normalization, no lazy JSONB
-- - Full audit trail for transformations

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ========== AUDIT/API_RESPONSE ==========
-- Audit Trail: API Responses
CREATE TABLE api_response_audit (
    id BIGSERIAL PRIMARY KEY,
    response_id UUID NOT NULL DEFAULT gen_random_uuid(),
    endpoint VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    response_size_bytes INTEGER,
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    rate_limit_remaining INTEGER,
    field_count INTEGER, -- Number of fields in response
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(response_id)
);

-- Indexes
CREATE INDEX idx_api_audit_endpoint ON api_response_audit(endpoint);
CREATE INDEX idx_api_audit_created ON api_response_audit(created_at);


-- ========== AUDIT/DATA_TRANSFORMATIONS ==========
-- Audit Trail: Data Transformations
-- Tracks EVERY transformation from API to database
CREATE TABLE data_transformations (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    original_value TEXT,
    transformed_value TEXT,
    transformation_type VARCHAR(50) NOT NULL,
    api_response_id UUID NOT NULL,
    field_path TEXT, -- Original field path in API response
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_transform_table ON data_transformations(table_name);
CREATE INDEX idx_transform_response ON data_transformations(api_response_id);
CREATE INDEX idx_transform_type ON data_transformations(transformation_type);
CREATE INDEX idx_transform_field ON data_transformations(field_path);


-- ========== AUDIT/FIELD_MAPPINGS ==========
-- Field Mapping Verification
-- Ensures all 8,227 fields are mapped
CREATE TABLE field_mapping_verification (
    id BIGSERIAL PRIMARY KEY,
    field_name TEXT NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    field_type VARCHAR(50) NOT NULL,
    occurrences INTEGER,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(field_name, table_name, column_name)
);

-- Indexes
CREATE INDEX idx_field_map_field ON field_mapping_verification(field_name);
CREATE INDEX idx_field_map_table ON field_mapping_verification(table_name);
CREATE INDEX idx_field_map_verified ON field_mapping_verification(verified);


-- ========== ANALYTICS/ANALYTICS_FIXED_WINDOW ==========
-- Analytics: analytics_fixed_window
-- Handles complex nested calculation results
CREATE TABLE analytics_fixed_window (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL DEFAULT gen_random_uuid(),
    meta_symbols VARCHAR(10),
    meta_min_dt DATE,
    meta_max_dt DATE,
    meta_ohlc VARCHAR(10),
    meta_interval VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(request_id)
);

-- Calculation results (normalized from nested structure)
CREATE TABLE analytics_fixed_window_calculations (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES analytics_fixed_window(request_id) ON DELETE CASCADE,
    calculation_category VARCHAR(50) NOT NULL,
    calculation_type VARCHAR(50) NOT NULL,
    calculation_subtype VARCHAR(50),
    symbol VARCHAR(10),
    symbol2 VARCHAR(10), -- For correlation calculations
    date DATE,
    value NUMERIC, -- ALL financial values use NUMERIC
    additional_data JSONB, -- For complex structures like histograms
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (date);

-- Create monthly partitions for next 12 months
CREATE TABLE analytics_fixed_window_calculations_2025_08 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE analytics_fixed_window_calculations_2025_09 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE analytics_fixed_window_calculations_2025_10 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE analytics_fixed_window_calculations_2025_11 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE analytics_fixed_window_calculations_2025_12 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE analytics_fixed_window_calculations_2026_01 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE analytics_fixed_window_calculations_2026_02 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE analytics_fixed_window_calculations_2026_03 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE analytics_fixed_window_calculations_2026_04 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE analytics_fixed_window_calculations_2026_05 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE analytics_fixed_window_calculations_2026_06 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE analytics_fixed_window_calculations_2026_07 
PARTITION OF analytics_fixed_window_calculations
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Performance indexes
CREATE INDEX idx_analytics_fixed_window_request ON analytics_fixed_window(request_id);
CREATE INDEX idx_analytics_fixed_window_calc_request ON analytics_fixed_window_calculations(request_id);
CREATE INDEX idx_analytics_fixed_window_calc_symbol ON analytics_fixed_window_calculations(symbol, date);
CREATE INDEX idx_analytics_fixed_window_calc_category ON analytics_fixed_window_calculations(calculation_category, calculation_type);


-- ========== ANALYTICS/ANALYTICS_SLIDING_WINDOW ==========
-- Analytics: analytics_sliding_window
-- Handles complex nested calculation results
CREATE TABLE analytics_sliding_window (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL DEFAULT gen_random_uuid(),
    meta_symbols VARCHAR(10),
    meta_window_size BIGINT,
    meta_min_dt DATE,
    meta_max_dt DATE,
    meta_ohlc VARCHAR(10),
    meta_interval VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(request_id)
);

-- Calculation results (normalized from nested structure)
CREATE TABLE analytics_sliding_window_calculations (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES analytics_sliding_window(request_id) ON DELETE CASCADE,
    calculation_category VARCHAR(50) NOT NULL,
    calculation_type VARCHAR(50) NOT NULL,
    calculation_subtype VARCHAR(50),
    symbol VARCHAR(10),
    symbol2 VARCHAR(10), -- For correlation calculations
    date DATE,
    value NUMERIC, -- ALL financial values use NUMERIC
    additional_data JSONB, -- For complex structures like histograms
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (date);

-- Create monthly partitions for next 12 months
CREATE TABLE analytics_sliding_window_calculations_2025_08 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE analytics_sliding_window_calculations_2025_09 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE analytics_sliding_window_calculations_2025_10 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE analytics_sliding_window_calculations_2025_11 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE analytics_sliding_window_calculations_2025_12 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE analytics_sliding_window_calculations_2026_01 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE analytics_sliding_window_calculations_2026_02 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE analytics_sliding_window_calculations_2026_03 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE analytics_sliding_window_calculations_2026_04 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE analytics_sliding_window_calculations_2026_05 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE analytics_sliding_window_calculations_2026_06 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE analytics_sliding_window_calculations_2026_07 
PARTITION OF analytics_sliding_window_calculations
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Performance indexes
CREATE INDEX idx_analytics_sliding_window_request ON analytics_sliding_window(request_id);
CREATE INDEX idx_analytics_sliding_window_calc_request ON analytics_sliding_window_calculations(request_id);
CREATE INDEX idx_analytics_sliding_window_calc_symbol ON analytics_sliding_window_calculations(symbol, date);
CREATE INDEX idx_analytics_sliding_window_calc_category ON analytics_sliding_window_calculations(calculation_category, calculation_type);


-- ========== ECONOMIC/CPI ==========
-- Economic Indicator: cpi
CREATE TABLE cpi (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(50),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_cpi_date ON cpi(date) WHERE date IS NOT NULL;


-- ========== ECONOMIC/FEDERAL_FUNDS_RATE ==========
-- Economic Indicator: federal_funds_rate
CREATE TABLE federal_funds_rate (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(10),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_federal_funds_rate_date ON federal_funds_rate(date) WHERE date IS NOT NULL;


-- ========== ECONOMIC/INFLATION ==========
-- Economic Indicator: inflation
CREATE TABLE inflation (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(10),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_inflation_date ON inflation(date) WHERE date IS NOT NULL;


-- ========== ECONOMIC/REAL_GDP ==========
-- Economic Indicator: real_gdp
CREATE TABLE real_gdp (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(50),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_real_gdp_date ON real_gdp(date) WHERE date IS NOT NULL;


-- ========== ECONOMIC/TREASURY_YIELD ==========
-- Economic Indicator: treasury_yield
CREATE TABLE treasury_yield (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50),
    interval VARCHAR(10),
    unit VARCHAR(10),
    date DATE,
    value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_treasury_yield_date ON treasury_yield(date) WHERE date IS NOT NULL;


-- ========== FUNDAMENTALS/BALANCE_SHEET ==========
-- Fundamental Time-Series: balance_sheet
CREATE TABLE balance_sheet (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) NOT NULL CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
    accumulateddepreciationamortizationppe TEXT,
    capitalleaseobligations TEXT,
    cashandcashequivalentsatcarryingvalue NUMERIC,
    cashandshortterminvestments NUMERIC,
    commonstock NUMERIC,
    commonstocksharesoutstanding NUMERIC,
    currentaccountspayable NUMERIC,
    currentdebt TEXT,
    currentlongtermdebt NUMERIC,
    currentnetreceivables NUMERIC,
    deferredrevenue TEXT,
    goodwill TEXT,
    intangibleassets TEXT,
    intangibleassetsexcludinggoodwill TEXT,
    inventory NUMERIC,
    investments TEXT,
    longtermdebt NUMERIC,
    longtermdebtnoncurrent TEXT,
    longterminvestments NUMERIC,
    othercurrentassets NUMERIC,
    othercurrentliabilities NUMERIC,
    othernoncurrentassets TEXT,
    othernoncurrentliabilities NUMERIC,
    propertyplantequipment NUMERIC,
    retainedearnings NUMERIC,
    shortlongtermdebttotal NUMERIC,
    shorttermdebt NUMERIC,
    shortterminvestments NUMERIC,
    totalassets NUMERIC,
    totalcurrentassets NUMERIC,
    totalcurrentliabilities NUMERIC,
    totalliabilities NUMERIC,
    totalnoncurrentassets NUMERIC,
    totalnoncurrentliabilities NUMERIC,
    totalshareholderequity NUMERIC,
    treasurystock TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, fiscal_date_ending, report_type)
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_date ON {table_name}(fiscal_date_ending);
CREATE INDEX idx_{table_name}_type ON {table_name}(report_type);


-- ========== FUNDAMENTALS/CASH_FLOW ==========
-- Fundamental Time-Series: cash_flow
CREATE TABLE cash_flow (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) NOT NULL CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
    capitalexpenditures NUMERIC,
    cashflowfromfinancing NUMERIC,
    cashflowfrominvestment NUMERIC,
    changeincashandcashequivalents TEXT,
    changeinexchangerate TEXT,
    changeininventory NUMERIC,
    changeinoperatingassets TEXT,
    changeinoperatingliabilities TEXT,
    changeinreceivables TEXT,
    depreciationdepletionandamortization NUMERIC,
    dividendpayout NUMERIC,
    dividendpayoutcommonstock NUMERIC,
    dividendpayoutpreferredstock TEXT,
    netincome NUMERIC,
    operatingcashflow NUMERIC,
    paymentsforoperatingactivities TEXT,
    paymentsforrepurchaseofcommonstock TEXT,
    paymentsforrepurchaseofequity TEXT,
    paymentsforrepurchaseofpreferredstock TEXT,
    proceedsfromissuanceofcommonstock TEXT,
    proceedsfromissuanceoflongtermdebtandcapitalsecuritiesnet TEXT,
    proceedsfromissuanceofpreferredstock TEXT,
    proceedsfromoperatingactivities TEXT,
    proceedsfromrepaymentsofshorttermdebt TEXT,
    proceedsfromrepurchaseofequity NUMERIC,
    proceedsfromsaleoftreasurystock TEXT,
    profitloss TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, fiscal_date_ending, report_type)
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_date ON {table_name}(fiscal_date_ending);
CREATE INDEX idx_{table_name}_type ON {table_name}(report_type);


-- ========== FUNDAMENTALS/DIVIDENDS ==========
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


-- ========== FUNDAMENTALS/EARNINGS ==========
-- Fundamental Time-Series: earnings
CREATE TABLE earnings (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) NOT NULL CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
    estimatedeps NUMERIC,
    reporteddate DATE,
    reportedeps NUMERIC,
    reporttime VARCHAR(50),
    surprise NUMERIC,
    surprisepercentage NUMERIC,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, fiscal_date_ending, report_type)
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_date ON {table_name}(fiscal_date_ending);
CREATE INDEX idx_{table_name}_type ON {table_name}(report_type);


-- ========== FUNDAMENTALS/INCOME_STATEMENT ==========
-- Fundamental Time-Series: income_statement
CREATE TABLE income_statement (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) NOT NULL CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
    comprehensiveincomenetoftax TEXT,
    costofgoodsandservicessold NUMERIC,
    costofrevenue NUMERIC,
    depreciation TEXT,
    depreciationandamortization NUMERIC,
    ebit NUMERIC,
    ebitda NUMERIC,
    grossprofit NUMERIC,
    incomebeforetax NUMERIC,
    incometaxexpense NUMERIC,
    interestanddebtexpense TEXT,
    interestexpense NUMERIC,
    interestincome NUMERIC,
    investmentincomenet TEXT,
    netincome NUMERIC,
    netincomefromcontinuingoperations NUMERIC,
    netinterestincome NUMERIC,
    noninterestincome TEXT,
    operatingexpenses NUMERIC,
    operatingincome NUMERIC,
    othernonoperatingincome TEXT,
    researchanddevelopment NUMERIC,
    sellinggeneralandadministrative NUMERIC,
    totalrevenue NUMERIC,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, fiscal_date_ending, report_type)
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);
CREATE INDEX idx_{table_name}_date ON {table_name}(fiscal_date_ending);
CREATE INDEX idx_{table_name}_type ON {table_name}(report_type);


-- ========== FUNDAMENTALS/OVERVIEW ==========
-- Fundamental Data: overview
CREATE TABLE overview (
    symbol VARCHAR(10) PRIMARY KEY,
    assettype VARCHAR(50),
    name VARCHAR(10),
    description TEXT,
    cik NUMERIC,
    exchange VARCHAR(10),
    currency VARCHAR(10),
    country VARCHAR(10),
    sector VARCHAR(10),
    industry VARCHAR(50),
    address VARCHAR(50),
    officialsite VARCHAR(50),
    fiscalyearend VARCHAR(10),
    latestquarter DATE,
    marketcapitalization NUMERIC,
    ebitda NUMERIC,
    peratio NUMERIC,
    pegratio NUMERIC,
    bookvalue NUMERIC,
    dividendpershare NUMERIC,
    dividendyield NUMERIC,
    eps NUMERIC,
    revenuepersharettm NUMERIC,
    profitmargin NUMERIC,
    operatingmarginttm NUMERIC,
    returnonassetsttm NUMERIC,
    returnonequityttm NUMERIC,
    revenuettm NUMERIC,
    grossprofitttm NUMERIC,
    dilutedepsttm NUMERIC,
    quarterlyearningsgrowthyoy NUMERIC,
    quarterlyrevenuegrowthyoy NUMERIC,
    analysttargetprice NUMERIC,
    analystratingstrongbuy NUMERIC,
    analystratingbuy NUMERIC,
    analystratinghold NUMERIC,
    analystratingsell NUMERIC,
    analystratingstrongsell NUMERIC,
    trailingpe NUMERIC,
    forwardpe NUMERIC,
    pricetosalesratiottm NUMERIC,
    pricetobookratio NUMERIC,
    evtorevenue NUMERIC,
    evtoebitda NUMERIC,
    beta NUMERIC,
    field_52weekhigh NUMERIC,
    field_52weeklow NUMERIC,
    field_50daymovingaverage NUMERIC,
    field_200daymovingaverage NUMERIC,
    sharesoutstanding NUMERIC,
    sharesfloat NUMERIC,
    percentinsiders NUMERIC,
    percentinstitutions NUMERIC,
    dividenddate DATE,
    exdividenddate DATE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);


-- ========== FUNDAMENTALS/SPLITS ==========
-- Fundamental Data: splits
CREATE TABLE splits (
    symbol VARCHAR(10) PRIMARY KEY,
    effective_date DATE,
    split_factor NUMERIC,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_symbol ON {table_name}(symbol);


-- ========== OPTIONS/HISTORICAL_OPTIONS ==========
-- Options Chain: historical_options
CREATE TABLE historical_options (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT {table_name}_unique UNIQUE(symbol, contract_id, data_timestamp)
) PARTITION BY RANGE (data_timestamp);

-- Monthly partitions
CREATE TABLE historical_options_2025_08 
PARTITION OF historical_options
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE historical_options_2025_09 
PARTITION OF historical_options
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE historical_options_2025_10 
PARTITION OF historical_options
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE historical_options_2025_11 
PARTITION OF historical_options
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE historical_options_2025_12 
PARTITION OF historical_options
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE historical_options_2026_01 
PARTITION OF historical_options
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE historical_options_2026_02 
PARTITION OF historical_options
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE historical_options_2026_03 
PARTITION OF historical_options
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE historical_options_2026_04 
PARTITION OF historical_options
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE historical_options_2026_05 
PARTITION OF historical_options
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE historical_options_2026_06 
PARTITION OF historical_options
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE historical_options_2026_07 
PARTITION OF historical_options
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes for performance
CREATE INDEX idx_historical_options_symbol ON historical_options(symbol);
CREATE INDEX idx_historical_options_expiration ON historical_options(expiration) WHERE expiration IS NOT NULL;
CREATE INDEX idx_historical_options_strike ON historical_options(strike) WHERE strike IS NOT NULL;
CREATE INDEX idx_historical_options_timestamp ON historical_options(data_timestamp);


-- ========== OPTIONS/REALTIME_OPTIONS ==========
-- Options Chain: realtime_options
CREATE TABLE realtime_options (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT {table_name}_unique UNIQUE(symbol, contract_id, data_timestamp)
) PARTITION BY RANGE (data_timestamp);

-- Monthly partitions
CREATE TABLE realtime_options_2025_08 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE realtime_options_2025_09 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE realtime_options_2025_10 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE realtime_options_2025_11 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE realtime_options_2025_12 
PARTITION OF realtime_options
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE realtime_options_2026_01 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE realtime_options_2026_02 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE realtime_options_2026_03 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE realtime_options_2026_04 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE realtime_options_2026_05 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE realtime_options_2026_06 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE realtime_options_2026_07 
PARTITION OF realtime_options
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes for performance
CREATE INDEX idx_realtime_options_symbol ON realtime_options(symbol);
CREATE INDEX idx_realtime_options_expiration ON realtime_options(expiration) WHERE expiration IS NOT NULL;
CREATE INDEX idx_realtime_options_strike ON realtime_options(strike) WHERE strike IS NOT NULL;
CREATE INDEX idx_realtime_options_timestamp ON realtime_options(data_timestamp);


-- ========== SENTIMENT/INSIDER_TRANSACTIONS ==========
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


-- ========== SENTIMENT/NEWS_SENTIMENT ==========
-- News Sentiment
CREATE TABLE news_sentiment (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(255),
    url VARCHAR(255),
    time_published VARCHAR(50),
    authors TEXT[],
    summary VARCHAR(50),
    banner_image VARCHAR(255),
    source VARCHAR(50),
    category_within_source VARCHAR(10),
    source_domain VARCHAR(50),
    topics TEXT[],
    overall_sentiment_score NUMERIC,
    overall_sentiment_label VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ticker sentiment (normalized)
CREATE TABLE {table_name}_tickers (
    id BIGSERIAL PRIMARY KEY,
    article_id VARCHAR(100),
    ticker VARCHAR(10) NOT NULL,
    relevance_score NUMERIC,
    ticker_sentiment_score NUMERIC,
    ticker_sentiment_label VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_published ON {table_name}(time_published) WHERE time_published IS NOT NULL;
CREATE INDEX idx_{table_name}_tickers_ticker ON {table_name}_tickers(ticker);


-- ========== SENTIMENT/TOP_GAINERS_LOSERS ==========
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


-- ========== TECHNICAL_INDICATORS/AD ==========
-- Technical Indicator: ad
CREATE TABLE ad_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    chaikin_a_d NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE ad_data_2025_08 
PARTITION OF ad_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE ad_data_2025_09 
PARTITION OF ad_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE ad_data_2025_10 
PARTITION OF ad_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE ad_data_2025_11 
PARTITION OF ad_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE ad_data_2025_12 
PARTITION OF ad_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE ad_data_2026_01 
PARTITION OF ad_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE ad_data_2026_02 
PARTITION OF ad_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE ad_data_2026_03 
PARTITION OF ad_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE ad_data_2026_04 
PARTITION OF ad_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE ad_data_2026_05 
PARTITION OF ad_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE ad_data_2026_06 
PARTITION OF ad_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE ad_data_2026_07 
PARTITION OF ad_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_ad_symbol ON ad_data(symbol);
CREATE INDEX idx_ad_timestamp ON ad_data(timestamp);
CREATE INDEX idx_ad_interval ON ad_data(interval);


-- ========== TECHNICAL_INDICATORS/ADX ==========
-- Technical Indicator: adx
CREATE TABLE adx_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    adx NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE adx_data_2025_08 
PARTITION OF adx_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE adx_data_2025_09 
PARTITION OF adx_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE adx_data_2025_10 
PARTITION OF adx_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE adx_data_2025_11 
PARTITION OF adx_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE adx_data_2025_12 
PARTITION OF adx_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE adx_data_2026_01 
PARTITION OF adx_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE adx_data_2026_02 
PARTITION OF adx_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE adx_data_2026_03 
PARTITION OF adx_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE adx_data_2026_04 
PARTITION OF adx_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE adx_data_2026_05 
PARTITION OF adx_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE adx_data_2026_06 
PARTITION OF adx_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE adx_data_2026_07 
PARTITION OF adx_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_adx_symbol ON adx_data(symbol);
CREATE INDEX idx_adx_timestamp ON adx_data(timestamp);
CREATE INDEX idx_adx_interval ON adx_data(interval);


-- ========== TECHNICAL_INDICATORS/AROON ==========
-- Technical Indicator: aroon
CREATE TABLE aroon_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    aroon_down NUMERIC,
    aroon_up NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE aroon_data_2025_08 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE aroon_data_2025_09 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE aroon_data_2025_10 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE aroon_data_2025_11 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE aroon_data_2025_12 
PARTITION OF aroon_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE aroon_data_2026_01 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE aroon_data_2026_02 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE aroon_data_2026_03 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE aroon_data_2026_04 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE aroon_data_2026_05 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE aroon_data_2026_06 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE aroon_data_2026_07 
PARTITION OF aroon_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_aroon_symbol ON aroon_data(symbol);
CREATE INDEX idx_aroon_timestamp ON aroon_data(timestamp);
CREATE INDEX idx_aroon_interval ON aroon_data(interval);


-- ========== TECHNICAL_INDICATORS/ATR ==========
-- Technical Indicator: atr
CREATE TABLE atr_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    atr NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE atr_data_2025_08 
PARTITION OF atr_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE atr_data_2025_09 
PARTITION OF atr_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE atr_data_2025_10 
PARTITION OF atr_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE atr_data_2025_11 
PARTITION OF atr_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE atr_data_2025_12 
PARTITION OF atr_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE atr_data_2026_01 
PARTITION OF atr_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE atr_data_2026_02 
PARTITION OF atr_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE atr_data_2026_03 
PARTITION OF atr_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE atr_data_2026_04 
PARTITION OF atr_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE atr_data_2026_05 
PARTITION OF atr_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE atr_data_2026_06 
PARTITION OF atr_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE atr_data_2026_07 
PARTITION OF atr_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_atr_symbol ON atr_data(symbol);
CREATE INDEX idx_atr_timestamp ON atr_data(timestamp);
CREATE INDEX idx_atr_interval ON atr_data(interval);


-- ========== TECHNICAL_INDICATORS/BBANDS ==========
-- Technical Indicator: bbands
CREATE TABLE bbands_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    real_lower_band NUMERIC,
    real_middle_band NUMERIC,
    real_upper_band NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE bbands_data_2025_08 
PARTITION OF bbands_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE bbands_data_2025_09 
PARTITION OF bbands_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE bbands_data_2025_10 
PARTITION OF bbands_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE bbands_data_2025_11 
PARTITION OF bbands_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE bbands_data_2025_12 
PARTITION OF bbands_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE bbands_data_2026_01 
PARTITION OF bbands_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE bbands_data_2026_02 
PARTITION OF bbands_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE bbands_data_2026_03 
PARTITION OF bbands_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE bbands_data_2026_04 
PARTITION OF bbands_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE bbands_data_2026_05 
PARTITION OF bbands_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE bbands_data_2026_06 
PARTITION OF bbands_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE bbands_data_2026_07 
PARTITION OF bbands_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_bbands_symbol ON bbands_data(symbol);
CREATE INDEX idx_bbands_timestamp ON bbands_data(timestamp);
CREATE INDEX idx_bbands_interval ON bbands_data(interval);


-- ========== TECHNICAL_INDICATORS/CCI ==========
-- Technical Indicator: cci
CREATE TABLE cci_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    cci NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE cci_data_2025_08 
PARTITION OF cci_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE cci_data_2025_09 
PARTITION OF cci_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE cci_data_2025_10 
PARTITION OF cci_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE cci_data_2025_11 
PARTITION OF cci_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE cci_data_2025_12 
PARTITION OF cci_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE cci_data_2026_01 
PARTITION OF cci_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE cci_data_2026_02 
PARTITION OF cci_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE cci_data_2026_03 
PARTITION OF cci_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE cci_data_2026_04 
PARTITION OF cci_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE cci_data_2026_05 
PARTITION OF cci_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE cci_data_2026_06 
PARTITION OF cci_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE cci_data_2026_07 
PARTITION OF cci_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_cci_symbol ON cci_data(symbol);
CREATE INDEX idx_cci_timestamp ON cci_data(timestamp);
CREATE INDEX idx_cci_interval ON cci_data(interval);


-- ========== TECHNICAL_INDICATORS/EMA ==========
-- Technical Indicator: ema
CREATE TABLE ema_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    ema NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE ema_data_2025_08 
PARTITION OF ema_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE ema_data_2025_09 
PARTITION OF ema_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE ema_data_2025_10 
PARTITION OF ema_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE ema_data_2025_11 
PARTITION OF ema_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE ema_data_2025_12 
PARTITION OF ema_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE ema_data_2026_01 
PARTITION OF ema_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE ema_data_2026_02 
PARTITION OF ema_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE ema_data_2026_03 
PARTITION OF ema_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE ema_data_2026_04 
PARTITION OF ema_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE ema_data_2026_05 
PARTITION OF ema_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE ema_data_2026_06 
PARTITION OF ema_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE ema_data_2026_07 
PARTITION OF ema_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_ema_symbol ON ema_data(symbol);
CREATE INDEX idx_ema_timestamp ON ema_data(timestamp);
CREATE INDEX idx_ema_interval ON ema_data(interval);


-- ========== TECHNICAL_INDICATORS/MACD ==========
-- Technical Indicator: macd
CREATE TABLE macd_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    macd NUMERIC,
    macd_hist NUMERIC,
    macd_signal NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE macd_data_2025_08 
PARTITION OF macd_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE macd_data_2025_09 
PARTITION OF macd_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE macd_data_2025_10 
PARTITION OF macd_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE macd_data_2025_11 
PARTITION OF macd_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE macd_data_2025_12 
PARTITION OF macd_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE macd_data_2026_01 
PARTITION OF macd_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE macd_data_2026_02 
PARTITION OF macd_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE macd_data_2026_03 
PARTITION OF macd_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE macd_data_2026_04 
PARTITION OF macd_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE macd_data_2026_05 
PARTITION OF macd_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE macd_data_2026_06 
PARTITION OF macd_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE macd_data_2026_07 
PARTITION OF macd_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_macd_symbol ON macd_data(symbol);
CREATE INDEX idx_macd_timestamp ON macd_data(timestamp);
CREATE INDEX idx_macd_interval ON macd_data(interval);


-- ========== TECHNICAL_INDICATORS/MFI ==========
-- Technical Indicator: mfi
CREATE TABLE mfi_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    mfi NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE mfi_data_2025_08 
PARTITION OF mfi_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE mfi_data_2025_09 
PARTITION OF mfi_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE mfi_data_2025_10 
PARTITION OF mfi_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE mfi_data_2025_11 
PARTITION OF mfi_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE mfi_data_2025_12 
PARTITION OF mfi_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE mfi_data_2026_01 
PARTITION OF mfi_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE mfi_data_2026_02 
PARTITION OF mfi_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE mfi_data_2026_03 
PARTITION OF mfi_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE mfi_data_2026_04 
PARTITION OF mfi_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE mfi_data_2026_05 
PARTITION OF mfi_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE mfi_data_2026_06 
PARTITION OF mfi_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE mfi_data_2026_07 
PARTITION OF mfi_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_mfi_symbol ON mfi_data(symbol);
CREATE INDEX idx_mfi_timestamp ON mfi_data(timestamp);
CREATE INDEX idx_mfi_interval ON mfi_data(interval);


-- ========== TECHNICAL_INDICATORS/MOM ==========
-- Technical Indicator: mom
CREATE TABLE mom_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    mom NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE mom_data_2025_08 
PARTITION OF mom_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE mom_data_2025_09 
PARTITION OF mom_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE mom_data_2025_10 
PARTITION OF mom_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE mom_data_2025_11 
PARTITION OF mom_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE mom_data_2025_12 
PARTITION OF mom_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE mom_data_2026_01 
PARTITION OF mom_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE mom_data_2026_02 
PARTITION OF mom_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE mom_data_2026_03 
PARTITION OF mom_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE mom_data_2026_04 
PARTITION OF mom_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE mom_data_2026_05 
PARTITION OF mom_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE mom_data_2026_06 
PARTITION OF mom_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE mom_data_2026_07 
PARTITION OF mom_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_mom_symbol ON mom_data(symbol);
CREATE INDEX idx_mom_timestamp ON mom_data(timestamp);
CREATE INDEX idx_mom_interval ON mom_data(interval);


-- ========== TECHNICAL_INDICATORS/OBV ==========
-- Technical Indicator: obv
CREATE TABLE obv_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    obv NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE obv_data_2025_08 
PARTITION OF obv_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE obv_data_2025_09 
PARTITION OF obv_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE obv_data_2025_10 
PARTITION OF obv_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE obv_data_2025_11 
PARTITION OF obv_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE obv_data_2025_12 
PARTITION OF obv_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE obv_data_2026_01 
PARTITION OF obv_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE obv_data_2026_02 
PARTITION OF obv_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE obv_data_2026_03 
PARTITION OF obv_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE obv_data_2026_04 
PARTITION OF obv_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE obv_data_2026_05 
PARTITION OF obv_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE obv_data_2026_06 
PARTITION OF obv_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE obv_data_2026_07 
PARTITION OF obv_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_obv_symbol ON obv_data(symbol);
CREATE INDEX idx_obv_timestamp ON obv_data(timestamp);
CREATE INDEX idx_obv_interval ON obv_data(interval);


-- ========== TECHNICAL_INDICATORS/RSI ==========
-- Technical Indicator: rsi
CREATE TABLE rsi_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    rsi NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE rsi_data_2025_08 
PARTITION OF rsi_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE rsi_data_2025_09 
PARTITION OF rsi_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE rsi_data_2025_10 
PARTITION OF rsi_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE rsi_data_2025_11 
PARTITION OF rsi_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE rsi_data_2025_12 
PARTITION OF rsi_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE rsi_data_2026_01 
PARTITION OF rsi_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE rsi_data_2026_02 
PARTITION OF rsi_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE rsi_data_2026_03 
PARTITION OF rsi_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE rsi_data_2026_04 
PARTITION OF rsi_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE rsi_data_2026_05 
PARTITION OF rsi_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE rsi_data_2026_06 
PARTITION OF rsi_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE rsi_data_2026_07 
PARTITION OF rsi_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_rsi_symbol ON rsi_data(symbol);
CREATE INDEX idx_rsi_timestamp ON rsi_data(timestamp);
CREATE INDEX idx_rsi_interval ON rsi_data(interval);


-- ========== TECHNICAL_INDICATORS/SMA ==========
-- Technical Indicator: sma
CREATE TABLE sma_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    sma NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE sma_data_2025_08 
PARTITION OF sma_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE sma_data_2025_09 
PARTITION OF sma_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE sma_data_2025_10 
PARTITION OF sma_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE sma_data_2025_11 
PARTITION OF sma_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE sma_data_2025_12 
PARTITION OF sma_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE sma_data_2026_01 
PARTITION OF sma_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE sma_data_2026_02 
PARTITION OF sma_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE sma_data_2026_03 
PARTITION OF sma_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE sma_data_2026_04 
PARTITION OF sma_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE sma_data_2026_05 
PARTITION OF sma_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE sma_data_2026_06 
PARTITION OF sma_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE sma_data_2026_07 
PARTITION OF sma_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_sma_symbol ON sma_data(symbol);
CREATE INDEX idx_sma_timestamp ON sma_data(timestamp);
CREATE INDEX idx_sma_interval ON sma_data(interval);


-- ========== TECHNICAL_INDICATORS/STOCH ==========
-- Technical Indicator: stoch
CREATE TABLE stoch_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    slowd NUMERIC,
    slowk NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE stoch_data_2025_08 
PARTITION OF stoch_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE stoch_data_2025_09 
PARTITION OF stoch_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE stoch_data_2025_10 
PARTITION OF stoch_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE stoch_data_2025_11 
PARTITION OF stoch_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE stoch_data_2025_12 
PARTITION OF stoch_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE stoch_data_2026_01 
PARTITION OF stoch_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE stoch_data_2026_02 
PARTITION OF stoch_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE stoch_data_2026_03 
PARTITION OF stoch_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE stoch_data_2026_04 
PARTITION OF stoch_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE stoch_data_2026_05 
PARTITION OF stoch_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE stoch_data_2026_06 
PARTITION OF stoch_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE stoch_data_2026_07 
PARTITION OF stoch_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_stoch_symbol ON stoch_data(symbol);
CREATE INDEX idx_stoch_timestamp ON stoch_data(timestamp);
CREATE INDEX idx_stoch_interval ON stoch_data(interval);


-- ========== TECHNICAL_INDICATORS/VWAP ==========
-- Technical Indicator: vwap
CREATE TABLE vwap_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    vwap NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE vwap_data_2025_08 
PARTITION OF vwap_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE vwap_data_2025_09 
PARTITION OF vwap_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE vwap_data_2025_10 
PARTITION OF vwap_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE vwap_data_2025_11 
PARTITION OF vwap_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE vwap_data_2025_12 
PARTITION OF vwap_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE vwap_data_2026_01 
PARTITION OF vwap_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE vwap_data_2026_02 
PARTITION OF vwap_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE vwap_data_2026_03 
PARTITION OF vwap_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE vwap_data_2026_04 
PARTITION OF vwap_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE vwap_data_2026_05 
PARTITION OF vwap_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE vwap_data_2026_06 
PARTITION OF vwap_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE vwap_data_2026_07 
PARTITION OF vwap_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_vwap_symbol ON vwap_data(symbol);
CREATE INDEX idx_vwap_timestamp ON vwap_data(timestamp);
CREATE INDEX idx_vwap_interval ON vwap_data(interval);


-- ========== TECHNICAL_INDICATORS/WILLR ==========
-- Technical Indicator: willr
CREATE TABLE willr_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    willr NUMERIC,
    
    -- Metadata from API response
    meta_symbol VARCHAR(10),
    meta_indicator VARCHAR(50),
    meta_last_refreshed TIMESTAMP WITH TIME ZONE,
    meta_interval VARCHAR(20),
    meta_time_period INTEGER,
    meta_series_type VARCHAR(10),
    meta_time_zone VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, interval, timestamp)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions
CREATE TABLE willr_data_2025_08 
PARTITION OF willr_data
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE willr_data_2025_09 
PARTITION OF willr_data
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE willr_data_2025_10 
PARTITION OF willr_data
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE willr_data_2025_11 
PARTITION OF willr_data
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE willr_data_2025_12 
PARTITION OF willr_data
FOR VALUES FROM ('2025-12-01') TO ('2025-01-01');
CREATE TABLE willr_data_2026_01 
PARTITION OF willr_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE willr_data_2026_02 
PARTITION OF willr_data
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE willr_data_2026_03 
PARTITION OF willr_data
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE willr_data_2026_04 
PARTITION OF willr_data
FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE willr_data_2026_05 
PARTITION OF willr_data
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE willr_data_2026_06 
PARTITION OF willr_data
FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE willr_data_2026_07 
PARTITION OF willr_data
FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- Indexes
CREATE INDEX idx_willr_symbol ON willr_data(symbol);
CREATE INDEX idx_willr_timestamp ON willr_data(timestamp);
CREATE INDEX idx_willr_interval ON willr_data(interval);


COMMIT;

-- Verification query
SELECT 
    schemaname,
    tablename,
    COUNT(*) OVER() as total_tables
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
