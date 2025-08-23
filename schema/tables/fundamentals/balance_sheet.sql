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
