-- FUNDAMENTALS: balance_sheet
-- Generated from ACTUAL API response investigation
-- Total fields in response: 79

CREATE TABLE balance_sheet (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
    fiscaldateending TEXT,
    reportedcurrency TEXT,
    totalassets TEXT,
    totalcurrentassets TEXT,
    cashandcashequivalentsatcarryingvalue TEXT,
    cashandshortterminvestments TEXT,
    inventory TEXT,
    currentnetreceivables TEXT,
    totalnoncurrentassets TEXT,
    propertyplantequipment TEXT,
    accumulateddepreciationamortizationppe TEXT,
    intangibleassets TEXT,
    intangibleassetsexcludinggoodwill TEXT,
    goodwill TEXT,
    investments TEXT,
    longterminvestments TEXT,
    shortterminvestments TEXT,
    othercurrentassets TEXT,
    othernoncurrentassets TEXT,
    totalliabilities TEXT,
    totalcurrentliabilities TEXT,
    currentaccountspayable TEXT,
    deferredrevenue TEXT,
    currentdebt TEXT,
    shorttermdebt TEXT,
    totalnoncurrentliabilities TEXT,
    capitalleaseobligations TEXT,
    longtermdebt TEXT,
    currentlongtermdebt TEXT,
    longtermdebtnoncurrent TEXT,
    shortlongtermdebttotal TEXT,
    othercurrentliabilities TEXT,
    othernoncurrentliabilities TEXT,
    totalshareholderequity TEXT,
    treasurystock TEXT,
    retainedearnings TEXT,
    commonstock TEXT,
    commonstocksharesoutstanding TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_balance_sheet_symbol ON balance_sheet(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_balance_sheet_date ON balance_sheet(date) WHERE date IS NOT NULL;
