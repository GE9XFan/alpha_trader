-- FUNDAMENTALS: income_statement
-- Generated from ACTUAL API response investigation
-- Total fields in response: 55

CREATE TABLE income_statement (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
    fiscaldateending TEXT,
    reportedcurrency TEXT,
    grossprofit TEXT,
    totalrevenue TEXT,
    costofrevenue TEXT,
    costofgoodsandservicessold TEXT,
    operatingincome TEXT,
    sellinggeneralandadministrative TEXT,
    researchanddevelopment TEXT,
    operatingexpenses TEXT,
    investmentincomenet TEXT,
    netinterestincome TEXT,
    interestincome TEXT,
    interestexpense TEXT,
    noninterestincome TEXT,
    othernonoperatingincome TEXT,
    depreciation TEXT,
    depreciationandamortization TEXT,
    incomebeforetax TEXT,
    incometaxexpense TEXT,
    interestanddebtexpense TEXT,
    netincomefromcontinuingoperations TEXT,
    comprehensiveincomenetoftax TEXT,
    ebit TEXT,
    ebitda TEXT,
    netincome TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_income_statement_symbol ON income_statement(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_income_statement_date ON income_statement(date) WHERE date IS NOT NULL;
