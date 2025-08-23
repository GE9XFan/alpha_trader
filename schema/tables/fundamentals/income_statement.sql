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
