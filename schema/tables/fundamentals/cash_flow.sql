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
