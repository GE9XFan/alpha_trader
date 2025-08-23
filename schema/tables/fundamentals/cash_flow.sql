-- FUNDAMENTALS: cash_flow
-- Generated from ACTUAL API response investigation
-- Total fields in response: 61

CREATE TABLE cash_flow (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date_ending DATE NOT NULL,
    report_type VARCHAR(10) CHECK (report_type IN ('annual', 'quarterly')),
    reported_currency VARCHAR(10),
    fiscaldateending TEXT,
    reportedcurrency TEXT,
    operatingcashflow TEXT,
    paymentsforoperatingactivities TEXT,
    proceedsfromoperatingactivities TEXT,
    changeinoperatingliabilities TEXT,
    changeinoperatingassets TEXT,
    depreciationdepletionandamortization TEXT,
    capitalexpenditures TEXT,
    changeinreceivables TEXT,
    changeininventory TEXT,
    profitloss TEXT,
    cashflowfrominvestment TEXT,
    cashflowfromfinancing TEXT,
    proceedsfromrepaymentsofshorttermdebt TEXT,
    paymentsforrepurchaseofcommonstock TEXT,
    paymentsforrepurchaseofequity TEXT,
    paymentsforrepurchaseofpreferredstock TEXT,
    dividendpayout TEXT,
    dividendpayoutcommonstock TEXT,
    dividendpayoutpreferredstock TEXT,
    proceedsfromissuanceofcommonstock TEXT,
    proceedsfromissuanceoflongtermdebtandcapitalsecuritiesnet TEXT,
    proceedsfromissuanceofpreferredstock TEXT,
    proceedsfromrepurchaseofequity TEXT,
    proceedsfromsaleoftreasurystock TEXT,
    changeincashandcashequivalents TEXT,
    changeinexchangerate TEXT,
    netincome TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_cash_flow_symbol ON cash_flow(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_cash_flow_date ON cash_flow(date) WHERE date IS NOT NULL;
