/**
 * Complete Alpha Vantage API Response Schemas
 * Generated from deep analysis of actual API responses
 * 
 * Key patterns:
 * - Most numeric values are returned as strings to preserve precision
 * - "None" string is used instead of null in many cases
 * - Dates use YYYY-MM-DD format, timestamps use YYYYMMDDTHHMMSS
 * - Technical indicators have consistent Meta Data + Technical Analysis structure
 */

// ============================================================================
// FUNDAMENTALS
// ============================================================================

export interface CompanyOverview {
  Symbol: string;
  AssetType: string;
  Name: string;
  Description: string;
  CIK: string;
  Exchange: string;
  Currency: string;
  Country: string;
  Sector: string;
  Industry: string;
  Address: string;
  OfficialSite: string;
  FiscalYearEnd: string;
  LatestQuarter: string; // YYYY-MM-DD
  MarketCapitalization: string; // numeric string
  EBITDA: string;
  PERatio: string;
  PEGRatio: string;
  BookValue: string;
  DividendPerShare: string;
  DividendYield: string;
  EPS: string;
  RevenuePerShareTTM: string;
  ProfitMargin: string;
  OperatingMarginTTM: string;
  ReturnOnAssetsTTM: string;
  ReturnOnEquityTTM: string;
  RevenueTTM: string;
  GrossProfitTTM: string;
  DilutedEPSTTM: string;
  QuarterlyEarningsGrowthYOY: string;
  QuarterlyRevenueGrowthYOY: string;
  AnalystTargetPrice: string;
  AnalystRatingStrongBuy: string;
  AnalystRatingBuy: string;
  AnalystRatingHold: string;
  AnalystRatingSell: string;
  AnalystRatingStrongSell: string;
  TrailingPE: string;
  ForwardPE: string;
  PriceToSalesRatioTTM: string;
  PriceToBookRatio: string;
  EVToRevenue: string;
  EVToEBITDA: string;
  Beta: string;
  '52WeekHigh': string;
  '52WeekLow': string;
  '50DayMovingAverage': string;
  '200DayMovingAverage': string;
  SharesOutstanding: string;
  SharesFloat: string;
  PercentInsiders: string;
  PercentInstitutions: string;
  DividendDate: string; // YYYY-MM-DD
  ExDividendDate: string; // YYYY-MM-DD
}

export interface BalanceSheetReport {
  fiscalDateEnding: string; // YYYY-MM-DD
  reportedCurrency: string;
  totalAssets: string;
  totalCurrentAssets: string;
  cashAndCashEquivalentsAtCarryingValue: string;
  cashAndShortTermInvestments: string;
  inventory: string;
  currentNetReceivables: string;
  totalNonCurrentAssets: string;
  propertyPlantEquipment: string;
  accumulatedDepreciationAmortizationPPE: string | "None";
  intangibleAssets: string | "None";
  intangibleAssetsExcludingGoodwill: string | "None";
  goodwill: string | "None";
  investments: string | "None";
  longTermInvestments: string;
  shortTermInvestments: string;
  otherCurrentAssets: string;
  otherNonCurrentAssets: string | "None";
  totalLiabilities: string;
  totalCurrentLiabilities: string;
  currentAccountsPayable: string;
  deferredRevenue: string | "None";
  currentDebt: string | "None";
  shortTermDebt: string;
  totalNonCurrentLiabilities: string;
  capitalLeaseObligations: string | "None";
  longTermDebt: string;
  currentLongTermDebt: string;
  longTermDebtNoncurrent: string | "None";
  shortLongTermDebtTotal: string;
  otherCurrentLiabilities: string;
  otherNonCurrentLiabilities: string;
  totalShareholderEquity: string;
  treasuryStock: string | "None";
  retainedEarnings: string;
  commonStock: string;
  commonStockSharesOutstanding: string;
}

export interface BalanceSheetResponse {
  symbol: string;
  annualReports: BalanceSheetReport[];
  quarterlyReports: BalanceSheetReport[];
}

export interface IncomeStatementReport {
  fiscalDateEnding: string;
  reportedCurrency: string;
  grossProfit: string;
  totalRevenue: string;
  costOfRevenue: string;
  costofGoodsAndServicesSold: string;
  operatingIncome: string;
  sellingGeneralAndAdministrative: string;
  researchAndDevelopment: string;
  operatingExpenses: string;
  investmentIncomeNet: string | "None";
  netInterestIncome: string;
  interestIncome: string;
  interestExpense: string;
  nonInterestIncome: string | "None";
  otherNonOperatingIncome: string;
  depreciation: string | "None";
  depreciationAndAmortization: string | "None";
  incomeBeforeTax: string;
  incomeTaxExpense: string;
  interestAndDebtExpense: string;
  netIncomeFromContinuingOperations: string;
  comprehensiveIncomeNetOfTax: string;
  ebit: string;
  ebitda: string;
  netIncome: string;
}

export interface IncomeStatementResponse {
  symbol: string;
  annualReports: IncomeStatementReport[];
  quarterlyReports: IncomeStatementReport[];
}

export interface CashFlowReport {
  fiscalDateEnding: string;
  reportedCurrency: string;
  operatingCashflow: string;
  paymentsForOperatingActivities: string | "None";
  proceedsFromOperatingActivities: string | "None";
  changeInOperatingLiabilities: string;
  changeInOperatingAssets: string;
  depreciationDepletionAndAmortization: string;
  capitalExpenditures: string;
  changeInReceivables: string;
  changeInInventory: string;
  profitLoss: string;
  cashflowFromInvestment: string;
  cashflowFromFinancing: string;
  proceedsFromRepaymentsOfShortTermDebt: string | "None";
  paymentsForRepurchaseOfCommonStock: string;
  paymentsForRepurchaseOfEquity: string | "None";
  paymentsForRepurchaseOfPreferredStock: string | "None";
  dividendPayout: string;
  dividendPayoutCommonStock: string;
  dividendPayoutPreferredStock: string | "None";
  proceedsFromIssuanceOfCommonStock: string;
  proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet: string;
  proceedsFromIssuanceOfPreferredStock: string | "None";
  proceedsFromRepurchaseOfEquity: string | "None";
  proceedsFromSaleOfTreasuryStock: string | "None";
  changeInCashAndCashEquivalents: string;
  changeInExchangeRate: string | "None";
  netIncome: string;
}

export interface CashFlowResponse {
  symbol: string;
  annualReports: CashFlowReport[];
  quarterlyReports: CashFlowReport[];
}

export interface EarningsReport {
  fiscalDateEnding: string;
  reportedDate: string;
  reportedEPS: string;
  estimatedEPS: string;
  surprise: string;
  surprisePercentage: string;
}

export interface EarningsResponse {
  symbol: string;
  annualEarnings: Array<{
    fiscalDateEnding: string;
    reportedEPS: string;
  }>;
  quarterlyEarnings: EarningsReport[];
}

export interface DividendEvent {
  date: string;
  dividendAmount: string;
}

export interface DividendsResponse {
  data: DividendEvent[];
}

export interface SplitEvent {
  date: string;
  splitRatio: string;
}

export interface SplitsResponse {
  data: SplitEvent[];
}

// ============================================================================
// TECHNICAL INDICATORS
// ============================================================================

interface TechnicalIndicatorMetaData {
  "1: Symbol": string;
  "2: Indicator": string;
  "3: Last Refreshed": string;
  "4: Interval": string;
  "5: Time Period": number;
  "6: Series Type": string;
  "7: Time Zone": string;
}

export interface RSIResponse {
  "Meta Data": TechnicalIndicatorMetaData;
  "Technical Analysis: RSI": {
    [date: string]: {
      RSI: string;
    };
  };
}

export interface MACDResponse {
  "Meta Data": {
    "1: Symbol": string;
    "2: Indicator": string;
    "3: Last Refreshed": string;
    "4: Interval": string;
    "5.1: Fast Period": number;
    "5.2: Slow Period": number;
    "5.3: Signal Period": number;
    "6: Series Type": string;
    "7: Time Zone": string;
  };
  "Technical Analysis: MACD": {
    [date: string]: {
      MACD: string;
      MACD_Signal: string;
      MACD_Hist: string;
    };
  };
}

export interface BollingerBandsResponse {
  "Meta Data": TechnicalIndicatorMetaData & {
    "8: Deviation multiplier up": number;
    "9: Deviation multiplier down": number;
    "10: MA Type": number;
  };
  "Technical Analysis: BBANDS": {
    [date: string]: {
      "Real Upper Band": string;
      "Real Middle Band": string;
      "Real Lower Band": string;
    };
  };
}

export interface StochasticResponse {
  "Meta Data": {
    "1: Symbol": string;
    "2: Indicator": string;
    "3: Last Refreshed": string;
    "4: Interval": string;
    "5.1: FastK Period": number;
    "5.2: SlowK Period": number;
    "5.3: SlowK MA Type": number;
    "5.4: SlowD Period": number;
    "5.5: SlowD MA Type": number;
    "6: Time Zone": string;
  };
  "Technical Analysis: STOCH": {
    [date: string]: {
      SlowK: string;
      SlowD: string;
    };
  };
}

export interface ADXResponse {
  "Meta Data": TechnicalIndicatorMetaData;
  "Technical Analysis: ADX": {
    [date: string]: {
      ADX: string;
    };
  };
}

export interface ATRResponse {
  "Meta Data": TechnicalIndicatorMetaData;
  "Technical Analysis: ATR": {
    [date: string]: {
      ATR: string;
    };
  };
}

export interface AroonResponse {
  "Meta Data": TechnicalIndicatorMetaData;
  "Technical Analysis: AROON": {
    [date: string]: {
      "Aroon Up": string;
      "Aroon Down": string;
    };
  };
}

export interface CCIResponse {
  "Meta Data": TechnicalIndicatorMetaData;
  "Technical Analysis: CCI": {
    [date: string]: {
      CCI: string;
    };
  };
}

export interface MFIResponse {
  "Meta Data": TechnicalIndicatorMetaData;
  "Technical Analysis: MFI": {
    [date: string]: {
      MFI: string;
    };
  };
}

export interface OBVResponse {
  "Meta Data": {
    "1: Symbol": string;
    "2: Indicator": string;
    "3: Last Refreshed": string;
    "4: Interval": string;
    "5: Time Zone": string;
  };
  "Technical Analysis: OBV": {
    [date: string]: {
      OBV: string;
    };
  };
}

export interface ChaikinADResponse {
  "Meta Data": {
    "1: Symbol": string;
    "2: Indicator": string;
    "3: Last Refreshed": string;
    "4: Interval": string;
    "5: Time Zone": string;
  };
  "Technical Analysis: Chaikin A/D": {
    [date: string]: {
      "Chaikin A/D": string;
    };
  };
}

export interface VWAPResponse {
  "Meta Data": {
    "1: Symbol": string;
    "2: Indicator": string;
    "3: Last Refreshed": string;
    "4: Interval": string;
    "5: Time Zone": string;
  };
  "Technical Analysis: VWAP": {
    [date: string]: {
      VWAP: string;
    };
  };
}

// ============================================================================
// OPTIONS
// ============================================================================

export interface OptionContract {
  contractID: string; // e.g., "AAPL250822C00110000"
  symbol: string;
  expiration: string; // YYYY-MM-DD
  strike: string;
  type: "call" | "put";
  last: string;
  mark: string;
  bid: string;
  bid_size: string;
  ask: string;
  ask_size: string;
  volume: string;
  open_interest: string;
  date: string; // YYYY-MM-DD
  implied_volatility: string;
  delta: string;
  gamma: string;
  theta: string;
  vega: string;
  rho: string;
}

export interface RealtimeOptionsResponse {
  endpoint: string;
  message: string;
  data: OptionContract[];
}

export interface HistoricalOptionsResponse {
  endpoint: string;
  message: string;
  data: OptionContract[];
}

// ============================================================================
// ECONOMIC INDICATORS
// ============================================================================

export interface EconomicDataPoint {
  date: string; // YYYY-MM-DD
  value: string;
}

export interface EconomicIndicatorResponse {
  name: string;
  interval: "monthly" | "quarterly" | "annual" | "weekly" | "daily";
  unit: string;
  data: EconomicDataPoint[];
}

// Specific economic indicators
export type CPIResponse = EconomicIndicatorResponse;
export type FederalFundsRateResponse = EconomicIndicatorResponse;
export type InflationResponse = EconomicIndicatorResponse;
export type RealGDPResponse = EconomicIndicatorResponse;
export type TreasuryYieldResponse = EconomicIndicatorResponse;

// ============================================================================
// SENTIMENT & NEWS
// ============================================================================

export interface NewsArticle {
  title: string;
  url: string;
  time_published: string; // YYYYMMDDTHHMMSS
  authors: string[];
  summary: string;
  banner_image?: string;
  source: string;
  category_within_source: string;
  source_domain: string;
  topics: Array<{
    topic: string;
    relevance_score: string;
  }>;
  overall_sentiment_score: number;
  overall_sentiment_label: "Bearish" | "Somewhat-Bearish" | "Neutral" | "Somewhat-Bullish" | "Bullish";
  ticker_sentiment: Array<{
    ticker: string;
    relevance_score: string;
    ticker_sentiment_score: string;
    ticker_sentiment_label: string;
  }>;
}

export interface NewsSentimentResponse {
  items: string;
  sentiment_score_definition: string;
  relevance_score_definition: string;
  feed: NewsArticle[];
}

export interface InsiderTransaction {
  transaction_date: string; // YYYY-MM-DD
  ticker: string;
  executive: string;
  executive_title: string;
  security_type: string;
  acquisition_or_disposal: "A" | "D"; // Acquisition or Disposal
  shares: string;
  share_price: string;
}

export interface InsiderTransactionsResponse {
  data: InsiderTransaction[];
}

export interface MarketMover {
  ticker: string;
  price: string;
  change_amount: string;
  change_percentage: string;
  volume: string;
}

export interface TopGainersLosersResponse {
  metadata: string;
  last_updated: string;
  top_gainers: MarketMover[];
  top_losers: MarketMover[];
  most_actively_traded: MarketMover[];
}

// ============================================================================
// ANALYTICS
// ============================================================================

export interface AnalyticsFixedWindowMetaData {
  symbols: string;
  min_dt: string;
  max_dt: string;
  ohlc: string;
  interval: string;
}

export interface DrawdownRange {
  start_drawdown: string;
  end_drawdown: string;
}

export interface MaxDrawdown {
  max_drawdown: number;
  drawdown_range: DrawdownRange;
}

export interface HistogramData {
  bin_count: number[];
  bin_edges: number[];
}

export interface AnalyticsFixedWindowPayload {
  RETURNS_CALCULATIONS: {
    MIN: { [symbol: string]: number };
    MAX: { [symbol: string]: number };
    MEAN: { [symbol: string]: number };
    MEDIAN: { [symbol: string]: number };
    CUMULATIVE_RETURN: { [symbol: string]: number };
    VARIANCE: { [symbol: string]: number };
    STDDEV: { [symbol: string]: number };
    MAX_DRAWDOWN: { [symbol: string]: MaxDrawdown };
    HISTOGRAM: { [symbol: string]: HistogramData };
    AUTOCORRELATION: { [symbol: string]: number };
    COVARIANCE: {
      index: string[];
      covariance: number[][];
    };
    CORRELATION: {
      index: string[];
      correlation: number[][];
    };
  };
}

export interface AnalyticsFixedWindowResponse {
  meta_data: AnalyticsFixedWindowMetaData;
  payload: AnalyticsFixedWindowPayload;
}

export interface AnalyticsSlidingWindowMetaData {
  symbols: string;
  window_size: number;
  min_dt: string;
  max_dt: string;
  ohlc: string;
  interval: string;
}

export interface RunningMetrics {
  [symbol: string]: {
    [date: string]: number;
  };
}

export interface WindowStartDates {
  [date: string]: string;
}

export interface AnalyticsSlidingWindowPayload {
  RETURNS_CALCULATIONS: {
    MEAN: {
      RUNNING_MEAN: RunningMetrics;
      window_start: WindowStartDates;
    };
    MEDIAN: {
      RUNNING_MEDIAN: RunningMetrics;
      window_start: WindowStartDates;
    };
    MIN: {
      RUNNING_MIN: RunningMetrics;
      window_start: WindowStartDates;
    };
    MAX: {
      RUNNING_MAX: RunningMetrics;
      window_start: WindowStartDates;
    };
    STDDEV: {
      RUNNING_STDDEV: RunningMetrics;
      window_start: WindowStartDates;
    };
    VARIANCE: {
      RUNNING_VARIANCE: RunningMetrics;
      window_start: WindowStartDates;
    };
  };
}

export interface AnalyticsSlidingWindowResponse {
  meta_data: AnalyticsSlidingWindowMetaData;
  payload: AnalyticsSlidingWindowPayload;
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

export type AlphaVantageResponse = 
  | CompanyOverview
  | BalanceSheetResponse
  | IncomeStatementResponse
  | CashFlowResponse
  | EarningsResponse
  | DividendsResponse
  | SplitsResponse
  | RSIResponse
  | MACDResponse
  | BollingerBandsResponse
  | StochasticResponse
  | ADXResponse
  | ATRResponse
  | AroonResponse
  | CCIResponse
  | MFIResponse
  | OBVResponse
  | ChaikinADResponse
  | VWAPResponse
  | RealtimeOptionsResponse
  | HistoricalOptionsResponse
  | EconomicIndicatorResponse
  | NewsSentimentResponse
  | InsiderTransactionsResponse
  | TopGainersLosersResponse
  | AnalyticsFixedWindowResponse
  | AnalyticsSlidingWindowResponse;

// Type guards for runtime validation
export function isErrorResponse(response: any): response is { Error?: string; Note?: string } {
  return response && (response.Error !== undefined || response.Note !== undefined);
}

export function hasMetaData(response: any): response is { "Meta Data": any } {
  return response && response["Meta Data"] !== undefined;
}

export function isOptionsResponse(response: any): response is RealtimeOptionsResponse | HistoricalOptionsResponse {
  return response && response.endpoint !== undefined && response.data !== undefined;
}