# Updated Data Source Division - Technical Reference
**Version:** 3.0  
**Critical Change:** IBKR provides ALL real-time pricing; Alpha Vantage focuses on Greeks, indicators, and analytics

---

## **Data Source Division (FINAL)**

### **IBKR Provides (ALL Real-Time Pricing)**
```python
IBKR_DATA = {
    'real_time_bars': [
        '1-min bars',
        '5-min bars',
        '10-min bars',
        '15-min bars',
        '30-min bars',
        '1-hour bars'
    ],
    'quotes': {
        'fields': ['bid', 'ask', 'last', 'bid_size', 'ask_size'],
        'update_frequency': 'tick-by-tick'
    },
    'moc_imbalance': {
        'exchanges': ['NYSE', 'NASDAQ'],
        'window': '15:40-15:55 ET',
        'update_frequency': '5 seconds during window'
    },
    'execution': {
        'order_types': ['LMT', 'MKT', 'STP', 'TRAIL'],
        'confirmation': 'real-time fills'
    },
    'account': {
        'positions': 'real-time updates',
        'pnl': 'real-time calculation',
        'buying_power': 'real-time available'
    }
}
```

### **Alpha Vantage Provides (41 APIs Total)**
```python
ALPHA_VANTAGE_DATA = {
    # OPTIONS & GREEKS (2 APIs) - PRIMARY GREEKS SOURCE
    'options': {
        'REALTIME_OPTIONS': {
            'provides': 'Full option chain with Greeks',
            'greeks': ['delta', 'gamma', 'theta', 'vega', 'rho'],
            'priority': 'CRITICAL - Primary Greeks source'
        },
        'HISTORICAL_OPTIONS': {
            'provides': 'Historical options data',
            'use_case': 'Backtesting and analysis'
        }
    },
    
    # TECHNICAL INDICATORS (16 APIs)
    'indicators': {
        'momentum': ['RSI', 'MACD', 'STOCH', 'MOM', 'WILLR', 'CCI'],
        'trend': ['ADX', 'AROON', 'EMA', 'SMA'],
        'volatility': ['BBANDS', 'ATR'],
        'volume': ['OBV', 'AD', 'MFI'],
        'support_resistance': ['VWAP']
    },
    
    # ADVANCED ANALYTICS (2 APIs)
    'analytics': {
        'ANALYTICS_FIXED_WINDOW': {
            'calculations': ['MIN', 'MAX', 'MEAN', 'MEDIAN', 'VARIANCE', 
                           'STDDEV', 'CORRELATION', 'COVARIANCE'],
            'use_case': 'Statistical analysis over fixed periods'
        },
        'ANALYTICS_SLIDING_WINDOW': {
            'calculations': ['Rolling means', 'Rolling volatility', 
                           'Rolling correlations'],
            'use_case': 'Dynamic analysis with moving windows'
        }
    },
    
    # SENTIMENT & NEWS (3 APIs)
    'sentiment': {
        'NEWS_SENTIMENT': 'Real-time news sentiment scores',
        'TOP_GAINERS_LOSERS': 'Market momentum indicators',
        'INSIDER_TRANSACTIONS': 'Insider trading activity'
    },
    
    # FUNDAMENTALS (10 APIs)
    'company_data': {
        'OVERVIEW': 'Company profile and key metrics',
        'EARNINGS': 'Historical earnings data',
        'EARNINGS_ESTIMATES': 'Analyst earnings estimates',
        'EARNINGS_CALENDAR': 'Upcoming earnings dates',
        'EARNINGS_CALL_TRANSCRIPT': 'Earnings call text',
        'INCOME_STATEMENT': 'Revenue and expenses',
        'BALANCE_SHEET': 'Assets and liabilities',
        'CASH_FLOW': 'Cash flow statements',
        'DIVIDENDS': 'Dividend history',
        'SPLITS': 'Stock split history'
    },
    
    # ECONOMIC INDICATORS (5 APIs)
    'economic': {
        'TREASURY_YIELD': 'Risk-free rate curves',
        'FEDERAL_FUNDS_RATE': 'Fed policy rate',
        'CPI': 'Consumer Price Index',
        'INFLATION': 'Inflation expectations',
        'REAL_GDP': 'Economic growth metrics'
    }
}

# REMOVED APIs (Not needed)
REMOVED_APIS = [
    'TIME_SERIES_INTRADAY',  # Using IBKR for all intraday pricing
    'LISTING_STATUS'         # Out of scope
]
```

---

## **Implementation Priority Order**

### **Critical Path APIs (Must have first)**
1. **IBKR Real-time bars** - Foundation for all price data
2. **REALTIME_OPTIONS** - Primary Greeks source
3. **Core Indicators** - RSI, MACD, BBANDS, VWAP

### **Enhancement APIs (Add incrementally)**
4. Supporting indicators (ATR, ADX, etc.)
5. Analytics functions
6. Sentiment analysis
7. Fundamental data
8. Economic indicators

---

## **API Call Distribution**

### **High Frequency (Every 10-60 seconds)**
- REALTIME_OPTIONS (Greeks)
- Core indicators for active positions

### **Medium Frequency (Every 5-15 minutes)**
- Supporting indicators
- Analytics calculations
- Sentiment updates

### **Low Frequency (Daily/Weekly)**
- Fundamental data
- Economic indicators
- Historical options

### **Real-Time Streams (Continuous)**
- IBKR bars (all timeframes)
- IBKR quotes
- MOC imbalance (3:40-3:55 PM only)

---

## **Rate Management Strategy**

```python
RATE_ALLOCATION = {
    'alpha_vantage': {
        'limit': 600,  # calls per minute
        'target': 500,  # stay below this
        'allocation': {
            'options_greeks': 200,  # 40% for critical Greeks
            'indicators': 200,      # 40% for indicators
            'analytics': 50,        # 10% for analytics
            'other': 50,           # 10% for sentiment/fundamentals
        }
    },
    'ibkr': {
        'subscriptions': 50,  # concurrent market data lines
        'allocation': {
            'tier_a': 20,  # SPY, QQQ, IWM, SPX
            'tier_b': 20,  # MAG7 stocks
            'tier_c': 10,  # Rotating watchlist
        }
    }
}
```

---

## **Data Flow Architecture**

```
[IBKR Real-Time Pricing]          [Alpha Vantage Greeks & Indicators]
         |                                      |
         v                                      v
    [Price Cache]                        [Rate Limiter]
         |                                      |
         +-----------------+-------------------+
                           |
                    [Data Ingestion]
                           |
                    [PostgreSQL + Redis]
                           |
                   [Analytics Engine]
                           |
                   [Decision Engine]
                           |
                    [Risk Manager]
                           |
                   [IBKR Executor]
```

---

## **Key Integration Points**

### **Price + Greeks Fusion**
```python
def get_option_data(symbol, strike, expiration):
    # Real-time price from IBKR
    price = ibkr.get_quote(symbol)
    
    # Greeks from Alpha Vantage
    greeks = av.get_realtime_options(symbol)
    
    # Combine for decision
    return {
        'price': price,
        'greeks': greeks,
        'timestamp': now()
    }
```

### **Indicator Calculation**
```python
def calculate_indicators(symbol):
    # Price data from IBKR
    bars = ibkr.get_bars(symbol, '5min')
    
    # Indicators from Alpha Vantage
    rsi = av.get_rsi(symbol)
    macd = av.get_macd(symbol)
    
    # Combine for signals
    return aggregate_signals(bars, rsi, macd)
```

---

## **Critical Dependencies**

1. **IBKR connection must be stable** - All pricing depends on it
2. **Greeks must be < 30 seconds old** - Stale Greeks = no trades
3. **Rate limiting must work** - Exceeding limits breaks everything
4. **Cache layer critical** - Reduces API calls by 40%

---

## **Schema Design Principles**

### **IBKR Tables**
- Optimized for time-series queries
- Partitioned by date
- Indexed on (symbol, timestamp)

### **Alpha Vantage Tables**
- Separate table per API
- JSONB for flexible response storage
- Indexed on symbol and data-specific fields

### **Example Schemas**

```sql
-- IBKR Pricing
CREATE TABLE ibkr_bars_5min (
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- Alpha Vantage Options
CREATE TABLE av_realtime_options (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    contract_id VARCHAR(50),
    strike DECIMAL(10,2),
    expiration DATE,
    option_type VARCHAR(4),
    delta DECIMAL(6,4),
    gamma DECIMAL(6,4),
    theta DECIMAL(8,4),
    vega DECIMAL(8,4),
    implied_volatility DECIMAL(6,4),
    raw_response JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_exp (symbol, expiration)
);
```

---

## **Testing Requirements**

### **Before Using Any Data Source**
1. Test connection and authentication
2. Verify response structure
3. Measure actual latency
4. Test error handling
5. Verify rate limiting
6. Test data quality
7. Measure storage requirements
8. Test query performance

### **Integration Testing**
1. Price + Greeks synchronization
2. Indicator + price alignment
3. Decision with all data types
4. Execution with real-time data
5. MOC window special handling

---

## **Monitoring Requirements**

### **Real-Time Monitoring**
- IBKR connection status
- Alpha Vantage rate usage
- Greeks freshness
- Data latency
- Cache hit rates

### **Daily Monitoring**
- API call distribution
- Data quality metrics
- Storage growth
- Query performance
- Cost analysis