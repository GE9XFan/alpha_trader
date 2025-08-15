# SSOT-Tech.md – Technical Specification
**Version:** 2.0 (Updated for Skeleton-First, API-Driven Development)  
**Last Updated:** Current  
**Audience:** Engineering / Claude Code  
**Purpose:** Defines **how** the production options trading system is implemented, with skeleton-first architecture, API-driven schema evolution, and configuration-driven design.  
**Scope:** All implementation-level details, frozen interfaces, and required artifacts for each build phase.  
**Relation to SSOT-Ops:** SSOT-Ops.md defines **what/when**; this document defines **how**.

---

## 1. Introduction & CRITICAL Updates

### 1.1 Version 2.0 Changes
- **Skeleton-First Development**: Build complete module skeleton before implementation
- **API-Driven Schema**: Schema evolves based on actual API responses
- **Configuration-Driven**: All parameters externalized to YAML/config files
- **IBKR Intraday Pricing**: All real-time price data from IBKR, not Alpha Vantage
- **Incremental Testing**: Each API fully tested before proceeding

### 1.2 Critical Warnings
- This system trades **real money** – every decision matters
- **Test each component in isolation first**
- **Schema must match actual API responses exactly**
- **No hard-coded configuration values**

---

## 2. Data Sources – REVISED DIVISION

```python
# IBKR Provides (ALL REAL-TIME PRICING)
IBKR_DATA = {
    'intraday_pricing': [
        '1-sec bars',
        '5-sec bars', 
        '1-min bars',
        '5-min bars',
        '15-min bars',
        '30-min bars',
        '1-hour bars'
    ],
    'quotes': ['bid', 'ask', 'last', 'bid_size', 'ask_size'],
    'moc_imbalance': ['NYSE', 'NASDAQ'],  # 3:40-3:55 PM ET
    'execution': 'TWS API',
    'fills': 'Confirmation and slippage check',
    'positions': 'Real-time position monitoring'
}

# Alpha Vantage Provides (GREEKS, INDICATORS, ANALYTICS)
ALPHA_VANTAGE_DATA = {
    'options': {
        'REALTIME_OPTIONS': 'WITH FULL GREEKS (PRIMARY SOURCE)',
        'HISTORICAL_OPTIONS': 'FOR BACKTESTING'
    },
    'technical_indicators': [
        'RSI', 'MACD', 'STOCH', 'BBANDS', 'ATR', 'ADX',
        'VWAP', 'EMA', 'SMA', 'AROON', 'CCI', 'MFI',
        'WILLR', 'MOM', 'AD', 'OBV'
    ],
    'analytics': [
        'ANALYTICS_FIXED_WINDOW',
        'ANALYTICS_SLIDING_WINDOW'
    ],
    'fundamentals': [
        'OVERVIEW', 'EARNINGS', 'INCOME_STATEMENT', 
        'BALANCE_SHEET', 'CASH_FLOW', 'DIVIDENDS', 
        'SPLITS', 'LISTING_STATUS', 'EARNINGS_ESTIMATES', 
        'EARNINGS_CALENDAR', 'EARNINGS_CALL_TRANSCRIPT'
    ],
    'economic': [
        'TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 
        'CPI', 'INFLATION', 'REAL_GDP'
    ],
    'sentiment': [
        'NEWS_SENTIMENT', 'TOP_GAINERS_LOSERS', 
        'INSIDER_TRANSACTIONS'
    ]
}
```

---

## 3. Configuration Architecture

### 3.1 Configuration Structure
```
config/
├── .env                          # Secrets ONLY
├── .env.example                  # Template
├── system/                       # System-wide settings
│   ├── database.yaml
│   ├── redis.yaml
│   ├── logging.yaml
│   └── paths.yaml
├── apis/                         # API configurations
│   ├── alpha_vantage.yaml       # All 43 endpoints
│   ├── ibkr.yaml
│   └── rate_limits.yaml
├── data/                         # Data management
│   ├── symbols.yaml
│   ├── schedules.yaml
│   ├── ingestion.yaml
│   └── validation.yaml
├── strategies/                   # Strategy parameters
│   ├── 0dte.yaml
│   ├── 1dte.yaml
│   ├── swing_14d.yaml
│   └── moc_imbalance.yaml
├── risk/                         # Risk management
│   ├── position_limits.yaml
│   ├── portfolio_limits.yaml
│   ├── circuit_breakers.yaml
│   └── sizing.yaml
├── ml/                          # ML configurations
│   ├── models.yaml
│   ├── features.yaml
│   └── thresholds.yaml
├── execution/                   # Trading execution
│   ├── trading_hours.yaml
│   ├── order_types.yaml
│   └── slippage.yaml
├── monitoring/                  # Output and monitoring
│   ├── alerts.yaml
│   ├── discord.yaml
│   └── dashboard.yaml
└── environments/               # Environment overrides
    ├── development.yaml
    ├── paper.yaml
    └── production.yaml
```

### 3.2 Configuration Manager
```python
class ConfigManager:
    """
    Central configuration management - NO HARDCODED VALUES
    """
    def __init__(self, environment='development'):
        self.environment = environment
        self.config = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        # Load base configurations
        self.config['system'] = self.load_yaml_dir('config/system/')
        self.config['apis'] = self.load_yaml_dir('config/apis/')
        self.config['data'] = self.load_yaml_dir('config/data/')
        self.config['strategies'] = self.load_yaml_dir('config/strategies/')
        self.config['risk'] = self.load_yaml_dir('config/risk/')
        self.config['ml'] = self.load_yaml_dir('config/ml/')
        self.config['execution'] = self.load_yaml_dir('config/execution/')
        self.config['monitoring'] = self.load_yaml_dir('config/monitoring/')
        
        # Apply environment overrides
        self.apply_environment_overrides()
        
        # Load secrets from .env
        self.load_secrets()
```

---

## 4. Architecture & Module Map (SKELETON-FIRST)

### 4.1 Directory Structure
```
src/
├── foundation/
│   ├── __init__.py
│   ├── config_manager.py        # Configuration management
│   └── base.py                  # Base classes for dependency injection
├── connections/
│   ├── __init__.py
│   ├── ibkr_connection.py       # IBKR TWS connection
│   └── av_client.py             # Alpha Vantage client
├── data/
│   ├── __init__.py
│   ├── rate_limiter.py          # Token bucket rate limiting
│   ├── scheduler.py             # API call scheduling
│   ├── ingestion.py             # Data normalization & storage
│   └── cache_manager.py         # Redis cache management
├── analytics/
│   ├── __init__.py
│   ├── indicator_processor.py   # Technical indicator processing
│   ├── greeks_validator.py      # Greeks validation
│   └── analytics_engine.py      # Analytics calculations
├── ml/
│   ├── __init__.py
│   ├── feature_builder.py       # Feature engineering
│   ├── model_suite.py           # Model loading and prediction
│   └── utils.py                 # ML utilities
├── decision/
│   ├── __init__.py
│   ├── decision_engine.py       # Master decision logic
│   └── strategy_engine.py       # Strategy orchestration
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py         # Base strategy class
│   ├── zero_dte.py             # 0DTE strategy
│   ├── one_dte.py              # 1DTE strategy
│   ├── swing_14d.py            # 14DTE swing strategy
│   └── moc_imbalance.py        # MOC imbalance strategy
├── risk/
│   ├── __init__.py
│   ├── risk_manager.py          # Risk management
│   └── position_sizer.py        # Position sizing
├── execution/
│   ├── __init__.py
│   └── ibkr_executor.py         # Order execution
├── monitoring/
│   ├── __init__.py
│   └── trade_monitor.py         # Trade monitoring
├── publishing/
│   ├── __init__.py
│   └── publisher.py             # Discord/alert publishing
└── api/
    ├── __init__.py
    └── dashboard_api.py          # FastAPI dashboard
```

### 4.2 Skeleton Implementation Order
```python
SKELETON_PHASES = {
    "Phase_0": {
        "name": "Base Skeleton",
        "modules": [
            "foundation/base.py",           # Abstract base classes
            "foundation/config_manager.py", # Config loader skeleton
        ],
        "purpose": "Establish inheritance and config patterns"
    },
    "Phase_1": {
        "name": "Connection Skeletons",
        "modules": [
            "connections/ibkr_connection.py",
            "connections/av_client.py"
        ],
        "purpose": "Define connection interfaces"
    },
    "Phase_2": {
        "name": "Data Layer Skeletons",
        "modules": [
            "data/rate_limiter.py",
            "data/scheduler.py",
            "data/ingestion.py",
            "data/cache_manager.py"
        ],
        "purpose": "Define data flow interfaces"
    },
    "Phase_3": {
        "name": "Processing Skeletons",
        "modules": [
            "analytics/indicator_processor.py",
            "analytics/greeks_validator.py",
            "analytics/analytics_engine.py",
            "ml/feature_builder.py",
            "ml/model_suite.py"
        ],
        "purpose": "Define processing pipelines"
    },
    "Phase_4": {
        "name": "Decision & Execution Skeletons",
        "modules": [
            "decision/decision_engine.py",
            "strategies/base_strategy.py",
            "strategies/zero_dte.py",
            "risk/risk_manager.py",
            "execution/ibkr_executor.py"
        ],
        "purpose": "Define decision and execution flow"
    },
    "Phase_5": {
        "name": "Output Skeletons",
        "modules": [
            "monitoring/trade_monitor.py",
            "publishing/publisher.py",
            "api/dashboard_api.py"
        ],
        "purpose": "Define output interfaces"
    }
}
```

---

## 5. Database Schema (EVOLUTIONARY)

### 5.1 Schema Evolution Strategy
```sql
-- IMPORTANT: Schema built incrementally based on API responses
-- DO NOT CREATE ALL TABLES AT ONCE

-- Phase 0: System tables only
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    api_name VARCHAR(50),
    table_name VARCHAR(100),
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_sample JSONB  -- Store sample API response
);

CREATE TABLE system_config (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tables will be added incrementally as each API is implemented
-- See Section 6 for API-driven table creation process
```

### 5.2 Table Creation Template
```sql
-- Template for each API's table (created after examining actual response)
CREATE TABLE IF NOT EXISTS {api_name}_{data_type} (
    -- Primary identification
    id SERIAL PRIMARY KEY,
    
    -- Common fields (adjust based on actual API response)
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    
    -- API-specific fields (discovered from response)
    -- ... fields determined by actual API response structure
    
    -- Metadata
    api_version VARCHAR(10),
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_response JSONB  -- Store complete response for debugging
);

-- Index template (created based on actual query patterns)
CREATE INDEX idx_{api_name}_{data_type}_lookup 
ON {api_name}_{data_type}(symbol, timestamp DESC);
```

---

## 6. API Implementation Process (CRITICAL)

### 6.1 API-by-API Implementation Flow
```python
API_IMPLEMENTATION_PROCESS = {
    "for_each_api": {
        "Step_1_Discovery": {
            "actions": [
                "Read API documentation",
                "Make test call with real symbol",
                "Save response to JSON file",
                "Analyze response structure",
                "Document rate limits observed",
                "Note any special parameters"
            ],
            "outputs": ["sample_response.json", "api_notes.md"]
        },
        "Step_2_Client_Implementation": {
            "actions": [
                "Create method in av_client.py or ibkr_connection.py",
                "Add configuration to config/apis/",
                "Implement error handling",
                "Add retry logic",
                "Test with multiple symbols"
            ],
            "outputs": ["working_api_method", "config_entry"]
        },
        "Step_3_Schema_Design": {
            "actions": [
                "Map JSON response to table columns",
                "Choose appropriate data types",
                "Design indexes for query patterns",
                "Create migration script",
                "Run migration"
            ],
            "outputs": ["migration_{api_name}.sql", "table_created"]
        },
        "Step_4_Ingestion_Implementation": {
            "actions": [
                "Create ingestion method",
                "Add data validation",
                "Implement normalization",
                "Add to scheduler if needed",
                "Test data persistence"
            ],
            "outputs": ["ingestion_method", "validated_data_flow"]
        },
        "Step_5_Testing": {
            "actions": [
                "Test with production symbol",
                "Verify data in database",
                "Test error scenarios",
                "Measure performance",
                "Document findings"
            ],
            "outputs": ["test_results.md", "performance_metrics"]
        },
        "Step_6_Integration": {
            "actions": [
                "Add to scheduler",
                "Configure rate limiting",
                "Update documentation",
                "Commit code"
            ],
            "outputs": ["integrated_api", "updated_docs"]
        }
    }
}
```

### 6.2 API Implementation Order
```python
# CRITICAL: Implement in this exact order for dependency management
API_IMPLEMENTATION_ORDER = [
    # Phase 1: IBKR Real-time Data (Foundation)
    {"api": "IBKR_Quotes", "priority": 1, "complexity": "Medium"},
    {"api": "IBKR_Bars_1min", "priority": 1, "complexity": "Medium"},
    {"api": "IBKR_Bars_5min", "priority": 1, "complexity": "Low"},
    
    # Phase 2: Alpha Vantage Greeks (Critical)
    {"api": "AV_REALTIME_OPTIONS", "priority": 1, "complexity": "High"},
    
    # Phase 3: Core Indicators
    {"api": "AV_RSI", "priority": 1, "complexity": "Low"},
    {"api": "AV_MACD", "priority": 1, "complexity": "Low"},
    {"api": "AV_BBANDS", "priority": 1, "complexity": "Low"},
    {"api": "AV_VWAP", "priority": 1, "complexity": "Low"},
    
    # Phase 4: Supporting Indicators
    {"api": "AV_ATR", "priority": 2, "complexity": "Low"},
    {"api": "AV_ADX", "priority": 2, "complexity": "Low"},
    {"api": "AV_STOCH", "priority": 2, "complexity": "Low"},
    
    # Phase 5: Volume Indicators
    {"api": "AV_OBV", "priority": 2, "complexity": "Low"},
    {"api": "AV_AD", "priority": 2, "complexity": "Low"},
    {"api": "AV_MFI", "priority": 2, "complexity": "Low"},
    
    # Phase 6: Analytics
    {"api": "AV_ANALYTICS_FIXED_WINDOW", "priority": 2, "complexity": "High"},
    {"api": "AV_ANALYTICS_SLIDING_WINDOW", "priority": 2, "complexity": "High"},
    
    # Phase 7: Sentiment & News
    {"api": "AV_NEWS_SENTIMENT", "priority": 3, "complexity": "Medium"},
    {"api": "AV_TOP_GAINERS_LOSERS", "priority": 3, "complexity": "Low"},
    
    # Phase 8: Fundamentals
    {"api": "AV_OVERVIEW", "priority": 3, "complexity": "Low"},
    {"api": "AV_EARNINGS_CALENDAR", "priority": 3, "complexity": "Medium"},
    
    # Phase 9: Economic Indicators
    {"api": "AV_TREASURY_YIELD", "priority": 3, "complexity": "Low"},
    {"api": "AV_FEDERAL_FUNDS_RATE", "priority": 3, "complexity": "Low"},
    
    # Phase 10: MOC Data
    {"api": "IBKR_MOC_Imbalance", "priority": 2, "complexity": "High"},
    
    # Continue for all 43 Alpha Vantage APIs...
]
```

---

## 7. Alpha Vantage API Catalog (Complete)

<!-- Using same complete API list from original, but marking which provide intraday -->
```python
# Complete list of all 43 Alpha Vantage APIs with configurations
AV_API_CATALOG = {
    # OPTIONS APIs (2)
    'REALTIME_OPTIONS': {
        'endpoint': 'https://www.alphavantage.co/query',
        'params': {
            'function': 'REALTIME_OPTIONS',
            'symbol': 'REQUIRED',
            'contract': 'OPTIONAL',
            'apikey': 'REQUIRED'
        },
        'provides_intraday': False,  # No intraday pricing
        'provides_greeks': True,      # PRIMARY GREEKS SOURCE
        'response_format': 'JSON',
        'cache_ttl': 10
    },
    
    'HISTORICAL_OPTIONS': {
        'endpoint': 'https://www.alphavantage.co/query',
        'params': {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': 'REQUIRED',
            'date': 'YYYY-MM-DD',
            'apikey': 'REQUIRED'
        },
        'provides_intraday': False,
        'provides_greeks': True,
        'response_format': 'JSON',
        'cache_ttl': 86400
    },
    
    # TECHNICAL INDICATORS (16)
    'RSI': {
        'params': {
            'function': 'RSI',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 14,
            'series_type': 'close',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'MOMENTUM_SIGNAL',
        'response_format': 'JSON',
        'cache_ttl': 60
    },
    
    'MACD': {
        'params': {
            'function': 'MACD',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'series_type': 'close',
            'fastperiod': 12,
            'slowperiod': 26,
            'signalperiod': 9,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'TREND_MOMENTUM',
        'response_format': 'JSON',
        'cache_ttl': 60
    },
    
    'STOCH': {
        'params': {
            'function': 'STOCH',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'fastkperiod': 5,
            'slowkperiod': 3,
            'slowdperiod': 3,
            'slowkmatype': 0,
            'slowdmatype': 0,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'OVERBOUGHT_OVERSOLD',
        'response_format': 'JSON',
        'cache_ttl': 60
    },
    
    'BBANDS': {
        'params': {
            'function': 'BBANDS',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 20,
            'series_type': 'close',
            'nbdevup': 2,
            'nbdevdn': 2,
            'matype': 0,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'VOLATILITY_BANDS',
        'response_format': 'JSON',
        'cache_ttl': 60
    },
    
    'ATR': {
        'params': {
            'function': 'ATR',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 14,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'VOLATILITY_MEASURE',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'ADX': {
        'params': {
            'function': 'ADX',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 14,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'TREND_STRENGTH',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'AROON': {
        'params': {
            'function': 'AROON',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 14,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'TREND_CHANGE_DETECTION',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'CCI': {
        'params': {
            'function': 'CCI',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 20,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'MEAN_REVERSION',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'MFI': {
        'params': {
            'function': 'MFI',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 14,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'VOLUME_WEIGHTED_MOMENTUM',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'WILLR': {
        'params': {
            'function': 'WILLR',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 14,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'OVERSOLD_OVERBOUGHT',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'MOM': {
        'params': {
            'function': 'MOM',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 10,
            'series_type': 'close',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'PRICE_MOMENTUM',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'VWAP': {
        'params': {
            'function': 'VWAP',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'INSTITUTIONAL_ACTIVITY',
        'response_format': 'JSON',
        'cache_ttl': 60
    },
    
    'EMA': {
        'params': {
            'function': 'EMA',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 'VARIABLE',
            'series_type': 'close',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'TREND_FOLLOWING',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'SMA': {
        'params': {
            'function': 'SMA',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'time_period': 'VARIABLE',
            'series_type': 'close',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'SUPPORT_RESISTANCE',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'OBV': {
        'params': {
            'function': 'OBV',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'VOLUME_TREND',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'AD': {
        'params': {
            'function': 'AD',
            'symbol': 'REQUIRED',
            'interval': '1min|5min|15min|30min|60min|daily|weekly|monthly',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'ACCUMULATION_DISTRIBUTION',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    # ANALYTICS APIs (2)
    'ANALYTICS_FIXED_WINDOW': {
        'params': {
            'function': 'ANALYTICS_FIXED_WINDOW',
            'SYMBOLS': 'symbol1,symbol2',
            'INTERVAL': '1min|5min|15min|30min|60min|DAILY|WEEKLY|MONTHLY',
            'CALCULATIONS': 'MIN,MAX,MEAN,MEDIAN,VARIANCE,STDDEV,CORRELATION',
            'RANGE': 'full|30day|60day|YYYY-MM-DD',
            'OHLC': 'close',
            'apikey': 'REQUIRED'
        },
        'calculations_available': [
            'MIN', 'MAX', 'MEAN', 'MEDIAN', 'CUMULATIVE_RETURN',
            'VARIANCE', 'VARIANCE(annualized=True)',
            'STDDEV', 'STDDEV(annualized=True)',
            'MAX_DRAWDOWN', 'HISTOGRAM(bins=20)',
            'AUTOCORRELATION(lag=2)',
            'COVARIANCE', 'COVARIANCE(annualized=True)',
            'CORRELATION', 'CORRELATION(method=KENDALL)',
            'CORRELATION(method=SPEARMAN)'
        ],
        'ml_usage': 'VOLATILITY_CALCULATIONS',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'ANALYTICS_SLIDING_WINDOW': {
        'params': {
            'function': 'ANALYTICS_SLIDING_WINDOW',
            'SYMBOLS': 'symbol1,symbol2',
            'INTERVAL': '1min|5min|15min|30min|60min|DAILY|WEEKLY|MONTHLY',
            'WINDOW_SIZE': 20,
            'CALCULATIONS': 'MEAN,STDDEV,CORRELATION',
            'RANGE': 'full|30day|60day',
            'OHLC': 'close',
            'apikey': 'REQUIRED'
        },
        'ml_usage': 'ROLLING_VOLATILITY_AND_CORRELATION',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    # SENTIMENT APIs (3)
    'NEWS_SENTIMENT': {
        'params': {
            'function': 'NEWS_SENTIMENT',
            'tickers': 'symbol1,symbol2',
            'topics': 'earnings,ipo,mergers_and_acquisitions,financial_markets',
            'time_from': 'YYYYMMDDTHHMM',
            'time_to': 'YYYYMMDDTHHMM',
            'sort': 'LATEST|EARLIEST|RELEVANCE',
            'limit': 1000,
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'SENTIMENT_SCORE',
        'response_format': 'JSON',
        'cache_ttl': 900
    },
    
    'TOP_GAINERS_LOSERS': {
        'params': {
            'function': 'TOP_GAINERS_LOSERS',
            'apikey': 'REQUIRED'
        },
        'usage': 'MARKET_MOMENTUM_SCAN',
        'response_format': 'JSON',
        'cache_ttl': 300
    },
    
    'INSIDER_TRANSACTIONS': {
        'params': {
            'function': 'INSIDER_TRANSACTIONS',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'INSIDER_SENTIMENT',
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    # ECONOMIC INDICATORS (5)
    'TREASURY_YIELD': {
        'params': {
            'function': 'TREASURY_YIELD',
            'interval': 'daily|weekly|monthly',
            'maturity': '3month|2year|5year|7year|10year|30year',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'RISK_FREE_RATE',
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'FEDERAL_FUNDS_RATE': {
        'params': {
            'function': 'FEDERAL_FUNDS_RATE',
            'interval': 'daily|weekly|monthly',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'MONETARY_POLICY',
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'CPI': {
        'params': {
            'function': 'CPI',
            'interval': 'monthly|semiannual',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'INFLATION_RATE',
        'response_format': 'JSON',
        'cache_ttl': 86400
    },
    
    'INFLATION': {
        'params': {
            'function': 'INFLATION',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'INFLATION_EXPECTATIONS',
        'response_format': 'JSON',
        'cache_ttl': 86400
    },
    
    'REAL_GDP': {
        'params': {
            'function': 'REAL_GDP',
            'interval': 'annual|quarterly',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'ECONOMIC_GROWTH',
        'response_format': 'JSON',
        'cache_ttl': 86400
    },
    
    # FUNDAMENTAL DATA (15)
    'OVERVIEW': {
        'params': {
            'function': 'OVERVIEW',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'data_includes': ['MarketCap', 'PE', 'EPS', 'Beta', 'DividendYield'],
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'EARNINGS': {
        'params': {
            'function': 'EARNINGS',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'usage': 'EARNINGS_HISTORY',
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'EARNINGS_ESTIMATES': {
        'params': {
            'function': 'EARNINGS_ESTIMATES',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'ml_feature': 'EARNINGS_SURPRISE_POTENTIAL',
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'EARNINGS_CALENDAR': {
        'params': {
            'function': 'EARNINGS_CALENDAR',
            'symbol': 'OPTIONAL',
            'horizon': '3month|6month|12month',
            'apikey': 'REQUIRED'
        },
        'response_format': 'CSV',
        'cache_ttl': 3600
    },
    
    'EARNINGS_CALL_TRANSCRIPT': {
        'params': {
            'function': 'EARNINGS_CALL_TRANSCRIPT',
            'symbol': 'REQUIRED',
            'quarter': 'YYYYQM',
            'apikey': 'REQUIRED'
        },
        'ml_usage': 'SENTIMENT_ANALYSIS',
        'response_format': 'JSON',
        'cache_ttl': 86400
    },
    
    'INCOME_STATEMENT': {
        'params': {
            'function': 'INCOME_STATEMENT',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'BALANCE_SHEET': {
        'params': {
            'function': 'BALANCE_SHEET',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'CASH_FLOW': {
        'params': {
            'function': 'CASH_FLOW',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'response_format': 'JSON',
        'cache_ttl': 3600
    },
    
    'DIVIDENDS': {
        'params': {
            'function': 'DIVIDENDS',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'response_format': 'JSON',
        'cache_ttl': 86400
    },
    
    'SPLITS': {
        'params': {
            'function': 'SPLITS',
            'symbol': 'REQUIRED',
            'apikey': 'REQUIRED'
        },
        'response_format': 'JSON',
        'cache_ttl': 86400
    },
    
    'LISTING_STATUS': {
        'params': {
            'function': 'LISTING_STATUS',
            'date': 'YYYY-MM-DD',
            'state': 'active|delisted',
            'apikey': 'REQUIRED'
        },
        'response_format': 'CSV',
        'cache_ttl': 86400
    }
}
```

---

## 8. Module Dependency Matrix (UPDATED)

```python
MODULE_DEPENDENCIES = {
    'foundation/config_manager.py': {
        'depends_on': [],
        'provides_to': ['ALL_MODULES'],
        'test_before_next': 'Config loads all YAML files correctly'
    },
    'connections/ibkr_connection.py': {
        'depends_on': ['config_manager'],
        'provides_to': ['data/ingestion', 'execution/ibkr_executor'],
        'test_before_next': 'TWS connection established, can fetch bars'
    },
    'connections/av_client.py': {
        'depends_on': ['config_manager', 'data/rate_limiter'],
        'provides_to': ['data/ingestion'],
        'test_before_next': 'All 43 endpoints callable with rate limiting'
    },
    'data/rate_limiter.py': {
        'depends_on': ['config_manager'],
        'provides_to': ['connections/av_client'],
        'test_before_next': 'Token bucket prevents >600/min'
    },
    'data/scheduler.py': {
        'depends_on': ['config_manager', 'connections/*'],
        'provides_to': ['data/ingestion'],
        'test_before_next': 'Scheduling logic triggers correctly'
    },
    'data/ingestion.py': {
        'depends_on': ['scheduler', 'cache_manager', 'database'],
        'provides_to': ['analytics/*', 'ml/*'],
        'test_before_next': 'Data normalized and stored correctly'
    },
    'data/cache_manager.py': {
        'depends_on': ['config_manager', 'redis'],
        'provides_to': ['data/ingestion', 'analytics/*'],
        'test_before_next': 'Redis read/write operations work'
    },
    'analytics/indicator_processor.py': {
        'depends_on': ['data/ingestion'],
        'provides_to': ['ml/feature_builder', 'decision/decision_engine'],
        'test_before_next': 'All indicators processed correctly'
    },
    'analytics/greeks_validator.py': {
        'depends_on': ['data/ingestion'],
        'provides_to': ['decision/decision_engine', 'risk/risk_manager'],
        'test_before_next': 'Greeks validation catches bad data'
    },
    'ml/feature_builder.py': {
        'depends_on': ['analytics/*'],
        'provides_to': ['ml/model_suite'],
        'test_before_next': 'Features extracted correctly'
    },
    'ml/model_suite.py': {
        'depends_on': ['ml/feature_builder'],
        'provides_to': ['decision/decision_engine'],
        'test_before_next': 'Models load and predict'
    },
    'decision/decision_engine.py': {
        'depends_on': ['ml/model_suite', 'analytics/*', 'strategies/*'],
        'provides_to': ['risk/risk_manager'],
        'test_before_next': 'Decision logic flows correctly'
    },
    'strategies/*.py': {
        'depends_on': ['decision/decision_engine'],
        'provides_to': ['risk/risk_manager'],
        'test_before_next': 'Strategy rules apply correctly'
    },
    'risk/risk_manager.py': {
        'depends_on': ['decision/decision_engine', 'portfolio_state'],
        'provides_to': ['execution/ibkr_executor'],
        'test_before_next': 'Risk limits enforced'
    },
    'execution/ibkr_executor.py': {
        'depends_on': ['risk/risk_manager', 'connections/ibkr_connection'],
        'provides_to': ['monitoring/trade_monitor'],
        'test_before_next': 'Orders execute in paper mode'
    },
    'monitoring/trade_monitor.py': {
        'depends_on': ['execution/ibkr_executor'],
        'provides_to': ['publishing/publisher'],
        'test_before_next': 'Positions tracked correctly'
    },
    'publishing/publisher.py': {
        'depends_on': ['monitoring/trade_monitor'],
        'provides_to': ['external_webhooks'],
        'test_before_next': 'Messages sent to Discord'
    },
    'api/dashboard_api.py': {
        'depends_on': ['ALL_READ_ONLY'],
        'provides_to': ['web_ui'],
        'test_before_next': 'Endpoints return correct data'
    }
}
```

---

## 9. Testing Strategy (API-DRIVEN)

### 9.1 Testing Phases
```python
TESTING_PHASES = {
    "Unit_Testing": {
        "scope": "Individual API methods",
        "approach": "Test each API with real calls",
        "validation": [
            "Response structure matches expectations",
            "Data types are correct",
            "Rate limiting works",
            "Error handling works"
        ]
    },
    "Integration_Testing": {
        "scope": "API -> Ingestion -> Database",
        "approach": "End-to-end data flow per API",
        "validation": [
            "Data persisted correctly",
            "Schema matches response",
            "Indexes work efficiently",
            "Cache operates correctly"
        ]
    },
    "System_Testing": {
        "scope": "Complete decision flow",
        "approach": "Paper trading with all components",
        "validation": [
            "Decisions made correctly",
            "Risk limits enforced",
            "Orders execute properly",
            "Monitoring accurate"
        ]
    }
}
```

### 9.2 API Test Checklist
```python
API_TEST_CHECKLIST = {
    "for_each_api": [
        "Make successful call with valid symbol",
        "Handle API errors gracefully",
        "Respect rate limits",
        "Parse response correctly",
        "Store data in correct table",
        "Retrieve data from database",
        "Cache works if applicable",
        "Performance meets targets",
        "Document any quirks found"
    ]
}
```

---

## 10. Implementation Timeline

### 10.1 Phase Timeline
```python
IMPLEMENTATION_TIMELINE = {
    "Week_1": {
        "goal": "Complete skeleton and infrastructure",
        "deliverables": [
            "All module files created",
            "Config management working",
            "Database and Redis connected",
            "Base classes implemented"
        ]
    },
    "Week_2-3": {
        "goal": "IBKR connection and data",
        "deliverables": [
            "IBKR connection established",
            "Real-time bars working",
            "Quote feed operational",
            "Data persisting to database"
        ]
    },
    "Week_4-6": {
        "goal": "Alpha Vantage core APIs",
        "deliverables": [
            "Options with Greeks working",
            "Core indicators (RSI, MACD, BBANDS, VWAP)",
            "Rate limiting tested",
            "Schema evolved for each API"
        ]
    },
    "Week_7-8": {
        "goal": "Remaining Alpha Vantage APIs",
        "deliverables": [
            "All 43 APIs implemented",
            "Complete schema in place",
            "Ingestion pipeline complete",
            "Scheduler operational"
        ]
    },
    "Week_9-10": {
        "goal": "Analytics and ML",
        "deliverables": [
            "Indicator processing",
            "Greeks validation",
            "Feature engineering",
            "Model integration"
        ]
    },
    "Week_11-12": {
        "goal": "Decision and execution",
        "deliverables": [
            "Decision engine complete",
            "All strategies implemented",
            "Risk management operational",
            "Paper trading execution"
        ]
    },
    "Week_13-14": {
        "goal": "Output and monitoring",
        "deliverables": [
            "Trade monitoring",
            "Discord publishing",
            "Dashboard API",
            "Complete system integration"
        ]
    },
    "Week_15-16": {
        "goal": "Paper trading validation",
        "deliverables": [
            "5+ days paper trading",
            "Performance gates met",
            "All failure modes tested",
            "Production readiness confirmed"
        ]
    }
}
```

---

## 11. Critical Success Factors

### 11.1 Must-Have Before Production
```python
PRODUCTION_REQUIREMENTS = {
    "technical": [
        "All 43 Alpha Vantage APIs tested and working",
        "IBKR real-time data reliable",
        "Rate limiting never exceeds 600/min",
        "All Greeks validated before use",
        "Schema matches all API responses",
        "Configuration fully externalized"
    ],
    "operational": [
        "5+ days successful paper trading",
        "Win rate > 45%",
        "All circuit breakers tested",
        "Emergency stop verified",
        "Backup procedures documented",
        "Monitoring dashboard operational"
    ],
    "risk": [
        "Position limits enforced",
        "Portfolio Greeks tracked",
        "Stop losses working",
        "Daily loss breaker tested",
        "Capital limits respected",
        "Slippage within tolerance"
    ]
}
```

### 11.2 Configuration Requirements
```python
CONFIGURATION_REQUIREMENTS = {
    "no_hardcoded_values": [
        "All API endpoints in config",
        "All rate limits in config",
        "All strategy parameters in config",
        "All risk limits in config",
        "All scheduling in config",
        "All thresholds in config"
    ],
    "environment_specific": [
        "Development config tested",
        "Paper trading config tested",
        "Production config validated",
        "Secrets never in code",
        "Easy parameter tuning",
        "Version controlled configs"
    ]
}
```

---

## 12. Documentation Requirements

### 12.1 Required Documentation
```python
DOCUMENTATION_REQUIREMENTS = {
    "per_api": [
        "Sample request",
        "Sample response",
        "Schema design",
        "Rate limits observed",
        "Quirks and gotchas",
        "Test results"
    ],
    "per_module": [
        "Class/function documentation",
        "Dependencies listed",
        "Configuration used",
        "Test coverage",
        "Performance metrics",
        "Error handling"
    ],
    "operational": [
        "Startup procedures",
        "Shutdown procedures",
        "Emergency procedures",
        "Monitoring guide",
        "Troubleshooting guide",
        "Configuration guide"
    ]
}
```

---

## Appendix A: Configuration Examples

### A.1 Strategy Configuration Example
```yaml
# config/strategies/0dte.yaml
strategy:
  name: "Zero DTE Strategy"
  enabled: true
  
confidence:
  minimum: 0.75
  ml_weight: 0.40
  indicators_weight: 0.30
  greeks_weight: 0.30

timing:
  entry_window:
    start: "09:45"
    end: "14:00"
  auto_close: "15:30"
  
position_limits:
  max_concurrent: 3
  max_per_symbol: 1
  min_premium: 0.50
  max_premium: 10.00

rules:
  rsi:
    enabled: true
    min_value: 30
    max_value: 70
    weight: 0.15
    
  delta:
    enabled: true
    min_abs_value: 0.25
    max_abs_value: 0.75
    weight: 0.20
    
  gamma:
    enabled: true
    max_value: 0.20
    weight: 0.15
    
  theta_decay:
    enabled: true
    min_ratio: 0.03  # theta/price
    weight: 0.15
    
  implied_volatility:
    enabled: true
    min_percentile: 20
    weight: 0.10
    
  volume:
    enabled: true
    min_ratio: 0.5  # current/average
    weight: 0.10
    
  bid_ask_spread:
    enabled: true
    max_spread: 0.10
    weight: 0.15

exit_rules:
  stop_loss: 0.25  # 25% loss
  take_profit: 0.50  # 50% gain
  time_stop: "15:30"  # Close all by 3:30 PM
```

### A.2 Risk Configuration Example
```yaml
# config/risk/position_limits.yaml
position_limits:
  greeks:
    max_delta: 0.80
    min_delta: 0.20
    max_gamma: 0.20
    max_vega: 200
    min_theta_ratio: 0.02
    
  sizing:
    max_position_size: 0.05  # 5% of capital
    max_contract_value: 1000  # $1000 max per contract
    min_contract_value: 50   # $50 min per contract
    
  entry_checks:
    - "delta_within_range"
    - "gamma_below_limit"
    - "theta_decay_sufficient"
    - "spread_acceptable"
    - "volume_sufficient"
    
# config/risk/portfolio_limits.yaml
portfolio_limits:
  greeks:
    max_net_delta: 0.30
    max_net_gamma: 0.75
    max_net_vega: 1000
    max_net_theta: -500
    
  exposure:
    max_capital_at_risk: 0.20  # 20% of total capital
    max_sector_concentration: 0.30  # 30% in one sector
    max_symbol_concentration: 0.15  # 15% in one symbol
    
  daily_limits:
    max_trades: 20
    max_loss: 0.02  # 2% daily loss limit
    max_consecutive_losses: 3
    
  circuit_breakers:
    daily_loss_threshold: 0.02
    weekly_loss_threshold: 0.05
    drawdown_threshold: 0.10
    action: "halt_new_trades"
```

---

## END OF SSOT-TECH.MD v2.0