# AlphaTrader Technical Specification v3.0
## Alpha Vantage + IBKR Dual-Source Architecture

---

## 1. EXECUTIVE SUMMARY

### 1.1 System Overview
AlphaTrader is a ML-driven options trading system that leverages Alpha Vantage's comprehensive data APIs for options analytics and Interactive Brokers for execution. The system uses Alpha Vantage's premium tier (600 calls/minute) to access real-time options data with Greeks, 20 years of historical options data, technical indicators, sentiment analysis, and market analytics, while IBKR handles real-time quotes, bars, and trade execution.

### 1.2 Key Architecture Change
**CRITICAL**: Greeks are PROVIDED by Alpha Vantage, not calculated locally. This eliminates the need for Black-Scholes calculations and ensures consistent, professional-grade Greeks across all options.

### 1.3 Data Source Architecture
```
Alpha Vantage (600 calls/min premium):
├── Real-time options chains with Greeks
├── 20 years historical options with Greeks  
├── 16 technical indicators (RSI, MACD, etc.)
├── News sentiment & analytics
├── Fundamental data
└── Economic indicators

Interactive Brokers:
├── Real-time spot prices & quotes
├── 5-second bars
├── Trade execution (paper & live)
└── Position management
```

### 1.4 Development Philosophy
- **Dual-source optimization**: Best data from each provider
- **Greeks as a service**: No local calculations needed
- **Cache-first design**: Minimize API calls, maximize performance  
- **Progressive complexity**: Start simple, add features incrementally
- **Observable by default**: Comprehensive monitoring of both data sources

### 1.5 Realistic Performance Targets

| Metric | Phase 1 Target | Phase 3 Target | Notes |
|--------|---------------|----------------|-------|
| Critical Path Latency | <300ms | <150ms | Includes AV API calls |
| Greeks Retrieval (cached) | <10ms | <5ms | From Redis cache |
| Greeks Fetch (API) | <500ms | <300ms | Direct from Alpha Vantage |
| AV API Efficiency | <10 calls/trade | <5 calls/trade | Through caching |
| ML Inference | <30ms | <15ms | Using AV features |
| Position Limit | 5 | 20 | Gradual increase |
| Daily Trades | 10 | 50 | As confidence grows |
| Discord Latency | N/A | <5 seconds | Async processing |

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Dual-Source Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   DATA LAYER (Dual Sources)                      │
├────────────────────────────────┬─────────────────────────────────┤
│   IBKR Connection              │   Alpha Vantage Client          │
│   - Real-time quotes           │   - Options chains w/ Greeks    │
│   - 5-second bars              │   - 20yr historical options     │
│   - Order execution            │   - Technical indicators        │
│   - Position management        │   - Sentiment analysis          │
└────────────┬───────────────────┴────────────┬────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DATA INTEGRATION LAYER                           │
├────────────────────────────────┬─────────────────────────────────┤
│   MarketDataManager            │   OptionsDataManager            │
│   - Spot prices (IBKR)         │   - Options data (AV)           │
│   - Price bars (IBKR)          │   - Greeks retrieval (AV)       │
│   - Execution routing (IBKR)   │   - IV analysis (AV)            │
└────────────┬───────────────────┴────────────┬────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              ANALYTICS LAYER (AV-Powered)                        │
├────────────────────────────────┬─────────────────────────────────┤
│   FeatureEngine                │   MLPredictor                   │
│   - Price features (IBKR)      │   - XGBoost model               │
│   - Technical indicators (AV)  │   - Trained on AV historical    │
│   - Options metrics (AV)       │   - 20 years of data available  │
│   - Sentiment scores (AV)      │   - Confidence scoring          │
└────────────┬───────────────────┴────────────┬────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│               TRADING LAYER (Hybrid)                             │
├────────────────────────────────┬─────────────────────────────────┤
│   SignalGenerator              │   RiskManager                   │
│   - Uses AV analytics          │   - Greeks limits (from AV)     │
│   - Combines both sources      │   - Position sizing (AV prices) │
│   - Entry/exit rules           │   - Portfolio Greeks (AV data)  │
└────────────┬───────────────────┴────────────┬────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXECUTION LAYER (IBKR)                              │
├────────────────────────────────┬─────────────────────────────────┤
│   PaperTrader                  │   LiveTrader                    │
│   - IBKR paper account         │   - IBKR live account           │
│   - Simulated fills            │   - Real fills                  │
│   - Tracks with AV Greeks      │   - Monitors with AV Greeks     │
└────────────┬───────────────────┴────────────┬────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│            COMMUNITY LAYER (AV Analytics)                        │
├────────────────────────────────┬─────────────────────────────────┤
│   SignalPublisher              │   DiscordBot                    │
│   - Includes AV Greeks         │   - Shows AV indicators         │
│   - Shows AV sentiment         │   - Publishes AV analytics      │
│   - Performance with AV data   │   - Real-time Greeks updates    │
└────────────────────────────────┴─────────────────────────────────┘
```

### 2.2 Data Flow Design

#### Synchronous Trading Pipeline
```
IBKR Quotes → AV Options → Features → ML Model → Signal → Risk → IBKR Order
    10ms    →    60ms    →   30ms   →   30ms   →  10ms  → 20ms →   100ms
                        Total: 260ms (Phase 1 with caching)
```

#### Parallel Data Fetching Strategy
```
┌──────────────┐
│   Trigger    │
└──────┬───────┘
       │
   ┌───┴────┐
   │ Parallel│
   └───┬────┘
       │
   ┌───┼─────────────┬──────────────┬────────────┐
   ▼   ▼             ▼              ▼            ▼
[IBKR Quotes]  [AV Options]  [AV Indicators]  [AV Sentiment]
   10ms           60ms           40ms            50ms
   │   │             │              │            │
   └───┼─────────────┼──────────────┼────────────┘
       ▼             ▼              ▼
       └─────────────┴──────────────┘
                     │
                     ▼
              [Feature Vector]
                   30ms
```

#### Caching Strategy
```
Data Type               | Cache TTL | Hit Target | Miss Penalty
------------------------|-----------|------------|-------------
Options (real-time)     | 60s       | 90%        | 500ms
Options (historical)    | 1 hour    | 95%        | 800ms
Technical Indicators    | 5 min     | 85%        | 400ms
Sentiment              | 15 min    | 90%        | 600ms
Fundamentals           | 24 hours  | 99%        | 1000ms
```

---

## 3. COMPONENT SPECIFICATIONS

### 3.1 Data Layer Components

#### AlphaVantageClient
```python
class AlphaVantageClient:
    """
    Premium Alpha Vantage API client (600 calls/minute)
    Provides all options, Greeks, indicators, and analytics
    """
    def __init__(self, config):
        self.api_key = config.av_api_key
        self.rate_limiter = RateLimiter(600, 60)  # 600/min
        self.cache = RedisCache()
        self.metrics = MetricsCollector()
        
    async def get_realtime_options(self, symbol, require_greeks=True):
        """
        Get real-time options WITH GREEKS
        Greeks are PROVIDED - no calculation needed!
        Returns: List[OptionContract] with delta, gamma, theta, vega, rho
        """
        
    async def get_historical_options(self, symbol, date):
        """
        Get historical options - up to 20 YEARS with Greeks!
        Perfect for ML training
        """
        
    async def get_technical_indicator(self, symbol, indicator, **params):
        """
        Get any of 16 technical indicators
        No TA-Lib needed - AV provides everything
        """
        
    async def get_news_sentiment(self, symbols):
        """
        Get news sentiment scores and analytics
        """
        
    async def get_analytics_fixed_window(self, symbols, calculations):
        """
        Get advanced analytics (correlation, variance, etc.)
        """
```

#### MarketDataManager (IBKR)
```python
class MarketDataManager:
    """
    IBKR connection for quotes, bars, and execution
    Complements Alpha Vantage options data
    """
    def __init__(self, config):
        self.ib = IB()
        self.execution_mode = config.mode  # paper/live
        
    async def get_latest_price(self, symbol):
        """Get spot price for options pricing"""
        
    async def get_bars(self, symbol, duration):
        """Get price bars for price action analysis"""
        
    async def execute_order(self, contract, order):
        """Execute trades through IBKR"""
```

#### OptionsDataManager
```python
class OptionsDataManager:
    """
    Manages options data from Alpha Vantage
    Greeks are RETRIEVED, not calculated!
    """
    def __init__(self, market_data_mgr, av_client):
        self.market = market_data_mgr  # IBKR for spot prices
        self.av = av_client  # Alpha Vantage for everything else
        self.chains = {}
        self.greeks_cache = {}  # Greeks from AV, not calculated!
        
    async def fetch_option_chain(self, symbol):
        """
        Fetch from Alpha Vantage with Greeks included
        No Black-Scholes calculation needed!
        """
        options = await self.av.get_realtime_options(symbol, require_greeks=True)
        
        # Greeks are already in the response!
        for option in options:
            print(f"Greeks from AV: Δ={option.delta}, Γ={option.gamma}")
            
        return options
        
    def get_option_greeks(self, symbol, strike, expiry, option_type):
        """
        Get Greeks - just retrieval from AV cache
        NO CALCULATION - Greeks come from Alpha Vantage!
        """
        # Simply return cached Greeks from Alpha Vantage
        return self.greeks_cache.get(key, default_greeks)
```

### 3.2 Analytics Layer

#### FeatureEngine
```python
class FeatureEngine:
    """
    Feature engineering using dual data sources
    IBKR for price action, Alpha Vantage for everything else
    """
    def __init__(self, options_mgr, av_client):
        self.options = options_mgr
        self.av = av_client
        
        self.feature_sources = {
            # IBKR features
            'price_action': ['returns_5m', 'returns_30m', 'volume_ratio'],
            
            # Alpha Vantage features  
            'technical': ['rsi', 'macd', 'bbands', 'atr', 'adx', 'obv'],
            'options': ['iv_rank', 'put_call_ratio', 'gamma_exposure'],
            'greeks': ['atm_delta', 'atm_gamma', 'atm_theta', 'atm_vega'],
            'sentiment': ['news_score', 'insider_sentiment'],
            'market': ['spy_correlation', 'vix_level']
        }
        
    async def calculate_features(self, symbol):
        """
        Parallel feature calculation from both sources
        """
        # Parallel API calls to Alpha Vantage
        tasks = [
            self.av.get_technical_indicator(symbol, 'RSI'),
            self.av.get_technical_indicator(symbol, 'MACD'),
            self.av.get_technical_indicator(symbol, 'BBANDS'),
            self.av.get_realtime_options(symbol),
            self.av.get_news_sentiment([symbol])
        ]
        
        results = await asyncio.gather(*tasks)
        # Process all AV data into features
        return feature_vector
```

#### MLPredictor
```python
class MLPredictor:
    """
    ML model trained on Alpha Vantage historical data
    Can use 20 years of options history!
    """
    def __init__(self, feature_engine):
        self.features = feature_engine
        self.model = self.load_or_create_model()
        
    async def train_with_av_historical(self, symbols, years_back=5):
        """
        Train on Alpha Vantage historical options
        Up to 20 years available with Greeks!
        """
        print(f"Training on {years_back} years of AV data...")
        
        for symbol in symbols:
            # Alpha Vantage provides complete historical Greeks
            historical = await self.av.get_historical_options_range(
                symbol, years_back * 365
            )
            
            # Train on millions of data points with real Greeks
            self.train_on_historical(historical)
```

### 3.3 Trading Layer

#### SignalGenerator
```python
class SignalGenerator:
    """
    Generates signals using Alpha Vantage analytics
    """
    def __init__(self, ml_model, feature_engine, options_data):
        self.ml = ml_model
        self.features = feature_engine
        self.options = options_data
        
    async def generate_signals(self, symbols):
        """
        Generate signals with AV data
        """
        signals = []
        
        for symbol in symbols:
            # Get features from both IBKR and AV
            features = await self.features.calculate_features(symbol)
            
            # ML prediction
            signal_type, confidence = self.ml.predict(features)
            
            if signal_type != 'HOLD':
                # Select option using AV Greeks
                option = await self._select_option_with_av_greeks(symbol, signal_type)
                
                signals.append({
                    'symbol': symbol,
                    'signal': signal_type,
                    'confidence': confidence,
                    'option': option,
                    'av_greeks': option['greeks'],  # Include AV Greeks
                    'av_iv': option['implied_volatility']
                })
                
        return signals
```

#### RiskManager
```python
class RiskManager:
    """
    Risk management using Alpha Vantage Greeks
    """
    def __init__(self, config, options_data):
        self.config = config
        self.options = options_data  # Gets Greeks from AV
        
        # Portfolio Greeks limits (monitored via AV)
        self.greeks_limits = {
            'delta': (-0.3, 0.3),
            'gamma': (-0.5, 0.5),
            'vega': (-500, 500),
            'theta': (-200, float('inf'))
        }
        
    async def can_trade(self, signal):
        """
        Validate trade using AV Greeks
        """
        # Get Greeks from signal (from Alpha Vantage)
        av_greeks = signal['av_greeks']
        
        # Check portfolio impact
        for greek, (min_val, max_val) in self.greeks_limits.items():
            new_value = self.portfolio_greeks[greek] + av_greeks[greek] * 5
            if not (min_val <= new_value <= max_val):
                return False, f"Greeks limit breach: {greek}={new_value:.3f}"
                
        return True, "OK"
        
    async def update_portfolio_greeks(self):
        """
        Update portfolio Greeks using fresh AV data
        """
        total_greeks = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        for symbol, position in self.positions.items():
            # Get current Greeks from Alpha Vantage
            current_greeks = await self.options.av.get_option_greeks(
                symbol, position['strike'], position['expiry']
            )
            
            for greek in total_greeks:
                total_greeks[greek] += current_greeks[greek] * position['quantity']
                
        self.portfolio_greeks = total_greeks
```

### 3.4 Execution Layer

#### PaperTrader
```python
class PaperTrader:
    """
    Paper trading with IBKR execution and AV analytics
    """
    def __init__(self, components):
        self.market = components['market']  # IBKR
        self.av = components['av']  # Alpha Vantage
        self.signals = components['signals']
        self.risk = components['risk']
        
    async def execute_trade(self, signal):
        """
        Execute paper trade with full AV analytics
        """
        # Get option details from Alpha Vantage
        av_option = await self.av.get_option_details(
            signal['symbol'], 
            signal['option']['strike'],
            signal['option']['expiry']
        )
        
        # Store trade with AV Greeks
        trade = {
            'symbol': signal['symbol'],
            'strike': signal['option']['strike'],
            'entry_greeks': av_option.greeks,  # From Alpha Vantage!
            'entry_iv': av_option.implied_volatility,
            'entry_price': av_option.mid_price
        }
        
        # Execute through IBKR paper account
        await self.market.execute_paper_order(trade)
```

---

## 4. DATABASE SCHEMA

### 4.1 Enhanced Schema for Dual Sources

```sql
-- Trades table with Alpha Vantage Greeks
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Execution data (IBKR)
    symbol VARCHAR(10),
    option_type VARCHAR(4),
    strike DECIMAL(10,2),
    expiry DATE,
    action VARCHAR(10),
    quantity INT,
    fill_price DECIMAL(10,4),  -- From IBKR
    commission DECIMAL(10,2),
    
    -- Alpha Vantage Greeks at entry
    entry_delta DECIMAL(6,4),
    entry_gamma DECIMAL(6,4),
    entry_theta DECIMAL(8,4),
    entry_vega DECIMAL(8,4),
    entry_rho DECIMAL(6,4),
    entry_iv DECIMAL(6,4),
    
    -- Alpha Vantage Greeks at exit
    exit_delta DECIMAL(6,4),
    exit_gamma DECIMAL(6,4),
    exit_theta DECIMAL(8,4),
    exit_vega DECIMAL(8,4),
    exit_iv DECIMAL(6,4),
    
    -- P&L
    realized_pnl DECIMAL(10,2),
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol (symbol)
);

-- Alpha Vantage API metrics
CREATE TABLE av_api_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    endpoint VARCHAR(50),  -- REALTIME_OPTIONS, RSI, etc.
    symbol VARCHAR(10),
    response_time_ms INT,
    cache_hit BOOLEAN,
    rate_limit_remaining INT,
    response_size_bytes INT,
    
    INDEX idx_endpoint (endpoint),
    INDEX idx_timestamp (timestamp)
);

-- Options chain snapshots from Alpha Vantage
CREATE TABLE av_options_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    symbol VARCHAR(10),
    expiry DATE,
    strike DECIMAL(10,2),
    option_type VARCHAR(4),
    
    -- Prices from AV
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    last DECIMAL(10,4),
    volume INT,
    open_interest INT,
    
    -- Greeks from AV (not calculated!)
    delta DECIMAL(6,4),
    gamma DECIMAL(6,4),
    theta DECIMAL(8,4),
    vega DECIMAL(8,4),
    rho DECIMAL(6,4),
    implied_volatility DECIMAL(6,4),
    
    INDEX idx_symbol_expiry (symbol, expiry),
    INDEX idx_timestamp (timestamp)
);

-- Signal tracking with data sources
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    symbol VARCHAR(10),
    signal_type VARCHAR(20),
    confidence DECIMAL(4,3),
    
    -- Feature sources
    ibkr_features JSONB,  -- Price action features
    av_technical JSONB,   -- AV technical indicators
    av_options JSONB,     -- AV options metrics
    av_sentiment JSONB,   -- AV sentiment scores
    
    executed BOOLEAN DEFAULT FALSE,
    trade_id INT REFERENCES trades(id)
);

-- Cache performance tracking
CREATE TABLE cache_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    cache_type VARCHAR(20),  -- options, indicators, sentiment
    hits INT,
    misses INT,
    avg_response_cached_ms DECIMAL(8,2),
    avg_response_uncached_ms DECIMAL(8,2),
    memory_usage_mb INT
);
```

---

## 5. CONFIGURATION MANAGEMENT

### 5.1 Comprehensive Dual-Source Configuration

```yaml
# config.yaml - Dual source configuration
system:
  name: AlphaTrader
  version: 3.0
  environment: development  # development/staging/production

# Data source configuration
data_sources:
  # Interactive Brokers - Execution & Quotes
  ibkr:
    host: 127.0.0.1
    port: 7497  # 7496 for live
    client_id: 1
    connection_timeout: 30
    heartbeat_interval: 10
    
    responsibilities:
      - real_time_quotes
      - price_bars_5sec
      - order_execution
      - position_management
      
  # Alpha Vantage - Analytics & Options
  alpha_vantage:
    api_key: ${AV_API_KEY}  # Environment variable
    tier: premium  # 600 calls/minute
    rate_limit: 600
    rate_window: 60  # seconds
    
    # Connection settings
    timeout: 30
    retry_count: 3
    retry_delay: 1
    concurrent_requests: 10
    
    # Cache configuration by data type
    cache_config:
      options:
        ttl: 60  # Real-time options cache for 1 minute
        max_size: 1000
      historical_options:
        ttl: 3600  # Historical data cache for 1 hour
        max_size: 10000
      indicators:
        ttl: 300  # Technical indicators cache for 5 minutes
        max_size: 500
      sentiment:
        ttl: 900  # News sentiment cache for 15 minutes
        max_size: 100
      fundamentals:
        ttl: 86400  # Fundamental data cache for 1 day
        max_size: 100
        
    # Enabled APIs (38 total available)
    enabled_apis:
      # Options APIs
      - REALTIME_OPTIONS
      - HISTORICAL_OPTIONS
      
      # Technical Indicators
      - RSI
      - MACD
      - STOCH
      - WILLR
      - MOM
      - BBANDS
      - ATR
      - ADX
      - AROON
      - CCI
      - EMA
      - SMA
      - MFI
      - OBV
      - AD
      - VWAP
      
      # Analytics
      - ANALYTICS_FIXED_WINDOW
      - ANALYTICS_SLIDING_WINDOW
      
      # Sentiment
      - NEWS_SENTIMENT
      - TOP_GAINERS_LOSERS
      - INSIDER_TRANSACTIONS
      
      # Fundamentals
      - OVERVIEW
      - EARNINGS
      - INCOME_STATEMENT
      - BALANCE_SHEET
      - CASH_FLOW
      
      # Economic
      - TREASURY_YIELD
      - FEDERAL_FUNDS_RATE
      - CPI
      - INFLATION
      - REAL_GDP

# Trading configuration
trading:
  mode: paper  # paper/live
  symbols: [SPY, QQQ, IWM]
  
  # Feature configuration by source
  features:
    from_ibkr:
      - spot_price
      - price_bars
      - volume
      - bid_ask_spread
      
    from_alpha_vantage:
      - options_chains
      - greeks  # PROVIDED, not calculated!
      - implied_volatility
      - technical_indicators
      - news_sentiment
      - market_analytics
      
  # Signal generation
  signals:
    min_confidence: 0.6
    max_signals_per_symbol: 3
    cooldown_seconds: 300
    
  # Execution routing
  execution:
    router: ibkr  # All orders through IBKR
    slippage_model: conservative
    commission_per_contract: 0.65

# Risk management with AV Greeks
risk:
  # Position limits
  max_positions: 5
  max_position_size: 10000
  position_sizing_method: kelly
  
  # Loss limits
  daily_loss_limit: 1000
  weekly_loss_limit: 3000
  monthly_loss_limit: 5000
  
  # Greeks limits (using Alpha Vantage data)
  portfolio_greeks:
    delta:
      min: -0.3
      max: 0.3
      source: alpha_vantage  # Greeks from AV!
    gamma:
      min: -0.5
      max: 0.5
      source: alpha_vantage
    vega:
      min: -500
      max: 500
      source: alpha_vantage
    theta:
      min: -200
      max: null  # No upper limit on theta
      source: alpha_vantage
      
  # Risk monitoring
  monitoring:
    check_interval: 30  # seconds
    alert_on_breach: true
    auto_close_on_limit: true

# Machine Learning configuration
ml:
  model:
    type: xgboost
    path: models/xgboost_v3.pkl
    version: 3.0
    
  training:
    data_source: alpha_vantage  # 20 years available!
    history_years: 5  # Use 5 years for training
    retrain_interval_days: 30
    validation_split: 0.2
    
  features:
    total_count: 45
    price_features: 5  # From IBKR
    technical_features: 16  # From Alpha Vantage
    options_features: 12  # From Alpha Vantage
    sentiment_features: 4  # From Alpha Vantage
    market_features: 8  # From Alpha Vantage
    
  inference:
    batch_size: 1
    timeout_ms: 100
    fallback_to_rules: true

# Database configuration
database:
  postgres:
    host: localhost
    port: 5432
    database: alphatrader
    user: postgres
    password: ${DB_PASSWORD}
    pool_size: 20
    
  redis:
    host: localhost
    port: 6379
    db: 0
    password: ${REDIS_PASSWORD}
    
    # Cache namespaces
    namespaces:
      av_options: 1
      av_indicators: 2
      av_sentiment: 3
      ibkr_quotes: 4
      ml_features: 5

# Monitoring and alerting
monitoring:
  log_level: INFO
  log_file: logs/alphatrader.log
  
  # Metrics collection
  metrics:
    enabled: true
    port: 9090
    interval: 10  # seconds
    
  # Health checks
  health_checks:
    - name: ibkr_connection
      interval: 60
      timeout: 10
      critical: true
      
    - name: av_api_health
      interval: 300
      timeout: 30
      critical: true
      
    - name: av_rate_limit
      interval: 30
      threshold: 100  # Alert if <100 calls remaining
      critical: false
      
    - name: database_connection
      interval: 60
      timeout: 5
      critical: true
      
    - name: redis_connection
      interval: 60
      timeout: 5
      critical: false
      
  # Alerting
  alerts:
    discord_webhook: ${DISCORD_WEBHOOK}
    email: ${ALERT_EMAIL}
    
    triggers:
      - metric: av_rate_limit_remaining
        condition: "<"
        threshold: 50
        severity: warning
        
      - metric: daily_loss
        condition: ">"
        threshold: 800
        severity: critical
        
      - metric: api_error_rate
        condition: ">"
        threshold: 0.05
        severity: warning

# Community features
community:
  discord:
    enabled: true
    bot_token: ${DISCORD_BOT_TOKEN}
    
    channels:
      signals: 123456789  # Channel ID
      performance: 234567890
      analytics: 345678901
      alerts: 456789012
      
    # Publishing delays by tier
    tiers:
      free:
        delay_seconds: 300
        max_signals_daily: 5
        show_greeks: false
        
      premium:
        delay_seconds: 30
        max_signals_daily: 20
        show_greeks: true
        
      vip:
        delay_seconds: 0
        max_signals_daily: -1  # Unlimited
        show_greeks: true
        show_analytics: true
```

---

## 6. API INTEGRATION SPECIFICATIONS

### 6.1 Alpha Vantage API Usage

#### Options APIs
```python
# REALTIME_OPTIONS - Get current options with Greeks
params = {
    'function': 'REALTIME_OPTIONS',
    'symbol': 'SPY',
    'require_greeks': 'true',  # ALWAYS request Greeks
    'apikey': api_key
}
# Returns: Options chain with delta, gamma, theta, vega, rho

# HISTORICAL_OPTIONS - Get historical options (20 years!)
params = {
    'function': 'HISTORICAL_OPTIONS',
    'symbol': 'SPY',
    'date': '2023-01-15',  # Any date in past 20 years
    'apikey': api_key
}
# Returns: Historical options with Greeks included
```

#### Technical Indicators (16 types)
```python
# All indicators available without local calculation
indicators = [
    'RSI', 'MACD', 'STOCH', 'WILLR', 'MOM', 'BBANDS',
    'ATR', 'ADX', 'AROON', 'CCI', 'EMA', 'SMA',
    'MFI', 'OBV', 'AD', 'VWAP'
]

for indicator in indicators:
    params = {
        'function': indicator,
        'symbol': symbol,
        'interval': '5min',  # or daily, weekly, monthly
        'apikey': api_key,
        **indicator_specific_params
    }
```

#### Analytics APIs
```python
# ANALYTICS_FIXED_WINDOW - Advanced calculations
params = {
    'SYMBOLS': 'SPY,QQQ',  # Note: uppercase params
    'INTERVAL': 'DAILY',
    'RANGE': '1month',
    'CALCULATIONS': 'MEAN,STDDEV,CORRELATION,MAX_DRAWDOWN',
    'apikey': api_key
}

# ANALYTICS_SLIDING_WINDOW - Rolling calculations
params = {
    'SYMBOLS': 'SPY',
    'INTERVAL': 'DAILY',
    'RANGE': '6month',
    'WINDOW_SIZE': 90,
    'CALCULATIONS': 'MEAN,VARIANCE,CORRELATION',
    'apikey': api_key
}
```

### 6.2 IBKR Integration

```python
# Real-time quotes and bars
contract = Stock('SPY', 'SMART', 'USD')
bars = ib.reqRealTimeBars(contract, 5, 'TRADES', False)

# Order execution
option_contract = Option('SPY', '20240119', 450, 'C', 'SMART')
order = MarketOrder('BUY', 5)
trade = ib.placeOrder(option_contract, order)
```

---

## 7. PERFORMANCE OPTIMIZATION

### 7.1 Caching Strategy

```python
class CacheManager:
    """
    Multi-tier caching for Alpha Vantage data
    """
    def __init__(self):
        self.l1_cache = {}  # In-memory (microseconds)
        self.l2_cache = Redis()  # Redis (milliseconds)
        self.l3_cache = PostgreSQL()  # Database (for historical)
        
    async def get_with_cache(self, key, fetch_func, ttl):
        # L1: Check memory
        if key in self.l1_cache:
            return self.l1_cache[key]
            
        # L2: Check Redis
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value
            
        # L3: Fetch from API
        value = await fetch_func()
        
        # Cache in all tiers
        self.l1_cache[key] = value
        await self.l2_cache.set(key, value, ttl)
        
        return value
```

### 7.2 Rate Limit Management

```python
class RateLimiter:
    """
    Smart rate limiting for 600 calls/minute
    """
    def __init__(self):
        self.calls_per_minute = 600
        self.bucket = 600
        self.last_refill = time.time()
        
    async def acquire(self, cost=1):
        # Refill bucket
        now = time.time()
        elapsed = now - self.last_refill
        refill = elapsed * (self.calls_per_minute / 60)
        self.bucket = min(600, self.bucket + refill)
        self.last_refill = now
        
        # Check if we can make call
        if self.bucket >= cost:
            self.bucket -= cost
            return True
        else:
            # Wait for refill
            wait_time = (cost - self.bucket) / (self.calls_per_minute / 60)
            await asyncio.sleep(wait_time)
            return await self.acquire(cost)
```

---

## 8. MONITORING & OBSERVABILITY

### 8.1 Key Metrics

#### Alpha Vantage Metrics
- API calls per minute (target: <500/600)
- Cache hit rate (target: >80%)
- Average response time (target: <300ms)
- Greeks data freshness (target: <60s)
- Rate limit buffer (target: >100 calls)

#### IBKR Metrics
- Connection uptime (target: >99.9%)
- Order fill rate (target: >95%)
- Average fill time (target: <500ms)
- Slippage (target: <0.1%)

#### System Metrics
- End-to-end latency (target: <200ms)
- Signal generation rate (target: 10-50/day)
- Position Greeks within limits (target: 100%)
- Daily P&L vs target

### 8.2 Dashboards

```python
# Prometheus metrics
av_api_calls = Counter('av_api_calls_total', 'Total AV API calls')
av_cache_hits = Counter('av_cache_hits_total', 'AV cache hits')
av_response_time = Histogram('av_response_seconds', 'AV response time')
portfolio_greeks = Gauge('portfolio_greeks', 'Portfolio Greeks', ['greek'])

# Grafana dashboards
dashboards = [
    'Data Source Health',  # IBKR + AV status
    'API Performance',  # Response times, rate limits
    'Trading Metrics',  # Signals, executions, P&L
    'Risk Monitoring',  # Greeks, position limits
    'Cache Performance'  # Hit rates, memory usage
]
```

---

## 9. ERROR HANDLING & RECOVERY

### 9.1 Graceful Degradation

```python
class DataSourceManager:
    """
    Handles failures in either data source
    """
    async def get_options_data(self, symbol):
        try:
            # Primary: Alpha Vantage
            return await self.av.get_realtime_options(symbol)
        except AVException:
            # Fallback: Use cached data
            cached = await self.cache.get_latest_options(symbol)
            if cached:
                return cached
            else:
                # Last resort: Skip this symbol
                logger.error(f"No options data for {symbol}")
                return None
                
    async def get_quote(self, symbol):
        try:
            # Primary: IBKR
            return await self.ibkr.get_quote(symbol)
        except IBKRException:
            # Fallback: Use last known price
            return self.last_prices.get(symbol)
```

### 9.2 Circuit Breakers

```python
class CircuitBreaker:
    """
    Prevents cascading failures
    """
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure = None
        self.state = 'closed'  # closed, open, half-open
        
    async def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure > self.timeout:
                self.state = 'half-open'
            else:
                raise CircuitBreakerOpen()
                
        try:
            result = await func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                
            raise e
```

---

## 10. TESTING STRATEGY

### 10.1 Unit Tests
```python
# Test Alpha Vantage Greeks retrieval
async def test_av_greeks_retrieval():
    options = await av_client.get_realtime_options('SPY')
    assert options[0].delta is not None
    assert -1 <= options[0].delta <= 1
    
# Test dual-source integration
async def test_dual_source_features():
    features = await feature_engine.calculate_features('SPY')
    assert 'spot_price' in features  # From IBKR
    assert 'rsi' in features  # From Alpha Vantage
    assert 'atm_delta' in features  # From Alpha Vantage
```

### 10.2 Integration Tests
```python
# Test complete signal generation pipeline
async def test_signal_generation_pipeline():
    # Setup
    await market_data.connect()  # IBKR
    await av_client.connect()    # Alpha Vantage
    
    # Generate signal
    signals = await signal_generator.generate_signals(['SPY'])
    
    # Verify signal has all required data
    assert signals[0]['av_greeks'] is not None
    assert signals[0]['confidence'] > 0.6
```

### 10.3 Performance Tests
```python
# Test Alpha Vantage API performance
async def test_av_performance():
    start = time.time()
    
    # Parallel API calls
    tasks = [
        av_client.get_realtime_options('SPY'),
        av_client.get_technical_indicator('SPY', 'RSI'),
        av_client.get_news_sentiment(['SPY'])
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in under 1 second
    assert all(r is not None for r in results)
```

---

## 11. DEPLOYMENT STRATEGY

### 11.1 Development Environment
- Local PostgreSQL and Redis
- IBKR paper account
- Alpha Vantage premium key (dev)
- Full logging and debugging

### 11.2 Staging Environment
- Dedicated servers for data sources
- IBKR paper account (separate)
- Alpha Vantage premium key (staging)
- Performance monitoring
- Load testing with historical data

### 11.3 Production Environment
- High-availability database cluster
- Redis cluster for caching
- IBKR live account with failover
- Multiple Alpha Vantage API keys
- Full monitoring and alerting
- Automated backups

---

## 12. SUCCESS CRITERIA

### Phase 1 (Weeks 1-4)
- ✅ Alpha Vantage integration complete
- ✅ Greeks retrieved, not calculated
- ✅ 10+ signals generated daily
- ✅ All 38 AV APIs accessible
- ✅ Caching reduces API calls by 80%

### Phase 2 (Weeks 5-8)
- ✅ Paper trading with AV Greeks
- ✅ <5% API errors
- ✅ Discord publishing with Greeks
- ✅ 50+ community members

### Phase 3 (Weeks 9-12)
- ✅ Live trading profitable
- ✅ AV rate limit never exceeded
- ✅ Greeks accuracy validated
- ✅ 100+ paying subscribers

### Phase 4 (Weeks 13-16)
- ✅ <150ms end-to-end latency
- ✅ Using 20+ AV indicators
- ✅ ML model trained on 5 years AV data
- ✅ Advanced analytics dashboard

---

## APPENDIX A: Alpha Vantage API Reference

### Available APIs (38 Total)
```
OPTIONS (2)
├── REALTIME_OPTIONS - Current options with Greeks
└── HISTORICAL_OPTIONS - 20 years history with Greeks

TECHNICAL INDICATORS (16)
├── RSI, MACD, STOCH, WILLR, MOM, BBANDS
├── ATR, ADX, AROON, CCI, EMA, SMA
└── MFI, OBV, AD, VWAP

ANALYTICS (2)
├── ANALYTICS_FIXED_WINDOW
└── ANALYTICS_SLIDING_WINDOW

SENTIMENT (3)
├── NEWS_SENTIMENT
├── TOP_GAINERS_LOSERS
└── INSIDER_TRANSACTIONS

FUNDAMENTALS (7)
├── OVERVIEW, EARNINGS, INCOME_STATEMENT
├── BALANCE_SHEET, CASH_FLOW
└── DIVIDENDS, SPLITS

ECONOMIC (5)
├── TREASURY_YIELD, FEDERAL_FUNDS_RATE
└── CPI, INFLATION, REAL_GDP
```

---

## APPENDIX B: Technology Stack

### Core Dependencies
```python
# requirements.txt
ib_insync==0.9.86  # IBKR connection
aiohttp==3.9.0  # Alpha Vantage async requests
xgboost==2.0.0  # ML model
pandas==2.1.0  # Data manipulation
numpy==1.25.0  # Numerical operations
redis==5.0.0  # Caching
psycopg2-binary==2.9.9  # PostgreSQL
discord.py==2.3.0  # Community bot
prometheus-client==0.19.0  # Metrics
```

### Infrastructure
- Python 3.11+
- PostgreSQL 16
- Redis 7.2
- Ubuntu 22.04 LTS (production)
- 16GB RAM minimum
- 100GB SSD storage
- 1Gbps network connection

---

END OF TECHNICAL SPECIFICATION v3.0