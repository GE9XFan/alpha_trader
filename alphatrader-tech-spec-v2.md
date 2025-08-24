# AlphaTrader Technical Specification v2.0
## Achievable Single-Developer Architecture

---

## 1. EXECUTIVE SUMMARY

### 1.1 System Overview
AlphaTrader is a ML-driven options trading system designed for single-developer implementation and maintenance. The system trades single-leg options on SPY, QQQ, and IWM using machine learning predictions, manages risk through real-time Greeks monitoring, and monetizes signals through a Discord-based community platform.

### 1.2 Development Philosophy
- **Build Once, Reuse Everywhere**: Every component is designed for maximum reusability
- **Progressive Complexity**: Start simple, add complexity only after stability
- **Fail Fast, Recover Faster**: Robust error handling with automatic recovery
- **Observable by Default**: Comprehensive logging and metrics from day one

### 1.3 Phased Delivery Approach
- **Phase 1 (Weeks 1-4)**: Core Trading Engine with ML
- **Phase 2 (Weeks 5-8)**: Paper Trading + Basic Community
- **Phase 3 (Weeks 9-12)**: Production Trading + Full Community
- **Phase 4 (Weeks 13-16)**: Optimization + Advanced Features

### 1.4 Realistic Performance Targets
| Metric | Phase 1 Target | Phase 3 Target | Notes |
|--------|---------------|----------------|-------|
| Critical Path Latency | <200ms | <100ms | Optimization over time |
| Greeks Calculation | <50ms | <20ms | For 50 contracts |
| ML Inference | <30ms | <15ms | Simple model first |
| Position Limit | 5 | 20 | Gradual increase |
| Daily Trades | 10 | 50 | As confidence grows |
| Discord Latency | N/A | <5 seconds | Async processing |

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Simplified Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA LAYER (Reusable)                  │
├──────────────────────────┬────────────────────────────────┤
│   MarketDataManager      │   OptionsDataManager          │
│   - IBKR connection      │   - Chain fetching            │
│   - 5-sec bars           │   - Greeks calculation        │
│   - Price caching        │   - IV computation            │
└──────────┬───────────────┴───────────┬───────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│              ANALYTICS LAYER (Reusable)                  │
├──────────────────────────┬────────────────────────────────┤
│   FeatureEngine          │   MLPredictor                 │
│   - Technical indicators │   - XGBoost model             │
│   - Options metrics      │   - Feature pipeline          │
│   - Market microstructure│   - Confidence scoring        │
└──────────┬───────────────┴───────────┬───────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│               TRADING LAYER (Core Logic)                 │
├──────────────────────────┬────────────────────────────────┤
│   SignalGenerator        │   RiskManager                 │
│   - Signal creation      │   - Position limits           │
│   - Confidence filtering │   - Greeks monitoring         │
│   - Entry/exit rules     │   - Stop loss logic           │
└──────────┬───────────────┴───────────┬───────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│              EXECUTION LAYER (Phase-Specific)            │
├──────────────────────────┬────────────────────────────────┤
│   PaperTrader            │   LiveTrader                  │
│   - Simulated fills      │   - IBKR orders               │
│   - Slippage modeling    │   - Real fills                │
│   - P&L tracking         │   - Position management       │
└──────────┬───────────────┴───────────┬───────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│            COMMUNITY LAYER (Progressive Build)           │
├──────────────────────────┬────────────────────────────────┤
│   SignalPublisher        │   DiscordBot                  │
│   - Format signals       │   - Channel management        │
│   - Apply delays         │   - Tier handling             │
│   - Track performance    │   - Command interface         │
└──────────────────────────┴────────────────────────────────┘
```

### 2.2 Data Flow Design

#### Synchronous Trading Pipeline
```
Market Data → Features → ML Model → Signal → Risk Check → Order
    10ms   →   30ms   →   30ms   →  10ms  →   20ms    →  100ms
                        Total: 200ms (Phase 1)
```

#### Asynchronous Community Pipeline
```
Trade Event → Queue → Format → Discord Post
    Instant →  1s   →  1s   →    3s
                Total: <5 seconds
```

---

## 3. COMPONENT SPECIFICATIONS

### 3.1 Data Layer (Weeks 1-2)

#### MarketDataManager
```python
class MarketDataManager:
    """
    Reusable market data component
    Built once in Week 1, used forever
    """
    def __init__(self, config):
        self.ibkr = None  # IBKR connection
        self.cache = {}   # Price cache
        self.bars = {}    # 5-second bars
        
    async def connect(self):
        # Connect to IBKR (paper or live based on config)
        
    async def subscribe(self, symbols):
        # Subscribe to market data
        
    def get_latest_price(self, symbol):
        # Get from cache (instant access)
        
    async def get_historical(self, symbol, days=5):
        # Get historical data for ML training
```

#### OptionsDataManager
```python
class OptionsDataManager:
    """
    Options-specific data handling
    Built in Week 2, extended over time
    """
    def __init__(self, market_data_mgr):
        self.market = market_data_mgr
        self.chains = {}
        self.greeks_cache = {}
        
    async def fetch_chain(self, symbol):
        # Get options chain from IBKR
        
    def calculate_greeks(self, option, spot_price):
        # Black-Scholes Greeks (cached)
        
    def find_atm_options(self, symbol, dte_range=(0, 7)):
        # Find ATM options for trading
```

### 3.2 Analytics Layer (Week 3)

#### FeatureEngine
```python
class FeatureEngine:
    """
    Feature calculation for ML
    Built once, extended as needed
    """
    def __init__(self):
        self.feature_names = [
            # Price features
            'returns_5m', 'returns_30m', 'returns_1d',
            'volume_ratio', 'price_momentum',
            
            # Technical indicators (10 core ones)
            'rsi', 'macd', 'bb_position', 'atr',
            'adx', 'obv', 'vwap_distance',
            
            # Options features
            'iv_rank', 'put_call_ratio', 'gamma_exposure',
            'delta_neutral_price', 'max_pain_distance',
            
            # Market structure
            'spy_correlation', 'sector_momentum',
            'vix_level', 'term_structure'
        ]
        
    def calculate(self, data):
        # Returns feature vector for ML
        # All features have defaults to prevent failures
```

#### MLPredictor
```python
class MLPredictor:
    """
    Simple ML model that works from Day 1
    Complexity added over time
    """
    def __init__(self):
        self.model = self.load_or_create_model()
        self.scaler = StandardScaler()
        self.min_confidence = 0.6
        
    def load_or_create_model(self):
        # Try to load trained model
        # If not found, create simple XGBoost with defaults
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        
    def predict(self, features):
        # Returns: (signal, confidence)
        # signal: 'BUY_CALL', 'BUY_PUT', 'HOLD', 'CLOSE'
        # confidence: 0.0 to 1.0
```

### 3.3 Trading Layer (Week 4)

#### SignalGenerator
```python
class SignalGenerator:
    """
    Combines ML predictions with trading rules
    Reusable for paper and live trading
    """
    def __init__(self, ml_predictor, risk_manager):
        self.ml = ml_predictor
        self.risk = risk_manager
        self.signals_today = []
        
    async def generate_signal(self, symbol, features):
        # Get ML prediction
        signal, confidence = self.ml.predict(features)
        
        # Apply trading rules
        if not self.validate_signal(signal, confidence):
            return None
            
        # Check risk limits
        if not await self.risk.can_trade(symbol, signal):
            return None
            
        # Create signal object
        return TradingSignal(
            symbol=symbol,
            action=signal,
            confidence=confidence,
            timestamp=datetime.now()
        )
```

#### RiskManager
```python
class RiskManager:
    """
    Enforces all risk limits
    Critical component - never bypassed
    """
    def __init__(self, config):
        self.max_positions = 5  # Start conservative
        self.max_position_size = 10000  # Small initially
        self.daily_loss_limit = 1000  # Tight stop
        self.portfolio_greeks_limits = {
            'delta': (-0.3, 0.3),
            'gamma': (-0.5, 0.5),
            'vega': (-500, 500),
            'theta': (-200, float('inf'))
        }
        
    async def can_trade(self, symbol, signal):
        # Check all limits
        # Return True only if all pass
```

### 3.4 Execution Layer (Weeks 5-6)

#### PaperTrader
```python
class PaperTrader:
    """
    Paper trading with realistic simulation
    Builds on all previous components
    """
    def __init__(self, market_data, options_data):
        self.market = market_data
        self.options = options_data
        self.positions = {}
        self.trades = []
        self.starting_capital = 100000
        self.cash = self.starting_capital
        
    async def execute_signal(self, signal):
        # Simulate execution with slippage
        # Track P&L realistically
        # Return execution report
```

#### LiveTrader
```python
class LiveTrader:
    """
    Live trading through IBKR
    Reuses PaperTrader logic with real orders
    """
    def __init__(self, ibkr_connection, paper_trader):
        self.ibkr = ibkr_connection
        self.paper = paper_trader  # Reuse logic!
        
    async def execute_signal(self, signal):
        # Use paper trader's logic for validation
        # Submit real order to IBKR
        # Handle partial fills and rejections
```

### 3.5 Community Layer (Weeks 7-8)

#### SignalPublisher
```python
class SignalPublisher:
    """
    Formats and publishes signals
    Works for paper and live trading
    """
    def __init__(self):
        self.discord = None  # Connected later
        self.performance_tracker = PerformanceTracker()
        
    async def publish(self, signal, execution):
        # Format signal for Discord
        # Track performance
        # Apply tier delays
        
    def format_signal(self, signal, execution):
        return {
            'symbol': signal.symbol,
            'action': signal.action,
            'entry_price': execution.fill_price,
            'contracts': execution.quantity,
            'confidence': signal.confidence,
            'timestamp': execution.timestamp
        }
```

---

## 4. DATABASE SCHEMA

### 4.1 Simplified Schema (Phase 1)

```sql
-- Core tables only
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ,
    symbol VARCHAR(10),
    option_type VARCHAR(4),  -- CALL/PUT
    strike DECIMAL,
    expiry DATE,
    action VARCHAR(10),  -- BUY/SELL
    quantity INT,
    price DECIMAL,
    commission DECIMAL,
    pnl DECIMAL
);

CREATE TABLE positions (
    symbol VARCHAR(10) PRIMARY KEY,
    quantity INT,
    avg_price DECIMAL,
    current_price DECIMAL,
    unrealized_pnl DECIMAL,
    delta DECIMAL,
    gamma DECIMAL,
    theta DECIMAL,
    vega DECIMAL,
    updated_at TIMESTAMPTZ
);

CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ,
    symbol VARCHAR(10),
    signal_type VARCHAR(20),
    confidence DECIMAL,
    executed BOOLEAN,
    execution_id INT REFERENCES trades(id)
);

-- Indexes for performance
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_signals_symbol ON signals(symbol, timestamp);
```

---

## 5. CONFIGURATION MANAGEMENT

### 5.1 Single Configuration File

```yaml
# config.yaml - Everything in one place
trading:
  mode: paper  # paper/live
  symbols: [SPY, QQQ, IWM]
  
risk:
  max_positions: 5
  max_position_size: 10000
  daily_loss_limit: 1000
  portfolio_greeks:
    delta: [-0.3, 0.3]
    gamma: [-0.5, 0.5]
    vega: [-500, 500]
    theta: [-200, null]
    
ml:
  model_path: models/xgboost_v1.pkl
  min_confidence: 0.6
  retrain_days: 30
  
community:
  discord_token: ${DISCORD_TOKEN}
  channels:
    paper: paper-trades
    live: live-trades
    alerts: system-alerts
  tiers:
    free:
      delay: 300
      max_signals: 5
    premium:
      delay: 30
      max_signals: 20
    vip:
      delay: 0
      max_signals: -1
      
monitoring:
  log_level: INFO
  metrics_port: 8080
  health_check_interval: 60
```

---

## 6. PROGRESSIVE ROLLOUT STRATEGY

### 6.1 Phase 1: Core Trading (Weeks 1-4)
- Build data layer components
- Implement ML prediction
- Create signal generation
- Test with historical data
- **Deliverable**: System that generates signals from market data

### 6.2 Phase 2: Paper Trading (Weeks 5-8)
- Add paper trading execution
- Implement P&L tracking
- Create basic Discord bot
- Start publishing paper trades
- **Deliverable**: Live paper trading with community viewing

### 6.3 Phase 3: Production (Weeks 9-12)
- Switch to live trading (small size)
- Add subscription management
- Implement full risk controls
- Scale position sizes gradually
- **Deliverable**: Revenue-generating trading system

### 6.4 Phase 4: Optimization (Weeks 13-16)
- Performance optimization
- Add spread strategies
- Enhance ML models
- Add more symbols
- **Deliverable**: Scaled, optimized system

---

## 7. ERROR HANDLING STRATEGY

### 7.1 Graceful Degradation
```python
# Every component has fallbacks
if not ml_model_available:
    use_simple_rules()
    
if not options_chain_available:
    trade_stock_only()
    
if not discord_connected:
    log_signals_locally()
```

### 7.2 Automatic Recovery
- IBKR reconnection with exponential backoff
- Model reloading on failure
- Database connection pooling
- Discord bot auto-restart

---

## 8. TESTING STRATEGY

### 8.1 Progressive Testing
1. **Unit tests** for each component
2. **Integration tests** for component pairs
3. **Paper trading** as system test
4. **Small money** as final validation

### 8.2 Performance Benchmarks
- Run benchmarks weekly
- Track latency trends
- Optimize bottlenecks iteratively

---

## 9. MONITORING & OPERATIONS

### 9.1 Key Metrics (Simple Dashboard)
- Current positions and P&L
- Today's signals and executions
- System latency (moving average)
- Error rate and types
- Discord message queue depth

### 9.2 Daily Checklist
- [ ] Check IBKR connection
- [ ] Verify market data flowing
- [ ] Review overnight positions
- [ ] Check risk limits
- [ ] Verify Discord bot online
- [ ] Review error logs

---

## 10. SUCCESS CRITERIA

### Phase 1 Success (Week 4)
- Generates 10+ signals per day
- ML confidence >60% on signals
- Backtesting shows positive expectancy
- All components have tests

### Phase 2 Success (Week 8)
- Paper trading for 2 weeks
- 50%+ win rate
- Discord publishing working
- 10+ community members viewing

### Phase 3 Success (Week 12)
- Live trading profitable
- <5% of trades have errors
- 50+ paying subscribers
- System stable for 30 days

### Phase 4 Success (Week 16)
- Latency <100ms consistently
- Trading 10+ symbols
- 100+ subscribers
- Adding spread strategies

---

## APPENDIX A: Technology Stack

### Core Requirements
- Python 3.11+ (latest stable)
- PostgreSQL 16 (already installed)
- Redis (already installed)
- ib_insync for IBKR
- XGBoost for ML
- discord.py for community

### Development Tools
- pytest for testing
- black for formatting
- loguru for logging
- prometheus for metrics
- grafana for dashboards (later)

### Hardware Requirements (MacBook)
- 16GB RAM minimum
- 50GB free disk space
- Stable internet connection
- Backup power recommended

---

END OF SPECIFICATION v2.0