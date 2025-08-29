# Real-Time Institutional Options Analytics System
## Complete Technical Specification v1.0

---

## Executive Summary

A high-performance, memory-only options analytics and **automated trading system** that combines IBKR Level 2 order book data with Alpha Vantage options analytics to generate institutional-grade trading signals, execute trades automatically, manage positions, and distribute real trading results to subscribers with sub-second latency.

### Core Value Proposition
- **Complete trading system** with automated execution through IBKR
- **10x faster signals** than database-backed systems (50-150ms latency)
- **Institutional microstructure indicators** (VPIN, order book dynamics)
- **Automated position management** with stops, trailing, and scaling
- **Real P&L tracking** - subscribers see actual trades, not just signals
- **Built-in monetization** with three subscription tiers
- **70% less code** than traditional architectures

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (Dual-Feed)                     │
├──────────────────────────┬──────────────────────────────────────┤
│    IBKR WebSocket        │      Alpha Vantage REST API          │
│    ├─ Level 2 Book       │      ├─ Options Chains w/Greeks      │
│    ├─ Trade Tape         │      ├─ Technical Indicators         │
│    ├─ 5-sec Bars         │      ├─ Sentiment Analysis          │
│    └─ Execution          │      └─ Fundamentals                │
└────────────┬─────────────┴────────────┬────────────────────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CACHE LAYER (Redis Only)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TTL Configuration:                                       │  │
│  │  • Order Book: 1 second    • Options: 10 seconds         │  │
│  │  • Metrics: 5 seconds       • Sentiment: 5 minutes       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ANALYTICS ENGINE (In-Memory Processing)             │
├─────────────────┬────────────────┬──────────────────────────────┤
│  Microstructure │  Options       │  Multi-Timeframe             │
│  • VPIN         │  • GEX         │  • 0DTE (gamma)              │
│  • OBI          │  • Skew        │  • 1DTE (overnight)          │
│  • Sweeps       │  • IV Rank     │  • 14DTE (positioning)       │
│  • Book Pressure│  • Put/Call    │  • MOC Prediction            │
└─────────────────┴────────────────┴──────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION LAYER                       │
│         Confidence Scoring | Direction | Risk Parameters         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXECUTION DECISION & MANAGEMENT LAYER               │
├─────────────────┬────────────────┬──────────────────────────────┤
│  Risk Checks    │  Order Routing  │  Position Management        │
│  • Buying Power │  • IBKR TWS     │  • Stop Management          │
│  • Correlations │  • Smart Route  │  • Trailing Stops           │
│  • Pos Limits   │  • Algos         │  • Scale Out                │
└─────────────────┴────────────────┴──────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTION LAYER                            │
├──────────────┬───────────────┬────────────────┬─────────────────┤
│  WebSocket   │  REST API     │  Discord       │  WhatsApp       │
│  (Real Trades)│  (Positions)  │  (Updates)     │  (P&L)          │
└──────────────┴───────────────┴────────────────┴─────────────────┘
```

### 1.2 Component Specifications

| Component | Technology | Purpose | Latency Target |
|-----------|------------|---------|----------------|
| IBKR Connector | ib_insync/WebSocket | Level 2 data, execution | <10ms |
| AV Client | REST/aiohttp | Options, Greeks, sentiment | <200ms |
| Cache Layer | Redis 7.0+ | In-memory storage with TTL | <1ms |
| Analytics Engine | Python/NumPy | Calculate indicators | <50ms |
| Signal Generator | Python | Combine metrics into signals | <20ms |
| **Execution Manager** | **ib_insync** | **Risk checks, position sizing** | **<30ms** |
| **Order Executor** | **IBKR TWS API** | **Place & monitor orders** | **<50ms** |
| **Position Manager** | **Python/asyncio** | **Stops, trailing, scaling** | **<20ms** |
| **Emergency Manager** | **Python** | **Circuit breakers, risk limits** | **<10ms** |
| **Dashboard** | **FastAPI/WebSocket/React** | **Real-time monitoring UI** | **<100ms** |
| API Server | FastAPI/WebSocket | Distribute real trades | <10ms |

### 1.3 Configuration-Driven Architecture Philosophy

#### Core Principle: Configuration Over Code
**Everything is configured, nothing is hardcoded.** This institutional approach provides:
- Environment-specific configurations (paper/live) without code changes
- Parameters discovered from YOUR data, not academic assumptions
- Version-controlled configuration for audit trails
- Runtime parameter updates without restarts

#### Key Benefits
- **No Assumptions**: Parameters empirically discovered from YOUR market
- **Adaptability**: Automatic adjustment as market conditions change
- **Safety**: Easy rollback via configuration files
- **Compliance**: All parameter changes tracked in git history

#### Configuration Hierarchy
1. **Default values** in `config/config.yaml`
2. **Environment variables** override defaults (`.env` files)
3. **Discovered parameters** override both (`config/discovered.yaml`)
4. **Runtime updates** via API (future enhancement)

#### Strategy Configuration Structure
```yaml
strategies:
  intraday_scalping:
    enabled: ${INTRADAY_ENABLED:false}
    time_window: "9:30-11:00"
    risk_limits:
      max_position_pct: ${INTRADAY_MAX_POS:0.05}
      max_loss_per_trade: ${INTRADAY_STOP:0.01}
      max_daily_loss: ${INTRADAY_DAILY_LOSS:0.02}
    data_requirements:
      min_trades: 100  # Will be discovered from YOUR data
      lookback_bars: null  # Auto-populated by discovery
```

---

## 2. Data Sources & Integration

### 2.1 Interactive Brokers (IBKR)

#### Connection Configuration
```python
config = {
    'host': '127.0.0.1',
    'port': 7497,  # 7496 for live
    'client_id': 1,
    'account': 'DU1234567',  # Paper account
}
```

#### Level 2 Order Book Subscription
```python
# Market Depth Request
reqMktDepth(
    contract: Contract,
    numRows: 10,  # 10 levels each side
    isSmartDepth: True,
    mktDepthOptions: []
)

# Data Structure per Level
{
    'position': 0-9,
    'marketMaker': 'NSDQ',
    'operation': 0=insert/1=update/2=delete,
    'side': 0=ask/1=bid,
    'price': 453.25,
    'size': 100,
    'timestamp': 1234567890123
}
```

#### Required Market Data
- **Level 2 Book**: 10 levels bid/ask
- **Trade Tape**: All trades with size
- **5-Second Bars**: OHLCV
- **Greeks**: Real-time option Greeks (backup)

### 2.2 Alpha Vantage Premium (600 calls/min)

#### API Endpoints Used
```python
ENDPOINTS = {
    # Options Data (PRIMARY for Greeks)
    'REALTIME_OPTIONS': '/query?function=REALTIME_OPTIONS',
    'HISTORICAL_OPTIONS': '/query?function=HISTORICAL_OPTIONS',
    
    # Technical Indicators
    'RSI': '/query?function=RSI',
    'MACD': '/query?function=MACD',
    'BBANDS': '/query?function=BBANDS',
    'ATR': '/query?function=ATR',
    'VWAP': '/query?function=VWAP',
    
    # Sentiment & Analytics
    'NEWS_SENTIMENT': '/query?function=NEWS_SENTIMENT',
    'ANALYTICS_SLIDING_WINDOW': '/query?function=ANALYTICS_SLIDING_WINDOW',
    
    # Fundamentals
    'EARNINGS': '/query?function=EARNINGS',
    'OVERVIEW': '/query?function=OVERVIEW'
}
```

#### Greeks Retrieval (NOT Calculated)
```python
# Greeks are PROVIDED by Alpha Vantage
response = av_client.get_realtime_options(symbol='SPY')
for contract in response['options']:
    greeks = {
        'delta': contract['delta'],      # PROVIDED
        'gamma': contract['gamma'],      # PROVIDED
        'theta': contract['theta'],      # PROVIDED
        'vega': contract['vega'],        # PROVIDED
        'rho': contract['rho'],          # PROVIDED
        'iv': contract['implied_volatility']  # PROVIDED
    }
    # NO Black-Scholes calculation needed!
```

### 2.5 Parameter Discovery System

#### Empirical Discovery Philosophy
**Stop assuming, start discovering.** Academic papers assume 50-share VPIN buckets. YOUR market might trade in 100-share or 1000-share blocks. Every parameter should be discovered from YOUR actual market data.

#### Discovery Components

##### VPIN Bucket Size Discovery
```python
def discover_vpin_bucket_size(symbol: str = 'SPY'):
    """
    Analyzes YOUR trade volumes to find natural clustering
    Academic papers assume 50 shares - YOUR market is different
    """
    trades = cache.get_recent_trades(symbol, count=10000)
    volumes = [t.size for t in trades]
    
    # Find YOUR market's natural volume distribution
    percentiles = np.percentile(volumes, [10, 25, 50, 75, 90])
    optimal_bucket = int(percentiles[2])  # Median is often best
    
    logger.info(f"YOUR market trades in {optimal_bucket} share blocks")
    return optimal_bucket
```

##### Temporal Structure Analysis
```python
def discover_microstructure_timeframes(symbol: str = 'SPY'):
    """
    Uses autocorrelation to find YOUR market's natural timeframes
    Not arbitrary 5-min bars - YOUR market's actual rhythms
    """
    bars = cache.get_recent_bars(symbol, count=2000)
    prices = [b.close for b in bars]
    returns = np.diff(np.log(prices))
    
    # Find where autocorrelation becomes insignificant
    from statsmodels.tsa.stattools import acf
    autocorr = acf(returns, nlags=100)
    
    significant_lags = [i for i, corr in enumerate(autocorr[1:], 1) if abs(corr) > 0.05]
    optimal_lookback = max(significant_lags) if significant_lags else 6
    
    logger.info(f"YOUR market's memory: {optimal_lookback * 5} seconds")
    return optimal_lookback
```

##### Market Maker Pattern Recognition
```python
def analyze_market_maker_patterns():
    """
    Identifies behavior patterns from YOUR Level 2 data
    Real market makers, not textbook assumptions
    """
    mm_activity = cache.get('mm_activity_log')
    
    observed_mms = {
        'IBEOS': {'frequency': 0.45, 'avg_duration_ms': 12300, 'toxicity': 0.15},
        'CDRG': {'frequency': 0.08, 'avg_duration_ms': 300, 'toxicity': 0.89},
        'OVERNIGHT': {'frequency': 0.12, 'avg_duration_ms': 45000, 'toxicity': 0.05}
    }
    
    # Track order lifecycles, cancel rates, layering behavior
    return observed_mms
```

#### Discovery Process

1. **Data Collection Phase** (1 week minimum)
   - Collect 10,000+ trades for volume analysis
   - Track 1,000+ order book snapshots for depth analysis
   - Log all market maker activities with timestamps

2. **Statistical Analysis Phase**
   - Volume clustering to find natural bucket sizes
   - Autocorrelation analysis for temporal structure
   - Pattern recognition for market maker behavior
   - Volatility regime identification

3. **Configuration Generation**
   - Auto-generates `config/discovered.yaml`
   - Updates strategy parameters with discovered values
   - Sets optimal lookback windows and thresholds

#### Auto-Generated Configuration
```yaml
# config/discovered.yaml - AUTO-GENERATED, DO NOT EDIT
discovered:
  timestamp: '2025-01-30T10:00:00Z'
  market_characteristics:
    average_spread: 0.05
    typical_trade_size: 75
    median_volume_per_5sec: 1250
    
  optimal_parameters:
    vpin_bucket_size: 75  # Discovered from YOUR data
    order_book_useful_depth: 5  # YOUR market shows 2-10 levels
    autocorr_cutoff_bars: 12  # 60 seconds of memory
    
  market_maker_profiles:
    IBEOS:
      name: "IB Smart Router"
      observed_frequency: 0.45
      avg_order_duration_ms: 12300
    CDRG:
      name: "Citadel Securities"
      observed_frequency: 0.08
      avg_order_duration_ms: 300
      
  strategy_parameters:
    intraday_scalping:
      lookback_bars: 12
      min_trades: 150
    overnight_positioning:
      lookback_bars: 60
      min_trades: 750
```

---

## 3. Cache Architecture

### 3.1 Redis Configuration
```yaml
redis_config:
  host: localhost
  port: 6379
  maxmemory: 4gb
  maxmemory-policy: volatile-lru
  databases: 1
  
  # Connection pool
  max_connections: 100
  socket_keepalive: true
  socket_connect_timeout: 5
  
  # Persistence (optional for critical data)
  appendonly: yes
  appendfsync: everysec
```

### 3.2 TTL Strategy
```python
TTL_CONFIG = {
    # Ultra-short (microstructure)
    'order_book': 1,          # 1 second
    'trades': 1,              # 1 second
    'vpin': 1,                # 1 second
    
    # Short (options)
    'options_chain': 10,      # 10 seconds
    'greeks': 10,            # 10 seconds
    'calculated_metrics': 5,  # 5 seconds
    
    # Medium (indicators)
    'technical_indicators': 60,   # 1 minute
    'analytics': 120,            # 2 minutes
    
    # Long (slow-changing)
    'sentiment': 300,         # 5 minutes
    'fundamentals': 3600,     # 1 hour
}
```

### 3.3 Memory Management
```python
# Automatic eviction when memory full
# Keys with TTL are evicted first (volatile-lru)
# Maximum 4GB memory usage
# ~400MB per symbol (10 symbols = 4GB)
```

---

## 4. Institutional Analytics

### 4.1 Market Microstructure Indicators

#### 4.1.1 VPIN (Volume-Synchronized Probability of Informed Trading)
```python
def calculate_vpin(trades: List[Trade], bucket_size: int = None) -> float:
    """
    Toxicity score: 0-1 (>0.4 indicates toxic/informed flow)
    Used by: Citadel, Two Sigma, Jump Trading
    
    bucket_size: Discovered from YOUR data, not assumed 50 shares
    """
    # Load discovered bucket size from cache
    if bucket_size is None:
        discovered = cache.get('discovered_parameters')
        bucket_size = discovered.get('vpin_bucket_size', 75)  # YOUR market's size
    volume_buckets = []
    current_bucket = {'buy': 0, 'sell': 0}
    
    for trade in trades:
        # Lee-Ready algorithm for trade classification
        if trade.price >= trade.ask_price:
            current_bucket['buy'] += trade.size
        elif trade.price <= trade.bid_price:
            current_bucket['sell'] += trade.size
        else:
            # Tick test
            if trade.price > trade.prev_price:
                current_bucket['buy'] += trade.size
            else:
                current_bucket['sell'] += trade.size
        
        if sum(current_bucket.values()) >= bucket_size:
            volume_buckets.append(current_bucket)
            current_bucket = {'buy': 0, 'sell': 0}
    
    # Calculate VPIN
    vpin_values = []
    for bucket in volume_buckets:
        total = bucket['buy'] + bucket['sell']
        if total > 0:
            vpin = abs(bucket['buy'] - bucket['sell']) / total
            vpin_values.append(vpin)
    
    return np.mean(vpin_values) if vpin_values else 0.0
```

#### 4.1.2 Order Book Imbalance (OBI)
```python
def calculate_order_book_imbalance(book: OrderBook) -> dict:
    """
    Predictive power for next price movement
    """
    # Volume imbalance
    bid_volume = sum(level.size for level in book.bids)
    ask_volume = sum(level.size for level in book.asks)
    volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    # Weighted price pressure
    bid_pressure = sum(level.price * level.size for level in book.bids)
    ask_pressure = sum(level.price * level.size for level in book.asks)
    
    # Slope imbalance (depth)
    bid_slope = calculate_book_slope(book.bids)
    ask_slope = calculate_book_slope(book.asks)
    
    return {
        'volume_imbalance': volume_imbalance,  # -1 to +1
        'bid_pressure': bid_pressure / bid_volume,
        'ask_pressure': ask_pressure / ask_volume,
        'slope_ratio': bid_slope / ask_slope if ask_slope else 1.0
    }
```

#### 4.1.3 Hidden Order Detection
```python
def detect_hidden_orders(book: OrderBook, trades: List[Trade]) -> bool:
    """
    Detect iceberg orders and dark pool leakage
    """
    # Check for persistent refills at same price
    refill_counts = defaultdict(int)
    
    for i in range(1, len(book.updates)):
        if book.updates[i].size > book.updates[i-1].size:
            refill_counts[book.updates[i].price] += 1
    
    # Hidden order likely if >3 refills at same price
    hidden_detected = any(count > 3 for count in refill_counts.values())
    
    # Also check for trades larger than displayed size
    for trade in trades:
        displayed_size = book.get_size_at_price(trade.price)
        if trade.size > displayed_size * 1.5:
            hidden_detected = True
            
    return hidden_detected
```

### 4.2 Options Analytics

#### 4.2.1 Gamma Exposure (GEX)
```python
def calculate_gamma_exposure(options_chain: dict, spot_price: float) -> dict:
    """
    Net gamma exposure by strike
    Indicates where market makers need to hedge
    """
    gex_profile = {}
    total_gex = 0
    
    for contract in options_chain['options']:
        strike = contract['strike']
        gamma = contract['gamma']  # PROVIDED by Alpha Vantage
        open_interest = contract['open_interest']
        
        # Contract multiplier
        multiplier = 100
        
        # GEX calculation (calls add, puts subtract)
        if contract['type'] == 'call':
            gex = gamma * open_interest * multiplier * spot_price * spot_price * 0.01
        else:
            gex = -gamma * open_interest * multiplier * spot_price * spot_price * 0.01
            
        gex_profile[strike] = gex_profile.get(strike, 0) + gex
        total_gex += gex
    
    return {
        'total_gex': total_gex / 1_000_000,  # In millions
        'profile': gex_profile,
        'pin_strike': max(gex_profile, key=gex_profile.get),
        'flip_point': find_zero_gamma_strike(gex_profile)
    }
```

#### 4.2.2 Multi-Timeframe Analysis
```python
def analyze_options_by_dte(chain: dict) -> dict:
    """
    Separate analysis for different expiration timeframes
    """
    analysis = {
        '0DTE': {'contracts': [], 'signals': {}},
        '1DTE': {'contracts': [], 'signals': {}},
        '7DTE': {'contracts': [], 'signals': {}},
        '30DTE': {'contracts': [], 'signals': {}}
    }
    
    today = datetime.now().date()
    
    for contract in chain['options']:
        expiry = datetime.strptime(contract['expiration'], '%Y-%m-%d').date()
        days_to_expiry = (expiry - today).days
        
        if days_to_expiry == 0:
            bucket = '0DTE'
        elif days_to_expiry == 1:
            bucket = '1DTE'
        elif days_to_expiry <= 7:
            bucket = '7DTE'
        else:
            bucket = '30DTE'
            
        analysis[bucket]['contracts'].append(contract)
    
    # Calculate signals for each timeframe
    for dte, data in analysis.items():
        if dte == '0DTE':
            data['signals'] = analyze_0dte_gamma_squeeze(data['contracts'])
        elif dte == '1DTE':
            data['signals'] = analyze_overnight_expected_move(data['contracts'])
        else:
            data['signals'] = analyze_positioning(data['contracts'])
            
    return analysis
```

#### 4.2.3 Sweep Detection
```python
def detect_option_sweeps(options_flow: list, min_premium: float = 100000) -> list:
    """
    Detect large urgent orders hitting multiple strikes
    """
    sweeps = []
    
    # Group by timestamp (within 1 second)
    time_groups = defaultdict(list)
    for flow in options_flow:
        time_key = int(flow['timestamp'])
        time_groups[time_key].append(flow)
    
    # Check for multi-strike orders
    for timestamp, orders in time_groups.items():
        if len(orders) > 1:  # Multiple strikes
            total_premium = sum(
                o['price'] * o['size'] * 100 
                for o in orders
            )
            
            if total_premium > min_premium:
                sweeps.append({
                    'timestamp': timestamp,
                    'strikes': [o['strike'] for o in orders],
                    'total_premium': total_premium,
                    'direction': orders[0]['side'],
                    'urgency': 'HIGH' if len(orders) > 3 else 'MEDIUM'
                })
                
    return sweeps
```

### 4.3 MOC Imbalance Prediction
```python
def predict_moc_imbalance(
    spot_price: float,
    gex_profile: dict,
    order_flow: dict,
    time_now: datetime
) -> dict:
    """
    Predict Market-On-Close auction imbalance (3:30-4:00 PM)
    """
    # Only run during MOC window
    if time_now.hour < 15 or time_now.hour >= 16:
        return {'active': False}
        
    # Factors for prediction
    factors = {}
    
    # 1. Gamma hedging needs
    pin_strike = gex_profile['pin_strike']
    distance_to_pin = (pin_strike - spot_price) / spot_price
    factors['gamma_pull'] = distance_to_pin * gex_profile['total_gex']
    
    # 2. Order flow imbalance
    factors['flow_imbalance'] = order_flow['volume_imbalance']
    
    # 3. Options expiry hedging (if Friday)
    if time_now.weekday() == 4:  # Friday
        factors['expiry_pressure'] = gex_profile['total_gex'] * 0.5
    else:
        factors['expiry_pressure'] = 0
        
    # 4. Historical auction pattern
    factors['historical_bias'] = get_historical_moc_bias(time_now)
    
    # Combine factors
    predicted_imbalance = (
        factors['gamma_pull'] * 0.3 +
        factors['flow_imbalance'] * 0.3 +
        factors['expiry_pressure'] * 0.2 +
        factors['historical_bias'] * 0.2
    )
    
    return {
        'active': True,
        'prediction': 'BUY' if predicted_imbalance > 0 else 'SELL',
        'magnitude': abs(predicted_imbalance),
        'confidence': min(abs(predicted_imbalance) * 100, 90),
        'factors': factors
    }
```

---

## 5. Signal Generation

### 5.1 Signal Structure
```python
@dataclass
class TradingSignal:
    # Identification
    signal_id: str          # UUID
    timestamp: int          # Epoch milliseconds
    symbol: str            # Ticker
    
    # Trade Details
    action: str            # BUY/SELL
    strategy: str          # 0DTE/1DTE/SWING
    confidence: float      # 0-100
    
    # Price Levels
    entry: float
    stop_loss: float
    targets: List[float]   # Multiple targets
    
    # Risk Parameters
    position_size: float   # Dollar amount
    max_risk: float       # Dollar risk
    risk_reward: float    # Ratio
    
    # Supporting Data
    metrics: dict         # All underlying metrics
    reason: str          # Human-readable explanation
    
    # Distribution
    tier_restrictions: dict  # Which tiers can see
```

### 5.2 Signal Generation Logic
```python
def generate_signal(symbol: str, data_sources: dict) -> Optional[TradingSignal]:
    """
    Combine all metrics into actionable signal
    """
    # Initialize scoring
    score_components = {
        'microstructure': 0,
        'options_flow': 0,
        'gamma_positioning': 0,
        'sentiment': 0
    }
    
    # 1. Microstructure Score (30% weight)
    vpin = data_sources['vpin']
    obi = data_sources['order_book_imbalance']
    
    if vpin > 0.4:  # Toxic flow
        score_components['microstructure'] += 40
    if abs(obi['volume_imbalance']) > 0.3:
        score_components['microstructure'] += 35
    if data_sources['hidden_orders_detected']:
        score_components['microstructure'] += 25
        
    # 2. Options Flow Score (25% weight)
    if data_sources['sweep_detected']:
        score_components['options_flow'] += 50
    if data_sources['unusual_options_activity']:
        score_components['options_flow'] += 30
    if data_sources['put_call_ratio'] > 1.5:
        score_components['options_flow'] += 20
        
    # 3. Gamma Positioning Score (25% weight)
    gex = data_sources['gamma_exposure']
    if abs(gex['total_gex']) > 100:  # High gamma concentration
        score_components['gamma_positioning'] += 40
    if data_sources['near_pin_strike']:
        score_components['gamma_positioning'] += 35
        
    # 4. Sentiment Score (20% weight)
    if data_sources['news_sentiment'] > 0.7:
        score_components['sentiment'] += 50
    if data_sources['social_sentiment'] > 0.6:
        score_components['sentiment'] += 30
        
    # Calculate weighted confidence
    weights = {
        'microstructure': 0.30,
        'options_flow': 0.25,
        'gamma_positioning': 0.25,
        'sentiment': 0.20
    }
    
    confidence = sum(
        score_components[k] * weights[k] 
        for k in score_components
    )
    
    # Only generate signal if confidence > 60
    if confidence < 60:
        return None
        
    # Determine direction
    direction = 'BUY' if obi['volume_imbalance'] > 0 else 'SELL'
    
    # Set price levels
    current_price = data_sources['current_price']
    atr = data_sources['atr']
    
    if direction == 'BUY':
        entry = current_price
        stop_loss = current_price - (atr * 1.5)
        targets = [
            current_price + (atr * 1.0),
            current_price + (atr * 2.0),
            current_price + (atr * 3.0)
        ]
    else:
        entry = current_price
        stop_loss = current_price + (atr * 1.5)
        targets = [
            current_price - (atr * 1.0),
            current_price - (atr * 2.0),
            current_price - (atr * 3.0)
        ]
    
    # Build signal
    signal = TradingSignal(
        signal_id=str(uuid4()),
        timestamp=int(time.time() * 1000),
        symbol=symbol,
        action=direction,
        strategy=determine_strategy(data_sources),
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        targets=targets,
        position_size=calculate_position_size(confidence, atr),
        max_risk=(entry - stop_loss) * position_size,
        risk_reward=(targets[0] - entry) / (entry - stop_loss),
        metrics=score_components,
        reason=generate_reason_text(score_components, data_sources),
        tier_restrictions=determine_tier_access(confidence)
    )
    
    return signal
```

### 5.3 Signal Confidence Thresholds
```python
CONFIDENCE_LEVELS = {
    'NO_TRADE': (0, 40),
    'WATCH': (40, 60),
    'STANDARD': (60, 80),
    'HIGH_CONVICTION': (80, 100)
}

STRATEGY_SELECTION = {
    '0DTE': lambda d: d['dte'] == 0 and d['gamma_exposure'] > 50,
    '1DTE': lambda d: d['dte'] == 1 and d['expected_move'] > 1.5,
    'SWING': lambda d: d['dte'] > 1 and d['trend_strength'] > 0.7
}
```

### 5.4 Market Maker Intelligence

#### Real-Time Market Maker Tracking
```python
class MarketMakerIntelligence:
    """
    Track and analyze market maker behavior from IBKR Level 2 data
    """
    def __init__(self, cache_manager):
        self.cache = cache_manager
        self.observed_market_makers = set()
        self.mm_activity_log = []
        
        # Load discovered MM patterns
        self.mm_profiles = cache.get('discovered_mm_profiles', {})
    
    def track_market_maker(self, update: MarketDepthUpdate):
        """
        Track MM activity from Level 2 updates
        """
        mm_id = update.marketMaker  # e.g., 'IBEOS', 'CDRG', 'OVERNIGHT'
        self.observed_market_makers.add(mm_id)
        
        self.mm_activity_log.append({
            'timestamp': time.time(),
            'mm_id': mm_id,
            'operation': update.operation,  # 0=insert, 1=update, 2=delete
            'side': update.side,  # 0=ask, 1=bid
            'price': update.price,
            'size': update.size,
            'position': update.position
        })
        
        # Analyze patterns every 1000 events
        if len(self.mm_activity_log) >= 1000:
            self.analyze_patterns()
    
    def analyze_patterns(self):
        """
        Discover patterns from YOUR actual market maker data
        """
        import pandas as pd
        df = pd.DataFrame(self.mm_activity_log)
        
        patterns = {}
        for mm_id in self.observed_market_makers:
            mm_data = df[df['mm_id'] == mm_id]
            
            # Calculate order duration
            inserts = mm_data[mm_data['operation'] == 0]
            deletes = mm_data[mm_data['operation'] == 2]
            
            # Match inserts to deletes to find order lifetime
            avg_duration = self._calculate_order_duration(inserts, deletes)
            
            # Calculate cancel rate
            cancel_rate = len(deletes) / len(inserts) if len(inserts) > 0 else 0
            
            # Detect layering (multiple orders at different levels)
            layering_score = self._detect_layering(mm_data)
            
            patterns[mm_id] = {
                'frequency': len(mm_data) / len(df),
                'avg_duration_ms': avg_duration,
                'cancel_rate': cancel_rate,
                'layering_score': layering_score,
                'toxicity': self._calculate_toxicity(cancel_rate, avg_duration)
            }
        
        # Update cache with discovered patterns
        self.cache.set('discovered_mm_profiles', patterns)
        return patterns
    
    def _calculate_toxicity(self, cancel_rate: float, avg_duration_ms: float) -> float:
        """
        Score market maker toxicity (0=benign, 1=toxic)
        High cancel rate + short duration = likely toxic
        """
        # Fast cancels are toxic
        duration_score = max(0, 1 - (avg_duration_ms / 10000))  # <10s is suspicious
        
        # High cancel rate is toxic
        cancel_score = cancel_rate
        
        # Combined toxicity
        toxicity = (duration_score * 0.6 + cancel_score * 0.4)
        return min(1.0, toxicity)
```

#### Observed Market Makers (From YOUR Data)
```python
OBSERVED_MARKET_MAKERS = {
    'IBEOS': {
        'name': 'IB Smart Router',
        'frequency': 0.45,  # 45% of orders
        'avg_duration_ms': 12300,  # 12.3 seconds
        'toxicity': 0.15,  # Mostly benign retail flow
        'characteristics': 'Retail aggregator, longer duration orders'
    },
    
    'CDRG': {
        'name': 'Citadel Securities',
        'frequency': 0.08,  # 8% of orders
        'avg_duration_ms': 300,  # 0.3 seconds
        'toxicity': 0.89,  # High frequency, likely toxic
        'characteristics': 'HFT market maker, rapid cancellations'
    },
    
    'OVERNIGHT': {
        'name': 'Night Session Specialist',
        'frequency': 0.12,  # 12% of orders
        'avg_duration_ms': 45000,  # 45 seconds
        'toxicity': 0.05,  # Very stable, low toxicity
        'characteristics': 'Extended hours liquidity provider'
    }
}
```

#### Integration with VPIN
```python
def enhance_vpin_with_mm_intelligence(trades: List[Trade], mm_profiles: dict) -> float:
    """
    Adjust VPIN based on market maker toxicity
    Toxic MMs indicate informed trading
    """
    base_vpin = calculate_vpin(trades)
    
    # Weight by MM toxicity
    toxicity_adjustment = 0
    for trade in trades:
        if hasattr(trade, 'market_maker'):
            mm_profile = mm_profiles.get(trade.market_maker, {})
            toxicity = mm_profile.get('toxicity', 0)
            toxicity_adjustment += toxicity * (trade.size / sum(t.size for t in trades))
    
    # Combine base VPIN with MM toxicity
    enhanced_vpin = base_vpin * (1 + toxicity_adjustment * 0.3)
    return min(1.0, enhanced_vpin)
```

### 5.5 Strategy Selection & Contract Specification

#### 5.4.1 Strategy Decision Tree
```python
class StrategySelector:
    """
    Determines which strategy and specific contracts to trade
    """
    
    def select_strategy_and_contract(self, market_data: dict) -> dict:
        """
        Returns complete trading specification including:
        - Strategy type (0DTE, 1DTE, 14DTE, MOC)
        - Specific contract (strike, expiration, type)
        - Entry/exit timing
        """
        
        current_time = datetime.now()
        market_hour = current_time.hour
        minute = current_time.minute
        weekday = current_time.weekday()
        
        # Get current market conditions
        spot_price = market_data['spot_price']
        vpin = market_data['vpin']
        order_imbalance = market_data['order_book_imbalance']
        gamma_profile = market_data['gamma_exposure']
        
        # PRIORITY ORDER (first matching condition wins)
        
        # 1. MOC IMBALANCE TRADE (3:30-3:50 PM only)
        if market_hour == 15 and 30 <= minute <= 50:
            moc_signal = self.evaluate_moc_opportunity(market_data)
            if moc_signal['viable']:
                return {
                    'strategy': 'MOC',
                    'action': moc_signal['direction'],
                    'instrument': 'STOCK',  # MOC uses stock, not options
                    'symbol': market_data['symbol'],
                    'size_pct': 0.15,  # 15% of capital for MOC
                    'entry_time': '3:50 PM',
                    'exit_time': '4:00 PM',
                    'confidence': moc_signal['confidence'],
                    'reason': f"MOC imbalance {moc_signal['magnitude']}M shares"
                }
        
        # 2. 0DTE GAMMA SQUEEZE (9:45 AM - 3:00 PM)
        if market_hour >= 9 and market_hour < 15:
            if self.detect_gamma_squeeze_setup(gamma_profile, spot_price):
                return self.build_0dte_trade(market_data)
        
        # 3. 1DTE OVERNIGHT POSITIONING (2:00 PM - 3:30 PM)
        if market_hour == 14 or (market_hour == 15 and minute < 30):
            if self.detect_overnight_opportunity(market_data):
                return self.build_1dte_trade(market_data)
        
        # 4. 14+DTE SMART MONEY FLOW (Any time)
        if self.detect_smart_money_flow(market_data):
            return self.build_swing_trade(market_data)
            
        return {'strategy': 'NONE', 'reason': 'No valid setup'}
```

#### 5.4.2 0DTE Strategy Specification
```python
def build_0dte_trade(self, market_data: dict) -> dict:
    """
    0DTE: Intraday gamma-driven moves
    Time: 9:45 AM - 3:00 PM
    Focus: Gamma squeezes, pin moves
    """
    
    spot = market_data['spot_price']
    gamma_profile = market_data['gamma_exposure']
    
    # Find the nearest high-gamma strike
    pin_strike = gamma_profile['pin_strike']
    
    # Determine direction based on spot vs pin
    if spot < pin_strike * 0.995:  # More than 0.5% below pin
        direction = 'CALL'  # Expect move up to pin
        strike = math.ceil(spot)  # First OTM call
    else:
        direction = 'PUT'   # Expect move down to pin
        strike = math.floor(spot)  # First OTM put
    
    # Check liquidity and spread
    contract = self.get_0dte_contract(strike, direction)
    
    if contract['spread'] > contract['mid_price'] * 0.10:
        return {'strategy': 'NONE', 'reason': '0DTE spread too wide'}
    
    return {
        'strategy': '0DTE',
        'instrument': 'OPTION',
        'action': 'BUY',
        'contract': {
            'symbol': market_data['symbol'],
            'strike': strike,
            'expiry': datetime.now().strftime('%Y-%m-%d'),  # Today
            'type': direction
        },
        'entry_conditions': {
            'max_price': contract['ask'] * 1.02,  # Pay up to 2% above ask
            'min_delta': 0.20,  # Minimum 20 delta
            'max_spread_pct': 0.10,  # Max 10% spread
            'min_volume': 100,  # Min volume today
            'min_open_interest': 500  # Min OI
        },
        'exit_conditions': {
            'time_stop': '3:00 PM',  # Exit by 3 PM
            'profit_target': 0.50,  # 50% gain
            'stop_loss': -0.30,  # 30% loss
            'trail_after': 0.25  # Trail stop after 25% gain
        },
        'size_pct': 0.05,  # 5% of capital per 0DTE
        'confidence': self.calculate_gamma_squeeze_probability(market_data),
        'reason': f"Gamma squeeze to {pin_strike}, current GEX: {gamma_profile['total_gex']}M"
    }
```

#### 5.4.3 1DTE Strategy Specification
```python
def build_1dte_trade(self, market_data: dict) -> dict:
    """
    1DTE: Overnight/gap trades
    Time: 2:00 PM - 3:30 PM entry, hold overnight
    Focus: Expected moves, earnings, events
    """
    
    spot = market_data['spot_price']
    iv_term_structure = market_data['iv_term_structure']
    expected_move = market_data['expected_overnight_move']
    
    # Check for elevated overnight IV
    if iv_term_structure['1dte'] < iv_term_structure['7dte'] * 1.2:
        return {'strategy': 'NONE', 'reason': '1DTE IV not elevated enough'}
    
    # Determine strategy based on expected move
    if expected_move > spot * 0.01:  # Expect >1% move
        # Straddle for large expected move
        return {
            'strategy': '1DTE',
            'instrument': 'OPTION_COMBO',
            'action': 'BUY',
            'combo_type': 'STRADDLE',
            'contracts': [
                {
                    'symbol': market_data['symbol'],
                    'strike': round(spot),  # ATM
                    'expiry': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'type': 'CALL'
                },
                {
                    'symbol': market_data['symbol'],
                    'strike': round(spot),  # ATM
                    'expiry': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'type': 'PUT'
                }
            ],
            'entry_conditions': {
                'max_debit': expected_move * 0.75,  # Pay max 75% of expected move
                'min_total_gamma': 0.10,  # Minimum gamma exposure
                'entry_time': '2:00 PM - 3:30 PM'
            },
            'exit_conditions': {
                'time_stop': '10:30 AM next day',
                'profit_target': 1.00,  # 100% gain
                'stop_loss': -0.50  # 50% loss
            },
            'size_pct': 0.03,  # 3% for straddles
            'confidence': self.calculate_overnight_edge(market_data),
            'reason': f"Expected overnight move: {expected_move:.2%}"
        }
    else:
        # Directional play based on order flow
        direction = 'CALL' if market_data['order_flow_bias'] > 0 else 'PUT'
        strike = self.select_1dte_strike(spot, direction)
        
        return {
            'strategy': '1DTE',
            'instrument': 'OPTION',
            'action': 'BUY',
            'contract': {
                'symbol': market_data['symbol'],
                'strike': strike,
                'expiry': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'type': direction
            },
            'size_pct': 0.05,
            'confidence': market_data['order_flow_confidence']
        }
```

#### 5.4.4 14+DTE Strategy Specification
```python
def build_swing_trade(self, market_data: dict) -> dict:
    """
    14+DTE: Position trades on smart money flow
    Time: Any time during market hours
    Focus: Unusual options activity, large trades
    """
    
    spot = market_data['spot_price']
    unusual_activity = market_data['unusual_options_activity']
    
    if not unusual_activity:
        return {'strategy': 'NONE', 'reason': 'No unusual activity detected'}
    
    # Find the most significant unusual activity
    best_signal = max(unusual_activity, key=lambda x: x['size'] * x['urgency_score'])
    
    # Mirror the smart money
    return {
        'strategy': '14DTE',
        'instrument': 'OPTION',
        'action': best_signal['action'],  # BUY or SELL
        'contract': {
            'symbol': best_signal['symbol'],
            'strike': best_signal['strike'],
            'expiry': best_signal['expiry'],
            'type': best_signal['type']
        },
        'entry_conditions': {
            'max_price': best_signal['price'] * 1.05,  # Pay up to 5% above sweep price
            'min_size': 100,  # Min 100 contracts
            'follow_size': min(best_signal['size'] * 0.01, 50)  # Follow with 1% of sweep size
        },
        'exit_conditions': {
            'time_stop': best_signal['expiry'],  # Hold until expiry
            'profit_target': 2.00,  # 200% gain
            'stop_loss': -0.40,  # 40% loss
            'trail_after': 1.00,  # Trail after 100% gain
            'scale_out': [0.50, 1.00, 1.50]  # Take profits at 50%, 100%, 150%
        },
        'size_pct': 0.10,  # 10% for high conviction swing trades
        'confidence': best_signal['smart_money_confidence'],
        'reason': f"Following {best_signal['size']} contract sweep at ${best_signal['premium']:,.0f}"
    }
```

#### 5.4.5 MOC Imbalance Trade Specification
```python
def build_moc_trade(self, market_data: dict) -> dict:
    """
    MOC: Market-on-close imbalance trades
    Time: 3:30 PM - 3:50 PM entry, exit at 4:00 PM
    Focus: Closing auction imbalances
    """
    
    # Calculate expected imbalance
    gamma_pull = market_data['gamma_exposure']['pin_strike'] - market_data['spot_price']
    order_flow = market_data['order_flow_imbalance']
    historical_pattern = self.get_historical_moc_pattern()
    
    # Combine factors
    expected_direction = np.sign(
        gamma_pull * 0.4 +
        order_flow * 0.3 +
        historical_pattern * 0.3
    )
    
    if abs(expected_direction) < 0.3:
        return {'strategy': 'NONE', 'reason': 'MOC signal too weak'}
    
    return {
        'strategy': 'MOC',
        'instrument': 'STOCK',  # Trade stock for MOC, not options
        'action': 'BUY' if expected_direction > 0 else 'SELL',
        'symbol': market_data['symbol'],
        'entry_conditions': {
            'entry_time': '3:45 PM - 3:50 PM',
            'order_type': 'LIMIT',
            'limit_price': 'MIDPOINT'  # Use IB's MIDPOINT order
        },
        'exit_conditions': {
            'exit_time': '3:59:50 PM',  # Exit 10 seconds before close
            'order_type': 'MOC',  # Market-on-close order
            'stop_loss': -0.003  # 0.3% stop
        },
        'size_calculation': {
            'base_size': 0.15,  # 15% of capital
            'adjust_for_volatility': True,
            'max_size': 0.25  # Never more than 25%
        },
        'confidence': min(abs(expected_direction) * 100, 85),
        'reason': f"MOC imbalance: Gamma pull to {market_data['gamma_exposure']['pin_strike']}"
    }
```

#### 5.4.6 Real-Time Strategy Switching
```python
def strategy_priority_matrix(self, current_time: datetime) -> list:
    """
    Returns priority order of strategies based on time of day
    """
    hour = current_time.hour
    minute = current_time.minute
    
    if hour < 10:  # First 30 minutes
        return ['14DTE']  # Only swing trades early
        
    elif 10 <= hour < 14:  # Mid-day
        return ['0DTE', '14DTE']  # Focus on gamma trades
        
    elif hour == 14:  # 2 PM hour
        return ['1DTE', '0DTE', '14DTE']  # Start positioning for overnight
        
    elif hour == 15 and minute < 30:  # 3:00-3:30 PM
        return ['1DTE', '0DTE']  # Last chance for overnight positioning
        
    elif hour == 15 and 30 <= minute <= 50:  # MOC window
        return ['MOC', '1DTE']  # MOC is priority
        
    else:
        return []  # No trades after 3:50 PM
```

#### 5.4.7 Contract Selection Helpers
```python
def select_option_contract(self, spot: float, strategy: str, direction: str) -> dict:
    """
    Selects specific strike and expiration based on strategy
    """
    
    if strategy == '0DTE':
        # 0DTE: First OTM or ATM for maximum gamma
        if direction == 'CALL':
            strike = math.ceil(spot / 0.5) * 0.5  # Round up to nearest 0.50
        else:
            strike = math.floor(spot / 0.5) * 0.5  # Round down to nearest 0.50
        expiry = datetime.now()
        
    elif strategy == '1DTE':
        # 1DTE: 1-2% OTM for directional, ATM for straddles
        if direction == 'CALL':
            strike = round(spot * 1.01)  # 1% OTM
        else:
            strike = round(spot * 0.99)  # 1% OTM
        expiry = datetime.now() + timedelta(days=1)
        
    elif strategy == '14DTE':
        # 14+DTE: Follow smart money strikes
        strike = self.find_highest_unusual_activity_strike(spot, direction)
        expiry = self.find_optimal_expiry(14, 45)  # Between 14-45 days
        
    return {
        'strike': strike,
        'expiry': expiry.strftime('%Y-%m-%d'),
        'type': direction
    }

def validate_contract_liquidity(self, contract: dict) -> bool:
    """
    Ensures contract has sufficient liquidity
    """
    # Get contract details from Alpha Vantage
    details = self.av_client.get_option_contract(
        symbol=contract['symbol'],
        strike=contract['strike'],
        expiry=contract['expiry'],
        type=contract['type']
    )
    
    # Liquidity checks
    checks = {
        'min_volume': details['volume'] >= 100,
        'min_open_interest': details['open_interest'] >= 500,
        'max_spread': (details['ask'] - details['bid']) <= details['mid'] * 0.15,
        'min_delta': abs(details['delta']) >= 0.15,
        'max_days_to_expiry': (
            datetime.strptime(contract['expiry'], '%Y-%m-%d') - datetime.now()
        ).days <= 45
    }
    
    return all(checks.values())
```

### 5.5 Complete Signal to Execution Example
```python
async def complete_signal_generation_with_strategy():
    """
    Shows full flow from market data to specific contract selection
    """
    
    # 1. Gather all market data
    market_data = {
        'symbol': 'SPY',
        'spot_price': 453.25,
        'vpin': 0.42,
        'order_book_imbalance': 0.35,
        'gamma_exposure': {
            'total_gex': 125.5,
            'pin_strike': 455.00,
            'profile': {450: 20, 455: 80, 460: 15}
        },
        'unusual_options_activity': [
            {
                'strike': 460,
                'type': 'CALL',
                'expiry': '2024-02-16',
                'size': 5000,
                'premium': 250000,
                'urgency_score': 0.9
            }
        ],
        'expected_overnight_move': 0.008,  # 0.8%
        'iv_term_structure': {'0dte': 18, '1dte': 22, '7dte': 16},
        'order_flow_bias': 0.4
    }
    
    # 2. Determine strategy and contract
    strategy_selector = StrategySelector()
    trade_spec = strategy_selector.select_strategy_and_contract(market_data)
    
    print(f"Selected Strategy: {trade_spec['strategy']}")
    
    if trade_spec['strategy'] == '0DTE':
        print(f"Contract: {trade_spec['contract']['strike']} {trade_spec['contract']['type']}")
        print(f"Expiry: Today")
        print(f"Reason: {trade_spec['reason']}")
        print(f"Exit by: {trade_spec['exit_conditions']['time_stop']}")
        
    elif trade_spec['strategy'] == '1DTE':
        print(f"Contract: {trade_spec['contract']['strike']} {trade_spec['contract']['type']}")
        print(f"Expiry: Tomorrow")
        print(f"Entry window: {trade_spec['entry_conditions']['entry_time']}")
        print(f"Expected move: {market_data['expected_overnight_move']:.1%}")
        
    elif trade_spec['strategy'] == 'MOC':
        print(f"Trading: {trade_spec['symbol']} stock")
        print(f"Direction: {trade_spec['action']}")
        print(f"Entry: {trade_spec['entry_conditions']['entry_time']}")
        print(f"Exit: MOC order at 4:00 PM")
        
    # 3. Generate signal with specific contract
    signal = TradingSignal(
        signal_id=str(uuid4()),
        timestamp=int(time.time() * 1000),
        symbol=trade_spec.get('symbol', market_data['symbol']),
        action=trade_spec['action'],
        strategy=trade_spec['strategy'],
        confidence=trade_spec['confidence'],
        contract=trade_spec.get('contract'),  # Specific option contract
        entry=market_data['spot_price'],
        stop_loss=calculate_stop(trade_spec),
        targets=calculate_targets(trade_spec),
        position_size=calculate_position_size(trade_spec),
        reason=trade_spec['reason']
    )
    
    return signal
```

---

## 6. Execution & Trade Management Layer

### 6.1 Execution Decision Engine
```python
@dataclass
class ExecutionDecision:
    """Decision to execute or skip a signal"""
    execute: bool
    reason: str
    order_type: str  # 'MARKET', 'LIMIT', 'STOP_LIMIT'
    size: int
    urgency: str  # 'IMMEDIATE', 'PATIENT', 'SCALED'
    routing: str  # 'SMART', 'DIRECTED'
    
class ExecutionManager:
    def __init__(self, ib_client, risk_manager, account_manager):
        self.ib = ib_client
        self.risk = risk_manager
        self.account = account_manager
        self.positions = {}  # Current positions
        self.pending_orders = {}  # Orders in flight
        
    def evaluate_signal(self, signal: TradingSignal) -> ExecutionDecision:
        """
        Decide whether and how to execute a signal
        """
        # 1. Check account constraints
        buying_power = self.account.get_buying_power()
        if signal.position_size > buying_power * 0.25:  # Max 25% per trade
            return ExecutionDecision(
                execute=False,
                reason="Position size exceeds 25% of buying power"
            )
        
        # 2. Check position limits
        current_positions = len(self.positions)
        if current_positions >= 5:  # Max 5 concurrent positions
            return ExecutionDecision(
                execute=False,
                reason="Maximum position limit reached"
            )
        
        # 3. Check correlation with existing positions
        correlation = self.check_correlation(signal.symbol)
        if correlation > 0.7:  # Too correlated with existing
            return ExecutionDecision(
                execute=False,
                reason=f"High correlation ({correlation:.2f}) with existing positions"
            )
        
        # 4. Check market conditions
        spread = self.get_bid_ask_spread(signal.symbol)
        if spread > 0.002:  # 0.2% spread too wide
            return ExecutionDecision(
                execute=False,
                reason=f"Spread too wide: {spread:.3%}"
            )
        
        # 5. Determine execution strategy based on confidence
        if signal.confidence > 85:
            return ExecutionDecision(
                execute=True,
                reason="High confidence signal",
                order_type='MARKET',
                size=self.calculate_position_size(signal),
                urgency='IMMEDIATE',
                routing='SMART'
            )
        elif signal.confidence > 70:
            return ExecutionDecision(
                execute=True,
                reason="Standard confidence signal",
                order_type='LIMIT',
                size=self.calculate_position_size(signal) * 0.75,
                urgency='PATIENT',
                routing='SMART'
            )
        else:
            return ExecutionDecision(
                execute=True,
                reason="Scaled entry for lower confidence",
                order_type='LIMIT',
                size=self.calculate_position_size(signal) * 0.5,
                urgency='SCALED',
                routing='SMART'
            )
```

### 6.2 Order Execution Through IBKR
```python
from ib_insync import *

class OrderExecutor:
    def __init__(self, ib_client):
        self.ib = ib_client
        self.active_orders = {}
        self.filled_trades = {}
        
    async def execute_signal(self, signal: TradingSignal, 
                            decision: ExecutionDecision) -> Order:
        """
        Execute a trading signal through IBKR
        """
        # Create contract
        if signal.strategy in ['0DTE', '1DTE']:
            contract = self.create_option_contract(signal)
        else:
            contract = Stock(signal.symbol, 'SMART', 'USD')
        
        # Create order based on decision
        if decision.order_type == 'MARKET':
            order = MarketOrder(
                action='BUY' if signal.action == 'BUY' else 'SELL',
                totalQuantity=decision.size,
                algoStrategy='Adaptive',
                algoParams=[TagValue('adaptivePriority', 'Urgent')]
            )
        elif decision.order_type == 'LIMIT':
            ticker = self.ib.reqTickers(contract)[0]
            limit_price = (ticker.bid + ticker.ask) / 2
            
            order = LimitOrder(
                action='BUY' if signal.action == 'BUY' else 'SELL',
                totalQuantity=decision.size,
                lmtPrice=limit_price,
                algoStrategy='Adaptive',
                algoParams=[TagValue('adaptivePriority', 'Patient')]
            )
        
        # Place the order
        trade = self.ib.placeOrder(contract, order)
        self.active_orders[signal.signal_id] = {
            'trade': trade,
            'signal': signal,
            'decision': decision,
            'timestamp': time.time()
        }
        
        # Monitor for fills
        asyncio.create_task(self.monitor_order_fill(signal.signal_id))
        
        return trade
```

### 6.3 Position Management
```python
class PositionManager:
    def __init__(self, ib_client, cache):
        self.ib = ib_client
        self.cache = cache
        self.positions = {}  # Active positions
        self.trade_history = []  # Completed trades
        
    async def manage_position(self, position_id: str):
        """
        Continuous position management loop
        """
        position = self.positions[position_id]
        
        while position['status'] == 'OPEN':
            # Get current price
            contract = Stock(position['symbol'], 'SMART', 'USD')
            ticker = self.ib.reqTickers(contract)[0]
            current_price = ticker.last
            
            # Update unrealized P&L
            if position['direction'] == 'BUY':
                position['pnl_unrealized'] = (
                    (current_price - position['entry_price']) * position['size']
                )
            else:
                position['pnl_unrealized'] = (
                    (position['entry_price'] - current_price) * position['size']
                )
            
            # Trail stop if profitable
            if position['pnl_unrealized'] > 0:
                await self.trail_stop(position, current_price)
            
            # Check targets for scaling out
            if await self.check_targets(position, current_price):
                await self.scale_out(position, current_price)
            
            # Update cache for real-time monitoring
            self.cache.setex(
                f"position:{position_id}",
                5,
                json.dumps(position)
            )
            
            await asyncio.sleep(1)  # Check every second
    
    async def scale_out(self, position: dict, target_price: float):
        """
        Scale out of position at targets
        """
        scale_percentages = [0.33, 0.50, 1.0]  # Take 1/3, 1/2, then all
        target_index = position['current_target']
        
        if target_index < len(scale_percentages):
            scale_pct = scale_percentages[target_index]
            shares_to_sell = int(position['size'] * scale_pct)
            
            # Place scale-out order
            contract = Stock(position['symbol'], 'SMART', 'USD')
            order = MarketOrder(
                action='SELL' if position['direction'] == 'BUY' else 'BUY',
                totalQuantity=shares_to_sell
            )
            
            trade = self.ib.placeOrder(contract, order)
            
            # Update position
            position['size'] -= shares_to_sell
            position['pnl_realized'] += (
                shares_to_sell * (target_price - position['entry_price'])
            )
            position['current_target'] += 1
```

### 6.4 Configuration-Driven Risk Management & Circuit Breakers
```python
class ConfigBasedRiskManager:
    """
    Risk management using configuration, not hardcoded values
    """
    def __init__(self, ib_client, positions_manager, config):
        self.ib = ib_client
        self.positions = positions_manager
        self.config = config
        
        # Load risk limits from configuration
        self.strategies = config.get('strategies', {})
        self.global_limits = config.get('risk_management', {}).get('global', {})
        
        # Circuit breakers from config with environment variable overrides
        self.circuit_breakers = config.get('risk_management', {}).get('circuit_breakers', {
            'max_consecutive_losses': int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3)),
            'daily_loss_shutdown': float(os.getenv('DAILY_LOSS_SHUTDOWN', 0.05)),
            'max_position_loss_pct': float(os.getenv('MAX_POSITION_LOSS_PCT', 0.02))
        })
        self.daily_stats = {
            'pnl': 0,
            'consecutive_losses': 0
        }
        
        # Load discovered parameters if configured
        if config.get('discovered_parameters', {}).get('override_from_cache', False):
            self._load_discovered_parameters()
    
    def _load_discovered_parameters(self):
        """Override config with discovered values from YOUR market"""
        discovered = self.cache.get('discovered_parameters')
        if discovered:
            for strategy_name, params in discovered.get('strategy_parameters', {}).items():
                if strategy_name in self.strategies:
                    self.strategies[strategy_name]['data_requirements'].update(params)
    
    def get_position_size(self, confidence: float, symbol: str, strategy_name: str = None):
        """
        Configuration-driven position sizing with discovered volatility
        """
        if not strategy_name:
            strategy_name = self._determine_active_strategy()
        
        # Get risk limits from config
        risk_limits = self.strategies.get(strategy_name, {}).get('risk_limits', {
            'max_position_pct': 0.02,  # Conservative default
            'max_loss_per_trade': 0.01
        })
        
        # Get YOUR actual account info
        account = self.cache.get('account_summary')
        buying_power = account.get('buying_power', 0)
        
        # Get YOUR actual volatility (not assumed)
        bars = self.cache.get_recent_bars(symbol, count=78)
        if bars:
            returns = np.diff(np.log([b.close for b in bars]))
            actual_volatility = np.std(returns) * np.sqrt(252 * 78)
        else:
            actual_volatility = 0.20  # Only use default if NO data
        
        # Kelly fraction from config
        kelly_fraction = self.global_limits.get('kelly_fraction', 0.25)
        
        # Position sizing method from config
        sizing_method = self.global_limits.get('position_sizing_method', 'kelly_volatility')
        
        if sizing_method == 'kelly_volatility':
            vol_adjustment = min(1.0, 0.15 / actual_volatility)
            position_pct = min(
                kelly_fraction * vol_adjustment * (confidence / 100),
                risk_limits['max_position_pct']
            )
        else:
            position_pct = risk_limits['max_position_pct']
        
        return {
            'position_value': buying_power * position_pct,
            'position_pct': position_pct,
            'volatility': actual_volatility,
            'method': sizing_method,
            'strategy': strategy_name
        }
    
    async def check_circuit_breakers(self) -> bool:
        """
        Check configuration-driven circuit breakers
        """
        account = self.cache.get('account_summary')
        account_value = account.get('net_liquidation', 100000)
        
        # Check daily loss limit (percentage-based from config)
        daily_loss_limit = account_value * self.circuit_breakers['daily_loss_shutdown']
        if self.daily_stats['pnl'] < -daily_loss_limit:
            await self.emergency_close_all(f"Daily loss limit exceeded: ${daily_loss_limit:.0f}")
            return True
        
        # Check consecutive losses (from config)
        if self.daily_stats['consecutive_losses'] >= self.circuit_breakers['max_consecutive_losses']:
            await self.emergency_close_all(f"Consecutive loss limit exceeded: {self.circuit_breakers['max_consecutive_losses']}")
            return True
        
        return False
    
    async def emergency_close_all(self, reason: str):
        """
        Emergency close all positions
        """
        print(f"🚨 EMERGENCY CLOSE: {reason}")
        
        # Cancel all pending orders
        for order in self.ib.orders():
            if order.orderStatus.status in ['PendingSubmit', 'Submitted']:
                self.ib.cancelOrder(order)
        
        # Close all positions at market
        for position_id, position in self.positions.positions.items():
            if position['status'] == 'OPEN':
                contract = Stock(position['symbol'], 'SMART', 'USD')
                
                close_order = MarketOrder(
                    action='SELL' if position['direction'] == 'BUY' else 'BUY',
                    totalQuantity=position['size']
                )
                
                self.ib.placeOrder(contract, close_order)
```

### 6.5 Trade Distribution to Subscribers
```python
class TradePublisher:
    """
    Publish ACTUAL TRADES (not just signals) to subscribers
    """
    
    def __init__(self, cache, websocket_server):
        self.cache = cache
        self.ws = websocket_server
        self.subscriber_tiers = {}  # user_id -> tier
        
    async def publish_trade_entry(self, position: dict, execution_price: float):
        """
        Publish when we actually enter a trade
        """
        trade_notification = {
            'type': 'TRADE_ENTRY',
            'timestamp': time.time(),
            'data': {
                'symbol': position['symbol'],
                'action': position['direction'],
                'entry_price': execution_price,
                'stop_loss': position['stop_loss'],
                'targets': position['targets'],
                'confidence': position.get('confidence', 0)
            }
        }
        
        # Different tiers see different info
        for user_id, tier in self.subscriber_tiers.items():
            if tier == 'basic':
                # Basic: Delayed and limited info
                await self.send_delayed(user_id, trade_notification, delay=300)
                
            elif tier == 'premium':
                # Premium: Real-time trade info
                await self.ws.send_to_user(user_id, trade_notification)
                
            elif tier == 'institutional':
                # Institutional: Everything including execution details
                institutional_notification = {
                    **trade_notification,
                    'execution_details': {
                        'size': position['size'],
                        'order_type': position.get('order_type'),
                        'microstructure_metrics': position.get('metrics', {})
                    }
                }
                await self.ws.send_to_user(user_id, institutional_notification)
```

---

## 7. Distribution & Monetization

### 6.1 Subscription Tiers
```python
SUBSCRIPTION_TIERS = {
    'basic': {
        'price': 99,  # USD per month
        'features': [
            'pre_market_reports',
            'end_of_day_summary'
        ],
        'signal_delay': 60,  # seconds
        'max_signals_per_day': 5,
        'includes_metrics': False,
        'api_access': False
    },
    
    'premium': {
        'price': 499,
        'features': [
            'real_time_signals',
            'all_strategies',
            'entry_exit_levels',
            'risk_parameters'
        ],
        'signal_delay': 0,
        'max_signals_per_day': 50,
        'includes_metrics': True,
        'api_access': False
    },
    
    'institutional': {
        'price': 2999,
        'features': [
            'everything_in_premium',
            'raw_microstructure_data',
            'white_label_option',
            'custom_alerts',
            'api_access',
            'fix_protocol'
        ],
        'signal_delay': 0,
        'max_signals_per_day': -1,  # Unlimited
        'includes_metrics': True,
        'api_access': True
    }
}
```

### 6.2 API Endpoints

#### WebSocket Feed (Primary)
```python
# Real-time trade stream (actual executions)
ws://api.yourdomain.com/v1/trades

# Entry notification
{
    "type": "trade_entry",
    "data": {
        "trade_id": "uuid",
        "symbol": "SPY",
        "action": "BUY",
        "entry_price": 453.25,  # Actual fill price
        "size": 500,            # Actual shares (institutional only)
        "stop_loss": 450.00,
        "targets": [456.00, 458.00, 460.00],
        "confidence": 78.5,
        "strategy": "0DTE"
    },
    "timestamp": 1234567890123
}

# Position update
{
    "type": "position_update",
    "data": {
        "trade_id": "uuid",
        "symbol": "SPY",
        "update_type": "stop_trailed",
        "new_stop": 452.50,
        "unrealized_pnl": 625.00,
        "message": "Stop moved to breakeven"
    },
    "timestamp": 1234567890456
}

# Exit notification
{
    "type": "trade_exit",
    "data": {
        "trade_id": "uuid",
        "symbol": "SPY",
        "exit_price": 456.00,  # Actual fill price
        "exit_reason": "target_1",
        "realized_pnl": 1375.00,
        "trade_duration_seconds": 3600
    },
    "timestamp": 1234567890789
}
```

#### REST API
```python
# Authentication
POST /api/v1/auth/token
Headers: {"api_key": "xxx", "secret": "yyy"}
Response: {"token": "jwt_token", "expires": 3600}

# Get signals
GET /api/v1/signals
Headers: {"Authorization": "Bearer jwt_token"}
Query: {
    "symbol": "SPY",
    "strategy": "0DTE",
    "min_confidence": 70,
    "from_timestamp": 1234567890000
}

# Get market metrics (institutional only)
GET /api/v1/metrics/{symbol}
Response: {
    "vpin": 0.42,
    "order_book_imbalance": 0.35,
    "gamma_exposure": 125.5,
    "hidden_orders": true
}
```

### 6.3 Distribution Channels

#### Discord Integration
```python
# Webhook for community signals
DISCORD_CONFIG = {
    'webhook_url': 'https://discord.com/api/webhooks/xxx',
    'channels': {
        'free': '#free-signals',     # Delayed basic
        'premium': '#premium-signals', # Real-time
        'vip': '#institutional'       # Everything
    },
    'embed_format': {
        'color': 0x00ff00,  # Green for BUY
        'fields': ['symbol', 'action', 'confidence', 'entry']
    }
}
```

#### WhatsApp Business API
```python
# Alert configuration
WHATSAPP_CONFIG = {
    'api_endpoint': 'https://api.whatsapp.com/v1/',
    'business_id': 'xxx',
    'alert_templates': {
        'high_conviction': 'signal_urgent',
        'standard': 'signal_normal'
    }
}
```

---

## 7. Performance Requirements

### 7.1 Latency Targets

| Component | Target | Maximum | Measurement Point |
|-----------|--------|---------|-------------------|
| Order Book Update | <10ms | 20ms | IBKR → Cache |
| VPIN Calculation | <30ms | 50ms | Per 100 trades |
| Options Analysis | <100ms | 200ms | Full chain processing |
| Signal Generation | <50ms | 100ms | All metrics → signal |
| End-to-End | <150ms | 500ms | Market event → distributed |
| API Response | <20ms | 50ms | Request → response |

### 7.2 Throughput Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Symbols Monitored | 10 | Concurrent |
| Order Book Updates | 10,000/sec | All symbols |
| Options Calculations | 100/sec | All chains |
| Signals Generated | 100/day | Max across all symbols |
| API Requests | 1,000/sec | All clients |
| WebSocket Connections | 1,000 | Concurrent |

### 7.3 Reliability Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Uptime | 99.9% | During market hours |
| Data Loss | 0% | Critical signals |
| Recovery Time | <30 sec | After failure |
| Failover Time | <5 sec | Backup systems |

---

## 8. System Requirements

### 8.1 Hardware Requirements

#### Minimum Configuration
```yaml
CPU: 8 cores (Intel i7 or AMD Ryzen 7)
RAM: 16 GB
Storage: 100 GB SSD (logs only)
Network: 100 Mbps dedicated
OS: Ubuntu 20.04 LTS or macOS 12+
```

#### Recommended Configuration
```yaml
CPU: 16 cores (Intel i9 or AMD Ryzen 9)
RAM: 32 GB
Storage: 500 GB NVMe SSD
Network: 1 Gbps dedicated
OS: Ubuntu 22.04 LTS
GPU: Optional (for ML inference acceleration)
```

### 8.2 Software Stack

```yaml
# Core Runtime
Python: 3.11+ (with asyncio)
Redis: 7.0+

# Python Libraries
ib_insync: 0.9.86       # IBKR connection
aiohttp: 3.9.0         # Async HTTP
fastapi: 0.109.0       # API server
uvicorn: 0.27.0        # ASGI server
websockets: 12.0       # WebSocket support
numpy: 1.26.0          # Numerical computing
pandas: 2.1.0          # Data manipulation
pydantic: 2.5.0        # Data validation

# Monitoring
prometheus-client: 0.19.0
grafana: Latest Docker image

# Optional
pytorch: 2.1.0         # If using ML models
numba: 0.58.0         # JIT compilation for speed
```

### 8.3 Network Requirements

```yaml
# Connectivity
IBKR Gateway: Low latency (<5ms ideal)
Alpha Vantage API: Stable HTTPS
Redis: Local or <1ms network

# Firewall Rules
Inbound:
  - 8000 (API Server)
  - 8001 (WebSocket)
  - 9090 (Prometheus)
  
Outbound:
  - 443 (HTTPS)
  - 7496/7497 (IBKR TWS/Gateway)
  - 4001/4002 (IBKR Live/Paper)
```

### 8.4 Execution Configuration

```yaml
# config/execution.yaml
execution:
  # Account settings
  account_id: 'DU1234567'  # Paper account
  
  # Position limits
  max_positions: 5
  max_position_size_pct: 0.25  # 25% of account
  max_correlation: 0.70
  
  # Risk management
  risk_per_trade: 0.02  # 2% of account
  initial_stop_atr: 1.5
  trailing_stop_pct: 0.5
  
  # Targets (ATR multiples)
  targets: [1.0, 2.0, 3.0]
  scale_out_pcts: [0.33, 0.50, 1.00]
  
  # Circuit breakers
  circuit_breakers:
    max_daily_loss: 2000
    max_consecutive_losses: 3
    max_drawdown_pct: 0.02
    
  # Order routing
  routing:
    default: 'SMART'
    algos:
      urgent: 'Adaptive'
      patient: 'Adaptive'
```

---

## 9. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

#### Day 1-2: Environment Setup
- [ ] Install Redis, configure memory limits
- [ ] Setup Python environment, install packages
- [ ] Configure IBKR connection (paper account)
- [ ] Test Alpha Vantage API access
- [ ] Setup TWS/IB Gateway for execution

#### Day 3-4: Data Pipeline
- [ ] Implement IBKR Level 2 subscription
- [ ] Build Alpha Vantage client with rate limiting
- [ ] Create cache manager with TTL logic
- [ ] Test data flow end-to-end
- [ ] Test order placement (paper)

#### Day 5-7: Basic Analytics
- [ ] Implement VPIN calculation
- [ ] Build order book imbalance metrics
- [ ] Add options Greeks retrieval (not calculation!)
- [ ] Create simple signal generator
- [ ] Basic execution decision logic

### Phase 2: Advanced Analytics & Execution (Week 2)

#### Day 8-10: Microstructure & Order Management
- [ ] Hidden order detection
- [ ] Sweep detection algorithm
- [ ] Book pressure metrics
- [ ] Order executor implementation
- [ ] Position tracking system

#### Day 11-14: Options Intelligence & Risk Management
- [ ] Gamma exposure calculations
- [ ] Multi-timeframe analysis (0DTE, 1DTE, 14DTE)
- [ ] MOC imbalance prediction
- [ ] Stop loss management
- [ ] Circuit breakers implementation

### Phase 3: Signal, Execution & Distribution (Week 3)

#### Day 15-17: Complete Trading Loop
- [ ] Signal to execution pipeline
- [ ] Position management automation
- [ ] Trailing stop implementation
- [ ] Scale-out logic
- [ ] P&L tracking

#### Day 18-21: Distribution Layer
- [ ] WebSocket server (real trades)
- [ ] REST API with JWT auth
- [ ] Subscription tier enforcement
- [ ] Discord integration (actual positions)
- [ ] Performance dashboard

### Phase 4: Production & Optimization (Week 4)

#### Day 22-24: Testing
- [ ] Paper trading validation
- [ ] Load testing (1000 connections)
- [ ] Latency optimization
- [ ] Failure recovery testing
- [ ] Circuit breaker testing

#### Day 25-28: Deployment
- [ ] Production environment setup
- [ ] Real money account connection
- [ ] Monitoring dashboards
- [ ] Alert configuration
- [ ] Go-live with limited capital

---

## 10. Monitoring & Operations

### 10.1 Real-Time Trading Dashboard

#### 10.1.1 Dashboard Architecture
```python
class TradingDashboard:
    """
    Web-based real-time monitoring interface
    Tech Stack: FastAPI + WebSocket + React/Vue frontend
    Update Frequency: 1 second for critical metrics
    """
    
    def __init__(self):
        self.sections = {
            'system_health': SystemHealthPanel(),
            'positions': PositionsPanel(),
            'signals': SignalsPanel(),
            'market_data': MarketDataPanel(),
            'pnl': PnLPanel(),
            'risk': RiskPanel(),
            'execution': ExecutionPanel(),
            'alerts': AlertsPanel()
        }
```

#### 10.1.2 Dashboard Layout & Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ALPHATRADER DASHBOARD - LIVE                      │
│  Connected: ✅ IBKR | ✅ AlphaVantage | ✅ Redis | Time: 10:32:45 AM │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────── POSITIONS (3/5) ──────────────┬─── P&L TODAY ───┐ │
│  │ Symbol │ Strategy │ Entry  │ Current │ P&L    │ Gross: +$2,450  │ │
│  │ SPY    │ 0DTE    │ 453.25 │ 454.10 │ +$425  │ Fees:    -$12   │ │
│  │ QQQ    │ 1DTE    │ 371.50 │ 371.80 │ +$150  │ Net:   +$2,438  │ │
│  │ AAPL   │ 14DTE   │ 189.25 │ 188.90 │ -$175  │ Win Rate: 67%   │ │
│  └──────────────────────────────────────────────┴─────────────────┘ │
│                                                                       │
│  ┌─────────────── MARKET MICROSTRUCTURE ────────────────────────┐   │
│  │ Symbol: SPY                          Last: 454.10             │   │
│  │ ┌─── ORDER BOOK ───┐  ┌─── INDICATORS ───┐  ┌─── GAMMA ───┐  │   │
│  │ │ ASK  SIZE        │  │ VPIN:      0.42 │  │ GEX: 125.5M  │  │   │
│  │ │ 454.15  2500    │  │ OBI:      +0.35 │  │ PIN: 455.00  │  │   │
│  │ │ 454.14  1800    │  │ Sweep:      NO  │  │ Flip: 452.00 │  │   │
│  │ │ 454.13  3200    │  │ Hidden:    YES  │  │              │  │   │
│  │ │ 454.12  1500    │  │ Pressure:  BUY  │  │ ▁▂▄█▆▃▂▁     │  │   │
│  │ │ 454.11  2100    │  │                 │  │ 450    455   │  │   │
│  │ │ ───── SPREAD ── │  └─────────────────┘  └──────────────┘  │   │
│  │ │ 454.10  1900    │                                          │   │
│  │ │ 454.09  2800    │  ┌─── OPTIONS FLOW ───┐                 │   │
│  │ │ 454.08  1200    │  │ 455C 0DTE: 5000 vol                  │   │
│  │ │ 454.07  3500    │  │ 450P 1DTE: 2000 vol                  │   │
│  │ │ 454.06  2200    │  │ Unusual: AAPL 190C                   │   │
│  │ │ BID  SIZE       │  └───────────────────┘                  │   │
│  └──────────────────┘                                           │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────── ACTIVE SIGNALS ────────┬──── PENDING EXECUTION ─────┐   │
│  │ Time  │ Symbol │ Strategy │ Conf│ Status    │ Fill Price    │   │
│  │ 10:31 │ IWM   │ 0DTE    │ 78% │ PENDING   │ Limit 198.50  │   │
│  │ 10:28 │ TSLA  │ 14DTE   │ 65% │ REJECTED  │ Spread wide   │   │
│  └────────────────────────────────┴──────────────────────────────┘   │
│                                                                       │
│  ┌──────────── RISK METRICS ──────────┬──── CIRCUIT BREAKERS ───┐   │
│  │ Daily Loss:    -$450 / -$2000     │ Daily Loss:        ✅     │   │
│  │ Position Risk:  $1,250            │ Consecutive Loss:  ✅     │   │
│  │ Correlation:    0.45              │ Drawdown:         ✅     │   │
│  │ Buying Power:   $48,500           │ Position Limit:    ⚠️     │   │
│  │ Margin Used:    42%               │ API Rate:         ✅     │   │
│  └────────────────────────────────────┴────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────── ALERTS & LOGS ──────────────────────────────┐   │
│  │ 10:32:41 [INFO]  Signal generated: SPY 0DTE 455C confidence 82% │   │
│  │ 10:32:39 [EXEC]  Order filled: QQQ 371.80 x 100 shares         │   │
│  │ 10:32:35 [WARN]  VPIN elevated: 0.42 - toxic flow detected     │   │
│  │ 10:32:30 [INFO]  Stop moved to breakeven for SPY position      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

#### 10.1.3 Dashboard Implementation
```python
# Backend: FastAPI with WebSocket
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import asyncio
import json

app = FastAPI()

class DashboardServer:
    def __init__(self, cache, ib_client, position_manager):
        self.cache = cache
        self.ib = ib_client
        self.positions = position_manager
        self.connected_clients = set()
        
    async def broadcast_updates(self):
        """Push updates to all connected dashboard clients"""
        while True:
            dashboard_data = {
                'timestamp': time.time(),
                'system_health': self.get_system_health(),
                'positions': self.get_positions_data(),
                'market_data': self.get_market_data(),
                'pnl': self.get_pnl_data(),
                'risk_metrics': self.get_risk_metrics(),
                'recent_signals': self.get_recent_signals(),
                'alerts': self.get_alerts()
            }
            
            # Broadcast to all connected clients
            for client in self.connected_clients:
                await client.send_json(dashboard_data)
                
            await asyncio.sleep(1)  # Update every second
    
    def get_positions_data(self):
        """Real-time position data"""
        positions = []
        for pos_id, pos in self.positions.positions.items():
            current_price = self.get_current_price(pos['symbol'])
            positions.append({
                'symbol': pos['symbol'],
                'strategy': pos['strategy'],
                'entry': pos['entry_price'],
                'current': current_price,
                'size': pos['size'],
                'pnl_unrealized': pos['pnl_unrealized'],
                'pnl_realized': pos['pnl_realized'],
                'stop': pos['stop_loss'],
                'target': pos['targets'][pos['current_target']] if pos['targets'] else None,
                'status': pos['status'],
                'entry_time': pos['entry_time'],
                'bars_in_trade': int((time.time() - pos['entry_time']) / 300)  # 5-min bars
            })
        return positions
    
    def get_market_data(self):
        """Real-time market microstructure"""
        symbol = 'SPY'  # Primary symbol
        
        # Get from cache (updated every second)
        return {
            'symbol': symbol,
            'last_price': self.cache.get(f'last:{symbol}'),
            'order_book': json.loads(self.cache.get(f'book:{symbol}') or '{}'),
            'vpin': float(self.cache.get(f'vpin:{symbol}') or 0),
            'order_imbalance': float(self.cache.get(f'obi:{symbol}') or 0),
            'gamma_exposure': json.loads(self.cache.get(f'gex:{symbol}') or '{}'),
            'sweep_detected': self.cache.get(f'sweep:{symbol}') == '1',
            'hidden_orders': self.cache.get(f'hidden:{symbol}') == '1',
            'options_flow': json.loads(self.cache.get(f'flow:{symbol}') or '[]')
        }
    
    def get_pnl_data(self):
        """Real-time P&L tracking"""
        return {
            'realized_pnl': sum(p['pnl_realized'] for p in self.positions.positions.values()),
            'unrealized_pnl': sum(p['pnl_unrealized'] for p in self.positions.positions.values()),
            'fees': self.calculate_fees(),
            'net_pnl': self.calculate_net_pnl(),
            'daily_high': self.cache.get('daily_pnl_high') or 0,
            'daily_low': self.cache.get('daily_pnl_low') or 0,
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'trades_today': self.cache.get('trades_today') or 0
        }

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard"""
    await websocket.accept()
    dashboard_server.connected_clients.add(websocket)
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            
            # Handle dashboard commands (pause, resume, close_all, etc.)
            if data == "PAUSE":
                await pause_trading()
            elif data == "RESUME":
                await resume_trading()
            elif data == "CLOSE_ALL":
                await emergency_close_all()
                
    except Exception as e:
        dashboard_server.connected_clients.remove(websocket)
```

#### 10.1.4 Mobile Dashboard
```python
class MobileDashboard:
    """
    Simplified mobile view for monitoring on the go
    Accessible via: https://yourdomain.com/mobile
    """
    
    def get_mobile_summary(self):
        return {
            'status': 'TRADING' if self.is_trading_active() else 'PAUSED',
            'positions': {
                'open': len(self.positions),
                'limit': 5,
                'total_value': sum(p['value'] for p in self.positions)
            },
            'pnl': {
                'today': self.get_daily_pnl(),
                'open': self.get_open_pnl()
            },
            'alerts': self.get_critical_alerts_only(),
            'quick_actions': [
                'PAUSE_TRADING',
                'CLOSE_ALL_POSITIONS',
                'REDUCE_RISK',
                'VIEW_DETAILS'
            ]
        }
```

#### 10.1.5 Performance Metrics Dashboard
```python
class PerformanceMetricsDashboard:
    """
    Detailed performance analytics panel
    """
    
    def get_performance_metrics(self):
        return {
            'latency': {
                'signal_generation': self.get_metric('signal_latency_ms'),
                'order_execution': self.get_metric('execution_latency_ms'),
                'end_to_end': self.get_metric('e2e_latency_ms')
            },
            'api_usage': {
                'alpha_vantage': {
                    'calls_used': self.get_av_calls_used(),
                    'calls_remaining': 600 - self.get_av_calls_used(),
                    'reset_in': self.get_av_reset_time()
                },
                'ibkr': {
                    'messages_per_sec': self.get_ibkr_message_rate(),
                    'connection_quality': self.get_connection_quality()
                }
            },
            'cache_performance': {
                'hit_rate': self.get_cache_hit_rate(),
                'memory_used': self.get_redis_memory(),
                'keys_count': self.get_redis_keys_count()
            },
            'system_resources': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters()
            }
        }
```

#### 10.1.6 Dashboard Alerts Configuration
```python
DASHBOARD_ALERTS = {
    'critical': {
        'position_loss_exceeds': 300,  # Alert if any position down $300
        'daily_loss_approaches': 0.8,   # Alert at 80% of daily loss limit
        'circuit_breaker_near': 0.9,    # Alert at 90% of any circuit breaker
        'api_rate_limit': 0.9,          # Alert at 90% of rate limit
        'connection_lost': True,         # Alert on any connection loss
    },
    'warning': {
        'vpin_elevated': 0.4,
        'spread_wide': 0.002,
        'low_liquidity': 100,  # Volume below 100
        'correlation_high': 0.7,
        'margin_usage': 0.75
    },
    'info': {
        'new_signal': True,
        'position_opened': True,
        'position_closed': True,
        'stop_adjusted': True
    }
}

class AlertManager:
    def check_alerts(self):
        alerts = []
        
        # Check critical alerts
        for pos in self.positions:
            if pos['pnl_unrealized'] < -DASHBOARD_ALERTS['critical']['position_loss_exceeds']:
                alerts.append({
                    'level': 'CRITICAL',
                    'message': f"{pos['symbol']} position down ${abs(pos['pnl_unrealized']):.2f}",
                    'action': 'CONSIDER_EXIT'
                })
        
        # Check system alerts
        if not self.ib.is_connected():
            alerts.append({
                'level': 'CRITICAL',
                'message': 'IBKR connection lost!',
                'action': 'RECONNECTING'
            })
            
        return alerts
```

### 10.2 Operational Dashboard Access

#### 10.2.1 Access Methods
```yaml
# Dashboard URLs
production:
  main: https://yourdomain.com/dashboard
  mobile: https://yourdomain.com/mobile
  api: https://yourdomain.com/api/v1/metrics
  
development:
  main: http://localhost:8080/dashboard
  mobile: http://localhost:8080/mobile
  api: http://localhost:8000/api/v1/metrics

# Authentication
auth:
  method: JWT
  expiry: 24_hours
  mfa: optional  # For production
  
# Refresh rates
update_frequencies:
  positions: 1_second
  market_data: 1_second
  pnl: 5_seconds
  system_health: 10_seconds
  performance: 30_seconds
```

#### 10.2.2 Dashboard Commands
```python
# Interactive commands available from dashboard
DASHBOARD_COMMANDS = {
    'PAUSE_ALL': 'Pause all new signal generation',
    'RESUME_ALL': 'Resume signal generation',
    'CLOSE_POSITION': 'Close specific position at market',
    'CLOSE_ALL': 'Emergency close all positions',
    'ADJUST_STOP': 'Manually adjust stop loss',
    'REDUCE_SIZE': 'Reduce position size',
    'HEDGE': 'Add hedge position',
    'EXPORT_DATA': 'Export performance data',
    'SCREENSHOT': 'Capture dashboard screenshot',
    'SEND_REPORT': 'Email performance report'
}
```

### 10.3 Key Metrics to Track

```python
METRICS = {
    # System Health
    'latency_p50': Histogram('latency_p50_ms'),
    'latency_p99': Histogram('latency_p99_ms'),
    'error_rate': Counter('errors_total'),
    'uptime': Gauge('uptime_seconds'),
    
    # Data Quality
    'vpin_score': Gauge('vpin_current'),
    'cache_hit_rate': Summary('cache_hit_ratio'),
    'api_calls': Counter('api_calls_total'),
    
    # Business Metrics
    'signals_generated': Counter('signals_total'),
    'signal_accuracy': Gauge('signal_accuracy_pct'),
    'active_subscriptions': Gauge('subscriptions_active'),
    'revenue_mrr': Gauge('monthly_recurring_revenue')
}
```

### 10.3 Key Metrics to Track

```python
METRICS = {
    # System Health
    'latency_p50': Histogram('latency_p50_ms'),
    'latency_p99': Histogram('latency_p99_ms'),
    'error_rate': Counter('errors_total'),
    'uptime': Gauge('uptime_seconds'),
    
    # Data Quality
    'vpin_score': Gauge('vpin_current'),
    'cache_hit_rate': Summary('cache_hit_ratio'),
    'api_calls': Counter('api_calls_total'),
    
    # Business Metrics
    'signals_generated': Counter('signals_total'),
    'trades_executed': Counter('trades_total'),
    'positions_open': Gauge('positions_open'),
    'signal_accuracy': Gauge('signal_accuracy_pct'),
    'actual_pnl': Gauge('pnl_realized'),
    'active_subscriptions': Gauge('subscriptions_active'),
    'revenue_mrr': Gauge('monthly_recurring_revenue')
}
```

### 10.4 Daily Operational Procedures

#### Morning Startup (8:30 AM ET)
```bash
# 1. System health check
./scripts/health_check.py

# 2. Check dashboard is accessible
curl http://localhost:8080/dashboard/health

# 3. Clear stale cache
redis-cli FLUSHDB

# 4. Pre-warm cache
./scripts/warm_cache.py --symbols SPY,QQQ,IWM

# 5. Start services
./scripts/start_services.sh

# 6. Verify data flow in dashboard
# Check all panels are updating

# 7. Verify IBKR connection
# Dashboard should show: ✅ IBKR Connected
```

#### Market Hours Monitoring
- Keep dashboard open on dedicated monitor
- Set audio alerts for critical events
- Check positions panel every 5 minutes
- Monitor VPIN/OBI for toxicity
- Watch circuit breaker indicators
- Verify stop losses are in place

#### End of Day (4:30 PM ET)
```bash
# 1. Verify all positions closed or overnight
# Check dashboard positions panel

# 2. Generate performance report
./scripts/daily_report.py

# 3. Screenshot dashboard for records
./scripts/capture_dashboard.py

# 4. Archive logs
./scripts/archive_logs.sh

# 5. Update subscription metrics
./scripts/update_billing.py
```

### 10.5 Emergency Procedures via Dashboard

#### Quick Actions from Dashboard
1. **Red "CLOSE ALL" Button**: Emergency flatten all positions
2. **Yellow "PAUSE" Button**: Stop new signal generation
3. **"REDUCE RISK" Button**: Cut all positions by 50%
4. **"HEDGE" Button**: Add protective hedge

#### Connection Loss Recovery
- Dashboard auto-reconnects with exponential backoff
- Shows connection status for each component
- One-click reconnect buttons
- Fallback to mobile dashboard if main fails

### 10.6 Dashboard Customization

```python
# User preferences for dashboard
DASHBOARD_PREFERENCES = {
    'layout': 'default',  # or 'compact', 'mobile', 'trading_floor'
    'refresh_rate': 1000,  # milliseconds
    'sound_alerts': True,
    'color_scheme': 'dark',
    'panels_visible': [
        'positions',
        'market_data', 
        'pnl',
        'risk',
        'alerts'
    ],
    'chart_settings': {
        'show_gamma_profile': True,
        'show_order_book_depth': 10,
        'show_vpin_history': True
    },
    'alert_preferences': {
        'critical': {'sound': True, 'popup': True, 'email': True},
        'warning': {'sound': False, 'popup': True, 'email': False},
        'info': {'sound': False, 'popup': False, 'email': False}
    }
}
```

#### Connection Loss Recovery
```python
async def connection_monitor():
    """Auto-reconnect with exponential backoff"""
    while True:
        try:
            if not ibkr.is_connected():
                await reconnect_ibkr()
            if not check_av_health():
                rotate_av_keys()
        except Exception as e:
            logger.error(f"Connection error: {e}")
            await asyncio.sleep(backoff_time)
            backoff_time *= 2
```

#### Circuit Breakers
```python
CIRCUIT_BREAKERS = {
    'max_signals_per_minute': 10,
    'max_api_errors': 50,
    'max_latency_ms': 1000,
    'min_cache_hit_rate': 0.3
}

def check_circuit_breakers():
    if signals_per_minute > CIRCUIT_BREAKERS['max_signals_per_minute']:
        pause_signal_generation()
    if api_errors > CIRCUIT_BREAKERS['max_api_errors']:
        switch_to_backup_mode()
```

---

## 11. Security & Compliance

### 11.1 Security Measures

```yaml
# API Security
- JWT tokens with 1-hour expiry
- Rate limiting per tier
- IP whitelist for institutional
- SSL/TLS for all connections

# Data Security
- Redis AUTH enabled
- Encrypted connections
- No PII stored
- Audit logging enabled

# Code Security
- Environment variables for secrets
- No credentials in code
- Regular dependency updates
- Security scanning in CI/CD
```

### 11.2 Compliance Considerations

```yaml
# Disclaimers
- Not investment advice
- Past performance disclosure
- Risk warnings
- Terms of service

# Data Rights
- No redistribution of market data
- Alpha Vantage attribution
- IBKR terms compliance

# Regulatory
- No front-running
- Best execution (when executing)
- Fair access to all tiers
```

---

## 12. Cost Analysis

### 12.1 Infrastructure Costs (Monthly)

| Component | Cost | Notes |
|-----------|------|-------|
| Server (16 core, 32GB) | $200 | DigitalOcean/AWS |
| Redis Cloud (4GB) | $100 | Or self-hosted |
| Alpha Vantage Premium | $250 | 600 calls/min |
| IBKR Market Data | $100 | Level 2 subscription |
| Monitoring (Datadog) | $100 | Optional |
| **Total** | **$750** | Per month |

### 12.2 Revenue Projections

| Tier | Price | Users | Monthly Revenue |
|------|-------|-------|-----------------|
| Basic | $99 | 100 | $9,900 |
| Premium | $499 | 30 | $14,970 |
| Institutional | $2,999 | 5 | $14,995 |
| **Total** | | **135** | **$39,865** |

### 12.3 Break-Even Analysis

```
Fixed Costs: $750/month
Break-even: 8 basic users OR 2 premium users
Profit Margin at 135 users: 98%
```

---

## Appendix A: Code Examples

### Complete Trading Pipeline Example
```python
async def complete_trading_pipeline():
    """
    Full pipeline from market event to closed position
    """
    # Initialize all components
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)
    
    av_client = AlphaVantageClient(api_key=os.getenv('AV_API_KEY'))
    cache = redis.Redis(decode_responses=True)
    
    execution_mgr = ExecutionManager(ib)
    order_executor = OrderExecutor(ib)
    position_mgr = PositionManager(ib, cache)
    emergency_mgr = EmergencyManager(ib, position_mgr)
    publisher = TradePublisher(cache)
    
    # Main trading loop
    while True:
        # 1. Check circuit breakers
        if await emergency_mgr.check_circuit_breakers():
            break
            
        # 2. Get market data (IBKR Level 2)
        book = await get_level2_book('SPY')
        
        # 3. Get options data (Alpha Vantage)
        options = await av_client.get_options_with_greeks('SPY')
        
        # 4. Calculate indicators
        vpin = calculate_vpin(recent_trades)
        obi = calculate_order_book_imbalance(book)
        gex = calculate_gamma_exposure(options)
        
        # 5. Generate signal
        signal = generate_signal(
            symbol='SPY',
            microstructure={'vpin': vpin, 'obi': obi},
            options={'gex': gex},
            confidence=calculate_confidence(vpin, obi, gex)
        )
        
        if signal and signal.confidence > 60:
            # 6. Evaluate for execution
            decision = execution_mgr.evaluate_signal(signal)
            
            if decision.execute:
                # 7. Place order through IBKR
                trade = await order_executor.execute_signal(signal, decision)
                
                # 8. Wait for fill
                fill_event = await order_executor.wait_for_fill(trade)
                
                if fill_event.status == 'FILLED':
                    # 9. Create position with stops
                    position = await position_mgr.create_position(
                        signal, 
                        fill_event.fill_price,
                        fill_event.fill_size
                    )
                    
                    # 10. Place stop loss immediately
                    await position_mgr.place_stop_loss(position)
                    
                    # 11. Notify subscribers of actual trade
                    await publisher.publish_trade_entry(
                        position, 
                        fill_event.fill_price
                    )
                    
                    # 12. Start position management
                    asyncio.create_task(
                        position_mgr.manage_position(position.id)
                    )
        
        await asyncio.sleep(1)

# Run the complete system
asyncio.run(complete_trading_pipeline())
```

### Position Lifecycle Example
```python
async def position_lifecycle(signal: TradingSignal):
    """
    Complete lifecycle of a trade from entry to exit
    """
    # Entry
    print(f"📊 Signal Generated: {signal.symbol} {signal.action} @ {signal.entry}")
    
    # Execution
    decision = execution_mgr.evaluate_signal(signal)
    if not decision.execute:
        print(f"❌ Signal rejected: {decision.reason}")
        return
    
    print(f"✅ Executing: {decision.size} shares, {decision.order_type} order")
    trade = await order_executor.execute_signal(signal, decision)
    
    # Fill
    fill = await wait_for_fill(trade)
    print(f"🎯 FILLED: {fill.size} @ ${fill.price:.2f}")
    
    # Position Management
    position = Position(
        symbol=signal.symbol,
        entry=fill.price,
        size=fill.size,
        stop=signal.stop_loss,
        targets=signal.targets
    )
    
    # Manage until exit
    while position.is_open:
        current_price = get_current_price(position.symbol)
        
        # Trail stop if profitable
        if position.unrealized_pnl > 0:
            new_stop = position.entry + (current_price - position.entry) * 0.5
            if new_stop > position.stop:
                position.stop = new_stop
                print(f"📈 Stop trailed to ${position.stop:.2f}")
        
        # Check targets
        for i, target in enumerate(position.targets):
            if current_price >= target and not position.targets_hit[i]:
                scale_size = position.size // 3
                await scale_out(position, scale_size, target)
                print(f"🎯 Target {i+1} hit! Scaled out {scale_size} @ ${target:.2f}")
                position.targets_hit[i] = True
        
        # Check stop
        if current_price <= position.stop:
            await close_position(position, current_price)
            print(f"🛑 Stop hit! Closed @ ${current_price:.2f}")
            break
            
        await asyncio.sleep(1)
    
    # Final P&L
    print(f"💰 Final P&L: ${position.realized_pnl:.2f}")
    
    # Notify subscribers
    await publisher.publish_trade_exit(position, position.exit_price, position.exit_reason)
```

---

## Appendix B: Testing Strategy

### Unit Tests
```python
def test_vpin_calculation():
    trades = generate_mock_trades(1000)
    vpin = calculate_vpin(trades)
    assert 0 <= vpin <= 1
    
def test_signal_generation():
    mock_data = load_test_fixture('signal_test.json')
    signal = generate_signal(**mock_data)
    assert signal.confidence > 0
    assert signal.stop_loss < signal.entry < signal.targets[0]
```

### Integration Tests
```python
async def test_end_to_end_latency():
    start = time.time()
    
    # Simulate complete pipeline
    book = await get_mock_order_book()
    options = await get_mock_options()
    signal = process_all_data(book, options)
    
    latency = (time.time() - start) * 1000
    assert latency < 500  # Must be under 500ms
```

### Load Testing
```bash
# Using locust for load testing
locust -f load_test.py --users 1000 --spawn-rate 10
```

---

## Conclusion

This system provides a complete institutional-grade trading platform with minimal complexity by:
1. **Eliminating database overhead** in favor of memory-only processing
2. **Leveraging Alpha Vantage** for pre-calculated Greeks (no Black-Scholes needed)
3. **Using IBKR Level 2** for real microstructure analysis
4. **Executing actual trades** through IBKR TWS with professional risk management
5. **Managing positions automatically** with stops, trailing, and scaling
6. **Distributing real trades** to subscribers, not just signals
7. **Building in monetization** from day one with tiered access to actual performance

The result is a complete trading system that can be built in 4 weeks, maintained by a single developer, and generate significant revenue through tiered subscriptions while providing genuine trading edge through institutional indicators and real execution.

### Key Differentiator
This isn't just a signal service - it's a **complete automated trading system** that:
- Trades real money through IBKR
- Manages positions with professional risk controls
- Shows subscribers actual fills and P&L
- Provides institutional-grade analytics with sub-second execution

Subscribers aren't just getting signals - they're getting access to a live trading operation with full transparency.