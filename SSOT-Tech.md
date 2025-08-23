# SSOT-Tech.md - Technical Implementation v2.0
**Version:** 2.0 (Plugin Architecture)  
**Last Updated:** Current  
**Purpose:** Technical implementation specification using event-driven plugin architecture  
**Scope:** Complete technical architecture for 36 Alpha Vantage APIs + IBKR integration

---

## 1. Core Architecture Overview

### 1.1 Event-Driven Plugin Architecture

The entire system is built on three foundational components that NEVER change:

```
Message Bus (Central Nervous System)
    ↕️
Plugin Manager (Component Orchestrator)
    ↕️
Event Store (Source of Truth)
```

### 1.2 Key Principles

1. **No Direct Communication**: Components only communicate through events
2. **Plugin Everything**: Every feature is a plugin that can be added/removed
3. **Event Sourcing**: All state changes are events, enabling replay and audit
4. **Configuration Driven**: Behavior controlled by YAML, not code changes
5. **Progressive Enhancement**: Start simple, add complexity through new plugins

---

## 2. Foundation Layer (Week 1)

### 2.1 Message Bus Implementation

```python
# core/bus.py
class MessageBus:
    """
    Central event bus - ALL communication flows through here
    Handles both sync and async message passing
    """
    
    def __init__(self, persistence_backend):
        self.subscribers = defaultdict(list)
        self.persistence = persistence_backend
        self.message_queue = Queue()
        self.metrics = MetricsCollector()
        
    def publish(self, event_type: str, data: dict, correlation_id: str = None):
        """
        Publish event to all subscribers
        
        Event Types Follow Pattern: domain.entity.action
        Examples:
        - api.alphavatange.data_received
        - feature.calculation.completed
        - ml.prediction.generated
        - risk.vpin.calculated
        - signal.entry.triggered
        """
        
        message = Message(
            id=str(uuid4()),
            correlation_id=correlation_id or str(uuid4()),
            event_type=event_type,
            data=data,
            timestamp=datetime.utcnow(),
            metadata={
                'publisher': inspect.stack()[1].function,
                'version': '1.0'
            }
        )
        
        # Persist first (event sourcing)
        self.persistence.store(message)
        
        # Then distribute
        self._distribute(message)
        
    def subscribe(self, pattern: str, handler: Callable):
        """
        Subscribe to events matching pattern
        Supports wildcards: 'api.*', 'ml.prediction.*'
        """
        self.subscribers[pattern].append(handler)
        
    def _distribute(self, message: Message):
        """Distribute message to matching subscribers"""
        for pattern, handlers in self.subscribers.items():
            if self._matches_pattern(pattern, message.event_type):
                for handler in handlers:
                    try:
                        handler(message)
                    except Exception as e:
                        self._handle_error(e, handler, message)
```

### 2.2 Plugin Base Class

```python
# core/plugin.py
class Plugin(ABC):
    """Base class for ALL system components"""
    
    def __init__(self, bus: MessageBus, config: dict):
        self.bus = bus
        self.config = config
        self.name = self.__class__.__name__
        self.logger = self._setup_logger()
        self.metrics = MetricsCollector(self.name)
        self.state = PluginState.INITIALIZED
        
    @abstractmethod
    def start(self):
        """Initialize subscriptions and start processing"""
        pass
        
    @abstractmethod
    def stop(self):
        """Cleanup and shutdown"""
        pass
        
    @abstractmethod
    def health_check(self) -> dict:
        """Return health status"""
        pass
        
    def publish(self, event_type: str, data: dict):
        """Convenience method for publishing"""
        self.bus.publish(f"{self.name.lower()}.{event_type}", data)
```

### 2.3 Event Store Schema

```sql
-- Single source of truth for all events
CREATE TABLE events (
    id UUID PRIMARY KEY,
    correlation_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Indexes for queries
    INDEX idx_events_type (event_type),
    INDEX idx_events_correlation (correlation_id),
    INDEX idx_events_created (created_at DESC),
    INDEX idx_events_payload_symbol ((payload->>'symbol')),
    INDEX idx_events_type_time (event_type, created_at DESC)
);

-- Partitioned by month for performance
CREATE TABLE events_2025_01 PARTITION OF events
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

---

## 3. Data Source Plugins (Week 2)

### 3.1 Alpha Vantage Plugin (Handles ALL 36 APIs)

```python
# plugins/datasources/alpha_vantage.py
class AlphaVantagePlugin(Plugin):
    """
    Single plugin handles all 36 Alpha Vantage APIs
    Configuration drives which APIs are active
    """
    
    def __init__(self, bus: MessageBus, config: dict):
        super().__init__(bus, config)
        self.api_key = config['api_key']
        self.rate_limiter = TokenBucket(
            capacity=600,  # per minute
            refill_rate=10  # per second
        )
        self.session = aiohttp.ClientSession()
        
    def start(self):
        """Load API configurations and start schedulers"""
        for api_config in self.config['apis']:
            self._setup_api_scheduler(api_config)
            
    def _setup_api_scheduler(self, api_config: dict):
        """Setup scheduled fetching for one API"""
        schedule = api_config['schedule']  # cron expression
        
        scheduler.add_job(
            func=self._fetch_api,
            trigger=CronTrigger.from_crontab(schedule),
            args=[api_config],
            id=f"av_{api_config['name']}",
            replace_existing=True
        )
        
    async def _fetch_api(self, api_config: dict):
        """Generic fetcher for any Alpha Vantage API"""
        
        # Rate limit check
        await self.rate_limiter.acquire()
        
        # Build request from config
        params = self._build_params(api_config)
        
        # Make request
        async with self.session.get(self.base_url, params=params) as response:
            data = await response.json()
            
        # Transform based on API type
        transformed = self._transform_response(api_config['name'], data)
        
        # Publish to bus
        self.publish(f"data.{api_config['name']}", {
            'api': api_config['name'],
            'symbol': api_config.get('symbol'),
            'data': transformed,
            'raw': data,
            'timestamp': datetime.utcnow()
        })
```

### 3.2 IBKR Plugin

```python
# plugins/datasources/ibkr.py
class IBKRPlugin(Plugin):
    """Handles all IBKR data and execution"""
    
    def __init__(self, bus: MessageBus, config: dict):
        super().__init__(bus, config)
        self.client = IBClient(
            host=config['host'],
            port=config['port'],
            client_id=config['client_id']
        )
        self.subscriptions = {}
        
    def start(self):
        """Connect and start data streams"""
        self.client.connect()
        
        # Subscribe to 5-second bars
        for symbol in self.config['symbols']:
            self._subscribe_bars(symbol)
            
        # Subscribe to events for order execution
        self.bus.subscribe('signal.execute', self.execute_order)
        
    def _on_bar_update(self, bar: Bar):
        """Handle incoming 5-second bar"""
        
        # Publish raw bar
        self.publish('bar.5s', {
            'symbol': bar.symbol,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'timestamp': bar.timestamp
        })
        
        # Aggregate to other timeframes
        self._aggregate_bars(bar)
```

### 3.3 API Configuration Structure

```yaml
# config/apis/alpha_vantage.yaml
plugin: AlphaVantagePlugin
api_key: ${ALPHA_VANTAGE_API_KEY}
base_url: https://www.alphavantage.co/query

apis:
  # Technical Indicators (16)
  - name: rsi
    function: RSI
    schedule: "*/5 * * * *"  # Every 5 minutes
    tiers:
      A: [SPY, QQQ, IWM, SPX]
      B: [AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA]
    params:
      interval: 5min
      time_period: 14
      series_type: close
    transform:
      type: time_series
      value_key: "Technical Analysis: RSI"
      
  - name: macd
    function: MACD
    schedule: "*/5 * * * *"
    # ... similar structure for all 36 APIs
    
  # Options (2)
  - name: realtime_options
    function: REALTIME_OPTIONS
    schedule: "*/30 * * * * *"  # Every 30 seconds
    params:
      require_greeks: true
    transform:
      type: options_chain
      extract_greeks: true
      
  # Analytics (2)
  - name: analytics_fixed_window
    function: ANALYTICS_FIXED_WINDOW
    schedule: "0 */5 * * *"  # Every 5 minutes
    params:
      RANGE: 1month
      CALCULATIONS: ALL
```

---

## 4. Data Processing Plugins (Week 3-4)

### 4.1 Bar Aggregator Plugin

```python
# plugins/processing/bar_aggregator.py
class BarAggregatorPlugin(Plugin):
    """Aggregates 5-second bars to all timeframes"""
    
    def start(self):
        self.bus.subscribe('ibkr.bar.5s', self.aggregate)
        
        # Initialize aggregation buffers
        self.buffers = {
            '1m': defaultdict(list),   # 12 bars
            '5m': defaultdict(list),   # 60 bars
            '10m': defaultdict(list),  # 120 bars
            '15m': defaultdict(list),  # 180 bars
            '30m': defaultdict(list),  # 360 bars
            '1h': defaultdict(list),   # 720 bars
        }
        
    def aggregate(self, message: Message):
        """Aggregate 5s bar to higher timeframes"""
        bar = message.data
        symbol = bar['symbol']
        
        # Add to all buffers
        for timeframe, buffer in self.buffers.items():
            buffer[symbol].append(bar)
            
            # Check if we have enough bars
            required_bars = self._get_required_bars(timeframe)
            if len(buffer[symbol]) >= required_bars:
                
                # Create aggregated bar
                agg_bar = self._create_aggregated_bar(
                    buffer[symbol][:required_bars]
                )
                
                # Publish aggregated bar
                self.publish(f'bar.{timeframe}', agg_bar)
                
                # Clear used bars
                buffer[symbol] = buffer[symbol][required_bars:]
```

### 4.2 Data Validator Plugin

```python
# plugins/processing/validator.py
class DataValidatorPlugin(Plugin):
    """Validates all incoming data"""
    
    def start(self):
        # Subscribe to all data events
        self.bus.subscribe('*.data.*', self.validate)
        
    def validate(self, message: Message):
        """Validate data quality"""
        data = message.data
        
        # Check for required fields
        if not self._has_required_fields(data):
            self.publish('validation.failed', {
                'event_type': message.event_type,
                'reason': 'missing_fields',
                'data': data
            })
            return
            
        # Check data freshness
        if self._is_stale(data):
            self.publish('validation.warning', {
                'event_type': message.event_type,
                'reason': 'stale_data',
                'age_seconds': self._get_age(data)
            })
            
        # Check value ranges
        if not self._check_ranges(data):
            self.publish('validation.failed', {
                'event_type': message.event_type,
                'reason': 'out_of_range',
                'data': data
            })
```

---

## 5. Feature Engineering Plugin (Week 4-5)

```python
# plugins/ml/feature_engine.py
class FeatureEnginePlugin(Plugin):
    """Calculates 200+ features for ML models"""
    
    def start(self):
        # Subscribe to all data sources
        self.bus.subscribe('ibkr.bar.*', self.update_price_features)
        self.bus.subscribe('alphavatange.data.rsi', self.update_indicator_features)
        self.bus.subscribe('alphavatange.data.realtime_options', self.update_greek_features)
        
        # Feature storage
        self.features = defaultdict(lambda: defaultdict(dict))
        
    def calculate_features(self, symbol: str):
        """Calculate all features for a symbol"""
        
        features = {
            # Price features (50)
            'returns_1m': self._calc_returns(symbol, '1m'),
            'returns_5m': self._calc_returns(symbol, '5m'),
            'returns_15m': self._calc_returns(symbol, '15m'),
            'volatility_5m': self._calc_volatility(symbol, '5m'),
            'volatility_15m': self._calc_volatility(symbol, '15m'),
            'vwap_ratio': self._calc_vwap_ratio(symbol),
            'high_low_ratio': self._calc_high_low_ratio(symbol),
            'volume_ratio': self._calc_volume_ratio(symbol),
            
            # Technical indicators (40)
            'rsi': self.features[symbol]['indicators']['rsi'],
            'macd_signal': self.features[symbol]['indicators']['macd_signal'],
            'macd_histogram': self.features[symbol]['indicators']['macd_hist'],
            'bb_position': self._calc_bb_position(symbol),
            'atr_normalized': self._calc_atr_normalized(symbol),
            
            # Greeks features (20)
            'total_gamma': self._sum_gamma(symbol),
            'total_delta': self._sum_delta(symbol),
            'put_call_ratio': self._calc_pcr(symbol),
            'iv_skew': self._calc_iv_skew(symbol),
            
            # Microstructure (20)
            'bid_ask_spread': self._calc_spread(symbol),
            'order_imbalance': self._calc_imbalance(symbol),
            
            # Market regime (20)
            'trend_strength': self._calc_trend_strength(symbol),
            'regime_change_prob': self._calc_regime_prob(symbol),
            
            # Sentiment (20)
            'news_sentiment': self.features[symbol]['sentiment']['news'],
            'insider_score': self.features[symbol]['sentiment']['insider'],
            
            # Cross-sectional (30)
            'sector_momentum': self._calc_sector_momentum(symbol),
            'relative_strength': self._calc_relative_strength(symbol),
        }
        
        # Publish complete feature set
        self.publish('features.calculated', {
            'symbol': symbol,
            'features': features,
            'feature_count': len(features),
            'timestamp': datetime.utcnow()
        })
```

---

## 6. ML Model Plugins (Week 6-7)

### 6.1 Model Server Plugin

```python
# plugins/ml/model_server.py
class ModelServerPlugin(Plugin):
    """Serves ML predictions from trained models"""
    
    def start(self):
        self.bus.subscribe('features.calculated', self.predict)
        
        # Load models
        self.models = {
            'xgboost': self._load_xgboost(),
            'lstm': self._load_lstm(),
            'gru': self._load_gru()
        }
        
    def predict(self, message: Message):
        """Generate predictions from all models"""
        
        features = message.data['features']
        symbol = message.data['symbol']
        
        # Prepare feature vector
        X = self._prepare_features(features)
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            predictions[name] = pred
            confidences[name] = prob.max() if prob is not None else 0.5
            
        # Ensemble prediction
        ensemble_pred = self._ensemble(predictions, confidences)
        
        # Calculate SHAP values for explainability
        shap_values = self._calculate_shap(X)
        
        # Publish prediction
        self.publish('prediction.generated', {
            'symbol': symbol,
            'prediction': ensemble_pred,
            'confidence': np.mean(list(confidences.values())),
            'models': predictions,
            'shap_values': shap_values,
            'timestamp': datetime.utcnow()
        })
```

### 6.2 Model Training Plugin (Offline)

```python
# plugins/ml/model_trainer.py
class ModelTrainerPlugin(Plugin):
    """Trains models on historical data"""
    
    def start(self):
        # Schedule periodic retraining
        self.bus.subscribe('scheduler.retrain', self.train_models)
        
    def train_models(self, message: Message):
        """Walk-forward training process"""
        
        # Load historical features
        X, y = self._load_training_data()
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train XGBoost
            xgb_model = self._train_xgboost(X_train, y_train)
            xgb_score = xgb_model.score(X_val, y_val)
            
            # Train LSTM
            lstm_model = self._train_lstm(X_train, y_train)
            lstm_score = lstm_model.evaluate(X_val, y_val)
            
        # Publish training complete
        self.publish('training.complete', {
            'models': ['xgboost', 'lstm', 'gru'],
            'scores': {'xgboost': xgb_score, 'lstm': lstm_score},
            'timestamp': datetime.utcnow()
        })
```

---

## 7. Analytics Plugins (Week 8)

### 7.1 VPIN Calculator Plugin

```python
# plugins/analytics/vpin.py
class VPINPlugin(Plugin):
    """Volume-Synchronized Probability of Informed Trading"""
    
    def start(self):
        self.bus.subscribe('ibkr.bar.5s', self.update_vpin)
        
        # VPIN parameters from config
        self.bucket_size = self.config['bucket_size']  # Volume bucket
        self.window = self.config['window']  # Number of buckets
        
        # Storage
        self.volume_buckets = defaultdict(list)
        
    def update_vpin(self, message: Message):
        """Update VPIN calculation with new bar"""
        
        bar = message.data
        symbol = bar['symbol']
        
        # Classify volume (BVC algorithm)
        buy_volume, sell_volume = self._classify_volume(bar)
        
        # Update current bucket
        current_bucket = self.volume_buckets[symbol][-1] if self.volume_buckets[symbol] else {'buy': 0, 'sell': 0, 'total': 0}
        
        current_bucket['buy'] += buy_volume
        current_bucket['sell'] += sell_volume
        current_bucket['total'] += bar['volume']
        
        # Check if bucket is full
        if current_bucket['total'] >= self.bucket_size:
            # Start new bucket with overflow
            overflow = current_bucket['total'] - self.bucket_size
            
            # Complete current bucket
            current_bucket['total'] = self.bucket_size
            self.volume_buckets[symbol].append(current_bucket)
            
            # Start new bucket with overflow
            if overflow > 0:
                self.volume_buckets[symbol].append({
                    'buy': buy_volume * (overflow / bar['volume']),
                    'sell': sell_volume * (overflow / bar['volume']),
                    'total': overflow
                })
                
        # Calculate VPIN if we have enough buckets
        if len(self.volume_buckets[symbol]) >= self.window:
            vpin = self._calculate_vpin(self.volume_buckets[symbol][-self.window:])
            
            # Publish VPIN update
            self.publish('vpin.calculated', {
                'symbol': symbol,
                'vpin': vpin,
                'toxic': vpin > 0.6,
                'critical': vpin > 0.7,
                'timestamp': datetime.utcnow()
            })
```

### 7.2 GEX Calculator Plugin

```python
# plugins/analytics/gex.py
class GEXPlugin(Plugin):
    """Gamma Exposure calculation"""
    
    def start(self):
        self.bus.subscribe('alphavatange.data.realtime_options', self.calculate_gex)
        
    def calculate_gex(self, message: Message):
        """Calculate total gamma exposure"""
        
        options_chain = message.data['data']
        symbol = message.data['symbol']
        
        # Calculate GEX for each strike
        gex_by_strike = {}
        
        for option in options_chain:
            strike = option['strike']
            gamma = option['gamma']
            oi = option['open_interest']
            
            # GEX = Gamma * OI * 100 (contract multiplier)
            gex = gamma * oi * 100
            
            # Adjust sign for puts (negative gamma)
            if option['type'] == 'put':
                gex = -gex
                
            gex_by_strike[strike] = gex_by_strike.get(strike, 0) + gex
            
        # Find key levels
        total_gex = sum(gex_by_strike.values())
        max_gex_strike = max(gex_by_strike, key=gex_by_strike.get)
        
        # Identify support/resistance
        support_levels = [k for k, v in gex_by_strike.items() if v > 0]
        resistance_levels = [k for k, v in gex_by_strike.items() if v < 0]
        
        # Publish GEX data
        self.publish('gex.calculated', {
            'symbol': symbol,
            'total_gex': total_gex,
            'gex_by_strike': gex_by_strike,
            'max_gex_strike': max_gex_strike,
            'support_levels': sorted(support_levels),
            'resistance_levels': sorted(resistance_levels),
            'timestamp': datetime.utcnow()
        })
```

### 7.3 Microstructure Analytics Plugin

```python
# plugins/analytics/microstructure.py
class MicrostructurePlugin(Plugin):
    """Market microstructure analytics"""
    
    def start(self):
        self.bus.subscribe('ibkr.tick.*', self.update_microstructure)
        
    def calculate_metrics(self, symbol: str):
        """Calculate microstructure metrics"""
        
        # Kyle's Lambda (price impact)
        lambda_kyle = self._calculate_kyle_lambda(symbol)
        
        # Amihud Illiquidity
        illiquidity = self._calculate_amihud(symbol)
        
        # Roll's Spread
        roll_spread = self._calculate_roll_spread(symbol)
        
        # Publish metrics
        self.publish('microstructure.calculated', {
            'symbol': symbol,
            'kyle_lambda': lambda_kyle,
            'amihud_illiquidity': illiquidity,
            'roll_spread': roll_spread,
            'timestamp': datetime.utcnow()
        })
```

---

## 8. Strategy Plugins (Week 9-10)

### 8.1 Base Strategy Plugin

```python
# plugins/strategies/base_strategy.py
class StrategyPlugin(Plugin):
    """Base class for all trading strategies"""
    
    def __init__(self, bus: MessageBus, config: dict):
        super().__init__(bus, config)
        self.positions = {}
        self.pending_signals = []
        
    def start(self):
        # Subscribe to all relevant data
        self.bus.subscribe('ml.prediction.generated', self.on_prediction)
        self.bus.subscribe('analytics.vpin.calculated', self.on_vpin)
        self.bus.subscribe('analytics.gex.calculated', self.on_gex)
        self.bus.subscribe('features.calculated', self.on_features)
        
        # Subscribe to execution confirmations
        self.bus.subscribe('execution.fill', self.on_fill)
        
    @abstractmethod
    def generate_signal(self) -> Signal:
        """Generate trading signal"""
        pass
        
    def execute_signal(self, signal: Signal):
        """Send signal for execution"""
        self.publish('signal.generated', {
            'strategy': self.name,
            'symbol': signal.symbol,
            'action': signal.action,
            'size': signal.size,
            'confidence': signal.confidence,
            'metadata': signal.metadata,
            'timestamp': datetime.utcnow()
        })
```

### 8.2 ML-Driven Strategy Plugin

```python
# plugins/strategies/ml_strategy.py
class MLStrategyPlugin(StrategyPlugin):
    """ML-driven options trading strategy"""
    
    def on_prediction(self, message: Message):
        """React to ML predictions"""
        
        data = message.data
        symbol = data['symbol']
        prediction = data['prediction']
        confidence = data['confidence']
        
        # Store latest prediction
        self.latest_predictions[symbol] = {
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': data['timestamp']
        }
        
        # Check if we should trade
        if self._should_trade(symbol):
            signal = self.generate_signal(symbol)
            if signal:
                self.execute_signal(signal)
                
    def generate_signal(self, symbol: str) -> Signal:
        """Generate signal based on all available data"""
        
        # Get latest data
        prediction = self.latest_predictions.get(symbol)
        vpin = self.latest_vpin.get(symbol)
        gex = self.latest_gex.get(symbol)
        features = self.latest_features.get(symbol)
        
        # Check confidence threshold
        if prediction['confidence'] < self.config['min_confidence']:
            return None
            
        # Check VPIN toxicity
        if vpin and vpin['toxic']:
            self.logger.warning(f"VPIN toxic for {symbol}: {vpin['vpin']}")
            return None
            
        # Check GEX levels
        current_price = self._get_current_price(symbol)
        if gex:
            nearest_strike = self._find_nearest_strike(current_price, gex['gex_by_strike'])
            if abs(current_price - nearest_strike) < 0.5:
                # Too close to max GEX strike (pinning risk)
                return None
                
        # Determine position size
        size = self._calculate_position_size(
            symbol,
            prediction['confidence'],
            features
        )
        
        # Create signal
        return Signal(
            symbol=symbol,
            action='BUY' if prediction['prediction'] > 0 else 'SELL',
            size=size,
            confidence=prediction['confidence'],
            metadata={
                'strategy': 'ml_strategy',
                'prediction': prediction,
                'vpin': vpin,
                'gex': gex,
                'features': features
            }
        )
```

### 8.3 0DTE Strategy Plugin

```python
# plugins/strategies/dte0_strategy.py
class DTE0StrategyPlugin(StrategyPlugin):
    """Zero Days to Expiration strategy"""
    
    def start(self):
        super().start()
        
        # 0DTE specific subscriptions
        self.bus.subscribe('alphavatange.data.realtime_options', self.on_options_update)
        
        # Trading windows
        self.entry_window = (time(9, 45), time(14, 0))
        self.exit_time = time(15, 30)
        
    def on_options_update(self, message: Message):
        """Process options chain updates"""
        
        chain = message.data['data']
        symbol = message.data['symbol']
        
        # Filter for 0DTE options
        today = datetime.now().date()
        dte0_options = [
            opt for opt in chain 
            if opt['expiration'].date() == today
        ]
        
        if not dte0_options:
            return
            
        # Find optimal strikes
        optimal_call = self._find_optimal_strike(dte0_options, 'call')
        optimal_put = self._find_optimal_strike(dte0_options, 'put')
        
        # Check entry criteria
        if self._check_entry_criteria(optimal_call, optimal_put):
            signal = Signal(
                symbol=symbol,
                action='BUY',
                instrument=optimal_call['contract_symbol'],
                size=self._calculate_0dte_size(optimal_call),
                confidence=self._calculate_0dte_confidence(optimal_call),
                metadata={
                    'strategy': '0dte',
                    'option': optimal_call,
                    'theta_decay': optimal_call['theta'],
                    'gamma_risk': optimal_call['gamma']
                }
            )
            self.execute_signal(signal)
```

---

## 9. Risk Management Plugin (Week 10)

```python
# plugins/risk/risk_manager.py
class RiskManagerPlugin(Plugin):
    """Central risk management"""
    
    def start(self):
        # Subscribe to all signals before execution
        self.bus.subscribe('signal.generated', self.validate_signal)
        
        # Subscribe to position updates
        self.bus.subscribe('execution.fill', self.update_positions)
        
        # Subscribe to risk metrics
        self.bus.subscribe('analytics.vpin.calculated', self.on_vpin)
        self.bus.subscribe('features.calculated', self.on_features)
        
        # Risk limits from config
        self.limits = self.config['limits']
        
    def validate_signal(self, message: Message):
        """Validate signal against risk limits"""
        
        signal = message.data
        
        # Check position limits
        if not self._check_position_limits(signal):
            self.publish('risk.rejected', {
                'signal': signal,
                'reason': 'position_limit_exceeded'
            })
            return
            
        # Check portfolio Greeks
        if not self._check_portfolio_greeks(signal):
            self.publish('risk.rejected', {
                'signal': signal,
                'reason': 'greeks_limit_exceeded'
            })
            return
            
        # Check daily loss limit
        if self._daily_pnl < -self.limits['max_daily_loss']:
            self.publish('risk.rejected', {
                'signal': signal,
                'reason': 'daily_loss_limit'
            })
            return
            
        # Calculate position size with Kelly Criterion
        sized_signal = self._apply_position_sizing(signal)
        
        # Signal approved - forward for execution
        self.publish('risk.approved', sized_signal)
```

---

## 10. Execution Plugin (Week 11)

```python
# plugins/execution/executor.py
class ExecutorPlugin(Plugin):
    """Handles order execution through IBKR"""
    
    def start(self):
        # Subscribe to approved signals
        self.bus.subscribe('risk.approved', self.execute_order)
        
        # Connect to IBKR
        self.ibkr = IBKRClient(self.config['ibkr'])
        
    def execute_order(self, message: Message):
        """Execute order through IBKR"""
        
        signal = message.data
        
        # Create order
        order = self._create_order(signal)
        
        # Submit to IBKR
        order_id = self.ibkr.place_order(order)
        
        # Track order
        self.pending_orders[order_id] = {
            'signal': signal,
            'order': order,
            'status': 'PENDING',
            'submitted_at': datetime.utcnow()
        }
        
        # Publish order submitted
        self.publish('order.submitted', {
            'order_id': order_id,
            'signal': signal,
            'order': order
        })
        
    def on_order_fill(self, fill_data):
        """Handle order fill from IBKR"""
        
        # Publish fill
        self.publish('execution.fill', {
            'order_id': fill_data.order_id,
            'symbol': fill_data.symbol,
            'size': fill_data.size,
            'price': fill_data.price,
            'commission': fill_data.commission,
            'timestamp': fill_data.timestamp
        })
```

---

## 11. Monitoring & Publishing Plugins (Week 12)

### 11.1 Discord Publisher Plugin

```python
# plugins/publishing/discord.py
class DiscordPublisherPlugin(Plugin):
    """Publishes to Discord channels"""
    
    def start(self):
        # Subscribe to events to publish
        self.bus.subscribe('signal.generated', self.publish_signal)
        self.bus.subscribe('execution.fill', self.publish_fill)
        self.bus.subscribe('risk.alert', self.publish_alert)
        
        # Discord webhook URLs from config
        self.webhooks = {
            'signals': self.config['webhooks']['signals'],
            'fills': self.config['webhooks']['fills'],
            'alerts': self.config['webhooks']['alerts']
        }
        
    def publish_signal(self, message: Message):
        """Publish signal to Discord"""
        
        signal = message.data
        
        embed = {
            'title': f"📊 Signal: {signal['action']} {signal['symbol']}",
            'color': 0x00ff00 if signal['action'] == 'BUY' else 0xff0000,
            'fields': [
                {'name': 'Strategy', 'value': signal['strategy']},
                {'name': 'Confidence', 'value': f"{signal['confidence']:.2%}"},
                {'name': 'Size', 'value': signal['size']},
            ],
            'timestamp': signal['timestamp'].isoformat()
        }
        
        self._send_webhook(self.webhooks['signals'], {'embeds': [embed]})
```

### 11.2 Performance Monitor Plugin

```python
# plugins/monitoring/performance.py
class PerformanceMonitorPlugin(Plugin):
    """Monitors system and trading performance"""
    
    def start(self):
        # Subscribe to all execution events
        self.bus.subscribe('execution.fill', self.track_trade)
        
        # Schedule periodic reports
        scheduler.add_job(
            self.generate_report,
            'cron',
            hour=16,
            minute=30
        )
        
    def generate_report(self):
        """Generate daily performance report"""
        
        report = {
            'date': datetime.now().date(),
            'trades': self._get_daily_trades(),
            'pnl': self._calculate_daily_pnl(),
            'win_rate': self._calculate_win_rate(),
            'sharpe': self._calculate_sharpe(),
            'max_drawdown': self._calculate_drawdown(),
            'var_95': self._calculate_var(0.95),
            'top_winners': self._get_top_trades(5, 'winners'),
            'top_losers': self._get_top_trades(5, 'losers')
        }
        
        # Publish report
        self.publish('report.daily', report)
```

---

## 12. Configuration Structure

### 12.1 Master Configuration

```yaml
# config/system.yaml
system:
  environment: ${ENVIRONMENT}
  
  message_bus:
    backend: postgresql
    async: true
    persistence:
      connection_string: ${DATABASE_URL}
      
  plugin_manager:
    auto_discover: true
    plugin_dirs:
      - plugins/datasources
      - plugins/processing
      - plugins/ml
      - plugins/analytics
      - plugins/strategies
      - plugins/risk
      - plugins/execution
      - plugins/monitoring
      - plugins/publishing
      
  scheduler:
    timezone: US/Eastern
    job_store: postgresql
```

### 12.2 Plugin Configurations

```yaml
# config/plugins/ml.yaml
feature_engine:
  enabled: true
  calculate_interval: 10  # seconds
  feature_window: 500  # bars to use
  
model_server:
  enabled: true
  models:
    xgboost:
      path: models/xgboost_latest.pkl
      features: config/features/xgboost.yaml
    lstm:
      path: models/lstm_latest.h5
      sequence_length: 100
    gru:
      path: models/gru_latest.h5
      sequence_length: 100
  ensemble:
    method: weighted_average
    weights:
      xgboost: 0.4
      lstm: 0.3
      gru: 0.3
```

---

## 13. Database Views for Querying

```sql
-- Current positions view
CREATE MATERIALIZED VIEW positions_current AS
SELECT 
    (e.payload->>'symbol') as symbol,
    (e.payload->>'action') as action,
    (e.payload->>'size')::int as size,
    (e.payload->>'price')::decimal as entry_price,
    e.created_at as entry_time
FROM events e
WHERE e.event_type = 'execution.fill'
AND NOT EXISTS (
    SELECT 1 FROM events e2 
    WHERE e2.event_type = 'position.closed'
    AND e2.payload->>'symbol' = e.payload->>'symbol'
    AND e2.created_at > e.created_at
);

-- Latest predictions view
CREATE MATERIALIZED VIEW predictions_latest AS
SELECT DISTINCT ON (symbol)
    (payload->>'symbol') as symbol,
    (payload->>'prediction')::decimal as prediction,
    (payload->>'confidence')::decimal as confidence,
    payload->'models' as model_predictions,
    created_at
FROM events
WHERE event_type = 'ml.prediction.generated'
ORDER BY (payload->>'symbol'), created_at DESC;

-- Risk metrics view
CREATE MATERIALIZED VIEW risk_metrics AS
SELECT 
    (payload->>'symbol') as symbol,
    (payload->>'vpin')::decimal as vpin,
    (payload->>'total_gex')::decimal as gex,
    (payload->'microstructure'->>'kyle_lambda')::decimal as kyle_lambda,
    created_at
FROM events
WHERE event_type IN ('analytics.vpin.calculated', 'analytics.gex.calculated')
ORDER BY created_at DESC;
```

---

## 14. Testing Strategy

### 14.1 Plugin Testing

```python
# tests/test_plugin.py
class TestPlugin:
    """Test individual plugin in isolation"""
    
    def test_plugin_isolation(self):
        # Create mock bus
        bus = MockMessageBus()
        
        # Create plugin
        plugin = YourPlugin(bus, config)
        
        # Start plugin
        plugin.start()
        
        # Send test message
        bus.publish('test.event', {'data': 'test'})
        
        # Assert plugin behavior
        assert bus.published_events['your_plugin.response']
```

### 14.2 Integration Testing

```python
# tests/test_integration.py
class TestIntegration:
    """Test multiple plugins together"""
    
    def test_data_flow(self):
        # Start real message bus
        bus = MessageBus(PostgresBackend())
        
        # Start plugins
        av_plugin = AlphaVantagePlugin(bus, av_config)
        feature_plugin = FeatureEnginePlugin(bus, feature_config)
        ml_plugin = ModelServerPlugin(bus, ml_config)
        
        # Start all
        av_plugin.start()
        feature_plugin.start()
        ml_plugin.start()
        
        # Simulate data
        av_plugin.publish('data.rsi', test_rsi_data)
        
        # Wait for pipeline
        time.sleep(1)
        
        # Check prediction was generated
        predictions = bus.get_events('ml.prediction.generated')
        assert len(predictions) > 0
```

---

## 15. Deployment Architecture

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: alphatrader
      POSTGRES_USER: alphatrader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    
  message_bus:
    build: .
    command: python -m core.bus
    depends_on:
      - postgres
      - redis
      
  plugin_manager:
    build: .
    command: python -m core.plugin_manager
    depends_on:
      - message_bus
    volumes:
      - ./plugins:/app/plugins
      - ./config:/app/config
      
  scheduler:
    build: .
    command: python -m core.scheduler
    depends_on:
      - plugin_manager
```

---

## END OF TECHNICAL SPECIFICATION

This architecture ensures:
1. **No Rebuilding**: Plugins added without touching existing code
2. **Clear Communication**: All through events
3. **Progressive Enhancement**: Start simple, add complexity
4. **Testability**: Each component in isolation
5. **Scalability**: Can distribute plugins across machines