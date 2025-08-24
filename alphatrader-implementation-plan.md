# AlphaTrader Implementation Plan
## 16-Week Progressive Build with Code Reuse

---

## OVERVIEW: Build Once, Extend Forever

This plan shows EXACTLY how each component is built once and reused throughout the project. Every week builds on previous work - no rewrites, no wasted effort.

```
Week 1-2:   Data Foundation (Reused by everything)
Week 3-4:   ML & Trading Logic (Core intelligence)
Week 5-6:   Paper Trading (Reuses all above)
Week 7-8:   Community Platform (Publishes paper trades)
Week 9-10:  Production Prep (Small real money)
Week 11-12: Full Production (Scale up)
Week 13-14: Optimization (Performance tuning)
Week 15-16: Advanced Features (Spreads, more symbols)
```

---

## PHASE 1: FOUNDATION (Weeks 1-4)
*Build the reusable core that everything depends on*

### Week 1: Data Layer Foundation

#### Day 1-2: Project Setup & IBKR Connection

**Create**: `src/core/config.py`
```python
from dataclasses import dataclass
from typing import List
import yaml
import os

@dataclass
class TradingConfig:
    """Single source of truth for all configuration"""
    # Trading settings
    mode: str  # 'paper' or 'live'
    symbols: List[str]
    
    # Risk settings
    max_positions: int = 5
    max_position_size: float = 10000
    daily_loss_limit: float = 1000
    
    # IBKR settings
    ibkr_host: str = '127.0.0.1'
    ibkr_port: int = 7497  # Paper port
    
    @classmethod
    def load(cls, path='config.yaml'):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

# THIS CONFIG IS REUSED BY EVERY COMPONENT
config = TradingConfig.load()
```

**Create**: `src/data/market_data.py`
```python
from ib_insync import IB, Stock, Option, MarketOrder
import asyncio
from typing import Dict, Optional
import pandas as pd

class MarketDataManager:
    """
    Core market data component - NEVER REWRITTEN
    Used by: ML, Trading, Risk, Paper, Live, Community
    """
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ib = IB()
        self.connected = False
        self.subscriptions = {}
        self.latest_prices = {}  # Cache for instant access
        self.bars_5sec = {}  # 5-second bars
        
    async def connect(self):
        """Connect to IBKR - reused for paper and live"""
        port = 7497 if self.config.mode == 'paper' else 7496
        await self.ib.connectAsync(
            self.config.ibkr_host, 
            port, 
            clientId=1
        )
        self.connected = True
        print(f"Connected to IBKR ({self.config.mode} mode)")
        
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to market data - reused forever"""
        for symbol in symbols:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Subscribe to 5-second bars
            bars = self.ib.reqRealTimeBars(
                contract, 5, 'TRADES', False
            )
            self.subscriptions[symbol] = bars
            
            # Set up callback for updates
            bars.updateEvent += lambda bars, symbol=symbol: 
                self._on_bar_update(symbol, bars)
                
    def _on_bar_update(self, symbol: str, bars):
        """Handle bar updates - feeds everything"""
        if bars:
            latest = bars[-1]
            self.latest_prices[symbol] = latest.close
            self.bars_5sec[symbol] = latest
            # This data feeds ML, risk, execution - EVERYTHING
            
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price - instant from cache"""
        return self.latest_prices.get(symbol, 0.0)
        
    async def get_historical_bars(self, symbol: str, days: int = 5):
        """Get historical data for ML training"""
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f'{days} D',
            barSizeSetting='5 secs',
            whatToShow='TRADES',
            useRTH=True
        )
        return pd.DataFrame(bars)

# CREATE ONCE, USE FOREVER
market_data = MarketDataManager(config)
```

#### Day 3-4: Options Data Management

**Create**: `src/data/options_data.py`
```python
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class OptionsDataManager:
    """
    Options data and Greeks - REUSED BY ALL TRADING
    Built on top of MarketDataManager
    """
    def __init__(self, market_data_manager: MarketDataManager):
        self.market = market_data_manager  # Reuse market data!
        self.chains = {}
        self.greeks_cache = {}
        self.risk_free_rate = 0.05
        
    async def fetch_option_chain(self, symbol: str) -> Dict:
        """Fetch option chain - used by ML and trading"""
        stock = Stock(symbol, 'SMART', 'USD')
        self.market.ib.qualifyContracts(stock)
        
        # Get next 4 weekly expirations
        chains = []
        for weeks in range(4):
            expiry = datetime.now() + timedelta(weeks=weeks, days=(4-datetime.now().weekday()))
            chain = await self.market.ib.reqSecDefOptParamsAsync(
                stock.symbol, '', stock.secType, stock.conId
            )
            chains.extend(chain)
            
        self.chains[symbol] = self._process_chain(chains)
        return self.chains[symbol]
        
    def calculate_greeks(self, spot: float, strike: float, 
                        time_to_expiry: float, volatility: float,
                        option_type: str) -> Dict[str, float]:
        """
        Black-Scholes Greeks - CRITICAL CALCULATION
        Reused by: Risk, ML features, position management
        """
        cache_key = f"{spot}_{strike}_{time_to_expiry}_{volatility}_{option_type}"
        
        # Check cache first
        if cache_key in self.greeks_cache:
            return self.greeks_cache[cache_key]
            
        # Calculate Greeks
        d1 = (np.log(spot/strike) + (self.risk_free_rate + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
        d2 = d1 - volatility*np.sqrt(time_to_expiry)
        
        if option_type == 'CALL':
            delta = norm.cdf(d1)
            theta = (-spot*norm.pdf(d1)*volatility/(2*np.sqrt(time_to_expiry)) - self.risk_free_rate*strike*np.exp(-self.risk_free_rate*time_to_expiry)*norm.cdf(d2))/365
        else:  # PUT
            delta = norm.cdf(d1) - 1
            theta = (-spot*norm.pdf(d1)*volatility/(2*np.sqrt(time_to_expiry)) + self.risk_free_rate*strike*np.exp(-self.risk_free_rate*time_to_expiry)*norm.cdf(-d2))/365
            
        gamma = norm.pdf(d1)/(spot*volatility*np.sqrt(time_to_expiry))
        vega = spot*norm.pdf(d1)*np.sqrt(time_to_expiry)/100
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
        
        # Cache for reuse
        self.greeks_cache[cache_key] = greeks
        return greeks
        
    def find_atm_options(self, symbol: str, dte_min: int = 0, dte_max: int = 7):
        """Find ATM options for trading - reused by signal generator"""
        spot = self.market.get_latest_price(symbol)
        chain = self.chains.get(symbol, {})
        
        atm_options = []
        for expiry, strikes in chain.items():
            dte = (expiry - datetime.now().date()).days
            if dte_min <= dte <= dte_max:
                # Find closest strike to spot
                closest_strike = min(strikes.keys(), 
                                   key=lambda k: abs(k - spot))
                atm_options.append({
                    'strike': closest_strike,
                    'expiry': expiry,
                    'dte': dte,
                    'call': strikes[closest_strike]['call'],
                    'put': strikes[closest_strike]['put']
                })
                
        return atm_options

# BUILD ON TOP OF MARKET DATA
options_data = OptionsDataManager(market_data)
```

#### Day 5: Database Setup

**Create**: `src/data/database.py`
```python
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import redis
import json

class DatabaseManager:
    """
    Database layer - REUSED BY ALL COMPONENTS
    Paper and live trading use same schema
    """
    def __init__(self, config: TradingConfig):
        # PostgreSQL connection pool
        self.pg_pool = ThreadedConnectionPool(
            1, 20,
            host='localhost',
            database='alphatrader',
            user='postgres',
            password='your_password'
        )
        
        # Redis for caching
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Create tables if not exist
        self._init_tables()
        
    def _init_tables(self):
        """Create tables - same for paper and live"""
        with self.get_db() as conn:
            cur = conn.cursor()
            
            # Trades table - stores everything
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    mode VARCHAR(10),  -- 'paper' or 'live'
                    symbol VARCHAR(10),
                    option_type VARCHAR(4),
                    strike DECIMAL,
                    expiry DATE,
                    action VARCHAR(10),
                    quantity INT,
                    price DECIMAL,
                    commission DECIMAL DEFAULT 0.65,
                    pnl DECIMAL
                )
            """)
            
            # Signals table - tracks all signals
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    symbol VARCHAR(10),
                    signal_type VARCHAR(20),
                    confidence DECIMAL,
                    features JSONB,  -- Store feature vector
                    executed BOOLEAN DEFAULT FALSE,
                    trade_id INT REFERENCES trades(id)
                )
            """)
            
            conn.commit()
            
    @contextmanager
    def get_db(self):
        """Get database connection - reused everywhere"""
        conn = self.pg_pool.getconn()
        try:
            yield conn
        finally:
            self.pg_pool.putconn(conn)
            
    def cache_set(self, key: str, value: any, ttl: int = 60):
        """Redis cache - reused for all caching"""
        self.redis.setex(key, ttl, json.dumps(value))
        
    def cache_get(self, key: str):
        """Get from cache - reused everywhere"""
        value = self.redis.get(key)
        return json.loads(value) if value else None

# ONE DATABASE MANAGER FOR EVERYTHING
db = DatabaseManager(config)
```

### Week 2: ML and Analytics Foundation

#### Day 1-2: Feature Engineering

**Create**: `src/analytics/features.py`
```python
import pandas as pd
import numpy as np
from typing import Dict, List
import talib

class FeatureEngine:
    """
    Feature engineering - FEEDS THE ML MODEL
    Reused for training and live prediction
    """
    def __init__(self, options_manager: OptionsDataManager):
        self.options = options_manager
        
        # Define feature names for consistency
        self.feature_names = [
            # Price action (5 features)
            'returns_5m', 'returns_30m', 'returns_1h',
            'volume_ratio', 'high_low_ratio',
            
            # Technical indicators (10 features)
            'rsi', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_position',
            'atr', 'adx', 'obv_slope', 'vwap_distance',
            
            # Options metrics (8 features)
            'iv_rank', 'iv_percentile', 'put_call_ratio',
            'gamma_exposure', 'delta_neutral_price',
            'max_pain_distance', 'call_volume', 'put_volume',
            
            # Market structure (5 features)
            'spy_correlation', 'qqq_correlation',
            'vix_level', 'term_structure', 'market_regime'
        ]
        
    def calculate_features(self, symbol: str, 
                          historical_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate all features - CRITICAL FOR ML
        Same features for training and live
        """
        features = {}
        
        # Price action features
        features['returns_5m'] = self._calculate_returns(historical_data, 60)  # 5min = 60 bars
        features['returns_30m'] = self._calculate_returns(historical_data, 360)
        features['returns_1h'] = self._calculate_returns(historical_data, 720)
        features['volume_ratio'] = self._calculate_volume_ratio(historical_data)
        features['high_low_ratio'] = (historical_data['high'].iloc[-1] - historical_data['low'].iloc[-1]) / historical_data['close'].iloc[-1]
        
        # Technical indicators using TA-Lib
        close = historical_data['close'].values
        high = historical_data['high'].values
        low = historical_data['low'].values
        volume = historical_data['volume'].values
        
        features['rsi'] = talib.RSI(close, timeperiod=14)[-1] / 100.0
        macd, signal, hist = talib.MACD(close)
        features['macd_signal'] = signal[-1] if len(signal) > 0 else 0
        features['macd_histogram'] = hist[-1] if len(hist) > 0 else 0
        
        upper, middle, lower = talib.BBANDS(close)
        features['bb_upper'] = upper[-1] if len(upper) > 0 else close[-1]
        features['bb_lower'] = lower[-1] if len(lower) > 0 else close[-1]
        features['bb_position'] = (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
        
        features['atr'] = talib.ATR(high, low, close)[-1]
        features['adx'] = talib.ADX(high, low, close)[-1] / 100.0
        features['obv_slope'] = self._calculate_obv_slope(close, volume)
        features['vwap_distance'] = self._calculate_vwap_distance(historical_data)
        
        # Options features
        options_features = self._calculate_options_features(symbol)
        features.update(options_features)
        
        # Market structure
        features['spy_correlation'] = 0.8  # Simplified for now
        features['qqq_correlation'] = 0.7
        features['vix_level'] = 20.0 / 100.0  # Normalized
        features['term_structure'] = 0.0
        features['market_regime'] = 0.5  # Neutral
        
        # Convert to array in consistent order
        feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Handle any NaN values
        feature_array = np.nan_to_num(feature_array, 0.0)
        
        return feature_array
        
    def _calculate_options_features(self, symbol: str) -> Dict:
        """Calculate options-specific features"""
        features = {}
        
        # Get ATM options
        atm_options = self.options.find_atm_options(symbol, dte_min=0, dte_max=7)
        
        if atm_options:
            option = atm_options[0]  # Use nearest expiry
            spot = self.options.market.get_latest_price(symbol)
            
            # Calculate Greeks for ATM
            call_greeks = self.options.calculate_greeks(
                spot, option['strike'], option['dte']/365.0, 0.20, 'CALL'
            )
            put_greeks = self.options.calculate_greeks(
                spot, option['strike'], option['dte']/365.0, 0.20, 'PUT'
            )
            
            features['iv_rank'] = 0.5  # Placeholder
            features['iv_percentile'] = 0.5
            features['put_call_ratio'] = 1.0
            features['gamma_exposure'] = call_greeks['gamma'] - put_greeks['gamma']
            features['delta_neutral_price'] = option['strike']
            features['max_pain_distance'] = (spot - option['strike']) / spot
            features['call_volume'] = 1000  # Placeholder
            features['put_volume'] = 1000
        else:
            # Default values if no options available
            features.update({
                'iv_rank': 0.5, 'iv_percentile': 0.5,
                'put_call_ratio': 1.0, 'gamma_exposure': 0.0,
                'delta_neutral_price': 0.0, 'max_pain_distance': 0.0,
                'call_volume': 0, 'put_volume': 0
            })
            
        return features

# BUILD ON OPTIONS DATA
feature_engine = FeatureEngine(options_data)
```

#### Day 3-4: ML Model

**Create**: `src/analytics/ml_model.py`
```python
import xgboost as xgb
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pandas as pd

class MLPredictor:
    """
    ML model for predictions - CORE INTELLIGENCE
    Reused for paper and live trading
    """
    def __init__(self, feature_engine: FeatureEngine):
        self.features = feature_engine
        self.model = None
        self.scaler = StandardScaler()
        self.confidence_threshold = 0.6
        
        # Try to load existing model
        self.load_model()
        
    def load_model(self, path: str = 'models/xgboost_v1.pkl'):
        """Load trained model or create default"""
        try:
            self.model = joblib.load(path)
            self.scaler = joblib.load(path.replace('.pkl', '_scaler.pkl'))
            print("Loaded existing model")
        except:
            print("Creating new model with defaults")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softprob',
                n_classes=4  # BUY_CALL, BUY_PUT, HOLD, CLOSE
            )
            
    def train(self, training_data: pd.DataFrame):
        """Train model on historical data"""
        # Extract features and labels
        X = []
        y = []
        
        for symbol in ['SPY', 'QQQ', 'IWM']:
            symbol_data = training_data[training_data['symbol'] == symbol]
            
            for i in range(100, len(symbol_data)):
                # Calculate features
                historical = symbol_data.iloc[i-100:i]
                features = self.features.calculate_features(symbol, historical)
                X.append(features)
                
                # Generate label based on future price movement
                future_return = (symbol_data.iloc[i+12]['close'] - symbol_data.iloc[i]['close']) / symbol_data.iloc[i]['close']
                
                if future_return > 0.002:  # 0.2% up
                    y.append(0)  # BUY_CALL
                elif future_return < -0.002:  # 0.2% down
                    y.append(1)  # BUY_PUT
                else:
                    y.append(2)  # HOLD
                    
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X, y)
        
        # Save model
        joblib.dump(self.model, 'models/xgboost_v1.pkl')
        joblib.dump(self.scaler, 'models/xgboost_v1_scaler.pkl')
        
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Make prediction - USED BY SIGNAL GENERATOR
        Returns: (signal, confidence)
        """
        if self.model is None:
            return 'HOLD', 0.0
            
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction probabilities
        probs = self.model.predict_proba(features_scaled)[0]
        
        # Get best prediction
        prediction = np.argmax(probs)
        confidence = probs[prediction]
        
        # Map to signal
        signals = ['BUY_CALL', 'BUY_PUT', 'HOLD', 'CLOSE']
        signal = signals[prediction]
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            signal = 'HOLD'
            
        return signal, confidence

# BUILD ON FEATURE ENGINE
ml_model = MLPredictor(feature_engine)
```

### Week 3: Trading Logic

#### Day 1-2: Signal Generation

**Create**: `src/trading/signals.py`
```python
from datetime import datetime, time
from typing import Optional, Dict
import asyncio

class SignalGenerator:
    """
    Generates trading signals - BRAIN OF THE SYSTEM
    Reused by paper and live trading
    """
    def __init__(self, ml_model: MLPredictor, 
                 feature_engine: FeatureEngine,
                 market_data: MarketDataManager,
                 options_data: OptionsDataManager):
        self.ml = ml_model
        self.features = feature_engine
        self.market = market_data
        self.options = options_data
        
        self.signals_today = []
        self.last_signal_time = {}
        self.min_time_between_signals = 300  # 5 minutes
        
    async def generate_signals(self, symbols: List[str]) -> List[Dict]:
        """
        Generate signals for symbols - CORE LOGIC
        Called by both paper and live traders
        """
        signals = []
        
        for symbol in symbols:
            # Check if enough time has passed
            if symbol in self.last_signal_time:
                time_since = (datetime.now() - self.last_signal_time[symbol]).seconds
                if time_since < self.min_time_between_signals:
                    continue
                    
            # Get historical data
            historical = await self.market.get_historical_bars(symbol, days=1)
            
            if len(historical) < 100:
                continue
                
            # Calculate features
            features = self.features.calculate_features(symbol, historical)
            
            # Get ML prediction
            signal_type, confidence = self.ml.predict(features)
            
            if signal_type != 'HOLD':
                # Find best option to trade
                option = self._select_option(symbol, signal_type)
                
                if option:
                    signal = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'option': option,
                        'features': features.tolist()
                    }
                    
                    signals.append(signal)
                    self.signals_today.append(signal)
                    self.last_signal_time[symbol] = datetime.now()
                    
        return signals
        
    def _select_option(self, symbol: str, signal_type: str) -> Optional[Dict]:
        """Select best option contract to trade"""
        atm_options = self.options.find_atm_options(symbol, dte_min=0, dte_max=7)
        
        if not atm_options:
            return None
            
        # Use shortest DTE for now (most gamma)
        option = atm_options[0]
        
        return {
            'strike': option['strike'],
            'expiry': option['expiry'],
            'type': 'CALL' if 'CALL' in signal_type else 'PUT',
            'contract': option['call'] if 'CALL' in signal_type else option['put']
        }

# SIGNAL GENERATOR REUSED BY ALL TRADING
signal_generator = SignalGenerator(ml_model, feature_engine, market_data, options_data)
```

#### Day 3-4: Risk Management

**Create**: `src/trading/risk.py`
```python
from typing import Dict, List, Tuple
import numpy as np

class RiskManager:
    """
    Risk management - NEVER BYPASSED
    Same rules for paper and live
    """
    def __init__(self, config: TradingConfig, 
                 options_data: OptionsDataManager,
                 db: DatabaseManager):
        self.config = config
        self.options = options_data
        self.db = db
        
        # Risk limits
        self.max_positions = config.max_positions
        self.max_position_size = config.max_position_size
        self.daily_loss_limit = config.daily_loss_limit
        
        # Greeks limits
        self.greeks_limits = {
            'delta': (-0.3, 0.3),
            'gamma': (-0.5, 0.5),
            'vega': (-500, 500),
            'theta': (-200, float('inf'))
        }
        
        # Current state
        self.positions = {}
        self.daily_pnl = 0.0
        self.portfolio_greeks = {
            'delta': 0.0, 'gamma': 0.0, 
            'vega': 0.0, 'theta': 0.0
        }
        
    def can_trade(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if trade is allowed - CRITICAL GATE
        Used by paper and live equally
        """
        # Check position count
        if len(self.positions) >= self.max_positions:
            return False, "Max positions reached"
            
        # Check daily loss
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, "Daily loss limit reached"
            
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        if position_size > self.max_position_size:
            return False, f"Position size ${position_size} exceeds limit"
            
        # Check Greeks impact
        projected_greeks = self._project_greeks_impact(signal)
        for greek, (min_val, max_val) in self.greeks_limits.items():
            new_value = self.portfolio_greeks[greek] + projected_greeks[greek]
            if new_value < min_val or new_value > max_val:
                return False, f"Would breach {greek} limit"
                
        return True, "OK"
        
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size in dollars"""
        # For options: 100 shares per contract * price
        option_price = 2.0  # Estimate $2 per contract
        contracts = 5  # Start with 5 contracts
        return contracts * 100 * option_price
        
    def _project_greeks_impact(self, signal: Dict) -> Dict[str, float]:
        """Project Greeks impact of new position"""
        spot = self.options.market.get_latest_price(signal['symbol'])
        
        greeks = self.options.calculate_greeks(
            spot=spot,
            strike=signal['option']['strike'],
            time_to_expiry=(signal['option']['expiry'] - datetime.now().date()).days / 365.0,
            volatility=0.20,  # Estimate
            option_type=signal['option']['type']
        )
        
        # Multiply by position size (5 contracts)
        for key in greeks:
            greeks[key] *= 5
            
        return greeks
        
    def update_position(self, symbol: str, position: Dict):
        """Update position tracking"""
        self.positions[symbol] = position
        self._recalculate_portfolio_greeks()
        
    def _recalculate_portfolio_greeks(self):
        """Recalculate total portfolio Greeks"""
        total_greeks = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        for symbol, position in self.positions.items():
            spot = self.options.market.get_latest_price(symbol)
            
            greeks = self.options.calculate_greeks(
                spot=spot,
                strike=position['strike'],
                time_to_expiry=position['dte'] / 365.0,
                volatility=0.20,
                option_type=position['option_type']
            )
            
            for key in total_greeks:
                total_greeks[key] += greeks[key] * position['quantity']
                
        self.portfolio_greeks = total_greeks

# RISK MANAGER USED BY ALL TRADING
risk_manager = RiskManager(config, options_data, db)
```

### Week 4: Integration Testing

#### Day 1-2: Integration Tests

**Create**: `tests/test_integration.py`
```python
import pytest
import asyncio
from datetime import datetime, timedelta

@pytest.mark.asyncio
async def test_data_flow():
    """Test that data flows through all components"""
    # Initialize components (reusing all from above)
    await market_data.connect()
    await market_data.subscribe_symbols(['SPY'])
    
    # Wait for data
    await asyncio.sleep(10)
    
    # Verify data is flowing
    assert market_data.get_latest_price('SPY') > 0
    
    # Test options data
    chain = await options_data.fetch_option_chain('SPY')
    assert len(chain) > 0
    
    # Test Greeks calculation
    greeks = options_data.calculate_greeks(450, 450, 0.1, 0.2, 'CALL')
    assert 0 <= greeks['delta'] <= 1
    
@pytest.mark.asyncio
async def test_signal_generation():
    """Test signal generation with real data"""
    # Get historical data
    historical = await market_data.get_historical_bars('SPY', days=1)
    
    # Calculate features
    features = feature_engine.calculate_features('SPY', historical)
    assert len(features) == len(feature_engine.feature_names)
    
    # Get prediction
    signal, confidence = ml_model.predict(features)
    assert signal in ['BUY_CALL', 'BUY_PUT', 'HOLD', 'CLOSE']
    assert 0 <= confidence <= 1
    
@pytest.mark.asyncio
async def test_risk_checks():
    """Test risk management"""
    signal = {
        'symbol': 'SPY',
        'signal_type': 'BUY_CALL',
        'option': {
            'strike': 450,
            'expiry': datetime.now().date() + timedelta(days=7),
            'type': 'CALL'
        }
    }
    
    can_trade, reason = risk_manager.can_trade(signal)
    assert isinstance(can_trade, bool)
    assert isinstance(reason, str)
```

#### Day 3-5: Performance Testing

**Create**: `tests/test_performance.py`
```python
import time
import numpy as np

def test_feature_calculation_speed():
    """Ensure features calculate quickly"""
    # Create dummy data
    data = pd.DataFrame({
        'close': np.random.randn(1000) + 450,
        'high': np.random.randn(1000) + 451,
        'low': np.random.randn(1000) + 449,
        'volume': np.random.randint(1000000, 5000000, 1000)
    })
    
    start = time.time()
    features = feature_engine.calculate_features('SPY', data)
    elapsed = time.time() - start
    
    assert elapsed < 0.1  # Should be under 100ms
    print(f"Feature calculation: {elapsed*1000:.2f}ms")
    
def test_greeks_calculation_speed():
    """Test Greeks calculation performance"""
    start = time.time()
    
    for _ in range(100):  # 100 contracts
        greeks = options_data.calculate_greeks(450, 450, 0.1, 0.2, 'CALL')
        
    elapsed = time.time() - start
    assert elapsed < 0.05  # 50ms for 100 contracts
    print(f"Greeks for 100 contracts: {elapsed*1000:.2f}ms")
```

---

## PHASE 2: PAPER TRADING (Weeks 5-8)
*Everything builds on Phase 1 - no rewrites!*

### Week 5-6: Paper Trading Implementation

**Create**: `src/trading/paper_trader.py`
```python
class PaperTrader:
    """
    Paper trading - REUSES ALL COMPONENTS FROM PHASE 1
    """
    def __init__(self, 
                 signal_generator: SignalGenerator,
                 risk_manager: RiskManager,
                 market_data: MarketDataManager,
                 options_data: OptionsDataManager,
                 db: DatabaseManager):
        # Reuse everything!
        self.signals = signal_generator
        self.risk = risk_manager
        self.market = market_data
        self.options = options_data
        self.db = db
        
        # Paper trading specific
        self.starting_capital = 100000
        self.cash = self.starting_capital
        self.positions = {}
        self.trades = []
        
    async def run(self):
        """Main paper trading loop"""
        print("Starting paper trading...")
        
        while True:
            # Market hours check
            if not self._is_market_open():
                await asyncio.sleep(60)
                continue
                
            # Generate signals using existing generator
            signals = await self.signals.generate_signals(['SPY', 'QQQ', 'IWM'])
            
            for signal in signals:
                # Use existing risk manager
                can_trade, reason = self.risk.can_trade(signal)
                
                if can_trade:
                    await self.execute_paper_trade(signal)
                else:
                    print(f"Signal rejected: {reason}")
                    
            # Update positions
            await self.update_positions()
            
            # Wait for next cycle
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def execute_paper_trade(self, signal: Dict):
        """Execute paper trade"""
        # Simulate fill with slippage
        fill_price = self._simulate_fill(signal)
        
        trade = {
            'timestamp': datetime.now(),
            'mode': 'paper',
            'symbol': signal['symbol'],
            'option_type': signal['option']['type'],
            'strike': signal['option']['strike'],
            'expiry': signal['option']['expiry'],
            'action': 'BUY',
            'quantity': 5,  # 5 contracts
            'price': fill_price,
            'commission': 0.65 * 5,
            'pnl': 0  # Updated later
        }
        
        # Store in database (reusing existing)
        with self.db.get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trades 
                (timestamp, mode, symbol, option_type, strike, expiry, 
                 action, quantity, price, commission, pnl)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, tuple(trade.values()))
            trade_id = cur.fetchone()[0]
            conn.commit()
            
        # Update position tracking
        self.positions[signal['symbol']] = trade
        self.risk.update_position(signal['symbol'], trade)
        
        print(f"Paper trade executed: {trade}")
        
# PAPER TRADER REUSES EVERYTHING
paper_trader = PaperTrader(signal_generator, risk_manager, 
                          market_data, options_data, db)
```

### Week 7-8: Community Platform

**Create**: `src/community/discord_bot.py`
```python
import discord
from discord.ext import commands
import asyncio

class TradingBot(commands.Bot):
    """
    Discord bot - PUBLISHES PAPER TRADES
    """
    def __init__(self, paper_trader: PaperTrader):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.paper = paper_trader  # Reuse paper trader!
        self.channels = {
            'signals': None,  # Set on ready
            'performance': None
        }
        
    async def on_ready(self):
        print(f'Bot connected as {self.user}')
        
        # Get channels
        self.channels['signals'] = self.get_channel(YOUR_SIGNALS_CHANNEL_ID)
        self.channels['performance'] = self.get_channel(YOUR_PERFORMANCE_CHANNEL_ID)
        
        # Start publishing loop
        self.loop.create_task(self.publish_trades())
        
    async def publish_trades(self):
        """Publish paper trades to Discord"""
        last_trade_id = 0
        
        while True:
            # Get new trades from database
            with self.paper.db.get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT * FROM trades 
                    WHERE id > %s AND mode = 'paper'
                    ORDER BY id
                """, (last_trade_id,))
                
                new_trades = cur.fetchall()
                
            for trade in new_trades:
                # Format trade as embed
                embed = discord.Embed(
                    title=f"📈 Paper Trade: {trade['symbol']}",
                    color=discord.Color.green() if trade['action'] == 'BUY' else discord.Color.red()
                )
                
                embed.add_field(name="Action", value=trade['action'])
                embed.add_field(name="Option", value=f"{trade['option_type']} ${trade['strike']}")
                embed.add_field(name="Quantity", value=f"{trade['quantity']} contracts")
                embed.add_field(name="Price", value=f"${trade['price']:.2f}")
                
                await self.channels['signals'].send(embed=embed)
                
                last_trade_id = trade['id']
                
            await asyncio.sleep(5)  # Check every 5 seconds

# BOT REUSES PAPER TRADER
bot = TradingBot(paper_trader)
```

---

## PHASE 3: PRODUCTION (Weeks 9-12)
*Switches from paper to live - reuses everything!*

### Week 9-10: Live Trading

**Create**: `src/trading/live_trader.py`
```python
class LiveTrader:
    """
    Live trading - REUSES PAPER TRADER LOGIC
    """
    def __init__(self, paper_trader: PaperTrader):
        # Reuse all paper trader components!
        self.paper = paper_trader
        
        # Just change execution
        self.market = paper_trader.market
        self.positions = {}
        
    async def run(self):
        """Live trading - same as paper but real orders"""
        print("Starting LIVE trading...")
        
        while True:
            # Generate signals (SAME AS PAPER)
            signals = await self.paper.signals.generate_signals(['SPY', 'QQQ', 'IWM'])
            
            for signal in signals:
                # Risk check (SAME AS PAPER)
                can_trade, reason = self.paper.risk.can_trade(signal)
                
                if can_trade:
                    # Only difference: real order
                    await self.execute_live_trade(signal)
                    
            await asyncio.sleep(30)
            
    async def execute_live_trade(self, signal: Dict):
        """Execute real trade through IBKR"""
        # Create option contract
        contract = Option(
            signal['symbol'], 
            signal['option']['expiry'].strftime('%Y%m%d'),
            signal['option']['strike'],
            signal['option']['type'],
            'SMART'
        )
        
        # Create order
        order = MarketOrder('BUY', 5)  # 5 contracts
        
        # Place order
        trade = self.market.ib.placeOrder(contract, order)
        
        # Wait for fill
        while not trade.isDone():
            await asyncio.sleep(0.1)
            
        # Store in database (SAME SCHEMA AS PAPER)
        # ... same database code as paper ...

# LIVE TRADER REUSES PAPER TRADER
live_trader = LiveTrader(paper_trader)
```

---

## PHASE 4: OPTIMIZATION (Weeks 13-16)
*Performance tuning and feature additions*

### Week 13-14: Performance Optimization
- Profile and optimize bottlenecks
- Add caching layers
- Implement parallel processing
- All optimizations work with existing code

### Week 15-16: Advanced Features
- Add spread strategies (builds on single-leg)
- Add more symbols (just configuration)
- Enhance ML model (retrain with live data)
- Scale community features

---

## CODE REUSE SUMMARY

```
Component           | Built In | Reused By                      | Never Rewritten
--------------------|----------|----------------------------------|----------------
MarketDataManager   | Week 1   | Everything                      | ✓
OptionsDataManager  | Week 1   | ML, Risk, Trading, Community   | ✓
DatabaseManager     | Week 1   | All components                  | ✓
FeatureEngine      | Week 2   | ML, Paper, Live                 | ✓
MLPredictor        | Week 2   | Signals, Paper, Live            | ✓
SignalGenerator    | Week 3   | Paper, Live, Community          | ✓
RiskManager        | Week 3   | Paper, Live                     | ✓
PaperTrader        | Week 5   | Live (reuses 90%), Community   | ✓
DiscordBot         | Week 7   | Paper and Live                  | ✓
LiveTrader         | Week 9   | Extends PaperTrader             | ✓
```

## SUCCESS METRICS BY WEEK

**Week 4**: System generates signals from real market data
**Week 6**: Paper trading running continuously
**Week 8**: Discord bot publishing all trades
**Week 10**: First live trades executed
**Week 12**: System profitable and stable
**Week 14**: Performance optimized (<100ms)
**Week 16**: Full production with advanced features

---

This plan shows EXACTLY how each component builds on previous work. No rewrites, no wasted effort - just progressive enhancement of a solid foundation.