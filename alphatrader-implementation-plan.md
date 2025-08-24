# AlphaTrader Implementation Plan v2.0
## 16-Week Progressive Build with Alpha Vantage Integration

---

## OVERVIEW: Build Once, Extend Forever

This plan shows EXACTLY how each component is built once and reused throughout the project. Every week builds on previous work - no rewrites, no wasted effort. Alpha Vantage provides comprehensive options data with Greeks, technical indicators, and market analytics.

```
Week 1-2:   Data Foundation (Alpha Vantage + IBKR Integration)
Week 3-4:   ML & Trading Logic (Using AV data feeds)
Week 5-6:   Paper Trading (Reuses all above)
Week 7-8:   Community Platform (Publishes paper trades)
Week 9-10:  Production Prep (Small real money)
Week 11-12: Full Production (Scale up)
Week 13-14: Optimization (Performance tuning)
Week 15-16: Advanced Features (Spreads, more symbols)
```

## KEY ARCHITECTURE PRINCIPLE
```
Alpha Vantage (600 calls/min) → Options chains, Greeks, IV, Technical Indicators, Sentiment, Fundamentals
IBKR → Market quotes, bars, trade execution
```

---

## PHASE 1: FOUNDATION (Weeks 1-4)
*Build the reusable core with Alpha Vantage and IBKR integration*

### Week 1: Data Layer Foundation

#### Day 1-2: Project Setup & Dual Data Source Architecture

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
    
    # IBKR settings (for quotes/bars/execution)
    ibkr_host: str = '127.0.0.1'
    ibkr_port: int = 7497  # Paper port
    
    # Alpha Vantage settings (for options/Greeks/indicators)
    av_api_key: str = os.getenv('AV_API_KEY')
    av_rate_limit: int = 600  # Premium tier
    av_cache_ttl: Dict[str, int] = {
        'options': 60,      # 1 minute for real-time options
        'historical': 3600, # 1 hour for historical data
        'indicators': 300,  # 5 minutes for technical indicators
        'sentiment': 900    # 15 minutes for news/sentiment
    }
    
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
from ib_insync import IB, Stock, MarketOrder
import asyncio
from typing import Dict, Optional
import pandas as pd

class MarketDataManager:
    """
    IBKR market data for quotes, bars, and execution
    Alpha Vantage handles options - this is just spot prices
    Used by: ML, Trading, Risk, Paper, Live, Community
    """
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ib = IB()
        self.connected = False
        self.latest_prices = {}  # Cache for instant access
        self.bars_5sec = {}  # 5-second bars from IBKR
        
    async def connect(self):
        """Connect to IBKR - reused for paper and live"""
        port = 7497 if self.config.mode == 'paper' else 7496
        await self.ib.connectAsync(
            self.config.ibkr_host, 
            port, 
            clientId=1
        )
        self.connected = True
        print(f"Connected to IBKR ({self.config.mode} mode) for quotes/execution")
        
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to market data - spot prices for options pricing"""
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
            # This spot price is used by Alpha Vantage options queries
            
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price - instant from cache"""
        return self.latest_prices.get(symbol, 0.0)
        
    async def get_ibkr_bars(self, symbol: str, duration: str = '1 D'):
        """Get IBKR historical bars for immediate price action"""
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting='5 secs',
            whatToShow='TRADES',
            useRTH=True
        )
        return pd.DataFrame(bars)
        
    async def execute_order(self, contract, order):
        """Execute trades through IBKR"""
        if self.config.mode == 'paper':
            # Paper trading through IBKR
            trade = self.ib.placeOrder(contract, order)
        else:
            # Live trading
            trade = self.ib.placeOrder(contract, order)
        return trade

# CREATE ONCE, USE FOREVER
market_data = MarketDataManager(config)
```

#### Day 3-4: Alpha Vantage Integration for Options & Greeks

**Create**: `src/data/alpha_vantage_client.py`
```python
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

@dataclass
class OptionContract:
    """Option data from Alpha Vantage - Greeks INCLUDED"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # CALL or PUT
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    # Greeks PROVIDED by Alpha Vantage - no calculation needed!
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class AlphaVantageClient:
    """
    Alpha Vantage API client - 600 calls/minute premium tier
    Provides: Options, Greeks, Technical Indicators, Sentiment, Fundamentals
    """
    def __init__(self, config: TradingConfig):
        self.config = config
        self.api_key = config.av_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = RateLimiter(600, 60)  # 600 calls per minute
        self.cache = {}
        self.session = None
        
    async def connect(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()
        print(f"Alpha Vantage client ready (600 calls/min tier)")
        
    async def get_realtime_options(self, symbol: str, 
                                  require_greeks: bool = True) -> List[OptionContract]:
        """
        Get real-time options WITH GREEKS from Alpha Vantage
        Greeks are PROVIDED - no calculation needed!
        """
        params = {
            'function': 'REALTIME_OPTIONS',
            'symbol': symbol,
            'require_greeks': 'true' if require_greeks else 'false',
            'apikey': self.api_key
        }
        
        cache_key = f"options_{symbol}_{datetime.now().minute}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        async with self.rate_limiter:
            async with self.session.get(self.base_url, params=params) as resp:
                data = await resp.json()
                
        options = []
        for contract_data in data.get('options', []):
            option = OptionContract(
                symbol=symbol,
                strike=contract_data['strike'],
                expiry=contract_data['expiry'],
                option_type=contract_data['type'],
                bid=contract_data['bid'],
                ask=contract_data['ask'],
                last=contract_data['last'],
                volume=contract_data['volume'],
                open_interest=contract_data['open_interest'],
                implied_volatility=contract_data['implied_volatility'],
                # Greeks PROVIDED by Alpha Vantage!
                delta=contract_data['delta'],
                gamma=contract_data['gamma'],
                theta=contract_data['theta'],
                vega=contract_data['vega'],
                rho=contract_data['rho']
            )
            options.append(option)
            
        self.cache[cache_key] = options
        return options
        
    async def get_historical_options(self, symbol: str, date: str) -> List[OptionContract]:
        """
        Get historical options data - up to 20 YEARS of history with Greeks!
        Alpha Vantage provides complete historical Greeks
        """
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'date': date,
            'apikey': self.api_key
        }
        
        async with self.rate_limiter:
            async with self.session.get(self.base_url, params=params) as resp:
                data = await resp.json()
                
        # Process historical options with Greeks included
        return self._process_options_response(data)
        
    async def get_technical_indicator(self, symbol: str, indicator: str, **kwargs) -> pd.DataFrame:
        """
        Get technical indicators from Alpha Vantage
        No local calculation - AV provides everything
        """
        params = {
            'function': indicator,
            'symbol': symbol,
            'apikey': self.api_key,
            **kwargs
        }
        
        cache_key = f"{indicator}_{symbol}_{datetime.now().hour}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        async with self.rate_limiter:
            async with self.session.get(self.base_url, params=params) as resp:
                data = await resp.json()
                
        df = pd.DataFrame(data.get(f'Technical Analysis: {indicator}', {})).T
        self.cache[cache_key] = df
        return df
        
    async def get_news_sentiment(self, symbols: List[str]) -> Dict:
        """Get news sentiment from Alpha Vantage"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ','.join(symbols),  # Note: 'tickers' not 'symbol'
            'apikey': self.api_key,
            'sort': 'LATEST',
            'limit': 50
        }
        
        async with self.rate_limiter:
            async with self.session.get(self.base_url, params=params) as resp:
                return await resp.json()

# Alpha Vantage client for all options and analytics
av_client = AlphaVantageClient(config)
```

**Create**: `src/data/options_data.py`
```python
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class OptionsDataManager:
    """
    Options data manager using Alpha Vantage
    Greeks are PROVIDED by Alpha Vantage - NO local calculation needed!
    Built on top of MarketDataManager for spot prices
    """
    def __init__(self, market_data_manager: MarketDataManager, 
                 av_client: AlphaVantageClient):
        self.market = market_data_manager  # For spot prices
        self.av = av_client  # For options and Greeks
        self.chains = {}
        self.latest_greeks = {}  # Cache latest Greeks from AV
        
    async def fetch_option_chain(self, symbol: str) -> List[OptionContract]:
        """
        Fetch option chain from Alpha Vantage
        Greeks are INCLUDED - no calculation needed!
        """
        # Get real-time options with Greeks from Alpha Vantage
        options = await self.av.get_realtime_options(symbol, require_greeks=True)
        
        # Cache the chain
        self.chains[symbol] = options
        
        # Cache Greeks for quick access
        for option in options:
            key = f"{symbol}_{option.strike}_{option.expiry}_{option.option_type}"
            self.latest_greeks[key] = {
                'delta': option.delta,
                'gamma': option.gamma,
                'theta': option.theta,
                'vega': option.vega,
                'rho': option.rho
            }
            
        return options
        
    def get_option_greeks(self, symbol: str, strike: float, 
                         expiry: str, option_type: str) -> Dict[str, float]:
        """
        Get Greeks for specific option - FROM ALPHA VANTAGE CACHE
        No calculation - just retrieval!
        """
        key = f"{symbol}_{strike}_{expiry}_{option_type}"
        
        if key in self.latest_greeks:
            return self.latest_greeks[key]
        else:
            # If not in cache, fetch fresh data
            asyncio.create_task(self.fetch_option_chain(symbol))
            # Return zeros temporarily
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
    def find_atm_options(self, symbol: str, dte_min: int = 0, dte_max: int = 7):
        """Find ATM options for trading - using Alpha Vantage data"""
        spot = self.market.get_latest_price(symbol)
        chain = self.chains.get(symbol, [])
        
        atm_options = []
        for option in chain:
            expiry_date = datetime.strptime(option.expiry, '%Y-%m-%d')
            dte = (expiry_date - datetime.now()).days
            
            if dte_min <= dte <= dte_max:
                # Find options near the money
                if abs(option.strike - spot) / spot < 0.02:  # Within 2% of spot
                    atm_options.append({
                        'contract': option,
                        'strike': option.strike,
                        'expiry': option.expiry,
                        'dte': dte,
                        'type': option.option_type,
                        # Greeks from Alpha Vantage!
                        'greeks': {
                            'delta': option.delta,
                            'gamma': option.gamma,
                            'theta': option.theta,
                            'vega': option.vega,
                            'rho': option.rho
                        }
                    })
                
        return sorted(atm_options, key=lambda x: abs(x['strike'] - spot))
        
    async def get_historical_options_ml_data(self, symbol: str, days_back: int = 30):
        """
        Get historical options data for ML training
        Alpha Vantage provides up to 20 YEARS of historical options with Greeks!
        """
        historical_data = []
        
        for day in range(days_back):
            date = (datetime.now() - timedelta(days=day)).strftime('%Y-%m-%d')
            options = await self.av.get_historical_options(symbol, date)
            
            for option in options:
                historical_data.append({
                    'date': date,
                    'symbol': symbol,
                    'strike': option.strike,
                    'expiry': option.expiry,
                    'type': option.option_type,
                    'price': option.last,
                    'iv': option.implied_volatility,
                    # Historical Greeks from Alpha Vantage!
                    'delta': option.delta,
                    'gamma': option.gamma,
                    'theta': option.theta,
                    'vega': option.vega,
                    'volume': option.volume,
                    'oi': option.open_interest
                })
                
        return pd.DataFrame(historical_data)

# BUILD ON TOP OF MARKET DATA AND ALPHA VANTAGE
options_data = OptionsDataManager(market_data, av_client)
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
    Stores both IBKR execution data and Alpha Vantage analytics
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
        
        # Redis for caching Alpha Vantage responses
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Create tables if not exist
        self._init_tables()
        
    def _init_tables(self):
        """Create tables - stores both IBKR and AV data"""
        with self.get_db() as conn:
            cur = conn.cursor()
            
            # Trades table - execution through IBKR
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
                    pnl DECIMAL,
                    -- Greeks at entry (from Alpha Vantage)
                    entry_delta DECIMAL,
                    entry_gamma DECIMAL,
                    entry_theta DECIMAL,
                    entry_vega DECIMAL,
                    entry_iv DECIMAL
                )
            """)
            
            # Signals table - tracks all signals with AV data
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    symbol VARCHAR(10),
                    signal_type VARCHAR(20),
                    confidence DECIMAL,
                    features JSONB,  -- Store feature vector
                    av_indicators JSONB,  -- Alpha Vantage indicators used
                    av_sentiment JSONB,  -- Alpha Vantage sentiment data
                    executed BOOLEAN DEFAULT FALSE,
                    trade_id INT REFERENCES trades(id)
                )
            """)
            
            # Alpha Vantage API monitoring
            cur.execute("""
                CREATE TABLE IF NOT EXISTS av_api_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    endpoint VARCHAR(50),
                    response_time_ms INT,
                    cache_hit BOOLEAN,
                    rate_limit_remaining INT
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
            
    def cache_av_response(self, key: str, value: any, ttl: int = None):
        """Cache Alpha Vantage responses with appropriate TTL"""
        if ttl is None:
            # Use default TTLs based on data type
            if 'options' in key:
                ttl = 60  # 1 minute for options
            elif 'indicator' in key:
                ttl = 300  # 5 minutes for indicators
            else:
                ttl = 900  # 15 minutes default
                
        self.redis.setex(key, ttl, json.dumps(value))
        
    def get_av_cache(self, key: str):
        """Get from Alpha Vantage cache"""
        value = self.redis.get(key)
        return json.loads(value) if value else None

# ONE DATABASE MANAGER FOR EVERYTHING
db = DatabaseManager(config)
```

### Week 2: ML and Analytics Foundation

#### Day 1-2: Feature Engineering with Alpha Vantage

**Create**: `src/analytics/features.py`
```python
import pandas as pd
import numpy as np
from typing import Dict, List
import asyncio

class FeatureEngine:
    """
    Feature engineering using Alpha Vantage data feeds
    All indicators from AV - no local calculation
    """
    def __init__(self, options_manager: OptionsDataManager, 
                 av_client: AlphaVantageClient):
        self.options = options_manager
        self.av = av_client
        
        # Define feature names for consistency
        self.feature_names = [
            # Price action (from IBKR bars)
            'returns_5m', 'returns_30m', 'returns_1h',
            'volume_ratio', 'high_low_ratio',
            
            # Technical indicators (from Alpha Vantage)
            'rsi', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_position',
            'atr', 'adx', 'obv_slope', 'vwap_distance',
            'ema_20', 'sma_50', 'momentum', 'cci',
            
            # Options metrics (from Alpha Vantage)
            'iv_rank', 'iv_percentile', 'put_call_ratio',
            'atm_delta', 'atm_gamma', 'atm_theta', 'atm_vega',
            'gamma_exposure', 'max_pain_distance',
            'call_volume', 'put_volume', 'oi_ratio',
            
            # Sentiment (from Alpha Vantage)
            'news_sentiment_score', 'news_volume',
            'insider_sentiment', 'social_sentiment',
            
            # Market structure (from Alpha Vantage)
            'spy_correlation', 'qqq_correlation',
            'vix_level', 'term_structure', 'market_regime'
        ]
        
    async def calculate_features(self, symbol: str) -> np.ndarray:
        """
        Calculate all features using Alpha Vantage APIs
        Parallel API calls for efficiency with 600 calls/min limit
        """
        features = {}
        
        # Get IBKR price data for basic returns
        bars = await self.options.market.get_ibkr_bars(symbol, '1 D')
        features['returns_5m'] = self._calculate_returns(bars, 60)
        features['returns_30m'] = self._calculate_returns(bars, 360)
        features['returns_1h'] = self._calculate_returns(bars, 720)
        features['volume_ratio'] = bars['volume'].iloc[-1] / bars['volume'].mean()
        features['high_low_ratio'] = (bars['high'].iloc[-1] - bars['low'].iloc[-1]) / bars['close'].iloc[-1]
        
        # Parallel Alpha Vantage API calls for technical indicators
        tasks = [
            self.av.get_technical_indicator(symbol, 'RSI', interval='5min', time_period=14),
            self.av.get_technical_indicator(symbol, 'MACD', interval='5min'),
            self.av.get_technical_indicator(symbol, 'BBANDS', interval='5min'),
            self.av.get_technical_indicator(symbol, 'ATR', interval='5min'),
            self.av.get_technical_indicator(symbol, 'ADX', interval='5min'),
            self.av.get_technical_indicator(symbol, 'OBV', interval='5min'),
            self.av.get_technical_indicator(symbol, 'VWAP', interval='5min'),
            self.av.get_technical_indicator(symbol, 'EMA', interval='5min', time_period=20),
            self.av.get_technical_indicator(symbol, 'SMA', interval='5min', time_period=50),
            self.av.get_technical_indicator(symbol, 'MOM', interval='5min'),
            self.av.get_technical_indicator(symbol, 'CCI', interval='5min')
        ]
        
        # Execute all indicator fetches in parallel
        indicator_results = await asyncio.gather(*tasks)
        
        # Process technical indicators from Alpha Vantage
        features['rsi'] = indicator_results[0].iloc[0]['RSI'] / 100.0 if not indicator_results[0].empty else 0.5
        
        macd_data = indicator_results[1]
        if not macd_data.empty:
            features['macd_signal'] = macd_data.iloc[0].get('MACD_Signal', 0)
            features['macd_histogram'] = macd_data.iloc[0].get('MACD_Hist', 0)
        else:
            features['macd_signal'] = 0
            features['macd_histogram'] = 0
            
        # Continue processing other indicators...
        
        # Get options features from Alpha Vantage
        options_features = await self._get_av_options_features(symbol)
        features.update(options_features)
        
        # Get sentiment from Alpha Vantage
        sentiment_features = await self._get_av_sentiment_features(symbol)
        features.update(sentiment_features)
        
        # Convert to array in consistent order
        feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Handle any NaN values
        feature_array = np.nan_to_num(feature_array, 0.0)
        
        return feature_array
        
    async def _get_av_options_features(self, symbol: str) -> Dict:
        """Get options-specific features from Alpha Vantage"""
        features = {}
        
        # Get current options chain with Greeks
        options = await self.av.get_realtime_options(symbol, require_greeks=True)
        
        if options:
            # Find ATM option
            spot = self.options.market.get_latest_price(symbol)
            atm_call = min(
                [opt for opt in options if opt.option_type == 'CALL'],
                key=lambda x: abs(x.strike - spot)
            )
            atm_put = min(
                [opt for opt in options if opt.option_type == 'PUT'],
                key=lambda x: abs(x.strike - spot)
            )
            
            # Greeks directly from Alpha Vantage - no calculation!
            features['atm_delta'] = atm_call.delta
            features['atm_gamma'] = atm_call.gamma
            features['atm_theta'] = atm_call.theta
            features['atm_vega'] = atm_call.vega
            
            # IV metrics
            all_ivs = [opt.implied_volatility for opt in options]
            features['iv_rank'] = np.percentile(all_ivs, 50) / 100.0
            features['iv_percentile'] = len([iv for iv in all_ivs if iv < atm_call.implied_volatility]) / len(all_ivs)
            
            # Volume metrics
            call_volume = sum(opt.volume for opt in options if opt.option_type == 'CALL')
            put_volume = sum(opt.volume for opt in options if opt.option_type == 'PUT')
            features['call_volume'] = call_volume
            features['put_volume'] = put_volume
            features['put_call_ratio'] = put_volume / call_volume if call_volume > 0 else 1.0
            
            # Open interest
            call_oi = sum(opt.open_interest for opt in options if opt.option_type == 'CALL')
            put_oi = sum(opt.open_interest for opt in options if opt.option_type == 'PUT')
            features['oi_ratio'] = put_oi / call_oi if call_oi > 0 else 1.0
            
            # Gamma exposure (using Greeks from AV)
            features['gamma_exposure'] = sum(
                opt.gamma * opt.open_interest * 100 * spot
                for opt in options
            )
            
            # Max pain calculation
            features['max_pain_distance'] = 0  # Simplified for now
            
        else:
            # Default values if no options available
            features.update({
                'atm_delta': 0.5, 'atm_gamma': 0.0, 'atm_theta': 0.0, 'atm_vega': 0.0,
                'iv_rank': 0.5, 'iv_percentile': 0.5,
                'put_call_ratio': 1.0, 'gamma_exposure': 0.0,
                'max_pain_distance': 0.0, 'call_volume': 0, 'put_volume': 0,
                'oi_ratio': 1.0
            })
            
        return features
        
    async def _get_av_sentiment_features(self, symbol: str) -> Dict:
        """Get sentiment features from Alpha Vantage"""
        features = {}
        
        # Get news sentiment
        sentiment_data = await self.av.get_news_sentiment([symbol])
        
        if sentiment_data and 'feed' in sentiment_data:
            sentiments = []
            for article in sentiment_data['feed'][:10]:  # Last 10 articles
                ticker_sentiment = next(
                    (s for s in article.get('ticker_sentiment', []) 
                     if s['ticker'] == symbol), 
                    None
                )
                if ticker_sentiment:
                    sentiments.append(float(ticker_sentiment.get('ticker_sentiment_score', 0)))
                    
            features['news_sentiment_score'] = np.mean(sentiments) if sentiments else 0.0
            features['news_volume'] = len(sentiments)
        else:
            features['news_sentiment_score'] = 0.0
            features['news_volume'] = 0
            
        # Placeholder for other sentiment sources
        features['insider_sentiment'] = 0.0  # Would come from INSIDER_TRANSACTIONS
        features['social_sentiment'] = 0.0
        
        # Market structure (simplified)
        features['spy_correlation'] = 0.8
        features['qqq_correlation'] = 0.7
        features['vix_level'] = 20.0 / 100.0
        features['term_structure'] = 0.0
        features['market_regime'] = 0.5
        
        return features

# BUILD ON OPTIONS DATA AND ALPHA VANTAGE
feature_engine = FeatureEngine(options_data, av_client)
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
    ML model for predictions using Alpha Vantage data
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
                num_class=4  # BUY_CALL, BUY_PUT, HOLD, CLOSE
            )
            
    async def train_with_av_historical(self, symbols: List[str], days_back: int = 30):
        """
        Train model using Alpha Vantage historical options data
        AV provides 20 years of history with Greeks!
        """
        print(f"Training with {days_back} days of Alpha Vantage historical data...")
        
        X = []
        y = []
        
        for symbol in symbols:
            # Get historical options data from Alpha Vantage
            hist_options = await self.features.options.get_historical_options_ml_data(
                symbol, days_back
            )
            
            # Get historical price data for labels
            hist_prices = await self.features.options.market.get_ibkr_bars(
                symbol, f'{days_back} D'
            )
            
            for i in range(len(hist_options) - 100):
                # Calculate features using Alpha Vantage data
                features = await self.features.calculate_features(symbol)
                X.append(features)
                
                # Generate label based on price movement
                future_return = (hist_prices.iloc[i+12]['close'] - 
                               hist_prices.iloc[i]['close']) / hist_prices.iloc[i]['close']
                
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
        
        print(f"Model trained on {len(X)} samples from Alpha Vantage historical data")
        
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
    Generates trading signals using Alpha Vantage data
    Reused by paper and live trading
    """
    def __init__(self, ml_model: MLPredictor, 
                 feature_engine: FeatureEngine,
                 market_data: MarketDataManager,
                 options_data: OptionsDataManager):
        self.ml = ml_model
        self.features = feature_engine
        self.market = market_data  # IBKR for spot prices
        self.options = options_data  # Alpha Vantage for options
        
        self.signals_today = []
        self.last_signal_time = {}
        self.min_time_between_signals = 300  # 5 minutes
        
    async def generate_signals(self, symbols: List[str]) -> List[Dict]:
        """
        Generate signals for symbols using Alpha Vantage data
        Called by both paper and live traders
        """
        signals = []
        
        for symbol in symbols:
            # Check if enough time has passed
            if symbol in self.last_signal_time:
                time_since = (datetime.now() - self.last_signal_time[symbol]).seconds
                if time_since < self.min_time_between_signals:
                    continue
                    
            # Calculate features using Alpha Vantage APIs
            features = await self.features.calculate_features(symbol)
            
            # Get ML prediction
            signal_type, confidence = self.ml.predict(features)
            
            if signal_type != 'HOLD':
                # Find best option to trade using Alpha Vantage data
                option = await self._select_option_from_av(symbol, signal_type)
                
                if option:
                    signal = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'option': option,
                        'features': features.tolist(),
                        'av_greeks': option['greeks'],  # Greeks from Alpha Vantage
                    }
                    
                    signals.append(signal)
                    self.signals_today.append(signal)
                    self.last_signal_time[symbol] = datetime.now()
                    
        return signals
        
    async def _select_option_from_av(self, symbol: str, signal_type: str) -> Optional[Dict]:
        """Select best option contract using Alpha Vantage data"""
        # Get ATM options with Greeks from Alpha Vantage
        atm_options = self.options.find_atm_options(symbol, dte_min=0, dte_max=7)
        
        if not atm_options:
            return None
            
        # Select based on Greeks from Alpha Vantage
        if 'CALL' in signal_type:
            # Find call with best delta/theta ratio
            best_option = max(
                [opt for opt in atm_options if opt['type'] == 'CALL'],
                key=lambda x: x['greeks']['delta'] / abs(x['greeks']['theta']) 
                if x['greeks']['theta'] != 0 else 0
            )
        else:
            # Find put with best delta/theta ratio
            best_option = max(
                [opt for opt in atm_options if opt['type'] == 'PUT'],
                key=lambda x: abs(x['greeks']['delta']) / abs(x['greeks']['theta'])
                if x['greeks']['theta'] != 0 else 0
            )
            
        return {
            'strike': best_option['strike'],
            'expiry': best_option['expiry'],
            'type': best_option['type'],
            'contract': best_option['contract'],
            'greeks': best_option['greeks']  # Include AV Greeks
        }

# SIGNAL GENERATOR USING ALPHA VANTAGE DATA
signal_generator = SignalGenerator(ml_model, feature_engine, market_data, options_data)
```

#### Day 3-4: Risk Management

**Create**: `src/trading/risk.py`
```python
from typing import Dict, List, Tuple
import numpy as np

class RiskManager:
    """
    Risk management using Alpha Vantage Greeks
    Same rules for paper and live
    """
    def __init__(self, config: TradingConfig, 
                 options_data: OptionsDataManager,
                 db: DatabaseManager):
        self.config = config
        self.options = options_data  # Gets Greeks from Alpha Vantage
        self.db = db
        
        # Risk limits
        self.max_positions = config.max_positions
        self.max_position_size = config.max_position_size
        self.daily_loss_limit = config.daily_loss_limit
        
        # Greeks limits (using AV data)
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
        
    async def can_trade(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if trade is allowed using Alpha Vantage Greeks
        Used by paper and live equally
        """
        # Check position count
        if len(self.positions) >= self.max_positions:
            return False, "Max positions reached"
            
        # Check daily loss
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, "Daily loss limit reached"
            
        # Calculate position size
        position_size = await self._calculate_position_size(signal)
        if position_size > self.max_position_size:
            return False, f"Position size ${position_size} exceeds limit"
            
        # Check Greeks impact using Alpha Vantage data
        projected_greeks = signal['av_greeks']  # Greeks from signal (from AV)
        
        for greek, (min_val, max_val) in self.greeks_limits.items():
            current = self.portfolio_greeks.get(greek, 0)
            # Multiply by 5 contracts (standard size)
            new_value = current + (projected_greeks.get(greek, 0) * 5)
            
            if new_value < min_val or new_value > max_val:
                return False, f"Would breach {greek} limit: {new_value:.3f}"
                
        return True, "OK"
        
    async def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size in dollars"""
        # Get current option price from Alpha Vantage
        options = await self.options.av.get_realtime_options(signal['symbol'])
        
        # Find the specific option
        option = next(
            (opt for opt in options 
             if opt.strike == signal['option']['strike'] 
             and opt.option_type == signal['option']['type']),
            None
        )
        
        if option:
            # Use mid price
            option_price = (option.bid + option.ask) / 2
        else:
            option_price = 2.0  # Default estimate
            
        contracts = 5  # Standard position size
        return contracts * 100 * option_price
        
    def update_position(self, symbol: str, position: Dict):
        """Update position tracking with Alpha Vantage Greeks"""
        self.positions[symbol] = position
        asyncio.create_task(self._update_portfolio_greeks_from_av())
        
    async def _update_portfolio_greeks_from_av(self):
        """Update portfolio Greeks using fresh Alpha Vantage data"""
        total_greeks = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        for symbol, position in self.positions.items():
            # Get current Greeks from Alpha Vantage
            greeks = self.options.get_option_greeks(
                symbol,
                position['strike'],
                position['expiry'],
                position['option_type']
            )
            
            for key in total_greeks:
                total_greeks[key] += greeks.get(key, 0) * position['quantity']
                
        self.portfolio_greeks = total_greeks
        
        # Log to database
        with self.db.get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO portfolio_greeks_history 
                (timestamp, delta, gamma, theta, vega)
                VALUES (NOW(), %s, %s, %s, %s)
            """, (total_greeks['delta'], total_greeks['gamma'], 
                  total_greeks['theta'], total_greeks['vega']))
            conn.commit()

# RISK MANAGER USING ALPHA VANTAGE GREEKS
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
async def test_dual_data_sources():
    """Test IBKR and Alpha Vantage integration"""
    # Initialize components
    await market_data.connect()  # IBKR for quotes
    await av_client.connect()    # Alpha Vantage for options
    
    # Subscribe to market data
    await market_data.subscribe_symbols(['SPY'])
    
    # Wait for data
    await asyncio.sleep(10)
    
    # Verify IBKR data flowing
    assert market_data.get_latest_price('SPY') > 0
    
    # Test Alpha Vantage options data
    options = await av_client.get_realtime_options('SPY', require_greeks=True)
    assert len(options) > 0
    
    # Verify Greeks are provided, not calculated
    first_option = options[0]
    assert hasattr(first_option, 'delta')
    assert hasattr(first_option, 'gamma')
    assert hasattr(first_option, 'theta')
    assert hasattr(first_option, 'vega')
    assert first_option.delta is not None
    
@pytest.mark.asyncio
async def test_alpha_vantage_indicators():
    """Test Alpha Vantage technical indicators"""
    # Test multiple indicators in parallel
    tasks = [
        av_client.get_technical_indicator('SPY', 'RSI', interval='5min'),
        av_client.get_technical_indicator('SPY', 'MACD', interval='5min'),
        av_client.get_technical_indicator('SPY', 'BBANDS', interval='5min')
    ]
    
    results = await asyncio.gather(*tasks)
    
    for result in results:
        assert result is not None
        assert not result.empty
        
@pytest.mark.asyncio
async def test_signal_generation_with_av():
    """Test signal generation with Alpha Vantage data"""
    # Calculate features using AV APIs
    features = await feature_engine.calculate_features('SPY')
    assert len(features) == len(feature_engine.feature_names)
    
    # Get prediction
    signal, confidence = ml_model.predict(features)
    assert signal in ['BUY_CALL', 'BUY_PUT', 'HOLD', 'CLOSE']
    assert 0 <= confidence <= 1
    
@pytest.mark.asyncio
async def test_risk_with_av_greeks():
    """Test risk management with Alpha Vantage Greeks"""
    # Create signal with AV Greeks
    signal = {
        'symbol': 'SPY',
        'signal_type': 'BUY_CALL',
        'option': {
            'strike': 450,
            'expiry': '2024-01-19',
            'type': 'CALL'
        },
        'av_greeks': {
            'delta': 0.55,
            'gamma': 0.02,
            'theta': -0.15,
            'vega': 0.20
        }
    }
    
    can_trade, reason = await risk_manager.can_trade(signal)
    assert isinstance(can_trade, bool)
    assert isinstance(reason, str)
    
@pytest.mark.asyncio
async def test_rate_limiting():
    """Test Alpha Vantage rate limiting (600 calls/min)"""
    start = datetime.now()
    calls = 0
    
    # Try to make 100 calls quickly
    for _ in range(100):
        await av_client.get_realtime_options('SPY')
        calls += 1
        
    elapsed = (datetime.now() - start).seconds
    
    # Should handle rate limiting properly
    assert calls == 100
    print(f"Made {calls} calls in {elapsed} seconds")
```

#### Day 3-5: Performance Testing

**Create**: `tests/test_performance.py`
```python
import time
import numpy as np

async def test_av_api_performance():
    """Test Alpha Vantage API response times"""
    # Test cached vs uncached performance
    
    # First call - not cached
    start = time.time()
    options = await av_client.get_realtime_options('SPY', require_greeks=True)
    uncached_time = time.time() - start
    
    # Second call - should be cached
    start = time.time()
    options = await av_client.get_realtime_options('SPY', require_greeks=True)
    cached_time = time.time() - start
    
    print(f"Alpha Vantage Options API:")
    print(f"  Uncached: {uncached_time*1000:.2f}ms")
    print(f"  Cached: {cached_time*1000:.2f}ms")
    
    assert cached_time < uncached_time / 10  # Cache should be 10x faster
    assert cached_time < 0.01  # Cached should be under 10ms
    
async def test_parallel_av_calls():
    """Test parallel Alpha Vantage API calls"""
    start = time.time()
    
    # Make 10 different API calls in parallel
    tasks = [
        av_client.get_technical_indicator('SPY', 'RSI'),
        av_client.get_technical_indicator('SPY', 'MACD'),
        av_client.get_technical_indicator('QQQ', 'RSI'),
        av_client.get_technical_indicator('QQQ', 'MACD'),
        av_client.get_technical_indicator('IWM', 'RSI'),
        av_client.get_news_sentiment(['SPY', 'QQQ']),
        av_client.get_realtime_options('SPY'),
        av_client.get_realtime_options('QQQ'),
        av_client.get_realtime_options('IWM'),
        av_client.get_technical_indicator('SPY', 'BBANDS')
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"10 parallel Alpha Vantage calls: {elapsed*1000:.2f}ms")
    assert elapsed < 2.0  # Should complete in under 2 seconds
    assert all(r is not None for r in results)
    
def test_feature_calculation_with_av():
    """Test feature calculation speed with Alpha Vantage data"""
    start = time.time()
    
    # This now uses Alpha Vantage APIs
    features = asyncio.run(feature_engine.calculate_features('SPY'))
    
    elapsed = time.time() - start
    
    print(f"Feature calculation with AV APIs: {elapsed*1000:.2f}ms")
    assert elapsed < 1.0  # Should be under 1 second with caching
```

---

## PHASE 2: PAPER TRADING (Weeks 5-8)
*Everything builds on Phase 1 - no rewrites!*

### Week 5-6: Paper Trading Implementation

**Create**: `src/trading/paper_trader.py`
```python
class PaperTrader:
    """
    Paper trading using IBKR execution and Alpha Vantage data
    REUSES ALL COMPONENTS FROM PHASE 1
    """
    def __init__(self, 
                 signal_generator: SignalGenerator,
                 risk_manager: RiskManager,
                 market_data: MarketDataManager,
                 options_data: OptionsDataManager,
                 av_client: AlphaVantageClient,
                 db: DatabaseManager):
        # Reuse everything!
        self.signals = signal_generator
        self.risk = risk_manager
        self.market = market_data  # IBKR for execution
        self.options = options_data  # Alpha Vantage for options
        self.av = av_client  # Alpha Vantage for all analytics
        self.db = db
        
        # Paper trading specific
        self.starting_capital = 100000
        self.cash = self.starting_capital
        self.positions = {}
        self.trades = []
        
    async def run(self):
        """Main paper trading loop"""
        print("Starting paper trading...")
        print("Data sources: IBKR (quotes/execution), Alpha Vantage (options/analytics)")
        
        while True:
            # Market hours check
            if not self._is_market_open():
                await asyncio.sleep(60)
                continue
                
            # Generate signals using Alpha Vantage data
            signals = await self.signals.generate_signals(['SPY', 'QQQ', 'IWM'])
            
            for signal in signals:
                # Use existing risk manager (with AV Greeks)
                can_trade, reason = await self.risk.can_trade(signal)
                
                if can_trade:
                    await self.execute_paper_trade(signal)
                else:
                    print(f"Signal rejected: {reason}")
                    
            # Update positions with fresh AV Greeks
            await self.update_positions()
            
            # Monitor Alpha Vantage rate limit
            print(f"AV API calls remaining: {self.av.rate_limiter.remaining}/600")
            
            # Wait for next cycle
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def execute_paper_trade(self, signal: Dict):
        """Execute paper trade with IBKR paper account"""
        # Get option price from Alpha Vantage
        options = await self.av.get_realtime_options(signal['symbol'])
        option = next(
            (opt for opt in options 
             if opt.strike == signal['option']['strike']),
            None
        )
        
        if option:
            fill_price = (option.bid + option.ask) / 2
        else:
            fill_price = 2.0  # Default
            
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
            'pnl': 0,  # Updated later
            # Store Greeks from Alpha Vantage at entry
            'entry_delta': signal['av_greeks']['delta'],
            'entry_gamma': signal['av_greeks']['gamma'],
            'entry_theta': signal['av_greeks']['theta'],
            'entry_vega': signal['av_greeks']['vega'],
            'entry_iv': option.implied_volatility if option else 0.2
        }
        
        # Store in database
        with self.db.get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trades 
                (timestamp, mode, symbol, option_type, strike, expiry, 
                 action, quantity, price, commission, pnl,
                 entry_delta, entry_gamma, entry_theta, entry_vega, entry_iv)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, tuple(trade.values()))
            trade_id = cur.fetchone()[0]
            conn.commit()
            
        # Update position tracking
        self.positions[signal['symbol']] = trade
        self.risk.update_position(signal['symbol'], trade)
        
        print(f"Paper trade executed: {trade}")
        print(f"  Greeks (from AV): Δ={trade['entry_delta']:.3f}, "
              f"Γ={trade['entry_gamma']:.3f}, Θ={trade['entry_theta']:.3f}")
        
# PAPER TRADER REUSES EVERYTHING
paper_trader = PaperTrader(signal_generator, risk_manager, 
                          market_data, options_data, av_client, db)
```

### Week 7-8: Community Platform

**Create**: `src/community/discord_bot.py`
```python
import discord
from discord.ext import commands
import asyncio

class TradingBot(commands.Bot):
    """
    Discord bot - PUBLISHES PAPER TRADES WITH ALPHA VANTAGE DATA
    """
    def __init__(self, paper_trader: PaperTrader):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.paper = paper_trader  # Reuse paper trader!
        self.channels = {
            'signals': None,  # Set on ready
            'performance': None,
            'analytics': None  # For Alpha Vantage analytics
        }
        
    async def on_ready(self):
        print(f'Bot connected as {self.user}')
        
        # Get channels
        self.channels['signals'] = self.get_channel(YOUR_SIGNALS_CHANNEL_ID)
        self.channels['performance'] = self.get_channel(YOUR_PERFORMANCE_CHANNEL_ID)
        self.channels['analytics'] = self.get_channel(YOUR_ANALYTICS_CHANNEL_ID)
        
        # Start publishing loops
        self.loop.create_task(self.publish_trades())
        self.loop.create_task(self.publish_av_analytics())
        
    async def publish_trades(self):
        """Publish paper trades to Discord with Alpha Vantage Greeks"""
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
                # Format trade as embed with AV data
                embed = discord.Embed(
                    title=f"📈 Paper Trade: {trade['symbol']}",
                    color=discord.Color.green() if trade['action'] == 'BUY' else discord.Color.red()
                )
                
                embed.add_field(name="Action", value=trade['action'])
                embed.add_field(name="Option", value=f"{trade['option_type']} ${trade['strike']}")
                embed.add_field(name="Quantity", value=f"{trade['quantity']} contracts")
                embed.add_field(name="Price", value=f"${trade['price']:.2f}")
                
                # Add Greeks from Alpha Vantage
                embed.add_field(
                    name="Greeks (AV)", 
                    value=f"Δ={trade['entry_delta']:.3f} Γ={trade['entry_gamma']:.3f} "
                          f"Θ={trade['entry_theta']:.3f} IV={trade['entry_iv']:.1%}"
                )
                
                await self.channels['signals'].send(embed=embed)
                
                last_trade_id = trade['id']
                
            await asyncio.sleep(5)  # Check every 5 seconds
            
    async def publish_av_analytics(self):
        """Publish Alpha Vantage analytics periodically"""
        while True:
            # Get sentiment from Alpha Vantage
            sentiment = await self.paper.av.get_news_sentiment(['SPY', 'QQQ', 'IWM'])
            
            if sentiment and 'feed' in sentiment:
                embed = discord.Embed(
                    title="📰 Market Sentiment (Alpha Vantage)",
                    color=discord.Color.blue()
                )
                
                for article in sentiment['feed'][:3]:
                    embed.add_field(
                        name=article['title'][:100],
                        value=f"Sentiment: {article.get('overall_sentiment_score', 'N/A')}",
                        inline=False
                    )
                    
                await self.channels['analytics'].send(embed=embed)
                
            await asyncio.sleep(900)  # Every 15 minutes

# BOT REUSES PAPER TRADER WITH ALPHA VANTAGE DATA
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
    Live trading - IBKR execution with Alpha Vantage analytics
    REUSES PAPER TRADER LOGIC
    """
    def __init__(self, paper_trader: PaperTrader):
        # Reuse all paper trader components!
        self.paper = paper_trader
        
        # Just change execution mode
        self.market = paper_trader.market  # IBKR
        self.av = paper_trader.av  # Alpha Vantage
        self.positions = {}
        
    async def run(self):
        """Live trading - same as paper but real orders through IBKR"""
        print("Starting LIVE trading...")
        print("Execution: IBKR | Analytics: Alpha Vantage")
        
        while True:
            # Generate signals using Alpha Vantage data (SAME AS PAPER)
            signals = await self.paper.signals.generate_signals(['SPY', 'QQQ', 'IWM'])
            
            for signal in signals:
                # Risk check with AV Greeks (SAME AS PAPER)
                can_trade, reason = await self.paper.risk.can_trade(signal)
                
                if can_trade:
                    # Only difference: real order through IBKR
                    await self.execute_live_trade(signal)
                    
            # Monitor AV rate limit
            print(f"AV API usage: {600 - self.av.rate_limiter.remaining}/600 calls")
                    
            await asyncio.sleep(30)
            
    async def execute_live_trade(self, signal: Dict):
        """Execute real trade through IBKR with Alpha Vantage analytics"""
        from ib_insync import Option, MarketOrder
        
        # Create option contract for IBKR
        contract = Option(
            signal['symbol'], 
            signal['option']['expiry'].replace('-', ''),
            signal['option']['strike'],
            signal['option']['type'][0],  # 'C' or 'P'
            'SMART'
        )
        
        # Create order
        order = MarketOrder('BUY', 5)  # 5 contracts
        
        # Place order through IBKR
        trade = await self.market.execute_order(contract, order)
        
        # Wait for fill
        while not trade.isDone():
            await asyncio.sleep(0.1)
            
        # Store in database with Alpha Vantage Greeks
        with self.paper.db.get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trades 
                (timestamp, mode, symbol, option_type, strike, expiry,
                 action, quantity, price, commission, pnl,
                 entry_delta, entry_gamma, entry_theta, entry_vega, entry_iv)
                VALUES (NOW(), 'live', %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s)
            """, (
                signal['symbol'], signal['option']['type'],
                signal['option']['strike'], signal['option']['expiry'],
                'BUY', 5, trade.orderStatus.avgFillPrice, 0.65 * 5, 0,
                signal['av_greeks']['delta'], signal['av_greeks']['gamma'],
                signal['av_greeks']['theta'], signal['av_greeks']['vega'],
                0.20  # IV placeholder
            ))
            conn.commit()

# LIVE TRADER REUSES PAPER TRADER WITH DUAL DATA SOURCES
live_trader = LiveTrader(paper_trader)
```

---

## PHASE 4: OPTIMIZATION (Weeks 13-16)
*Performance tuning and feature additions*

### Week 13-14: Performance Optimization
- Optimize Alpha Vantage API calls with better caching
- Implement predictive cache warming for frequently used data
- Add Redis clustering for cache scaling
- Parallelize API calls where possible (respecting 600/min limit)

### Week 15-16: Advanced Features
- Add spread strategies using Alpha Vantage options chains
- Implement complex Greeks analysis using AV historical data
- Add more symbols (scale gradually with AV rate limits)
- Enhanced ML models trained on 20 years of AV historical options
- Advanced sentiment analysis using all AV news/social APIs

---

## CODE REUSE SUMMARY

```
Component           | Built In | Data Source      | Reused By                    
--------------------|----------|------------------|------------------------------
MarketDataManager   | Week 1   | IBKR             | Everything (spot prices)     
AlphaVantageClient  | Week 1   | Alpha Vantage    | Everything (options/analytics)
OptionsDataManager  | Week 1   | Alpha Vantage    | ML, Risk, Trading, Community
DatabaseManager     | Week 1   | Local            | All components              
FeatureEngine      | Week 2   | Alpha Vantage    | ML, Paper, Live             
MLPredictor        | Week 2   | Alpha Vantage    | Signals, Paper, Live        
SignalGenerator    | Week 3   | Both sources     | Paper, Live, Community      
RiskManager        | Week 3   | Alpha Vantage    | Paper, Live                 
PaperTrader        | Week 5   | Both sources     | Live (reuses 90%), Community
DiscordBot         | Week 7   | Alpha Vantage    | Paper and Live              
LiveTrader         | Week 9   | Both sources     | Extends PaperTrader         
```

## DATA SOURCE ARCHITECTURE

```
IBKR (Interactive Brokers):
- Real-time market quotes
- 5-second bars
- Trade execution (paper & live)
- Order management

Alpha Vantage (600 calls/min):
- Real-time options chains with Greeks
- 20 years historical options with Greeks
- All technical indicators (RSI, MACD, etc.)
- News sentiment & social sentiment
- Fundamental data
- Economic indicators
- Market analytics
```

## SUCCESS METRICS BY WEEK

**Week 4**: System generates signals using Alpha Vantage data
**Week 6**: Paper trading with real-time AV Greeks monitoring
**Week 8**: Discord bot publishing trades with AV analytics
**Week 10**: First live trades with dual data sources
**Week 12**: System profitable using AV's comprehensive data
**Week 14**: Optimized to <100ms with smart caching
**Week 16**: Full production leveraging all 38 AV APIs

---

This plan shows EXACTLY how Alpha Vantage provides options data with Greeks (no calculation needed) while IBKR handles execution. No rewrites, no wasted effort - just progressive enhancement with proper data source architecture.