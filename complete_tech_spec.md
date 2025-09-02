# Real-Time Institutional Options Analytics System
## Complete Technical Specification v2.0 - Redis-Centric Architecture

---

## Executive Summary

A high-performance, Redis-centric options analytics and automated trading system that eliminates module interdependencies through a centralized state store. Each component operates independently, reading from and writing to Redis, creating a robust, scalable architecture that combines IBKR Level 2 order book data with Alpha Vantage options analytics to generate institutional-grade trading signals and execute trades automatically.

### Architecture Revolution
- **Zero module-to-module communication** - Everything flows through Redis
- **Natural fault isolation** - Modules can fail/restart independently
- **Simplified debugging** - All state visible in Redis CLI
- **Automatic backpressure** - Queues naturally form in Redis
- **10x easier integration** - Each module only knows Redis keys

---

## 1. System Architecture

### 1.1 Complete Redis-Centric Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (Dual-Feed)                     │
├──────────────────────────┬──────────────────────────────────────┤
│    IBKR WebSocket        │      Alpha Vantage REST API          │
│    ├─ Level 2 Book       │      ├─ Options Chains w/Greeks      │
│    ├─ Trade Tape         │      ├─ Technical Indicators         │
│    ├─ 5-sec Bars         │      ├─ Sentiment Analysis          │
│    └─ Execution Status   │      └─ Fundamentals                │
└────────────┬─────────────┴────────────┬────────────────────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA INGESTION MODULES                         │
│                   (Write to Redis Only)                          │
├──────────────────────────┬──────────────────────────────────────┤
│    IBKR Writer           │      AV Writer                       │
│    Writes to Redis:      │      Writes to Redis:                │
│    market:SPY:book       │      options:SPY:chain              │
│    market:SPY:trades     │      options:SPY:greeks             │
│    market:QQQ:book       │      sentiment:SPY:score            │
│    market:QQQ:trades     │      technicals:SPY:rsi             │
│    (for all symbols)     │      (for all symbols)               │
└──────────────────────────┴──────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REDIS (Central State Store)                   │
└─────────────────────────────────────────────────────────────────┘
        ▲       ▲       ▲       ▲       ▲       ▲       ▲
        │       │       │       │       │       │       │
        ▼       ▼       ▼       ▼       ▼       ▼       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING MODULES                            │
│                 (All Read/Write to Redis)                        │
├───────────┬───────────┬───────────┬───────────┬─────────────────┤
│ PARAMETER │ ANALYTICS │  SIGNAL   │ EXECUTION │   POSITION      │
│ DISCOVERY │  ENGINE   │ GENERATOR │  MANAGER  │   MANAGER       │
├───────────┼───────────┼───────────┼───────────┼─────────────────┤
│   RISK    │  CIRCUIT  │ EMERGENCY │  SIGNAL   │   DISCORD       │
│  MANAGER  │  BREAKERS │  MANAGER  │DISTRIBUTOR│     BOT         │
├───────────┼───────────┼───────────┼───────────┼─────────────────┤
│ WEBSOCKET │  REST API │ DASHBOARD │  METRICS  │   LOGGING       │
│  SERVER   │  SERVER   │  (WEB UI) │ COLLECTOR │  AGGREGATOR     │
└───────────┴───────────┴───────────┴───────────┴─────────────────┘
```

---

## 2. Redis Schema

### 2.1 Complete Key Structure

```python
REDIS_SCHEMA = {
    # Market Data (1 second TTL)
    'market:{symbol}:book': 'Full Level 2 order book JSON',
    'market:{symbol}:trades': 'List of recent trades',
    'market:{symbol}:last': 'Last trade price',
    'market:{symbol}:bars': 'Recent 5-second OHLCV bars',
    'market:{symbol}:timestamp': 'Last update epoch milliseconds',
    
    # Options Data (10 second TTL)
    'options:{symbol}:chain': 'Full options chain',
    'options:{symbol}:greeks': 'Greeks by strike/expiry',
    'options:{symbol}:flow': 'Recent options flow',
    'options:{symbol}:unusual': 'Unusual activity detected',
    
    # Discovered Parameters (24 hour TTL)
    'discovered:vpin_bucket_size': 'Empirically discovered bucket size',
    'discovered:lookback_bars': 'Optimal lookback window',
    'discovered:mm_profiles': 'Market maker patterns JSON',
    'discovered:vol_regimes': 'Volatility regime parameters',
    'discovered:correlation_matrix': 'Symbol correlation matrix',
    
    # Calculated Metrics (5 second TTL)
    'metrics:{symbol}:vpin': 'VPIN toxicity score',
    'metrics:{symbol}:obi': 'Order book imbalance JSON',
    'metrics:{symbol}:gex': 'Gamma exposure JSON',
    'metrics:{symbol}:dex': 'Delta exposure JSON',
    'metrics:{symbol}:sweep': 'Sweep detection JSON',
    'metrics:{symbol}:hidden': 'Hidden orders detected boolean',
    'metrics:{symbol}:regime': 'Current market regime',
    
    # Signals (60 second TTL)
    'signals:{symbol}:pending': 'Queue of pending signals',
    'signals:{symbol}:active': 'Currently active signals',
    'signals:global:count': 'Total signals generated today',
    
    # Positions (No TTL - persistent)
    'positions:{symbol}:{id}': 'Individual position data',
    'positions:summary': 'All positions summary',
    'orders:pending:{id}': 'Pending orders',
    'orders:working:{id}': 'Working orders at IBKR',
    
    # Global State (No TTL)
    'global:buying_power': 'Account buying power',
    'global:positions:count': 'Current position count',
    'global:pnl:realized': 'Realized P&L today',
    'global:pnl:unrealized': 'Unrealized P&L',
    'global:risk:correlation': 'Position correlation',
    'global:risk:var': 'Value at Risk',
    'global:halt': 'Trading halted flag',
    
    # Distribution (Variable TTL)
    'distribution:basic:queue': 'Basic tier queue (60s delay)',
    'distribution:premium:queue': 'Premium tier queue (realtime)',
    'distribution:signals:count': 'Signals distributed today',
    
    # Monitoring (5 second TTL)
    'monitoring:latency:*': 'Latency metrics by module',
    'monitoring:api:av:calls': 'Alpha Vantage API usage',
    'monitoring:api:ibkr:messages': 'IBKR message rate',
    'monitoring:errors:count': 'Error count by module',
    'monitoring:health:*': 'Module health status',
    
    # Social Media Integration (Variable TTL)
    'twitter:tweet:{id}': 'Tweet tracking (24h TTL)',
    'twitter:posted_signals': 'Set of posted signal hashes',
    'telegram:user:{id}:tier': 'Telegram user subscription tier',
    'telegram:signals:{symbol}:queue': 'Telegram signal distribution queue',
    'telegram:analysis:queue': 'Morning analysis queue for Telegram',
    
    # Morning Analysis (24 hour TTL)
    'analysis:morning:full': 'Complete morning analysis JSON',
    'analysis:morning:preview': 'Public preview of morning analysis',
    'dashboard:morning_analysis': 'Analysis for web dashboard',
    
    # Social Media Distribution
    'distribution:premium:analysis': 'Premium analysis distribution queue',
    'status:morning_analysis': 'Morning analysis completion status'
}
```

---

## 3. Data Ingestion Modules

### 3.1 IBKR WebSocket Ingestion

```python
import asyncio
import json
import time
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import redis
from typing import Dict, List

class IBKRIngestion:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ib = IB()
        self.symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'PLTR', 'VXX']
        self.contracts = {}
        self.order_books = {}
        
    async def start(self):
        # Connect to IBKR
        await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        
        # Subscribe to all symbols
        for symbol in self.symbols:
            contract = Stock(symbol, 'SMART', 'USD')
            self.contracts[symbol] = contract
            
            # Request Level 2 market depth
            self.ib.reqMktDepth(contract, numRows=10, isSmartDepth=True)
            self.ib.reqMktDepthExchanges()
            
            # Request market data
            self.ib.reqMktData(contract, '', False, False)
            
            # Request 5-second bars
            self.ib.reqRealTimeBars(contract, 5, 'TRADES', False)
            
            # Initialize order book
            self.order_books[symbol] = {'bids': [], 'asks': []}
        
        # Set up event handlers
        self.ib.updateMktDepthEvent += self.on_depth_update
        self.ib.pendingTickersEvent += self.on_ticker_update
        self.ib.barUpdateEvent += self.on_bar_update
        
        # Keep running
        while True:
            await asyncio.sleep(0.01)
    
    def on_depth_update(self, depth):
        symbol = depth.contract.symbol
        
        # Update local order book
        if depth.side == 0:  # Bid
            self.update_book_level(symbol, 'bids', depth)
        else:  # Ask
            self.update_book_level(symbol, 'asks', depth)
        
        # Write to Redis
        book_json = json.dumps(self.order_books[symbol])
        pipe = self.redis.pipeline()
        pipe.setex(f'market:{symbol}:book', 1, book_json)
        pipe.setex(f'market:{symbol}:timestamp', 1, int(time.time() * 1000))
        pipe.execute()
    
    def update_book_level(self, symbol: str, side: str, depth):
        levels = self.order_books[symbol][side]
        
        # Find or create level
        level_data = {
            'price': depth.price,
            'size': depth.size,
            'market_maker': depth.marketMaker,
            'position': depth.position
        }
        
        if depth.operation == 0:  # Insert
            if depth.position < len(levels):
                levels.insert(depth.position, level_data)
            else:
                levels.append(level_data)
        elif depth.operation == 1:  # Update
            if depth.position < len(levels):
                levels[depth.position] = level_data
        elif depth.operation == 2:  # Delete
            if depth.position < len(levels):
                del levels[depth.position]
    
    def on_ticker_update(self, tickers):
        for ticker in tickers:
            if ticker.contract.symbol in self.symbols:
                symbol = ticker.contract.symbol
                
                # Store last price
                if ticker.last:
                    self.redis.setex(f'market:{symbol}:last', 1, ticker.last)
                
                # Store trade data
                if ticker.lastSize:
                    trade = {
                        'price': ticker.last,
                        'size': ticker.lastSize,
                        'time': time.time(),
                        'bid': ticker.bid,
                        'ask': ticker.ask
                    }
                    self.redis.lpush(f'market:{symbol}:trades', json.dumps(trade))
                    self.redis.ltrim(f'market:{symbol}:trades', 0, 999)
                    self.redis.expire(f'market:{symbol}:trades', 1)
    
    def on_bar_update(self, bars, hasNewBar):
        if hasNewBar:
            for bar in bars:
                if bar.contract.symbol in self.symbols:
                    symbol = bar.contract.symbol
                    bar_data = {
                        'time': bar.time.timestamp(),
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    
                    # Get existing bars
                    bars_json = self.redis.get(f'market:{symbol}:bars')
                    if bars_json:
                        bars_list = json.loads(bars_json)
                    else:
                        bars_list = []
                    
                    # Add new bar and keep last 100
                    bars_list.append(bar_data)
                    bars_list = bars_list[-100:]
                    
                    self.redis.setex(f'market:{symbol}:bars', 10, json.dumps(bars_list))
```

### 3.2 Alpha Vantage Ingestion

```python
import aiohttp
import asyncio
import json
import time
from collections import deque

class AlphaVantageIngestion:
    def __init__(self, api_key: str, redis_host='localhost', redis_port=6379):
        self.api_key = api_key
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.base_url = 'https://www.alphavantage.co/query'
        self.symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'PLTR', 'VXX']
        self.call_times = deque(maxlen=600)  # Track last 600 calls for rate limiting
        
    async def start(self):
        async with aiohttp.ClientSession() as session:
            while True:
                for symbol in self.symbols:
                    await self.fetch_symbol_data(session, symbol)
                    await self.rate_limit_check()
                
                await asyncio.sleep(10)  # Full cycle every 10 seconds
    
    async def rate_limit_check(self):
        now = time.time()
        self.call_times.append(now)
        
        # Check if we've made 600 calls in the last minute
        minute_ago = now - 60
        recent_calls = sum(1 for t in self.call_times if t > minute_ago)
        
        if recent_calls >= 590:  # Leave buffer
            sleep_time = 61 - (now - self.call_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def fetch_symbol_data(self, session: aiohttp.ClientSession, symbol: str):
        # Fetch options chain with Greeks
        options_url = f"{self.base_url}?function=REALTIME_OPTIONS&symbol={symbol}&apikey={self.api_key}"
        
        async with session.get(options_url) as response:
            if response.status == 200:
                data = await response.json()
                
                # Parse options chain
                options_chain = []
                greeks = {}
                
                for contract in data.get('options', []):
                    # Store full contract
                    options_chain.append(contract)
                    
                    # Extract Greeks (PROVIDED by Alpha Vantage, not calculated)
                    key = f"{contract['strike']}_{contract['expiration']}_{contract['type']}"
                    greeks[key] = {
                        'delta': contract.get('delta', 0),
                        'gamma': contract.get('gamma', 0),
                        'theta': contract.get('theta', 0),
                        'vega': contract.get('vega', 0),
                        'rho': contract.get('rho', 0),
                        'iv': contract.get('implied_volatility', 0),
                        'open_interest': contract.get('open_interest', 0),
                        'volume': contract.get('volume', 0)
                    }
                
                # Store in Redis
                pipe = self.redis.pipeline()
                pipe.setex(f'options:{symbol}:chain', 10, json.dumps(options_chain))
                pipe.setex(f'options:{symbol}:greeks', 10, json.dumps(greeks))
                
                # Detect unusual activity
                unusual = self.detect_unusual_activity(options_chain)
                if unusual:
                    pipe.setex(f'options:{symbol}:unusual', 10, json.dumps(unusual))
                
                pipe.execute()
        
        # Fetch sentiment
        await self.fetch_sentiment(session, symbol)
    
    async def fetch_sentiment(self, session: aiohttp.ClientSession, symbol: str):
        sentiment_url = f"{self.base_url}?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
        
        async with session.get(sentiment_url) as response:
            if response.status == 200:
                data = await response.json()
                
                # Calculate aggregate sentiment
                sentiment_scores = []
                for article in data.get('feed', []):
                    ticker_sentiment = article.get('ticker_sentiment', [])
                    for ts in ticker_sentiment:
                        if ts['ticker'] == symbol:
                            sentiment_scores.append(float(ts['ticker_sentiment_score']))
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    self.redis.setex(f'sentiment:{symbol}:score', 300, json.dumps({
                        'score': avg_sentiment,
                        'count': len(sentiment_scores),
                        'timestamp': time.time()
                    }))
    
    def detect_unusual_activity(self, options_chain: list) -> dict:
        unusual = []
        
        for contract in options_chain:
            volume = contract.get('volume', 0)
            open_interest = contract.get('open_interest', 0)
            
            # Unusual if volume > 2x open interest
            if open_interest > 0 and volume > open_interest * 2:
                unusual.append({
                    'strike': contract['strike'],
                    'expiration': contract['expiration'],
                    'type': contract['type'],
                    'volume': volume,
                    'open_interest': open_interest,
                    'ratio': volume / open_interest
                })
        
        if unusual:
            # Sort by ratio
            unusual.sort(key=lambda x: x['ratio'], reverse=True)
            return {'detected': True, 'contracts': unusual[:10]}  # Top 10
        
        return {'detected': False}
```

---

## 4. Parameter Discovery Module

```python
import numpy as np
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import acf
from collections import defaultdict
import json
import yaml

class ParameterDiscovery:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
    async def run_discovery(self):
        """Run complete parameter discovery pipeline"""
        
        print("Starting parameter discovery...")
        
        # 1. Discover VPIN bucket size from actual trade volumes
        vpin_bucket = self.discover_vpin_bucket_size()
        self.redis.setex('discovered:vpin_bucket_size', 86400, vpin_bucket)
        print(f"Discovered VPIN bucket size: {vpin_bucket} shares")
        
        # 2. Discover temporal structure
        lookback = self.discover_temporal_structure()
        self.redis.setex('discovered:lookback_bars', 86400, lookback)
        print(f"Discovered lookback: {lookback} bars")
        
        # 3. Analyze market makers
        mm_profiles = self.analyze_market_makers()
        self.redis.setex('discovered:mm_profiles', 86400, json.dumps(mm_profiles))
        print(f"Discovered {len(mm_profiles)} market makers")
        
        # 4. Detect volatility regimes
        vol_regimes = self.discover_volatility_regimes()
        self.redis.setex('discovered:vol_regimes', 86400, json.dumps(vol_regimes))
        print(f"Current volatility regime: {vol_regimes['current']}")
        
        # 5. Calculate correlations
        correlations = self.calculate_correlations()
        self.redis.setex('discovered:correlation_matrix', 86400, json.dumps(correlations))
        
        # 6. Generate config file
        self.generate_config_file()
        
        print("Parameter discovery complete!")
    
    def discover_vpin_bucket_size(self) -> int:
        """Discover natural trade size clustering"""
        
        # Get recent trades
        trades_json = self.redis.lrange('market:SPY:trades', 0, 9999)
        if len(trades_json) < 1000:
            return 100  # Default
        
        trades = [json.loads(t) for t in trades_json]
        sizes = [t['size'] for t in trades]
        
        # Find natural clustering
        sizes_array = np.array(sizes).reshape(-1, 1)
        
        # Use percentiles first
        percentiles = np.percentile(sizes, [25, 50, 75])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(sizes_array)
        
        # Find median cluster
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        optimal_bucket = int(cluster_centers[2])  # Middle cluster
        
        # Ensure reasonable size
        optimal_bucket = max(50, min(optimal_bucket, 500))
        
        return optimal_bucket
    
    def discover_temporal_structure(self) -> int:
        """Find how many bars back have predictive power"""
        
        bars_json = self.redis.get('market:SPY:bars')
        if not bars_json:
            return 12  # Default
        
        bars = json.loads(bars_json)
        if len(bars) < 100:
            return 12
        
        # Calculate returns
        prices = [b['close'] for b in bars]
        returns = np.diff(np.log(prices))
        
        # Calculate autocorrelation
        autocorr = acf(returns, nlags=min(50, len(returns)//4), fft=True)
        
        # Find significant lags
        significance_threshold = 2.0 / np.sqrt(len(returns))
        significant_lags = []
        
        for i, corr in enumerate(autocorr[1:], 1):
            if abs(corr) > significance_threshold:
                significant_lags.append(i)
        
        if significant_lags:
            optimal_lookback = max(significant_lags)
        else:
            optimal_lookback = 12
        
        return min(optimal_lookback, 30)  # Cap at 30 bars
    
    def analyze_market_makers(self) -> dict:
        """Profile market makers from Level 2 data"""
        
        mm_stats = defaultdict(lambda: {
            'order_count': 0,
            'total_size': 0,
            'bid_count': 0,
            'ask_count': 0,
            'prices': []
        })
        
        # Analyze order books for all symbols
        for symbol in ['SPY', 'QQQ', 'IWM']:
            book_json = self.redis.get(f'market:{symbol}:book')
            if not book_json:
                continue
            
            book = json.loads(book_json)
            
            # Count market maker activity
            for level in book.get('bids', []):
                mm = level.get('market_maker', 'UNKNOWN')
                mm_stats[mm]['order_count'] += 1
                mm_stats[mm]['total_size'] += level['size']
                mm_stats[mm]['bid_count'] += 1
                mm_stats[mm]['prices'].append(level['price'])
            
            for level in book.get('asks', []):
                mm = level.get('market_maker', 'UNKNOWN')
                mm_stats[mm]['order_count'] += 1
                mm_stats[mm]['total_size'] += level['size']
                mm_stats[mm]['ask_count'] += 1
                mm_stats[mm]['prices'].append(level['price'])
        
        # Calculate profiles
        profiles = {}
        total_orders = sum(s['order_count'] for s in mm_stats.values())
        
        for mm_id, stats in mm_stats.items():
            if stats['order_count'] > 0:
                avg_size = stats['total_size'] / stats['order_count']
                
                # Calculate toxicity heuristically
                toxicity = 0.0
                
                # Small size indicates potential HFT
                if avg_size < 100:
                    toxicity += 0.3
                
                # Known HFT firms
                if mm_id in ['CDRG', 'JANE', 'VIRTU', 'SUSG']:
                    toxicity += 0.4
                
                # Balanced bid/ask indicates market making
                bid_ask_ratio = stats['bid_count'] / (stats['ask_count'] + 1)
                if 0.8 < bid_ask_ratio < 1.2:
                    toxicity -= 0.1
                
                toxicity = max(0, min(toxicity, 1.0))
                
                profiles[mm_id] = {
                    'frequency': stats['order_count'] / total_orders if total_orders > 0 else 0,
                    'avg_size': avg_size,
                    'toxicity': toxicity,
                    'classification': 'HFT' if toxicity > 0.5 else 'INSTITUTIONAL'
                }
        
        return profiles
    
    def discover_volatility_regimes(self) -> dict:
        """Identify current volatility regime"""
        
        bars_json = self.redis.get('market:SPY:bars')
        if not bars_json:
            return {'current': 'NORMAL', 'threshold_low': 0.10, 'threshold_high': 0.25}
        
        bars = json.loads(bars_json)
        if len(bars) < 20:
            return {'current': 'NORMAL', 'threshold_low': 0.10, 'threshold_high': 0.25}
        
        # Calculate current volatility
        recent_prices = [b['close'] for b in bars[-20:]]
        recent_returns = np.diff(np.log(recent_prices))
        current_vol = np.std(recent_returns) * np.sqrt(252 * 78)  # Annualized
        
        # Calculate historical volatility distribution
        vol_history = []
        for i in range(0, len(bars) - 20, 5):
            window_prices = [b['close'] for b in bars[i:i+20]]
            if len(window_prices) > 1:
                window_returns = np.diff(np.log(window_prices))
                window_vol = np.std(window_returns) * np.sqrt(252 * 78)
                vol_history.append(window_vol)
        
        if vol_history:
            p33 = np.percentile(vol_history, 33)
            p67 = np.percentile(vol_history, 67)
            
            if current_vol < p33:
                regime = 'LOW'
            elif current_vol > p67:
                regime = 'HIGH'
            else:
                regime = 'NORMAL'
            
            return {
                'current': regime,
                'threshold_low': p33,
                'threshold_high': p67,
                'current_vol': current_vol
            }
        
        return {
            'current': 'NORMAL',
            'threshold_low': 0.10,
            'threshold_high': 0.25,
            'current_vol': current_vol
        }
    
    def calculate_correlations(self) -> dict:
        """Calculate correlation matrix between symbols"""
        
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA']
        correlations = {}
        
        for symbol1 in symbols:
            correlations[symbol1] = {}
            
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlations[symbol1][symbol2] = 1.0
                else:
                    # Get price data
                    bars1_json = self.redis.get(f'market:{symbol1}:bars')
                    bars2_json = self.redis.get(f'market:{symbol2}:bars')
                    
                    if bars1_json and bars2_json:
                        bars1 = json.loads(bars1_json)
                        bars2 = json.loads(bars2_json)
                        
                        if len(bars1) > 10 and len(bars2) > 10:
                            prices1 = [b['close'] for b in bars1[-50:]]
                            prices2 = [b['close'] for b in bars2[-50:]]
                            
                            min_len = min(len(prices1), len(prices2))
                            if min_len > 2:
                                returns1 = np.diff(np.log(prices1[:min_len]))
                                returns2 = np.diff(np.log(prices2[:min_len]))
                                
                                if len(returns1) > 0 and len(returns2) > 0:
                                    corr = np.corrcoef(returns1, returns2)[0, 1]
                                    correlations[symbol1][symbol2] = round(corr, 3)
                                else:
                                    correlations[symbol1][symbol2] = 0.0
                            else:
                                correlations[symbol1][symbol2] = 0.0
                        else:
                            correlations[symbol1][symbol2] = 0.0
                    else:
                        correlations[symbol1][symbol2] = 0.0
        
        return correlations
    
    def generate_config_file(self):
        """Generate discovered parameters config file"""
        
        config = {
            'discovered': {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'parameters': {
                    'vpin_bucket_size': int(self.redis.get('discovered:vpin_bucket_size') or 100),
                    'lookback_bars': int(self.redis.get('discovered:lookback_bars') or 12),
                },
                'market_makers': json.loads(self.redis.get('discovered:mm_profiles') or '{}'),
                'volatility': json.loads(self.redis.get('discovered:vol_regimes') or '{}'),
                'correlations': json.loads(self.redis.get('discovered:correlation_matrix') or '{}')
            }
        }
        
        # Save to file
        with open('config/discovered.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
```

---

## 5. Analytics Engine

```python
import numpy as np
from collections import defaultdict
import json
import time

class AnalyticsEngine:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'PLTR', 'VXX']
        
    async def start(self):
        """Main analytics processing loop"""
        
        while True:
            for symbol in self.symbols:
                # Check if new data available
                last_processed = self.redis.get(f'analytics:{symbol}:timestamp')
                current = self.redis.get(f'market:{symbol}:timestamp')
                
                if current and (not last_processed or float(current) > float(last_processed)):
                    self.process_symbol(symbol)
            
            await asyncio.sleep(0.1)  # Process at 10Hz
    
    def process_symbol(self, symbol: str):
        """Calculate all metrics for one symbol"""
        
        # Load discovered parameters
        vpin_bucket = int(self.redis.get('discovered:vpin_bucket_size') or 100)
        lookback = int(self.redis.get('discovered:lookback_bars') or 12)
        mm_profiles = json.loads(self.redis.get('discovered:mm_profiles') or '{}')
        
        # Get market data
        book_json = self.redis.get(f'market:{symbol}:book')
        trades_json = self.redis.lrange(f'market:{symbol}:trades', 0, 999)
        last_price = float(self.redis.get(f'market:{symbol}:last') or 0)
        
        # Get options data
        greeks_json = self.redis.get(f'options:{symbol}:greeks')
        chain_json = self.redis.get(f'options:{symbol}:chain')
        
        # Parse data
        book = json.loads(book_json) if book_json else {}
        trades = [json.loads(t) for t in trades_json]
        greeks = json.loads(greeks_json) if greeks_json else {}
        chain = json.loads(chain_json) if chain_json else []
        
        # Calculate all metrics
        metrics = {}
        
        # 1. Enhanced VPIN
        metrics['vpin'] = self.calculate_enhanced_vpin(trades, vpin_bucket, mm_profiles)
        
        # 2. Order Book Imbalance
        metrics['obi'] = self.calculate_order_book_imbalance(book)
        
        # 3. Hidden Orders
        metrics['hidden'] = self.detect_hidden_orders(book, trades)
        
        # 4. Gamma Exposure
        metrics['gex'] = self.calculate_gamma_exposure(greeks, chain, last_price)
        
        # 5. Delta Exposure
        metrics['dex'] = self.calculate_delta_exposure(greeks, chain, last_price)
        
        # 6. Sweep Detection
        metrics['sweep'] = self.detect_sweeps(trades)
        
        # 7. Market Regime
        vol_regimes = json.loads(self.redis.get('discovered:vol_regimes') or '{}')
        metrics['regime'] = vol_regimes.get('current', 'NORMAL')
        
        # Store all metrics
        pipe = self.redis.pipeline()
        for metric_name, value in metrics.items():
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            elif not isinstance(value, str):
                value = json.dumps(value)
            pipe.setex(f'metrics:{symbol}:{metric_name}', 5, value)
        pipe.setex(f'analytics:{symbol}:timestamp', 5, time.time())
        pipe.execute()
    
    def calculate_enhanced_vpin(self, trades: list, bucket_size: int, mm_profiles: dict) -> float:
        """VPIN with market maker toxicity adjustment"""
        
        if len(trades) < 10:
            return 0.0
        
        buckets = []
        current_bucket = {'buy': 0, 'sell': 0, 'toxic': 0}
        
        for i, trade in enumerate(trades):
            # Classify trade direction
            if 'bid' in trade and 'ask' in trade:
                if trade['price'] >= trade['ask']:
                    current_bucket['buy'] += trade['size']
                elif trade['price'] <= trade['bid']:
                    current_bucket['sell'] += trade['size']
                else:
                    # Tick test
                    if i > 0 and trade['price'] > trades[i-1]['price']:
                        current_bucket['buy'] += trade['size']
                    else:
                        current_bucket['sell'] += trade['size']
            else:
                # No bid/ask, use tick test
                if i > 0 and trade['price'] > trades[i-1]['price']:
                    current_bucket['buy'] += trade['size']
                else:
                    current_bucket['sell'] += trade['size']
            
            # Add toxicity
            mm_id = trade.get('market_maker', 'UNKNOWN')
            if mm_id in mm_profiles:
                toxicity = mm_profiles[mm_id].get('toxicity', 0)
                current_bucket['toxic'] += trade['size'] * toxicity
            
            # Check if bucket full
            if current_bucket['buy'] + current_bucket['sell'] >= bucket_size:
                buckets.append(current_bucket)
                current_bucket = {'buy': 0, 'sell': 0, 'toxic': 0}
        
        # Calculate VPIN
        if not buckets:
            return 0.0
        
        vpin_values = []
        for bucket in buckets:
            total = bucket['buy'] + bucket['sell']
            if total > 0:
                base_vpin = abs(bucket['buy'] - bucket['sell']) / total
                toxicity_adjustment = bucket['toxic'] / total
                enhanced_vpin = base_vpin * (1 + toxicity_adjustment * 0.3)
                vpin_values.append(min(enhanced_vpin, 1.0))
        
        return np.mean(vpin_values)
    
    def calculate_order_book_imbalance(self, book: dict) -> dict:
        """Calculate multi-factor order book imbalance"""
        
        if not book or 'bids' not in book or 'asks' not in book:
            return {'volume': 0, 'pressure': 0, 'slope': 0}
        
        bids = book['bids'][:10]  # Top 10 levels
        asks = book['asks'][:10]
        
        if not bids or not asks:
            return {'volume': 0, 'pressure': 0, 'slope': 0}
        
        # Volume imbalance
        bid_volume = sum(level['size'] for level in bids)
        ask_volume = sum(level['size'] for level in asks)
        
        if bid_volume + ask_volume == 0:
            return {'volume': 0, 'pressure': 0, 'slope': 0}
        
        volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Weighted price pressure
        if bid_volume > 0:
            bid_pressure = sum(level['price'] * level['size'] for level in bids) / bid_volume
        else:
            bid_pressure = bids[0]['price'] if bids else 0
        
        if ask_volume > 0:
            ask_pressure = sum(level['price'] * level['size'] for level in asks) / ask_volume
        else:
            ask_pressure = asks[0]['price'] if asks else 0
        
        if bid_pressure + ask_pressure > 0:
            pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
        else:
            pressure_imbalance = 0
        
        # Book slope (depth decay)
        bid_slope = self.calculate_book_slope(bids)
        ask_slope = self.calculate_book_slope(asks)
        slope_ratio = bid_slope / ask_slope if ask_slope != 0 else 1.0
        
        return {
            'volume': round(volume_imbalance, 4),
            'pressure': round(pressure_imbalance, 4),
            'slope': round(slope_ratio, 4)
        }
    
    def calculate_book_slope(self, levels: list) -> float:
        """Calculate order book depth decay"""
        
        if len(levels) < 2:
            return 0.0
        
        sizes = [level['size'] for level in levels]
        positions = list(range(len(sizes)))
        
        # Linear regression for slope
        if len(sizes) > 1:
            z = np.polyfit(positions, sizes, 1)
            return abs(z[0])  # Return absolute slope
        
        return 0.0
    
    def detect_hidden_orders(self, book: dict, trades: list) -> bool:
        """Detect iceberg orders and hidden liquidity"""
        
        if not book or not trades:
            return False
        
        # Check for trades larger than displayed size
        for trade in trades[-20:]:  # Last 20 trades
            trade_price = trade['price']
            trade_size = trade['size']
            
            # Find displayed size at trade price
            displayed_size = 0
            
            for level in book.get('bids', []):
                if abs(level['price'] - trade_price) < 0.01:
                    displayed_size = level['size']
                    break
            
            for level in book.get('asks', []):
                if abs(level['price'] - trade_price) < 0.01:
                    displayed_size = max(displayed_size, level['size'])
                    break
            
            # Hidden order detected if trade > 1.5x displayed
            if displayed_size > 0 and trade_size > displayed_size * 1.5:
                return True
        
        return False
    
    def calculate_gamma_exposure(self, greeks: dict, chain: list, spot_price: float) -> dict:
        """Calculate net gamma exposure"""
        
        if not greeks or spot_price == 0:
            return {'total': 0, 'pin': spot_price, 'flip': spot_price, 'profile': {}}
        
        gex_by_strike = defaultdict(float)
        total_gex = 0
        
        for contract_key, greek_values in greeks.items():
            parts = contract_key.split('_')
            if len(parts) != 3:
                continue
            
            try:
                strike = float(parts[0])
                contract_type = parts[2]
                
                gamma = greek_values.get('gamma', 0)
                open_interest = greek_values.get('open_interest', 100)
                
                # GEX calculation
                contract_multiplier = 100
                spot_squared = spot_price * spot_price * 0.01
                
                if contract_type == 'CALL':
                    strike_gex = gamma * open_interest * contract_multiplier * spot_squared
                else:  # PUT
                    strike_gex = -gamma * open_interest * contract_multiplier * spot_squared
                
                gex_by_strike[strike] += strike_gex
                total_gex += strike_gex
                
            except (ValueError, KeyError):
                continue
        
        # Find pin strike (maximum gamma)
        if gex_by_strike:
            pin_strike = max(gex_by_strike, key=lambda k: abs(gex_by_strike[k]))
        else:
            pin_strike = spot_price
        
        # Find flip point (zero gamma crossing)
        flip_point = spot_price
        sorted_strikes = sorted(gex_by_strike.keys())
        for i in range(len(sorted_strikes) - 1):
            if gex_by_strike[sorted_strikes[i]] * gex_by_strike[sorted_strikes[i+1]] < 0:
                flip_point = sorted_strikes[i]
                break
        
        return {
            'total': round(total_gex / 1_000_000, 2),  # In millions
            'pin': pin_strike,
            'flip': flip_point,
            'profile': {str(k): round(v/1_000_000, 2) for k, v in gex_by_strike.items()}
        }
    
    def calculate_delta_exposure(self, greeks: dict, chain: list, spot_price: float) -> dict:
        """Calculate net delta exposure"""
        
        if not greeks:
            return {'total': 0, 'by_strike': {}}
        
        dex_by_strike = defaultdict(float)
        total_dex = 0
        
        for contract_key, greek_values in greeks.items():
            parts = contract_key.split('_')
            if len(parts) != 3:
                continue
            
            try:
                strike = float(parts[0])
                contract_type = parts[2]
                
                delta = greek_values.get('delta', 0)
                open_interest = greek_values.get('open_interest', 100)
                
                # DEX calculation
                contract_multiplier = 100
                
                if contract_type == 'CALL':
                    strike_dex = delta * open_interest * contract_multiplier * spot_price
                else:  # PUT
                    strike_dex = delta * open_interest * contract_multiplier * spot_price
                
                dex_by_strike[strike] += strike_dex
                total_dex += strike_dex
                
            except (ValueError, KeyError):
                continue
        
        return {
            'total': round(total_dex / 1_000_000, 2),  # In millions
            'by_strike': {str(k): round(v/1_000_000, 2) for k, v in dex_by_strike.items()}
        }
    
    def detect_sweeps(self, trades: list) -> dict:
        """Detect sweep orders based on clustering"""
        
        if len(trades) < 5:
            return {'detected': False}
        
        # Group trades by time window (1 second)
        time_groups = defaultdict(list)
        
        for trade in trades[-100:]:  # Last 100 trades
            time_key = int(trade.get('time', time.time()))
            time_groups[time_key].append(trade)
        
        # Look for sweeps
        for timestamp, group_trades in time_groups.items():
            if len(group_trades) >= 3:  # Multiple trades in same second
                total_size = sum(t['size'] for t in group_trades)
                avg_price = sum(t['price'] * t['size'] for t in group_trades) / total_size
                
                if total_size > 5000:  # Large total size
                    return {
                        'detected': True,
                        'timestamp': timestamp,
                        'total_size': total_size,
                        'avg_price': round(avg_price, 2),
                        'trade_count': len(group_trades)
                    }
        
        return {'detected': False}
```

---

## 6. Signal Generation

```python
from datetime import datetime, timedelta
import uuid

class SignalGenerator:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'PLTR', 'VXX']
        
    async def start(self):
        """Main signal generation loop"""
        
        while True:
            current_time = datetime.now()
            
            for symbol in self.symbols:
                # Get symbol configuration
                strategies = self.get_symbol_strategies(symbol)
                
                for strategy in strategies:
                    if self.is_strategy_active(strategy, current_time):
                        self.check_for_signal(symbol, strategy)
            
            await asyncio.sleep(0.5)
    
    def get_symbol_strategies(self, symbol: str) -> list:
        """Get strategies for symbol"""
        
        strategy_map = {
            'SPY': ['0DTE', '1DTE', 'MOC'],
            'QQQ': ['0DTE', '1DTE', 'MOC'],
            'IWM': ['0DTE', '1DTE'],
            'AAPL': ['14DTE'],
            'TSLA': ['14DTE'],
            'NVDA': ['14DTE'],
            'PLTR': ['14DTE'],
            'VXX': ['1DTE']
        }
        
        return strategy_map.get(symbol, [])
    
    def is_strategy_active(self, strategy: str, current_time: datetime) -> bool:
        """Check if strategy is in its time window"""
        
        hour = current_time.hour
        minute = current_time.minute
        current_minutes = hour * 60 + minute
        
        windows = {
            '0DTE': (9*60+45, 15*60),      # 9:45 AM - 3:00 PM
            '1DTE': (14*60, 15*60+30),     # 2:00 PM - 3:30 PM
            '14DTE': (9*60+30, 16*60),     # 9:30 AM - 4:00 PM
            'MOC': (15*60+30, 15*60+50)    # 3:30 PM - 3:50 PM
        }
        
        if strategy in windows:
            start, end = windows[strategy]
            return start <= current_minutes < end
        
        return False
    
    def check_for_signal(self, symbol: str, strategy: str):
        """Check if conditions met for signal"""
        
        # Load metrics
        vpin = float(self.redis.get(f'metrics:{symbol}:vpin') or 0)
        obi = json.loads(self.redis.get(f'metrics:{symbol}:obi') or '{}')
        gex = json.loads(self.redis.get(f'metrics:{symbol}:gex') or '{}')
        dex = json.loads(self.redis.get(f'metrics:{symbol}:dex') or '{}')
        sweep = json.loads(self.redis.get(f'metrics:{symbol}:sweep') or '{}')
        hidden = self.redis.get(f'metrics:{symbol}:hidden') == 'true'
        regime = self.redis.get(f'metrics:{symbol}:regime') or 'NORMAL'
        
        # Calculate confidence
        confidence = 0
        reason = []
        
        if strategy == '0DTE':
            # 0DTE: Gamma-driven intraday moves
            if vpin > 0.4:
                confidence += 30
                reason.append(f'VPIN elevated: {vpin:.2f}')
            
            if abs(obi.get('volume', 0)) > 0.3:
                confidence += 25
                reason.append(f'Order imbalance: {obi["volume"]:.2f}')
            
            if gex and gex.get('total', 0) > 100:
                spot = float(self.redis.get(f'market:{symbol}:last') or 0)
                pin = gex.get('pin', spot)
                if abs(spot - pin) / spot < 0.005:
                    confidence += 30
                    reason.append(f'Near gamma pin: {pin:.2f}')
            
            if sweep.get('detected', False):
                confidence += 15
                reason.append('Sweep detected')
        
        elif strategy == '1DTE':
            # 1DTE: Overnight positioning
            if regime == 'HIGH':
                confidence += 20
                reason.append('High volatility regime')
            
            if abs(obi.get('pressure', 0)) > 0.2:
                confidence += 30
                reason.append(f'Price pressure: {obi["pressure"]:.2f}')
            
            if gex and gex.get('total', 0) > 100:
                confidence += 25
                reason.append(f'High gamma: {gex["total"]:.0f}M')
            
            if vpin > 0.35:
                confidence += 25
                reason.append(f'Positioning detected: {vpin:.2f}')
        
        elif strategy == '14DTE':
            # 14DTE: Swing trades on unusual activity
            unusual = json.loads(self.redis.get(f'options:{symbol}:unusual') or '{}')
            
            if unusual.get('detected', False):
                confidence += 40
                reason.append('Unusual options activity')
            
            if sweep.get('detected', False):
                confidence += 30
                reason.append(f'Sweep: {sweep["total_size"]} shares')
            
            if hidden:
                confidence += 20
                reason.append('Hidden orders detected')
            
            if abs(dex.get('total', 0)) > 50:
                confidence += 10
                reason.append(f'Delta exposure: {dex["total"]:.0f}M')
        
        elif strategy == 'MOC':
            # MOC: Market-on-close imbalance
            if gex and gex.get('total', 0) > 50:
                spot = float(self.redis.get(f'market:{symbol}:last') or 0)
                pin = gex.get('pin', spot)
                pull = (pin - spot) / spot
                
                if abs(pull) > 0.002:
                    confidence += 40
                    reason.append(f'Gamma pull to {pin:.2f}')
            
            if abs(obi.get('volume', 0)) > 0.2:
                confidence += 30
                reason.append(f'Order flow imbalance: {obi["volume"]:.2f}')
            
            # Friday expiry
            if datetime.now().weekday() == 4:
                confidence += 30
                reason.append('Friday expiry dynamics')
        
        # Generate signal if confidence sufficient
        if confidence >= 60:
            signal = self.create_signal(symbol, strategy, confidence, reason, {
                'vpin': vpin,
                'obi': obi,
                'gex': gex,
                'dex': dex,
                'sweep': sweep,
                'hidden': hidden,
                'regime': regime
            })
            
            # Add to queue
            self.redis.lpush(f'signals:{symbol}:pending', json.dumps(signal))
            self.redis.expire(f'signals:{symbol}:pending', 60)
            
            # Update counter
            self.redis.incr('signals:global:count')
            
            print(f"Signal generated: {symbol} {strategy} confidence={confidence}%")
    
    def create_signal(self, symbol: str, strategy: str, confidence: float, reason: list, metrics: dict) -> dict:
        """Create complete signal object"""
        
        spot_price = float(self.redis.get(f'market:{symbol}:last') or 0)
        
        # Determine direction
        obi = metrics['obi']
        if strategy == 'MOC' and metrics['gex']:
            # MOC uses gamma pull
            pin = metrics['gex'].get('pin', spot_price)
            direction = 'BUY' if pin > spot_price else 'SELL'
        else:
            # Others use order imbalance
            direction = 'BUY' if obi.get('volume', 0) > 0 else 'SELL'
        
        # Calculate ATR for stops/targets
        atr = self.calculate_atr(symbol)
        
        # Set price levels
        if direction == 'BUY':
            stop_loss = spot_price - (atr * 1.5)
            targets = [
                spot_price + (atr * 1.0),
                spot_price + (atr * 2.0),
                spot_price + (atr * 3.0)
            ]
        else:
            stop_loss = spot_price + (atr * 1.5)
            targets = [
                spot_price - (atr * 1.0),
                spot_price - (atr * 2.0),
                spot_price - (atr * 3.0)
            ]
        
        # Select specific contract
        contract = self.select_contract(symbol, strategy, direction, spot_price)
        
        # Position sizing
        position_size = self.calculate_position_size(confidence, strategy)
        
        signal = {
            'signal_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'symbol': symbol,
            'strategy': strategy,
            'action': direction,
            'confidence': confidence,
            'reason': ' | '.join(reason),
            'entry_price': spot_price,
            'stop_loss': round(stop_loss, 2),
            'targets': [round(t, 2) for t in targets],
            'contract': contract,
            'position_size': position_size,
            'max_risk': position_size * abs(spot_price - stop_loss),
            'risk_reward': abs(targets[0] - spot_price) / abs(spot_price - stop_loss),
            'metrics': {
                'vpin': metrics['vpin'],
                'obi_volume': obi.get('volume', 0),
                'gex_total': metrics['gex'].get('total', 0) if metrics['gex'] else 0,
                'sweep': metrics['sweep'].get('detected', False),
                'regime': metrics['regime']
            }
        }
        
        return signal
    
    def select_contract(self, symbol: str, strategy: str, direction: str, spot: float) -> dict:
        """Select specific option contract"""
        
        if strategy == '0DTE':
            # First OTM strike expiring today
            if direction == 'BUY':
                strike = int(spot) + 1 if spot % 1 < 0.5 else int(spot) + 0.5
            else:
                strike = int(spot) if spot % 1 < 0.5 else int(spot) + 0.5
            
            expiry = datetime.now().strftime('%Y-%m-%d')
            contract_type = 'CALL' if direction == 'BUY' else 'PUT'
        
        elif strategy == '1DTE':
            # 1% OTM expiring tomorrow
            if direction == 'BUY':
                strike = round(spot * 1.01)
            else:
                strike = round(spot * 0.99)
            
            expiry = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            contract_type = 'CALL' if direction == 'BUY' else 'PUT'
        
        elif strategy == '14DTE':
            # Follow unusual activity or 2% OTM
            unusual = json.loads(self.redis.get(f'options:{symbol}:unusual') or '{}')
            
            if unusual.get('detected') and unusual.get('contracts'):
                # Follow the most unusual
                best = unusual['contracts'][0]
                strike = best['strike']
                expiry = best['expiration']
                contract_type = best['type']
            else:
                # Default 2% OTM, 14 days out
                if direction == 'BUY':
                    strike = round(spot * 1.02)
                else:
                    strike = round(spot * 0.98)
                
                expiry = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                contract_type = 'CALL' if direction == 'BUY' else 'PUT'
        
        else:  # MOC
            # MOC trades stock, not options
            return {
                'type': 'STOCK',
                'symbol': symbol
            }
        
        return {
            'type': 'OPTION',
            'symbol': symbol,
            'strike': strike,
            'expiry': expiry,
            'contract_type': contract_type
        }
    
    def calculate_position_size(self, confidence: float, strategy: str) -> float:
        """Calculate position size based on Kelly criterion"""
        
        # Get account value
        buying_power = float(self.redis.get('global:buying_power') or 100000)
        
        # Base allocation by strategy
        base_allocation = {
            '0DTE': 0.05,   # 5% for 0DTE
            '1DTE': 0.07,   # 7% for overnight
            '14DTE': 0.10,  # 10% for swings
            'MOC': 0.15     # 15% for MOC
        }
        
        # Kelly adjustment based on confidence
        kelly_fraction = 0.25  # Conservative Kelly
        confidence_factor = (confidence - 60) / 40  # 0 to 1 scale
        
        allocation = base_allocation.get(strategy, 0.05)
        adjusted_allocation = allocation * (1 + confidence_factor * kelly_fraction)
        
        # Cap at maximum
        max_allocation = 0.20  # Never more than 20% per trade
        final_allocation = min(adjusted_allocation, max_allocation)
        
        return round(buying_power * final_allocation, 0)
    
    def calculate_atr(self, symbol: str) -> float:
        """Calculate Average True Range"""
        
        bars_json = self.redis.get(f'market:{symbol}:bars')
        if not bars_json:
            return 1.0  # Default
        
        bars = json.loads(bars_json)
        if len(bars) < 14:
            return 1.0
        
        # Calculate TR for each bar
        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # ATR is average of last 14 TRs
        if len(true_ranges) >= 14:
            atr = np.mean(true_ranges[-14:])
        else:
            atr = np.mean(true_ranges)
        
        return max(atr, 0.5)  # Minimum 0.5
```

---

## 7. Execution Manager

```python
from ib_insync import Option, Stock, MarketOrder, LimitOrder

class ExecutionManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ib = IB()
        self.max_positions = 5
        self.max_per_symbol = 2
        
    async def start(self):
        """Main execution loop"""
        
        # Connect to IBKR
        await self.ib.connectAsync('127.0.0.1', 7497, clientId=2)
        
        while True:
            # Check if trading is halted
            if self.redis.get('global:halt') == 'true':
                await asyncio.sleep(1)
                continue
            
            # Check position limits
            position_count = int(self.redis.get('global:positions:count') or 0)
            if position_count >= self.max_positions:
                await asyncio.sleep(1)
                continue
            
            # Process signals for each symbol
            for symbol in ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'PLTR', 'VXX']:
                # Check symbol position limit
                symbol_positions = len(self.redis.keys(f'positions:{symbol}:*'))
                if symbol_positions >= self.max_per_symbol:
                    continue
                
                # Get pending signal
                signal_json = self.redis.rpop(f'signals:{symbol}:pending')
                if signal_json:
                    signal = json.loads(signal_json)
                    
                    # Risk checks
                    if self.passes_risk_checks(signal):
                        await self.execute_signal(signal)
            
            await asyncio.sleep(0.2)
    
    def passes_risk_checks(self, signal: dict) -> bool:
        """Perform risk checks before execution"""
        
        # Check buying power
        buying_power = float(self.redis.get('global:buying_power') or 0)
        if signal['position_size'] > buying_power * 0.25:
            print(f"Risk check failed: Position too large for {signal['symbol']}")
            return False
        
        # Check correlation
        correlation_matrix = json.loads(self.redis.get('discovered:correlation_matrix') or '{}')
        existing_positions = self.get_existing_position_symbols()
        
        for existing_symbol in existing_positions:
            if existing_symbol in correlation_matrix.get(signal['symbol'], {}):
                correlation = correlation_matrix[signal['symbol']][existing_symbol]
                if abs(correlation) > 0.7:
                    print(f"Risk check failed: High correlation {signal['symbol']} vs {existing_symbol}")
                    return False
        
        # Check daily loss
        daily_pnl = float(self.redis.get('global:pnl:realized') or 0)
        if daily_pnl < -2000:  # $2000 daily loss limit
            print("Risk check failed: Daily loss limit reached")
            self.redis.set('global:halt', 'true')
            return False
        
        return True
    
    def get_existing_position_symbols(self) -> list:
        """Get symbols with open positions"""
        
        position_keys = self.redis.keys('positions:*:*')
        symbols = set()
        
        for key in position_keys:
            parts = key.split(':')
            if len(parts) >= 2:
                symbols.add(parts[1])
        
        return list(symbols)
    
    async def execute_signal(self, signal: dict):
        """Execute a trading signal"""
        
        print(f"Executing signal: {signal['symbol']} {signal['strategy']} {signal['action']}")
        
        # Create contract
        if signal['contract']['type'] == 'OPTION':
            contract = Option(
                symbol=signal['symbol'],
                lastTradeDateOrContractMonth=signal['contract']['expiry'].replace('-', ''),
                strike=signal['contract']['strike'],
                right='C' if signal['contract']['contract_type'] == 'CALL' else 'P',
                exchange='SMART'
            )
        else:  # STOCK
            contract = Stock(signal['symbol'], 'SMART', 'USD')
        
        # Qualify the contract
        self.ib.qualifyContracts(contract)
        
        # Get current market data
        ticker = self.ib.reqTickers(contract)[0]
        
        # Determine order type and size
        if signal['confidence'] > 85:
            # High confidence: market order
            order = MarketOrder(
                action='BUY' if signal['action'] == 'BUY' else 'SELL',
                totalQuantity=self.calculate_order_size(signal, ticker)
            )
        else:
            # Normal confidence: limit order
            if signal['action'] == 'BUY':
                limit_price = ticker.ask if ticker.ask else ticker.last
            else:
                limit_price = ticker.bid if ticker.bid else ticker.last
            
            order = LimitOrder(
                action='BUY' if signal['action'] == 'BUY' else 'SELL',
                totalQuantity=self.calculate_order_size(signal, ticker),
                lmtPrice=limit_price
            )
        
        # Place the order
        trade = self.ib.placeOrder(contract, order)
        
        # Store pending order
        order_data = {
            'order_id': trade.order.orderId,
            'signal_id': signal['signal_id'],
            'symbol': signal['symbol'],
            'strategy': signal['strategy'],
            'action': signal['action'],
            'quantity': order.totalQuantity,
            'order_type': order.orderType,
            'status': 'PENDING',
            'timestamp': time.time()
        }
        
        self.redis.setex(
            f'orders:pending:{trade.order.orderId}',
            300,  # 5 minute TTL
            json.dumps(order_data)
        )
        
        # Wait for fill
        asyncio.create_task(self.monitor_order(trade, signal))
    
    def calculate_order_size(self, signal: dict, ticker) -> int:
        """Calculate order size for options/stock"""
        
        if signal['contract']['type'] == 'OPTION':
            # Options: calculate contracts based on premium
            if ticker.ask and ticker.bid:
                mid_price = (ticker.ask + ticker.bid) / 2
            else:
                mid_price = ticker.last or 1.0
            
            # Each option controls 100 shares
            premium_per_contract = mid_price * 100
            
            if premium_per_contract > 0:
                contracts = int(signal['position_size'] / premium_per_contract)
                return max(1, min(contracts, 50))  # 1-50 contracts
            else:
                return 1
        
        else:  # STOCK
            # Stock: calculate shares
            if ticker.last:
                shares = int(signal['position_size'] / ticker.last)
                return max(1, shares)
            else:
                return 100
    
    async def monitor_order(self, trade, signal: dict):
        """Monitor order until filled or cancelled"""
        
        while not trade.isDone():
            await asyncio.sleep(0.1)
        
        if trade.orderStatus.status == 'Filled':
            # Create position
            position_id = str(uuid.uuid4())
            position = {
                'position_id': position_id,
                'signal_id': signal['signal_id'],
                'symbol': signal['symbol'],
                'strategy': signal['strategy'],
                'direction': signal['action'],
                'entry_price': trade.orderStatus.avgFillPrice,
                'entry_time': time.time(),
                'size': trade.orderStatus.filled,
                'stop_loss': signal['stop_loss'],
                'targets': signal['targets'],
                'current_target': 0,
                'status': 'OPEN',
                'pnl_realized': 0,
                'pnl_unrealized': 0,
                'contract': signal['contract']
            }
            
            # Store position
            self.redis.set(f'positions:{signal["symbol"]}:{position_id}', json.dumps(position))
            
            # Update global counts
            self.redis.incr('global:positions:count')
            
            # Place stop loss order
            await self.place_stop_loss(position)
            
            print(f"Position opened: {signal['symbol']} @ {trade.orderStatus.avgFillPrice}")
        
        else:
            print(f"Order failed: {signal['symbol']} - {trade.orderStatus.status}")
        
        # Clean up pending order
        self.redis.delete(f'orders:pending:{trade.order.orderId}')
    
    async def place_stop_loss(self, position: dict):
        """Place stop loss order for position"""
        
        # Create contract
        if position['contract']['type'] == 'OPTION':
            contract = Option(
                symbol=position['symbol'],
                lastTradeDateOrContractMonth=position['contract']['expiry'].replace('-', ''),
                strike=position['contract']['strike'],
                right='C' if position['contract']['contract_type'] == 'CALL' else 'P',
                exchange='SMART'
            )
        else:
            contract = Stock(position['symbol'], 'SMART', 'USD')
        
        # Create stop order (opposite direction)
        if position['direction'] == 'BUY':
            action = 'SELL'
        else:
            action = 'BUY'
        
        stop_order = StopOrder(
            action=action,
            totalQuantity=position['size'],
            stopPrice=position['stop_loss']
        )
        
        # Place the stop
        stop_trade = self.ib.placeOrder(contract, stop_order)
        
        # Store stop order ID with position
        position['stop_order_id'] = stop_trade.order.orderId
        self.redis.set(
            f'positions:{position["symbol"]}:{position["position_id"]}',
            json.dumps(position)
        )
```

---

## 8. Position Manager

```python
class PositionManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
    async def start(self):
        """Main position management loop"""
        
        while True:
            # Get all open positions
            position_keys = self.redis.keys('positions:*:*')
            
            for key in position_keys:
                position_json = self.redis.get(key)
                if position_json:
                    position = json.loads(position_json)
                    
                    if position['status'] == 'OPEN':
                        self.update_position(position)
            
            await asyncio.sleep(1)
    
    def update_position(self, position: dict):
        """Update position P&L and manage stops"""
        
        # Get current price
        current_price = float(self.redis.get(f'market:{position["symbol"]}:last') or 0)
        
        if current_price == 0:
            return
        
        # Calculate unrealized P&L
        if position['contract']['type'] == 'OPTION':
            # For options, need the option price, not stock price
            # This would require getting option quotes
            # Simplified: assume linear relationship
            price_change = current_price - position['entry_price']
            
            if position['direction'] == 'BUY':
                position['pnl_unrealized'] = price_change * position['size'] * 100
            else:
                position['pnl_unrealized'] = -price_change * position['size'] * 100
        else:
            # Stock P&L
            if position['direction'] == 'BUY':
                position['pnl_unrealized'] = (current_price - position['entry_price']) * position['size']
            else:
                position['pnl_unrealized'] = (position['entry_price'] - current_price) * position['size']
        
        # Trail stop if profitable
        if position['pnl_unrealized'] > 0:
            self.trail_stop(position, current_price)
        
        # Check targets for scaling
        self.check_targets(position, current_price)
        
        # Update position in Redis
        self.redis.set(
            f'positions:{position["symbol"]}:{position["position_id"]}',
            json.dumps(position)
        )
    
    def trail_stop(self, position: dict, current_price: float):
        """Trail stop loss if position is profitable"""
        
        if position['direction'] == 'BUY':
            # For longs, trail stop up
            new_stop = position['entry_price'] + (current_price - position['entry_price']) * 0.5
            
            if new_stop > position['stop_loss']:
                position['stop_loss'] = round(new_stop, 2)
                print(f"Stop trailed to {position['stop_loss']} for {position['symbol']}")
        else:
            # For shorts, trail stop down
            new_stop = position['entry_price'] - (position['entry_price'] - current_price) * 0.5
            
            if new_stop < position['stop_loss']:
                position['stop_loss'] = round(new_stop, 2)
                print(f"Stop trailed to {position['stop_loss']} for {position['symbol']}")
    
    def check_targets(self, position: dict, current_price: float):
        """Check if targets hit for scaling out"""
        
        current_target_index = position.get('current_target', 0)
        
        if current_target_index >= len(position['targets']):
            return  # All targets hit
        
        target_price = position['targets'][current_target_index]
        
        if position['direction'] == 'BUY':
            if current_price >= target_price:
                self.scale_out(position, target_price, current_target_index)
        else:
            if current_price <= target_price:
                self.scale_out(position, target_price, current_target_index)
    
    def scale_out(self, position: dict, target_price: float, target_index: int):
        """Scale out of position at target"""
        
        scale_percentages = [0.33, 0.50, 1.0]  # Take 1/3, 1/2, then all
        
        if target_index < len(scale_percentages):
            scale_pct = scale_percentages[target_index]
            shares_to_sell = int(position['size'] * scale_pct)
            
            print(f"Target {target_index + 1} hit for {position['symbol']}, scaling out {shares_to_sell} shares")
            
            # Update position
            position['size'] -= shares_to_sell
            position['pnl_realized'] += shares_to_sell * (target_price - position['entry_price'])
            position['current_target'] = target_index + 1
            
            if position['size'] == 0:
                position['status'] = 'CLOSED'
                position['exit_price'] = target_price
                position['exit_time'] = time.time()
                position['exit_reason'] = f'Target {target_index + 1}'
                
                # Update global P&L
                self.redis.incrbyfloat('global:pnl:realized', position['pnl_realized'])
                self.redis.decr('global:positions:count')
```

---

## 9. Risk Manager & Circuit Breakers

```python
class RiskManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.daily_loss_limit = 2000
        self.max_consecutive_losses = 3
        self.consecutive_losses = 0
        
    async def start(self):
        """Monitor risk metrics continuously"""
        
        while True:
            self.check_circuit_breakers()
            self.update_risk_metrics()
            await asyncio.sleep(1)
    
    def check_circuit_breakers(self):
        """Check all circuit breakers"""
        
        # Daily loss circuit breaker
        daily_pnl = float(self.redis.get('global:pnl:realized') or 0)
        
        if daily_pnl < -self.daily_loss_limit:
            print(f"CIRCUIT BREAKER: Daily loss limit hit: ${daily_pnl:.2f}")
            self.halt_trading("Daily loss limit exceeded")
            return
        
        # Position correlation circuit breaker
        correlation = self.calculate_portfolio_correlation()
        
        if correlation > 0.8:
            print(f"CIRCUIT BREAKER: Portfolio correlation too high: {correlation:.2f}")
            self.halt_trading("Portfolio correlation exceeded 0.8")
            return
        
        # Drawdown circuit breaker
        drawdown = self.calculate_drawdown()
        
        if drawdown > 0.10:  # 10% drawdown
            print(f"CIRCUIT BREAKER: Drawdown limit hit: {drawdown:.1%}")
            self.halt_trading("10% drawdown limit exceeded")
            return
    
    def halt_trading(self, reason: str):
        """Halt all trading"""
        
        self.redis.set('global:halt', 'true')
        self.redis.set('global:halt:reason', reason)
        self.redis.set('global:halt:time', time.time())
        
        # Send alerts
        print(f"TRADING HALTED: {reason}")
    
    def calculate_portfolio_correlation(self) -> float:
        """Calculate average correlation of portfolio"""
        
        correlation_matrix = json.loads(self.redis.get('discovered:correlation_matrix') or '{}')
        position_symbols = self.get_position_symbols()
        
        if len(position_symbols) < 2:
            return 0.0
        
        total_correlation = 0
        count = 0
        
        for i, symbol1 in enumerate(position_symbols):
            for symbol2 in position_symbols[i+1:]:
                if symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                    total_correlation += abs(correlation_matrix[symbol1][symbol2])
                    count += 1
        
        return total_correlation / count if count > 0 else 0.0
    
    def get_position_symbols(self) -> list:
        """Get all symbols with open positions"""
        
        position_keys = self.redis.keys('positions:*:*')
        symbols = set()
        
        for key in position_keys:
            parts = key.split(':')
            if len(parts) >= 2:
                symbols.add(parts[1])
        
        return list(symbols)
    
    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from high water mark"""
        
        # This would track account value over time
        # Simplified version
        account_value = float(self.redis.get('global:buying_power') or 100000)
        high_water_mark = float(self.redis.get('global:high_water_mark') or account_value)
        
        if account_value > high_water_mark:
            self.redis.set('global:high_water_mark', account_value)
            return 0.0
        
        return (high_water_mark - account_value) / high_water_mark
    
    def update_risk_metrics(self):
        """Update risk metrics in Redis"""
        
        # Calculate Value at Risk (VaR)
        var = self.calculate_var()
        self.redis.setex('global:risk:var', 5, var)
        
        # Calculate position sizes
        position_sizes = self.calculate_position_sizes()
        self.redis.setex('global:risk:position_sizes', 5, json.dumps(position_sizes))
        
        # Update correlation
        correlation = self.calculate_portfolio_correlation()
        self.redis.setex('global:risk:correlation', 5, correlation)
    
    def calculate_var(self) -> float:
        """Calculate 95% Value at Risk"""
        
        # Simplified VaR calculation
        positions = []
        position_keys = self.redis.keys('positions:*:*')
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                positions.append(position['pnl_unrealized'])
        
        if not positions:
            return 0.0
        
        # 95% VaR (1.65 standard deviations)
        return np.mean(positions) - 1.65 * np.std(positions)
    
    def calculate_position_sizes(self) -> dict:
        """Calculate current position sizes"""
        
        sizes = {}
        position_keys = self.redis.keys('positions:*:*')
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                symbol = position['symbol']
                
                if symbol not in sizes:
                    sizes[symbol] = 0
                
                # Calculate position value
                current_price = float(self.redis.get(f'market:{symbol}:last') or position['entry_price'])
                position_value = current_price * position['size']
                
                if position['contract']['type'] == 'OPTION':
                    position_value *= 100  # Options multiplier
                
                sizes[symbol] += position_value
        
        return sizes


class EmergencyManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ib = IB()
        
    async def emergency_close_all(self):
        """Emergency close all positions"""
        
        print("EMERGENCY: Closing all positions!")
        
        # Connect to IBKR if not connected
        if not self.ib.isConnected():
            await self.ib.connectAsync('127.0.0.1', 7497, clientId=3)
        
        # Cancel all pending orders
        for order in self.ib.orders():
            if order.orderStatus.status in ['PendingSubmit', 'Submitted', 'PreSubmitted']:
                self.ib.cancelOrder(order)
                print(f"Cancelled order: {order.orderId}")
        
        # Close all positions
        position_keys = self.redis.keys('positions:*:*')
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                
                if position['status'] == 'OPEN':
                    await self.close_position(position)
        
        # Halt trading
        self.redis.set('global:halt', 'true')
        self.redis.set('global:halt:reason', 'Emergency close executed')
    
    async def close_position(self, position: dict):
        """Close a single position at market"""
        
        # Create contract
        if position['contract']['type'] == 'OPTION':
            contract = Option(
                symbol=position['symbol'],
                lastTradeDateOrContractMonth=position['contract']['expiry'].replace('-', ''),
                strike=position['contract']['strike'],
                right='C' if position['contract']['contract_type'] == 'CALL' else 'P',
                exchange='SMART'
            )
        else:
            contract = Stock(position['symbol'], 'SMART', 'USD')
        
        # Create market order (opposite direction)
        if position['direction'] == 'BUY':
            action = 'SELL'
        else:
            action = 'BUY'
        
        order = MarketOrder(
            action=action,
            totalQuantity=position['size']
        )
        
        # Place the order
        trade = self.ib.placeOrder(contract, order)
        
        print(f"Emergency close: {position['symbol']} x {position['size']}")
        
        # Update position status
        position['status'] = 'CLOSING'
        position['exit_reason'] = 'EMERGENCY'
        
        self.redis.set(
            f'positions:{position["symbol"]}:{position["position_id"]}',
            json.dumps(position)
        )
```

---

## 10. Distribution System

```python
class SignalDistributor:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
    async def start(self):
        """Distribute signals to different tiers"""
        
        while True:
            # Check for new signals
            signal_keys = self.redis.keys('signals:*:active')
            
            for key in signal_keys:
                signal_json = self.redis.get(key)
                if signal_json:
                    signal = json.loads(signal_json)
                    await self.distribute_signal(signal)
            
            # Check for position updates
            position_keys = self.redis.keys('positions:*:*')
            
            for key in position_keys:
                position_json = self.redis.get(key)
                if position_json:
                    position = json.loads(position_json)
                    await self.distribute_position_update(position)
            
            await asyncio.sleep(1)
    
    async def distribute_signal(self, signal: dict):
        """Distribute signal to appropriate tiers"""
        
        # Premium tier: real-time
        premium_signal = {
            'type': 'SIGNAL',
            'timestamp': signal['timestamp'],
            'symbol': signal['symbol'],
            'strategy': signal['strategy'],
            'action': signal['action'],
            'confidence': signal['confidence'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'targets': signal['targets'],
            'reason': signal['reason']
        }
        
        self.redis.lpush('distribution:premium:queue', json.dumps(premium_signal))
        
        # Basic tier: delayed and limited info
        basic_signal = {
            'type': 'SIGNAL',
            'timestamp': signal['timestamp'] + 60,  # 60 second delay
            'symbol': signal['symbol'],
            'action': signal['action'],
            'confidence_range': 'HIGH' if signal['confidence'] > 80 else 'MEDIUM'
        }
        
        # Schedule for delayed distribution
        asyncio.create_task(self.delayed_publish('distribution:basic:queue', basic_signal, 60))
    
    async def delayed_publish(self, queue: str, data: dict, delay: int):
        """Publish with delay"""
        await asyncio.sleep(delay)
        self.redis.lpush(queue, json.dumps(data))
    
    async def distribute_position_update(self, position: dict):
        """Distribute position updates"""
        
        update = {
            'type': 'POSITION_UPDATE',
            'timestamp': time.time(),
            'symbol': position['symbol'],
            'status': position['status'],
            'pnl_unrealized': position['pnl_unrealized'],
            'pnl_realized': position['pnl_realized']
        }
        
        # Premium gets everything
        self.redis.lpush('distribution:premium:queue', json.dumps(update))
        
        # Basic gets summary only
        if position['status'] == 'CLOSED':
            basic_update = {
                'type': 'TRADE_CLOSED',
                'symbol': position['symbol'],
                'result': 'WIN' if position['pnl_realized'] > 0 else 'LOSS'
            }
            self.redis.lpush('distribution:basic:queue', json.dumps(basic_update))


class DiscordBot:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.webhook_urls = {
            'basic': 'https://discord.com/api/webhooks/YOUR_BASIC_WEBHOOK',
            'premium': 'https://discord.com/api/webhooks/YOUR_PREMIUM_WEBHOOK'
        }
        
    async def start(self):
        """Send updates to Discord"""
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Process premium queue
                premium_msg = self.redis.rpop('distribution:premium:queue')
                if premium_msg:
                    await self.send_to_discord(session, 'premium', json.loads(premium_msg))
                
                # Process basic queue
                basic_msg = self.redis.rpop('distribution:basic:queue')
                if basic_msg:
                    await self.send_to_discord(session, 'basic', json.loads(basic_msg))
                
                await asyncio.sleep(0.5)
    
    async def send_to_discord(self, session: aiohttp.ClientSession, tier: str, data: dict):
        """Send formatted message to Discord"""
        
        if data['type'] == 'SIGNAL':
            embed = self.create_signal_embed(data, tier)
        elif data['type'] == 'POSITION_UPDATE':
            embed = self.create_position_embed(data)
        else:
            return
        
        webhook_data = {'embeds': [embed]}
        
        async with session.post(self.webhook_urls[tier], json=webhook_data) as response:
            if response.status != 204:
                print(f"Discord webhook failed: {response.status}")
    
    def create_signal_embed(self, signal: dict, tier: str) -> dict:
        """Create Discord embed for signal"""
        
        color = 0x00FF00 if signal['action'] == 'BUY' else 0xFF0000
        
        embed = {
            'title': f"🚨 {signal['symbol']} - {signal['action']}",
            'color': color,
            'timestamp': datetime.fromtimestamp(signal['timestamp']).isoformat(),
            'fields': []
        }
        
        if tier == 'premium':
            embed['fields'].extend([
                {'name': 'Strategy', 'value': signal.get('strategy', 'N/A'), 'inline': True},
                {'name': 'Confidence', 'value': f"{signal.get('confidence', 0):.0f}%", 'inline': True},
                {'name': 'Entry', 'value': f"${signal.get('entry_price', 0):.2f}", 'inline': True},
                {'name': 'Stop Loss', 'value': f"${signal.get('stop_loss', 0):.2f}", 'inline': True},
                {'name': 'Target 1', 'value': f"${signal.get('targets', [0])[0]:.2f}", 'inline': True},
                {'name': 'Reason', 'value': signal.get('reason', 'Multiple factors'), 'inline': False}
            ])
        else:  # basic
            embed['fields'].extend([
                {'name': 'Direction', 'value': signal['action'], 'inline': True},
                {'name': 'Confidence', 'value': signal.get('confidence_range', 'MEDIUM'), 'inline': True}
            ])
        
        return embed
    
    def create_position_embed(self, position: dict) -> dict:
        """Create Discord embed for position update"""
        
        pnl = position.get('pnl_unrealized', 0)
        color = 0x00FF00 if pnl > 0 else 0xFF0000 if pnl < 0 else 0xFFFF00
        
        embed = {
            'title': f"📊 {position['symbol']} Position Update",
            'color': color,
            'timestamp': datetime.fromtimestamp(position['timestamp']).isoformat(),
            'fields': [
                {'name': 'Status', 'value': position['status'], 'inline': True},
                {'name': 'Unrealized P&L', 'value': f"${pnl:.2f}", 'inline': True},
                {'name': 'Realized P&L', 'value': f"${position.get('pnl_realized', 0):.2f}", 'inline': True}
            ]
        }
        
        return embed
```

---

## 11. Dashboard & Monitoring

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

class Dashboard:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.app = FastAPI()
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.get("/")
        async def get_dashboard():
            return HTMLResponse(self.generate_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            while True:
                # Send updates every second
                data = self.get_dashboard_data()
                await websocket.send_json(data)
                await asyncio.sleep(1)
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            return self.get_dashboard_data()
    
    def get_dashboard_data(self) -> dict:
        """Get all dashboard data from Redis"""
        
        return {
            'timestamp': time.time(),
            'system_health': {
                'trading_active': self.redis.get('global:halt') != 'true',
                'ibkr_connected': True,  # Would check actual connection
                'av_api_calls': int(self.redis.get('monitoring:api:av:calls') or 0),
                'errors': int(self.redis.get('monitoring:errors:count') or 0)
            },
            'positions': self.get_positions(),
            'pnl': {
                'realized': float(self.redis.get('global:pnl:realized') or 0),
                'unrealized': self.calculate_unrealized_pnl(),
                'total': float(self.redis.get('global:pnl:realized') or 0) + self.calculate_unrealized_pnl()
            },
            'signals': {
                'today': int(self.redis.get('signals:global:count') or 0),
                'pending': self.count_pending_signals()
            },
            'risk': {
                'var': float(self.redis.get('global:risk:var') or 0),
                'correlation': float(self.redis.get('global:risk:correlation') or 0),
                'buying_power': float(self.redis.get('global:buying_power') or 100000)
            },
            'market_data': self.get_market_data()
        }
    
    def get_positions(self) -> list:
        """Get all open positions"""
        
        positions = []
        position_keys = self.redis.keys('positions:*:*')
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                if position['status'] == 'OPEN':
                    positions.append({
                        'symbol': position['symbol'],
                        'strategy': position['strategy'],
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'current_price': float(self.redis.get(f'market:{position["symbol"]}:last') or position['entry_price']),
                        'size': position['size'],
                        'pnl_unrealized': position['pnl_unrealized'],
                        'stop_loss': position['stop_loss']
                    })
        
        return positions
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        
        total = 0
        position_keys = self.redis.keys('positions:*:*')
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                if position['status'] == 'OPEN':
                    total += position.get('pnl_unrealized', 0)
        
        return total
    
    def count_pending_signals(self) -> int:
        """Count pending signals"""
        
        count = 0
        for symbol in ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'PLTR', 'VXX']:
            count += self.redis.llen(f'signals:{symbol}:pending')
        
        return count
    
    def get_market_data(self) -> dict:
        """Get market data for primary symbol"""
        
        symbol = 'SPY'
        
        return {
            'symbol': symbol,
            'last': float(self.redis.get(f'market:{symbol}:last') or 0),
            'vpin': float(self.redis.get(f'metrics:{symbol}:vpin') or 0),
            'obi': json.loads(self.redis.get(f'metrics:{symbol}:obi') or '{}'),
            'gex': json.loads(self.redis.get(f'metrics:{symbol}:gex') or '{}'),
            'regime': self.redis.get(f'metrics:{symbol}:regime') or 'NORMAL'
        }
    
    def generate_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlphaTrader Dashboard</title>
            <style>
                body { font-family: monospace; background: #1e1e1e; color: #0f0; padding: 20px; }
                .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
                .panel { border: 1px solid #0f0; padding: 15px; background: #0a0a0a; }
                .header { font-size: 18px; margin-bottom: 10px; color: #0ff; }
                .value { font-size: 24px; font-weight: bold; }
                .positive { color: #0f0; }
                .negative { color: #f00; }
                table { width: 100%; border-collapse: collapse; }
                td { padding: 5px; border-bottom: 1px solid #333; }
            </style>
        </head>
        <body>
            <h1>ALPHATRADER REAL-TIME DASHBOARD</h1>
            
            <div class="grid">
                <div class="panel">
                    <div class="header">SYSTEM STATUS</div>
                    <div id="system-status"></div>
                </div>
                
                <div class="panel">
                    <div class="header">P&L</div>
                    <div id="pnl"></div>
                </div>
                
                <div class="panel">
                    <div class="header">RISK METRICS</div>
                    <div id="risk"></div>
                </div>
            </div>
            
            <div class="panel" style="margin-top: 20px;">
                <div class="header">OPEN POSITIONS</div>
                <table id="positions"></table>
            </div>
            
            <div class="panel" style="margin-top: 20px;">
                <div class="header">MARKET DATA</div>
                <div id="market"></div>
            </div>
            
            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    // System Status
                    document.getElementById('system-status').innerHTML = `
                        Trading: ${data.system_health.trading_active ? '✅ ACTIVE' : '❌ HALTED'}<br>
                        IBKR: ${data.system_health.ibkr_connected ? '✅' : '❌'}<br>
                        AV Calls: ${data.system_health.av_api_calls}/600<br>
                        Errors: ${data.system_health.errors}
                    `;
                    
                    // P&L
                    const totalPnl = data.pnl.total;
                    document.getElementById('pnl').innerHTML = `
                        <div class="value ${totalPnl >= 0 ? 'positive' : 'negative'}">
                            $${totalPnl.toFixed(2)}
                        </div>
                        Realized: $${data.pnl.realized.toFixed(2)}<br>
                        Unrealized: $${data.pnl.unrealized.toFixed(2)}
                    `;
                    
                    // Risk
                    document.getElementById('risk').innerHTML = `
                        VaR: $${data.risk.var.toFixed(2)}<br>
                        Correlation: ${(data.risk.correlation * 100).toFixed(1)}%<br>
                        Buying Power: $${data.risk.buying_power.toFixed(0)}
                    `;
                    
                    // Positions
                    let positionsHtml = '<tr><th>Symbol</th><th>Dir</th><th>Entry</th><th>Current</th><th>P&L</th></tr>';
                    for (const pos of data.positions) {
                        const pnlClass = pos.pnl_unrealized >= 0 ? 'positive' : 'negative';
                        positionsHtml += `
                            <tr>
                                <td>${pos.symbol}</td>
                                <td>${pos.direction}</td>
                                <td>$${pos.entry_price.toFixed(2)}</td>
                                <td>$${pos.current_price.toFixed(2)}</td>
                                <td class="${pnlClass}">$${pos.pnl_unrealized.toFixed(2)}</td>
                            </tr>
                        `;
                    }
                    document.getElementById('positions').innerHTML = positionsHtml;
                    
                    // Market Data
                    document.getElementById('market').innerHTML = `
                        SPY: $${data.market_data.last.toFixed(2)}<br>
                        VPIN: ${data.market_data.vpin.toFixed(3)}<br>
                        OBI: ${data.market_data.obi.volume ? data.market_data.obi.volume.toFixed(3) : 'N/A'}<br>
                        GEX: ${data.market_data.gex.total ? data.market_data.gex.total.toFixed(1) + 'M' : 'N/A'}<br>
                        Regime: ${data.market_data.regime}
                    `;
                }
            </script>
        </body>
        </html>
        """
    
    async def start(self):
        """Start the dashboard server"""
        config = uvicorn.Config(app=self.app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        await server.serve()
```

---

## 12. Main Application

```python
import asyncio
import yaml
import sys

class AlphaTrader:
    def __init__(self, config_file='config/config.yaml'):
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Redis
        self.redis = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            decode_responses=True
        )
        
        # Initialize all modules
        self.modules = {
            'ibkr_ingestion': IBKRIngestion(),
            'av_ingestion': AlphaVantageIngestion(self.config['alpha_vantage']['api_key']),
            'parameter_discovery': ParameterDiscovery(),
            'analytics_engine': AnalyticsEngine(),
            'signal_generator': SignalGenerator(),
            'execution_manager': ExecutionManager(),
            'position_manager': PositionManager(),
            'risk_manager': RiskManager(),
            'emergency_manager': EmergencyManager(),
            'signal_distributor': SignalDistributor(),
            'discord_bot': DiscordBot(),
            'dashboard': Dashboard(),
            'twitter_bot': TwitterBot(),
            'telegram_bot': TelegramBot(),
            'market_analysis': MarketAnalysisGenerator(),
            'scheduled_tasks': ScheduledTasks()
        }
    
    async def start(self):
        """Start all modules"""
        
        print("Starting AlphaTrader System...")
        
        # Run parameter discovery first
        print("Running parameter discovery...")
        await self.modules['parameter_discovery'].run_discovery()
        
        # Start all modules
        tasks = []
        
        for name, module in self.modules.items():
            if name != 'parameter_discovery':  # Already ran
                print(f"Starting {name}...")
                tasks.append(asyncio.create_task(module.start()))
        
        # Start social media modules
        print("Starting Twitter bot...")
        tasks.append(asyncio.create_task(self.modules['twitter_bot'].start()))
        
        print("Starting Telegram bot...")
        tasks.append(asyncio.create_task(self.modules['telegram_bot'].start()))
        
        print("Starting scheduled tasks...")
        tasks.append(asyncio.create_task(self.modules['scheduled_tasks'].start()))
        
        # Run forever
        await asyncio.gather(*tasks)
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        print("Shutting down AlphaTrader...")
        
        # Halt trading
        self.redis.set('global:halt', 'true')
        
        # Close all positions
        await self.modules['emergency_manager'].emergency_close_all()
        
        # Wait for cleanup
        await asyncio.sleep(5)
        
        print("Shutdown complete")


async def main():
    trader = AlphaTrader()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Configuration Files

### config/config.yaml

```yaml
# Redis Configuration
redis:
  host: localhost
  port: 6379
  decode_responses: true

# IBKR Configuration
ibkr:
  host: 127.0.0.1
  port: 7497  # Paper trading port
  client_id: 1
  account: DU1234567  # Paper account

# Alpha Vantage Configuration
alpha_vantage:
  api_key: YOUR_API_KEY_HERE
  calls_per_minute: 600

# Trading Configuration
trading:
  max_positions: 5
  max_per_symbol: 2
  daily_loss_limit: 2000
  max_consecutive_losses: 3
  
# Risk Management
risk:
  kelly_fraction: 0.25
  max_correlation: 0.7
  max_drawdown: 0.10
  
# Strategy Allocations
strategies:
  0DTE:
    max_allocation: 0.05
    time_window: "09:45-15:00"
  1DTE:
    max_allocation: 0.07
    time_window: "14:00-15:30"
  14DTE:
    max_allocation: 0.10
    time_window: "09:30-16:00"
  MOC:
    max_allocation: 0.15
    time_window: "15:30-15:50"

# Distribution
distribution:
  discord:
    basic_webhook: YOUR_WEBHOOK_URL
    premium_webhook: YOUR_WEBHOOK_URL
  
# Dashboard
dashboard:
  host: 0.0.0.0
  port: 8000

# Twitter Configuration
twitter:
  bearer_token: YOUR_BEARER_TOKEN
  consumer_key: YOUR_CONSUMER_KEY
  consumer_secret: YOUR_CONSUMER_SECRET
  access_token: YOUR_ACCESS_TOKEN
  access_token_secret: YOUR_ACCESS_TOKEN_SECRET
  
# Telegram Configuration
telegram:
  bot_token: YOUR_TELEGRAM_BOT_TOKEN
  channels:
    public: "@alphatrader_public"
    basic: "@alphatrader_basic"
    premium: "@alphatrader_premium"
  
# OpenAI Configuration
openai:
  api_key: YOUR_OPENAI_API_KEY
  
# Stripe Configuration
stripe:
  api_key: YOUR_STRIPE_API_KEY
  
# Scheduled Tasks
scheduled_tasks:
  morning_analysis: "08:00"
  market_open: "09:30"
  midday_update: "12:00"
  market_close: "16:00"
  evening_wrapup: "18:00"
```

### requirements.txt

```
redis==5.0.1
ib_insync==0.9.86
aiohttp==3.9.1
fastapi==0.109.0
uvicorn==0.27.0
numpy==1.26.2
scikit-learn==1.3.2
statsmodels==0.14.1
pyyaml==6.0.1
websockets==12.0
tweepy==4.14.0
python-telegram-bot==20.7
stripe==7.8.0
openai==1.6.1
yfinance==0.2.33
```

---

## Deployment Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Redis
```bash
redis-server --maxmemory 4gb --maxmemory-policy volatile-lru
```

### 3. Start IBKR Gateway/TWS
- Open TWS or IB Gateway
- Enable API connections
- Set port to 7497 for paper trading

### 4. Configure API Keys
- Add Alpha Vantage API key to config/config.yaml
- Add Discord webhooks if using

### 5. Run the System
```bash
python main.py
```

### 6. Access Dashboard
- Open browser to http://localhost:8000
- Monitor all metrics in real-time

---

## 13. Twitter Integration Module

```python
import tweepy
import asyncio
import json
import time
from datetime import datetime, timedelta
import hashlib

class TwitterBot:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Twitter API v2 credentials
        self.client = tweepy.Client(
            bearer_token="YOUR_BEARER_TOKEN",
            consumer_key="YOUR_CONSUMER_KEY",
            consumer_secret="YOUR_CONSUMER_SECRET",
            access_token="YOUR_ACCESS_TOKEN",
            access_token_secret="YOUR_ACCESS_TOKEN_SECRET"
        )
        
        # Track posted signals to avoid duplicates
        self.posted_signals = set()
        
    async def start(self):
        """Main Twitter posting loop"""
        
        while True:
            try:
                # Post winning trades
                await self.post_winning_trades()
                
                # Post signal teasers
                await self.post_signal_teasers()
                
                # Post daily performance
                if datetime.now().hour == 16 and datetime.now().minute == 0:
                    await self.post_daily_summary()
                
                # Post market analysis preview (premium content teaser)
                if datetime.now().hour == 8 and datetime.now().minute == 30:
                    await self.post_analysis_teaser()
                
            except Exception as e:
                print(f"Twitter error: {e}")
                
            await asyncio.sleep(60)  # Check every minute
    
    async def post_winning_trades(self):
        """Post closed winning positions to Twitter"""
        
        # Get recent closed positions
        position_keys = self.redis.keys('positions:*:*')
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                
                # Check if recently closed and profitable
                if (position['status'] == 'CLOSED' and 
                    position['pnl_realized'] > 0 and
                    time.time() - position.get('exit_time', 0) < 300):  # Last 5 mins
                    
                    # Create unique hash to avoid reposting
                    trade_hash = hashlib.md5(f"{position['position_id']}".encode()).hexdigest()
                    
                    if trade_hash not in self.posted_signals:
                        self.posted_signals.add(trade_hash)
                        
                        # Calculate return percentage
                        return_pct = (position['pnl_realized'] / 
                                     (position['entry_price'] * position['size'])) * 100
                        
                        # Create tweet
                        tweet = self.format_winning_trade_tweet(position, return_pct)
                        
                        # Post to Twitter
                        try:
                            response = self.client.create_tweet(text=tweet)
                            print(f"Posted winning trade to Twitter: {response.data['id']}")
                            
                            # Track engagement
                            self.redis.setex(
                                f'twitter:tweet:{response.data["id"]}',
                                86400,
                                json.dumps({
                                    'type': 'winning_trade',
                                    'position_id': position['position_id'],
                                    'timestamp': time.time()
                                })
                            )
                        except Exception as e:
                            print(f"Failed to post tweet: {e}")
    
    def format_winning_trade_tweet(self, position, return_pct):
        """Format winning trade for Twitter"""
        
        emoji_map = {
            '0DTE': '⚡',
            '1DTE': '🌙',
            '14DTE': '📊',
            'MOC': '🔔'
        }
        
        strategy_emoji = emoji_map.get(position['strategy'], '📈')
        
        # Different formats for different return levels
        if return_pct > 100:
            tweet = f"""
{strategy_emoji} BANGER ALERT {strategy_emoji}

${position['symbol']} {position['strategy']} 
+{return_pct:.1f}% GAIN 🚀

Entry: ${position['entry_price']:.2f}
Exit: ${position.get('exit_price', 0):.2f}
P&L: ${position['pnl_realized']:.2f}

Premium members got this in real-time.
Join the feast: whop.com/alphatrader
"""
        elif return_pct > 50:
            tweet = f"""
{strategy_emoji} {position['symbol']} {position['strategy']} printed!

+{return_pct:.1f}% gain 💰
P&L: ${position['pnl_realized']:.2f}

This is why we trade with data, not emotion.

🔥 More signals daily: whop.com/alphatrader
"""
        else:
            tweet = f"""
✅ ${position['symbol']}: +{return_pct:.1f}%

Another win for the {position['strategy']} strategy.
Slow and steady wins the race.

Get all signals: whop.com/alphatrader
"""
        
        return tweet[:280]  # Twitter character limit
    
    async def post_signal_teasers(self):
        """Post teasers for high-confidence signals"""
        
        for symbol in ['SPY', 'QQQ', 'TSLA', 'NVDA']:
            signal_json = self.redis.lindex(f'signals:{symbol}:active', 0)
            
            if signal_json:
                signal = json.loads(signal_json)
                
                # Only post high confidence signals
                if signal['confidence'] > 85:
                    signal_hash = hashlib.md5(f"{signal['signal_id']}".encode()).hexdigest()
                    
                    if signal_hash not in self.posted_signals:
                        self.posted_signals.add(signal_hash)
                        
                        tweet = f"""
🎯 HIGH CONFIDENCE SIGNAL 🎯

${signal['symbol']} looking {signal['action']}ISH
Confidence: {signal['confidence']:.0f}%
Strategy: {signal['strategy']}

Premium members already positioned.
Basic members get it in 60 seconds.

Real-time access: whop.com/alphatrader
"""
                        
                        try:
                            self.client.create_tweet(text=tweet)
                        except Exception as e:
                            print(f"Failed to post signal teaser: {e}")
    
    async def post_daily_summary(self):
        """Post daily performance summary"""
        
        # Get daily stats
        daily_pnl = float(self.redis.get('global:pnl:realized') or 0)
        signal_count = int(self.redis.get('signals:global:count') or 0)
        win_rate = self.calculate_win_rate()
        
        # Get best trade of the day
        best_trade = self.get_best_trade_today()
        
        if daily_pnl > 0:
            tweet = f"""
📊 DAILY RESULTS 📊

P&L: +${daily_pnl:.2f} ✅
Signals: {signal_count}
Win Rate: {win_rate:.1f}%
"""
            if best_trade:
                tweet += f"""
Best Trade: ${best_trade['symbol']} +{best_trade['return']:.1f}%
"""
            tweet += """
Another profitable day in the books.
Join us tomorrow: whop.com/alphatrader
"""
        else:
            tweet = f"""
📊 Daily Summary

P&L: ${daily_pnl:.2f}
Signals: {signal_count}
Win Rate: {win_rate:.1f}%

Every day is a learning opportunity.
Risk management is key.

Join the journey: whop.com/alphatrader
"""
        
        try:
            self.client.create_tweet(text=tweet[:280])
        except Exception as e:
            print(f"Failed to post daily summary: {e}")
    
    async def post_analysis_teaser(self):
        """Post morning analysis preview"""
        
        # Get market analysis preview
        analysis = json.loads(self.redis.get('analysis:morning:preview') or '{}')
        
        if analysis:
            tweet = f"""
☕ MORNING MARKET ANALYSIS ☕

{analysis.get('headline', 'Major levels identified')}

Key Levels:
• SPY: {analysis.get('spy_level', 'N/A')}
• QQQ: {analysis.get('qqq_level', 'N/A')}

Full analysis with entry points available for premium members.

Read now: whop.com/alphatrader
"""
            
            try:
                self.client.create_tweet(text=tweet)
            except Exception as e:
                print(f"Failed to post analysis teaser: {e}")
    
    def calculate_win_rate(self):
        """Calculate today's win rate"""
        position_keys = self.redis.keys('positions:*:*')
        wins = 0
        total = 0
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                if position['status'] == 'CLOSED':
                    total += 1
                    if position['pnl_realized'] > 0:
                        wins += 1
        
        return (wins / total * 100) if total > 0 else 0
    
    def get_best_trade_today(self):
        """Get best performing trade of the day"""
        position_keys = self.redis.keys('positions:*:*')
        best_return = 0
        best_trade = None
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                if position['status'] == 'CLOSED':
                    return_pct = (position['pnl_realized'] / 
                                 (position['entry_price'] * position['size'])) * 100
                    if return_pct > best_return:
                        best_return = return_pct
                        best_trade = {
                            'symbol': position['symbol'],
                            'return': return_pct
                        }
        
        return best_trade
```

---

## 14. Telegram Integration Module

```python
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters
import stripe

class TelegramBot:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.bot_token = "YOUR_TELEGRAM_BOT_TOKEN"
        self.bot = Bot(token=self.bot_token)
        self.stripe = stripe
        self.stripe.api_key = "YOUR_STRIPE_KEY"
        
        # Channel IDs
        self.channels = {
            'public': '@alphatrader_public',
            'premium': '@alphatrader_premium',
            'basic': '@alphatrader_basic'
        }
        
        # Payment links (using Telegram's native payments or Stripe)
        self.payment_links = {
            'basic': 'https://t.me/alphatrader_bot?start=basic_plan',
            'premium': 'https://t.me/alphatrader_bot?start=premium_plan'
        }
        
    async def start(self):
        """Initialize Telegram bot with handlers"""
        
        # Create application
        app = Application.builder().token(self.bot_token).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("subscribe", self.subscribe_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CommandHandler("help", self.help_command))
        
        # Add payment handler
        app.add_handler(CallbackQueryHandler(self.handle_payment))
        
        # Start the bot
        await app.initialize()
        await app.start()
        
        # Main loop for sending signals
        while True:
            await self.check_and_send_signals()
            await asyncio.sleep(1)
    
    async def start_command(self, update: Update, context):
        """Handle /start command"""
        
        user_id = update.effective_user.id
        
        # Check if user came with a referral/plan parameter
        if context.args:
            plan = context.args[0]
            if plan in ['basic_plan', 'premium_plan']:
                await self.initiate_payment(update, plan)
                return
        
        welcome_message = """
🚀 Welcome to AlphaTrader Signals!

I provide institutional-grade options trading signals based on:
• Level 2 order flow analysis
• Options gamma exposure
• Hidden order detection
• AI-powered market regime classification

Choose your plan:
/subscribe - View subscription options
/status - Check your current subscription
/help - Get help

Free members get delayed signals in @alphatrader_public
"""
        
        await update.message.reply_text(welcome_message)
    
    async def subscribe_command(self, update: Update, context):
        """Show subscription options"""
        
        keyboard = [
            [InlineKeyboardButton("💎 Premium ($149/mo)", callback_data='subscribe_premium')],
            [InlineKeyboardButton("📊 Basic ($49/mo)", callback_data='subscribe_basic')],
            [InlineKeyboardButton("🆓 Free Trial (3 days)", callback_data='subscribe_trial')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = """
📈 Choose Your Trading Edge:

💎 **Premium** ($149/month)
• Real-time signals (0 delay)
• All strategies (0DTE, 1DTE, 14DTE, MOC)
• Entry, stops, and 3 targets
• Options contract specifics
• Morning market analysis
• Premium-only channel access

📊 **Basic** ($49/month)
• 60-second delayed signals
• Symbol, direction, strategy
• Basic channel access
• Daily performance updates

🆓 **Free Trial** (3 days)
• Full premium access
• Then converts to free tier
• 5-minute delayed signals
• Limited to 2 signals/day
"""
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def handle_payment(self, update: Update, context):
        """Handle subscription payment"""
        
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        plan = query.data.replace('subscribe_', '')
        
        if plan == 'trial':
            # Activate free trial
            self.redis.setex(
                f'telegram:user:{user_id}:tier',
                259200,  # 3 days
                'premium'
            )
            
            # Add to premium channel
            invite_link = await self.bot.create_chat_invite_link(
                self.channels['premium'],
                member_limit=1
            )
            
            await query.edit_message_text(
                f"✅ 3-day trial activated!\n\nJoin premium channel: {invite_link.invite_link}"
            )
            
        else:
            # Create Stripe checkout session
            session = self.stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'AlphaTrader {plan.title()} Plan',
                        },
                        'unit_amount': 14900 if plan == 'premium' else 4900,
                        'recurring': {'interval': 'month'}
                    },
                    'quantity': 1,
                }],
                mode='subscription',
                success_url='https://t.me/alphatrader_bot?start=payment_success',
                metadata={'telegram_user_id': str(user_id), 'plan': plan}
            )
            
            await query.edit_message_text(
                f"Complete your subscription:\n{session.url}\n\nAfter payment, you'll receive the private channel invite."
            )
    
    async def check_and_send_signals(self):
        """Check for new signals and distribute to Telegram channels"""
        
        # Check for active signals
        for symbol in ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA']:
            signal_json = self.redis.rpop(f'telegram:signals:{symbol}:queue')
            
            if signal_json:
                signal = json.loads(signal_json)
                await self.distribute_signal(signal)
    
    async def distribute_signal(self, signal):
        """Send signal to appropriate Telegram channels"""
        
        # Premium signal (full details)
        premium_message = self.format_premium_signal(signal)
        await self.bot.send_message(
            self.channels['premium'],
            premium_message,
            parse_mode='Markdown'
        )
        
        # Basic signal (60 second delay)
        await asyncio.sleep(60)
        basic_message = self.format_basic_signal(signal)
        await self.bot.send_message(
            self.channels['basic'],
            basic_message,
            parse_mode='Markdown'
        )
        
        # Public teaser (5 minute delay)
        await asyncio.sleep(240)
        public_message = f"""
🔔 Signal Alert

${signal['symbol']} - {signal['action']}

Premium members positioned 5 minutes ago.
Get real-time signals: /subscribe
"""
        await self.bot.send_message(
            self.channels['public'],
            public_message
        )
    
    def format_premium_signal(self, signal):
        """Format signal for premium members"""
        
        return f"""
🎯 **PREMIUM SIGNAL**

**Symbol:** ${signal['symbol']}
**Action:** {signal['action']}
**Strategy:** {signal['strategy']}
**Confidence:** {signal['confidence']:.0f}%

**Entry:** ${signal['entry_price']:.2f}
**Stop Loss:** ${signal['stop_loss']:.2f}
**Target 1:** ${signal['targets'][0]:.2f}
**Target 2:** ${signal['targets'][1]:.2f}
**Target 3:** ${signal['targets'][2]:.2f}

**Contract:**
• Type: {signal['contract']['contract_type']}
• Strike: ${signal['contract']['strike']}
• Expiry: {signal['contract']['expiry']}

**Reason:** {signal['reason']}

**Risk:** ${signal['max_risk']:.2f}
**Size:** {signal['position_size']:.0f}
"""
    
    def format_basic_signal(self, signal):
        """Format signal for basic members"""
        
        return f"""
📊 **BASIC SIGNAL**

**Symbol:** ${signal['symbol']}
**Action:** {signal['action']}
**Strategy:** {signal['strategy']}
**Confidence Range:** {'HIGH' if signal['confidence'] > 80 else 'MEDIUM'}

Upgrade to premium for entries, stops, and targets.
"""
    
    async def send_morning_analysis(self, analysis):
        """Send morning market analysis to premium Telegram"""
        
        message = f"""
☕ **MORNING MARKET ANALYSIS**
{datetime.now().strftime('%B %d, %Y')}

{analysis['content']}

**Trading Plan:**
{analysis['trading_plan']}

**Risk Notes:**
{analysis['risk_notes']}

_This analysis is for premium members only._
"""
        
        await self.bot.send_message(
            self.channels['premium'],
            message,
            parse_mode='Markdown'
        )
```

---

## 15. Morning Market Analysis Generator

```python
import openai
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

class MarketAnalysisGenerator:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.openai_client = openai.Client(api_key="YOUR_OPENAI_KEY")
        
    async def generate_morning_analysis(self):
        """Generate comprehensive morning analysis for premium users"""
        
        print(f"Generating morning analysis for {datetime.now().strftime('%Y-%m-%d')}")
        
        # Gather market data
        market_data = await self.gather_overnight_data()
        
        # Get technical levels
        technical_levels = self.calculate_key_levels()
        
        # Get options positioning
        options_data = self.analyze_options_positioning()
        
        # Get economic calendar
        economic_events = self.get_economic_calendar()
        
        # Generate analysis using GPT-4
        analysis = self.generate_ai_analysis(
            market_data,
            technical_levels,
            options_data,
            economic_events
        )
        
        # Store in Redis
        self.redis.setex(
            'analysis:morning:full',
            86400,
            json.dumps(analysis)
        )
        
        # Create preview for Twitter/public
        preview = {
            'headline': analysis['headline'],
            'spy_level': technical_levels['SPY']['pivot'],
            'qqq_level': technical_levels['QQQ']['pivot'],
            'vix_level': market_data['vix_close']
        }
        self.redis.setex(
            'analysis:morning:preview',
            86400,
            json.dumps(preview)
        )
        
        # Distribute to all platforms
        await self.distribute_analysis(analysis)
        
        return analysis
    
    async def gather_overnight_data(self):
        """Gather overnight futures and international markets data"""
        
        data = {}
        
        # Get futures data
        futures_symbols = {
            'ES=F': 'S&P Futures',
            'NQ=F': 'Nasdaq Futures',
            'YM=F': 'Dow Futures',
            'RTY=F': 'Russell Futures',
            'VX=F': 'VIX Futures'
        }
        
        for symbol, name in futures_symbols.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                data[name] = {
                    'close': hist['Close'].iloc[-1],
                    'change': (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100,
                    'volume': hist['Volume'].sum()
                }
        
        # Get international markets
        intl_symbols = {
            '^N225': 'Nikkei',
            '^HSI': 'Hang Seng',
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX'
        }
        
        for symbol, name in intl_symbols.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                data[name] = {
                    'close': hist['Close'].iloc[-1],
                    'change': (hist['Close'].iloc[-1] / hist['Open'].iloc[0] - 1) * 100
                }
        
        # Get VIX
        vix = yf.Ticker('^VIX')
        vix_hist = vix.history(period="5d")
        data['vix_close'] = vix_hist['Close'].iloc[-1]
        data['vix_change'] = (vix_hist['Close'].iloc[-1] / vix_hist['Close'].iloc[-2] - 1) * 100
        
        return data
    
    def calculate_key_levels(self):
        """Calculate key technical levels for major symbols"""
        
        levels = {}
        symbols = ['SPY', 'QQQ', 'IWM']
        
        for symbol in symbols:
            # Get recent bars from Redis
            bars_json = self.redis.get(f'market:{symbol}:bars')
            
            if bars_json:
                bars = json.loads(bars_json)
                df = pd.DataFrame(bars)
                
                # Calculate levels
                high = df['high'].max()
                low = df['low'].min()
                close = df['close'].iloc[-1]
                
                # Pivot points
                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                r2 = pivot + (high - low)
                s1 = 2 * pivot - high
                s2 = pivot - (high - low)
                
                # Volume profile (simplified)
                volume_weighted_price = (df['close'] * df['volume']).sum() / df['volume'].sum()
                
                levels[symbol] = {
                    'pivot': round(pivot, 2),
                    'r1': round(r1, 2),
                    'r2': round(r2, 2),
                    's1': round(s1, 2),
                    's2': round(s2, 2),
                    'vwap': round(volume_weighted_price, 2),
                    'yesterday_close': close
                }
            else:
                # Fallback to Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    high = hist['High'].iloc[-1]
                    low = hist['Low'].iloc[-1]
                    close = hist['Close'].iloc[-1]
                    
                    pivot = (high + low + close) / 3
                    
                    levels[symbol] = {
                        'pivot': round(pivot, 2),
                        'r1': round(2 * pivot - low, 2),
                        's1': round(2 * pivot - high, 2),
                        'yesterday_close': round(close, 2)
                    }
        
        return levels
    
    def analyze_options_positioning(self):
        """Analyze options positioning from Redis data"""
        
        positioning = {}
        
        for symbol in ['SPY', 'QQQ', 'IWM']:
            # Get GEX and DEX from Redis
            gex_json = self.redis.get(f'metrics:{symbol}:gex')
            dex_json = self.redis.get(f'metrics:{symbol}:dex')
            
            if gex_json and dex_json:
                gex = json.loads(gex_json)
                dex = json.loads(dex_json)
                
                positioning[symbol] = {
                    'gamma_exposure': gex.get('total', 0),
                    'gamma_pin': gex.get('pin', 0),
                    'gamma_flip': gex.get('flip', 0),
                    'delta_exposure': dex.get('total', 0),
                    'positioning': 'BULLISH' if dex.get('total', 0) > 0 else 'BEARISH'
                }
            else:
                positioning[symbol] = {
                    'gamma_exposure': 0,
                    'gamma_pin': 0,
                    'gamma_flip': 0,
                    'delta_exposure': 0,
                    'positioning': 'NEUTRAL'
                }
        
        return positioning
    
    def get_economic_calendar(self):
        """Get today's economic events"""
        
        # In production, this would pull from an economic calendar API
        # For now, return placeholder
        
        events = [
            {
                'time': '08:30',
                'event': 'Initial Jobless Claims',
                'importance': 'HIGH',
                'forecast': '230K',
                'previous': '225K'
            },
            {
                'time': '10:00',
                'event': 'Consumer Sentiment',
                'importance': 'MEDIUM',
                'forecast': '71.0',
                'previous': '69.5'
            }
        ]
        
        return events
    
    def generate_ai_analysis(self, market_data, technical_levels, options_data, economic_events):
        """Use GPT-4 to generate professional analysis"""
        
        # Prepare context for GPT-4
        context = f"""
        You are a professional options trader preparing a morning analysis.
        
        Market Data:
        - S&P Futures: {market_data.get('S&P Futures', {}).get('change', 0):.2f}%
        - VIX: {market_data.get('vix_close', 0):.2f} ({market_data.get('vix_change', 0):.2f}%)
        
        Key Levels:
        SPY: Pivot {technical_levels.get('SPY', {}).get('pivot', 0)}, R1 {technical_levels.get('SPY', {}).get('r1', 0)}
        
        Options Positioning:
        SPY Gamma: {options_data.get('SPY', {}).get('gamma_exposure', 0)}M
        SPY Pin: {options_data.get('SPY', {}).get('gamma_pin', 0)}
        
        Economic Events Today:
        {json.dumps(economic_events, indent=2)}
        
        Generate a concise morning analysis with:
        1. Market outlook (bullish/bearish/neutral)
        2. Key levels to watch
        3. Potential trade setups
        4. Risk factors
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional derivatives trader."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        analysis_text = response.choices[0].message.content
        
        # Structure the analysis
        analysis = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'headline': self.extract_headline(analysis_text),
            'content': analysis_text,
            'market_data': market_data,
            'technical_levels': technical_levels,
            'options_positioning': options_data,
            'economic_calendar': economic_events,
            'trading_plan': self.generate_trading_plan(technical_levels, options_data),
            'risk_notes': self.generate_risk_notes(market_data)
        }
        
        return analysis
    
    def extract_headline(self, analysis_text):
        """Extract headline from analysis"""
        lines = analysis_text.split('\n')
        for line in lines:
            if line.strip():
                return line.strip()[:100]
        return "Market Analysis Available"
    
    def generate_trading_plan(self, levels, options):
        """Generate specific trading plan"""
        
        plan = []
        
        for symbol in ['SPY', 'QQQ']:
            if symbol in levels and symbol in options:
                level_data = levels[symbol]
                option_data = options[symbol]
                
                # Bullish setup
                if option_data['positioning'] == 'BULLISH':
                    plan.append(f"• {symbol}: Look for calls above {level_data['pivot']}, target {level_data['r1']}")
                
                # Bearish setup
                elif option_data['positioning'] == 'BEARISH':
                    plan.append(f"• {symbol}: Consider puts below {level_data['pivot']}, target {level_data['s1']}")
                
                # Gamma pin setup
                if abs(option_data['gamma_pin'] - level_data['yesterday_close']) < 1:
                    plan.append(f"• {symbol}: Gamma pin at {option_data['gamma_pin']}, expect range-bound action")
        
        return '\n'.join(plan) if plan else "Wait for clearer setups to develop"
    
    def generate_risk_notes(self, market_data):
        """Generate risk warnings"""
        
        risks = []
        
        # VIX check
        if market_data.get('vix_close', 0) > 20:
            risks.append("• Elevated VIX indicates higher volatility expected")
        
        # Futures divergence
        sp_change = market_data.get('S&P Futures', {}).get('change', 0)
        nq_change = market_data.get('Nasdaq Futures', {}).get('change', 0)
        
        if abs(sp_change - nq_change) > 0.5:
            risks.append("• Futures divergence suggests sector rotation")
        
        # International markets
        if any(market_data.get(mkt, {}).get('change', 0) < -2 
               for mkt in ['Nikkei', 'Hang Seng', 'DAX']):
            risks.append("• International markets showing weakness")
        
        return '\n'.join(risks) if risks else "Normal risk parameters apply"
    
    async def distribute_analysis(self, analysis):
        """Distribute analysis to all platforms"""
        
        # Queue for Discord premium channel
        self.redis.lpush('distribution:premium:analysis', json.dumps({
            'type': 'MORNING_ANALYSIS',
            'content': analysis['content'],
            'levels': analysis['technical_levels'],
            'plan': analysis['trading_plan']
        }))
        
        # Queue for Telegram
        self.redis.lpush('telegram:analysis:queue', json.dumps(analysis))
        
        # Store for web dashboard
        self.redis.setex('dashboard:morning_analysis', 86400, json.dumps(analysis))
        
        print(f"Morning analysis distributed at {datetime.now()}")
```

---

## 16. Scheduled Tasks Coordinator

```python
class ScheduledTasks:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.twitter = TwitterBot()
        self.telegram = TelegramBot()
        self.analysis = MarketAnalysisGenerator()
        
    async def start(self):
        """Run scheduled tasks"""
        
        while True:
            current_time = datetime.now()
            
            # Morning analysis (8:00 AM EST)
            if current_time.hour == 8 and current_time.minute == 0:
                await self.run_morning_routine()
            
            # Market open routine (9:30 AM EST)
            elif current_time.hour == 9 and current_time.minute == 30:
                await self.market_open_routine()
            
            # Midday update (12:00 PM EST)
            elif current_time.hour == 12 and current_time.minute == 0:
                await self.midday_update()
            
            # Market close routine (4:00 PM EST)
            elif current_time.hour == 16 and current_time.minute == 0:
                await self.market_close_routine()
            
            # Evening wrap-up (6:00 PM EST)
            elif current_time.hour == 18 and current_time.minute == 0:
                await self.evening_wrapup()
            
            await asyncio.sleep(60)  # Check every minute
    
    async def run_morning_routine(self):
        """8:00 AM - Generate and distribute morning analysis"""
        
        print("Running morning routine...")
        
        # Generate analysis
        analysis = await self.analysis.generate_morning_analysis()
        
        # Post to all platforms
        await self.twitter.post_analysis_teaser()
        await self.telegram.send_morning_analysis(analysis)
        
        # Update dashboard
        self.redis.setex('status:morning_analysis', 86400, 'completed')
    
    async def market_open_routine(self):
        """9:30 AM - Market open tasks"""
        
        # Reset daily counters
        self.redis.set('signals:global:count', 0)
        self.redis.set('global:pnl:realized', 0)
        
        # Post market open message
        message = "🔔 Markets are open! Systems online and scanning for opportunities."
        
        # Twitter
        self.twitter.client.create_tweet(text=message)
        
        # Telegram
        await self.telegram.bot.send_message(
            self.telegram.channels['public'],
            message
        )
    
    async def midday_update(self):
        """12:00 PM - Midday performance update"""
        
        stats = {
            'signals': int(self.redis.get('signals:global:count') or 0),
            'pnl': float(self.redis.get('global:pnl:realized') or 0),
            'positions': int(self.redis.get('global:positions:count') or 0)
        }
        
        message = f"""
📊 Midday Update

Signals Generated: {stats['signals']}
Open Positions: {stats['positions']}
P&L: ${stats['pnl']:.2f}

Afternoon session starting. Stay focused!
"""
        
        # Post to Twitter and Telegram
        self.twitter.client.create_tweet(text=message)
        await self.telegram.bot.send_message(
            self.telegram.channels['public'],
            message
        )
    
    async def market_close_routine(self):
        """4:00 PM - Market close and daily summary"""
        
        # Generate daily report
        daily_stats = self.generate_daily_report()
        
        # Post to all platforms
        await self.twitter.post_daily_summary()
        
        # Send detailed report to premium members
        await self.send_premium_daily_report(daily_stats)
    
    async def evening_wrapup(self):
        """6:00 PM - Evening wrap-up and next day preview"""
        
        # Analyze after-hours movement
        ah_analysis = self.analyze_after_hours()
        
        # Prepare next day watchlist
        watchlist = self.generate_watchlist()
        
        message = f"""
🌙 Evening Wrap-Up

After-Hours Movement:
{ah_analysis}

Tomorrow's Watchlist:
{watchlist}

Rest well, traders. See you tomorrow!
"""
        
        # Premium members only
        await self.telegram.bot.send_message(
            self.telegram.channels['premium'],
            message
        )
    
    def generate_daily_report(self):
        """Generate comprehensive daily report"""
        
        # Gather all stats
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signals_generated': int(self.redis.get('signals:global:count') or 0),
            'trades_executed': self.count_executed_trades(),
            'win_rate': self.calculate_win_rate(),
            'total_pnl': float(self.redis.get('global:pnl:realized') or 0),
            'best_trade': self.get_best_trade(),
            'worst_trade': self.get_worst_trade()
        }
    
    def calculate_win_rate(self):
        """Calculate win rate from closed positions"""
        position_keys = self.redis.keys('positions:*:*')
        wins = 0
        total = 0
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                if position['status'] == 'CLOSED':
                    total += 1
                    if position['pnl_realized'] > 0:
                        wins += 1
        
        return (wins / total * 100) if total > 0 else 0
    
    def count_executed_trades(self):
        """Count executed trades for the day"""
        position_keys = self.redis.keys('positions:*:*')
        count = 0
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                # Check if position was opened today
                if position.get('entry_time', 0) > time.time() - 86400:
                    count += 1
        
        return count
    
    def get_best_trade(self):
        """Get best trade of the day"""
        position_keys = self.redis.keys('positions:*:*')
        best_pnl = 0
        best_trade = None
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                if position['status'] == 'CLOSED' and position['pnl_realized'] > best_pnl:
                    best_pnl = position['pnl_realized']
                    best_trade = {
                        'symbol': position['symbol'],
                        'pnl': best_pnl,
                        'strategy': position['strategy']
                    }
        
        return best_trade
    
    def get_worst_trade(self):
        """Get worst trade of the day"""
        position_keys = self.redis.keys('positions:*:*')
        worst_pnl = 0
        worst_trade = None
        
        for key in position_keys:
            position_json = self.redis.get(key)
            if position_json:
                position = json.loads(position_json)
                if position['status'] == 'CLOSED' and position['pnl_realized'] < worst_pnl:
                    worst_pnl = position['pnl_realized']
                    worst_trade = {
                        'symbol': position['symbol'],
                        'pnl': worst_pnl,
                        'strategy': position['strategy']
                    }
        
        return worst_trade
    
    def analyze_after_hours(self):
        """Analyze after-hours movement"""
        # Simplified - would check actual AH data
        return "SPY +0.15%, QQQ +0.22% in after-hours"
    
    def generate_watchlist(self):
        """Generate next day watchlist based on signals"""
        # Pull from queued signals and analysis
        return "SPY calls above 450, QQQ puts below 375"
    
    async def send_premium_daily_report(self, stats):
        """Send detailed daily report to premium members"""
        
        report = f"""
📈 **DAILY PERFORMANCE REPORT**
{stats['date']}

**Statistics:**
• Signals Generated: {stats['signals_generated']}
• Trades Executed: {stats['trades_executed']}
• Win Rate: {stats['win_rate']:.1f}%
• Total P&L: ${stats['total_pnl']:.2f}
"""
        
        if stats['best_trade']:
            report += f"""

**Best Trade:**
{stats['best_trade']['symbol']} ({stats['best_trade']['strategy']})
P&L: ${stats['best_trade']['pnl']:.2f}
"""
        
        if stats['worst_trade']:
            report += f"""

**Worst Trade:**
{stats['worst_trade']['symbol']} ({stats['worst_trade']['strategy']})
P&L: ${stats['worst_trade']['pnl']:.2f}
"""
        
        # Send to premium Telegram
        await self.telegram.bot.send_message(
            self.telegram.channels['premium'],
            report,
            parse_mode='Markdown'
        )
        
        # Queue for Discord premium
        self.redis.lpush('distribution:premium:queue', json.dumps({
            'type': 'DAILY_REPORT',
            'content': report
        }))
```

---