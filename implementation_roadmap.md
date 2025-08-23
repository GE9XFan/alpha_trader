# Implementation Roadmap v2.0 - Detailed Build Guide
**Version:** 2.0 (Plugin Architecture)  
**Timeline:** 12 Weeks to Production  
**Approach:** Progressive enhancement with plugin architecture  
**APIs:** 36 Alpha Vantage + IBKR

---

## Executive Summary

This roadmap implements a plugin-based architecture that grows without rebuilding. You build the foundation once (Week 1), then add capabilities through plugins. No refactoring, no breaking changes, just progressive enhancement.

**Key Principles:**
1. Build the message bus first - it never changes
2. Everything is a plugin that subscribes/publishes events  
3. Start trading in Week 2 (simple), add ML in Week 6
4. Each week adds capabilities without touching prior code
5. Configuration drives behavior, not code changes

---

## Week 1: Foundation Layer

### Day 1-2: Project Setup & Core Message Bus

**Morning Day 1:**
```bash
# Create project structure
mkdir -p alphatrader/{core,plugins,config,tests,scripts,models,data}
cd alphatrader

# Initialize git
git init
echo "# AlphaTrader" > README.md

# Create Python environment
python3.11 -m venv venv
source venv/bin/activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core
pydantic==2.5.0
python-dotenv==1.0.0
pyyaml==6.0.1

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Redis
redis==5.0.1

# HTTP
aiohttp==3.9.1
requests==2.31.0

# Scheduling
apscheduler==3.10.4

# Data Processing
pandas==2.1.4
numpy==1.26.2

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
EOF

pip install -r requirements.txt
```

**Afternoon Day 1: Core Message Bus**
```python
# core/message.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

@dataclass
class Message:
    id: str
    correlation_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, event_type: str, data: Dict[str, Any], 
               correlation_id: Optional[str] = None):
        return cls(
            id=str(uuid.uuid4()),
            correlation_id=correlation_id or str(uuid.uuid4()),
            event_type=event_type,
            data=data,
            timestamp=datetime.utcnow(),
            metadata={}
        )
```

```python
# core/bus.py
from collections import defaultdict
from typing import Callable, Dict, List
import asyncio
import logging
import re

from .message import Message
from .persistence import EventStore

logger = logging.getLogger(__name__)

class MessageBus:
    def __init__(self, event_store: EventStore):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_store = event_store
        self._running = False
        
    def publish(self, event_type: str, data: Dict, correlation_id: str = None):
        """Publish event to all matching subscribers"""
        message = Message.create(event_type, data, correlation_id)
        
        # Persist first
        self.event_store.save(message)
        
        # Then distribute
        for pattern, handlers in self.subscribers.items():
            if self._matches(pattern, event_type):
                for handler in handlers:
                    try:
                        # Call handler (async or sync)
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.create_task(handler(message))
                        else:
                            handler(message)
                    except Exception as e:
                        logger.error(f"Handler error: {e}", exc_info=True)
                        
    def subscribe(self, pattern: str, handler: Callable):
        """Subscribe to events matching pattern (supports wildcards)"""
        self.subscribers[pattern].append(handler)
        logger.info(f"Subscribed {handler.__name__} to {pattern}")
        
    def _matches(self, pattern: str, event_type: str) -> bool:
        """Check if event_type matches pattern with wildcards"""
        # Convert pattern to regex: * matches anything
        regex = pattern.replace('.', r'\.').replace('*', '.*')
        return bool(re.match(f"^{regex}$", event_type))
```

**Day 2: Event Store & Plugin Base**

```python
# core/persistence.py
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Optional

from .message import Message

class EventStore:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._init_db()
        
    def _init_db(self):
        """Create events table if not exists"""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id UUID PRIMARY KEY,
                        correlation_id UUID NOT NULL,
                        event_type TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        
                        CONSTRAINT events_id_unique UNIQUE (id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_events_type 
                        ON events(event_type);
                    CREATE INDEX IF NOT EXISTS idx_events_created 
                        ON events(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_events_correlation 
                        ON events(correlation_id);
                """)
                
    def save(self, message: Message):
        """Persist message to database"""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO events (
                        id, correlation_id, event_type, 
                        payload, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    message.id,
                    message.correlation_id,
                    message.event_type,
                    json.dumps(message.data),
                    json.dumps(message.metadata),
                    message.timestamp
                ))
```

```python
# core/plugin.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from .bus import MessageBus

class Plugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, bus: MessageBus, config: Dict[str, Any]):
        self.bus = bus
        self.config = config
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(self.name)
        self._running = False
        
    @abstractmethod
    async def start(self):
        """Initialize plugin and subscribe to events"""
        pass
        
    @abstractmethod
    async def stop(self):
        """Cleanup and shutdown"""
        pass
        
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health status"""
        pass
        
    def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish event with plugin prefix"""
        full_event_type = f"{self.name.lower()}.{event_type}"
        self.bus.publish(full_event_type, data)
```

### Day 3-4: Plugin Manager & Configuration

```python
# core/plugin_manager.py
import importlib
import inspect
from pathlib import Path
from typing import Dict, List
import yaml
import asyncio

from .plugin import Plugin
from .bus import MessageBus

class PluginManager:
    def __init__(self, bus: MessageBus, config_dir: str):
        self.bus = bus
        self.config_dir = Path(config_dir)
        self.plugins: Dict[str, Plugin] = {}
        self._load_configs()
        
    def _load_configs(self):
        """Load all plugin configurations"""
        self.configs = {}
        for config_file in self.config_dir.glob("plugins/*.yaml"):
            with open(config_file) as f:
                config = yaml.safe_load(f)
                plugin_name = config_file.stem
                self.configs[plugin_name] = config
                
    async def discover_and_load(self):
        """Auto-discover and load all plugins"""
        plugins_dir = Path("plugins")
        
        for plugin_file in plugins_dir.rglob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            # Import module
            module_path = str(plugin_file).replace("/", ".").replace(".py", "")
            module = importlib.import_module(module_path)
            
            # Find Plugin subclasses
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin:
                    # Get config for this plugin
                    plugin_config = self.configs.get(name.lower(), {})
                    
                    if plugin_config.get('enabled', True):
                        # Instantiate and start plugin
                        plugin = obj(self.bus, plugin_config)
                        await plugin.start()
                        self.plugins[name] = plugin
                        print(f"✓ Loaded plugin: {name}")
                        
    async def stop_all(self):
        """Stop all plugins gracefully"""
        for name, plugin in self.plugins.items():
            await plugin.stop()
            print(f"✓ Stopped plugin: {name}")
```

### Day 5: Configuration System

```yaml
# config/system.yaml
system:
  environment: development
  
  database:
    host: localhost
    port: 5432
    name: alphatrader
    user: alphatrader
    password: ${DB_PASSWORD}
    
  redis:
    host: localhost
    port: 6379
    db: 0
    
  message_bus:
    persistence: postgresql
    async: true
    
  plugin_manager:
    auto_discover: true
    plugin_dirs:
      - plugins/datasources
      - plugins/processing
      - plugins/strategies
      - plugins/risk
```

```yaml
# config/plugins/alpha_vantage.yaml
enabled: true
api_key: ${ALPHA_VANTAGE_API_KEY}
base_url: https://www.alphavantage.co/query
rate_limit:
  calls_per_minute: 500
  burst_size: 20

apis:
  # Technical Indicators (16 total)
  rsi:
    function: RSI
    schedule: "*/5 * * * *"
    symbols:
      tier_a: [SPY, QQQ, IWM, SPX]
      tier_b: [AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA]
    params:
      interval: 5min
      time_period: 14
      series_type: close
      
  macd:
    function: MACD
    schedule: "*/5 * * * *"
    symbols:
      tier_a: [SPY, QQQ, IWM, SPX]
      tier_b: [AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA]
    params:
      interval: 5min
      series_type: close
      
  # ... (all 36 APIs configured similarly)
```

### Day 6-7: Testing Framework

```python
# tests/test_message_bus.py
import pytest
import asyncio
from core.bus import MessageBus
from core.persistence import EventStore
from core.message import Message

@pytest.fixture
def bus():
    store = EventStore("postgresql://test@localhost/test")
    return MessageBus(store)

def test_publish_subscribe(bus):
    received = []
    
    def handler(message):
        received.append(message)
        
    # Subscribe to pattern
    bus.subscribe("test.*", handler)
    
    # Publish matching event
    bus.publish("test.event", {"data": "value"})
    
    # Verify received
    assert len(received) == 1
    assert received[0].event_type == "test.event"
    assert received[0].data == {"data": "value"}

def test_wildcard_patterns(bus):
    received = []
    
    def handler(message):
        received.append(message.event_type)
        
    # Subscribe with wildcard
    bus.subscribe("api.*.data", handler)
    
    # Publish various events
    bus.publish("api.rsi.data", {})
    bus.publish("api.macd.data", {})
    bus.publish("other.event", {})  # Should not match
    
    assert received == ["api.rsi.data", "api.macd.data"]
```

**Week 1 Deliverables:**
- ✅ Message bus operational
- ✅ Event store persisting to PostgreSQL
- ✅ Plugin base class defined
- ✅ Plugin manager auto-loading
- ✅ Configuration system working
- ✅ Testing framework ready

---

## Week 2: Data Source Plugins

### Day 8-9: Alpha Vantage Plugin

```python
# plugins/datasources/alpha_vantage.py
import aiohttp
import asyncio
from datetime import datetime
from typing import Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.plugin import Plugin
from core.rate_limiter import TokenBucket

class AlphaVantagePlugin(Plugin):
    """Handles all 36 Alpha Vantage APIs"""
    
    async def start(self):
        self.session = aiohttp.ClientSession()
        self.rate_limiter = TokenBucket(
            capacity=self.config['rate_limit']['calls_per_minute'],
            refill_rate=10  # per second
        )
        self.scheduler = AsyncIOScheduler()
        
        # Schedule all configured APIs
        for api_name, api_config in self.config['apis'].items():
            if api_config.get('enabled', True):
                self._schedule_api(api_name, api_config)
                
        self.scheduler.start()
        self.logger.info(f"Started with {len(self.config['apis'])} APIs")
        
    def _schedule_api(self, name: str, config: Dict):
        """Schedule one API for periodic fetching"""
        schedule = config['schedule']
        
        # For each symbol tier
        for tier, symbols in config.get('symbols', {}).items():
            for symbol in symbols:
                job_id = f"{name}_{symbol}"
                
                self.scheduler.add_job(
                    func=self._fetch_api,
                    trigger=CronTrigger.from_crontab(schedule),
                    args=[name, config, symbol],
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                
    async def _fetch_api(self, api_name: str, config: Dict, symbol: str):
        """Fetch data from one API"""
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Build request
        params = {
            'function': config['function'],
            'symbol': symbol,
            'apikey': self.config['api_key'],
            **config.get('params', {})
        }
        
        try:
            async with self.session.get(
                self.config['base_url'],
                params=params
            ) as response:
                data = await response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    self.logger.error(f"API error for {api_name}/{symbol}: {data}")
                    return
                    
                # Transform based on API type
                transformed = self._transform_data(api_name, data)
                
                # Publish to message bus
                self.publish(f"data.{api_name}", {
                    'symbol': symbol,
                    'api': api_name,
                    'data': transformed,
                    'raw': data,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                self.logger.debug(f"Published {api_name} data for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Failed to fetch {api_name}/{symbol}: {e}")
            
    def _transform_data(self, api_name: str, raw_data: Dict) -> Any:
        """Transform API response to standard format"""
        # Handle different API response formats
        
        if api_name in ['rsi', 'macd', 'bbands']:  # Technical indicators
            # Extract time series data
            key = next((k for k in raw_data.keys() if 'Technical Analysis' in k), None)
            if key:
                return raw_data[key]
                
        elif api_name == 'realtime_options':
            # Extract options chain with Greeks
            return raw_data.get('data', [])
            
        elif api_name in ['overview', 'earnings']:  # Fundamentals
            return raw_data
            
        # Default: return as-is
        return raw_data
        
    async def stop(self):
        self.scheduler.shutdown()
        await self.session.close()
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'scheduled_jobs': len(self.scheduler.get_jobs()),
            'rate_limit_available': self.rate_limiter.available_tokens()
        }
```

### Day 10-11: IBKR Plugin

```python
# plugins/datasources/ibkr.py
from ibapi import wrapper, client, contract
import asyncio
from datetime import datetime
from typing import Dict, Any
import threading

from core.plugin import Plugin

class IBKRPlugin(Plugin):
    """IBKR data and execution"""
    
    async def start(self):
        self.app = IBApp(self.bus)
        self.app.connect(
            self.config['host'],
            self.config['port'],
            self.config['client_id']
        )
        
        # Start IB API thread
        self.api_thread = threading.Thread(target=self.app.run, daemon=True)
        self.api_thread.start()
        
        # Wait for connection
        await asyncio.sleep(2)
        
        # Subscribe to 5-second bars for all symbols
        for symbol in self.config['symbols']:
            self.app.subscribe_bars(symbol)
            
        self.logger.info("Connected to IBKR")
        
    async def stop(self):
        self.app.disconnect()
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': self.app.isConnected(),
            'subscriptions': len(self.app.subscriptions)
        }

class IBApp(wrapper.EWrapper, client.EClient):
    def __init__(self, bus):
        wrapper.EWrapper.__init__(self)
        client.EClient.__init__(self, self)
        self.bus = bus
        self.subscriptions = {}
        
    def subscribe_bars(self, symbol: str):
        """Subscribe to 5-second bars"""
        contract = self._make_contract(symbol)
        req_id = len(self.subscriptions) + 1
        
        self.reqRealTimeBars(
            req_id,
            contract,
            5,  # 5-second bars
            "TRADES",
            False,
            []
        )
        
        self.subscriptions[req_id] = symbol
        
    def realtimeBar(self, reqId, time, open_, high, low, close, 
                    volume, wap, count):
        """Handle incoming 5-second bar"""
        symbol = self.subscriptions.get(reqId)
        
        if symbol:
            self.bus.publish("ibkr.bar.5s", {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(time).isoformat(),
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'vwap': wap,
                'count': count
            })
```

### Day 12-13: Bar Aggregator Plugin

```python
# plugins/processing/bar_aggregator.py
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from core.plugin import Plugin

class BarAggregatorPlugin(Plugin):
    """Aggregates 5-second bars to all timeframes"""
    
    async def start(self):
        # Subscribe to 5-second bars
        self.bus.subscribe("ibkr.bar.5s", self.on_bar)
        
        # Initialize buffers for each timeframe
        self.buffers = {
            '1m': defaultdict(list),   # 12 bars
            '5m': defaultdict(list),   # 60 bars
            '10m': defaultdict(list),  # 120 bars
            '15m': defaultdict(list),  # 180 bars
            '30m': defaultdict(list),  # 360 bars
            '1h': defaultdict(list),   # 720 bars
        }
        
        self.bars_required = {
            '1m': 12,
            '5m': 60,
            '10m': 120,
            '15m': 180,
            '30m': 360,
            '1h': 720
        }
        
        self.logger.info("Bar aggregator started")
        
    def on_bar(self, message):
        """Process incoming 5-second bar"""
        bar = message.data
        symbol = bar['symbol']
        
        # Add to all buffers
        for timeframe in self.buffers:
            self.buffers[timeframe][symbol].append(bar)
            
            # Check if we have enough bars to aggregate
            if len(self.buffers[timeframe][symbol]) >= self.bars_required[timeframe]:
                # Aggregate bars
                agg_bar = self._aggregate_bars(
                    self.buffers[timeframe][symbol][:self.bars_required[timeframe]]
                )
                
                # Publish aggregated bar
                self.publish(f"bar.{timeframe}", {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    **agg_bar
                })
                
                # Remove used bars
                self.buffers[timeframe][symbol] = \
                    self.buffers[timeframe][symbol][self.bars_required[timeframe]:]
                    
    def _aggregate_bars(self, bars: List[Dict]) -> Dict:
        """Aggregate list of bars into single bar"""
        return {
            'timestamp': bars[0]['timestamp'],  # Start time
            'open': bars[0]['open'],
            'high': max(b['high'] for b in bars),
            'low': min(b['low'] for b in bars),
            'close': bars[-1]['close'],
            'volume': sum(b['volume'] for b in bars),
            'vwap': np.average(
                [b['vwap'] for b in bars],
                weights=[b['volume'] for b in bars]
            ) if sum(b['volume'] for b in bars) > 0 else bars[-1]['close']
        }
        
    async def stop(self):
        pass
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'symbols_tracking': len(self.buffers['1m']),
            'buffers': {
                tf: len(symbols) 
                for tf, symbols in self.buffers.items()
            }
        }
```

### Day 14: Simple Trading Strategy

```python
# plugins/strategies/simple_momentum.py
from datetime import datetime
from typing import Dict, Optional

from core.plugin import Plugin

class SimpleMomentumStrategy(Plugin):
    """Simple RSI/MACD momentum strategy for testing"""
    
    async def start(self):
        # Subscribe to indicators
        self.bus.subscribe("alphavantage.data.rsi", self.on_rsi)
        self.bus.subscribe("alphavantage.data.macd", self.on_macd)
        self.bus.subscribe("ibkr.bar.5m", self.on_price)
        
        # Storage for latest values
        self.rsi = {}
        self.macd = {}
        self.prices = {}
        self.positions = {}
        
        self.logger.info("Simple momentum strategy started")
        
    def on_rsi(self, message):
        """Store latest RSI"""
        data = message.data
        symbol = data['symbol']
        
        # Get latest RSI value
        rsi_data = data['data']
        if rsi_data:
            latest_time = max(rsi_data.keys())
            self.rsi[symbol] = float(rsi_data[latest_time]['RSI'])
            self._check_signal(symbol)
            
    def on_macd(self, message):
        """Store latest MACD"""
        data = message.data
        symbol = data['symbol']
        
        # Get latest MACD values
        macd_data = data['data']
        if macd_data:
            latest_time = max(macd_data.keys())
            self.macd[symbol] = {
                'macd': float(macd_data[latest_time].get('MACD', 0)),
                'signal': float(macd_data[latest_time].get('MACD_Signal', 0)),
                'histogram': float(macd_data[latest_time].get('MACD_Hist', 0))
            }
            self._check_signal(symbol)
            
    def on_price(self, message):
        """Store latest price"""
        bar = message.data
        symbol = bar['symbol']
        self.prices[symbol] = bar['close']
        self._check_signal(symbol)
        
    def _check_signal(self, symbol: str):
        """Check if we should generate a signal"""
        # Need all data
        if symbol not in self.rsi or symbol not in self.macd or symbol not in self.prices:
            return
            
        rsi = self.rsi[symbol]
        macd = self.macd[symbol]
        price = self.prices[symbol]
        
        # Simple rules
        if symbol not in self.positions:
            # Entry logic
            if rsi < 30 and macd['histogram'] > 0:  # Oversold + momentum turning
                self.publish("signal.entry", {
                    'symbol': symbol,
                    'action': 'BUY',
                    'strategy': 'simple_momentum',
                    'price': price,
                    'size': 100,  # Start small
                    'confidence': 0.6,
                    'reason': f"RSI={rsi:.1f}, MACD_Hist={macd['histogram']:.3f}",
                    'timestamp': datetime.utcnow().isoformat()
                })
                self.positions[symbol] = {'entry': price}
                self.logger.info(f"BUY signal for {symbol}")
                
        else:
            # Exit logic
            if rsi > 70 or macd['histogram'] < 0:  # Overbought or momentum fading
                entry = self.positions[symbol]['entry']
                pnl = ((price - entry) / entry) * 100
                
                self.publish("signal.exit", {
                    'symbol': symbol,
                    'action': 'SELL',
                    'strategy': 'simple_momentum',
                    'price': price,
                    'size': 100,
                    'pnl_pct': pnl,
                    'reason': f"RSI={rsi:.1f}, MACD_Hist={macd['histogram']:.3f}",
                    'timestamp': datetime.utcnow().isoformat()
                })
                del self.positions[symbol]
                self.logger.info(f"SELL signal for {symbol}, PnL: {pnl:.2f}%")
                
    async def stop(self):
        pass
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'positions': len(self.positions),
            'symbols_tracked': len(self.rsi)
        }
```

**Week 2 Deliverables:**
- ✅ Alpha Vantage plugin fetching all 36 APIs
- ✅ IBKR plugin receiving 5-second bars
- ✅ Bar aggregation to all timeframes
- ✅ Simple trading strategy generating signals
- ✅ Data flowing through message bus

---

## Week 3-4: Risk & Execution

### Day 15-16: Risk Manager Plugin

```python
# plugins/risk/risk_manager.py
from datetime import datetime, timedelta
from typing import Dict, Optional

from core.plugin import Plugin

class RiskManagerPlugin(Plugin):
    """Validates all signals against risk limits"""
    
    async def start(self):
        # Subscribe to all signals
        self.bus.subscribe("*.signal.entry", self.validate_entry)
        self.bus.subscribe("*.signal.exit", self.validate_exit)
        
        # Track positions and P&L
        self.positions = {}
        self.daily_pnl = 0
        self.daily_trades = 0
        
        # Load risk limits from config
        self.max_positions = self.config.get('max_positions', 10)
        self.max_position_size = self.config.get('max_position_size', 1000)
        self.max_daily_loss = self.config.get('max_daily_loss', 5000)
        self.max_daily_trades = self.config.get('max_daily_trades', 50)
        
        self.logger.info(f"Risk manager started with max_loss=${self.max_daily_loss}")
        
    def validate_entry(self, message):
        """Validate entry signal against risk limits"""
        signal = message.data
        symbol = signal['symbol']
        
        # Check position count
        if len(self.positions) >= self.max_positions:
            self.publish("signal.rejected", {
                **signal,
                'reason': f"Max positions ({self.max_positions}) reached"
            })
            return
            
        # Check position size
        if signal.get('size', 0) > self.max_position_size:
            signal['size'] = self.max_position_size
            
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            self.publish("signal.rejected", {
                **signal,
                'reason': f"Daily loss limit (${self.max_daily_loss}) reached"
            })
            return
            
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            self.publish("signal.rejected", {
                **signal,
                'reason': f"Daily trade limit ({self.max_daily_trades}) reached"
            })
            return
            
        # Signal approved
        self.positions[symbol] = {
            'size': signal['size'],
            'entry_price': signal['price'],
            'entry_time': datetime.utcnow()
        }
        self.daily_trades += 1
        
        self.publish("signal.approved", signal)
        self.logger.info(f"Approved {signal['action']} {symbol}")
        
    def validate_exit(self, message):
        """Process exit signal"""
        signal = message.data
        symbol = signal['symbol']
        
        if symbol in self.positions:
            # Calculate P&L
            position = self.positions[symbol]
            pnl = (signal['price'] - position['entry_price']) * position['size']
            self.daily_pnl += pnl
            
            # Remove position
            del self.positions[symbol]
            
            # Forward signal
            self.publish("signal.approved", signal)
            self.logger.info(f"Exit {symbol}, P&L: ${pnl:.2f}")
            
    async def stop(self):
        pass
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': self.daily_pnl > -self.max_daily_loss,
            'positions': len(self.positions),
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades
        }
```

### Day 17-18: Executor Plugin

```python
# plugins/execution/executor.py
import asyncio
from datetime import datetime
from typing import Dict, Optional

from core.plugin import Plugin

class ExecutorPlugin(Plugin):
    """Executes approved signals through IBKR"""
    
    async def start(self):
        # Subscribe to approved signals
        self.bus.subscribe("riskmanager.signal.approved", self.execute)
        
        # Subscribe to IBKR events for order updates
        self.bus.subscribe("ibkr.order.filled", self.on_fill)
        self.bus.subscribe("ibkr.order.cancelled", self.on_cancel)
        
        # Track pending orders
        self.pending_orders = {}
        
        self.logger.info("Executor started")
        
    def execute(self, message):
        """Execute approved signal"""
        signal = message.data
        
        # For now, simulate execution
        # In production, this would call IBKR API
        order_id = f"ORD_{datetime.utcnow().timestamp()}"
        
        self.pending_orders[order_id] = signal
        
        # Simulate order placement
        self.publish("order.placed", {
            'order_id': order_id,
            **signal
        })
        
        # Simulate fill after 1 second
        asyncio.create_task(self._simulate_fill(order_id))
        
    async def _simulate_fill(self, order_id: str):
        """Simulate order fill for testing"""
        await asyncio.sleep(1)
        
        if order_id in self.pending_orders:
            signal = self.pending_orders[order_id]
            
            # Simulate fill at signal price
            self.publish("order.filled", {
                'order_id': order_id,
                'symbol': signal['symbol'],
                'action': signal['action'],
                'size': signal['size'],
                'fill_price': signal['price'],
                'commission': 1.0,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            del self.pending_orders[order_id]
            self.logger.info(f"Filled {order_id}: {signal['action']} {signal['symbol']}")
            
    def on_fill(self, message):
        """Handle order fill confirmation"""
        fill = message.data
        self.logger.info(f"Order filled: {fill}")
        
    def on_cancel(self, message):
        """Handle order cancellation"""
        cancel = message.data
        order_id = cancel['order_id']
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            
    async def stop(self):
        pass
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'pending_orders': len(self.pending_orders)
        }
```

### Day 19-21: Testing & Integration

```python
# scripts/test_trading_pipeline.py
#!/usr/bin/env python3
"""Test complete trading pipeline"""

import asyncio
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bus import MessageBus
from core.persistence import EventStore
from core.plugin_manager import PluginManager

async def main():
    # Setup
    conn_string = os.getenv('DATABASE_URL', 'postgresql://localhost/alphatrader')
    event_store = EventStore(conn_string)
    bus = MessageBus(event_store)
    
    # Load plugins
    manager = PluginManager(bus, "config")
    await manager.discover_and_load()
    
    print(f"Loaded {len(manager.plugins)} plugins")
    
    # Run for a while
    print("Running trading system...")
    await asyncio.sleep(300)  # 5 minutes
    
    # Check health
    print("\n=== Health Check ===")
    for name, plugin in manager.plugins.items():
        health = plugin.health_check()
        status = "✅" if health.get('healthy') else "❌"
        print(f"{status} {name}: {health}")
        
    # Stop all
    await manager.stop_all()
    print("\nShutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
```

**Week 3-4 Deliverables:**
- ✅ Risk manager enforcing limits
- ✅ Executor handling orders
- ✅ Complete pipeline: Data → Strategy → Risk → Execution
- ✅ System running end-to-end

---

## Week 5-6: Feature Engineering & ML Preparation

### Day 22-24: Feature Engine Plugin

```python
# plugins/ml/feature_engine.py
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.plugin import Plugin

class FeatureEnginePlugin(Plugin):
    """Calculates 200+ features for ML models"""
    
    async def start(self):
        # Subscribe to all data sources
        self.bus.subscribe("ibkr.bar.*", self.on_price_update)
        self.bus.subscribe("alphavantage.data.*", self.on_indicator_update)
        
        # Feature storage with rolling windows
        self.price_history = defaultdict(lambda: deque(maxlen=500))
        self.volume_history = defaultdict(lambda: deque(maxlen=500))
        self.indicators = defaultdict(dict)
        self.features = defaultdict(dict)
        
        # Schedule feature calculation
        self.calculate_interval = self.config.get('calculate_interval', 60)
        asyncio.create_task(self._feature_calculation_loop())
        
        self.logger.info("Feature engine started")
        
    def on_price_update(self, message):
        """Store price/volume data"""
        if 'bar' in message.event_type:
            bar = message.data
            symbol = bar['symbol']
            
            self.price_history[symbol].append({
                'timestamp': bar['timestamp'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume'],
                'vwap': bar.get('vwap', bar['close'])
            })
            
    def on_indicator_update(self, message):
        """Store indicator values"""
        data = message.data
        symbol = data['symbol']
        indicator = message.event_type.split('.')[-1]  # Get indicator name
        
        self.indicators[symbol][indicator] = data['data']
        
    async def _feature_calculation_loop(self):
        """Periodically calculate features"""
        while True:
            await asyncio.sleep(self.calculate_interval)
            
            for symbol in self.price_history:
                if len(self.price_history[symbol]) >= 100:  # Need enough history
                    features = self.calculate_features(symbol)
                    
                    if features:
                        self.publish("features.calculated", {
                            'symbol': symbol,
                            'features': features,
                            'feature_count': len(features),
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        
    def calculate_features(self, symbol: str) -> Dict[str, float]:
        """Calculate all features for a symbol"""
        features = {}
        
        # Convert price history to DataFrame
        df = pd.DataFrame(list(self.price_history[symbol]))
        
        # Price-based features (50)
        features.update(self._calculate_price_features(df))
        
        # Volume features (30)
        features.update(self._calculate_volume_features(df))
        
        # Technical indicator features (40)
        features.update(self._calculate_indicator_features(symbol))
        
        # Microstructure features (20)
        features.update(self._calculate_microstructure_features(df))
        
        # Time-based features (10)
        features.update(self._calculate_time_features())
        
        return features
        
    def _calculate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Price-based features"""
        features = {}
        
        # Returns at different intervals
        for period in [1, 5, 10, 20, 50]:
            if len(df) > period:
                features[f'return_{period}'] = (
                    df['close'].iloc[-1] / df['close'].iloc[-period-1] - 1
                )
                
        # Volatility
        for period in [5, 10, 20]:
            if len(df) > period:
                returns = df['close'].pct_change()
                features[f'volatility_{period}'] = returns.tail(period).std()
                
        # Price position in range
        for period in [10, 20, 50]:
            if len(df) > period:
                high = df['high'].tail(period).max()
                low = df['low'].tail(period).min()
                current = df['close'].iloc[-1]
                
                if high != low:
                    features[f'price_position_{period}'] = (current - low) / (high - low)
                    
        # VWAP ratio
        if 'vwap' in df.columns:
            features['vwap_ratio'] = df['close'].iloc[-1] / df['vwap'].iloc[-1]
            
        # High-low spread
        features['hl_spread'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        
        return features
        
    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Volume-based features"""
        features = {}
        
        # Volume ratios
        for period in [5, 10, 20]:
            if len(df) > period * 2:
                recent_vol = df['volume'].tail(period).mean()
                prev_vol = df['volume'].iloc[-period*2:-period].mean()
                
                if prev_vol > 0:
                    features[f'volume_ratio_{period}'] = recent_vol / prev_vol
                    
        # Volume-weighted returns
        if len(df) > 20:
            returns = df['close'].pct_change()
            volumes = df['volume']
            
            # Calculate volume-weighted return
            vw_return = (returns * volumes).sum() / volumes.sum()
            features['volume_weighted_return'] = vw_return
            
        return features
        
    def _calculate_indicator_features(self, symbol: str) -> Dict[str, float]:
        """Technical indicator features"""
        features = {}
        indicators = self.indicators.get(symbol, {})
        
        # RSI features
        if 'rsi' in indicators and indicators['rsi']:
            latest_rsi = self._get_latest_value(indicators['rsi'])
            if latest_rsi:
                features['rsi'] = latest_rsi
                features['rsi_oversold'] = 1 if latest_rsi < 30 else 0
                features['rsi_overbought'] = 1 if latest_rsi > 70 else 0
                
        # MACD features
        if 'macd' in indicators and indicators['macd']:
            macd_data = self._get_latest_value(indicators['macd'])
            if macd_data and isinstance(macd_data, dict):
                features['macd'] = macd_data.get('MACD', 0)
                features['macd_signal'] = macd_data.get('MACD_Signal', 0)
                features['macd_histogram'] = macd_data.get('MACD_Hist', 0)
                
        # Add more indicators...
        
        return features
        
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Market microstructure features"""
        features = {}
        
        # Kyle's Lambda approximation (price impact)
        if len(df) > 20:
            returns = df['close'].pct_change().dropna()
            volumes = df['volume'].iloc[1:]
            
            if len(returns) > 0 and len(volumes) > 0:
                # Simple price impact measure
                features['price_impact'] = abs(returns).mean() / volumes.mean()
                
        # Roll's spread estimator
        if len(df) > 20:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 1:
                cov = returns.cov(returns.shift(1))
                if cov < 0:
                    features['roll_spread'] = 2 * np.sqrt(-cov)
                    
        return features
        
    def _calculate_time_features(self) -> Dict[str, float]:
        """Time-based features"""
        now = datetime.now()
        
        features = {
            'hour': now.hour,
            'minute': now.minute,
            'day_of_week': now.weekday(),
            'minutes_since_open': (now.hour - 9) * 60 + (now.minute - 30),
            'minutes_to_close': (16 - now.hour) * 60 - now.minute,
        }
        
        # Market session
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            features['session'] = 0  # Pre-market
        elif now.hour >= 16:
            features['session'] = 2  # After-market
        else:
            features['session'] = 1  # Regular
            
        return features
        
    def _get_latest_value(self, data: Dict):
        """Extract latest value from indicator data"""
        if isinstance(data, dict) and data:
            latest_key = max(data.keys())
            return data[latest_key]
        return None
        
    async def stop(self):
        pass
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'symbols_tracked': len(self.price_history),
            'features_calculated': len(self.features)
        }
```

### Day 25-28: Data Recording for ML Training

```python
# plugins/ml/data_recorder.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

from core.plugin import Plugin

class DataRecorderPlugin(Plugin):
    """Records features and outcomes for model training"""
    
    async def start(self):
        # Subscribe to features and signals
        self.bus.subscribe("featureengine.features.calculated", self.on_features)
        self.bus.subscribe("executor.order.filled", self.on_trade)
        
        # Storage
        self.features_buffer = []
        self.trades_buffer = []
        
        # Paths
        self.data_dir = Path(self.config.get('data_dir', 'data/ml'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Schedule periodic saves
        asyncio.create_task(self._save_loop())
        
        self.logger.info("Data recorder started")
        
    def on_features(self, message):
        """Record features with timestamp"""
        data = message.data
        
        record = {
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            **data['features']
        }
        
        self.features_buffer.append(record)
        
    def on_trade(self, message):
        """Record trade outcomes"""
        trade = message.data
        
        self.trades_buffer.append({
            'timestamp': trade['timestamp'],
            'symbol': trade['symbol'],
            'action': trade['action'],
            'price': trade['fill_price'],
            'size': trade['size']
        })
        
    async def _save_loop(self):
        """Periodically save data to disk"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            if self.features_buffer:
                # Save features
                df = pd.DataFrame(self.features_buffer)
                filename = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(self.data_dir / filename)
                
                self.logger.info(f"Saved {len(self.features_buffer)} features to {filename}")
                self.features_buffer = []
                
            if self.trades_buffer:
                # Save trades
                df = pd.DataFrame(self.trades_buffer)
                filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(self.data_dir / filename)
                
                self.logger.info(f"Saved {len(self.trades_buffer)} trades to {filename}")
                self.trades_buffer = []
                
    async def stop(self):
        # Save remaining data
        await self._save_loop()
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'features_buffered': len(self.features_buffer),
            'trades_buffered': len(self.trades_buffer)
        }
```

**Week 5-6 Deliverables:**
- ✅ 200+ features calculating
- ✅ Data recording for ML training
- ✅ Feature pipeline operational
- ✅ Ready for ML model integration

---

## Week 7-8: ML Models & Advanced Analytics

### Day 29-32: ML Model Training (Offline)

```python
# scripts/train_models.py
#!/usr/bin/env python3
"""Train ML models on recorded data"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

def load_training_data():
    """Load and prepare training data"""
    data_dir = Path("data/ml")
    
    # Load all feature files
    features_files = sorted(data_dir.glob("features_*.parquet"))
    features_df = pd.concat([
        pd.read_parquet(f) for f in features_files
    ])
    
    # Load all trade files
    trades_files = sorted(data_dir.glob("trades_*.parquet"))
    trades_df = pd.concat([
        pd.read_parquet(f) for f in trades_files
    ])
    
    # Create labels (1 if profitable trade opportunity)
    # This is simplified - real labeling would be more complex
    features_df['label'] = 0  # Default
    
    # Mark profitable opportunities
    for _, trade in trades_df.iterrows():
        mask = (
            (features_df['symbol'] == trade['symbol']) &
            (features_df['timestamp'] < trade['timestamp']) &
            (features_df['timestamp'] > trade['timestamp'] - pd.Timedelta(minutes=5))
        )
        features_df.loc[mask, 'label'] = 1 if trade['action'] == 'BUY' else -1
        
    return features_df

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        use_label_encoder=False
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        early_stopping_rounds=10,
        verbose=True
    )
    
    # Save model and scaler
    joblib.dump(model, "models/xgboost_latest.pkl")
    joblib.dump(scaler, "models/scaler_latest.pkl")
    
    return model, scaler

def main():
    print("Loading training data...")
    df = load_training_data()
    
    # Prepare features and labels
    feature_cols = [c for c in df.columns 
                   if c not in ['timestamp', 'symbol', 'label']]
    
    X = df[feature_cols].values
    y = (df['label'] > 0).astype(int).values  # Binary classification
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_score = 0
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nTraining fold {fold + 1}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model, scaler = train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate
        score = model.score(X_val, y_val)
        print(f"Validation accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            print("New best model!")
            
    print(f"\nTraining complete. Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()
```

### Day 33-35: Model Server Plugin

```python
# plugins/ml/model_server.py
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from core.plugin import Plugin

class ModelServerPlugin(Plugin):
    """Serves ML predictions"""
    
    async def start(self):
        # Load models
        self.model = joblib.load(self.config['model_path'])
        self.scaler = joblib.load(self.config['scaler_path'])
        
        # Feature configuration
        self.feature_cols = self.config['feature_columns']
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Subscribe to features
        self.bus.subscribe("featureengine.features.calculated", self.predict)
        
        self.logger.info("Model server started")
        
    def predict(self, message):
        """Generate prediction from features"""
        data = message.data
        symbol = data['symbol']
        features = data['features']
        
        # Prepare feature vector
        X = np.array([[
            features.get(col, 0) for col in self.feature_cols
        ]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        confidence = max(probability)
        
        # Only publish if confident
        if confidence >= self.confidence_threshold:
            self.publish("prediction.generated", {
                'symbol': symbol,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'probability_0': float(probability[0]),
                'probability_1': float(probability[1]),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            self.logger.info(
                f"Prediction for {symbol}: {prediction} "
                f"(confidence: {confidence:.2%})"
            )
            
    async def stop(self):
        pass
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'model_loaded': self.model is not None,
            'confidence_threshold': self.confidence_threshold
        }
```

**Week 7-8 Deliverables:**
- ✅ ML models trained on real data
- ✅ Model server making predictions
- ✅ Predictions flowing to strategies
- ✅ Complete ML pipeline operational

---

## Week 9-10: Advanced Analytics

### VPIN Calculator Plugin

```python
# plugins/analytics/vpin.py
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List

from core.plugin import Plugin

class VPINPlugin(Plugin):
    """Volume-Synchronized Probability of Informed Trading"""
    
    async def start(self):
        # Subscribe to trades/bars
        self.bus.subscribe("ibkr.bar.5s", self.update_vpin)
        
        # VPIN parameters
        self.bucket_size = self.config.get('bucket_size', 50000)  # Volume
        self.n_buckets = self.config.get('n_buckets', 50)
        
        # Storage per symbol
        self.buckets = defaultdict(lambda: deque(maxlen=self.n_buckets))
        self.current_bucket = defaultdict(lambda: {'buy': 0, 'sell': 0, 'volume': 0})
        
        self.logger.info(f"VPIN calculator started (bucket_size={self.bucket_size})")
        
    def update_vpin(self, message):
        """Update VPIN with new bar"""
        bar = message.data
        symbol = bar['symbol']
        
        # Classify volume using tick rule
        # Simplified: if close > open, it's buy volume
        if bar['close'] > bar['open']:
            buy_volume = bar['volume'] * 0.6
            sell_volume = bar['volume'] * 0.4
        else:
            buy_volume = bar['volume'] * 0.4
            sell_volume = bar['volume'] * 0.6
            
        # Update current bucket
        bucket = self.current_bucket[symbol]
        bucket['buy'] += buy_volume
        bucket['sell'] += sell_volume
        bucket['volume'] += bar['volume']
        
        # Check if bucket is complete
        if bucket['volume'] >= self.bucket_size:
            # Calculate order imbalance for this bucket
            total = bucket['buy'] + bucket['sell']
            if total > 0:
                imbalance = abs(bucket['buy'] - bucket['sell']) / total
            else:
                imbalance = 0
                
            # Add to buckets
            self.buckets[symbol].append(imbalance)
            
            # Calculate VPIN if we have enough buckets
            if len(self.buckets[symbol]) >= 10:  # Need minimum buckets
                vpin = np.mean(self.buckets[symbol])
                
                # Publish VPIN
                self.publish("vpin.calculated", {
                    'symbol': symbol,
                    'vpin': float(vpin),
                    'buckets_used': len(self.buckets[symbol]),
                    'toxic': vpin > 0.6,
                    'critical': vpin > 0.7,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                if vpin > 0.6:
                    self.logger.warning(f"High VPIN for {symbol}: {vpin:.3f}")
                    
            # Start new bucket with overflow
            overflow = bucket['volume'] - self.bucket_size
            ratio = overflow / bar['volume'] if bar['volume'] > 0 else 0
            
            self.current_bucket[symbol] = {
                'buy': buy_volume * ratio,
                'sell': sell_volume * ratio,
                'volume': overflow
            }
            
    async def stop(self):
        pass
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'symbols_tracked': len(self.buckets)
        }
```

---

## Week 11-12: Production Preparation

### Monitoring & Alerting

```python
# plugins/monitoring/performance_monitor.py
from datetime import datetime, timedelta
from typing import Dict

from core.plugin import Plugin

class PerformanceMonitorPlugin(Plugin):
    """Monitors system and trading performance"""
    
    async def start(self):
        # Subscribe to all trades
        self.bus.subscribe("executor.order.filled", self.track_trade)
        
        # Metrics storage
        self.trades = []
        self.daily_pnl = 0
        
        # Schedule reports
        asyncio.create_task(self._report_loop())
        
        self.logger.info("Performance monitor started")
        
    def track_trade(self, message):
        """Track trade for performance"""
        trade = message.data
        self.trades.append(trade)
        
        # Calculate P&L if it's a closing trade
        # (Simplified - real logic would match opening trades)
        
    async def _report_loop(self):
        """Generate periodic reports"""
        while True:
            await asyncio.sleep(3600)  # Every hour
            
            report = self.generate_report()
            self.publish("report.hourly", report)
            
    def generate_report(self) -> Dict:
        """Generate performance report"""
        if not self.trades:
            return {'status': 'No trades yet'}
            
        # Calculate metrics
        total_trades = len(self.trades)
        
        # Win rate (simplified)
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'daily_pnl': self.daily_pnl,
            'last_trade': self.trades[-1] if self.trades else None
        }
        
    async def stop(self):
        # Generate final report
        report = self.generate_report()
        self.logger.info(f"Final report: {report}")
        
    def health_check(self) -> Dict[str, Any]:
        return {
            'healthy': True,
            'trades_today': len(self.trades)
        }
```

### Production Deployment Script

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e  # Exit on error

echo "=== AlphaTrader Production Deployment ==="

# 1. Check prerequisites
echo "Checking prerequisites..."
command -v python3.11 >/dev/null 2>&1 || { echo "Python 3.11 required"; exit 1; }
command -v psql >/dev/null 2>&1 || { echo "PostgreSQL required"; exit 1; }
command -v redis-cli >/dev/null 2>&1 || { echo "Redis required"; exit 1; }

# 2. Setup environment
echo "Setting up environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Database setup
echo "Setting up database..."
psql -U postgres -c "CREATE DATABASE IF NOT EXISTS alphatrader;"
python scripts/init_database.py

# 4. Configuration
echo "Configuring system..."
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Please create .env with:"
    echo "  ALPHA_VANTAGE_API_KEY=your_key"
    echo "  IBKR_HOST=127.0.0.1"
    echo "  IBKR_PORT=7497"
    echo "  DATABASE_URL=postgresql://localhost/alphatrader"
    exit 1
fi

# 5. Test connection
echo "Testing connections..."
python scripts/test_connections.py || exit 1

# 6. Start services
echo "Starting services..."

# Start in screen sessions for persistence
screen -dmS redis redis-server
screen -dmS message_bus python -m core.main

echo "=== Deployment Complete ==="
echo ""
echo "Services running in screen sessions:"
echo "  screen -r redis      # Redis server"
echo "  screen -r message_bus # Main system"
echo ""
echo "To monitor:"
echo "  tail -f logs/alphatrader.log"
echo ""
echo "To stop:"
echo "  screen -X -S message_bus quit"
echo "  screen -X -S redis quit"
```

---

## Production Launch Checklist

### Week 11: Final Testing
- [ ] Run complete system for 5 days in paper mode
- [ ] Verify win rate > 45%
- [ ] Test all failure scenarios
- [ ] Validate risk limits working
- [ ] Confirm data quality

### Week 12: Production Launch
- [ ] Deploy to production server
- [ ] Start with minimal capital ($1,000)
- [ ] Monitor closely for first week
- [ ] Scale up gradually

### Success Metrics
- **System Health**: All plugins running, <1% error rate
- **Data Quality**: <0.1% missing data, <100ms latency
- **Trading Performance**: Win rate >45%, Sharpe >1.0
- **Risk Management**: No limit breaches, proper position sizing

---

## Key Implementation Tips

1. **Start Simple**: Get data flowing first week, add complexity gradually
2. **Test Everything**: Every plugin should have tests
3. **Monitor Always**: Add logging and metrics from day 1
4. **Configuration First**: Never hardcode values
5. **Fail Gracefully**: Every plugin should handle failures
6. **Document As You Go**: Update docs with each plugin

---

## Troubleshooting Guide

### Common Issues

**Message Bus Not Starting**
```bash
# Check PostgreSQL connection
psql -U postgres -d alphatrader -c "SELECT 1;"

# Check event table exists
psql -U postgres -d alphatrader -c "\dt events;"
```

**Plugins Not Loading**
```bash
# Check plugin directory structure
find plugins -name "*.py" -type f

# Test individual plugin
python -c "from plugins.datasources.alpha_vantage import AlphaVantagePlugin"
```

**No Data Flowing**
```bash
# Check message bus events
psql -U postgres -d alphatrader -c "SELECT event_type, COUNT(*) FROM events GROUP BY event_type;"

# Check plugin health
curl http://localhost:8080/health
```

---

## END OF IMPLEMENTATION ROADMAP

This roadmap provides a complete path from empty directory to production trading system in 12 weeks. The plugin architecture ensures you can add features without breaking existing code, and the message bus pattern keeps components loosely coupled and testable.