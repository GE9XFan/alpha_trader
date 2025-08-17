# AlphaTrader - Automated Options Trading System

**Version:** 0.5.1 (Phase 5.1 Complete - RSI Operational)  
**Status:** Development - First Technical Indicator Integrated  
**Last Updated:** August 17, 2025 (12:30 PM ET)  
**Development Day:** 18 of 106 (17.0% Complete)

## 🎯 Project Overview

AlphaTrader is a production-grade automated options trading system that combines real-time market data from IBKR, options analytics from Alpha Vantage, and machine learning to execute systematic trading strategies. Built with a focus on reliability, scalability, educational content generation, and zero hardcoded values.

### Vision
Build a fully automated trading system that not only trades profitably but also generates comprehensive educational content, market analysis, and community insights.

### Current Capabilities (Phases 0-5.1 Complete)
- ✅ **Configuration Management** - YAML-based with environment variables, zero hardcoding
- ✅ **Database Infrastructure** - PostgreSQL with 9 optimized tables (added av_rsi)
- ✅ **Alpha Vantage Integration** - Options + RSI operational, 40 more APIs ready
- ✅ **IBKR Real-Time Data** - Live bars and quotes from Interactive Brokers
- ✅ **Rate Limiting** - Token bucket (600/min), currently using 9.2% capacity
- ✅ **Data Ingestion Pipeline** - Unified ingestion for all data sources
- ✅ **Redis Cache Layer** - 109.4x performance on RSI, 30x on options
- ✅ **Automated Scheduler** - 69 jobs (46 options + 23 RSI) across 23 symbols
- ✅ **Market Hours Awareness** - Smart scheduling with test mode override
- ✅ **Options Data** - 49,854+ contracts with complete Greeks
- ✅ **Technical Indicators** - RSI operational with 83,239 data points

## 📊 System Architecture

### Current Implementation (Phase 5.1)
```
┌─────────────────────────────────────────────────────┐
│                  DATA SCHEDULER                      │
│         69 Jobs | 23 Symbols | 4 Data Types         │
│    Options: 30-180s | RSI: 60-600s | Daily: 6AM     │
└────────────┬──────────────────────┬─────────────────┘
             │                      │
             ▼                      ▼
┌──────────────────────────┬──────────────────────────┐
│    Alpha Vantage API     │      IBKR TWS API        │
│  • REALTIME_OPTIONS      │  • Real-time Bars (5s)   │
│  • HISTORICAL_OPTIONS    │  • Real-time Quotes      │
│  • RSI (NEW)            │  • MOC Imbalance         │
│  • Greeks & Analytics    │  • All Pricing Data      │
└────────────┬─────────────┴────────────┬─────────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────────────────────────────────┐
│          Rate Limiter          │   Direct Stream     │
│   Token Bucket (9.2% used)    │   (No limiting)     │
└────────────┬──────────────────┴────────┬────────────┘
             │                            │
             ▼                            ▼
      ┌──────────────┐              ┌──────────────┐
      │ Redis Cache  │◄─────────────│   Ingestion  │
      │  60s RSI     │              │    Engine    │
      │  30s Options │              │              │
      └──────────────┘              └──────┬───────┘
             │                              │
             └──────────┬───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│         PostgreSQL Database (9 Tables)              │
│  • av_realtime_options   • ibkr_bars_5sec          │
│  • av_historical_options • ibkr_bars_1min          │
│  • av_rsi (NEW)         • ibkr_bars_5min          │
│  • system_config         • ibkr_quotes             │
│  • api_response_log                                │
└─────────────────────────────────────────────────────┘
```

### Automated Scheduling Architecture (Updated)
| Data Type | Tier | Symbols | Update Frequency | API Calls/Hour | Jobs |
|-----------|------|---------|------------------|----------------|------|
| **Options** | A | SPY, QQQ, IWM, IBIT | 30 seconds | 480 | 4 |
| **Options** | B | AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA | 60 seconds | 420 | 7 |
| **Options** | C | DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW | 180 seconds | 240 | 12 |
| **RSI** | A | SPY, QQQ, IWM, IBIT | 60 seconds | 240 | 4 |
| **RSI** | B | MAG7 stocks | 300 seconds | 84 | 7 |
| **RSI** | C | 12 other stocks | 600 seconds | 72 | 12 |
| **Historical** | All | All 23 symbols | Daily 6:00 AM | 23 | 23 |

**Total API Usage:** ~46 calls/minute (9.2% of 500/min budget)

### Cache Performance Metrics (Updated)
| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| Options Data Fetch | 1.01s | 0.03s | **30.6x faster** |
| RSI Data Fetch | 0.58s | 0.01s | **109.4x faster** |
| API Calls Saved | N/A | N/A | **~60%** |
| Memory Usage | N/A | 35MB | Efficient |
| Cache Hit Rate | 0% | 66.7%+ | Excellent |

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.11+** (tested on 3.11.11)
- **PostgreSQL 14+** (for production data storage)
- **Redis 8.0+** (for caching layer)
- **Interactive Brokers TWS** (paper or live account)
- **Alpha Vantage API Key** (Premium recommended for 600 calls/min)
- **macOS/Linux** (primary development on macOS)
- **4GB+ RAM** for data processing
- **50GB+ disk space** for growing indicator data
- **APScheduler 3.10.4** for automated scheduling

### Step 1: Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/AlphaTrader.git
cd AlphaTrader

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install ibapi  # IBKR API
```

### Step 2: Database Setup
```bash
# Create the database
psql -U your_username -d postgres
CREATE DATABASE trading_system_db;
\q

# Initialize all tables
psql -U your_username -d trading_system_db -f scripts/init_db.sql
psql -U your_username -d trading_system_db -f scripts/create_options_table.sql
psql -U your_username -d trading_system_db -f scripts/create_historical_options_table.sql
psql -U your_username -d trading_system_db -f scripts/create_ibkr_tables.sql
psql -U your_username -d trading_system_db -f scripts/create_rsi_table.sql  # NEW
```

### Step 3: Redis Setup
```bash
# Install Redis
brew install redis

# Start Redis service
brew services start redis

# Verify Redis is running
redis-cli ping  # Should return PONG
```

### Step 4: Configuration
```bash
# Create your environment file
cp config/.env.example config/.env

# Edit config/.env with your credentials:
DATABASE_URL=postgresql://username:password@localhost:5432/trading_system_db
AV_API_KEY=your_alpha_vantage_api_key_here
REDIS_URL=redis://localhost:6379/0

# IBKR Settings
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497 for paper, 7496 for live
IBKR_CLIENT_ID=1
```

### Step 5: TWS Configuration
1. Open TWS (or IB Gateway)
2. File → Global Configuration → API → Settings
3. Enable "Enable ActiveX and Socket Clients"
4. Enable "Allow connections from localhost only"
5. Socket port: 7497 (paper) or 7496 (live)
6. Click OK and restart TWS

### Step 6: Verify Installation
```bash
# Test Phase 0-2: Foundation & Alpha Vantage
python scripts/test_phase0.py
python scripts/test_phase2_complete.py

# Test Phase 3: IBKR Connection
python scripts/test_ibkr_connection.py
python scripts/test_ibkr_market_data.py

# Test Phase 4: Cache & Scheduler
python scripts/test_cache_manager.py
python scripts/test_cached_av_client.py
python scripts/test_cache_integration.py
python scripts/test_scheduler.py

# Test Phase 5.1: RSI (NEW)
python scripts/test_rsi_api.py
python scripts/test_rsi_client.py
python scripts/test_rsi_pipeline.py
python scripts/test_rsi_scheduler.py
python scripts/test_rsi_complete.py

# Test Live Data (run during market hours)
python scripts/test_ibkr_live_data.py
```

### Step 7: Start the Automated Scheduler
```bash
# Production mode (respects market hours)
python scripts/run_scheduler.py

# Test mode (for weekends/development)
python scripts/run_scheduler.py --test
```

## 📁 Project Structure

```
AlphaTrader/
├── src/                           # Source code
│   ├── foundation/
│   │   └── config_manager.py     # Zero hardcoded values
│   ├── connections/
│   │   ├── av_client.py          # AV client with RSI support (fixed hardcoding)
│   │   └── ibkr_connection.py    # IBKR TWS connection
│   ├── data/
│   │   ├── ingestion.py          # RSI + options ingestion
│   │   ├── rate_limiter.py       # Token bucket (9.2% usage)
│   │   ├── cache_manager.py      # Redis cache manager
│   │   └── scheduler.py          # 69 automated jobs
│   └── __init__.py
│
├── config/                        # Configuration files
│   ├── .env                       # Environment variables
│   ├── .env.example               # Template for environment
│   ├── apis/
│   │   └── alpha_vantage.yaml    # RSI + options endpoints
│   ├── data/
│   │   └── schedules.yaml        # RSI in indicators_fast
│   └── system/
│       └── redis.yaml             # Cache configuration
│
├── scripts/                       # Utility & test scripts
│   ├── init_db.sql               # System tables
│   ├── create_options_table.sql  # AV options schemas
│   ├── create_historical_options_table.sql
│   ├── create_ibkr_tables.sql    # IBKR data schemas
│   ├── create_rsi_table.sql      # RSI schema (NEW)
│   ├── test_phase0.py            # Foundation tests
│   ├── test_phase2_complete.py   # AV integration tests
│   ├── test_ibkr_connection.py   # IBKR connection test
│   ├── test_ibkr_bars.py         # Bar data test
│   ├── test_ibkr_market_data.py  # Quotes test
│   ├── test_ibkr_live_data.py    # Live ingestion test
│   ├── test_cache_manager.py     # Cache operations
│   ├── test_cached_av_client.py  # Cache integration
│   ├── test_cache_integration.py # Full pipeline test
│   ├── test_scheduler.py         # Scheduler tests
│   ├── test_rsi_api.py          # RSI API discovery (NEW)
│   ├── test_rsi_client.py       # RSI client test (NEW)
│   ├── test_rsi_pipeline.py     # RSI pipeline test (NEW)
│   ├── test_rsi_scheduler.py    # RSI scheduler test (NEW)
│   ├── test_rsi_complete.py     # RSI complete test (NEW)
│   ├── run_scheduler.py          # Production scheduler
│   └── query_options_data.py     # Data analysis queries
│
├── data/                          # Data storage
│   └── api_responses/            # Saved API responses
│       ├── realtime_options_*.json
│       ├── historical_options_*.json
│       └── rsi_*.json            # RSI response samples (NEW)
│
├── docs/                          # Documentation
│   ├── SSOT-Ops.md              # Operational specification
│   ├── SSOT-Tech.md             # Technical specification
│   ├── granular-phased-plan.md  # 19-phase roadmap
│   ├── educational-timeline.md  # Educational development
│   ├── educational-content-plan.md # Content strategy
│   └── phase5_rsi_summary.md    # RSI implementation (NEW)
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 💾 Database Schema

### Alpha Vantage Tables
| Table | Records | Description | Update Frequency |
|-------|---------|-------------|------------------|
| `av_realtime_options` | 49,854+ | Live options chains with Greeks | 30-180s |
| `av_historical_options` | 49,854+ | Historical snapshots | Daily 6 AM |
| **`av_rsi`** | **83,239** | **RSI indicator values** | **60-600s** |

### RSI Table Schema (NEW)
```sql
CREATE TABLE av_rsi (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    rsi DECIMAL(10, 4),
    interval VARCHAR(10) NOT NULL,
    time_period INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period)
);
```

### IBKR Tables (Ready for Monday)
| Table | Purpose | Update Frequency | Status |
|-------|---------|------------------|--------|
| `ibkr_bars_5sec` | Granular price action | Every 5 seconds | Schema ready |
| `ibkr_bars_1min` | Short-term analysis | Every minute | Schema ready |
| `ibkr_bars_5min` | Trading signals | Every 5 minutes | Schema ready |
| `ibkr_quotes` | Bid/ask spreads | Tick-by-tick | Schema ready |

### Redis Cache Keys (Updated)
| Key Pattern | TTL | Purpose | Hit Rate |
|-------------|-----|---------|----------|
| `av:realtime_options:{symbol}` | 30s | Options chains | 66.7% |
| `av:historical_options:{symbol}:{date}` | 24h | Historical options | N/A |
| **`av:rsi:{symbol}:{interval}_{period}`** | **60s** | **RSI values** | **95%+** |
| `av:indicators:{type}:{symbol}` | 60-300s | Other indicators | Planned |

## 📈 API Usage & Examples

### Automated Scheduler Operations (Updated)
```python
from src.data.scheduler import DataScheduler

# Initialize scheduler (use test_mode=True on weekends)
scheduler = DataScheduler(test_mode=False)

# Start automated data collection
scheduler.start()

# Check status
status = scheduler.get_status()
print(f"Running: {status['running']}")
print(f"Total jobs: {status['total_jobs']}")  # Now 69 jobs
print(f"Market hours: {status['is_market_hours']}")

# View scheduled jobs
for job in status['jobs'][:10]:
    print(f"{job['name']}: Next run at {job['next_run']}")

# Stop scheduler
scheduler.stop()
```

### Fetch RSI with Caching (NEW)
```python
from src.connections.av_client import AlphaVantageClient

# Initialize client (no hardcoded defaults!)
client = AlphaVantageClient()

# Fetch RSI - required symbol, optional params from config
rsi_data = client.get_rsi('SPY')  # Uses config defaults
# Or specify parameters
rsi_data = client.get_rsi('QQQ', interval='5min', time_period=21)

# Check cache performance
# First call: 0.58s
# Second call: 0.01s (109.4x faster!)
```

### RSI Data Ingestion (NEW)
```python
from src.data.ingestion import DataIngestion

ingestion = DataIngestion()

# Ingest RSI data
records = ingestion.ingest_rsi_data(rsi_data, 'SPY', interval='1min', time_period=14)
print(f"Ingested {records} RSI data points")
```

### Cache Management
```python
from src.data.cache_manager import get_cache

cache = get_cache()

# Store with TTL
cache.set("my_key", {"data": "value"}, ttl=60)

# Retrieve
data = cache.get("my_key")

# Check RSI cache keys
rsi_keys = len(cache.redis_client.keys("av:rsi:*"))
print(f"RSI keys cached: {rsi_keys}")

# Check stats
stats = cache.get_stats()
print(f"Memory used: {stats['used_memory']}")
```

### Query RSI Data (NEW)
```sql
-- Check RSI data freshness
SELECT 
    symbol,
    COUNT(*) as data_points,
    MAX(timestamp) as latest_data,
    MIN(rsi) as min_rsi,
    MAX(rsi) as max_rsi,
    AVG(rsi) as avg_rsi
FROM av_rsi
GROUP BY symbol
ORDER BY symbol;

-- Find oversold/overbought conditions
SELECT 
    symbol,
    timestamp,
    rsi,
    CASE 
        WHEN rsi < 30 THEN 'OVERSOLD'
        WHEN rsi > 70 THEN 'OVERBOUGHT'
        ELSE 'NEUTRAL'
    END as condition
FROM av_rsi
WHERE timestamp > NOW() - INTERVAL '1 hour'
    AND (rsi < 30 OR rsi > 70)
ORDER BY timestamp DESC;
```

## 🧪 Testing

### Test Suite by Phase
```bash
# Phase 0-2: Foundation & Alpha Vantage
python scripts/test_phase0.py          # Config & database
python scripts/test_rate_limiter.py    # Rate limiting
python scripts/test_phase2_complete.py # Both AV APIs

# Phase 3: IBKR Integration
python scripts/test_ibkr_connection.py    # TWS connection
python scripts/test_ibkr_bars.py          # Real-time bars
python scripts/test_ibkr_market_data.py   # Quotes & bars
python scripts/test_ibkr_live_data.py     # Full ingestion

# Phase 4: Cache & Scheduler
python scripts/test_cache_manager.py      # Basic cache ops
python scripts/test_cached_av_client.py   # Cached API calls
python scripts/test_cache_integration.py  # Full pipeline with cache
python scripts/test_scheduler.py          # Scheduler with all symbols

# Phase 5.1: RSI (NEW)
python scripts/test_rsi_api.py           # API discovery
python scripts/test_rsi_client.py        # Client method
python scripts/test_rsi_pipeline.py      # Full pipeline
python scripts/test_rsi_scheduler.py     # Scheduler integration
python scripts/test_rsi_complete.py      # Comprehensive test

# Data Analysis
python scripts/query_options_data.py      # Analyze stored data
```

### Performance Benchmarks (Updated)
| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| AV API Call (no cache) | < 2s | ~1.01s | ✅ |
| AV API Call (cached) | < 100ms | ~30ms | ✅ |
| RSI Fetch (no cache) | < 2s | 0.58s | ✅ |
| RSI Fetch (cached) | < 100ms | 0.01s | ✅ |
| Cache Hit Rate | > 50% | 80%+ avg | ✅ |
| Scheduler Jobs | N/A | 69 | ✅ |
| Job Execution | < 2s | ~1s | ✅ |
| IBKR Connection | < 5s | ~2s | ✅ |
| Database Insert (21K records) | < 30s | ~8s | ✅ |
| Query Options Chain | < 100ms | ~45ms | ✅ |
| Query RSI Data | < 100ms | ~42ms | ✅ |
| Real-time Bar Latency | < 500ms | ~100ms | ✅ |
| Rate Limiter Check | < 10ms | ~1ms | ✅ |
| Cache Get/Set | < 10ms | ~3ms | ✅ |

## 📊 Current Data Holdings

| Data Type | Source | Count | Update Frequency | Cache TTL |
|-----------|--------|-------|------------------|-----------|
| **Options Contracts** | Alpha Vantage | 49,854+ | Every 30-180s | 30 seconds |
| **Historical Options** | Alpha Vantage | 49,854+ | Daily at 6 AM | 24 hours |
| **RSI Indicator** | Alpha Vantage | 83,239 | Every 60-600s | 60 seconds |
| **Real-time Bars** | IBKR | 0 (Monday) | 5 seconds | N/A |
| **Quotes** | IBKR | 0 (Monday) | Tick-by-tick | N/A |
| **Symbols Tracked** | Both | 23 | Continuous | - |
| **Scheduled Jobs** | Scheduler | 69 | Various | - |

### Live Market Data Insights (August 17, 2025)
- **Symbols Tracked:** 23 across 3 tiers
- **Total Data Points:** 133,093 (options + RSI)
- **Cache Hit Rate:** 80%+ average
- **API Usage:** Only 9.2% of capacity
- **Automation Level:** 100% hands-free
- **RSI Coverage:** 22 days of 1-minute data

### Symbols by Tier
**Tier A (High Priority - Fastest updates):**
- SPY, QQQ, IWM, IBIT
- Options: 30s, RSI: 60s

**Tier B (Medium Priority):**
- AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA
- Options: 60s, RSI: 300s

**Tier C (Standard Priority):**
- DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW
- Options: 180s, RSI: 600s

## 🛠️ Configuration

### Scheduler Configuration (`config/data/schedules.yaml`)
```yaml
# Market hours configuration
market_hours:
  pre_market_start: "04:00"
  market_open: "09:30"
  market_close: "16:00"
  after_hours_end: "20:00"
  timezone: "America/New_York"

# Symbol tiers with priority
symbol_tiers:
  tier_a:
    symbols: ["SPY", "QQQ", "IWM", "IBIT"]
    priority: 1
    
  tier_b:
    symbols: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    priority: 2
    
  tier_c:
    symbols: ["DIS", "NFLX", "COST", "WMA", "HOOD", "MSTR", "PLTR", 
              "SMCI", "AMD", "INTC", "ORCL", "SNOW"]
    priority: 3

# API scheduling groups
api_groups:
  critical:
    tier_a_interval: 30  # seconds
    tier_b_interval: 60
    tier_c_interval: 180
    
  indicators_fast:  # NEW - RSI added
    apis: ["RSI"]
    tier_a_interval: 60
    tier_b_interval: 300
    tier_c_interval: 600
    
  daily:
    schedule_time: "06:00"
```

### Alpha Vantage Configuration (`config/apis/alpha_vantage.yaml`)
```yaml
rate_limit:
  max_per_minute: 600
  target_per_minute: 500
  refill_rate: 10
  burst_capacity: 20

endpoints:
  realtime_options:
    function: "REALTIME_OPTIONS"
    require_greeks: "true"
    
  historical_options:
    function: "HISTORICAL_OPTIONS"
    
  rsi:  # NEW
    function: "RSI"
    datatype: "json"
    cache_ttl: 60
    default_params:
      interval: "1min"
      time_period: 14
      series_type: "close"
```

### Redis Cache (`config/system/redis.yaml`)
```yaml
cache_ttl:
  realtime_options: 30      # 30 seconds
  historical_options: 86400  # 24 hours
  rsi: 60                   # 60 seconds (NEW)
  api_responses: 300        # 5 minutes
  
pool:
  max_connections: 10
```

### Environment Variables (`.env`)
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_system_db

# APIs
AV_API_KEY=your_alpha_vantage_key
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper: 7497, Live: 7496
IBKR_CLIENT_ID=1

# Cache
REDIS_URL=redis://localhost:6379/0
```

## 🚦 Development Status

### ✅ Completed Phases (Days 1-18)
| Phase | Days | Description | Status | Key Achievement |
|-------|------|-------------|--------|-----------------|
| **0** | 1-3 | Minimal Foundation | ✅ Complete | Zero hardcoding |
| **1** | 4-7 | First API Pipeline | ✅ Complete | 9,294 contracts |
| **2** | 8-10 | Rate Limiting & 2nd API | ✅ Complete | 600/min protection |
| **3** | 11-14 | IBKR Integration | ✅ Complete | Real-time ready |
| **4** | 15-17 | Scheduler & Cache | ✅ Complete | 46 jobs automated |
| **5.1** | 18 | RSI Indicator | ✅ Complete | 83,239 data points |

### 🚧 In Progress - Phase 5 Continuation
| Indicator | Day | Status | Implementation | Notes |
|-----------|-----|--------|---------------|-------|
| RSI | 18 | ✅ Complete | 83,239 records | 109.4x cache |
| MACD | 19 | 📋 Next | Ready to start | 3 values |
| BBANDS | 20 | 📋 Planned | - | 3 bands |
| VWAP | 21 | 📋 Planned | - | 1 value |
| ATR | 22 | 📋 Planned | - | 1 value |
| ADX | 23 | 📋 Planned | - | 3 values |
| Integration | 24 | 📋 Planned | - | All testing |

### 📅 Upcoming Phases (Days 25-106)
| Phase | Days | Description | Key Deliverable |
|-------|------|-------------|-----------------|
| **6** | 25-28 | Analytics & Validation | Greeks validator |
| **7** | 29-35 | First Strategy (0DTE) | **Trading logic** |
| **8** | 36-39 | Risk Management | Position limits |
| **9** | 40-43 | Paper Trading | **First trades!** |
| **10** | 44-50 | All Indicators | 16 total indicators |
| **11** | 51-57 | Additional Strategies | 1DTE, 14DTE, MOC |
| **12** | 58-63 | ML Integration | Model predictions |
| **13** | 64-67 | Sentiment & News | News integration |
| **14** | 68-74 | Fundamentals | Company data |
| **15** | 75-82 | Output & Monitoring | Discord, Dashboard |
| **16** | 83-89 | Market Analysis | Educational content |
| **17** | 90-99 | Integration Testing | Full system test |
| **18** | 89-95 | Production Prep | Documentation |
| **19** | 96+ | Production | **Live trading** |

## 🐛 Troubleshooting

### Scheduler Issues
```bash
# Check if scheduler is running
ps aux | grep python | grep scheduler

# Test scheduler manually
python scripts/test_scheduler.py

# Use test mode on weekends
python scripts/run_scheduler.py --test
```

### Redis Issues
```bash
# Check Redis status
redis-cli ping

# Monitor Redis
redis-cli MONITOR

# Check memory usage
redis-cli INFO memory

# Clear all cache
redis-cli FLUSHDB

# Check RSI cache keys
redis-cli --scan --pattern "av:rsi:*"
```

### Cache Miss Issues
```bash
# Check cache contents
redis-cli KEYS "av:*"

# Get TTL for a key
redis-cli TTL "av:rsi:SPY:1min_14"

# Manual cache check
python -c "from src.data.cache_manager import get_cache; c = get_cache(); print(c.get_stats())"
```

### IBKR Connection Issues
```bash
# Check TWS is running
ps aux | grep tws

# Verify API settings in TWS
# File → Global Configuration → API → Settings

# Test connection
python scripts/test_ibkr_connection.py
```

### API Rate Limit Issues
```python
# Check current usage
from src.connections.av_client import AlphaVantageClient

client = AlphaVantageClient()
stats = client.get_rate_limit_status()
print(f"Calls this minute: {stats['minute_window_calls']}/600")
print(f"Tokens available: {stats['tokens_available']}/20")
```

### Hardcoded Values Check (Critical!)
```bash
# Search for hardcoded defaults
grep -r "='SPY'" src/
grep -r "interval='1min'" src/
grep -r "time_period=14" src/

# Should return NOTHING - all defaults from config
```

## 📚 Documentation

- **[SSOT-Ops.md](docs/SSOT-Ops.md)** - Operational specifications
- **[SSOT-Tech.md](docs/SSOT-Tech.md)** - Technical implementation
- **[Phased Plan](docs/granular-phased-plan.md)** - Complete 19-phase roadmap
- **[RSI Implementation](docs/phase5_rsi_summary.md)** - RSI details (NEW)
- **[Quick Reference](quick_reference_guide.md)** - Common commands
- **[Status Report](project_status_report.md)** - Current progress
- **[Educational Plan](docs/educational-content-plan.md)** - Content strategy

## 🎯 Next Steps

### Immediate (Day 19 - Sunday/Monday)
1. **MACD Implementation**
   - Follow 8-step process from RSI
   - Expect 3 values per timestamp (MACD, Signal, Histogram)
   - Consider separate columns or JSONB storage
   - Test with SPY first

### This Week (Phase 5 Completion)
- [ ] Day 19: MACD implementation
- [ ] Day 20: BBANDS implementation
- [ ] Day 21: VWAP implementation
- [ ] Day 22: ATR implementation
- [ ] Day 23: ADX implementation
- [ ] Day 24: Integration testing all indicators

### This Month (Through Phase 9)
- [ ] Complete all 16 technical indicators
- [ ] Implement Greeks validation
- [ ] Build first trading strategy (0DTE)
- [ ] Add risk management
- [ ] Begin paper trading (Day 40)

## 📊 Key Metrics

| Metric | Value | Change |
|--------|-------|--------|
| **Development Day** | 18 of 106 | +2 |
| **Progress** | 17.0% complete | +1.9% |
| **Phase Status** | 5.1 of 19 complete | +0.1 |
| **Lines of Code** | ~2,500 | +500 |
| **Database Tables** | 9 | +1 |
| **Database Size** | 47MB | +10MB |
| **Scheduled Jobs** | 69 | +23 |
| **API Usage** | 46/min (9.2%) | +27/min |
| **Cache Performance** | 109.4x on RSI | NEW |
| **Total Data Points** | 133,093 | +83,239 |
| **Test Scripts** | 23 | +5 |
| **Status** | ✅ On Schedule | - |

## 🏆 Achievements

### Phase 5.1 Specific
- ✅ RSI implementation in ~2 hours
- ✅ 83,239 RSI data points ingested
- ✅ Fixed ALL hardcoded values in av_client.py
- ✅ 109.4x cache performance on RSI
- ✅ Configuration-driven defaults
- ✅ 23 new scheduled jobs
- ✅ Clean 8-step implementation process

### Overall Project
- ✅ Clean architecture maintained throughout
- ✅ Zero hardcoded values achieved
- ✅ Rate limiting never exceeded
- ✅ IBKR integration successful
- ✅ Cache layer performing exceptionally
- ✅ Production-ready automation
- ✅ Scalable to 100+ symbols
- ✅ Test coverage comprehensive
- ✅ Documentation detailed and current

## 🔧 Production Operations

### Starting the System
```bash
# Full production start
python scripts/run_scheduler.py

# Weekend/test mode
python scripts/run_scheduler.py --test

# Monitor system
python scripts/run_scheduler.py --interval 30
```

### System Monitoring
```python
# Check system health
from src.data.scheduler import DataScheduler
from src.connections.av_client import AlphaVantageClient

scheduler = DataScheduler()
client = AlphaVantageClient()

print(f"Scheduler jobs: {len(scheduler.get_status()['jobs'])}")
print(f"API usage: {client.get_rate_limit_status()['minute_window_calls']}/500")
print(f"Cache stats: {client.get_cache_status()}")
```

### Daily Operations Checklist
- [ ] Morning: Check scheduler status
- [ ] Verify cache hit rate > 60%
- [ ] Monitor API usage < 100/min
- [ ] Check database growth
- [ ] Review any error logs
- [ ] Evening: Daily performance summary

### RSI-Specific Monitoring
```sql
-- Check RSI data freshness
SELECT 
    symbol,
    COUNT(*) as data_points,
    MAX(timestamp) as latest_data,
    NOW() - MAX(updated_at) as age,
    AVG(rsi) as avg_rsi
FROM av_rsi
GROUP BY symbol
ORDER BY symbol;

-- Find trading signals
SELECT 
    symbol,
    timestamp,
    rsi,
    CASE 
        WHEN rsi < 30 THEN 'OVERSOLD - BUY SIGNAL'
        WHEN rsi > 70 THEN 'OVERBOUGHT - SELL SIGNAL'
        ELSE 'NEUTRAL'
    END as signal
FROM av_rsi
WHERE timestamp > NOW() - INTERVAL '1 hour'
    AND (rsi < 30 OR rsi > 70)
ORDER BY timestamp DESC;
```

## 📮 Contact

For questions or issues, create an issue in the repository.

---

**Current Phase:** 5.1 Complete - RSI Operational ✅  
**Next Phase:** 5.2 - MACD Implementation (Day 19)  
**First Paper Trade:** Day 40 (22 days away)  
**Production Launch:** Day 107 (89 days away)

*Last Updated: August 17, 2025, 12:30 PM ET - RSI with 83,239 data points across 23 symbols, 109.4x cache performance*