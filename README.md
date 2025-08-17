# AlphaTrader - Automated Options Trading System

**Version:** 0.5.3 (Phase 5.3 Complete - BBANDS Operational)  
**Status:** Development - Three Technical Indicators Integrated  
**Last Updated:** August 17, 2025 (2:00 PM ET)  
**Development Day:** 20 of 106 (18.9% Complete)

## 🎯 Project Overview

AlphaTrader is a production-grade automated options trading system that combines real-time market data from IBKR, options analytics from Alpha Vantage, and machine learning to execute systematic trading strategies. Built with a focus on reliability, scalability, educational content generation, and absolutely zero hardcoded values.

### Vision
Build a fully automated trading system that not only trades profitably but also generates comprehensive educational content, market analysis, and community insights.

### Current Capabilities (Phases 0-5.3 Complete)
- ✅ **Configuration Management** - YAML-based with environment variables, zero hardcoding enforced
- ✅ **Database Infrastructure** - PostgreSQL with 11 optimized tables (added av_macd, av_bbands)
- ✅ **Alpha Vantage Integration** - Options + 3 indicators operational, 38 more APIs ready
- ✅ **IBKR Real-Time Data** - Live bars and quotes from Interactive Brokers
- ✅ **Rate Limiting** - Token bucket (600/min), currently using 20% capacity
- ✅ **Data Ingestion Pipeline** - Unified ingestion for all data sources
- ✅ **Redis Cache Layer** - 127x performance on BBANDS, 110x on MACD, 109x on RSI
- ✅ **Automated Scheduler** - 115 jobs (46 options + 69 indicators) across 23 symbols
- ✅ **Market Hours Awareness** - Smart scheduling with test mode override
- ✅ **Options Data** - 49,854+ contracts with complete Greeks
- ✅ **Technical Indicators** - RSI (momentum), MACD (trend), BBANDS (volatility) operational

## 📊 System Architecture

### Current Implementation (Phase 5.3)
```
┌─────────────────────────────────────────────────────┐
│                  DATA SCHEDULER                      │
│        115 Jobs | 23 Symbols | 5 Data Types         │
│  Options: 30-180s | Indicators: 60-600s | Daily: 6AM │
└────────────┬──────────────────────┬─────────────────┘
             │                      │
             ▼                      ▼
┌──────────────────────────┬──────────────────────────┐
│    Alpha Vantage API     │      IBKR TWS API        │
│  • REALTIME_OPTIONS      │  • Real-time Bars (5s)   │
│  • HISTORICAL_OPTIONS    │  • Real-time Quotes      │
│  • RSI (OPERATIONAL)     │  • MOC Imbalance         │
│  • MACD (OPERATIONAL)    │  • All Pricing Data      │
│  • BBANDS (OPERATIONAL)  │                          │
│  • Greeks & Analytics    │                          │
└────────────┬─────────────┴────────────┬─────────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────────────────────────────────┐
│          Rate Limiter          │   Direct Stream     │
│   Token Bucket (20% used)     │   (No limiting)     │
└────────────┬──────────────────┴────────┬────────────┘
             │                            │
             ▼                            ▼
      ┌──────────────┐              ┌──────────────┐
      │ Redis Cache  │◄─────────────│   Ingestion  │
      │  60s TTL     │              │    Engine    │
      │  109-127x    │              │              │
      └──────────────┘              └──────┬───────┘
             │                              │
             └──────────┬───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│         PostgreSQL Database (11 Tables)             │
│  • av_realtime_options   • av_rsi                  │
│  • av_historical_options • av_macd                 │
│  • av_bbands            • ibkr_bars_5sec           │
│  • ibkr_bars_1min       • ibkr_bars_5min           │
│  • ibkr_quotes          • system_config            │
│  • api_response_log                                │
└─────────────────────────────────────────────────────┘
```

### Automated Scheduling Architecture (Phase 5.3 Complete)
| Data Type | Tier | Symbols | Update Frequency | API Calls/Hour | Jobs |
|-----------|------|---------|------------------|----------------|------|
| **Options** | A | SPY, QQQ, IWM, IBIT | 30 seconds | 480 | 4 |
| **Options** | B | AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA | 60 seconds | 420 | 7 |
| **Options** | C | DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW | 180 seconds | 240 | 12 |
| **RSI** | A | SPY, QQQ, IWM, IBIT | 60 seconds | 240 | 4 |
| **RSI** | B | MAG7 stocks | 300 seconds | 84 | 7 |
| **RSI** | C | 12 other stocks | 600 seconds | 72 | 12 |
| **MACD** | A | SPY, QQQ, IWM, IBIT | 60 seconds | 240 | 4 |
| **MACD** | B | MAG7 stocks | 300 seconds | 84 | 7 |
| **MACD** | C | 12 other stocks | 600 seconds | 72 | 12 |
| **BBANDS** | A | SPY, QQQ, IWM, IBIT | 60 seconds | 240 | 4 |
| **BBANDS** | B | MAG7 stocks | 300 seconds | 84 | 7 |
| **BBANDS** | C | 12 other stocks | 600 seconds | 72 | 12 |
| **Historical** | All | All 23 symbols | Daily 6:00 AM | 23 | 23 |

**Total API Usage:** ~100 calls/minute (20% of 500/min budget)

### Cache Performance Metrics (Phase 5.3)
| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| Options Data Fetch | 1.01s | 0.03s | **30.6x faster** |
| RSI Data Fetch | 0.58s | 0.01s | **109.4x faster** |
| MACD Data Fetch | 0.95s | 0.01s | **110.2x faster** |
| BBANDS Data Fetch | 0.44s | 0.003s | **127.4x faster** |
| API Calls Saved | N/A | N/A | **~75%** |
| Memory Usage | N/A | 45MB | Efficient |
| Cache Hit Rate | 0% | 80%+ | Excellent |

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.11+** (tested on 3.11.11)
- **PostgreSQL 14+** (for production data storage)
- **Redis 8.0+** (for caching layer)
- **Interactive Brokers TWS** (paper or live account)
- **Alpha Vantage API Key** (Premium recommended for 600 calls/min)
- **macOS/Linux** (primary development on macOS)
- **4GB+ RAM** for data processing
- **100GB+ disk space** for growing indicator data
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
psql -U your_username -d trading_system_db -f scripts/create_rsi_table.sql
psql -U your_username -d trading_system_db -f scripts/create_macd_table.sql
psql -U your_username -d trading_system_db -f scripts/create_bbands_table.sql
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

# Test Phase 5.1: RSI
python scripts/test_rsi_api.py
python scripts/test_rsi_client.py
python scripts/test_rsi_pipeline.py
python scripts/test_rsi_scheduler.py
python scripts/test_rsi_complete.py

# Test Phase 5.2: MACD
python scripts/test_macd_api.py
python scripts/test_macd_client.py
python scripts/test_macd_pipeline.py
python scripts/test_macd_scheduler.py
python scripts/test_macd_complete.py

# Test Phase 5.3: BBANDS
python scripts/test_bbands_api.py
python scripts/test_bbands_client.py
python scripts/test_bbands_pipeline.py
python scripts/test_bbands_scheduler.py
python scripts/test_bbands_complete.py

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
│   │   ├── av_client.py          # AV client with 3 indicators (NO HARDCODING!)
│   │   └── ibkr_connection.py    # IBKR TWS connection
│   ├── data/
│   │   ├── ingestion.py          # RSI, MACD, BBANDS + options ingestion
│   │   ├── rate_limiter.py       # Token bucket (20% usage)
│   │   ├── cache_manager.py      # Redis cache manager
│   │   └── scheduler.py          # 115 automated jobs
│   └── __init__.py
│
├── config/                        # Configuration files
│   ├── .env                       # Environment variables
│   ├── .env.example               # Template for environment
│   ├── apis/
│   │   └── alpha_vantage.yaml    # RSI, MACD, BBANDS + options endpoints
│   ├── data/
│   │   └── schedules.yaml        # 3 indicators in indicators_fast
│   └── system/
│       └── redis.yaml             # Cache configuration
│
├── scripts/                       # Utility & test scripts
│   ├── init_db.sql               # System tables
│   ├── create_options_table.sql  # AV options schemas
│   ├── create_historical_options_table.sql
│   ├── create_ibkr_tables.sql    # IBKR data schemas
│   ├── create_rsi_table.sql      # RSI schema
│   ├── create_macd_table.sql     # MACD schema (3 values)
│   ├── create_bbands_table.sql   # BBANDS schema (3 bands)
│   ├── test_phase0.py            # Foundation tests
│   ├── test_phase2_complete.py   # AV integration tests
│   ├── test_ibkr_*.py            # IBKR tests (5 scripts)
│   ├── test_cache_*.py           # Cache tests (3 scripts)
│   ├── test_scheduler.py         # Scheduler tests
│   ├── test_rsi_*.py            # RSI tests (5 scripts)
│   ├── test_macd_*.py           # MACD tests (5 scripts)
│   ├── test_bbands_*.py         # BBANDS tests (5 scripts)
│   ├── run_scheduler.py          # Production scheduler
│   └── query_options_data.py     # Data analysis queries
│
├── data/                          # Data storage
│   └── api_responses/            # Saved API responses
│       ├── realtime_options_*.json
│       ├── historical_options_*.json
│       ├── rsi_*.json            # RSI response samples
│       ├── macd_*.json           # MACD response samples
│       └── bbands_*.json         # BBANDS response samples
│
├── docs/                          # Documentation
│   ├── SSOT-Ops.md              # Operational specification
│   ├── SSOT-Tech.md             # Technical specification
│   ├── granular-phased-plan.md  # 19-phase roadmap
│   ├── educational-timeline.md  # Educational development
│   ├── educational-content-plan.md # Content strategy
│   ├── phase5_rsi_summary.md    # RSI implementation
│   ├── phase5_macd_summary.md   # MACD implementation
│   └── phase5_bbands_summary.md # BBANDS implementation
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
| `av_rsi` | 83,239 | RSI momentum indicator | 60-600s |
| `av_macd` | 83,163 | MACD trend indicator (3 values) | 60-600s |
| `av_bbands` | 16,863 | Bollinger Bands volatility (3 bands) | 60-600s |

### Technical Indicator Schemas

#### RSI Table Schema
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

#### MACD Table Schema
```sql
CREATE TABLE av_macd (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    macd DECIMAL(10, 4),           -- MACD line value
    macd_signal DECIMAL(10, 4),    -- Signal line value  
    macd_hist DECIMAL(10, 4),       -- Histogram value
    interval VARCHAR(10) NOT NULL,
    fastperiod INTEGER NOT NULL,
    slowperiod INTEGER NOT NULL,
    signalperiod INTEGER NOT NULL,
    series_type VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, fastperiod, slowperiod, signalperiod)
);
```

#### BBANDS Table Schema
```sql
CREATE TABLE av_bbands (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    upper_band DECIMAL(10, 4),      -- Real Upper Band
    middle_band DECIMAL(10, 4),     -- Real Middle Band (SMA)
    lower_band DECIMAL(10, 4),      -- Real Lower Band
    interval VARCHAR(10) NOT NULL,
    time_period INTEGER NOT NULL,
    nbdevup INTEGER NOT NULL,
    nbdevdn INTEGER NOT NULL,
    matype INTEGER NOT NULL,
    series_type VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval, time_period, nbdevup, nbdevdn, matype)
);
```

### IBKR Tables (Ready for Monday)
| Table | Purpose | Update Frequency | Status |
|-------|---------|------------------|--------|
| `ibkr_bars_5sec` | Granular price action | Every 5 seconds | Schema ready |
| `ibkr_bars_1min` | Short-term analysis | Every minute | Schema ready |
| `ibkr_bars_5min` | Trading signals | Every 5 minutes | Schema ready |
| `ibkr_quotes` | Bid/ask spreads | Tick-by-tick | Schema ready |

### Redis Cache Keys (Phase 5.3)
| Key Pattern | TTL | Purpose | Hit Rate |
|-------------|-----|---------|----------|
| `av:realtime_options:{symbol}` | 30s | Options chains | 66.7% |
| `av:historical_options:{symbol}:{date}` | 24h | Historical options | N/A |
| `av:rsi:{symbol}:{interval}_{period}` | 60s | RSI values | 95%+ |
| `av:macd:{symbol}:{interval}_{f}_{s}_{sig}` | 60s | MACD values | 95%+ |
| `av:bbands:{symbol}:{interval}_{period}_{up}_{dn}_{ma}` | 60s | BBANDS values | 95%+ |

## 📈 API Usage & Examples

### Automated Scheduler Operations (Phase 5.3)
```python
from src.data.scheduler import DataScheduler

# Initialize scheduler (use test_mode=True on weekends)
scheduler = DataScheduler(test_mode=False)

# Start automated data collection
scheduler.start()

# Check status
status = scheduler.get_status()
print(f"Running: {status['running']}")
print(f"Total jobs: {status['total_jobs']}")  # Now 115 jobs
print(f"Market hours: {status['is_market_hours']}")

# View scheduled jobs
for job in status['jobs'][:10]:
    print(f"{job['name']}: Next run at {job['next_run']}")

# Stop scheduler
scheduler.stop()
```

### Fetch Indicators with NO HARDCODING
```python
from src.connections.av_client import AlphaVantageClient
from src.foundation.config_manager import ConfigManager

# Initialize client and config
client = AlphaVantageClient()
config = ConfigManager()

# RSI - Get config values, NO DEFAULTS!
rsi_config = config.av_config['endpoints']['rsi']['default_params']
rsi_data = client.get_rsi(
    'SPY',
    interval=rsi_config['interval'],
    time_period=rsi_config['time_period'],
    series_type=rsi_config['series_type']
)

# MACD - All parameters required
macd_config = config.av_config['endpoints']['macd']['default_params']
macd_data = client.get_macd(
    'SPY',
    interval=macd_config['interval'],
    fastperiod=macd_config['fastperiod'],
    slowperiod=macd_config['slowperiod'],
    signalperiod=macd_config['signalperiod'],
    series_type=macd_config['series_type']
)

# BBANDS - No hardcoded values!
bbands_config = config.av_config['endpoints']['bbands']['default_params']
bbands_data = client.get_bbands(
    'SPY',
    interval=bbands_config['interval'],
    time_period=bbands_config['time_period'],
    series_type=bbands_config['series_type'],
    nbdevup=bbands_config['nbdevup'],
    nbdevdn=bbands_config['nbdevdn'],
    matype=bbands_config['matype']
)
```

### Data Ingestion (NO DEFAULTS!)
```python
from src.data.ingestion import DataIngestion

ingestion = DataIngestion()

# RSI ingestion - ALL parameters required
records = ingestion.ingest_rsi_data(
    rsi_data, 'SPY', 
    interval='1min',  # Required, no default!
    time_period=14    # Required, no default!
)

# MACD ingestion - ALL parameters required
records = ingestion.ingest_macd_data(
    macd_data, 'SPY',
    interval='1min',
    fastperiod=12,
    slowperiod=26,
    signalperiod=9,
    series_type='close'
)

# BBANDS ingestion - ALL parameters required
records = ingestion.ingest_bbands_data(
    bbands_data, 'SPY',
    interval='5min',
    time_period=20,
    nbdevup=2,
    nbdevdn=2,
    matype=0,
    series_type='close'
)
```

### Query Indicator Data
```sql
-- Check all indicators freshness
SELECT 
    'RSI' as indicator,
    COUNT(*) as records,
    MAX(updated_at) as last_update
FROM av_rsi
UNION ALL
SELECT 
    'MACD' as indicator,
    COUNT(*) as records,
    MAX(updated_at) as last_update
FROM av_macd
UNION ALL
SELECT 
    'BBANDS' as indicator,
    COUNT(*) as records,
    MAX(updated_at) as last_update
FROM av_bbands;

-- Find trading signals across indicators
WITH signals AS (
    SELECT symbol, timestamp, 'RSI_OVERSOLD' as signal
    FROM av_rsi WHERE rsi < 30
    UNION ALL
    SELECT symbol, timestamp, 'RSI_OVERBOUGHT' as signal
    FROM av_rsi WHERE rsi > 70
    UNION ALL
    SELECT symbol, timestamp, 'MACD_BULLISH' as signal
    FROM av_macd 
    WHERE macd_hist > 0 AND LAG(macd_hist) OVER (PARTITION BY symbol ORDER BY timestamp) < 0
    UNION ALL
    SELECT symbol, timestamp, 'BBANDS_SQUEEZE' as signal
    FROM av_bbands
    WHERE (upper_band - lower_band) < (SELECT AVG(upper_band - lower_band) * 0.5 FROM av_bbands)
)
SELECT * FROM signals
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;
```

## 🧪 Testing

### Test Suite by Phase and Indicator
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

# Phase 5.1: RSI (Momentum)
python scripts/test_rsi_api.py           # API discovery
python scripts/test_rsi_client.py        # Client method
python scripts/test_rsi_pipeline.py      # Full pipeline
python scripts/test_rsi_scheduler.py     # Scheduler integration
python scripts/test_rsi_complete.py      # Comprehensive test

# Phase 5.2: MACD (Trend)
python scripts/test_macd_api.py          # API discovery
python scripts/test_macd_client.py       # Client method
python scripts/test_macd_pipeline.py     # Full pipeline
python scripts/test_macd_scheduler.py    # Scheduler integration
python scripts/test_macd_complete.py     # Comprehensive test

# Phase 5.3: BBANDS (Volatility)
python scripts/test_bbands_api.py        # API discovery
python scripts/test_bbands_client.py     # Client method
python scripts/test_bbands_pipeline.py   # Full pipeline
python scripts/test_bbands_scheduler.py  # Scheduler integration
python scripts/test_bbands_complete.py   # Comprehensive test

# Data Analysis
python scripts/query_options_data.py      # Analyze stored data
```

### Performance Benchmarks (Phase 5.3)
| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| AV API Call (no cache) | < 2s | ~1.01s | ✅ |
| AV API Call (cached) | < 100ms | ~10ms | ✅ |
| RSI Fetch (no cache) | < 2s | 0.58s | ✅ |
| RSI Fetch (cached) | < 100ms | 0.01s | ✅ |
| MACD Fetch (no cache) | < 2s | 0.95s | ✅ |
| MACD Fetch (cached) | < 100ms | 0.01s | ✅ |
| BBANDS Fetch (no cache) | < 2s | 0.44s | ✅ |
| BBANDS Fetch (cached) | < 100ms | 0.003s | ✅ |
| Cache Hit Rate | > 50% | 80%+ avg | ✅ |
| Scheduler Jobs | N/A | 115 | ✅ |
| Job Execution | < 2s | ~1s | ✅ |
| IBKR Connection | < 5s | ~2s | ✅ |
| Database Insert (21K records) | < 30s | ~8s | ✅ |
| Database Insert (4K BBANDS) | < 10s | ~4s | ✅ |
| Query Options Chain | < 100ms | ~45ms | ✅ |
| Query Indicator Data | < 100ms | ~40ms | ✅ |
| Real-time Bar Latency | < 500ms | ~100ms | ✅ |
| Rate Limiter Check | < 10ms | ~1ms | ✅ |
| Cache Get/Set | < 10ms | ~3ms | ✅ |

## 📊 Current Data Holdings

| Data Type | Source | Count | Update Frequency | Cache TTL |
|-----------|--------|-------|------------------|-----------|
| **Options Contracts** | Alpha Vantage | 49,854+ | Every 30-180s | 30 seconds |
| **Historical Options** | Alpha Vantage | 49,854+ | Daily at 6 AM | 24 hours |
| **RSI Indicator** | Alpha Vantage | 83,239 | Every 60-600s | 60 seconds |
| **MACD Indicator** | Alpha Vantage | 83,163 | Every 60-600s | 60 seconds |
| **BBANDS Indicator** | Alpha Vantage | 16,863 | Every 60-600s | 60 seconds |
| **Real-time Bars** | IBKR | 0 (Monday) | 5 seconds | N/A |
| **Quotes** | IBKR | 0 (Monday) | Tick-by-tick | N/A |
| **Symbols Tracked** | Both | 23 | Continuous | - |
| **Scheduled Jobs** | Scheduler | 115 | Various | - |
| **Total Data Points** | All | 233,119+ | - | - |

### Live Market Data Insights (August 17, 2025)
- **Symbols Tracked:** 23 across 3 tiers
- **Total Data Points:** 233,119 (options + 3 indicators)
- **Cache Hit Rate:** 80%+ average
- **API Usage:** Only 20% of capacity
- **Automation Level:** 100% hands-free
- **RSI Coverage:** 22 days of 1-minute data
- **MACD Coverage:** 22 days with 6,672 crossovers
- **BBANDS Coverage:** 22 days with squeeze detection

### Symbols by Tier
**Tier A (High Priority - Fastest updates):**
- SPY, QQQ, IWM, IBIT
- Options: 30s, Indicators: 60s

**Tier B (Medium Priority):**
- AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA
- Options: 60s, Indicators: 300s

**Tier C (Standard Priority):**
- DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW
- Options: 180s, Indicators: 600s

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
    
  indicators_fast:  # RSI, MACD, BBANDS added
    apis: ["RSI", "MACD", "BBANDS"]
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
    
  rsi:
    function: "RSI"
    datatype: "json"
    cache_ttl: 60
    default_params:
      interval: "1min"
      time_period: 14
      series_type: "close"
      
  macd:
    function: "MACD"
    datatype: "json"
    cache_ttl: 60
    default_params:
      interval: "1min"
      fastperiod: 12
      slowperiod: 26
      signalperiod: 9
      series_type: "close"
      
  bbands:
    function: "BBANDS"
    datatype: "json"
    cache_ttl: 60
    default_params:
      interval: "5min"
      time_period: 20
      series_type: "close"
      nbdevup: 2
      nbdevdn: 2
      matype: 0
```

### Redis Cache (`config/system/redis.yaml`)
```yaml
cache_ttl:
  realtime_options: 30      # 30 seconds
  historical_options: 86400  # 24 hours
  rsi: 60                   # 60 seconds
  macd: 60                  # 60 seconds
  bbands: 60                # 60 seconds
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

### ✅ Completed Phases (Days 1-20)
| Phase | Days | Description | Status | Key Achievement |
|-------|------|-------------|--------|-----------------|
| **0** | 1-3 | Minimal Foundation | ✅ Complete | Zero hardcoding |
| **1** | 4-7 | First API Pipeline | ✅ Complete | 9,294 contracts |
| **2** | 8-10 | Rate Limiting & 2nd API | ✅ Complete | 600/min protection |
| **3** | 11-14 | IBKR Integration | ✅ Complete | Real-time ready |
| **4** | 15-17 | Scheduler & Cache | ✅ Complete | 46 jobs automated |
| **5.1** | 18 | RSI Indicator | ✅ Complete | 83,239 data points |
| **5.2** | 19 | MACD Indicator | ✅ Complete | 83,163 data points |
| **5.3** | 20 | BBANDS Indicator | ✅ Complete | 16,863 data points |

### 🚧 In Progress - Phase 5 Continuation
| Indicator | Day | Status | Implementation | Notes |
|-----------|-----|--------|---------------|-------|
| RSI | 18 | ✅ Complete | 83,239 records | 109.4x cache |
| MACD | 19 | ✅ Complete | 83,163 records | 110.2x cache |
| BBANDS | 20 | ✅ Complete | 16,863 records | 127.4x cache |
| VWAP | 21 | 📋 Next | Ready to start | Volume indicator |
| ATR | 22 | 📋 Planned | - | Volatility |
| ADX | 23 | 📋 Planned | - | Trend strength |
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

# Check indicator cache keys
redis-cli --scan --pattern "av:rsi:*"
redis-cli --scan --pattern "av:macd:*"
redis-cli --scan --pattern "av:bbands:*"
```

### Cache Miss Issues
```bash
# Check cache contents
redis-cli KEYS "av:*"

# Get TTL for a key
redis-cli TTL "av:rsi:SPY:1min_14"
redis-cli TTL "av:macd:SPY:1min_12_26_9"
redis-cli TTL "av:bbands:SPY:5min_20_2_2_0"

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

### Hardcoded Values Check (CRITICAL!)
```bash
# Search for hardcoded defaults - SHOULD RETURN NOTHING!
grep -r "='SPY'" src/
grep -r "interval='1min'" src/
grep -r "interval='5min'" src/
grep -r "time_period=14" src/
grep -r "time_period=20" src/
grep -r "fastperiod=12" src/
grep -r "slowperiod=26" src/
grep -r "signalperiod=9" src/
grep -r "nbdevup=2" src/
grep -r "nbdevdn=2" src/
grep -r "matype=0" src/

# All should return NOTHING - all defaults from config
```

## 📚 Documentation

- **[SSOT-Ops.md](docs/SSOT-Ops.md)** - Operational specifications
- **[SSOT-Tech.md](docs/SSOT-Tech.md)** - Technical implementation
- **[Phased Plan](docs/granular-phased-plan.md)** - Complete 19-phase roadmap
- **[RSI Implementation](docs/phase5_rsi_summary.md)** - RSI details
- **[MACD Implementation](docs/phase5_macd_summary.md)** - MACD details
- **[BBANDS Implementation](docs/phase5_bbands_summary.md)** - BBANDS details
- **[Quick Reference](quick_reference_guide.md)** - Common commands
- **[Status Report](project_status_report.md)** - Current progress
- **[Educational Plan](docs/educational-content-plan.md)** - Content strategy

## 🎯 Next Steps

### Immediate (Day 21 - Monday)
1. **VWAP Implementation**
   - Follow 8-step process exactly
   - Expect 1 value per timestamp (volume-weighted)
   - Consider intraday vs daily VWAP
   - Test with SPY first
   - NO HARDCODED VALUES!

### This Week (Phase 5 Completion)
- [ ] Day 21: VWAP implementation (Volume indicator)
- [ ] Day 22: ATR implementation (Volatility)
- [ ] Day 23: ADX implementation (Trend strength)
- [ ] Day 24: Integration testing all 6 indicators

### This Month (Through Phase 9)
- [ ] Complete all 16 technical indicators
- [ ] Implement Greeks validation
- [ ] Build first trading strategy (0DTE)
- [ ] Add risk management
- [ ] Begin paper trading (Day 40)

## 📊 Key Metrics

| Metric | Value | Change |
|--------|-------|--------|
| **Development Day** | 20 of 106 | +2 |
| **Progress** | 18.9% complete | +1.9% |
| **Phase Status** | 5.3 of 19 complete | +0.2 |
| **Lines of Code** | ~3,500 | +1000 |
| **Database Tables** | 11 | +2 |
| **Database Size** | 65MB | +18MB |
| **Scheduled Jobs** | 115 | +46 |
| **API Usage** | 100/min (20%) | +54/min |
| **Cache Performance** | 127.4x on BBANDS | NEW |
| **Total Data Points** | 233,119 | +100K |
| **Test Scripts** | 33 | +10 |
| **Status** | ✅ On Schedule | - |

## 🏆 Achievements

### Phase 5.3 Specific
- ✅ BBANDS implementation in ~1.5 hours
- ✅ 16,863 BBANDS data points ingested
- ✅ Squeeze conditions detected
- ✅ 127.4x cache performance (best yet)
- ✅ 5-minute intervals for better analysis
- ✅ 23 new scheduled jobs
- ✅ Clean 8-step implementation process

### Phase 5 Progress (50% Complete)
- ✅ 3 of 6 indicators operational
- ✅ Momentum (RSI), Trend (MACD), Volatility (BBANDS) covered
- ✅ 183,265 indicator data points total
- ✅ Zero hardcoded values maintained throughout
- ✅ All parameters configuration-driven
- ✅ Cache performance exceeding 100x on all indicators

### Overall Project
- ✅ Clean architecture maintained throughout
- ✅ Zero hardcoded values achieved and enforced
- ✅ Rate limiting never exceeded
- ✅ IBKR integration successful
- ✅ Cache layer performing exceptionally
- ✅ Production-ready automation
- ✅ Scalable to 100+ symbols
- ✅ Test coverage comprehensive (33 test scripts)
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
- [ ] Morning: Check scheduler status (115 jobs)
- [ ] Verify cache hit rate > 80%
- [ ] Monitor API usage < 150/min
- [ ] Check database growth (~5MB/day)
- [ ] Review any error logs
- [ ] Evening: Daily performance summary

### Indicator-Specific Monitoring
```sql
-- Check all indicator data freshness
SELECT 
    indicator,
    symbol,
    data_points,
    last_update,
    age_minutes
FROM (
    SELECT 'RSI' as indicator, symbol, COUNT(*) as data_points, 
           MAX(updated_at) as last_update,
           EXTRACT(EPOCH FROM (NOW() - MAX(updated_at)))/60 as age_minutes
    FROM av_rsi GROUP BY symbol
    UNION ALL
    SELECT 'MACD', symbol, COUNT(*), MAX(updated_at),
           EXTRACT(EPOCH FROM (NOW() - MAX(updated_at)))/60
    FROM av_macd GROUP BY symbol
    UNION ALL
    SELECT 'BBANDS', symbol, COUNT(*), MAX(updated_at),
           EXTRACT(EPOCH FROM (NOW() - MAX(updated_at)))/60
    FROM av_bbands GROUP BY symbol
) t
ORDER BY indicator, symbol;

-- Find convergence of signals
WITH combined_signals AS (
    SELECT 
        r.symbol,
        r.timestamp,
        r.rsi,
        m.macd_hist,
        (b.upper_band - b.lower_band) as bandwidth,
        CASE 
            WHEN r.rsi < 30 AND m.macd_hist > 0 THEN 'STRONG_BUY'
            WHEN r.rsi > 70 AND m.macd_hist < 0 THEN 'STRONG_SELL'
            WHEN r.rsi < 30 THEN 'BUY_SIGNAL'
            WHEN r.rsi > 70 THEN 'SELL_SIGNAL'
            ELSE 'NEUTRAL'
        END as signal
    FROM av_rsi r
    JOIN av_macd m ON r.symbol = m.symbol 
        AND r.timestamp = m.timestamp
    LEFT JOIN av_bbands b ON r.symbol = b.symbol 
        AND DATE_TRUNC('minute', r.timestamp) = DATE_TRUNC('minute', b.timestamp)
    WHERE r.timestamp > NOW() - INTERVAL '1 hour'
)
SELECT * FROM combined_signals
WHERE signal != 'NEUTRAL'
ORDER BY timestamp DESC;
```

## 📮 Contact

For questions or issues, create an issue in the repository.

---

**Current Phase:** 5.3 Complete - BBANDS Operational ✅  
**Next Phase:** 5.4 - VWAP Implementation (Day 21)  
**First Paper Trade:** Day 40 (20 days away)  
**Production Launch:** Day 107 (87 days away)

*Last Updated: August 17, 2025, 2:00 PM ET - BBANDS with 16,863 data points across 23 symbols, 127.4x cache performance, 3 of 6 indicators complete*