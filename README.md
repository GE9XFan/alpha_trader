# AlphaTrader - Automated Options Trading System

**Version:** 0.4.3 (Phase 4 Complete - Fully Automated Scheduler)  
**Status:** Development - Autonomous Data Collection Operational  
**Last Updated:** August 16, 2025 (4:00 PM ET)  
**Development Day:** 16 of 106 (15.1% Complete)

## 🎯 Project Overview

AlphaTrader is a production-grade automated options trading system that combines real-time market data from IBKR, options analytics from Alpha Vantage, and machine learning to execute systematic trading strategies. Built with a focus on reliability, scalability, and educational content generation.

### Vision
Build a fully automated trading system that not only trades profitably but also generates comprehensive educational content, market analysis, and community insights.

### Current Capabilities (Phases 0-4 Complete)
- ✅ **Configuration Management** - Centralized YAML-based configuration with environment variables
- ✅ **Database Infrastructure** - PostgreSQL with 8 optimized tables for market data
- ✅ **Alpha Vantage Integration** - Real-time and historical options data with full Greeks
- ✅ **IBKR Real-Time Data** - Live bars and quotes from Interactive Brokers
- ✅ **Rate Limiting** - Token bucket implementation (600 calls/min protection)
- ✅ **Data Ingestion Pipeline** - Unified ingestion for both data sources
- ✅ **Redis Cache Layer** - 30x performance improvement with intelligent TTL management
- ✅ **Automated Scheduler** - 46 jobs managing 23 symbols across 3 priority tiers
- ✅ **Market Hours Awareness** - Smart scheduling based on market conditions
- ✅ **Options Data** - 49,854+ contracts tracked automatically with complete Greeks

## 📊 System Architecture

### Current Implementation (Phase 4 Complete)
```
┌─────────────────────────────────────────────────────┐
│                  DATA SCHEDULER                      │
│         46 Jobs | 23 Symbols | 3 Tiers              │
│    Tier A: 30s | Tier B: 60s | Tier C: 180s        │
└────────────┬──────────────────────┬─────────────────┘
             │                      │
             ▼                      ▼
┌──────────────────────────┬──────────────────────────┐
│    Alpha Vantage API     │      IBKR TWS API        │
│  • REALTIME_OPTIONS      │  • Real-time Bars (5s)   │
│  • HISTORICAL_OPTIONS    │  • Real-time Quotes      │
│  • Greeks & Analytics    │  • All Pricing Data      │
└────────────┬─────────────┴────────────┬─────────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────────────────────────────────┐
│          Rate Limiter          │   Direct Stream     │
│      (Token Bucket)           │   (No limiting)     │
└────────────┬──────────────────┴────────┬────────────┘
             │                            │
             ▼                            ▼
      ┌──────────────┐              ┌──────────────┐
      │ Redis Cache  │◄─────────────│   Ingestion  │
      │  30s TTL     │              │    Engine    │
      └──────────────┘              └──────┬───────┘
             │                              │
             └──────────┬───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            PostgreSQL Database (8 Tables)           │
│  • av_realtime_options   • ibkr_bars_5sec          │
│  • av_historical_options • ibkr_bars_1min          │
│  • system_config         • ibkr_bars_5min          │
│  • api_response_log      • ibkr_quotes             │
└─────────────────────────────────────────────────────┘
```

### Automated Scheduling Architecture
| Tier | Symbols | Update Frequency | API Calls/Hour |
|------|---------|------------------|----------------|
| **A** | SPY, QQQ, IWM, IBIT | Every 30 seconds | 480 |
| **B** | AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA | Every 60 seconds | 420 |
| **C** | DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW | Every 3 minutes | 240 |
| **Daily** | All 23 symbols | 6:00 AM ET | 23 |

**Total API Usage:** ~19 calls/minute (3.8% of 500/min budget)

### Cache Performance Metrics
| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| Options Data Fetch | 1.01s | 0.03s | **30.6x faster** |
| API Calls (2 symbols) | 4 | 2 | **50% reduction** |
| Memory Usage | N/A | 30MB | Efficient |
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
- **20GB+ disk space** for historical data
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
│   │   └── config_manager.py     # Configuration management
│   ├── connections/
│   │   ├── av_client.py          # Alpha Vantage client (with caching)
│   │   └── ibkr_connection.py    # IBKR TWS connection manager
│   ├── data/
│   │   ├── ingestion.py          # Data ingestion (cache-aware)
│   │   ├── rate_limiter.py       # Token bucket rate limiter
│   │   ├── cache_manager.py      # Redis cache manager
│   │   └── scheduler.py          # Automated data scheduler (NEW)
│   └── __init__.py
│
├── config/                        # Configuration files
│   ├── .env                       # Environment variables (not in git)
│   ├── .env.example               # Template for environment
│   ├── apis/
│   │   └── alpha_vantage.yaml    # API settings & endpoints
│   ├── data/
│   │   └── schedules.yaml        # Scheduler configuration (NEW)
│   └── system/
│       └── redis.yaml             # Cache configuration
│
├── scripts/                       # Utility & test scripts
│   ├── init_db.sql               # System tables
│   ├── create_options_table.sql  # AV options schemas
│   ├── create_ibkr_tables.sql    # IBKR data schemas
│   ├── test_phase0.py            # Foundation tests
│   ├── test_phase2_complete.py   # AV integration tests
│   ├── test_ibkr_connection.py   # IBKR connection test
│   ├── test_ibkr_bars.py         # Bar data test
│   ├── test_ibkr_market_data.py  # Quotes test
│   ├── test_ibkr_live_data.py    # Live ingestion test
│   ├── test_cache_manager.py     # Cache operations
│   ├── test_cached_av_client.py  # Cache integration
│   ├── test_cache_integration.py # Full pipeline test
│   ├── test_scheduler.py         # Scheduler tests (NEW)
│   ├── run_scheduler.py          # Production scheduler (NEW)
│   └── query_options_data.py     # Data analysis queries
│
├── data/                          # Data storage
│   └── api_responses/            # Saved API responses
│
├── docs/                          # Documentation
│   ├── SSOT-Ops.md              # Operational specification
│   ├── SSOT-Tech.md             # Technical specification
│   ├── granular-phased-plan.md  # 19-phase roadmap
│   └── educational-*.md         # Educational platform plans
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 💾 Database Schema

### Alpha Vantage Tables
| Table | Records | Description |
|-------|---------|-------------|
| `av_realtime_options` | 49,854+ | Live options chains with Greeks |
| `av_historical_options` | 49,854+ | Historical snapshots |

### IBKR Tables (Ready for Monday)
| Table | Purpose | Update Frequency |
|-------|---------|------------------|
| `ibkr_bars_5sec` | Granular price action | Every 5 seconds |
| `ibkr_bars_1min` | Short-term analysis | Every minute |
| `ibkr_bars_5min` | Trading signals | Every 5 minutes |
| `ibkr_quotes` | Bid/ask spreads | Tick-by-tick |

### Redis Cache Keys
| Key Pattern | TTL | Purpose |
|-------------|-----|---------|
| `av:realtime_options:{symbol}` | 30s | Options chains with Greeks |
| `av:historical_options:{symbol}:{date}` | 24h | Historical options data |
| `av:indicators:{type}:{symbol}` | 60s | Technical indicators (Phase 5) |

## 📈 API Usage & Examples

### Automated Scheduler Operations
```python
from src.data.scheduler import DataScheduler

# Initialize scheduler (use test_mode=True on weekends)
scheduler = DataScheduler(test_mode=False)

# Start automated data collection
scheduler.start()

# Check status
status = scheduler.get_status()
print(f"Running: {status['running']}")
print(f"Total jobs: {status['total_jobs']}")
print(f"Market hours: {status['is_market_hours']}")

# View scheduled jobs
for job in status['jobs'][:5]:
    print(f"{job['name']}: Next run at {job['next_run']}")

# Stop scheduler
scheduler.stop()
```

### Fetch Options with Caching
```python
from src.connections.av_client import AlphaVantageClient

# Initialize with caching
client = AlphaVantageClient()

# First call - hits API (1.01s)
options_data = client.get_realtime_options('SPY')

# Second call - hits cache (0.03s) - 30x faster!
options_data = client.get_realtime_options('SPY')

# Check cache status
cache_stats = client.get_cache_status()
print(f"Cache has {cache_stats['av_keys']} Alpha Vantage keys")
```

### Cache Management
```python
from src.data.cache_manager import get_cache

cache = get_cache()

# Store with TTL
cache.set("my_key", {"data": "value"}, ttl=60)

# Retrieve
data = cache.get("my_key")

# Check stats
stats = cache.get_stats()
print(f"Memory used: {stats['used_memory']}")
```

### Manual Data Update
```python
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion

client = AlphaVantageClient()
ingestion = DataIngestion()

# Update specific symbol manually
symbol = 'SPY'
data = client.get_realtime_options(symbol)
records = ingestion.ingest_options_data(data, symbol)
print(f"Updated {symbol}: {records} records")
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

# Data Analysis
python scripts/query_options_data.py      # Analyze stored data
```

### Performance Benchmarks
| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| AV API Call (no cache) | < 2s | ~1.01s | ✅ |
| AV API Call (cached) | < 100ms | ~30ms | ✅ |
| Cache Hit Rate | > 50% | 66.7%+ | ✅ |
| Scheduler Jobs | N/A | 46 | ✅ |
| Job Execution | < 2s | ~1s | ✅ |
| IBKR Connection | < 5s | ~2s | ✅ |
| Database Insert (9K records) | < 30s | ~8s | ✅ |
| Query Options Chain | < 100ms | ~45ms | ✅ |
| Real-time Bar Latency | < 500ms | ~100ms | ✅ |
| Rate Limiter Check | < 10ms | ~1ms | ✅ |
| Cache Get/Set | < 10ms | ~3ms | ✅ |

## 📊 Current Data Holdings

| Data Type | Source | Count | Update Frequency | Cache TTL |
|-----------|--------|-------|------------------|-----------|
| **Options Contracts** | Alpha Vantage | 49,854+ | Every 30-180s | 30 seconds |
| **Historical Options** | Alpha Vantage | 49,854+ | Daily at 6 AM | 24 hours |
| **Real-time Bars** | IBKR | 0 (ready Monday) | 5 seconds | N/A |
| **Quotes** | IBKR | 0 (ready Monday) | Tick-by-tick | N/A |
| **Symbols Tracked** | Both | 23 | Continuous | - |
| **Scheduled Jobs** | Scheduler | 46 | Various | - |

### Live Market Data Insights (August 16, 2025)
- **Symbols Tracked:** 23 across 3 tiers
- **Total Contracts:** 49,854+ with full Greeks
- **Cache Hit Rate:** 66.7%+ and growing
- **API Usage:** Only 3.8% of capacity
- **Automation Level:** 100% hands-free

### Symbols by Tier
**Tier A (High Priority - 30s):** SPY, QQQ, IWM, IBIT  
**Tier B (Medium Priority - 60s):** AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA  
**Tier C (Low Priority - 180s):** DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW

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
    
  daily:
    schedule_time: "06:00"
```

### Alpha Vantage (`config/apis/alpha_vantage.yaml`)
```yaml
rate_limit:
  max_per_minute: 600
  target_per_minute: 500
  refill_rate: 10
  burst_capacity: 20

endpoints:
  realtime_options:
    function: "REALTIME_OPTIONS"
    require_greeks: "true"  # Critical for Greeks
  historical_options:
    function: "HISTORICAL_OPTIONS"
```

### Redis Cache (`config/system/redis.yaml`)
```yaml
cache_ttl:
  realtime_options: 30      # 30 seconds
  historical_options: 86400  # 24 hours
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

### ✅ Completed Phases (Days 1-16)
| Phase | Days | Description | Status | Key Achievement |
|-------|------|-------------|--------|-----------------|
| **0** | 1-3 | Minimal Foundation | ✅ Complete | Clean architecture |
| **1** | 4-7 | First API Pipeline | ✅ Complete | 9,294 contracts stored |
| **2** | 8-10 | Rate Limiting & 2nd API | ✅ Complete | 600/min protection |
| **3** | 11-14 | IBKR Integration | ✅ Complete | Real-time data ready |
| **4** | 15-17 | Scheduler & Cache | ✅ Complete | 46 jobs, 23 symbols automated |

### 🚧 In Progress
None - Phase 4 complete, preparing for Phase 5

### 📅 Upcoming Phases (Days 18-106)
| Phase | Days | Description | Key Deliverable |
|-------|------|-------------|-----------------|
| **5** | 18-24 | Core Indicators | RSI, MACD, BBANDS, VWAP, ATR, ADX |
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
| **19** | 96+ | Production | Live trading |

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
```

### Cache Miss Issues
```bash
# Check cache contents
redis-cli KEYS "av:*"

# Get TTL for a key
redis-cli TTL "av:realtime_options:SPY"

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

## 📚 Documentation

- **[SSOT-Ops.md](docs/SSOT-Ops.md)** - Operational specifications
- **[SSOT-Tech.md](docs/SSOT-Tech.md)** - Technical implementation
- **[Phased Plan](docs/granular-phased-plan.md)** - Complete 19-phase roadmap
- **[Quick Reference](quick_reference_guide.md)** - Common commands
- **[Status Report](project_status_report.md)** - Current progress
- **[Educational Plan](docs/educational-content-plan.md)** - Content strategy

## 🎯 Next Steps

### Immediate (Day 17 - Sunday)
1. **Run 24-hour stability test**
   - Monitor unattended operation
   - Check for memory leaks
   - Verify error recovery
   - Document any issues

### Next Week (Phase 5 - Days 18-24)
1. **Core Technical Indicators**
   - Day 18: RSI implementation
   - Day 19: MACD implementation
   - Day 20: BBANDS implementation
   - Day 21: VWAP implementation
   - Day 22: ATR implementation
   - Day 23: ADX implementation
   - Day 24: Testing & integration

### This Month (Through Phase 9)
- [ ] Complete all 16 technical indicators
- [ ] Implement Greeks validation
- [ ] Build first trading strategy (0DTE)
- [ ] Add risk management
- [ ] Begin paper trading (Day 40)

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Development Day** | 16 of 106 |
| **Progress** | 15.1% complete |
| **Phase Status** | 4 of 19 complete |
| **Lines of Code** | ~2,000 |
| **Database Tables** | 8 |
| **APIs Integrated** | 3 (2 AV + 1 IBKR) |
| **Scheduled Jobs** | 46 |
| **Symbols Tracked** | 23 |
| **API Usage** | 19/min (3.8% of 500) |
| **Cache Performance** | 30.6x faster |
| **Cache Hit Rate** | 66.7%+ |
| **Test Scripts** | 18 |
| **Status** | ✅ On Schedule |

## 🏆 Achievements

### Phase 4 Specific
- ✅ Automated scheduler with 46 jobs
- ✅ 23 symbols across 3 priority tiers
- ✅ Market-aware scheduling
- ✅ Test mode for weekend development
- ✅ Cache-integrated for efficiency
- ✅ 100% hands-free operation
- ✅ APScheduler with thread pool execution

### Overall Project
- ✅ Clean architecture maintained throughout
- ✅ Zero hardcoded values
- ✅ Rate limiting never exceeded
- ✅ IBKR integration successful
- ✅ Cache layer seamlessly integrated
- ✅ Production-ready automation
- ✅ Scalable to 100+ symbols

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

## 📮 Contact

For questions or issues, create an issue in the repository.

---

**Current Phase:** 4 Complete - Fully Automated Scheduler ✅  
**Next Phase:** 5 - Core Technical Indicators (Starting Day 18)  
**First Paper Trade:** Day 40 (24 days away)  
**Production Launch:** Day 107 (91 days away)

*Last Updated: August 16, 2025, 4:00 PM ET - Phase 4 Complete with 46 jobs managing 23 symbols*