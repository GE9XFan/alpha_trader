# AlphaTrader - Automated Options Trading System

**Version:** 0.4.1 (Phase 4 In Progress - Cache Layer Complete)  
**Status:** Development - Automated Data Collection Building  
**Last Updated:** August 16, 2025 (3:00 PM ET)  
**Development Day:** 15 of 106 (14.2% Complete)

## 🎯 Project Overview

AlphaTrader is a production-grade automated options trading system that combines real-time market data from IBKR, options analytics from Alpha Vantage, and machine learning to execute systematic trading strategies. Built with a focus on reliability, scalability, and educational content generation.

### Vision
Build a fully automated trading system that not only trades profitably but also generates comprehensive educational content, market analysis, and community insights.

### Current Capabilities (Phases 0-4.1 Complete)
- ✅ **Configuration Management** - Centralized YAML-based configuration with environment variables
- ✅ **Database Infrastructure** - PostgreSQL with 8 optimized tables for market data
- ✅ **Alpha Vantage Integration** - Real-time and historical options data with full Greeks
- ✅ **IBKR Real-Time Data** - Live bars and quotes from Interactive Brokers
- ✅ **Rate Limiting** - Token bucket implementation (600 calls/min protection)
- ✅ **Data Ingestion Pipeline** - Unified ingestion for both data sources
- ✅ **Redis Cache Layer** - 30x performance improvement with intelligent TTL management
- ✅ **Options Data** - 17,554+ contracts stored with complete Greeks (SPY + QQQ)
- ✅ **Modular Architecture** - Clean separation of concerns with extensible design

## 📊 System Architecture

### Current Implementation (Phase 4.1)
```
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

### Cache Performance Metrics
| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| Options Data Fetch | 1.01s | 0.03s | **30.6x faster** |
| API Calls (2 symbols) | 4 | 2 | **50% reduction** |
| Memory Usage | N/A | 8.48MB | Efficient |
| Cache Hit Rate | 0% | 66.7% | Excellent |

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.11+** (tested on 3.11.11)
- **PostgreSQL 14+** (for production data storage)
- **Redis 8.0+** (for caching layer) ✅ NEW
- **Interactive Brokers TWS** (paper or live account)
- **Alpha Vantage API Key** (Premium recommended for 600 calls/min)
- **macOS/Linux** (primary development on macOS)
- **4GB+ RAM** for data processing
- **20GB+ disk space** for historical data

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

### Step 3: Redis Setup (NEW)
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
REDIS_URL=redis://localhost:6379/0  # NEW

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

# Test Phase 4.1: Cache Manager (NEW)
python scripts/test_cache_manager.py
python scripts/test_cached_av_client.py
python scripts/test_cache_integration.py

# Test Live Data (run during market hours)
python scripts/test_ibkr_live_data.py
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
│   │   └── cache_manager.py      # Redis cache manager (NEW)
│   └── __init__.py
│
├── config/                        # Configuration files
│   ├── .env                       # Environment variables (not in git)
│   ├── .env.example               # Template for environment
│   ├── apis/
│   │   └── alpha_vantage.yaml    # API settings & endpoints
│   └── system/
│       └── redis.yaml             # Cache configuration (NEW)
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
│   ├── test_cache_manager.py     # Cache operations (NEW)
│   ├── test_cached_av_client.py  # Cache integration (NEW)
│   ├── test_cache_integration.py # Full pipeline test (NEW)
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
| `av_realtime_options` | 17,554 | Live options chains with Greeks |
| `av_historical_options` | 17,554 | Historical snapshots |

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

### Fetch Options with Caching (NEW)
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

### Cache Management (NEW)
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

# Phase 4.1: Cache Layer (NEW)
python scripts/test_cache_manager.py      # Basic cache ops
python scripts/test_cached_av_client.py   # Cached API calls
python scripts/test_cache_integration.py  # Full pipeline with cache

# Data Analysis
python scripts/query_options_data.py      # Analyze stored data
```

### Performance Benchmarks
| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| AV API Call (no cache) | < 2s | ~1.01s | ✅ |
| AV API Call (cached) | < 100ms | ~30ms | ✅ NEW |
| Cache Hit Rate | > 50% | 66.7% | ✅ NEW |
| IBKR Connection | < 5s | ~2s | ✅ |
| Database Insert (9K records) | < 30s | ~8s | ✅ |
| Query Options Chain | < 100ms | ~45ms | ✅ |
| Real-time Bar Latency | < 500ms | ~100ms | ✅ |
| Rate Limiter Check | < 10ms | ~1ms | ✅ |
| Cache Get/Set | < 10ms | ~3ms | ✅ NEW |

## 📊 Current Data Holdings

| Data Type | Source | Count | Update Frequency | Cache TTL |
|-----------|--------|-------|------------------|-----------|
| **Options Contracts** | Alpha Vantage | 17,554 | Manual (30s planned) | 30 seconds |
| **Historical Options** | Alpha Vantage | 17,554 | Daily | 24 hours |
| **Real-time Bars** | IBKR | 0 (ready Monday) | 5 seconds | N/A |
| **Quotes** | IBKR | 0 (ready Monday) | Tick-by-tick | N/A |
| **Symbols Supported** | Both | SPY, QQQ, IWM | Expanding Phase 5 | - |

### Live Market Data Insights (August 16, 2025)
- **Massive 0DTE Volume:** SPY 645C with 610,529 contracts!
- **Market Level:** SPY around $643
- **Active Strikes:** $150 to $1,000 range
- **Expirations:** 33 dates from 0DTE to Dec 2027

## 🛠️ Configuration

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
```

### Redis Cache (`config/system/redis.yaml`) - NEW
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

# Cache (NEW)
REDIS_URL=redis://localhost:6379/0
```

## 🚦 Development Status

### ✅ Completed Phases (Days 1-15)
| Phase | Days | Description | Status | Key Achievement |
|-------|------|-------------|--------|-----------------|
| **0** | 1-3 | Minimal Foundation | ✅ Complete | Clean architecture |
| **1** | 4-7 | First API Pipeline | ✅ Complete | 9,294 contracts stored |
| **2** | 8-10 | Rate Limiting & 2nd API | ✅ Complete | 600/min protection |
| **3** | 11-14 | IBKR Integration | ✅ Complete | Real-time data ready |
| **4.1** | 15 | Cache Manager | ✅ Complete | 30x speed improvement |

### 🚧 In Progress (Day 16-17)
| Phase | Days | Description | Next Step |
|-------|------|-------------|-----------|
| **4.2** | 16 | Scheduler | Implement DataScheduler class |
| **4.3** | 17 | Integration & Testing | Full automation |

### 📅 Upcoming Phases (Days 18-106)
| Phase | Days | Description | Key Deliverable |
|-------|------|-------------|-----------------|
| **5** | 18-24 | Core Indicators | RSI, MACD, etc. |
| **6** | 25-28 | Analytics & Validation | Greeks validator |
| **7** | 29-35 | First Strategy (0DTE) | Trading logic |
| **8** | 36-39 | Risk Management | Position limits |
| **9** | 40-43 | Paper Trading | **First trades!** |
| **10-19** | 44-106 | Full System | ML, Publishing, Production |

## 🐛 Troubleshooting

### Redis Issues (NEW)
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

## 📚 Documentation

- **[SSOT-Ops.md](docs/SSOT-Ops.md)** - Operational specifications
- **[SSOT-Tech.md](docs/SSOT-Tech.md)** - Technical implementation
- **[Phased Plan](docs/granular-phased-plan.md)** - Complete 19-phase roadmap
- **[Quick Reference](quick_reference_guide.md)** - Common commands (UPDATED)
- **[Status Report](project_status_report.md)** - Current progress (UPDATED)

## 🎯 Next Steps

### Immediate (Day 16 - Tomorrow)
1. **Implement DataScheduler class**
   - Automated API calls every 30 seconds
   - Market hours awareness
   - Priority queue for symbols

2. **Test scheduler with cache**
   - Verify cache hit rate increases
   - Monitor API call reduction

### This Weekend (Days 16-17)
- [ ] Complete Phase 4 (Scheduler)
- [ ] Test 24-hour unattended operation
- [ ] Document scheduler configuration
- [ ] Prepare for Phase 5 (Indicators)

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Development Day** | 15 of 106 |
| **Progress** | 14.2% complete |
| **Phase 4 Progress** | 33% (Day 1 of 3) |
| **Lines of Code** | ~1,500 |
| **Database Tables** | 8 |
| **APIs Integrated** | 3 (2 AV + 1 IBKR) |
| **Cache Performance** | 30.6x faster |
| **Test Scripts** | 15 |
| **Status** | ✅ On Schedule |

## 🏆 Achievements

### Phase 4.1 Specific
- ✅ Redis cache layer operational
- ✅ 30x performance improvement achieved
- ✅ 50% API call reduction with just 2 symbols
- ✅ Cache-aware ingestion pipeline
- ✅ Intelligent TTL management

### Overall Project
- ✅ Clean architecture maintained throughout
- ✅ Zero hardcoded values
- ✅ Rate limiting never exceeded
- ✅ IBKR integration successful
- ✅ Cache layer seamlessly integrated
- ✅ Ready for automated scheduling

## 📮 Contact

For questions or issues, create an issue in the repository.

---

**Current Phase:** 4.1 Complete - Cache Manager ✅  
**Next Phase:** 4.2 - Scheduler (Starting Day 16)  
**First Paper Trade:** Day 40 (25 days away)  
**Production Launch:** Day 107 (92 days away)

*Last Updated: August 16, 2025, 3:00 PM ET - Phase 4.1 (Cache) Complete*