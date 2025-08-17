# AlphaTrader 🚀

An advanced automated trading system leveraging Alpha Vantage data, IBKR execution, and machine learning for intelligent options trading.

## 🎯 Project Overview

AlphaTrader is a configuration-driven, fully automated trading system that combines real-time market data, technical indicators, options analytics, and machine learning to execute sophisticated trading strategies. Built with a zero-hardcoding philosophy, the system adapts entirely through configuration files.

### Key Features
- **Real-time Options Trading** with Greeks validation
- **Technical Indicators** (RSI, MACD, BBANDS, VWAP, ATR, ADX)
- **Automated Scheduling** with market hours awareness
- **ML-Driven Decisions** using frozen model inference
- **Risk Management** with position and portfolio limits
- **Redis Cache Layer** achieving 100x+ performance
- **Educational Platform** for market analysis and content generation

## 🏗️ System Architecture

```
[Alpha Vantage] → [Rate Limiter] → [AV Client] → [Cache Layer]
                                                      ↓
[IBKR TWS] → [IBKR Connection Manager] → [Data Ingestion]
                                              ↓
                                    [PostgreSQL Database]
                                              ↓
                                    [Analytics Engine]
                                              ↓
                                    [ML Models] → [Decision Engine]
                                              ↓
                                    [Risk Manager] → [IBKR Executor]
                                              ↓
                                    [Trade Monitor] → [Discord/Dashboard]
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **PostgreSQL 14+**
- **Redis 7+**
- **IBKR TWS or IB Gateway**
- **Alpha Vantage API Key** (Premium recommended)

### Installation

#### Step 1: Clone & Environment Setup
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

#### Step 2: Database Setup
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
psql -U your_username -d trading_system_db -f scripts/create_vwap_table.sql
```

#### Step 3: Redis Setup
```bash
# Install Redis
brew install redis  # macOS
# or
sudo apt-get install redis  # Ubuntu

# Start Redis service
brew services start redis  # macOS
# or
sudo systemctl start redis  # Ubuntu

# Verify Redis is running
redis-cli ping  # Should return PONG
```

#### Step 4: Configuration
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

#### Step 5: TWS Configuration
1. Open TWS (or IB Gateway)
2. File → Global Configuration → API → Settings
3. Enable "Enable ActiveX and Socket Clients"
4. Enable "Allow connections from localhost only"
5. Socket port: 7497 (paper) or 7496 (live)
6. Click OK and restart TWS

#### Step 6: Verify Installation
```bash
# Test all phases
python scripts/test_phase0.py
python scripts/test_phase2_complete.py
python scripts/test_ibkr_connection.py
python scripts/test_scheduler.py
python scripts/test_rsi_complete.py
python scripts/test_macd_complete.py
python scripts/test_bbands_complete.py
python scripts/test_vwap_complete.py

# Test live data (run during market hours)
python scripts/test_ibkr_live_data.py
```

#### Step 7: Start the Scheduler
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
│   │   ├── av_client.py          # AV client with 4 indicators
│   │   ├── ibkr_connection.py    # TWS integration
│   │   └── rate_limiter.py       # 600/min protection
│   ├── data/
│   │   ├── ingestion.py          # Data processing (710 lines)
│   │   ├── scheduler.py          # 138 automated jobs (750 lines)
│   │   └── cache_manager.py      # Redis caching (100x+ perf)
│   └── analytics/                # Coming in Phase 6
│       ├── greeks_validator.py
│       └── analytics_engine.py
│
├── config/                        # Configuration files
│   ├── .env                      # Environment variables
│   ├── apis/
│   │   └── alpha_vantage.yaml    # API configurations
│   ├── data/
│   │   ├── symbols.yaml          # Symbol tiers
│   │   └── schedules.yaml        # Polling schedules
│   └── strategies/               # Strategy configs (Phase 7)
│
├── scripts/                       # Test and utility scripts
│   ├── test_*.py                 # 40+ test scripts
│   ├── create_*_table.sql        # Database schemas
│   └── run_scheduler.py          # Main scheduler
│
├── data/                          # Data storage
│   └── api_responses/            # Saved API responses
│       ├── realtime_options_*.json
│       ├── rsi_*.json
│       ├── macd_*.json
│       ├── bbands_*.json
│       └── vwap_*.json
│
├── docs/                          # Documentation
│   ├── SSOT-Ops.md              # Operational specification
│   ├── SSOT-Tech.md             # Technical specification
│   ├── granular-phased-plan.md  # 19-phase roadmap
│   ├── phase5_rsi_summary.md    # RSI implementation
│   ├── phase5_macd_summary.md   # MACD implementation
│   ├── phase5_bbands_summary.md # BBANDS implementation
│   └── phase5_vwap_summary.md   # VWAP implementation
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
| `av_vwap` | 4,246 | Volume Weighted Average Price | 60-600s |

### IBKR Tables (Ready for Monday)
| Table | Purpose | Update Rate |
|-------|---------|-------------|
| `ibkr_bars_5sec` | 5-second bars | Every 5s |
| `ibkr_bars_1min` | 1-minute bars | Every 60s |
| `ibkr_bars_5min` | 5-minute bars | Every 5min |
| `ibkr_quotes` | Bid/Ask/Last | Tick-by-tick |

## 🔧 Configuration

### Symbol Tiers (`config/data/symbols.yaml`)
```yaml
tiers:
  tier_a:  # High priority - 60s updates
    symbols: [SPY, QQQ, IWM, SPX]
  tier_b:  # Medium priority - 300s updates
    symbols: [AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA]
  tier_c:  # Low priority - 600s updates
    symbols: [AMD, INTC, NFLX, DIS, BA, GS, JPM, BAC, WMT, TGT, XOM, CVX]
```

### Cache Configuration (`config/cache/redis.yaml`)
```yaml
cache:
  default_ttl: 300          # 5 minutes default
  
ttls:
  realtime_options: 30      # 30 seconds
  historical_options: 3600  # 1 hour
  rsi: 60                   # 60 seconds
  macd: 60                  # 60 seconds
  bbands: 60                # 60 seconds
  vwap: 60                  # 60 seconds
  api_responses: 300        # 5 minutes
```

## 🚦 Development Status

### ✅ Completed Phases (Days 1-21)
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
| **5.4** | 21 | VWAP Indicator | ✅ Complete | 4,246 data points |

### 🚧 In Progress - Phase 5 Continuation
| Indicator | Day | Status | Implementation | Notes |
|-----------|-----|--------|---------------|-------|
| RSI | 18 | ✅ Complete | 83,239 records | 109.4x cache |
| MACD | 19 | ✅ Complete | 83,163 records | 110.2x cache |
| BBANDS | 20 | ✅ Complete | 16,863 records | 127.4x cache |
| VWAP | 21 | ✅ Complete | 4,246 records | 150x cache |
| ATR | 22 | 📋 Next | Ready to start | Volatility/daily |
| ADX | 23 | 📋 Planned | - | Trend strength |
| Integration | 24 | 📋 Planned | - | All testing |

### 📅 Upcoming Phases (Days 25-106)
| Phase | Days | Description | Key Deliverable |
|-------|------|-------------|-----------------|
| **6** | 25-28 | Analytics & Validation | Greeks validator |
| **7** | 29-35 | First Strategy (0DTE) | **Trading logic** |
| **8** | 36-39 | Risk Management | Position limits |
| **9** | 40-43 | Paper Trading | **First trades!** |
| **10** | 44-57 | All Indicators | 16 total indicators |
| **11** | 58-63 | All Strategies | 4 strategies |
| **12** | 64-67 | ML Integration | Frozen models |
| **13** | 68-74 | Sentiment & News | Market sentiment |
| **14** | 75-82 | Fundamentals | Company data |
| **15** | 83-89 | Output & Monitoring | Discord/Dashboard |
| **16** | 90-99 | Educational Platform | Content generation |
| **17** | 100-106 | Production Prep | Go-live ready |

## 📊 Key Metrics

| Metric | Value | Change |
|--------|-------|--------|
| **Development Day** | 21 of 106 | +1 |
| **Progress** | 19.8% complete | +0.9% |
| **Phase Status** | 5.4 of 19 complete | +0.1 |
| **Lines of Code** | ~4,000 | +250 |
| **Database Tables** | 12 | - |
| **Database Size** | 75MB | +10MB |
| **Scheduled Jobs** | 138 | +23 |
| **API Usage** | 154/min (30.8%) | +27/min |
| **Cache Performance** | 150x on VWAP | NEW |
| **Total Data Points** | 237,365 | +4,246 |
| **Test Scripts** | 40+ | +5 |
| **Status** | ✅ On Schedule | - |

## 🎯 Next Steps

### Immediate (Day 22 - Sunday)
1. **ATR Implementation**
   - Test with daily intervals (different from others)
   - Expect single value per timestamp
   - Consider longer cache TTL for daily data
   - Test with SPY first
   - NO HARDCODED VALUES!

### This Week (Phase 5 Completion)
- [ ] Day 22: ATR implementation (Volatility/daily)
- [ ] Day 23: ADX implementation (Trend strength)
- [ ] Day 24: Integration testing all 6 indicators
- [ ] Test IBKR connection over weekend
- [ ] Prepare for Monday market open with real data

### This Month (Through Phase 9)
- [ ] Complete all 16 technical indicators
- [ ] Implement Greeks validation
- [ ] Build first trading strategy (0DTE)
- [ ] Add risk management
- [ ] Begin paper trading (Day 40)

## 🏆 Achievements

### Phase 5.4 Specific (VWAP)
- ✅ VWAP implementation in ~2 hours
- ✅ 4,246 data points ingested (5min intervals)
- ✅ No time_period parameter (unique to VWAP)
- ✅ 150x cache performance (best yet)
- ✅ Timestamp format flexibility
- ✅ 23 new scheduled jobs
- ✅ Clean 8-step implementation process

### Phase 5 Progress (66.7% Complete)
- ✅ 4 of 6 indicators operational
- ✅ Momentum (RSI), Trend (MACD), Volatility (BBANDS), Volume (VWAP)
- ✅ 187,511 indicator data points total
- ✅ Zero hardcoded values maintained throughout
- ✅ Configuration-driven architecture proven

## 🔐 Configuration Philosophy

**Zero Hardcoding Verification:**
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

## 🧪 Testing

### Test Scripts by Phase
```bash
# Phase 0-2: Foundation
python scripts/test_phase0.py
python scripts/test_phase2_complete.py

# Phase 3: IBKR
python scripts/test_ibkr_connection.py
python scripts/test_ibkr_market_data.py
python scripts/test_ibkr_live_data.py     # Run during market hours

# Phase 4: Cache & Scheduler
python scripts/test_cache_manager.py
python scripts/test_cached_av_client.py
python scripts/test_cache_integration.py
python scripts/test_scheduler.py

# Phase 5.1: RSI (Momentum)
python scripts/test_rsi_api.py
python scripts/test_rsi_client.py
python scripts/test_rsi_pipeline.py
python scripts/test_rsi_scheduler.py
python scripts/test_rsi_complete.py

# Phase 5.2: MACD (Trend)
python scripts/test_macd_api.py
python scripts/test_macd_client.py
python scripts/test_macd_pipeline.py
python scripts/test_macd_scheduler.py
python scripts/test_macd_complete.py

# Phase 5.3: BBANDS (Volatility)
python scripts/test_bbands_api.py
python scripts/test_bbands_client.py
python scripts/test_bbands_pipeline.py
python scripts/test_bbands_scheduler.py
python scripts/test_bbands_complete.py

# Phase 5.4: VWAP (Volume)
python scripts/test_vwap_api.py
python scripts/test_vwap_client.py
python scripts/test_vwap_pipeline.py
python scripts/test_vwap_scheduler.py
python scripts/test_vwap_complete.py

# Data Analysis
python scripts/query_options_data.py
```

### Performance Benchmarks
| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| AV API Call (no cache) | < 2s | ~0.5s | ✅ |
| AV API Call (cached) | < 100ms | ~2ms | ✅ |
| RSI Fetch (cached) | < 100ms | 0.01s | ✅ |
| MACD Fetch (cached) | < 100ms | 0.01s | ✅ |
| BBANDS Fetch (cached) | < 100ms | 0.003s | ✅ |
| VWAP Fetch (cached) | < 100ms | 0.002s | ✅ |
| Cache Hit Rate | > 50% | 80%+ avg | ✅ |
| Scheduler Jobs | N/A | 138 | ✅ |
| Job Execution | < 2s | ~1s | ✅ |
| IBKR Connection | < 5s | 2s | ✅ |

## 📚 Documentation

- **[SSOT-Ops.md](docs/SSOT-Ops.md)** - Operational specifications
- **[SSOT-Tech.md](docs/SSOT-Tech.md)** - Technical implementation
- **[Phased Plan](docs/granular-phased-plan.md)** - Complete 19-phase roadmap
- **[RSI Implementation](docs/phase5_rsi_summary.md)** - RSI details
- **[MACD Implementation](docs/phase5_macd_summary.md)** - MACD details
- **[BBANDS Implementation](docs/phase5_bbands_summary.md)** - BBANDS details
- **[VWAP Implementation](docs/phase5_vwap_summary.md)** - VWAP details
- **[Quick Reference](quick_reference_guide.md)** - Common commands
- **[Status Report](project_status_report.md)** - Current progress
- **[Educational Plan](docs/educational-content-plan.md)** - Content strategy

## 🔍 SQL Queries

### Useful Analysis Queries
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
    UNION ALL
    SELECT 'VWAP', symbol, COUNT(*), MAX(updated_at),
           EXTRACT(EPOCH FROM (NOW() - MAX(updated_at)))/60
    FROM av_vwap GROUP BY symbol
) t
ORDER BY indicator, symbol;

-- Find signal convergence
WITH combined_signals AS (
    SELECT 
        r.symbol,
        r.timestamp,
        r.rsi,
        m.macd_hist,
        (b.upper_band - b.lower_band) as bandwidth,
        v.vwap,
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
    LEFT JOIN av_vwap v ON r.symbol = v.symbol
        AND DATE_TRUNC('minute', r.timestamp) = DATE_TRUNC('minute', v.timestamp)
    WHERE r.timestamp > NOW() - INTERVAL '1 hour'
)
SELECT * FROM combined_signals
WHERE signal != 'NEUTRAL'
ORDER BY timestamp DESC;
```

## 📮 Contact

For questions or issues, create an issue in the repository.

---

**Current Phase:** 5.4 Complete - VWAP Operational ✅  
**Next Phase:** 5.5 - ATR Implementation (Day 22)  
**First Paper Trade:** Day 40 (19 days away)  
**Production Launch:** Day 107 (86 days away)

*Last Updated: August 17, 2025, 3:00 PM ET - VWAP with 4,246 data points, 150x cache performance, 4 of 6 indicators complete*