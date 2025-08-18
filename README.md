# AlphaTrader 🚀

An advanced automated trading system leveraging Alpha Vantage data, IBKR execution, and machine learning for intelligent options trading.

## 🎯 Project Overview

AlphaTrader is a configuration-driven, fully automated trading system that combines real-time market data, technical indicators, options analytics, and machine learning to execute sophisticated trading strategies. Built with a zero-hardcoding philosophy, the system adapts entirely through configuration files.

### Key Features
- **Real-time Options Trading** with Greeks validation
- **6 Technical Indicators** (RSI, MACD, BBANDS, VWAP, ATR, ADX) ✅ ALL COMPLETE
- **Automated Scheduling** with market hours awareness (184 jobs)
- **ML-Driven Decisions** using frozen model inference
- **Risk Management** with position and portfolio limits
- **Redis Cache Layer** achieving 100x+ performance
- **Educational Platform** for market analysis and content generation

## 🗏 System Architecture

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
psql -U your_username -d trading_system_db -f scripts/create_atr_table.sql
psql -U your_username -d trading_system_db -f scripts/create_adx_table.sql
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
python scripts/test_atr_daily_volatility.py
python scripts/test_adx_complete.py  # NEW!

# Run comprehensive integration test
python scripts/run_scheduler.py  # Should show 31/31 tests passing

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
│   │   ├── av_client.py          # AV client with 6 indicators (570 lines)
│   │   ├── ibkr_connection.py    # TWS integration (230 lines)
│   │   └── rate_limiter.py       # 600/min protection (115 lines)
│   ├── data/
│   │   ├── ingestion.py          # Data processing (950 lines)
│   │   ├── scheduler.py          # 184 automated jobs (1100 lines)
│   │   └── cache_manager.py      # Redis caching (125 lines)
│   └── analytics/                # Coming in Phase 6
│       ├── greeks_validator.py   # Next implementation
│       └── analytics_engine.py
│
├── config/                        # Configuration files
│   ├── .env                      # Environment variables
│   ├── apis/
│   │   └── alpha_vantage.yaml    # API configurations (6 indicators)
│   ├── data/
│   │   ├── symbols.yaml          # Symbol tiers
│   │   └── schedules.yaml        # Polling schedules (3 groups + daily)
│   └── strategies/               # Strategy configs (Phase 7)
│
├── scripts/                       # Test and utility scripts
│   ├── test_*.py                 # 50+ test scripts
│   ├── create_*_table.sql        # Database schemas
│   └── run_scheduler.py          # Main scheduler
│
├── data/                          # Data storage
│   └── api_responses/            # Saved API responses
│       ├── realtime_options_*.json
│       ├── rsi_*.json
│       ├── macd_*.json
│       ├── bbands_*.json
│       ├── vwap_*.json
│       ├── atr_*.json
│       └── adx_*.json           # NEW!
│
├── docs/                          # Documentation
│   ├── SSOT-Ops.md              # Operational specification
│   ├── SSOT-Tech.md             # Technical specification
│   ├── granular-phased-plan.md  # 19-phase roadmap
│   ├── phase5_rsi_summary.md    # RSI implementation
│   ├── phase5_macd_summary.md   # MACD implementation
│   ├── phase5_bbands_summary.md # BBANDS implementation
│   ├── phase5_vwap_summary.md   # VWAP implementation
│   ├── phase5_atr_summary.md    # ATR implementation
│   └── phase5_adx_summary.md    # ADX implementation (NEW!)
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 💾 Database Schema

### Alpha Vantage Tables - ALL COMPLETE ✅
| Table | Records | Description | Update Frequency |
|-------|---------|-------------|------------------|
| `av_realtime_options` | 49,854+ | Live options chains with Greeks | 30-180s |
| `av_historical_options` | 49,854+ | Historical snapshots | Daily 6 AM |
| `av_rsi` | 83,239 | RSI momentum indicator | 60-600s |
| `av_macd` | 83,163 | MACD trend indicator (3 values) | 60-600s |
| `av_bbands` | 16,863 | Bollinger Bands volatility (3 bands) | 60-600s |
| `av_vwap` | 4,246 | Volume Weighted Average Price | 60-600s |
| `av_atr` | 6,473 | Average True Range (daily volatility) | Daily 16:30 ET |
| `av_adx` | 4,219 | Average Directional Index (trend strength) | 900-3600s |

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
    symbols: [SPY, QQQ, IWM, IBIT]
  tier_b:  # Medium priority - 300s updates
    symbols: [AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA]
  tier_c:  # Low priority - 600s updates
    symbols: [DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW]
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
  atr: 300                  # 5 minutes (daily data)
  adx: 300                  # 5 minutes (trend strength)
  api_responses: 300        # 5 minutes
```

### Scheduler Groups (`config/data/schedules.yaml`)
```yaml
api_groups:
  indicators_fast:           # RSI, MACD, BBANDS, VWAP
    apis: ["RSI","MACD","BBANDS","VWAP"]
    tier_a_interval: 60      # Every minute
    tier_b_interval: 300     # Every 5 minutes
    tier_c_interval: 600     # Every 10 minutes
    
  indicators_slow:           # ADX - Trend strength
    apis: ["ADX"]
    tier_a_interval: 900     # Every 15 minutes
    tier_b_interval: 1800    # Every 30 minutes
    tier_c_interval: 3600    # Every 60 minutes
    
  daily_volatility:          # ATR - Smart scheduling for daily data
    apis: ["ATR"]
    schedule_time: "16:30"   # Once daily, 30 min after market close
    calls_per_symbol: 0.04   # Minimal API usage
```

## 🚦 Development Status

### ✅ Completed Phases (Days 1-22)
| Phase | Days | Description | Status | Key Achievement |
|-------|------|-------------|--------|-----------------|
| **0** | 1-3 | Minimal Foundation | ✅ Complete | Zero hardcoding |
| **1** | 4-7 | First API Pipeline | ✅ Complete | 9,294 contracts |
| **2** | 8-10 | Rate Limiting & 2nd API | ✅ Complete | 600/min protection |
| **3** | 11-14 | IBKR Integration | ✅ Complete | Real-time ready |
| **4** | 15-17 | Scheduler & Cache | ✅ Complete | 46 jobs automated |
| **5** | 18-22 | Core Technical Indicators | ✅ COMPLETE | All 6 indicators |

### ✅ Phase 5 Status - 100% COMPLETE! 🎉
| Indicator | Day | Status | Records | Cache Perf | Scheduling | Notes |
|-----------|-----|--------|---------|------------|------------|-------|
| RSI | 18 | ✅ Complete | 83,239 | 109.4x | 60-600s intervals | Momentum |
| MACD | 19 | ✅ Complete | 83,163 | 110.2x | 60-600s intervals | Trend |
| BBANDS | 20 | ✅ Complete | 16,863 | 127.4x | 60-600s intervals | Volatility bands |
| VWAP | 21 | ✅ Complete | 4,246 | 150x | 60-600s intervals | Volume-weighted |
| ATR | 22 | ✅ Complete | 6,473 | Efficient | Daily at 16:30 ET | Daily volatility |
| **ADX** | **22** | **✅ Complete** | **4,219** | **112x** | **900-3600s intervals** | **Trend strength** |

### 📅 Upcoming Phases (Days 23-106)
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
| **Development Day** | 22 of 106 | Complete |
| **Progress** | 31.6% complete | +10.8% |
| **Phase Status** | 6 of 19 complete | +1 |
| **Lines of Code** | ~4,790 | +590 |
| **Database Tables** | 14 | +1 (av_adx) |
| **Database Size** | 85MB | +5MB |
| **Scheduled Jobs** | 184 | +46 (ADX added) |
| **API Usage** | 154/min (30.8%) | Optimized |
| **Cache Performance** | 68.7x-150x | Collision-free |
| **Total Data Points** | 248,057 | +10,412 |
| **Test Scripts** | 50+ | +5 |
| **Integration Tests** | 31/31 passing | Perfect |
| **Status** | ✅ AHEAD OF SCHEDULE | Phase 5 Complete! |

## 🎯 Next Steps

### Immediate (Day 23 - Monday) 🚨
1. **IBKR GOES LIVE** 
   - Real-time market data starts
   - Monitor connection stability
   - Validate quote quality
   - Track system performance

2. **Begin Phase 6**
   - Greeks Validator implementation
   - Analytics engine framework
   - ANALYTICS_FIXED_WINDOW
   - ANALYTICS_SLIDING_WINDOW

### This Week (Phase 6 Start)
- [x] Day 22: ATR implementation ✅ COMPLETE
- [x] Day 22: ADX implementation ✅ COMPLETE (Same day!)
- [ ] Day 23: IBKR LIVE + Begin Phase 6
- [ ] Day 24: Greeks Validator
- [ ] Day 25: Analytics APIs

### This Month (Through Phase 9)
- [ ] Complete analytics layer
- [ ] Implement Greeks validation
- [ ] Build first trading strategy (0DTE)
- [ ] Add risk management
- [ ] Begin paper trading (Day 40)

## 🏆 Achievements

### Phase 5 Complete - All Indicators Operational
- ✅ **RSI** - 83,239 momentum data points
- ✅ **MACD** - 83,163 trend signals with 3 components
- ✅ **BBANDS** - 16,863 volatility measurements (3 bands)
- ✅ **VWAP** - 4,246 volume-weighted price points
- ✅ **ATR** - 6,473 daily volatility records (26 years)
- ✅ **ADX** - 4,219 trend strength indicators (avg 38.48)

### System Excellence
- ✅ 248,057 total indicator data points
- ✅ Zero hardcoded values maintained throughout
- ✅ Configuration-driven architecture proven
- ✅ Cache collision bug fixed - data integrity guaranteed
- ✅ Smart scheduling implemented (3 groups + daily)
- ✅ 31/31 integration tests passing perfectly
- ✅ 184 scheduled jobs running efficiently

### ADX Specific Achievements
- ✅ Implementation completed on same day as ATR
- ✅ Shows strong trending market (avg 38.48)
- ✅ Max ADX of 95.37 (extremely rare trend)
- ✅ Properly scheduled in indicators_slow group
- ✅ Full pipeline working (client → ingestion → scheduler)

## 🔐 Configuration Philosophy

**Zero Hardcoding Verification:**
```bash
# Search for hardcoded defaults - SHOULD RETURN NOTHING!
grep -r "='SPY'" src/
grep -r "interval='1min'" src/
grep -r "interval='5min'" src/
grep -r "interval='daily'" src/
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

# Phase 5.5: ATR (Daily Volatility)
python scripts/test_atr_api.py
python scripts/test_atr_client.py
python scripts/test_atr_pipeline.py
python scripts/test_atr_scheduler.py
python scripts/test_atr_daily_volatility.py

# Phase 5.6: ADX (Trend Strength) - NEW!
python scripts/test_adx_api.py
python scripts/test_adx_client.py
python scripts/test_adx_pipeline.py
python scripts/test_adx_scheduler.py
python scripts/test_adx_complete.py

# Integration Testing - PERFECT SCORE!
python scripts/run_scheduler.py  # 31/31 tests passing

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
| ATR Fetch (cached) | < 100ms | 0.005s | ✅ |
| ADX Fetch (cached) | < 100ms | 0.004s | ✅ |
| Cache Hit Rate | > 50% | 80%+ avg | ✅ |
| Cache Accuracy | 100% | 100% | ✅ Fixed |
| Scheduler Jobs | N/A | 184 | ✅ |
| Job Execution | < 2s | ~1s | ✅ |
| IBKR Connection | < 5s | 2s | ✅ |
| Integration Tests | 100% | 31/31 | ✅ PERFECT |

## 📚 Documentation

- **[SSOT-Ops.md](docs/SSOT-Ops.md)** - Operational specifications
- **[SSOT-Tech.md](docs/SSOT-Tech.md)** - Technical implementation
- **[Phased Plan](docs/granular-phased-plan.md)** - Complete 19-phase roadmap
- **[RSI Implementation](docs/phase5_rsi_summary.md)** - RSI details
- **[MACD Implementation](docs/phase5_macd_summary.md)** - MACD details
- **[BBANDS Implementation](docs/phase5_bbands_summary.md)** - BBANDS details
- **[VWAP Implementation](docs/phase5_vwap_summary.md)** - VWAP details
- **[ATR Implementation](docs/phase5_atr_summary.md)** - ATR daily volatility details
- **[ADX Implementation](docs/phase5_adx_summary.md)** - ADX trend strength (NEW!)
- **[Quick Reference](quick_reference_guide.md)** - Common commands
- **[Status Report](project_status_report.md)** - Current progress (UPDATED!)
- **[Educational Plan](docs/educational-content-plan.md)** - Content strategy

## 🔍 SQL Queries

### Useful Analysis Queries
```sql
-- Check all indicator data freshness (ALL 6 INDICATORS)
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
    UNION ALL
    SELECT 'ATR', symbol, COUNT(*), MAX(updated_at),
           EXTRACT(EPOCH FROM (NOW() - MAX(updated_at)))/60
    FROM av_atr GROUP BY symbol
    UNION ALL
    SELECT 'ADX', symbol, COUNT(*), MAX(updated_at),
           EXTRACT(EPOCH FROM (NOW() - MAX(updated_at)))/60
    FROM av_adx GROUP BY symbol
) t
ORDER BY indicator, symbol;

-- Find signal convergence with ATR for position sizing and ADX for trend
WITH combined_signals AS (
    SELECT 
        r.symbol,
        r.timestamp,
        r.rsi,
        m.macd_hist,
        (b.upper_band - b.lower_band) as bandwidth,
        v.vwap,
        a.atr,
        x.adx,
        CASE 
            WHEN r.rsi < 30 AND m.macd_hist > 0 THEN 'STRONG_BUY'
            WHEN r.rsi > 70 AND m.macd_hist < 0 THEN 'STRONG_SELL'
            WHEN r.rsi < 30 THEN 'BUY_SIGNAL'
            WHEN r.rsi > 70 THEN 'SELL_SIGNAL'
            ELSE 'NEUTRAL'
        END as signal,
        -- Position size based on ATR
        CASE 
            WHEN a.atr < 2 THEN 'FULL_SIZE'
            WHEN a.atr BETWEEN 2 AND 5 THEN 'HALF_SIZE'
            ELSE 'QUARTER_SIZE'
        END as position_size,
        -- Trend strength from ADX
        CASE
            WHEN x.adx < 25 THEN 'NO_TREND'
            WHEN x.adx BETWEEN 25 AND 50 THEN 'STRONG_TREND'
            WHEN x.adx BETWEEN 50 AND 75 THEN 'VERY_STRONG_TREND'
            ELSE 'EXTREME_TREND'
        END as trend_strength
    FROM av_rsi r
    JOIN av_macd m ON r.symbol = m.symbol 
        AND r.timestamp = m.timestamp
    LEFT JOIN av_bbands b ON r.symbol = b.symbol 
        AND DATE_TRUNC('minute', r.timestamp) = DATE_TRUNC('minute', b.timestamp)
    LEFT JOIN av_vwap v ON r.symbol = v.symbol
        AND DATE_TRUNC('minute', r.timestamp) = DATE_TRUNC('minute', v.timestamp)
    LEFT JOIN av_atr a ON r.symbol = a.symbol
        AND DATE(r.timestamp) = a.timestamp
    LEFT JOIN av_adx x ON r.symbol = x.symbol
        AND DATE_TRUNC('hour', r.timestamp) = DATE_TRUNC('hour', x.timestamp)
    WHERE r.timestamp > NOW() - INTERVAL '1 hour'
)
SELECT * FROM combined_signals
WHERE signal != 'NEUTRAL'
ORDER BY timestamp DESC;

-- ADX Analysis for Market Conditions
SELECT 
    symbol,
    AVG(adx) as avg_adx,
    MIN(adx) as min_adx,
    MAX(adx) as max_adx,
    COUNT(*) as data_points,
    CASE 
        WHEN AVG(adx) < 25 THEN 'Ranging Market'
        WHEN AVG(adx) BETWEEN 25 AND 50 THEN 'Trending Market'
        WHEN AVG(adx) BETWEEN 50 AND 75 THEN 'Strong Trending'
        ELSE 'Extreme Trending'
    END as market_condition
FROM av_adx
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY symbol
ORDER BY avg_adx DESC;
```

## 📮 Contact

For questions or issues, create an issue in the repository.

---

**Current Phase:** 6 Ready to Start - Analytics & Greeks ✅  
**Phase 5 Status:** COMPLETE - All 6 Indicators Operational 🎉  
**Next Milestone:** IBKR LIVE (Day 23) + Phase 6 Start 🚨  
**First Paper Trade:** Day 40 (18 days away)  
**Production Launch:** Day 107 (85 days away)

*Last Updated: August 17, 2025, 10:15 PM ET - Phase 5 COMPLETE with ADX operational, 248,057 total data points, 31/31 integration tests passing*