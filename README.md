# AlphaTrader - Automated Options Trading System

[![Phase](https://img.shields.io/badge/Phase-2%20Data%20Integration-blue)]()
[![APIs](https://img.shields.io/badge/APIs-38%2F38%20Tested-success)]()
[![Database](https://img.shields.io/badge/Database-Production%20Ready-green)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue)]()

Production-grade automated options trading system with real money capabilities. 

## 🚀 Current Status: Phase 2 - Data Integration

### ✅ Completed Components

#### Phase 0: Infrastructure (100% COMPLETE)
- ✅ Complete project skeleton with 40+ modules
- ✅ Configuration management system (30+ YAML files)
- ✅ PostgreSQL database with 21 production tables
- ✅ Redis cache layer configured
- ✅ Base classes and module initialization
- ✅ Environment support (dev/paper/production)

#### Phase 0.5: API Discovery (100% COMPLETE)
- ✅ All 38 Alpha Vantage APIs tested and working
- ✅ IBKR real-time bars tested and verified
- ✅ Database schema aligned with API responses
- ✅ Options chain with full Greeks support
- ✅ Complete test suite (`test_api.py`)

#### Phase 1: Connections (30% COMPLETE)
- ✅ API test framework operational
- ✅ Database schemas production-ready
- 🔄 Rate limiter implementation needed
- 🔄 Full client implementations needed

### 🎯 Current Focus: Phase 2 - Data Ingestion

Implementing data ingestion pipelines for all tested APIs.

## 📊 System Architecture

### Database Schema (PRODUCTION READY)
- **21 Tables** covering all data types
- **Options Chain** with complete Greeks (Δ, Γ, Θ, Vega, Rho)
- **Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
- **Fundamentals** (Balance Sheet, Income Statement, Cash Flow)
- **Market Data** (Earnings, Dividends, Splits)
- **Analytics** (Fixed & Sliding Window)
- **Economic Indicators** (CPI, GDP, Treasury Yields)
- **Sentiment & News** analysis
- **System Health** monitoring

### API Integration Status
| API Provider | APIs | Status | Notes |
|-------------|------|--------|-------|
| Alpha Vantage | 38/38 | ✅ Tested | All working, rate limit: 600/min |
| IBKR | 4/4 | ✅ Tested | Bars, quotes, MOC, execution ready |

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11
- PostgreSQL 16 (database already created)
- Redis (configured)
- Alpha Vantage Premium API key (600 calls/min)
- IBKR account (paper or live)

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/yourusername/AlphaTrader.git
cd AlphaTrader

# 2. Setup environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Add your API keys to .env

# 4. Database is already set up with schema.sql

# 5. Test APIs (ALREADY COMPLETE)
python scripts/test_api.py  # ✅ All passing

# 6. Start data ingestion (CURRENT PHASE)
python scripts/ingest_data.py
```

## 📈 Development Roadmap

### ✅ Completed Phases
- **Phase 0**: Infrastructure Foundation (100%)
- **Phase 0.5**: API Discovery & Testing (100%)

### 🔄 In Progress
- **Phase 1**: Connection Layer Implementation (30%)
- **Phase 2**: Data Ingestion Pipelines (10%)

### 📋 Upcoming Phases
- **Phase 3**: Analytics Engine
- **Phase 4**: ML Integration
- **Phase 5**: Decision Engine
- **Phase 6**: Risk & Execution
- **Phase 7**: Output Layer
- **Phase 8**: Integration Testing
- **Phase 9**: Production Deployment

### Revised Timeline
- **Week 1-2**: Complete data ingestion (CURRENT)
- **Week 3**: Analytics and indicators
- **Week 4**: ML and decision engine
- **Week 5**: Risk management
- **Week 6**: Paper trading
- **Week 7-8**: Performance validation
- **Week 9**: Production deployment

## 🎯 Next Steps

1. **Implement Data Ingestion** (Priority)
   - Create ingestion methods for each API
   - Map responses to database tables
   - Implement upsert logic

2. **Activate Rate Limiting**
   - Implement TokenBucketRateLimiter
   - Configure for 600 calls/minute

3. **Build Scheduler**
   - Tier-based polling (A, B, C symbols)
   - MOC window handling
   - Optimize API usage

4. **Connect IBKR Live Data**
   - Real-time bar subscriptions
   - Quote feed integration
   - MOC imbalance monitoring

## 📊 Project Metrics

- **Database Tables**: 21/21 ✅
- **APIs Tested**: 38/38 ✅
- **Modules Created**: 40/40 ✅
- **Configuration Files**: 30+ ✅
- **Test Coverage**: Growing 📈

## 🔒 Trading Strategies

### Configured Strategies
1. **0DTE (Zero Days to Expiration)**
   - Min Confidence: 75%
   - Entry Window: 09:45 - 14:00 ET
   - Auto-close: 15:30 ET

2. **1DTE (One Day to Expiration)**
   - Min Confidence: 70%
   - Entry Window: 09:45 - 15:00 ET
   - Can hold overnight

3. **14-Day Swing**
   - Min Confidence: 65%
   - Hold Period: 1-14 days
   - Position rolling enabled

4. **MOC Imbalance**
   - Active Window: 15:40 - 15:55 ET
   - Min Imbalance: $10M
   - Uses straddles or directional

## 🔍 Data Sources

### IBKR (Interactive Brokers)
- ✅ Real-time pricing (1s, 5s, 1m, 5m bars)
- ✅ Real-time quotes
- ✅ MOC imbalance data
- ✅ Trade execution
- ✅ Position monitoring

### Alpha Vantage (38 APIs - ALL WORKING)
- ✅ Options chains with Greeks
- ✅ Technical indicators (16 types)
- ✅ Analytics (fixed/sliding window)
- ✅ Fundamentals data (11 endpoints)
- ✅ Economic indicators (5 types)
- ✅ News sentiment (3 endpoints)

## ⚠️ Important Notes

- System will trade **real money** in production
- All 38 Alpha Vantage APIs confirmed working
- Database schema is production-ready
- Paper trading required before production
- Rate limit: 600 calls/minute (confirmed)

## 📜 License

Proprietary - All Rights Reserved

---

**Current Phase**: 2 (Data Integration) 🔄  
**APIs Working**: 38/38 ✅  
**Database**: Production Ready ✅  
**Trading Mode**: Development