# AlphaTrader 🏛️

> **Institutional-Grade Automated Options Trading System**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/postgresql-14+-336791.svg)
![Redis](https://img.shields.io/badge/redis-7+-DC382D.svg)
![Status](https://img.shields.io/badge/status-development-yellow.svg)
![License](https://img.shields.io/badge/license-proprietary-red.svg)

## 🎯 Overview

AlphaTrader is an institutional-grade automated options trading system designed for real money trading. Built with zero compromises on quality, every line of code is production-ready from day one.

### Core Principles
- **Zero Hardcoding**: 100% configuration-driven architecture
- **Institutional Grade**: Enterprise patterns from line 1
- **Real Testing**: No mocks, test against actual systems
- **Production Ready**: Monitoring, logging, and metrics from start

### Key Features
- **36+ Alpha Vantage APIs** with comprehensive schema analysis
- **IBKR Integration** for real-time 5-second bars and execution
- **Institutional Analytics**: VPIN, GEX, Microstructure analysis
- **ML-Driven Decisions**: XGBoost, LSTM, GRU models with 200+ features
- **Four Trading Strategies**: 0DTE, 1DTE, 14-day swing, MOC imbalance
- **Risk Management**: VaR/CVaR, circuit breakers, position limits
- **Educational Platform**: 10+ daily content pieces for community

---

## 🏗️ Architecture

### System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources Layer                       │
├─────────────────────────┬───────────────────────────────────┤
│      IBKR (5-sec)       │      Alpha Vantage (36 APIs)      │
│  • Real-time bars       │  • Options & Greeks               │
│  • Order execution      │  • Technical indicators (16)      │
│  • MOC imbalance        │  • Fundamentals (8)              │
│                         │  • Sentiment & Economic (10)      │
└─────────────┬───────────┴───────────────┬───────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                     │
│  • Bar Aggregation (5s → 1m, 5m, 10m, 15m, 30m, 1h)       │
│  • Rate Limiting (<600/min)                                 │
│  • Data Validation & Caching                                │
│  • Schema-Driven Type Safety (8,227 fields mapped)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Analytics Layer                           │
│  • VPIN (Toxicity Detection)                                │
│  • GEX (Gamma Exposure)                                     │
│  • Microstructure Metrics                                   │
│  • 200+ ML Features                                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Decision Layer                            │
│  • ML Models (XGBoost, LSTM, GRU)                          │
│  • Strategy Rules (0DTE, 1DTE, Swing, MOC)                 │
│  • Risk Management (VaR/CVaR)                              │
│  • Circuit Breakers                                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Execution Layer                           │
│  • IBKR Order Management                                    │
│  • Position Monitoring                                      │
│  • P&L Tracking                                            │
│  • Publishing (Discord, Dashboard)                          │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Language**: Python 3.11+
- **Database**: PostgreSQL 14+ (time-series optimized)
- **Cache**: Redis 7+ (sub-millisecond latency)
- **ML Framework**: XGBoost 2.0+, TensorFlow/Keras
- **Monitoring**: Prometheus + OpenTelemetry
- **Scheduling**: APScheduler
- **Testing**: Pytest (real system tests only)
- **Logging**: structlog (structured JSON)
- **Type Safety**: TypeScript schemas for all APIs

---

## 🚀 Getting Started

### Prerequisites
- macOS (optimized for MacBook Pro)
- Python 3.11 or higher
- PostgreSQL 14+
- Redis 7+
- IBKR TWS or Gateway
- Alpha Vantage Premium API key (600 calls/min)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/AlphaTrader.git
cd AlphaTrader
```

#### 2. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

#### 3. Configure Environment
```bash
# Copy template
cp .env.template .env

# Edit .env with your configuration
# CRITICAL: Set all 73 configuration values
nano .env
```

#### 4. Set Up PostgreSQL Database
```bash
# Ensure PostgreSQL is running
brew services start postgresql  # macOS with Homebrew

# Run setup script
python scripts/setup_database.py
# Enter postgres password when prompted
```

#### 5. Set Up Redis Cache
```bash
# Ensure Redis is running
brew services start redis  # macOS with Homebrew

# Test connection
redis-cli ping
# Should return: PONG
```

#### 6. Test Alpha Vantage APIs (Day 3)
```bash
# Interactive API testing
python test_av_apis_interactive.py
```

---

## 📊 Development Progress

### Current Status: Day 3 of 87 In Progress (3.45%)

| Phase | Days | Status | Progress | Description |
|-------|------|--------|----------|-------------|
| **Foundation** | 1-2 | ✅ Complete | ████████████████ 100% | Core system components |
| **Alpha Vantage** | 3-8 | 🔄 In Progress | ███░░░░░░░░░░░░░ 16.7% | 36 APIs schema analysis |
| **IBKR Integration** | 9-14 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | Real-time data & execution |
| **Integration** | 15-17 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | System optimization |
| **Analytics** | 18-24 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | VPIN, GEX, Microstructure |
| **ML Features** | 25-28 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | 200+ features engineering |
| **ML Models** | 29-35 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | XGBoost, LSTM, GRU |
| **Strategies** | 36-43 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | Trading strategies |
| **Risk Management** | 44-47 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | VaR, circuit breakers |
| **Execution** | 48-51 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | Order management |
| **Paper Trading** | 52-59 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | System validation |
| **Publishing** | 60-66 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | Discord, Dashboard |
| **Education** | 67-73 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | Content engine |
| **Testing** | 74-80 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | Integration testing |
| **Production** | 81-87 | ⏳ Pending | ░░░░░░░░░░░░░░░░ 0% | Go-live preparation |

### Day 3 Achievements 🎯
- **Deep Schema Analysis**: 8,227 unique fields discovered and cataloged
- **TypeScript Definitions**: Complete type safety for all 36 API endpoints
- **Automated Analysis**: Zero manual schema writing
- **Pattern Recognition**: Identified critical data patterns (numeric strings, date formats)
- **Production Ready**: Full type definitions and validation ready

### Recent Commits
- `b022ab8` - Starting day 3
- `5a8233c` - Foundation Phase 0 - Day 2: Core foundation components
- `5e8c9f4` - Documentation: Comprehensive README and Project Status Report
- `36e3fad` - Foundation Phase 0 - Day 1: Project structure and dependencies

---

## ⚙️ Configuration

All configuration is externalized with **ZERO hardcoding**:

### Configuration Statistics
- **73 configurable values** in `.env`
- **100% externalized** - no magic numbers
- **Environment-based** - development/production modes
- **Hot-reloadable** - changes without code modification

### Configuration Structure
```
config/
├── system/              # Core system settings
│   ├── database.yaml    # PostgreSQL configuration
│   ├── redis.yaml       # Cache settings
│   └── logging.yaml     # Logging configuration
├── apis/                # API configurations
│   ├── alpha_vantage.yaml
│   └── ibkr.yaml
├── data/                # Data management
│   ├── symbols.yaml     # Symbol tiers
│   └── schedules.yaml   # Polling schedules
├── strategies/          # Strategy parameters
│   ├── 0dte.yaml
│   ├── 1dte.yaml
│   ├── swing_14d.yaml
│   └── moc_imbalance.yaml
├── risk/                # Risk limits
│   ├── position_limits.yaml
│   └── circuit_breakers.yaml
└── ml/                  # ML model configs
    ├── models.yaml
    └── features.yaml
```

---

## 📁 Project Structure

```
AlphaTrader/
├── src/                    # Source code
│   ├── foundation/         # Core utilities ✅ Complete
│   │   ├── config_manager.py
│   │   ├── database.py
│   │   ├── cache.py
│   │   ├── logger.py
│   │   └── metrics.py
│   ├── types/              # Type definitions 🔄 Day 3
│   │   └── alpha_vantage_schemas.ts  # 8,227 fields typed
│   ├── connections/        # API clients (Days 3-14)
│   │   ├── av_client.py
│   │   └── ibkr_client.py
│   ├── data/              # Data management
│   │   ├── ingestion.py
│   │   └── validation.py
│   ├── analytics/         # Analytics engine (Days 18-24)
│   │   ├── vpin.py
│   │   ├── gex.py
│   │   └── microstructure.py
│   ├── ml/                # Machine learning (Days 25-35)
│   │   ├── features.py
│   │   └── models.py
│   ├── strategies/        # Trading strategies (Days 36-43)
│   ├── risk/              # Risk management (Days 44-47)
│   ├── execution/         # Order execution (Days 48-51)
│   └── monitoring/        # System monitoring
├── config/                # Configuration files
├── scripts/               # Utility scripts
│   ├── setup_database.py # Database setup ✅
│   └── test_*.py         # Test scripts
├── migrations/            # Database migrations
├── tests/                 # Test suite
│   └── integration/       # Real system tests
├── logs/                  # Application logs
├── docs/                  # Documentation
└── data/                  # Data storage
    ├── api_responses/     # Raw API data (36 endpoints) ✅
    ├── api_schemas.json   # Schema analysis ✅
    ├── deep_api_schemas.json # Deep analysis ✅
    ├── field_statistics.json # Field stats ✅
    ├── cache/            # Local cache
    └── models/           # Trained models
```

---

## 🧪 Testing Philosophy

### REAL TESTS ONLY - No Mocks, No Stubs

Our testing approach:
- ✅ Test against **real PostgreSQL** database
- ✅ Test against **real Redis** cache
- ✅ Test with **actual API calls** (rate-limited)
- ✅ Test with **real market data**
- ✅ Test **failure scenarios** with actual failures
- ❌ No perfect world assumptions
- ❌ No mocked responses
- ❌ No stubbed services

Example test:
```python
def test_real_database_connection():
    """Test REAL PostgreSQL operations"""
    db = DatabaseManager(config)
    
    # This actually connects to PostgreSQL
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
```

---

## 📈 Performance Targets

### System Performance Requirements

| Metric | Target | Maximum | Test Method |
|--------|--------|---------|-------------|
| **5-sec bar latency** | <100ms | 200ms | Time from IBKR to storage |
| **Decision latency** | <1s | 2s | Data to decision time |
| **API calls/minute** | <500 | 600 | Alpha Vantage rate limit |
| **Database query** | <100ms | 500ms | Complex query benchmark |
| **Cache retrieval** | <10ms | 50ms | Redis GET operation |
| **Bar aggregation** | <50ms | 100ms | 5-sec to 1-min conversion |
| **VPIN calculation** | <200ms | 500ms | Per symbol calculation |
| **ML prediction** | <200ms | 500ms | Model inference time |

### Trading Performance Targets

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| **Win Rate** | 45% | 55% | 65% |
| **Profit Factor** | 1.2 | 1.5 | 2.0 |
| **Sharpe Ratio** | 1.0 | 1.5 | 2.0 |
| **Max Drawdown** | 15% | 10% | 5% |
| **Daily VaR (95%)** | 3% | 2% | 1% |

---

## 🛡️ Risk Management

### Position-Level Controls
```yaml
Max Delta: 0.80          # Maximum delta per position
Max Gamma: 0.20          # Maximum gamma exposure
Max Vega: 200            # Maximum vega per position
Min Theta Ratio: 0.02    # Minimum theta/price ratio
Max Position VaR: 5%     # Maximum VaR per position
```

### Portfolio-Level Controls
```yaml
Max Net Delta: 0.30      # Portfolio delta limit
Max Net Gamma: 0.75      # Portfolio gamma limit
Max Capital at Risk: 20% # Maximum capital deployed
Portfolio VaR (95%): 10% # Portfolio Value at Risk
Portfolio CVaR (95%): 15% # Conditional VaR
```

### Circuit Breakers
- **Daily Loss > 2%**: Halt all trading
- **VPIN > 0.7**: Emergency close all positions
- **Model Confidence < 0.3**: Stop new trades
- **5 Consecutive Losses**: Review required
- **Network Issues**: Graceful degradation

---

## 📚 Alpha Vantage API Coverage

### Complete Schema Analysis (Day 3)
Total APIs Analyzed: **36 endpoints**
Total Fields Mapped: **8,227 unique fields**

#### Fundamentals (8 APIs)
- Company Overview
- Balance Sheet (Annual/Quarterly)
- Income Statement (Annual/Quarterly)
- Cash Flow (Annual/Quarterly)
- Earnings
- Dividends
- Stock Splits
- Earnings Calendar

#### Technical Indicators (16 APIs)
- RSI, MACD, SMA, EMA
- Bollinger Bands, Stochastic
- ADX, ATR, CCI, Aroon
- MFI, OBV, AD, VWAP
- Momentum, Williams %R

#### Options & Greeks (2 APIs)
- Realtime Options Chain
- Historical Options Data

#### Economic Indicators (5 APIs)
- CPI (Consumer Price Index)
- Federal Funds Rate
- Inflation Rate
- Real GDP
- Treasury Yield

#### Sentiment & Market Data (3 APIs)
- News Sentiment Analysis
- Insider Transactions
- Top Gainers/Losers

#### Analytics (2 APIs)
- Fixed Window Analytics
- Sliding Window Analytics

---

## 📊 Monitoring & Observability

### Metrics (Prometheus)
- **Port**: 9090
- **Namespace**: `alphatrader`
- **Key Metrics**:
  - API call rates (36 endpoints tracked)
  - Database query performance
  - Cache hit/miss rates
  - Trading performance
  - System resources

### Health Checks
- **Port**: 8080
- **Endpoint**: `/health`
- **Checks**:
  - Database connectivity
  - Redis availability
  - API rate limits
  - System resources

### Logging (structlog)
- **Format**: Structured JSON
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Features**:
  - Correlation IDs
  - Performance tracking
  - Audit trail
  - Error aggregation

---

## 🚦 Production Readiness Checklist

### Pre-Production Requirements (Day 87)

#### Data Infrastructure
- [ ] All 36 Alpha Vantage APIs operational
- [ ] IBKR 5-second bars streaming reliably
- [ ] Bar aggregation accurate to all timeframes
- [ ] Complete database schema implemented
- [ ] Rate limiting never exceeded in testing

#### Analytics & ML
- [ ] VPIN calculations validated
- [ ] GEX analysis operational
- [ ] 200+ features generated per symbol
- [ ] ML models achieving >55% accuracy
- [ ] Walk-forward backtesting complete

#### Trading System
- [ ] All 4 strategies implemented
- [ ] Risk limits enforcing correctly
- [ ] Circuit breakers tested
- [ ] 5+ consecutive profitable paper trading days
- [ ] Win rate consistently >45%

#### Operational
- [ ] Complete documentation
- [ ] Disaster recovery plan tested
- [ ] Team training complete
- [ ] Monitoring dashboards operational
- [ ] Educational content pipeline active

---

## 🤝 Contributing

This is a proprietary project. Contributors must:

1. **Follow Standards**
   - Institutional coding standards
   - Zero hardcoding policy
   - Production-ready from line 1

2. **Testing Requirements**
   - Write real tests (no mocks)
   - Test against actual services
   - Include failure scenarios

3. **Documentation**
   - Document all changes
   - Update configuration guides
   - Maintain audit trail

4. **Code Quality**
   - Type hints required
   - Docstrings mandatory
   - Performance benchmarks met

---

## 📞 Support & Troubleshooting

### Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Database connection failed | Check PostgreSQL is running: `brew services list` |
| Redis connection failed | Check Redis is running: `redis-cli ping` |
| Import errors | Ensure venv activated: `source venv/bin/activate` |
| Configuration errors | Verify `.env` has all 73 values set |
| Schema analysis errors | Check `data/deep_api_schemas.json` exists |

### Getting Help
1. Check [Documentation](#documentation)
2. Review [Project Status](PROJECT_STATUS.md)
3. Check schema analysis in `data/DEEP_SCHEMA_ANALYSIS.md`
4. Follow error correlation IDs in logs

---

## 📚 Documentation

### Core Documents
- [**Project Status**](PROJECT_STATUS.md) - Current progress report (Day 3)
- [**Implementation Plan**](docs/implementation_plan.md) - 87-day detailed roadmap
- [**API Schema Analysis**](data/DEEP_SCHEMA_ANALYSIS.md) - Complete API structure
- [**TypeScript Schemas**](src/types/alpha_vantage_schemas.ts) - Type definitions

### Quick Links
- [Setup Guide](#installation)
- [Configuration Guide](#configuration)
- [Testing Guide](#testing-philosophy)
- [Risk Controls](#risk-management)
- [API Coverage](#alpha-vantage-api-coverage)

---

## ⚖️ License

**Proprietary - All Rights Reserved**

This software is proprietary and confidential. Unauthorized copying, distribution, or use of this software, via any medium, is strictly prohibited.

---

## 🙏 Acknowledgments

Built with institutional-grade components:
- PostgreSQL for reliable data persistence
- Redis for high-performance caching
- Prometheus for comprehensive monitoring
- Alpha Vantage for market data (36 APIs analyzed)
- Interactive Brokers for execution

---

<div align="center">

**Built with 🎯 precision for institutional-grade trading**

*AlphaTrader - Where Quality Meets Performance*

**Day 3 Progress: 8,227 Fields Analyzed | 36 APIs Mapped | Type Safety Achieved**

[Project Status](PROJECT_STATUS.md) | [Schema Analysis](data/DEEP_SCHEMA_ANALYSIS.md) | [Documentation](#documentation)

</div>