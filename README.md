# AlphaTrader - Automated Options Trading System

[![Phase](https://img.shields.io/badge/Phase-2%20Data%20Integration-blue)]()
[![APIs](https://img.shields.io/badge/APIs-53%2F53%20Working-success)]()
[![Components](https://img.shields.io/badge/Components-3%2F12%20Complete-yellow)]()
[![Database](https://img.shields.io/badge/Database-Production%20Ready-green)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue)]()

Production-grade automated options trading system with real money capabilities.

## 🚀 Current Status: Phase 2 - Data Integration

### ✅ Latest Achievements (August 15, 2025)

- **Phase 1 COMPLETE**: All connection components working
- **53/53 APIs Working**: Alpha Vantage (38) + IBKR (15 data feeds)
- **Rate Limiting**: Sophisticated token bucket with priority queues
- **Real-Time Data**: 1-minute bars from IBKR confirmed working
- **Configuration-Driven**: Zero hardcoded values

## 🏗️ System Architecture

```
WORKING COMPONENTS (Phase 1 Complete):
├── 🔧 ConfigManager           ✅ Loads all YAML configurations
├── 🚦 TokenBucketRateLimiter  ✅ 600 calls/min with priority queues
├── 📊 AlphaVantageClient       ✅ 38 APIs fully tested
└── 📈 IBKRConnectionManager    ✅ Real-time bars & quotes

IN PROGRESS (Phase 2):
├── ⏱️ DataScheduler            🔄 Tier-based API orchestration
├── 🔄 DataIngestionPipeline    🔄 Normalize & store data
└── 💾 CacheManager             📋 Redis integration

PLANNED (Phases 3-9):
├── 📊 AnalyticsEngine          📋 Week 3
├── 🤖 MLPipeline               📋 Week 4
├── 🎯 DecisionEngine           📋 Week 5
├── ⚠️ RiskManager              📋 Week 5
├── 🏃 ExecutionEngine          📋 Week 6
└── 📡 MonitoringSystem         📋 Week 7
```

## 📊 Data Architecture

### Real-Time Data Sources

| Source | Data Type | Status | Latency |
|--------|-----------|--------|---------|
| **IBKR** | 1-min bars | ✅ Working | < 100ms |
| **IBKR** | 5-sec bars | ✅ Working | < 100ms |
| **IBKR** | Real-time quotes | ✅ Working | < 50ms |
| **IBKR** | MOC Imbalance | ✅ Configured | 3:40-3:55 PM |
| **Alpha Vantage** | Options Greeks | ✅ Working | < 500ms |
| **Alpha Vantage** | Technical Indicators | ✅ Working | < 500ms |
| **Alpha Vantage** | Analytics | ✅ Working | < 500ms |
| **Alpha Vantage** | Fundamentals | ✅ Working | < 1s |

### Database Schema (21 Tables)

```sql
-- Core Trading Tables
options_chain           -- Full Greeks (Δ, Γ, Θ, Vega, Rho)
intraday_bars          -- IBKR real-time bars
technical_indicators   -- RSI, MACD, Bollinger, etc.

-- Fundamentals
company_overview       -- Company profiles
balance_sheet         -- Quarterly/Annual
income_statement      -- Revenue & earnings
cash_flow            -- Cash flow statements

-- Market Data
earnings             -- Historical earnings
dividends           -- Dividend history
stock_splits        -- Split history
market_movers       -- Top gainers/losers

-- Analytics
analytics_fixed_window    -- Fixed period analysis
analytics_sliding_window  -- Rolling analysis
news_sentiment           -- News analysis
economic_indicators      -- CPI, GDP, etc.
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 16
- Redis
- IBKR TWS or Gateway
- Alpha Vantage Premium API (600 calls/min)

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/AlphaTrader.git
cd AlphaTrader

# 2. Setup environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your credentials:
# - AV_API_KEY
# - IBKR_USERNAME
# - IBKR_PASSWORD

# 4. Database setup (already complete)
psql -U postgres -d trading_system -f schema.sql

# 5. Test connections
python test_av_client_live.py      # Test Alpha Vantage
python test_ibkr_connection.py     # Test IBKR

# 6. Start system (when ready)
python main.py --mode paper        # Paper trading mode
```

## 📈 Trading Strategies

### Implemented Strategies

| Strategy | Confidence | Entry Window | Hold Period | Status |
|----------|------------|--------------|-------------|--------|
| **0DTE** | 75% | 09:45-14:00 | Intraday | 📋 Pending |
| **1DTE** | 70% | 09:45-15:00 | 1 day | 📋 Pending |
| **14-Day Swing** | 65% | Market hours | 1-14 days | 📋 Pending |
| **MOC Imbalance** | 80% | 15:40-15:55 | Minutes | 📋 Pending |

## 🎯 Development Roadmap

### ✅ Completed (Phases 0-1)
- Infrastructure setup
- Configuration management
- Database schema
- API discovery & testing
- Connection layer implementation
- Rate limiting system
- Real-time data feeds

### 🔄 In Progress (Phase 2)
- Data scheduling system
- Ingestion pipelines
- Cache layer
- Data validation

### 📋 Upcoming (Phases 3-9)
- **Week 3**: Analytics engine
- **Week 4**: ML integration
- **Week 5**: Decision & risk engines
- **Week 6**: Paper trading
- **Week 7-8**: Performance validation
- **Week 9**: Production deployment

## 📊 Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| API Success Rate | 100% | > 99.9% |
| Data Latency | < 500ms | < 1s |
| Rate Limit Usage | 495/600 | < 500/600 |
| System Uptime | 100% | > 99.9% |
| Memory Usage | < 500MB | < 2GB |
| Components Complete | 3/12 | 12/12 |

## 🔒 Risk Management

### Position Limits (Configured)
- Max position delta: 100
- Max portfolio delta: 500
- Max position size: $50,000
- Max daily loss: 2%

### Circuit Breakers (Configured)
- Daily loss > 2%: Halt new trades
- Weekly loss > 5%: Reduce position sizes
- Drawdown > 10%: Emergency shutdown
- 5 losses in 60min: Pause 30min

## 🛠️ Testing

```bash
# Run all tests
pytest tests/

# Test specific components
pytest tests/test_rate_limiter.py
pytest tests/test_av_client.py
pytest tests/test_ibkr_connection.py

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ -v
```

## 📝 Configuration

All configuration is externalized to YAML files:

```
config/
├── apis/               # API configurations
│   ├── alpha_vantage.yaml
│   ├── ibkr.yaml
│   └── rate_limits.yaml
├── data/              # Data management
│   ├── symbols.yaml   # Tier A/B/C symbols
│   ├── schedules.yaml # Polling schedules
│   └── validation.yaml
├── strategies/        # Strategy parameters
├── risk/             # Risk limits
├── monitoring/       # Alerts & monitoring
└── environments/     # Dev/Paper/Prod
```

## 🚨 Important Notes

- **Real Money**: System will trade real money in production mode
- **Paper First**: Always test in paper mode before production
- **Rate Limits**: Strictly enforced at 600 calls/minute
- **Market Hours**: Configured for US Eastern Time
- **MOC Window**: Special handling 3:40-3:55 PM ET

## 🤝 Contributing

This is a proprietary trading system. Contributions are not accepted.

## 📜 License

Proprietary - All Rights Reserved

## 📞 Support

For issues or questions, contact the development team.

---

**Current Status**: Phase 2 (Data Integration) 🔄  
**APIs Working**: 53/53 ✅  
**Database**: Production Ready ✅  
**Trading Mode**: Development 🔧  
**Schedule**: 5-7 weeks ahead 🚀