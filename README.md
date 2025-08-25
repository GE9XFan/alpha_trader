# 🚀 AlphaTrader v3.0

**An ML-Driven Algorithmic Options Trading System with Dual-Source Architecture**

![Status](https://img.shields.io/badge/Status-Week%201%20Day%203-blue)
![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-success)
![Greeks](https://img.shields.io/badge/Greeks-PROVIDED%20by%20AV-green)
![Python](https://img.shields.io/badge/Python-3.13.2-blue)

## 📊 Project Overview

AlphaTrader is a sophisticated algorithmic trading system that leverages machine learning and a dual-source data architecture to trade options with professional-grade analytics. The system uniquely combines:

- **Alpha Vantage Premium (600 calls/min)**: Options chains with pre-calculated Greeks, technical indicators, sentiment analysis
- **Interactive Brokers**: Real-time quotes, 5-second bars, order execution, position management

### 🎯 Core Philosophy: "Greeks PROVIDED, Not Calculated"

Unlike traditional systems that implement Black-Scholes calculations, AlphaTrader leverages Alpha Vantage's professional-grade Greeks calculations, eliminating computational complexity and ensuring consistency.

## 📈 Current Status: Week 1, Day 3

### ✅ What's Working
- **Alpha Vantage Client**: Fully functional with all 36 APIs
- **Greeks Retrieval**: Successfully receiving Delta, Gamma, Theta, Vega, Rho
- **Options Data**: Real-time and historical options chains
- **Technical Indicators**: All 16 indicators operational
- **Sentiment Analysis**: News sentiment, top gainers/losers, insider transactions
- **Rate Limiting**: Professional 600/min tier properly managed
- **Caching System**: In-memory cache with TTL management
- **Testing Suite**: 100% pass rate (25/25 tests)

### 🚧 In Progress
- IBKR market data connection
- ML model training pipeline
- Signal generation logic
- Risk management system

### 📝 Not Yet Started
- Paper trading implementation
- Live trading system
- Discord bot integration
- Performance monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   DATA LAYER (Dual Sources)                      │
├────────────────────────────────┬─────────────────────────────────┤
│   Alpha Vantage (600/min)      │   Interactive Brokers           │
│   ✅ Options with Greeks       │   🚧 Real-time quotes          │
│   ✅ Technical Indicators      │   🚧 5-second bars             │
│   ✅ Sentiment Analysis        │   🚧 Order execution           │
│   ✅ Historical Data (20yr)    │   🚧 Position management       │
└────────────────────────────────┴─────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYTICS LAYER                               │
│   📊 Feature Engineering | 🤖 ML Model (XGBoost)                │
│   🚧 45 Features Total  | 🚧 Training Pipeline                  │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                     TRADING LAYER                                │
│   📈 Signal Generation | ⚠️ Risk Management | 💼 Execution      │
│   🚧 Entry/Exit Rules | 🚧 Position Sizing  | 🚧 Order Routing │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+ (tested with 3.13.2)
- PostgreSQL 16+
- Redis 8.0+
- Alpha Vantage Premium API Key
- Interactive Brokers Account (paper or live)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AlphaTrader.git
cd AlphaTrader
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.template .env
# Edit .env and add your API keys:
# AV_API_KEY=your_alpha_vantage_key
```

5. **Set up databases**
```bash
# PostgreSQL (already configured if following setup)
createdb alphatrader

# Redis should be running via brew services (macOS)
brew services start redis
```

6. **Verify installation**
```bash
python verify_environment.py
```

## ⚙️ Configuration

### Alpha Vantage API Key
Obtain a premium API key from [Alpha Vantage](https://www.alphavantage.co/premium/) for 600 calls/minute access.

### Key Configuration Files
- `config/config.yaml` - Main configuration
- `.env` - API keys and secrets
- `config/logging.yaml` - Logging settings
- `config/alerts.yaml` - Alert thresholds

### Important Settings
```yaml
# config/config.yaml
data_sources:
  alpha_vantage:
    rate_limit: 600  # Premium tier
    cache_ttls:
      options: 60     # 1 minute for real-time
      historical: 3600 # 1 hour for historical
      indicators: 300  # 5 minutes
      sentiment: 900   # 15 minutes
```

## 🧪 Testing

### Run Full Test Suite
```bash
python scripts/test_av_client.py --full
```

### Quick Greeks Verification
```bash
python scripts/test_av_client.py --quick
```

### Test Specific API
```bash
python scripts/test_av_client.py --api get_realtime_options --symbol AAPL
```

### Current Test Results
```
✅ Passed: 25
❌ Failed: 0
📈 Success Rate: 100.0%
```

## 📊 API Status

| Category | APIs | Status | Notes |
|----------|------|--------|-------|
| **Options** | 2 | ✅ Working | REALTIME_OPTIONS, HISTORICAL_OPTIONS with Greeks |
| **Technical Indicators** | 16 | ✅ Working | RSI, MACD, STOCH, WILLR, MOM, BBANDS, ATR, ADX, AROON, CCI, EMA, SMA, MFI, OBV, AD, VWAP |
| **Analytics** | 2 | ✅ Working | ANALYTICS_FIXED_WINDOW, ANALYTICS_SLIDING_WINDOW |
| **Sentiment** | 3 | ✅ Working | NEWS_SENTIMENT (stocks only, not ETFs), TOP_GAINERS_LOSERS, INSIDER_TRANSACTIONS |
| **Fundamentals** | 8 | ✅ Working | OVERVIEW, EARNINGS, INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, DIVIDENDS, SPLITS, EARNINGS_CALENDAR |
| **Economic** | 5 | ✅ Working | TREASURY_YIELD, FEDERAL_FUNDS_RATE, CPI, INFLATION, REAL_GDP |

**Total: 36 APIs Operational**

## 🐛 Known Issues & Limitations

### Alpha Vantage Limitations
- **NEWS_SENTIMENT**: Does not support ETFs (SPY, QQQ, IWM). Use individual stocks (AAPL, MSFT, etc.)
- **Greeks**: Some deep ITM/OTM options may have zero Greeks (this is valid, not an error)
- **Historical Options**: Requires past dates; future dates return empty data

### Current Implementation Gaps
- IBKR connection not yet implemented
- ML model needs training data
- Paper trader is skeleton only
- Risk management rules not defined

## 📅 Development Roadmap

### Week 1: Foundation ✅ (Days 1-3 Complete)
- [x] Day 1-2: Project setup, configuration
- [x] Day 3: Alpha Vantage client implementation
- [ ] Day 4: IBKR connection
- [ ] Day 5: Options data manager

### Week 2: ML & Analytics 🚧
- [ ] Feature engineering (45 features)
- [ ] ML model training pipeline
- [ ] Backtesting framework
- [ ] Signal generation

### Week 3-4: Trading Logic 📝
- [ ] Risk management implementation
- [ ] Position sizing algorithms
- [ ] Entry/exit rules
- [ ] Order execution

### Week 5-6: Paper Trading 📝
- [ ] Paper trader implementation
- [ ] Performance tracking
- [ ] Real-time monitoring
- [ ] Alert system

### Week 7-8: Community Features 📝
- [ ] Discord bot
- [ ] Signal publishing
- [ ] Subscription tiers
- [ ] Performance dashboard

## 🚀 Quick Start Trading (Future)

Once fully implemented, trading will be as simple as:

```python
from src.trading.paper_trader import PaperTrader

trader = PaperTrader()
await trader.run()  # Starts paper trading with AV data + IBKR execution
```

## 📚 Documentation

- [Technical Specification v3](alphatrader-tech-spec-v3.md) - Complete system design
- [Implementation Plan](alphatrader-implementation-plan.md) - 16-week development roadmap
- [Operations Manual](alphatrader-ops-manual.md) - Production deployment guide
- [API Reference](av_api_reference.py) - Alpha Vantage API details

## 🤝 Contributing

This is currently a private project in active development. Contributions will be welcomed after the core system is operational (estimated Week 8).

## 📄 License

Proprietary - All rights reserved

## 🙏 Acknowledgments

- **Alpha Vantage** for providing professional-grade options Greeks
- **Interactive Brokers** for execution infrastructure
- **XGBoost** for ML capabilities

## 📞 Contact

For questions about this project, please open an issue in the repository.

---

**Current Focus**: Implementing IBKR connection (Week 1, Day 4)

**Next Milestone**: Complete paper trading system (Week 5-6)

**Ultimate Goal**: Fully automated ML-driven options trading with professional Greeks