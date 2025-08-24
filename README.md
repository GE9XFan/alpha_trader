# AlphaTrader v3.0

ML-driven options trading system using Alpha Vantage for comprehensive options data and analytics, with Interactive Brokers for execution.

## 🚀 Key Features

- **Dual Data Sources**: Alpha Vantage (options/Greeks/analytics) + IBKR (execution)
- **Greeks PROVIDED**: Greeks come from Alpha Vantage - NO Black-Scholes calculations
- **38 Alpha Vantage APIs**: Complete options, technical, sentiment, and fundamental data
- **ML-Driven Signals**: XGBoost trained on 20 years of AV historical options data
- **Risk Management**: Portfolio Greeks limits using AV real-time data
- **Paper & Live Trading**: Seamless transition from paper to live
- **Discord Community**: Tiered signal publishing with Greeks

## 📊 Data Architecture

### Alpha Vantage (600 calls/min premium)
- Real-time options chains WITH Greeks ✅
- 20 years historical options WITH Greeks ✅
- 16 technical indicators (RSI, MACD, etc.)
- News sentiment & analytics
- Fundamental data
- Economic indicators

### Interactive Brokers
- Real-time quotes & 5-second bars
- Order execution (paper & live)
- Position management

## 🏗️ System Architecture

Based on Tech Spec v3.0, Implementation Plan v2.0, and Operations Manual v2.0.

```
Data Layer (AV + IBKR)
    ↓
Analytics Layer (ML + Features)
    ↓
Trading Layer (Signals + Risk)
    ↓
Execution Layer (Paper + Live)
    ↓
Community Layer (Discord)
```

## 🚦 Quick Start

1. **Setup Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure**
```bash
cp config/config.template.yaml config/config.yaml
# Edit config.yaml with your API keys
export AV_API_KEY="your_alpha_vantage_key"
```

3. **Initialize Database**
```bash
psql -U postgres
CREATE DATABASE alphatrader;
\q
```

4. **Run Health Checks**
```bash
python scripts/health/morning_checks.py
```

5. **Start Trading**
```bash
./scripts/startup/start_all.sh
```

## 📈 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Greeks Retrieval | <5ms cached | From Alpha Vantage |
| API Efficiency | <5 calls/trade | Through caching |
| Signal Latency | <150ms | End-to-end |
| Cache Hit Rate | >80% | Multi-tier cache |

## 🔑 Critical Understanding

**Greeks are PROVIDED by Alpha Vantage, NEVER calculated locally!**

This eliminates the need for:
- Black-Scholes formulas
- Local Greeks calculation
- Complex options math

## 📚 Documentation

- [Technical Specification v3.0](docs/tech-spec.md)
- [Implementation Plan v2.0](docs/implementation-plan.md)
- [Operations Manual v2.0](docs/ops-manual.md)

## 🛠️ Development Phases

- **Weeks 1-4**: Foundation (Data, ML, Trading logic) ✅
- **Weeks 5-8**: Paper Trading + Community ✅
- **Weeks 9-12**: Production + Full Community
- **Weeks 13-16**: Optimization + Advanced Features

## ⚠️ Risk Disclaimer

Options trading involves substantial risk. This software is for educational purposes. Always understand the risks before trading with real money.

## 📝 License

MIT License - See LICENSE file for details.
