# AlphaTrader - Automated Options Trading System

[![Phase](https://img.shields.io/badge/Phase-0%20Complete-green)]()
[![Environment](https://img.shields.io/badge/Environment-Development-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)]()
[![Redis](https://img.shields.io/badge/Redis-Latest-red)]()

Production-grade automated options trading system with real money capabilities. Built with a skeleton-first, API-driven, configuration-based architecture.

## 🚀 Current Status: Phase 0 Complete

### ✅ Completed Components

- **Infrastructure Foundation** - Complete project skeleton with 40+ modules
- **Configuration Management** - 30+ YAML files, zero hardcoded values
- **Database Setup** - PostgreSQL with system tables (data tables created during API discovery)
- **Cache Layer** - Redis for high-performance caching
- **Base Classes** - Consistent interfaces across all modules
- **Module Initialization** - Dependency-aware initialization chain
- **Environment Support** - Development, Paper, Production modes

## 📊 System Architecture

```
AlphaTrader/
├── src/
│   ├── foundation/        # Core infrastructure (ConfigManager, Base Classes)
│   ├── connections/       # API connections (IBKR, Alpha Vantage)
│   ├── data/             # Data ingestion, caching, scheduling
│   ├── analytics/        # Indicators, Greeks validation
│   ├── ml/               # Feature engineering, model integration
│   ├── decision/         # Decision engine, strategy orchestration
│   ├── strategies/       # Trading strategies (0DTE, 1DTE, Swing, MOC)
│   ├── risk/             # Risk management, position sizing
│   ├── execution/        # Order execution via IBKR
│   ├── monitoring/       # Trade monitoring, performance tracking
│   ├── publishing/       # Discord alerts, notifications
│   └── api/              # Dashboard API
├── config/               # All configuration files (YAML)
│   ├── system/          # Database, Redis, logging, paths
│   ├── apis/            # API configurations and rate limits
│   ├── data/            # Symbols, schedules, validation
│   ├── strategies/      # Strategy parameters
│   ├── risk/            # Risk limits and circuit breakers
│   ├── ml/              # Model paths and features
│   ├── execution/       # Trading hours, order types
│   ├── monitoring/      # Alerts and dashboard settings
│   └── environments/    # Environment-specific overrides
├── scripts/             # Utility scripts
├── tests/               # Test suites
├── models/              # ML model storage
├── logs/                # Application logs
└── data/                # Data storage

```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.11 (required - not 3.13)
- PostgreSQL 16
- Redis
- Alpha Vantage API key
- Interactive Brokers account (paper or live)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/AlphaTrader.git
cd AlphaTrader
```

### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
# Required: AV_API_KEY, IBKR_USERNAME, IBKR_PASSWORD, IBKR_ACCOUNT
```

### Step 5: Setup Database

```bash
# Database and tables are already created
# Verify connection:
python scripts/initialize_system.py --health-check-only
```

### Step 6: Verify Installation

```bash
# Test infrastructure
python test_infrastructure.py

# Test base classes
python test_base_classes.py

# Test configuration
python -m src.foundation.config_manager
```

## 🏃 Running the System

### Development Mode (Default)

```bash
# Initialize system in development mode
python scripts/initialize_system.py --env development
```

### Paper Trading Mode

```bash
# Initialize system for paper trading
python scripts/initialize_system.py --env paper
```

### Production Mode (REAL MONEY - Use with Caution!)

```bash
# Initialize system for live trading
python scripts/initialize_system.py --env production
# Will require explicit confirmation
```

### Health Check Only

```bash
# Check system health without starting
python scripts/initialize_system.py --health-check-only
```

## ⚙️ Configuration

All system behavior is controlled through YAML configuration files in the `config/` directory:

- **NO HARDCODED VALUES** - Everything is configurable
- **Environment-specific overrides** - Different settings for dev/paper/production
- **Strategy parameters** - Fully tunable without code changes
- **Risk limits** - Position and portfolio constraints
- **Circuit breakers** - Automatic safety stops

Key configurations:
- `config/apis/` - API settings and rate limits
- `config/strategies/` - Trading strategy parameters
- `config/risk/` - Risk management rules
- `config/execution/` - Trading hours and order types

## 🔒 Safety Features

- **Paper Trading Default** - System defaults to paper trading
- **Environment Confirmation** - Production mode requires explicit confirmation
- **Circuit Breakers** - Automatic halt on 2% daily loss
- **Rate Limiting** - Never exceeds API limits (600/min for Alpha Vantage)
- **Position Limits** - Maximum position sizes enforced
- **Greeks Validation** - All options data validated before use

## 📈 Trading Strategies

### Implemented Strategies (Skeletons Ready)

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

## 🚧 Development Roadmap

### ✅ Phase 0: Infrastructure (COMPLETE)
- Project skeleton
- Configuration management
- Database setup
- Base classes
- Module initialization

### 🔄 Phase 0.5: API Discovery (NEXT)
- Test each API endpoint
- Analyze response structures
- Create schemas from actual data
- Build ingestion pipelines

### 📋 Upcoming Phases
- **Phase 1**: Complete Connections Layer
- **Phase 2**: Data Management Layer
- **Phase 3**: Analytics Engine
- **Phase 4**: ML Integration
- **Phase 5**: Decision Engine
- **Phase 6**: Risk & Execution
- **Phase 7**: Output Layer
- **Phase 8**: Integration Testing
- **Phase 9**: Production Deployment

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Test specific module
pytest tests/unit/test_config_manager.py

# Integration tests
pytest tests/integration/
```

## 📊 Monitoring

- **Health Checks** - Continuous system health monitoring
- **Performance Metrics** - Tracked in database
- **API Call Logging** - All API calls logged
- **Emergency Log** - Critical events tracked
- **Discord Alerts** - Real-time notifications

## 🔍 Data Sources

### IBKR (Interactive Brokers)
- Real-time pricing (1s, 5s, 1m, 5m bars)
- Real-time quotes
- MOC imbalance data
- Trade execution
- Position monitoring

### Alpha Vantage (43 APIs)
- Options chains with Greeks
- Technical indicators (RSI, MACD, BBANDS, etc.)
- Analytics (fixed/sliding window)
- Fundamentals data
- Economic indicators
- News sentiment

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   - Always run scripts from project root
   - Use `python -m src.module.name` format

2. **Database Connection Failed**
   - Check PostgreSQL is running: `brew services list`
   - Verify credentials in `.env`

3. **Redis Connection Failed**
   - Check Redis is running: `redis-cli ping`
   - Should return `PONG`

4. **API Keys Not Working**
   - Verify keys in `.env`
   - Check not using example values

## 🤝 Contributing

This is a production trading system handling real money. All contributions must:
- Include comprehensive tests
- Update documentation
- Pass all existing tests
- Be reviewed before merge

## ⚠️ Disclaimer

**IMPORTANT**: This system trades real money when in production mode. 
- Test thoroughly in paper mode first
- Never trade with money you can't afford to lose
- Past performance doesn't guarantee future results
- You are responsible for all trades executed

## 📚 Documentation

- [Operational Specification (SSOT-Ops.md)](SSOT-Ops.md) - What the system does
- [Technical Specification (SSOT-Tech.md)](SSOT-Tech.md) - How it's implemented
- [Granular Implementation Plan](granular-phased-plan.md) - Development roadmap

## 📞 Support

For issues or questions:
1. Check documentation first
2. Review troubleshooting guide
3. Check existing issues
4. Create new issue with details

## 📜 License

Proprietary - All Rights Reserved

---

**Current Phase**: 0 (Infrastructure) ✅ COMPLETE  
**Next Phase**: 0.5 (API Discovery) 🔄 READY TO START  
**Environment**: Development  
**Trading Mode**: Paper (Safe)  