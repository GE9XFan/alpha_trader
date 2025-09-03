# AlphaTrader Pro

A high-performance, Redis-centric institutional options analytics and automated trading system.

## Current Status: Day 2 Complete ✅

### Completed Components

#### Day 1 (Infrastructure) ✅
- ✅ Redis infrastructure configured with persistence
- ✅ Main application skeleton (main.py)
- ✅ Configuration system (YAML + environment variables)
- ✅ Module initialization framework (10 modules ready)
- ✅ Comprehensive test suite (tests/test_day1.py)
- ✅ Graceful shutdown handlers
- ✅ Health monitoring system
- ✅ Connection pooling for Redis

#### Day 2 (IBKR Data Ingestion) ✅
- ✅ IBKR Gateway connection with reconnection logic
- ✅ Level 2 market depth for SPY/QQQ/IWM (0DTE/1DTE/MOC)
- ✅ Standard data for other symbols (14+ DTE)
- ✅ Real-time order book management
- ✅ Trade data collection with buffers
- ✅ 5-second OHLCV bars
- ✅ Redis storage with proper TTLs
- ✅ Performance monitoring and metrics
- ✅ Comprehensive error handling
- ✅ Data quality validation
- ✅ Thread-safe concurrent updates
- ✅ Sweep detection for Level 2 symbols
- ✅ Unusual volume detection

### Test Results
```bash
# Day 1 tests: All passing ✅
# Day 2 tests: 7/8 passing ✅
✅ IBKR Connection
✅ Symbol Classification (Level 2 vs Standard)
✅ Market Data Subscriptions
✅ Data Flow to Redis
✅ Data Quality Validation
⚠️ Performance Metrics (updates every 10s)
✅ Redis Key Structure
```

## Installation

### Prerequisites
- ✅ Python 3.11+
- ✅ Redis 7.0+ (configured with persistence)
- ✅ IBKR Gateway or TWS (paper trading on port 7497)
- ⏳ Alpha Vantage Premium API key (600 calls/min) - Day 3

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AlphaTraderPro.git
cd AlphaTraderPro
```

2. **Ensure IBKR Gateway/TWS is running**
```bash
# Paper trading account on port 7497
# Configure for socket clients
# Enable API connections
```

3. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Start Redis (if not already running)**
```bash
redis-server config/redis.conf
```

6. **Configure the application**
```bash
# Copy example environment file
cp .env.example .env
# Edit .env and add your API keys
```

7. **Run tests to verify setup**
```bash
# Test infrastructure
python tests/test_day1.py

# Test IBKR data ingestion (requires IBKR Gateway)
python tests/test_day2.py
```

8. **Start the application**
```bash
python main.py
```

## Architecture

### Module Structure
```
src/
├── data_ingestion.py    # ✅ IBKR data ingestion COMPLETE
│                       # ⏳ Alpha Vantage (Day 3)
├── analytics.py         # ⏳ VPIN, GEX, parameter discovery (Day 4-5)
├── signals.py          # ⏳ Signal generation & distribution (Day 6)
├── execution.py        # ⏳ Order & position management (Day 7-10)
├── dashboard.py        # ⏳ FastAPI web interface (Day 12)
├── morning_analysis.py # ⏳ GPT-4 analysis generator (Day 16-17)
└── social_media.py     # ⏳ Twitter/Discord/Telegram bots (Day 13-15)
```

### Data Flow Architecture
```
IBKR Gateway (Port 7497)
    ↓
┌─────────────────────────────┐
│   IBKRIngestion Class       │
├─────────────────────────────┤
│ • Level 2 (SPY/QQQ/IWM)    │
│ • Standard (Other symbols)  │
│ • 5-second bars             │
│ • Trade data                │
└─────────────────────────────┘
    ↓ Real-time updates
┌─────────────────────────────┐
│      Redis (In-Memory)      │
├─────────────────────────────┤
│ • Order books (1s TTL)      │
│ • Trades (1s TTL)           │
│ • Bars (10s TTL)            │
│ • Metrics & monitoring      │
└─────────────────────────────┘
```

### Redis Keys (Active)
```
# Market Data - IBKR (✅ LIVE)
market:{symbol}:book       # Level 2 order book (SPY/QQQ/IWM only)
market:{symbol}:trades     # Recent trades list (last 1000)
market:{symbol}:last       # Last trade price
market:{symbol}:bars       # 5-second OHLCV bars (last 100)
market:{symbol}:ticker     # Current bid/ask/volume
market:{symbol}:spread     # Bid-ask spread
market:{symbol}:imbalance  # Order book imbalance (-1 to 1)
market:{symbol}:sweep      # Sweep detection (Level 2 only)
market:{symbol}:unusual_volume # Unusual volume alerts

# Connection Status (✅ LIVE)
ibkr:connected            # Connection status (0/1)
ibkr:account             # Connected account ID

# Monitoring (✅ LIVE)
module:heartbeat:ibkr_ingestion  # Module health
monitoring:ibkr:metrics          # Performance metrics
monitoring:ibkr:errors           # Error log
monitoring:data:stale            # Stale data warnings

# Options Data (⏳ Day 3)
options:{symbol}:chain    # Options chain
options:{symbol}:greeks   # Greeks by strike
sentiment:{symbol}:score  # News sentiment
```

## Development Progress

### Phase 1: Core Infrastructure (Days 1-5)
| Day | Component | Status | Details |
|-----|-----------|--------|----------|
| 1 | Main Application | ✅ Complete | Config, Redis, modules, monitoring |
| 2 | IBKR Ingestion | ✅ Complete | Level 2, trades, bars, real-time flow |
| 3 | Alpha Vantage | ⏳ Next | Options chains, Greeks, sentiment |
| 4 | Parameter Discovery | ⏳ Planned | VPIN, volatility regimes |
| 5 | Analytics Engine | ⏳ Planned | VPIN, GEX, DEX calculations |

### Phase 2: Signal & Execution (Days 6-10)
| Day | Component | Status |
|-----|-----------|--------|
| 6 | Signal Generation | ⏳ Planned |
| 7 | Risk Management | ⏳ Planned |
| 8 | Execution Manager | ⏳ Planned |
| 9 | Position Management | ⏳ Planned |
| 10 | Emergency Systems | ⏳ Planned |

### Phase 3: Distribution (Days 11-15)
| Day | Component | Status |
|-----|-----------|--------|
| 11 | Signal Distribution | ⏳ Planned |
| 12 | Dashboard | ⏳ Planned |
| 13 | Twitter Bot | ⏳ Planned |
| 14 | Telegram Bot | ⏳ Planned |
| 15 | Discord Bot | ⏳ Planned |

## Next Steps

### Day 3: Alpha Vantage Integration
- [ ] Implement rate limiting (600 calls/min)
- [ ] Add options chain fetching with Greeks
- [ ] Implement sentiment data collection
- [ ] Add unusual activity detection
- [ ] Store all data in Redis

### Day 4: Parameter Discovery
- [ ] Implement VPIN bucket size discovery
- [ ] Add temporal structure analysis
- [ ] Implement market maker profiling
- [ ] Add volatility regime detection

## Project Structure
```
AlphaTraderPro/
├── config/
│   ├── config.yaml        # Main configuration
│   ├── redis.conf         # Redis configuration
│   └── discovered.yaml    # Auto-discovered parameters
├── src/
│   ├── data_ingestion.py  # IBKR & Alpha Vantage
│   ├── analytics.py       # Analytics engine
│   ├── signals.py         # Signal generation
│   ├── execution.py       # Order & position management
│   ├── social_media.py    # Twitter, Telegram, Discord
│   ├── dashboard.py       # Web UI
│   └── morning_analysis.py # AI analysis
├── tests/
│   ├── test_day1.py       # Infrastructure tests
│   └── test_day2.py       # IBKR ingestion tests
├── logs/                  # Application logs
├── data/redis/           # Redis persistence
├── main.py               # Main application
├── requirements.txt      # Python dependencies
├── implementation_plan.md # Detailed 30-day plan
└── .env                  # Environment variables
```

## Monitoring

### Health Checks
```bash
# System status
redis-cli get system:halt
redis-cli get system:health:main

# IBKR connection
redis-cli get ibkr:connected
redis-cli get ibkr:account

# Module heartbeats
redis-cli keys 'module:heartbeat:*'

# Performance metrics
redis-cli hgetall monitoring:ibkr:metrics

# Watch live data
redis-cli --scan --pattern 'market:*'
```

### Debug Commands
```bash
# Watch order book updates (SPY)
redis-cli get market:SPY:book | python -m json.tool

# Check latest trades
redis-cli get market:SPY:trades | python -m json.tool | head -20

# Monitor errors
redis-cli lrange monitoring:ibkr:errors 0 10
```

## License
Proprietary - All Rights Reserved

## Support
For issues or questions, please refer to the implementation plan and technical specification documents.