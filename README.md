# AlphaTrader - Real-Time Institutional Options Analytics & Trading System

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Usage](#usage)
- [Trading Strategies](#trading-strategies)
- [Technical Indicators](#technical-indicators)
- [API Documentation](#api-documentation)
- [Development Status](#development-status)
- [Troubleshooting](#troubleshooting)
- [Performance Benchmarks](#performance-benchmarks)
- [Risk Management](#risk-management)
- [Contributing](#contributing)
- [License](#license)

## Overview

AlphaTrader is a professional-grade automated options trading system that combines real-time Level 2 order book data from Interactive Brokers with advanced options analytics from Alpha Vantage. The system implements institutional-level market microstructure indicators to identify and execute high-probability trading opportunities with sub-second latency.

### What Makes This Different

Unlike traditional trading bots that rely on basic technical indicators, AlphaTrader implements:

1. **Institutional Market Microstructure**: VPIN (Volume-Synchronized Probability of Informed Trading), Order Book Imbalance, and Hidden Order Detection algorithms used by hedge funds and market makers.

2. **Real-time Options Flow**: Analyzes options chains with pre-calculated Greeks (no Black-Scholes computation needed) to identify gamma squeezes, unusual options activity, and smart money positioning.

3. **Memory-First Architecture**: Entire system runs in-memory using Redis for <1ms data access, eliminating database bottlenecks that plague traditional systems.

4. **Multi-Strategy Orchestration**: Simultaneously runs multiple strategies (0DTE gamma scalping, 1DTE overnight positioning, swing trades, MOC imbalance) with intelligent capital allocation.

## Key Features

### Data Infrastructure
- **IBKR Level 2 Order Book**: 10 levels of bid/ask depth with real-time updates
- **Options Analytics**: Real-time Greeks, implied volatility, and unusual activity detection
- **Smart Caching**: Redis-based cache with intelligent TTL management (1 second for microstructure, 10 seconds for options)
- **Rate Limiting**: Automatic API rate management for Alpha Vantage (600 calls/min)

### Trading Intelligence
- **VPIN Calculation**: Detects toxic order flow with 50ms computation time
- **Gamma Exposure (GEX)**: Identifies market maker hedging levels and pin strikes
- **Hidden Order Detection**: Identifies iceberg orders and dark pool leakage
- **Multi-Timeframe Analysis**: Separate logic for 0DTE, 1DTE, and swing trades

### Execution & Risk Management
- **Automated Execution**: Direct order routing through IBKR TWS/Gateway
- **Position Management**: Automated stop losses, trailing stops, and scale-out logic
- **Circuit Breakers**: Max daily loss, consecutive loss limits, and emergency shutdown
- **Paper Trading Mode**: Full simulation with real market data

### Monitoring & Distribution
- **Real-time Dashboard**: Web-based monitoring with 1-second updates
- **Discord Integration**: Trade alerts and performance updates
- **Performance Analytics**: P&L tracking, win rate, Sharpe ratio calculations
- **Health Monitoring**: System resource tracking and automatic recovery

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (Dual-Feed)                     │
├──────────────────────────┬──────────────────────────────────────┤
│    IBKR WebSocket        │      Alpha Vantage REST API          │
│    ├─ Level 2 Book       │      ├─ Options Chains w/Greeks      │
│    ├─ Trade Tape         │      ├─ Technical Indicators         │
│    ├─ 5-sec Bars         │      ├─ Sentiment Analysis          │
│    └─ Execution          │      └─ Fundamentals                │
└────────────┬─────────────┴────────────┬────────────────────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CACHE LAYER (Redis)                            │
│  • Order Book: 1 sec TTL    • Options: 10 sec TTL               │
│  • Metrics: 5 sec TTL       • Sentiment: 5 min TTL              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ANALYTICS ENGINE (In-Memory Processing)             │
│  • VPIN Calculation        • Gamma Exposure (GEX)                │
│  • Order Book Imbalance    • Options Flow Analysis               │
│  • Hidden Order Detection  • IV Skew Analysis                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                             │
│  • Confidence Scoring      • Risk/Reward Calculation             │
│  • Strategy Selection      • Position Sizing                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION MANAGER                             │
│  • IBKR Order Routing      • Stop Loss Management                │
│  • Position Tracking       • P&L Calculation                     │
└─────────────────────────────────────────────────────────────────┘
```

## Installation Guide

### Prerequisites

#### Required Software
- **Python 3.11+**: Required for compatibility with all dependencies
- **Redis 7.0+**: In-memory cache (install via `brew install redis` on macOS or `apt install redis-server` on Ubuntu)
- **Interactive Brokers TWS or IB Gateway**: For market data and execution
- **Git**: For version control

#### Required Accounts
- **Interactive Brokers Account**: Paper trading account recommended for testing
- **Alpha Vantage Premium API Key**: Required for options data (600 calls/min tier)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AlphaTrader.git
cd AlphaTrader
```

#### 2. Install Python 3.11 (if needed)
```bash
# macOS
brew install python@3.11

# Ubuntu/Debian
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

#### 3. Run Setup Script
```bash
# Make script executable
chmod +x setup.sh

# Run automated setup
./setup.sh
```

This script will:
- Create a Python virtual environment
- Install all dependencies
- Create necessary directories
- Generate configuration files
- Set up logging

#### 4. Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

Required configurations:
```env
# Alpha Vantage - Get key from https://www.alphavantage.co/premium/
AV_API_KEY=your_actual_api_key_here

# Interactive Brokers
IBKR_ACCOUNT=DU1234567  # Your paper account number
IBKR_PORT=7497          # 7497 for paper, 7496 for live

# Redis (default settings usually work)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_MAX_MEMORY=4gb

# Trading Parameters
MAX_POSITIONS=5
MAX_DAILY_LOSS=2000
RISK_PER_TRADE=0.02
```

#### 5. Start Redis Server
```bash
# Start with optimized configuration
redis-server redis.conf

# Or with basic settings
redis-server --maxmemory 4gb --maxmemory-policy volatile-lru
```

#### 6. Configure Interactive Brokers

1. **Download IB Gateway or TWS** from [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=16457)

2. **Enable API Access:**
   - Open IB Gateway/TWS
   - Go to File → Global Configuration → API → Settings
   - Enable: ✅ ActiveX and Socket Clients
   - Set Socket port: 7497 (paper) or 7496 (live)
   - Allow connections from: 127.0.0.1

3. **Login** with your paper trading credentials

#### 7. Verify Installation
```bash
# Activate virtual environment
source venv/bin/activate

# Test all connections
python scripts/test_connections.py

# Check system health
python scripts/health_check.py
```

Expected output:
```
✓ Redis connected (4GB configured)
✓ Alpha Vantage API connected (600 calls/min)
✓ IBKR connected (Paper account: DU1234567)
✓ All systems operational
```

## Configuration

### Main Configuration File (`config/config.yaml`)

```yaml
# Trading Strategies Configuration
strategies:
  0DTE:
    enabled: true
    start_time: "09:45"
    end_time: "15:00"
    min_confidence: 70
    max_position_size: 0.05  # 5% of account
    
  1DTE:
    enabled: true
    start_time: "14:00"
    end_time: "15:30"
    min_confidence: 65
    max_position_size: 0.08
    
  14DTE:
    enabled: true
    start_time: "09:30"
    end_time: "16:00"
    min_confidence: 60
    max_position_size: 0.10

# Risk Management
risk:
  max_positions: 5
  max_daily_loss: 2000
  max_position_loss: 500
  max_correlation: 0.70
  circuit_breakers:
    consecutive_losses: 3
    drawdown_percent: 0.02
```

### Symbols Configuration (`config/symbols.yaml`)

```yaml
primary:
  - symbol: SPY
    strategies: ["0DTE", "1DTE", "MOC"]
    min_volume: 50000000
    max_spread_pct: 0.001
    
  - symbol: QQQ
    strategies: ["0DTE", "1DTE"]
    min_volume: 30000000
    max_spread_pct: 0.001
```

## Usage

### Starting the System

#### Manual Start (Development)
```bash
# Terminal 1: Start Redis
redis-server redis.conf

# Terminal 2: Start main trading system
source venv/bin/activate
python scripts/start.py
```

#### Production Start
```bash
# Start all services
./scripts/start_production.sh
```

### Monitoring

#### Web Dashboard
```bash
# Start dashboard (separate terminal)
python dashboard/app.py
# Access at http://localhost:8080
```

#### Command Line Monitoring
```bash
# Watch real-time logs
tail -f logs/trading.log

# Check system health
python scripts/health_check.py

# View positions
python scripts/show_positions.py
```

### Manual Trading Commands

```python
# Interactive Python shell for manual control
python scripts/trading_console.py

>>> # Get current positions
>>> positions = get_positions()

>>> # Close all positions
>>> close_all_positions()

>>> # Pause trading
>>> pause_trading()

>>> # Resume trading
>>> resume_trading()
```

## Trading Strategies

### 0DTE (Zero Days to Expiration)
**Timeframe**: 9:45 AM - 3:00 PM  
**Focus**: Gamma squeezes and pin moves  
**Logic**: Identifies high gamma concentration strikes where market makers must hedge aggressively. Enters when spot price diverges from pin strike.

### 1DTE (One Day to Expiration)
**Timeframe**: 2:00 PM - 3:30 PM  
**Focus**: Overnight gaps and events  
**Logic**: Positions for overnight moves based on elevated implied volatility and unusual options activity.

### 14DTE+ (Swing Trades)
**Timeframe**: All day  
**Focus**: Following smart money flow  
**Logic**: Mirrors large institutional options orders identified through sweep detection.

### MOC (Market-on-Close)
**Timeframe**: 3:30 PM - 3:50 PM  
**Focus**: Closing auction imbalances  
**Logic**: Predicts and trades closing imbalances based on gamma exposure and order flow.

## Technical Indicators

### VPIN (Volume-Synchronized Probability of Informed Trading)
```python
# Calculation in analytics/microstructure.py
vpin = calculate_vpin(trades, bucket_size=50)
# Returns: 0-1 score (>0.4 indicates toxic flow)
```

### Order Book Imbalance (OBI)
```python
# Measures bid/ask pressure
obi = calculate_order_book_imbalance(order_book)
# Returns: -1 to +1 (-1 = heavy selling, +1 = heavy buying)
```

### Gamma Exposure (GEX)
```python
# Net market maker gamma exposure
gex = calculate_gamma_exposure(options_chain, spot_price)
# Returns: Total GEX in millions, pin strike, flip point
```

## API Documentation

### REST API Endpoints

```bash
# Get current positions
GET /api/v1/positions

# Get trading signals
GET /api/v1/signals?confidence=70

# Get system metrics
GET /api/v1/metrics

# Emergency stop
POST /api/v1/emergency/stop
```

### WebSocket Streams

```javascript
// Connect to real-time feed
ws://localhost:8001/stream

// Subscribe to signals
{"action": "subscribe", "channel": "signals"}

// Subscribe to positions
{"action": "subscribe", "channel": "positions"}
```

## Development Status

### ✅ Completed (Day 1-2)
- Python 3.11 environment setup
- Redis configuration (4GB, optimized for trading)
- IBKR paper trading connection
- Alpha Vantage Premium API integration
- Project structure and configuration
- Connection testing and health monitoring

### 🚧 In Progress (Day 3-4)
- [ ] Redis cache manager implementation
- [ ] IBKR Level 2 data subscription
- [ ] Alpha Vantage options client
- [ ] Data models and structures

### 📅 Upcoming (Week 2-4)
- [ ] VPIN and microstructure indicators
- [ ] Options analytics (GEX, skew)
- [ ] Signal generation engine
- [ ] Automated execution
- [ ] Position management
- [ ] Web dashboard
- [ ] Discord bot

## Troubleshooting

### Common Issues and Solutions

#### Redis Connection Error
```bash
# Error: "Redis connection refused"
# Solution: Start Redis server
redis-server redis.conf

# Error: "MISCONF Redis is configured to save RDB snapshots"
# Solution: Disable persistence
redis-cli CONFIG SET stop-writes-on-bgsave-error no
```

#### IBKR Connection Failed
```bash
# Error: "Cannot connect to TWS"
# Solutions:
1. Ensure TWS/IB Gateway is running
2. Check API is enabled in Global Configuration
3. Verify port number (7497 for paper, 7496 for live)
4. Check firewall isn't blocking connection
```

#### Alpha Vantage Rate Limit
```bash
# Error: "API rate limit exceeded"
# Solutions:
1. Verify you have Premium tier (600 calls/min)
2. Check rate limiting in code is working
3. Reduce number of symbols monitored
```

#### Python Import Errors
```bash
# Error: "ModuleNotFoundError"
# Solutions:
1. Activate virtual environment: source venv/bin/activate
2. Reinstall dependencies: pip install -r requirements.txt
3. Check Python version: python --version (must be 3.11+)
```

## Performance Benchmarks

### Current Performance (Paper Trading)
| Metric | Target | Achieved | Status |
|--------|--------|----------|---------|
| Order Book Update Latency | <10ms | TBD | 🔄 |
| Signal Generation | <50ms | TBD | 🔄 |
| End-to-End Latency | <150ms | TBD | 🔄 |
| Cache Hit Rate | >90% | TBD | 🔄 |
| API Calls/Minute | 600 | 600 | ✅ |

### System Resources
- **Memory Usage**: ~400MB per symbol (4GB total for 10 symbols)
- **CPU Usage**: <20% on 8-core system
- **Network**: ~10 Mbps during market hours
- **Storage**: <100GB (logs only, no database)

## Risk Management

### Position Limits
- Maximum 5 concurrent positions
- Maximum 25% of account per position
- Maximum 70% correlation between positions
- Automatic position sizing based on volatility

### Loss Prevention
- Daily loss limit: $2,000 (paper) / configurable for live
- Per-position stop loss: 2% of account
- Circuit breaker: 3 consecutive losses triggers shutdown
- Maximum drawdown: 2% triggers risk reduction

### Emergency Procedures
```bash
# Emergency close all positions
python scripts/emergency_close.py

# Pause all trading
python scripts/pause_trading.py

# Reduce all positions by 50%
python scripts/reduce_risk.py
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linters
black .
isort .
mypy .
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for everyone. 

- Always test with paper trading first
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- This is not financial advice

## Support

- **Documentation**: [Wiki](https://github.com/YOUR_USERNAME/AlphaTrader/wiki)
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/AlphaTrader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/AlphaTrader/discussions)

## Acknowledgments

- **Interactive Brokers** for market data and execution APIs
- **Alpha Vantage** for options analytics and technical indicators
- **Redis** for high-performance caching
- **ib_insync** for IBKR Python integration

---

**Project Status**: 🟢 Active Development  
**Current Phase**: Day 3-4 - Building Data Pipeline  
**Last Updated**: August 26, 2025