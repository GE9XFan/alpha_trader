# AlphaTrader Pro

Institutional-grade options trading system with real-time signal distribution.

## System Overview

AlphaTrader Pro is a sophisticated automated trading system that combines:
- **Level 2 Order Flow**: Real-time market depth from IBKR
- **Options Analytics**: Greeks and chains from Alpha Vantage
- **Advanced Algorithms**: VPIN, GEX/DEX, hidden order detection
- **Multi-Strategy**: 0DTE, 1DTE, 14DTE, MOC strategies
- **Social Distribution**: Discord, Twitter, Telegram integration
- **AI Analysis**: GPT-4 powered morning market analysis

## Architecture

### Redis-Centric Design
- **Zero module dependencies**: All communication through Redis
- **Natural fault isolation**: Modules can fail/restart independently
- **Perfect observability**: All state visible in Redis CLI
- **Automatic backpressure**: Queues naturally form in Redis

### 16 Independent Modules
1. IBKR WebSocket Ingestion
2. Alpha Vantage Ingestion
3. Parameter Discovery
4. Analytics Engine
5. Signal Generator
6. Execution Manager
7. Position Manager
8. Risk Manager & Circuit Breakers
9. Signal Distributor
10. Discord Bot
11. Dashboard & Monitoring
12. Main Application
13. Twitter Integration
14. Telegram Integration
15. Morning Market Analysis
16. Scheduled Tasks

## Trading Symbols

```
SPY, QQQ, IWM, AAPL, TSLA, NVDA, AMD, GOOGL, META, AMZN, MSFT, VXX
```

## Quick Start

### Prerequisites
- MacOS (development) or Linux (production)
- Python 3.9+
- Redis 6.0+
- IBKR TWS/Gateway with API enabled
- Alpha Vantage Premium API key (600 calls/min)

### Installation

1. **Clone the repository**
```bash
cd /Users/michaelmerrick/AlphaTraderPro
```

2. **Install Redis**
```bash
brew install redis
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure the system**
```bash
cp .env.example .env
# Edit .env with your API keys

# Edit config/config.yaml with your specific settings
```

5. **Start Redis**
```bash
redis-server config/redis.conf
```

6. **Run tests**
```bash
python tests/test_config_load.py
python tests/test_redis_connection.py
```

7. **Start the system**
```bash
python main.py
```

## Project Structure

```
AlphaTraderPro/
├── config/               # Configuration files
├── modules/              # Trading modules (16 total)
│   ├── ingestion/       # Data ingestion (IBKR, Alpha Vantage)
│   ├── analytics/       # Analytics engine
│   ├── signals/         # Signal generation
│   ├── execution/       # Order execution
│   ├── risk/           # Risk management
│   ├── distribution/    # Signal distribution
│   ├── dashboard/       # Web dashboard
│   └── social/         # Social media integration
├── tests/               # Test suite
├── logs/               # System logs
├── data/               # Redis persistence
├── scripts/            # Utility scripts
└── main.py             # Main application

```

## Configuration

### Redis Keys (TTL Strategy)
- **Market Data**: 1 second TTL
- **Options Data**: 10 second TTL
- **Calculated Metrics**: 5 second TTL
- **Signals**: 60 second TTL
- **Discovered Parameters**: 24 hour TTL
- **Positions**: No TTL (persistent)

### Trading Strategies

| Strategy | Time Window | Focus |
|----------|------------|-------|
| 0DTE | 9:45 AM - 3:00 PM | Gamma-driven intraday |
| 1DTE | 2:00 PM - 3:30 PM | Overnight positioning |
| 14DTE | 9:30 AM - 4:00 PM | Unusual activity |
| MOC | 3:30 PM - 3:50 PM | Close imbalances |

### Risk Limits
- Max positions: 5 total, 2 per symbol
- Daily loss limit: $2,000
- Max consecutive losses: 3
- Max drawdown: 10%
- Kelly fraction: 0.25

## Implementation Timeline

| Phase | Week | Focus |
|-------|------|-------|
| 1 | Week 1 | Data Infrastructure & Collection |
| 2 | Week 2 | Analytics & Signal Generation |
| 3 | Week 3 | Execution System |
| 4 | Week 4 | Distribution & Monetization |
| 5 | Week 5 | Social & Advanced Features |
| 6 | Week 6 | Production Deployment |

## Monetization

### Subscription Tiers
- **Premium**: $149/month - Real-time signals, all features
- **Basic**: $49/month - 60-second delayed signals
- **Free**: 5-minute delay, limited signals

### Distribution Channels
- Discord (primary)
- Telegram (secondary)
- Twitter (marketing)

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Paper Trading
Configure IBKR paper account in `config/config.yaml`:
```yaml
ibkr:
  port: 7497  # Paper trading port
  account: DU1234567  # Your paper account
```

### Validation Metrics
- Win rate > 45%
- Sharpe ratio > 1.0
- Max drawdown < 10%
- Execution latency < 100ms

## Monitoring

### Redis CLI
```bash
redis-cli
> KEYS *
> GET market:SPY:last
> MONITOR  # Watch all Redis operations
```

### Dashboard
Access the web dashboard at `http://localhost:8000` (once implemented)

## Documentation

- **Technical Specification**: [complete_tech_spec.md](complete_tech_spec.md)
- **Implementation Plan**: [implementation_plan.md](implementation_plan.md)

## Support

For issues or questions about the implementation, refer to:
1. The complete technical specification
2. The implementation plan with daily tasks
3. Redis key schema in Section 2 of tech spec

## License

Proprietary - All rights reserved

---

**Note**: This is Day 1-2 of the implementation. Modules will be added incrementally according to the implementation plan.