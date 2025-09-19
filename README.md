# QuantiCity Capital

## What This Is

QuantiCity Capital is a production-grade algorithmic trading platform that ingests real-time market microstructure from Interactive Brokers and Alpha Vantage, computes dealer positioning metrics (Vanna, Charm, GEX/DEX), and orchestrates automated options execution with integrated risk controls. The system processes Level 2 depth across major indices, maintains sub-second analytics pipelines, and distributes signals through tiered channels while managing live positions with bracket orders and trailing stops.

## Why It Matters

Traditional retail platforms lack visibility into dealer hedging flows and market maker positioning that drive intraday price action. This platform bridges that gap by combining institutional-grade analytics (VPIN toxicity, flow clustering, volatility regimes) with retail-accessible execution infrastructure, enabling systematic options strategies that exploit dealer rebalancing patterns. The entire stack runs on commodity hardware with Redis as the single source of truth, making sophisticated market analysis accessible without enterprise infrastructure.

## At-a-Glance Capabilities

| Capability | Coverage | Update Frequency | Data Sources |
|------------|----------|------------------|--------------|
| **Level 2 Market Depth** | SPY, QQQ, IWM | < 5ms latency | IBKR SMART routing |
| **Options Analytics** | All liquid strikes | 5 minutes | Alpha Vantage chains |
| **Dealer Positioning** | Vanna, Charm, 0DTE skew | 30 seconds | Computed from chains |
| **Trade Flow Classification** | Momentum/hedging/reversion | 60 seconds | ML clustering |
| **Signal Generation** | 0/1/14 DTE, MOC | Continuous | Multi-factor scoring |
| **Automated Execution** | Bracket orders, trailing stops | Real-time | IBKR API |
| **Risk Management** | Position limits, circuit breakers | Real-time | Internal controls |

## System Components

| Component | Responsibilities | Redis Touchpoints |
|-----------|------------------|-------------------|
| **Ingestion** | • Stream IBKR Level 2 depth with venue codes<br>• Aggregate 1-minute bars from 5-second samples<br>• Fetch options chains with full Greeks | `market:{symbol}:book`<br>`market:{symbol}:bars:1min`<br>`options:{symbol}:chain` |
| **Analytics** | • Calculate VPIN toxicity and order imbalance<br>• Compute GEX/DEX exposure profiles<br>• Classify trade flow via KMeans clustering | `analytics:{symbol}:vpin`<br>`analytics:{symbol}:gex`<br>`analytics:flow_clusters:{symbol}` |
| **Signal Engine** | • Score multi-factor DTE opportunities<br>• Detect MOC auction imbalances<br>• Enforce contract-level deduplication | `signals:pending:{symbol}`<br>`signals:execution:{symbol}`<br>`signals:emitted:{fingerprint}` |
| **Execution** | • Place bracket orders on fills<br>• Manage trailing stop adjustments<br>• Sync positions with IBKR in real-time | `positions:open:{account}:{conId}`<br>`orders:pending:{orderId}`<br>`execution:fills:{symbol}` |
| **Distribution** | • Route signals to tier queues<br>• Enforce 0/60/300s delays<br>• Track delivery metrics | `distribution:premium:queue`<br>`distribution:basic:queue`<br>`distribution:metrics` |

## Architecture Overview

```
┌─────────────┐                          ┌─────────────┐
│ IBKR Gateway├──WebSocket──────────────▶│  Ingestion  │
└─────────────┘                          └──────┬──────┘
                                                 │
┌─────────────┐                                 ▼
│Alpha Vantage├──REST+RateLimit────────▶┌─────────────┐
└─────────────┘                          │    Redis    │
                                         │  (Central)  │
                                         └──────┬──────┘
                                                 │
    ┌──────────────┬──────────────┬─────────────┼─────────────┬──────────────┐
    ▼              ▼              ▼             ▼             ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Analytics│  │ Signals  │  │Execution │  │   Risk   │  │  Distro  │  │Dashboard │
└────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Quick Validation

```bash
# Check system health
redis-cli get heartbeat:ibkr_ingestion  # Should return recent timestamp
redis-cli hlen market:SPY:book          # Should show depth levels

# Verify analytics
redis-cli hget analytics:SPY:vpin value # Should show toxicity score
redis-cli hget analytics:SPY:gex total  # Should show gamma exposure

# Monitor signals
redis-cli llen signals:pending:SPY      # Pending signal count
```

## Runbook

### Prerequisites

- Python 3.10+ with venv support
- Redis 6.2+ (local or managed)
- IBKR TWS/Gateway on port 7497
- Alpha Vantage API key (600 calls/min tier)

### Environment Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/quanticity/capital
cd capital

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env with your keys:
# - ALPHA_VANTAGE_API_KEY
# - TELEGRAM_BOT_TOKEN (optional)
# - TWITTER_API_KEY (optional)

# 5. Verify Redis
redis-cli ping  # Should return PONG
```

### Process Orchestration

```bash
# Start main coordinator (manages all modules)
python main.py

# Or run specific modules
python -m src.ibkr_ingestion        # Market data only
python -m src.analytics_engine      # Analytics only
python -m src.signal_generator      # Signals only
```

### Monitoring & Alerts

| Metric | Redis Key | Alert Threshold |
|--------|-----------|-----------------|
| IBKR Connection | `ibkr:connected` | = 0 for > 30s |
| Stale Data | `monitoring:data:stale` | Any symbol > 10s |
| Risk Violations | `risk:breaches:count` | > 3 per hour |
| Failed Orders | `orders:failed:count` | > 5 per hour |
| API Rate Limit | `monitoring:alpha_vantage:metrics` | tokens < 10 |

### Redis Schema & TTLs

| Namespace | TTL | Description |
|-----------|-----|-------------|
| `market:*` | 60-3600s | Real-time market data |
| `options:*` | 900s | Options chains and Greeks |
| `analytics:*` | 300s | Computed metrics |
| `signals:*` | 3600s | Generated signals |
| `positions:*` | None | Position state (persistent) |
| `risk:*` | 86400s | Daily risk metrics |

## Testing

```bash
# Unit tests (no external dependencies)
pytest tests/unit/ -v

# Integration tests (requires Redis)
pytest tests/integration/ -v

# Specific module
pytest tests/test_ibkr_processor.py -v

# With coverage
pytest --cov=src --cov-report=html
```

## Code Touchpoints

### Ingestion Subsystem
- Market data streaming: `src/ibkr_ingestion.py:124-163`
- Level 2 depth processing: `src/ibkr_ingestion.py:354-396`
- Options chain processing: `src/av_options.py:166-265`
- Sentiment analysis: `src/av_sentiment.py:206-297`

### Analytics Subsystem
- VPIN calculation: `src/vpin_calculator.py:31-116`
- GEX/DEX computation: `src/gex_dex_calculator.py:47-105`
- Flow clustering: `src/flow_clustering.py:30-199`
- Dealer metrics: `src/dealer_flow_calculator.py:75-206`

### Signal Subsystem
- Feature extraction: `src/signal_generator.py:542-803`
- DTE strategies: `src/dte_strategies.py:391-540`
- MOC strategy: `src/moc_strategy.py:55-214`
- Deduplication: `src/signal_deduplication.py:76-133`

### Execution Subsystem
- Order placement: `src/execution_manager.py:840-1593`
- Bracket management: `src/execution_manager.py:828-972`
- Position sync: `src/position_manager.py:165-427`
- Risk checks: `src/risk_manager.py:57-350`

## License

Proprietary - QuantiCity Capital © 2025