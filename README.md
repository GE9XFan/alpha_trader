# AlphaTrader - Automated Options Trading System

**Version:** 0.2.0 (Phase 2 Complete)  
**Status:** Development - Foundation & Data Pipeline Operational  
**Last Updated:** August 15, 2025  
**Development Approach:** Skeleton-First, API-Driven, Configuration-Based

## 🎯 Project Overview

AlphaTrader is a production-grade automated options trading system that combines real-time market data, options analytics, and machine learning to execute systematic trading strategies. Built with a focus on reliability, scalability, and educational content generation.

### Vision
Build a fully automated trading system that not only trades profitably but also generates comprehensive educational content, market analysis, and community insights.

### Current Capabilities (Phases 0-2 Complete)
- ✅ **Configuration Management** - Centralized YAML-based configuration with environment variables
- ✅ **Database Infrastructure** - PostgreSQL with optimized schemas for options data
- ✅ **Alpha Vantage Integration** - Real-time and historical options data with full Greeks
- ✅ **Rate Limiting** - Token bucket implementation (600 calls/min protection)
- ✅ **Data Ingestion Pipeline** - Automated data flow from API to database
- ✅ **Options Data** - 18,588 contracts stored (SPY full chain with Greeks)
- ✅ **Modular Architecture** - Clean separation of concerns with extensible design

## 📊 System Architecture

### Current Implementation
```
┌─────────────────────────────────────────────────┐
│             Alpha Vantage API                   │
│  • REALTIME_OPTIONS (with Greeks)               │
│  • HISTORICAL_OPTIONS                           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│          Rate Limiter (Token Bucket)            │
│  • 600 calls/min hard limit                     │
│  • 10 tokens/sec refill rate                    │
│  • 20 token burst capacity                      │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│           Data Ingestion Engine                 │
│  • Parse & validate data                        │
│  • Update or insert logic                       │
│  • Type conversion & error handling             │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│            PostgreSQL Database                  │
│  • av_realtime_options table                    │
│  • av_historical_options table                  │
│  • system_config & api_response_log            │
└─────────────────────────────────────────────────┘
```

### Planned Architecture (Phases 3-19)
```
[IBKR Real-Time] + [Alpha Vantage APIs] → [Rate Limited Ingestion]
                            ↓
                    [PostgreSQL + Redis Cache]
                            ↓
                    [Analytics & ML Engine]
                            ↓
                    [Strategy Decision Engine]
                            ↓
                    [Risk Management Layer]
                            ↓
                    [Order Execution (IBKR)]
                            ↓
                [Publishing & Educational Content]
```

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.11+** (tested on 3.11.11)
- **PostgreSQL 14+** (for production data storage)
- **Alpha Vantage API Key** (Premium recommended for 600 calls/min)
- **macOS** (primary development platform)
- **2GB+ RAM** for data processing
- **10GB+ disk space** for historical data

### Step 1: Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/AlphaTrader.git
cd AlphaTrader

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Database Setup
```bash
# Create the database
psql -U your_username -d postgres
CREATE DATABASE trading_system_db;
\q

# Initialize tables
psql -U your_username -d trading_system_db -f scripts/init_db.sql
psql -U your_username -d trading_system_db -f scripts/create_options_table.sql
psql -U your_username -d trading_system_db -f scripts/create_historical_options_table.sql
```

### Step 3: Configuration
```bash
# Create your environment file
cp config/.env.example config/.env

# Edit config/.env with your credentials:
DATABASE_URL=postgresql://username:password@localhost:5432/trading_system_db
AV_API_KEY=your_alpha_vantage_api_key_here
```

### Step 4: Verify Installation
```bash
# Test Phase 0 - Foundation
python scripts/test_phase0.py

# Test Phase 1 - First API
python scripts/test_full_pipeline.py

# Test Phase 2 - Rate Limiting
python scripts/test_rate_limiter.py
python scripts/test_phase2_complete.py
```

## 📁 Project Structure

```
AlphaTrader/
├── src/                           # Source code
│   ├── foundation/
│   │   └── config_manager.py     # Configuration management
│   ├── connections/
│   │   └── av_client.py          # Alpha Vantage API client
│   ├── data/
│   │   ├── ingestion.py          # Data ingestion pipeline
│   │   └── rate_limiter.py       # Token bucket rate limiter
│   └── __init__.py
│
├── config/                        # Configuration files
│   ├── .env                       # Environment variables (not in git)
│   ├── .env.example               # Template for environment
│   └── apis/
│       └── alpha_vantage.yaml    # API settings & endpoints
│
├── scripts/                       # Utility scripts
│   ├── init_db.sql               # System tables
│   ├── create_options_table.sql  # Realtime options schema
│   ├── create_historical_options_table.sql
│   ├── test_phase0.py            # Foundation verification
│   ├── test_full_pipeline.py     # End-to-end test
│   ├── test_rate_limiter.py      # Rate limit testing
│   ├── test_phase2_complete.py   # Phase 2 verification
│   └── query_options_data.py     # Data analysis queries
│
├── data/                          # Data storage
│   └── api_responses/            # Saved API responses for analysis
│
├── docs/                          # Documentation
│   ├── SSOT-Ops.md              # Operational specification
│   ├── SSOT-Tech.md             # Technical specification
│   ├── granular-phased-plan.md  # Implementation roadmap
│   └── educational-*.md         # Educational platform plans
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 💾 Database Schema

### Table: `av_realtime_options`
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| contract_id | VARCHAR(50) | Unique option contract ID |
| symbol | VARCHAR(10) | Underlying symbol (e.g., SPY) |
| expiration | DATE | Option expiration date |
| strike | DECIMAL(10,2) | Strike price |
| option_type | VARCHAR(4) | 'call' or 'put' |
| last_price | DECIMAL(10,2) | Last traded price |
| bid/ask | DECIMAL(10,2) | Current bid/ask |
| volume | INTEGER | Daily volume |
| open_interest | INTEGER | Open interest |
| delta | DECIMAL(7,5) | Option delta |
| gamma | DECIMAL(7,5) | Option gamma |
| theta | DECIMAL(8,5) | Option theta |
| vega | DECIMAL(7,5) | Option vega |
| implied_volatility | DECIMAL(8,5) | IV |
| created_at | TIMESTAMP | Record creation |
| updated_at | TIMESTAMP | Last update |

### Table: `av_historical_options`
Same structure as realtime with additional:
| Column | Type | Description |
|--------|------|-------------|
| data_date | DATE | Historical data date |

## 📈 API Usage & Examples

### Basic Options Data Retrieval
```python
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion

# Initialize client with rate limiting
client = AlphaVantageClient()

# Fetch realtime options with Greeks
options_data = client.get_realtime_options('SPY')
print(f"Retrieved {len(options_data['data'])} contracts")

# Ingest to database
ingestion = DataIngestion()
records = ingestion.ingest_options_data(options_data, 'SPY')
print(f"Processed {records} contracts")

# Check rate limit status
stats = client.get_rate_limit_status()
print(f"API calls: {stats['calls_made']}/{stats['max_per_minute']}")
```

### Query Options Data
```python
from sqlalchemy import create_engine, text
from src.foundation.config_manager import ConfigManager

config = ConfigManager()
engine = create_engine(config.database_url)

# Find near-the-money options
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT contract_id, strike, option_type, 
               last_price, delta, implied_volatility
        FROM av_realtime_options
        WHERE symbol = :symbol
          AND strike BETWEEN :min_strike AND :max_strike
          AND expiration = :exp_date
        ORDER BY strike
    """), {
        'symbol': 'SPY',
        'min_strike': 640,
        'max_strike': 650,
        'exp_date': '2025-08-15'
    })
    
    for row in result:
        print(f"{row.contract_id}: ${row.strike} {row.option_type}, "
              f"Last=${row.last_price}, Delta={row.delta}, IV={row.implied_volatility}")
```

### Rate Limiter Usage
```python
from src.data.rate_limiter import get_rate_limiter

# Get global rate limiter instance
limiter = get_rate_limiter()

# Acquire token before API call
if limiter.acquire(blocking=True, timeout=30):
    # Make API call
    response = make_api_call()
else:
    print("Rate limit exceeded, please wait")

# Check current status
stats = limiter.get_stats()
print(f"Tokens available: {stats['tokens_available']}")
print(f"Calls this minute: {stats['minute_window_calls']}")
```

## 🧪 Testing

### Run All Tests
```bash
# Phase 0: Foundation
python scripts/test_phase0.py

# Phase 1: First API Pipeline
python scripts/test_full_pipeline.py

# Phase 2: Rate Limiting & Multiple APIs
python scripts/test_rate_limiter.py
python scripts/test_rate_limited_api.py
python scripts/test_phase2_complete.py

# Data Analysis
python scripts/query_options_data.py
```

### Performance Benchmarks
| Operation | Target | Actual |
|-----------|--------|--------|
| API Call (with rate limit) | < 2s | ~1.3s |
| Database Insert (9,294 records) | < 30s | ~8s |
| Query Options Chain | < 100ms | ~45ms |
| Rate Limiter Check | < 10ms | ~1ms |

## 📊 Current Data Statistics

| Metric | Value |
|--------|-------|
| **Total Option Contracts** | 18,588 (9,294 realtime + 9,294 historical) |
| **Active Symbol** | SPY (QQQ, IWM ready for Phase 3) |
| **Expiration Range** | Aug 2025 - Dec 2027 (33 dates) |
| **Strike Range** | $150.00 - $1,000.00 |
| **Greeks Coverage** | 100% (Delta, Gamma, Theta, Vega, Rho, IV) |
| **Highest Volume Contract** | SPY250815C00645000 (610,529 volume) |
| **API Calls Used** | ~2-6 per data refresh |
| **Rate Limit Buffer** | 100+ calls/minute available |

## 🛠️ Configuration Details

### Alpha Vantage Configuration (`config/apis/alpha_vantage.yaml`)
```yaml
base_url: "https://www.alphavantage.co/query"
timeout: 30
retry_count: 3

rate_limit:
  max_per_minute: 600
  target_per_minute: 500
  refill_rate: 10
  burst_capacity: 20

endpoints:
  realtime_options:
    function: "REALTIME_OPTIONS"
    require_greeks: "true"  # Critical for Greeks data
    datatype: "json"
  
  historical_options:
    function: "HISTORICAL_OPTIONS"
    datatype: "json"
```

### Environment Variables (`.env`)
```bash
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/trading_system_db

# API Keys
AV_API_KEY=your_alpha_vantage_api_key

# Future additions for Phase 3+
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
```

## 🚦 Development Status

### ✅ Completed Phases (Days 1-10)
- **Phase 0:** Minimal Foundation - Project setup, config, database
- **Phase 1:** First Working Pipeline - REALTIME_OPTIONS API integrated
- **Phase 2:** Rate Limiting & Second API - Token bucket, HISTORICAL_OPTIONS

### 🚧 Next Phases (Days 11-106)
- **Phase 3:** IBKR Connection & Real-Time Pricing
- **Phase 4:** Scheduler & Cache Layer
- **Phase 5:** Core Technical Indicators (RSI, MACD, etc.)
- **Phase 6:** Analytics & Greeks Validation
- **Phase 7:** First Strategy (0DTE)
- **Phase 8:** Risk Management
- **Phase 9:** Paper Trading Execution
- **Phases 10-19:** Complete system with ML, publishing, and production

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Greeks showing as NULL
**Problem:** Options data missing Greeks values  
**Solution:** Ensure `require_greeks: "true"` in `alpha_vantage.yaml`

#### Rate limit errors
**Problem:** API returning rate limit messages  
**Solution:** Rate limiter is working; wait for tokens to refill (10/sec)

#### Database connection failed
**Problem:** Can't connect to PostgreSQL  
**Solution:** 
1. Check PostgreSQL is running: `pg_ctl status`
2. Verify credentials in `.env`
3. Ensure database exists: `psql -U username -l`

#### Import errors
**Problem:** Module import failures  
**Solution:** 
1. Activate virtual environment: `source venv/bin/activate`
2. Reinstall requirements: `pip install -r requirements.txt`
3. Check Python version: `python --version` (needs 3.11+)

## 📚 Documentation

- **[SSOT-Ops.md](docs/SSOT-Ops.md)** - Operational specifications and procedures
- **[SSOT-Tech.md](docs/SSOT-Tech.md)** - Technical implementation details
- **[Granular Phased Plan](docs/granular-phased-plan.md)** - Complete development roadmap
- **[Educational Content Plan](docs/educational-content-plan.md)** - Content generation strategy

## 🤝 Contributing

This is currently a private project in active development. Contribution guidelines will be published when the project reaches Phase 10 (MVP).

## 📄 License

Proprietary - All rights reserved

## 🙏 Acknowledgments

- Alpha Vantage for comprehensive options data API
- PostgreSQL for reliable data storage
- Python community for excellent libraries

## 📮 Contact

For questions or issues:
- Create an issue in the repository

---

**Next Step:** Begin Phase 3 - IBKR Integration for real-time pricing streams

*Last Updated: August 15, 2025 - Phase 2 Complete*