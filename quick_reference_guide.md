# AlphaTrader Quick Reference Guide

## 🚀 Daily Development Commands

### Start Development Session
```bash
cd ~/AlphaTrader
source venv/bin/activate
export DATABASE_URL=postgresql://michaelmerrick:password@localhost:5432/trading_system_db
```

### Common Operations
```bash
# Test current setup
python scripts/test_phase0.py          # Test foundation
python scripts/test_phase2_complete.py  # Test all Phase 0-2

# Fetch fresh data
python -c "from src.connections.av_client import AlphaVantageClient; client = AlphaVantageClient(); client.get_realtime_options('SPY')"

# Query data
python scripts/query_options_data.py

# Check rate limit status
python -c "from src.connections.av_client import AlphaVantageClient; client = AlphaVantageClient(); print(client.get_rate_limit_status())"
```

### Database Operations
```bash
# Connect to database
psql -U michaelmerrick -d trading_system_db

# Common queries
\dt                                    # List all tables
SELECT COUNT(*) FROM av_realtime_options;
SELECT * FROM av_realtime_options LIMIT 5;
\d av_realtime_options                # Show table structure
\q                                     # Exit
```

## 💻 Code Snippets

### Fetch & Store Options Data
```python
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion

# Get options with Greeks
client = AlphaVantageClient()
data = client.get_realtime_options('SPY')

# Store in database
ingestion = DataIngestion()
records = ingestion.ingest_options_data(data, 'SPY')
print(f"Stored {records} contracts")
```

### Query Near-The-Money Options
```python
from sqlalchemy import create_engine, text
from src.foundation.config_manager import ConfigManager

config = ConfigManager()
engine = create_engine(config.database_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT contract_id, strike, option_type, last_price, delta
        FROM av_realtime_options
        WHERE symbol = 'SPY'
          AND strike BETWEEN 640 AND 650
          AND expiration = (SELECT MIN(expiration) FROM av_realtime_options)
        ORDER BY strike
    """))
    
    for row in result:
        print(f"{row.contract_id}: ${row.strike} {row.option_type}, Delta={row.delta}")
```

### Check Rate Limiter
```python
from src.data.rate_limiter import get_rate_limiter

limiter = get_rate_limiter()
stats = limiter.get_stats()
print(f"Calls made: {stats['calls_made']}")
print(f"Tokens available: {stats['tokens_available']}/20")
print(f"This minute: {stats['minute_window_calls']}/600")
```

## 📊 Important SQL Queries

### Data Overview
```sql
-- Summary statistics
SELECT 
    COUNT(*) as total_contracts,
    COUNT(DISTINCT symbol) as symbols,
    COUNT(DISTINCT expiration) as expirations,
    MIN(strike) as min_strike,
    MAX(strike) as max_strike
FROM av_realtime_options;

-- Contracts by expiration
SELECT 
    expiration,
    COUNT(*) as contracts,
    SUM(volume) as total_volume
FROM av_realtime_options
GROUP BY expiration
ORDER BY expiration
LIMIT 10;

-- High volume options
SELECT 
    contract_id,
    strike,
    option_type,
    volume,
    last_price,
    delta
FROM av_realtime_options
WHERE volume > 100000
ORDER BY volume DESC
LIMIT 10;

-- Near-the-money options
SELECT *
FROM av_realtime_options
WHERE symbol = 'SPY'
  AND strike BETWEEN 640 AND 650
  AND expiration = '2025-08-15'
ORDER BY strike, option_type;
```

## 🔧 Configuration Reference

### Key Files
```
config/.env                    # API keys & database URL
config/apis/alpha_vantage.yaml # API settings & endpoints
```

### Environment Variables
```bash
DATABASE_URL=postgresql://michaelmerrick:password@localhost:5432/trading_system_db
AV_API_KEY=your_actual_api_key_here
```

### Alpha Vantage Settings
```yaml
# config/apis/alpha_vantage.yaml
endpoints:
  realtime_options:
    function: "REALTIME_OPTIONS"
    require_greeks: "true"    # CRITICAL - enables Greeks!
    datatype: "json"
```

## 🐛 Troubleshooting

### Greeks showing as NULL
```python
# Check configuration
config = ConfigManager()
print(config.av_config['endpoints']['realtime_options'])
# Should show: require_greeks: "true"
```

### Rate limit issues
```python
# Check current status
from src.connections.av_client import AlphaVantageClient
client = AlphaVantageClient()
print(client.get_rate_limit_status())
# Wait if tokens depleted
```

### Database connection errors
```bash
# Test connection
psql -U michaelmerrick -d trading_system_db -c "SELECT 1;"

# Check PostgreSQL status
pg_ctl status
# or
brew services list | grep postgresql
```

### Import errors
```bash
# Ensure virtual environment active
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt
```

## 📁 Project Structure Reminder

```
AlphaTrader/
├── src/
│   ├── foundation/config_manager.py    # Config management
│   ├── connections/av_client.py        # API client
│   └── data/
│       ├── ingestion.py               # Data pipeline
│       └── rate_limiter.py            # Rate limiting
├── config/
│   ├── .env                           # Credentials
│   └── apis/alpha_vantage.yaml        # API config
├── scripts/                           # Test & utility scripts
└── data/api_responses/                # Saved responses
```

## 📈 Current System Capabilities

| Component | Status | Details |
|-----------|--------|---------|
| **Config Manager** | ✅ Active | Loads .env and YAML |
| **AV Client** | ✅ Active | 2 endpoints integrated |
| **Rate Limiter** | ✅ Active | 600/min protection |
| **Data Ingestion** | ✅ Active | Insert/update logic |
| **Database** | ✅ Active | 4 tables, 18K+ records |

## 🎯 Phase 3 Preview (IBKR)

### Required Setup
```bash
# Install IB API
pip install ibapi

# IBKR Settings (.env)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper: 7497, Live: 7496
IBKR_CLIENT_ID=1
```

### Basic Connection (Phase 3.1)
```python
# Coming in Phase 3
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class IBKRConnection(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
    
    def connect_tws(self):
        self.connect("127.0.0.1", 7497, clientId=1)
```

## 📝 Git Commands

```bash
# Check status
git status

# Add all changes
git add .

# Commit with message
git commit -m "Phase 2 complete: Rate limiting and dual APIs"

# Create branch for Phase 3
git checkout -b phase-3-ibkr-integration
```

## 🔗 Important Links

- [Alpha Vantage Docs](https://www.alphavantage.co/documentation/)
- [IBKR API Docs](https://interactivebrokers.github.io/tws-api/)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)

## 📊 Performance Benchmarks

| Operation | Target | Current |
|-----------|--------|---------|
| API Call | < 2s | ~1.3s ✅ |
| DB Insert (10K) | < 30s | ~8s ✅ |
| Query (complex) | < 100ms | ~45ms ✅ |
| Rate Limit Check | < 10ms | ~1ms ✅ |

## 🚦 System Health Checks

```python
# Complete health check
def system_health_check():
    from src.foundation.config_manager import ConfigManager
    from src.connections.av_client import AlphaVantageClient
    from sqlalchemy import create_engine, text
    
    print("=== System Health Check ===\n")
    
    # 1. Config
    try:
        config = ConfigManager()
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    # 2. Database
    try:
        engine = create_engine(config.database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM av_realtime_options"))
            count = result.scalar()
            print(f"✅ Database connected ({count} records)")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # 3. API Client
    try:
        client = AlphaVantageClient()
        stats = client.get_rate_limit_status()
        print(f"✅ API client ready ({stats['tokens_available']} tokens)")
    except Exception as e:
        print(f"❌ API client error: {e}")
        return False
    
    print("\n✅ All systems operational!")
    return True

# Run health check
system_health_check()
```

---

**Quick Help**
- Virtual env not active? → `source venv/bin/activate`
- Database not connecting? → Check PostgreSQL: `brew services list`
- Greeks missing? → Check `require_greeks: "true"` in config
- Rate limited? → Wait for tokens: 10 per second refill

**Current Phase:** 2 Complete ✅  
**Next Phase:** 3 - IBKR Integration  
**Timeline:** Day 10 of 106 (On Schedule)