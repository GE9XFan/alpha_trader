# AlphaTrader Quick Reference Guide
**Phase 3 Complete - IBKR Integration Active**

---

## 🚀 Daily Startup Routine

### 1. Environment Setup
```bash
cd ~/AlphaTrader
source venv/bin/activate
export DATABASE_URL=postgresql://michaelmerrick:password@localhost:5432/trading_system_db
```

### 2. Service Check
```bash
# Check PostgreSQL
pg_ctl status
# or
brew services list | grep postgresql

# Check TWS is running (if during market hours)
ps aux | grep tws

# Check Redis (Phase 4)
redis-cli ping  # Should return PONG
```

### 3. Market Hours Quick Test
```bash
# During market hours (9:30 AM - 4:00 PM ET)
python scripts/test_ibkr_live_data.py
```

---

## 💻 Common Operations

### Alpha Vantage Operations
```bash
# Fetch fresh options data
python -c "from src.connections.av_client import AlphaVantageClient; client = AlphaVantageClient(); data = client.get_realtime_options('SPY'); print(f'Got {len(data[\"data\"])} contracts')"

# Check rate limit status
python -c "from src.connections.av_client import AlphaVantageClient; client = AlphaVantageClient(); print(client.get_rate_limit_status())"

# Quick options update for SPY
python scripts/test_realtime_options.py
```

### IBKR Operations
```bash
# Test TWS connection
python scripts/test_ibkr_connection.py

# Test market data (bars + quotes)
python scripts/test_ibkr_market_data.py

# Full live data test (best during market hours)
python scripts/test_ibkr_live_data.py
```

### Database Queries
```bash
# Connect to database
psql -U michaelmerrick -d trading_system_db

# Quick counts
SELECT COUNT(*) FROM av_realtime_options;
SELECT COUNT(*) FROM ibkr_bars_5sec WHERE timestamp > NOW() - INTERVAL '1 hour';
SELECT COUNT(*) FROM ibkr_quotes WHERE timestamp > NOW() - INTERVAL '1 hour';

# Latest market data
SELECT * FROM ibkr_bars_5sec ORDER BY timestamp DESC LIMIT 5;
SELECT * FROM ibkr_quotes ORDER BY timestamp DESC LIMIT 5;

# Options summary
SELECT symbol, COUNT(*), MIN(strike), MAX(strike), COUNT(DISTINCT expiration)
FROM av_realtime_options 
GROUP BY symbol;
```

---

## 🔧 Code Snippets

### Complete Data Fetch & Store
```python
# Alpha Vantage Options
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion

client = AlphaVantageClient()
ingestion = DataIngestion()

# Fetch and store options
data = client.get_realtime_options('SPY')
records = ingestion.ingest_options_data(data, 'SPY')
print(f"Stored {records} option contracts")
```

### IBKR Real-Time Stream
```python
# IBKR Real-time data
from src.connections.ibkr_connection import IBKRConnectionManager
import time

ibkr = IBKRConnectionManager()

# Connect
if ibkr.connect_tws():
    # Subscribe to data
    ibkr.get_quotes('SPY')  # Quotes
    ibkr.subscribe_bars('SPY', '5 secs')  # Bars
    
    # Let it run for 30 seconds
    time.sleep(30)
    
    # Disconnect
    ibkr.disconnect_tws()
```

### Query Latest Market Data
```python
from sqlalchemy import create_engine, text
from src.foundation.config_manager import ConfigManager

config = ConfigManager()
engine = create_engine(config.database_url)

with engine.connect() as conn:
    # Get latest bars
    result = conn.execute(text("""
        SELECT timestamp, symbol, open, high, low, close, volume, vwap
        FROM ibkr_bars_5sec
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
        ORDER BY timestamp DESC
        LIMIT 10
    """))
    
    for row in result:
        print(f"{row.timestamp}: {row.symbol} C={row.close} V={row.volume}")
```

### Near-The-Money Options with Greeks
```python
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT contract_id, strike, option_type, 
               last_price, delta, gamma, theta, implied_volatility
        FROM av_realtime_options
        WHERE symbol = 'SPY'
          AND strike BETWEEN 640 AND 650
          AND expiration = (SELECT MIN(expiration) FROM av_realtime_options)
        ORDER BY strike, option_type
    """))
    
    for row in result:
        print(f"{row.strike} {row.option_type}: Δ={row.delta:.3f}, θ={row.theta:.3f}")
```

---

## 📊 System Health Checks

### Complete System Check
```python
def full_system_check():
    from src.foundation.config_manager import ConfigManager
    from src.connections.av_client import AlphaVantageClient
    from src.connections.ibkr_connection import IBKRConnectionManager
    from sqlalchemy import create_engine, text
    
    print("=== System Health Check ===\n")
    
    # 1. Configuration
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
            # Check all tables
            result = conn.execute(text("""
                SELECT COUNT(*) as options, 
                       (SELECT COUNT(*) FROM ibkr_bars_5sec) as bars,
                       (SELECT COUNT(*) FROM ibkr_quotes) as quotes
                FROM av_realtime_options
            """))
            counts = result.fetchone()
            print(f"✅ Database: {counts.options} options, {counts.bars} bars, {counts.quotes} quotes")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # 3. Alpha Vantage
    try:
        av_client = AlphaVantageClient()
        stats = av_client.get_rate_limit_status()
        print(f"✅ Alpha Vantage: {stats['tokens_available']:.0f}/20 tokens available")
    except Exception as e:
        print(f"❌ AV client error: {e}")
    
    # 4. IBKR
    try:
        ibkr = IBKRConnectionManager()
        if ibkr.connect_tws():
            print("✅ IBKR TWS connected")
            ibkr.disconnect_tws()
        else:
            print("⚠️  IBKR not connected (TWS may be down)")
    except Exception as e:
        print(f"⚠️  IBKR error: {e}")
    
    print("\n✅ System check complete!")
    return True

# Run it
full_system_check()
```

---

## 🛠️ Configuration Files

### Key Locations
```
config/.env                       # API keys & credentials
config/apis/alpha_vantage.yaml   # AV settings & rate limits
```

### Critical Settings
```yaml
# Alpha Vantage (config/apis/alpha_vantage.yaml)
endpoints:
  realtime_options:
    require_greeks: "true"    # MUST be true for Greeks!

rate_limit:
  max_per_minute: 600
  target_per_minute: 500
```

### Environment Variables
```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_system_db
AV_API_KEY=your_alpha_vantage_key
IBKR_HOST=127.0.0.1
IBKR_PORT=7497    # 7497=paper, 7496=live
IBKR_CLIENT_ID=1
```

---

## 🐛 Troubleshooting

### IBKR Issues

#### TWS Not Connecting
```bash
# 1. Check TWS is running
ps aux | grep tws

# 2. Verify API settings in TWS:
#    File → Global Configuration → API → Settings
#    - Enable ActiveX and Socket Clients ✓
#    - Allow connections from localhost only ✓
#    - Socket port: 7497 (paper) or 7496 (live)

# 3. Test connection
python scripts/test_ibkr_connection.py

# 4. If still failing, restart TWS
```

#### No Market Data from IBKR
```bash
# Check if market is open
python -c "
from datetime import datetime
now = datetime.now()
if now.weekday() >= 5:
    print('Weekend - market closed')
elif 9 <= now.hour < 16:
    if now.hour == 9 and now.minute < 30:
        print('Pre-market')
    else:
        print('Market OPEN')
else:
    print('After hours')
"

# Check data subscriptions
python scripts/test_ibkr_market_data.py
```

### Database Issues

#### Connection Problems
```bash
# Check PostgreSQL running
brew services list | grep postgresql

# Start if needed
brew services start postgresql@14

# Test connection
psql -U michaelmerrick -d trading_system_db -c "SELECT 1;"
```

#### Check Data Freshness
```sql
-- Most recent options update
SELECT MAX(updated_at) FROM av_realtime_options;

-- Recent IBKR data
SELECT COUNT(*), MAX(timestamp) 
FROM ibkr_bars_5sec 
WHERE timestamp > NOW() - INTERVAL '1 hour';

-- Data by symbol
SELECT symbol, COUNT(*), MAX(timestamp) as latest
FROM ibkr_bars_5sec
GROUP BY symbol
ORDER BY latest DESC;
```

### Alpha Vantage Issues

#### Greeks Missing
```python
# Check config has require_greeks
from src.foundation.config_manager import ConfigManager
config = ConfigManager()
print(config.av_config['endpoints']['realtime_options'])
# Should show: {'function': 'REALTIME_OPTIONS', 'require_greeks': 'true', ...}
```

#### Rate Limit Hit
```python
# Check current status
from src.connections.av_client import AlphaVantageClient
client = AlphaVantageClient()
stats = client.get_rate_limit_status()
print(f"Tokens: {stats['tokens_available']}/20")
print(f"This minute: {stats['minute_window_calls']}/600")
# Wait if depleted (refills at 10 tokens/second)
```

---

## 📁 Project Structure Reference

```
AlphaTrader/
├── src/
│   ├── foundation/
│   │   └── config_manager.py      # Loads .env and YAML
│   ├── connections/
│   │   ├── av_client.py           # Alpha Vantage client
│   │   └── ibkr_connection.py     # IBKR TWS connection
│   └── data/
│       ├── ingestion.py           # Unified data storage
│       └── rate_limiter.py        # Token bucket (600/min)
├── config/
│   ├── .env                       # Your credentials
│   └── apis/
│       └── alpha_vantage.yaml     # API configuration
├── scripts/
│   ├── test_ibkr_*.py            # IBKR test scripts
│   └── test_phase*.py            # Phase validation
└── data/
    └── api_responses/             # Saved JSON responses
```

---

## 📈 Performance Benchmarks

| Operation | Target | Current | Status |
|-----------|--------|---------|--------|
| AV API Call | < 2s | ~1.3s | ✅ |
| IBKR Connect | < 5s | ~2s | ✅ |
| DB Insert (10K) | < 30s | ~8s | ✅ |
| Query Complex | < 100ms | ~45ms | ✅ |
| Rate Limit Check | < 10ms | ~1ms | ✅ |
| Bar Latency | < 500ms | ~100ms | ✅ |

---

## 🎯 Current Capabilities

| Feature | Status | Details |
|---------|--------|---------|
| **Alpha Vantage Options** | ✅ Active | 18,588 contracts with Greeks |
| **IBKR Real-time Bars** | ✅ Ready | 5-second bars (needs market open) |
| **IBKR Quotes** | ✅ Ready | Bid/ask/last streaming |
| **Rate Limiting** | ✅ Active | 600/min protection |
| **Data Ingestion** | ✅ Active | Unified storage layer |
| **Database** | ✅ Active | 8 tables, optimized |

---

## 🔗 Quick Links

### Documentation
- [README.md](README.md) - Project overview
- [SSOT-Ops.md](docs/SSOT-Ops.md) - Operational specs
- [SSOT-Tech.md](docs/SSOT-Tech.md) - Technical details
- [Phased Plan](docs/granular-phased-plan.md) - Complete roadmap

### Key Scripts
```bash
# Phase validation
scripts/test_phase0.py          # Foundation check
scripts/test_phase2_complete.py # AV integration
scripts/test_ibkr_connection.py # IBKR connection

# Live testing
scripts/test_ibkr_live_data.py  # Full market test
scripts/query_options_data.py   # Data analysis
```

---

## 📅 Phase Status

| Phase | Days | Status | Next Step |
|-------|------|--------|-----------|
| 0: Foundation | 1-3 | ✅ Complete | - |
| 1: First API | 4-7 | ✅ Complete | - |
| 2: Rate Limiting | 8-10 | ✅ Complete | - |
| 3: IBKR | 11-14 | ✅ Complete | - |
| **4: Scheduler** | **15-17** | **📋 Next** | **Install Redis** |
| 5: Indicators | 18-24 | 🔜 Coming | RSI, MACD, etc. |
| 7: First Strategy | 29-35 | 🎯 Critical | 0DTE strategy |
| 9: Paper Trading | 40-43 | 🚀 Milestone | First trades! |

---

## 🚀 Monday Checklist

```bash
# 1. Start services
brew services start postgresql@14
brew services start redis  # After installing

# 2. Activate environment
cd ~/AlphaTrader
source venv/bin/activate

# 3. Check TWS
# Open TWS, verify API enabled

# 4. Test live data (after 9:30 AM ET)
python scripts/test_ibkr_live_data.py

# 5. Verify data flowing
psql -U michaelmerrick -d trading_system_db -c "
SELECT COUNT(*), MAX(timestamp) 
FROM ibkr_bars_5sec 
WHERE timestamp > NOW() - INTERVAL '1 hour';"

# 6. Begin Phase 4
# Implement DataScheduler class
```

---

## 💡 Pro Tips

1. **Market Hours Testing** - Best results 10 AM - 3 PM ET
2. **Rate Limiting** - Stay under 500/min for safety
3. **Database Cleanup** - Archive old bars weekly
4. **TWS Settings** - Save workspace after config
5. **Error Logs** - Check TWS logs at `~/Jts/`

---

**Quick Help**
- Virtual env issues? → `deactivate` then `source venv/bin/activate`
- TWS won't connect? → Restart TWS, check port 7497
- Database full? → `psql -c "VACUUM ANALYZE;"`
- Greeks missing? → Verify `require_greeks: "true"`

**Current Status:** Phase 3 Complete ✅  
**Next Phase:** 4 - Scheduler & Cache  
**Progress:** Day 14 of 106 (13.2%)  
**First Trade:** Day 40 (26 days away)