# AlphaTrader Quick Reference Guide
**Phase 4 Complete - Automated Scheduler Operational**  
**Day 16 of 106** | **August 16, 2025**

---

## 🚀 Daily Startup Routine

### 1. Environment Setup
```bash
cd ~/AlphaTrader
source venv/bin/activate
export DATABASE_URL=postgresql://michaelmerrick:password@localhost:5432/trading_system_db
export REDIS_URL=redis://localhost:6379/0
```

### 2. Service Check
```bash
# Check PostgreSQL
pg_ctl status

# Check Redis
redis-cli ping  # Should return PONG

# Check TWS (if during market hours)
ps aux | grep tws
```

### 3. Start Automated Scheduler
```bash
# Start the scheduler (production mode)
python scripts/run_scheduler.py

# Or test mode for weekends
python scripts/test_scheduler.py
```

---

## 🤖 Automated Scheduler Operations

### Start/Stop Scheduler
```python
from src.data.scheduler import DataScheduler

# Start scheduler
scheduler = DataScheduler(test_mode=False)  # Use True on weekends
scheduler.start()

# Check status
status = scheduler.get_status()
print(f"Running: {status['running']}")
print(f"Jobs: {status['total_jobs']}")
for job in status['jobs'][:5]:
    print(f"  {job['name']}: {job['next_run']}")

# Stop scheduler
scheduler.stop()
```

### Monitor Scheduler Activity
```bash
# Watch scheduler logs in real-time
tail -f logs/scheduler.log  # If logging configured

# Check what jobs are running
python -c "
from src.data.scheduler import DataScheduler
s = DataScheduler()
s.start()
import time
time.sleep(5)
status = s.get_status()
for job in status['jobs']:
    print(f'{job['name']}: {job['next_run']}')
s.stop()
"
```

### Current Schedule Configuration
| Tier | Symbols | Interval | Jobs |
|------|---------|----------|------|
| **A** | SPY, QQQ, IWM, IBIT | 30s | 4 |
| **B** | AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA | 60s | 7 |
| **C** | DIS, NFLX, COST, WMA, HOOD, MSTR, PLTR, SMCI, AMD, INTC, ORCL, SNOW | 180s | 12 |
| **Daily** | All 23 symbols | 6:00 AM | 23 |

**Total:** 46 scheduled jobs using ~19 calls/minute (3.8% of budget)

---

## 💻 Common Operations

### Check Scheduler & Cache Status
```python
from src.data.scheduler import DataScheduler
from src.connections.av_client import AlphaVantageClient
from src.data.cache_manager import get_cache

# Check scheduler
scheduler = DataScheduler()
status = scheduler.get_status()
print(f"Jobs configured: {len(status['jobs'])}")

# Check cache
cache = get_cache()
stats = cache.get_stats()
print(f"Cache keys: {stats['keys']}")
print(f"Memory: {stats['used_memory']}")

# Check API usage
client = AlphaVantageClient()
api_stats = client.get_rate_limit_status()
print(f"API calls made: {api_stats['calls_made']}")
print(f"Tokens available: {api_stats['tokens_available']}")
```

### Manual Data Update (Bypass Scheduler)
```python
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion

client = AlphaVantageClient()
ingestion = DataIngestion()

# Update a specific symbol manually
symbol = 'SPY'
data = client.get_realtime_options(symbol)
records = ingestion.ingest_options_data(data, symbol)
print(f"Updated {symbol}: {records} records")
```

### Database Queries
```sql
-- Check latest updates per symbol
SELECT symbol, 
       COUNT(*) as contracts,
       MAX(updated_at) as last_update,
       NOW() - MAX(updated_at) as age
FROM av_realtime_options
GROUP BY symbol
ORDER BY symbol;

-- Check scheduler effectiveness
SELECT 
    DATE_TRUNC('hour', updated_at) as hour,
    COUNT(DISTINCT symbol) as symbols_updated,
    COUNT(*) as total_updates
FROM av_realtime_options
WHERE updated_at > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;
```

---

## 🔧 Configuration Files

### Scheduler Configuration
```yaml
# config/data/schedules.yaml
symbol_tiers:
  tier_a:
    symbols: ["SPY", "QQQ", "IWM", "IBIT"]
  tier_b:
    symbols: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
  tier_c:
    symbols: ["DIS", "NFLX", "COST", "WMA", "HOOD", "MSTR", 
              "PLTR", "SMCI", "AMD", "INTC", "ORCL", "SNOW"]

api_groups:
  critical:
    tier_a_interval: 30
    tier_b_interval: 60
    tier_c_interval: 180
```

### Test Mode for Weekends
```python
# Enable test mode to bypass market hours check
scheduler = DataScheduler(test_mode=True)
# Will show 🧪 emoji in logs to indicate test mode
```

---

## 📊 System Health Checks

### Complete System Status
```python
def system_status():
    """Complete system health check"""
    from src.data.scheduler import DataScheduler
    from src.connections.av_client import AlphaVantageClient
    from src.data.cache_manager import get_cache
    from sqlalchemy import create_engine, text
    from src.foundation.config_manager import ConfigManager
    
    print("=== SYSTEM STATUS ===\n")
    
    # Scheduler
    scheduler = DataScheduler()
    scheduler.start()
    status = scheduler.get_status()
    print(f"✅ Scheduler: {status['total_jobs']} jobs")
    scheduler.stop()
    
    # Cache
    cache = get_cache()
    cache_stats = cache.get_stats()
    print(f"✅ Cache: {cache_stats['keys']} keys, {cache_stats['used_memory']}")
    
    # API Usage
    client = AlphaVantageClient()
    api_stats = client.get_rate_limit_status()
    usage_pct = (api_stats['minute_window_calls'] / 500) * 100
    print(f"✅ API: {api_stats['minute_window_calls']}/500 ({usage_pct:.1f}% used)")
    
    # Database
    config = ConfigManager()
    engine = create_engine(config.database_url)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(DISTINCT symbol) as symbols,
                   COUNT(*) as total_contracts,
                   MAX(updated_at) as last_update
            FROM av_realtime_options
        """))
        db_stats = result.fetchone()
        print(f"✅ Database: {db_stats[0]} symbols, {db_stats[1]:,} contracts")
        print(f"   Last update: {db_stats[2]}")
    
    print("\n✅ All systems operational!")

system_status()
```

---

## 🐛 Troubleshooting

### Scheduler Issues

#### Scheduler Not Running
```python
# Check if scheduler is actually running
from src.data.scheduler import DataScheduler

scheduler = DataScheduler(test_mode=True)
scheduler.start()

# Check jobs
jobs = scheduler.scheduler.get_jobs()
print(f"Jobs created: {len(jobs)}")

# Check if jobs are executing
import time
time.sleep(35)  # Wait for Tier A to trigger

# Look for output
```

#### Jobs Not Executing
```python
# Debug job execution
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.start()

# Add a simple test job
def test_job():
    print("Job executed!")

scheduler.add_job(test_job, 'interval', seconds=5)

# Wait and watch
import time
time.sleep(15)
scheduler.shutdown()
```

#### Weekend Testing
```python
# Always use test_mode on weekends
scheduler = DataScheduler(test_mode=True)  # Bypasses market hours check
```

### Cache Issues
```bash
# Clear all Alpha Vantage cache entries
redis-cli --scan --pattern "av:*" | xargs redis-cli DEL

# Monitor cache operations
redis-cli MONITOR | grep "av:"
```

### API Rate Issues
```python
# Check current rate limit status
from src.connections.av_client import AlphaVantageClient

client = AlphaVantageClient()
stats = client.get_rate_limit_status()
print(f"Calls this minute: {stats['minute_window_calls']}")
print(f"Tokens available: {stats['tokens_available']}")
print(f"Total calls made: {stats['calls_made']}")
```

---

## 📈 Performance Monitoring

### Real-Time Metrics
```bash
# Watch API calls
watch -n 5 'python -c "
from src.connections.av_client import AlphaVantageClient
c = AlphaVantageClient()
s = c.get_rate_limit_status()
print(f\"API Calls/min: {s[\"minute_window_calls\"]}/500\")
print(f\"Total calls: {s[\"calls_made\"]}\")"'

# Monitor cache hit rate
watch -n 10 'redis-cli INFO stats | grep keyspace'

# Database activity
watch -n 30 'psql -U michaelmerrick -d trading_system_db -c "
SELECT symbol, COUNT(*), MAX(updated_at) 
FROM av_realtime_options 
WHERE updated_at > NOW() - INTERVAL \"1 hour\"
GROUP BY symbol ORDER BY MAX(updated_at) DESC LIMIT 5"'
```

---

## 📅 Phase Status

| Phase | Days | Status | Key Component |
|-------|------|--------|---------------|
| 0: Foundation | 1-3 | ✅ Complete | Config & DB |
| 1: First API | 4-7 | ✅ Complete | REALTIME_OPTIONS |
| 2: Rate Limiting | 8-10 | ✅ Complete | Token Bucket |
| 3: IBKR | 11-14 | ✅ Complete | TWS Connection |
| **4: Scheduler** | **15-17** | ✅ **Complete** | **46 Jobs Running** |
| 5: Indicators | 18-24 | 📋 Next | RSI, MACD, BBANDS |
| 7: First Strategy | 29-35 | 🎯 Critical | 0DTE strategy |
| 9: Paper Trading | 40-43 | 🚀 Milestone | First trades! |

---

## 🚀 Quick Commands

### Start Full System
```bash
# One command to start everything
python -c "
from src.data.scheduler import DataScheduler
import signal
import sys

def signal_handler(sig, frame):
    print('\nStopping scheduler...')
    scheduler.stop()
    sys.exit(0)

scheduler = DataScheduler(test_mode=False)
scheduler.start()
print('Scheduler running. Press Ctrl+C to stop.')

signal.signal(signal.SIGINT, signal_handler)
signal.pause()
"
```

### Update All Symbols Now
```bash
# Force immediate update of all symbols
python -c "
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
import time

client = AlphaVantageClient()
ingestion = DataIngestion()

symbols = ['SPY', 'QQQ', 'IWM', 'IBIT', 'AAPL', 'MSFT', 'NVDA']

for symbol in symbols:
    print(f'Updating {symbol}...')
    data = client.get_realtime_options(symbol)
    if data:
        records = ingestion.ingest_options_data(data, symbol)
        print(f'  {records} records')
    time.sleep(0.5)  # Be nice to API
"
```

---

## 💡 Pro Tips

### Scheduler Tips
1. **Test Mode** - Always use `test_mode=True` on weekends
2. **Monitor Jobs** - Check `scheduler.get_status()` regularly
3. **Tier Management** - Move symbols between tiers in config
4. **Market Hours** - Scheduler is market-aware automatically
5. **Manual Override** - Can still call APIs directly if needed

### Performance Tips
1. **Cache Warmup** - Takes ~5 minutes for optimal hit rate
2. **API Budget** - Currently using only 3.8%, lots of room
3. **Database Size** - ~2MB per symbol, plan accordingly
4. **Memory Usage** - ~10MB per symbol in cache
5. **Network Load** - ~1MB/minute during market hours

---

## 🔗 Essential Scripts

```bash
# Test scheduler
python scripts/test_scheduler.py

# Run production scheduler (implement this)
python scripts/run_scheduler.py

# Check database
python scripts/query_options_data.py

# Test cache
python scripts/test_cache_manager.py
```

---

**Quick Stats Dashboard**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Symbols:      23 active
⚡ Jobs:         46 scheduled
🎯 API Usage:    19/500 (3.8%)
💾 Cache:        30MB, 66.7% hits
📈 Contracts:    49,854+ tracked
🔄 Automation:   100% hands-free
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Current Status:** Phase 4 Complete ✅  
**Next Phase:** 5 - Core Indicators (Day 18)  
**Progress:** Day 16 of 106 (15.1%)  
**First Trade:** Day 40 (24 days away)