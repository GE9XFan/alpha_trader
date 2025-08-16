# AlphaTrader Quick Reference Guide
**Phase 4.1 Complete - Cache Layer Operational**  
**Day 15 of 106** | **August 16, 2025**

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
# or
brew services list | grep postgresql

# Check Redis (NEW)
redis-cli ping  # Should return PONG
redis-cli INFO memory | head -5

# Check TWS (if during market hours)
ps aux | grep tws
```

### 3. Quick System Health Check
```bash
# Full system check with cache
python -c "
from src.data.cache_manager import get_cache
from src.connections.av_client import AlphaVantageClient
cache = get_cache()
client = AlphaVantageClient()
print('Cache Stats:', cache.get_stats())
print('Rate Limit:', client.get_rate_limit_status())
"
```

---

## 💻 Common Operations

### Redis Cache Operations (NEW)
```bash
# Check cache status
redis-cli INFO stats | grep instantaneous_ops

# Monitor cache activity in real-time
redis-cli MONITOR  # Ctrl+C to stop

# View all Alpha Vantage cache keys
redis-cli KEYS "av:*"

# Check specific key TTL
redis-cli TTL "av:realtime_options:SPY"

# Clear all cache
redis-cli FLUSHDB

# Get cache memory usage
redis-cli INFO memory | grep used_memory_human
```

### Alpha Vantage with Cache (UPDATED)
```bash
# Fetch with caching (30.6x faster on second call!)
python -c "
from src.connections.av_client import AlphaVantageClient
import time
client = AlphaVantageClient()

# First call - hits API
start = time.time()
data = client.get_realtime_options('SPY')
print(f'API call: {time.time()-start:.2f}s')

# Second call - hits cache
start = time.time()
data = client.get_realtime_options('SPY')
print(f'Cache call: {time.time()-start:.2f}s')

# Show cache stats
print('Cache:', client.get_cache_status())
"
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

### Database Queries with Cache Impact
```bash
# Connect to database
psql -U michaelmerrick -d trading_system_db

# Quick data summary
SELECT 
    'Options' as type, COUNT(*) as records, MAX(updated_at) as latest
FROM av_realtime_options
UNION ALL
SELECT 
    'Historical', COUNT(*), MAX(created_at)
FROM av_historical_options
UNION ALL
SELECT 
    'IBKR Bars', COUNT(*), MAX(timestamp)
FROM ibkr_bars_5sec
WHERE timestamp > NOW() - INTERVAL '1 hour';

# High volume options today
SELECT contract_id, strike, option_type, volume, delta, implied_volatility
FROM av_realtime_options
WHERE symbol = 'SPY' AND volume > 100000
ORDER BY volume DESC LIMIT 5;
```

---

## 🔧 Code Snippets

### Complete Data Fetch with Cache (NEW)
```python
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.data.cache_manager import get_cache

client = AlphaVantageClient()
ingestion = DataIngestion()
cache = get_cache()

# This flow now includes automatic caching!
# 1. Check cache first
# 2. Hit API only if cache miss
# 3. Store in DB
# 4. Update cache

data = client.get_realtime_options('SPY')  # Fast if cached!
records = ingestion.ingest_options_data(data, 'SPY')  # Also updates cache
print(f"Processed {records} records")
print(f"Cache stats: {cache.get_stats()}")
```

### Cache Management Utilities (NEW)
```python
from src.data.cache_manager import get_cache

cache = get_cache()

# Check what's cached
def show_cache_contents():
    stats = cache.get_stats()
    print(f"Total keys: {stats['keys']}")
    print(f"Memory: {stats['used_memory']}")
    
    # Check specific symbols
    for symbol in ['SPY', 'QQQ', 'IWM']:
        key = f"av:realtime_options:{symbol}"
        if cache.exists(key):
            ttl = cache.get_ttl(key)
            print(f"{symbol}: Cached (TTL: {ttl}s)")
        else:
            print(f"{symbol}: Not cached")

# Clear stale data
def clear_options_cache():
    deleted = cache.flush_pattern("av:realtime_options:*")
    print(f"Cleared {deleted} options cache entries")

show_cache_contents()
```

### Performance Comparison (NEW)
```python
import time
from src.connections.av_client import AlphaVantageClient

def benchmark_cache():
    client = AlphaVantageClient()
    results = {}
    
    # Test each symbol twice
    for symbol in ['SPY', 'QQQ', 'IWM']:
        # First call (API)
        start = time.time()
        data1 = client.get_realtime_options(symbol)
        api_time = time.time() - start
        
        # Second call (Cache)
        start = time.time()
        data2 = client.get_realtime_options(symbol)
        cache_time = time.time() - start
        
        results[symbol] = {
            'api_time': api_time,
            'cache_time': cache_time,
            'speedup': api_time / cache_time,
            'contracts': len(data1.get('data', []))
        }
    
    # Display results
    for symbol, metrics in results.items():
        print(f"\n{symbol}:")
        print(f"  API: {metrics['api_time']:.3f}s")
        print(f"  Cache: {metrics['cache_time']:.3f}s")
        print(f"  Speedup: {metrics['speedup']:.1f}x")
        print(f"  Contracts: {metrics['contracts']}")

benchmark_cache()
```

---

## 📊 System Health Checks

### Complete System Check with Cache (UPDATED)
```python
def full_system_check():
    from src.foundation.config_manager import ConfigManager
    from src.connections.av_client import AlphaVantageClient
    from src.connections.ibkr_connection import IBKRConnectionManager
    from src.data.cache_manager import get_cache
    from sqlalchemy import create_engine, text
    
    print("=== System Health Check ===\n")
    
    # 1. Configuration
    try:
        config = ConfigManager()
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    # 2. Redis Cache (NEW)
    try:
        cache = get_cache()
        stats = cache.get_stats()
        print(f"✅ Redis Cache: {stats['keys']} keys, {stats['used_memory']}")
    except Exception as e:
        print(f"❌ Cache error: {e}")
    
    # 3. Database
    try:
        engine = create_engine(config.database_url)
        with engine.connect() as conn:
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
    
    # 4. Alpha Vantage with Cache
    try:
        av_client = AlphaVantageClient()
        rate_stats = av_client.get_rate_limit_status()
        cache_stats = av_client.get_cache_status()
        print(f"✅ Alpha Vantage: {rate_stats['tokens_available']:.0f}/20 tokens")
        print(f"✅ AV Cache: {cache_stats['av_keys']} cached endpoints")
    except Exception as e:
        print(f"❌ AV client error: {e}")
    
    # 5. IBKR
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

### Key Locations (UPDATED)
```
config/.env                       # API keys & credentials + REDIS_URL
config/apis/alpha_vantage.yaml   # AV settings & rate limits
config/system/redis.yaml         # Cache TTL settings (NEW)
```

### Redis Configuration (NEW)
```yaml
# config/system/redis.yaml
connection:
  host: localhost
  port: 6379
  db: 0
  decode_responses: true

cache_ttl:
  realtime_options: 30      # 30 seconds for Greeks
  historical_options: 86400  # 24 hours
  api_responses: 300        # 5 minutes general
  
pool:
  max_connections: 10
```

### Environment Variables (UPDATED)
```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_system_db
AV_API_KEY=your_alpha_vantage_key
REDIS_URL=redis://localhost:6379/0  # NEW
IBKR_HOST=127.0.0.1
IBKR_PORT=7497    # 7497=paper, 7496=live
IBKR_CLIENT_ID=1
```

---

## 🐛 Troubleshooting

### Redis/Cache Issues (NEW)

#### Redis Not Running
```bash
# Check if Redis is running
redis-cli ping
# If "Could not connect", start it:
brew services start redis

# Check Redis logs
brew services info redis
```

#### Cache Not Working
```python
# Debug cache operations
from src.data.cache_manager import get_cache

cache = get_cache()

# Test basic operations
cache.set("test", "value", ttl=5)
print(cache.get("test"))  # Should print "value"
time.sleep(6)
print(cache.get("test"))  # Should print None (expired)

# Check Redis connection
print(cache.redis_client.ping())  # Should be True
```

#### Cache Hit Rate Low
```python
# Analyze cache performance
from src.connections.av_client import AlphaVantageClient

client = AlphaVantageClient()

# Force cache refresh
cache = client.cache
cache.flush_pattern("av:*")

# Make calls and check timing
import time
times = []
for i in range(3):
    start = time.time()
    client.get_realtime_options('SPY')
    times.append(time.time() - start)
    
print(f"Call times: {times}")
print(f"First call (API): {times[0]:.3f}s")
print(f"Cached calls: {[f'{t:.3f}s' for t in times[1:]]}")
```

### Database Issues
```bash
# Check data freshness with cache impact
psql -U michaelmerrick -d trading_system_db -c "
SELECT 
    source,
    COUNT(*) as records,
    MAX(updated_at) as last_update,
    NOW() - MAX(updated_at) as age
FROM (
    SELECT 'realtime' as source, updated_at FROM av_realtime_options
    UNION ALL
    SELECT 'historical', created_at FROM av_historical_options
) t
GROUP BY source;"
```

---

## 📁 Test Scripts Reference

### Phase 4.1 Tests (NEW)
```bash
# Cache-specific tests
python scripts/test_cache_manager.py       # Basic cache operations
python scripts/test_cached_av_client.py    # API client with caching
python scripts/test_cache_integration.py   # Full pipeline test

# Run all Phase 4.1 tests
for test in test_cache_manager test_cached_av_client test_cache_integration; do
    echo "Running $test..."
    python scripts/$test.py
    echo "---"
done
```

### Complete Test Suite
```bash
# All phases in order
python scripts/test_phase0.py              # Foundation
python scripts/test_phase2_complete.py     # Alpha Vantage
python scripts/test_ibkr_connection.py     # IBKR setup
python scripts/test_cache_integration.py   # Cache layer
```

---

## 📈 Performance Quick Stats

### Cache Performance (NEW)
| Metric | Value | Impact |
|--------|-------|--------|
| Speed Improvement | 30.6x | 1.01s → 0.03s |
| API Calls Saved | 50% | With 2 symbols |
| Cache Hit Rate | 66.7% | Excellent |
| Memory Usage | 8.48MB | Very efficient |
| TTL (Options) | 30s | Perfect for Greeks |
| TTL (Historical) | 24h | Daily refresh |

### System Performance
| Operation | Before Cache | With Cache | Status |
|-----------|--------------|------------|--------|
| SPY Options Fetch | 1.01s | 0.03s | ✅ 30x faster |
| Database Query | 45ms | 45ms | ✅ No change |
| Rate Limit Check | 1ms | 1ms | ✅ Fast |
| Memory Total | 200MB | 208MB | ✅ +8MB only |

---

## 🔗 Quick Commands

### Data Collection with Cache
```bash
# Quick options update (now cached!)
python -c "
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
c = AlphaVantageClient()
i = DataIngestion()
for symbol in ['SPY', 'QQQ']:
    data = c.get_realtime_options(symbol)
    records = i.ingest_options_data(data, symbol)
    print(f'{symbol}: {records} records')
"
```

### Cache Monitoring
```bash
# Watch cache in real-time
redis-cli MONITOR | grep "av:"

# Cache statistics dashboard
watch -n 1 'redis-cli INFO stats | grep -E "keyspace|ops|memory"'
```

---

## 📅 Phase Status

| Phase | Days | Status | Next Step |
|-------|------|--------|-----------|
| 0: Foundation | 1-3 | ✅ Complete | - |
| 1: First API | 4-7 | ✅ Complete | - |
| 2: Rate Limiting | 8-10 | ✅ Complete | - |
| 3: IBKR | 11-14 | ✅ Complete | - |
| **4.1: Cache** | **15** | ✅ **Complete** | - |
| **4.2: Scheduler** | **16** | 📋 **Tomorrow** | **DataScheduler class** |
| 4.3: Integration | 17 | 🔜 Sunday | 24-hour test |
| 5: Indicators | 18-24 | 🎯 Next Week | RSI, MACD, etc. |
| 7: First Strategy | 29-35 | 🎯 Critical | 0DTE strategy |
| 9: Paper Trading | 40-43 | 🚀 Milestone | First trades! |

---

## 🚀 Tomorrow's Checklist (Day 16)

```bash
# Morning Setup
cd ~/AlphaTrader
source venv/bin/activate

# Verify services
redis-cli ping
pg_ctl status

# Start Day 16 - Scheduler
# 1. Create src/data/scheduler.py
# 2. Create config/data/schedules.yaml
# 3. Implement DataScheduler class
# 4. Test with live data
# 5. Verify cache integration

# Key Goals:
# - Automated API calls every 30 seconds
# - Market hours awareness
# - Zero manual intervention
```

---

## 💡 Pro Tips

### Cache-Specific Tips (NEW)
1. **Monitor TTL** - Use `redis-cli TTL key` to check expiration
2. **Cache Warming** - Not needed with 30s TTL
3. **Memory Limit** - Set max memory in Redis config if needed
4. **Key Patterns** - Use `av:type:symbol:date` consistently
5. **Debug Mode** - Use `redis-cli MONITOR` to see all operations

### General Tips
1. **Market Hours Testing** - Best 10 AM - 3 PM ET
2. **Rate Limiting** - Stay under 500/min for safety
3. **Cache Before API** - Always check cache first
4. **Database Cleanup** - Archive old bars weekly
5. **Error Logs** - Check TWS logs at `~/Jts/`

---

**Quick Help**
- Redis issues? → `brew services restart redis`
- Cache miss? → Check TTL with `redis-cli TTL "av:realtime_options:SPY"`
- Slow performance? → Verify cache with benchmark_cache()
- Memory concerns? → `redis-cli INFO memory`

**Current Status:** Phase 4.1 Complete ✅  
**Next Phase:** 4.2 - Scheduler (Day 16)  
**Progress:** Day 15 of 106 (14.2%)  
**Cache Performance:** 30.6x faster! 🚀  
**First Trade:** Day 40 (25 days away)