# AlphaTrader Operations Manual v2.0
## Daily Operations Guide with Alpha Vantage Integration

---

## SYSTEM ARCHITECTURE OVERVIEW

### Data Sources
```
Alpha Vantage (600 calls/min premium):
├── Options chains with Greeks (real-time & 20 years historical)
├── Technical indicators (all 16 types)
├── News sentiment & analytics
├── Fundamental data
└── Economic indicators

Interactive Brokers:
├── Real-time quotes & bars
├── Order execution (paper & live)
└── Position management
```

---

## QUICK START CHECKLIST

### First Time Setup (45 minutes)
```bash
# 1. Create Python environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (no TA-Lib needed!)
pip install ib_insync xgboost pandas numpy scipy psycopg2-binary redis discord.py aiohttp

# 3. Create database
psql -U postgres
CREATE DATABASE alphatrader;
\q

# 4. Set up configuration with Alpha Vantage
cp config.template.yaml config.yaml
# Edit config.yaml with:
# - Alpha Vantage API key (premium 600 calls/min)
# - IBKR connection settings
# - Risk parameters

# 5. Set environment variable
export AV_API_KEY="your_premium_alpha_vantage_key"

# 6. Test Alpha Vantage connection
python scripts/test_av_connection.py

# 7. Test IBKR connection
python scripts/test_ibkr_connection.py

# Ready to go!
```

---

## DAILY OPERATIONS

### Pre-Market Routine (8:30 AM - 9:30 AM)

#### 1. System Startup (8:30 AM)
```bash
# Terminal 1: Start market data (IBKR + Alpha Vantage)
cd ~/AlphaTrader
source venv/bin/activate
export AV_API_KEY="your_key"
python scripts/start_market_data.py

# Terminal 2: Start Alpha Vantage monitor
python scripts/av_rate_monitor.py
# Shows: API calls used, remaining, cache hits

# Terminal 3: Start paper trader
python scripts/start_paper_trader.py

# Terminal 4: Start Discord bot
python scripts/start_discord_bot.py
```

#### 2. Health Checks (8:45 AM)
```python
# scripts/morning_checks.py
import asyncio
from src.data.market_data import market_data
from src.data.alpha_vantage_client import av_client
from src.data.options_data import options_data

async def morning_checks():
    print("🔍 Running morning checks...")
    
    # Check IBKR connection for quotes/execution
    await market_data.connect()
    print("✅ IBKR connected (quotes & execution)")
    
    # Check Alpha Vantage connection
    await av_client.connect()
    print(f"✅ Alpha Vantage connected (600 calls/min tier)")
    
    # Test IBKR market data
    await market_data.subscribe_symbols(['SPY'])
    await asyncio.sleep(5)
    price = market_data.get_latest_price('SPY')
    print(f"✅ IBKR SPY price: ${price:.2f}")
    
    # Test Alpha Vantage options data WITH GREEKS
    options = await av_client.get_realtime_options('SPY', require_greeks=True)
    print(f"✅ Alpha Vantage options: {len(options)} contracts with Greeks")
    
    # Verify Greeks are provided
    if options:
        sample = options[0]
        print(f"✅ Sample Greeks from AV: Δ={sample.delta:.3f}, Γ={sample.gamma:.3f}")
    
    # Check Alpha Vantage rate limit status
    print(f"✅ AV Rate limit: {av_client.rate_limiter.remaining}/600 calls remaining")
    
    # Test Alpha Vantage indicators
    rsi = await av_client.get_technical_indicator('SPY', 'RSI', interval='5min')
    print(f"✅ Alpha Vantage RSI: {rsi.iloc[0]['RSI']:.2f}")
    
    # Check database
    with db.get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = CURRENT_DATE")
        trades_today = cur.fetchone()[0]
        print(f"✅ Database: {trades_today} trades today")
        
    # Check risk limits with AV Greeks
    print(f"✅ Risk limits: {risk_manager.positions} / {risk_manager.max_positions} positions")
    print(f"✅ Portfolio Greeks (from AV):")
    for greek, value in risk_manager.portfolio_greeks.items():
        print(f"   {greek}: {value:.3f}")
    print(f"✅ Daily P&L: ${risk_manager.daily_pnl:.2f} / -${risk_manager.daily_loss_limit}")
    
    print("\n🎯 System ready for trading!")
    print("📊 Data sources: IBKR (quotes/execution) + Alpha Vantage (options/analytics)")

asyncio.run(morning_checks())
```

#### 3. Alpha Vantage API Health Check (9:00 AM)
```python
# scripts/av_api_health.py
async def check_av_endpoints():
    """Test all critical Alpha Vantage endpoints"""
    print("\n🔬 Testing Alpha Vantage APIs...")
    
    endpoints = [
        ('OPTIONS', lambda: av_client.get_realtime_options('SPY', require_greeks=True)),
        ('RSI', lambda: av_client.get_technical_indicator('SPY', 'RSI')),
        ('MACD', lambda: av_client.get_technical_indicator('SPY', 'MACD')),
        ('NEWS', lambda: av_client.get_news_sentiment(['SPY'])),
        ('ANALYTICS', lambda: av_client.get_analytics_fixed_window('SPY'))
    ]
    
    for name, func in endpoints:
        try:
            start = time.time()
            result = await func()
            elapsed = (time.time() - start) * 1000
            print(f"✅ {name}: {elapsed:.0f}ms")
        except Exception as e:
            print(f"❌ {name}: {e}")
            
    print(f"\n📊 Rate limit status: {av_client.rate_limiter.remaining}/600")
```

#### 4. Review Positions with Live Greeks (9:15 AM)
```python
# scripts/review_positions.py
async def review_positions():
    with db.get_db() as conn:
        cur = conn.cursor()
        
        # Get current positions
        cur.execute("""
            SELECT symbol, option_type, strike, expiry, quantity, 
                   price, entry_delta, entry_gamma, entry_theta, entry_vega
            FROM positions
            WHERE quantity != 0
        """)
        
        positions = cur.fetchall()
        
        print("\n📊 CURRENT POSITIONS WITH ALPHA VANTAGE GREEKS")
        print("-" * 80)
        
        total_pnl = 0
        for pos in positions:
            # Get current Greeks from Alpha Vantage
            current_greeks = options_data.get_option_greeks(
                pos['symbol'], pos['strike'], pos['expiry'], pos['option_type']
            )
            
            print(f"{pos['symbol']} {pos['option_type']} ${pos['strike']} exp:{pos['expiry']}")
            print(f"  Qty: {pos['quantity']} | Entry: ${pos['price']:.2f}")
            print(f"  Entry Greeks (AV): Δ={pos['entry_delta']:.3f}, Θ={pos['entry_theta']:.3f}")
            print(f"  Current Greeks (AV): Δ={current_greeks['delta']:.3f}, Θ={current_greeks['theta']:.3f}")
            
            # Check for 0DTE positions
            if pos['expiry'] == datetime.now().date():
                print(f"  ⚠️ EXPIRES TODAY! Θ decay: ${current_greeks['theta'] * 100:.2f}/day")
```

### Market Hours Operations (9:30 AM - 4:00 PM)

#### Enhanced Monitoring Dashboard with Alpha Vantage
```python
# scripts/monitor.py
import time
import os
from datetime import datetime

async def monitor_loop():
    while True:
        clear_screen()
        
        print("=" * 80)
        print(f"ALPHATRADER MONITOR - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        # Market Data (IBKR)
        print("\n📈 MARKET DATA (IBKR)")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            price = market_data.get_latest_price(symbol)
            print(f"{symbol}: ${price:.2f}")
            
        # Options Data (Alpha Vantage)
        print("\n📊 OPTIONS DATA (Alpha Vantage)")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            atm_options = options_data.find_atm_options(symbol)
            if atm_options:
                opt = atm_options[0]
                print(f"{symbol} ATM: Strike=${opt['strike']}, "
                      f"Greeks: Δ={opt['greeks']['delta']:.3f}, "
                      f"Θ={opt['greeks']['theta']:.3f}")
                      
        # Alpha Vantage API Status
        print(f"\n🌐 ALPHA VANTAGE STATUS")
        print(f"  API Calls Used: {600 - av_client.rate_limiter.remaining}/600")
        print(f"  Cache Hit Rate: {av_client.cache_hits / max(av_client.total_calls, 1) * 100:.1f}%")
        print(f"  Avg Response Time: {av_client.avg_response_time:.0f}ms")
        
        # Positions with AV Greeks
        print(f"\n💼 POSITIONS: {len(risk_manager.positions)} / {risk_manager.max_positions}")
        for symbol, pos in risk_manager.positions.items():
            greeks = options_data.get_option_greeks(
                symbol, pos['strike'], pos['expiry'], pos['option_type']
            )
            print(f"  {symbol}: {pos['quantity']} contracts, "
                  f"Δ={greeks['delta']:.3f}, Θ={greeks['theta']:.3f}")
            
        # Portfolio Greeks (from Alpha Vantage)
        print(f"\n🎯 PORTFOLIO GREEKS (Alpha Vantage)")
        for greek, value in risk_manager.portfolio_greeks.items():
            limit_min, limit_max = risk_manager.greeks_limits[greek]
            status = "✅" if limit_min <= value <= limit_max else "⚠️"
            print(f"  {greek}: {value:.3f} [{limit_min:.2f}, {limit_max:.2f}] {status}")
            
        # P&L
        print(f"\n💰 DAILY P&L: ${risk_manager.daily_pnl:.2f}")
        if risk_manager.daily_pnl < 0:
            pct_of_limit = abs(risk_manager.daily_pnl / risk_manager.daily_loss_limit * 100)
            print(f"  Loss: {pct_of_limit:.1f}% of daily limit")
            
        # Recent Signals with AV indicators
        print(f"\n📡 RECENT SIGNALS (with Alpha Vantage indicators)")
        recent = signal_generator.signals_today[-3:] if signal_generator.signals_today else []
        for sig in recent:
            print(f"  {sig['timestamp'].strftime('%H:%M')} - {sig['symbol']} "
                  f"{sig['signal_type']} (conf: {sig['confidence']:.2f})")
            if 'av_greeks' in sig:
                print(f"    Greeks: Δ={sig['av_greeks']['delta']:.3f}")
            
        # System Status
        print(f"\n⚙️  SYSTEM STATUS")
        print(f"  IBKR: {'🟢 Connected' if market_data.connected else '🔴 Disconnected'}")
        print(f"  Alpha Vantage: {'🟢 Online' if av_client.session else '🔴 Offline'}")
        print(f"  Discord: {'🟢 Online' if bot.is_ready() else '🔴 Offline'}")
        print(f"  Mode: {config.mode.upper()}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

asyncio.run(monitor_loop())
```

#### Manual Interventions

**Check Alpha Vantage Rate Limit:**
```python
# scripts/check_av_rate.py
def check_av_rate_limit():
    print(f"Alpha Vantage API Status:")
    print(f"  Calls remaining: {av_client.rate_limiter.remaining}/600")
    print(f"  Reset in: {av_client.rate_limiter.reset_time} seconds")
    print(f"  Cache stats:")
    print(f"    Hit rate: {av_client.cache_hit_rate:.1%}")
    print(f"    Cached items: {len(av_client.cache)}")
```

**Force Refresh Alpha Vantage Data:**
```python
# scripts/refresh_av_data.py
async def refresh_av_options(symbol: str):
    """Force refresh options data from Alpha Vantage"""
    print(f"Refreshing {symbol} options from Alpha Vantage...")
    
    # Clear cache
    av_client.cache.pop(f"options_{symbol}_{datetime.now().minute}", None)
    
    # Fetch fresh data with Greeks
    options = await av_client.get_realtime_options(symbol, require_greeks=True)
    
    print(f"Fetched {len(options)} options with Greeks")
    
    # Update options manager
    options_data.chains[symbol] = options
    
    # Update Greeks cache
    for option in options:
        key = f"{symbol}_{option.strike}_{option.expiry}_{option.option_type}"
        options_data.latest_greeks[key] = {
            'delta': option.delta,
            'gamma': option.gamma,
            'theta': option.theta,
            'vega': option.vega
        }
    
    print(f"✅ {symbol} options and Greeks updated from Alpha Vantage")
```

**Close Position with Updated Greeks:**
```python
# scripts/close_position.py
async def close_position(symbol: str):
    position = risk_manager.positions.get(symbol)
    
    if not position:
        print(f"No position in {symbol}")
        return
        
    # Get current Greeks from Alpha Vantage for logging
    current_greeks = options_data.get_option_greeks(
        symbol, position['strike'], position['expiry'], position['option_type']
    )
    
    print(f"Closing {symbol} position")
    print(f"  Entry Greeks (AV): Δ={position['entry_delta']:.3f}")
    print(f"  Current Greeks (AV): Δ={current_greeks['delta']:.3f}")
    print(f"  Greeks P&L: ΔΔ={(current_greeks['delta'] - position['entry_delta']):.3f}")
    
    # Create closing order through IBKR
    from ib_insync import Option, MarketOrder
    
    contract = Option(
        symbol,
        position['expiry'].strftime('%Y%m%d'),
        position['strike'],
        position['option_type'][0],
        'SMART'
    )
    
    order = MarketOrder('SELL', position['quantity'])
    
    if config.mode == 'live':
        trade = await market_data.execute_order(contract, order)
    else:
        print(f"Paper close: {symbol} {position['quantity']} contracts")
        
    # Update database with exit Greeks
    with db.get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE trades 
            SET exit_delta = %s, exit_gamma = %s, exit_theta = %s, exit_vega = %s
            WHERE symbol = %s AND position_open = true
        """, (current_greeks['delta'], current_greeks['gamma'], 
              current_greeks['theta'], current_greeks['vega'], symbol))
        conn.commit()
        
    # Remove from risk manager
    del risk_manager.positions[symbol]
    await risk_manager._update_portfolio_greeks_from_av()
```

### End of Day Operations (3:30 PM - 5:00 PM)

#### 3:30 PM - Pre-Close Checks with Alpha Vantage
```python
# scripts/pre_close.py
async def pre_close_checks():
    print("\n🔔 PRE-CLOSE CHECKS (3:30 PM)")
    
    # Check for 0DTE positions
    with db.get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM positions 
            WHERE expiry = CURRENT_DATE AND quantity != 0
        """)
        
        expiring = cur.fetchall()
        
        if expiring:
            print(f"\n⚠️  {len(expiring)} POSITIONS EXPIRING TODAY:")
            for pos in expiring:
                # Get current Greeks from Alpha Vantage
                greeks = options_data.get_option_greeks(
                    pos['symbol'], pos['strike'], pos['expiry'], pos['option_type']
                )
                
                print(f"  {pos['symbol']} {pos['option_type']} ${pos['strike']}")
                print(f"    Current Greeks (AV): Δ={greeks['delta']:.3f}, "
                      f"Θ={greeks['theta']:.3f}")
                print(f"    Theta decay today: ${greeks['theta'] * 100:.2f}")
                
            print("\n  These will be closed at 3:59 PM")
    
    # Check Alpha Vantage API usage
    print(f"\n📊 Alpha Vantage API Usage Today:")
    print(f"  Total calls: {av_client.total_calls_today}")
    print(f"  Cache hits: {av_client.cache_hits_today}")
    print(f"  Hit rate: {av_client.cache_hits_today / av_client.total_calls_today * 100:.1f}%")
    print(f"  Remaining: {av_client.rate_limiter.remaining}/600")
```

#### 4:15 PM - Daily Report with Alpha Vantage Analytics
```python
# scripts/daily_report.py
async def generate_daily_report():
    print("\n" + "="*80)
    print(f"DAILY REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*80)
    
    with db.get_db() as conn:
        cur = conn.cursor()
        
        # Today's trades with AV Greeks analysis
        cur.execute("""
            SELECT COUNT(*), SUM(pnl),
                   AVG(entry_delta), AVG(entry_gamma), AVG(entry_theta), AVG(entry_vega)
            FROM trades 
            WHERE DATE(timestamp) = CURRENT_DATE
        """)
        trades_count, total_pnl, avg_delta, avg_gamma, avg_theta, avg_vega = cur.fetchone()
        
        print(f"\n📊 TRADING SUMMARY")
        print(f"  Total Trades: {trades_count}")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"\n📈 AVERAGE ENTRY GREEKS (Alpha Vantage):")
        print(f"  Delta: {avg_delta:.3f}")
        print(f"  Gamma: {avg_gamma:.3f}")
        print(f"  Theta: {avg_theta:.3f}")
        print(f"  Vega: {avg_vega:.3f}")
        
        # Alpha Vantage API usage
        cur.execute("""
            SELECT endpoint, COUNT(*), AVG(response_time_ms),
                   SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::float / COUNT(*) as hit_rate
            FROM av_api_metrics
            WHERE DATE(timestamp) = CURRENT_DATE
            GROUP BY endpoint
            ORDER BY COUNT(*) DESC
        """)
        
        print(f"\n🌐 ALPHA VANTAGE API USAGE:")
        for endpoint, calls, avg_time, hit_rate in cur.fetchall():
            print(f"  {endpoint}: {calls} calls, {avg_time:.0f}ms avg, {hit_rate:.1%} cache")
        
        # Get market sentiment from Alpha Vantage
        sentiment = await av_client.get_news_sentiment(['SPY', 'QQQ', 'IWM'])
        
        if sentiment and 'feed' in sentiment:
            avg_sentiment = np.mean([
                float(article.get('overall_sentiment_score', 0))
                for article in sentiment['feed'][:20]
            ])
            print(f"\n📰 MARKET SENTIMENT (Alpha Vantage):")
            print(f"  Average sentiment score: {avg_sentiment:.3f}")
            print(f"  News articles analyzed: {len(sentiment['feed'])}")
        
        # Technical indicators summary from Alpha Vantage
        print(f"\n📊 CLOSING INDICATORS (Alpha Vantage):")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            rsi = await av_client.get_technical_indicator(symbol, 'RSI', interval='daily')
            if not rsi.empty:
                print(f"  {symbol} RSI: {rsi.iloc[0]['RSI']:.2f}")
```

---

## TROUBLESHOOTING

### Common Issues and Solutions

#### 1. Alpha Vantage Rate Limit Hit
```python
# Monitor and manage rate limits
async def handle_av_rate_limit():
    if av_client.rate_limiter.remaining < 50:
        print("⚠️ Alpha Vantage rate limit low!")
        print(f"Remaining: {av_client.rate_limiter.remaining}/600")
        
        # Increase cache TTLs temporarily
        av_client.cache_ttl_multiplier = 2.0
        
        # Wait for reset if needed
        if av_client.rate_limiter.remaining < 10:
            wait_time = av_client.rate_limiter.reset_time
            print(f"Waiting {wait_time}s for rate limit reset...")
            await asyncio.sleep(wait_time)
```

#### 2. Alpha Vantage API Error
```python
# Fallback when AV is down
async def av_fallback_mode():
    print("⚠️ Alpha Vantage unavailable, using fallback mode")
    
    # Use cached data if available
    cached_options = db.redis.keys("av_options_*")
    print(f"Found {len(cached_options)} cached option chains")
    
    # Reduce trading frequency
    signal_generator.min_time_between_signals = 600  # 10 minutes
    
    # Alert Discord
    await bot.send_alert("Alpha Vantage API issues - using cached data")
```

#### 3. Greeks Data Missing
```python
# Handle missing Greeks from Alpha Vantage
async def handle_missing_greeks(symbol: str):
    print(f"⚠️ Greeks missing for {symbol}")
    
    # Try historical endpoint
    historical = await av_client.get_historical_options(
        symbol, 
        datetime.now().strftime('%Y-%m-%d')
    )
    
    if historical and historical[0].delta is not None:
        print("✅ Retrieved Greeks from historical endpoint")
        return historical[0]
    else:
        print("❌ Greeks unavailable - skipping signal")
        return None
```

#### 4. Cache Management
```bash
# Clear Alpha Vantage cache if stale
redis-cli
> KEYS av_*
> DEL av_options_SPY_*
> exit

# Or via Python
python scripts/clear_av_cache.py --pattern "av_options_*"
```

#### 5. IBKR Connection Lost (Execution)
```python
# Reconnect IBKR while maintaining AV connection
async def reconnect_ibkr():
    print("Reconnecting to IBKR...")
    market_data.ib.disconnect()
    await asyncio.sleep(5)
    await market_data.connect()
    
    # Alpha Vantage remains connected
    print("IBKR reconnected, Alpha Vantage still online")
```

---

## WEEKLY MAINTENANCE

### Sunday Tasks (Market Closed)

#### 1. Backup Database with AV Metrics
```bash
# Include Alpha Vantage metrics in backup
pg_dump -U postgres alphatrader > backups/alphatrader_$(date +%Y%m%d).sql

# Backup Redis cache
redis-cli --rdb backups/redis_av_cache_$(date +%Y%m%d).rdb
```

#### 2. Analyze Alpha Vantage Usage
```python
# scripts/weekly_av_analysis.py
async def analyze_av_usage():
    """Analyze Alpha Vantage API usage patterns"""
    with db.get_db() as conn:
        cur = conn.cursor()
        
        # Weekly API usage
        cur.execute("""
            SELECT 
                DATE(timestamp) as day,
                COUNT(*) as total_calls,
                AVG(response_time_ms) as avg_response,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::float / COUNT(*) as hit_rate
            FROM av_api_metrics
            WHERE timestamp > NOW() - INTERVAL '7 days'
            GROUP BY DATE(timestamp)
            ORDER BY day
        """)
        
        print("📊 ALPHA VANTAGE WEEKLY USAGE:")
        for day, calls, response, hit_rate in cur.fetchall():
            print(f"{day}: {calls} calls, {response:.0f}ms, {hit_rate:.1%} cache")
            
        # Most used endpoints
        cur.execute("""
            SELECT endpoint, COUNT(*) as calls
            FROM av_api_metrics
            WHERE timestamp > NOW() - INTERVAL '7 days'
            GROUP BY endpoint
            ORDER BY calls DESC
            LIMIT 10
        """)
        
        print("\n🔝 TOP ALPHA VANTAGE ENDPOINTS:")
        for endpoint, calls in cur.fetchall():
            print(f"  {endpoint}: {calls} calls")
```

#### 3. Update ML Model with AV Historical Data
```python
# Retrain using Alpha Vantage's 20 years of options data
async def weekly_model_update():
    print("Training model with Alpha Vantage historical data...")
    
    # Can use up to 20 years of data!
    await ml_model.train_with_av_historical(
        symbols=['SPY', 'QQQ', 'IWM'],
        days_back=365  # 1 year for weekly training
    )
    
    print("Model updated with comprehensive AV historical options data")
```

#### 4. Optimize Alpha Vantage Cache
```python
# scripts/optimize_av_cache.py
def optimize_av_cache():
    """Adjust cache TTLs based on usage patterns"""
    
    # Analyze cache performance
    hit_rates = {}
    for cache_type in ['options', 'indicators', 'sentiment']:
        pattern = f"av_{cache_type}_*"
        hits = db.redis.get(f"cache_hits_{cache_type}") or 0
        misses = db.redis.get(f"cache_misses_{cache_type}") or 0
        
        if hits + misses > 0:
            hit_rates[cache_type] = hits / (hits + misses)
            
    # Adjust TTLs based on hit rates
    for cache_type, rate in hit_rates.items():
        if rate < 0.5:  # Low hit rate
            # Increase TTL
            config.av_cache_ttl[cache_type] *= 1.5
        elif rate > 0.9:  # Very high hit rate
            # Might be too long, decrease slightly
            config.av_cache_ttl[cache_type] *= 0.9
            
    print(f"Cache TTLs optimized: {config.av_cache_ttl}")
```

---

## SCALING OPERATIONS

### Phase Transitions with Alpha Vantage

#### Paper → Small Live (Week 9)
1. Verify Alpha Vantage data quality:
   - Compare AV Greeks with IBKR's theoretical values
   - Validate AV options prices against market
2. Monitor AV API stability for 1 week
3. Implement fallback for AV outages
4. Cache critical data aggressively

#### Small Live → Full Production (Week 11)
1. Upgrade Alpha Vantage plan if needed (beyond 600 calls/min)
2. Implement distributed caching for AV data
3. Add redundant AV API keys
4. Monitor Greeks accuracy closely

#### Scaling Alpha Vantage Usage
```python
# Monitor API efficiency
def check_av_efficiency():
    calls_per_trade = av_client.total_calls_today / trades_today
    print(f"Alpha Vantage efficiency: {calls_per_trade:.1f} calls per trade")
    
    if calls_per_trade > 10:
        print("⚠️ Consider optimizing API usage:")
        print("  - Increase cache TTLs")
        print("  - Batch similar requests")
        print("  - Use historical data where possible")
```

---

## EMERGENCY PROCEDURES

### Alpha Vantage Complete Outage
```python
# scripts/emergency_av_outage.py
async def handle_av_outage():
    print("🚨 ALPHA VANTAGE OUTAGE DETECTED")
    
    # 1. Switch to cached data only
    print("Switching to cached data mode...")
    config.av_cache_only = True
    
    # 2. Disable new positions (can't get Greeks)
    risk_manager.max_positions = len(risk_manager.positions)
    print(f"New positions disabled, managing {len(risk_manager.positions)} existing")
    
    # 3. Use IBKR theoretical prices for existing positions
    print("Using IBKR theoretical values for position management")
    
    # 4. Alert Discord
    await bot.send_emergency_alert(
        "Alpha Vantage outage - operating in degraded mode\n"
        "- No new positions\n"
        "- Using cached Greeks\n"
        "- Managing existing positions only"
    )
    
    # 5. Monitor for recovery
    asyncio.create_task(monitor_av_recovery())
```

### Rate Limit Exhaustion
```python
async def handle_rate_exhaustion():
    print("🚨 Alpha Vantage rate limit exhausted!")
    
    # Calculate wait time
    wait_seconds = av_client.rate_limiter.reset_time
    print(f"Must wait {wait_seconds}s until reset")
    
    if wait_seconds > 300:  # More than 5 minutes
        # Enter conservation mode
        print("Entering API conservation mode")
        
        # Only fetch critical data
        av_client.critical_only = True
        
        # Increase all cache TTLs by 10x
        for key in config.av_cache_ttl:
            config.av_cache_ttl[key] *= 10
```

---

## PERFORMANCE MONITORING

### Key Metrics with Alpha Vantage

| Metric | Target | Alert Level | Action |
|--------|--------|-------------|--------|
| AV API Calls/min | <500 | >550 | Increase cache TTL |
| AV Cache Hit Rate | >70% | <50% | Optimize cache strategy |
| AV Response Time | <300ms | >1000ms | Check API status |
| Greeks Accuracy | ±0.02 | ±0.05 | Verify data quality |
| Options Data Age | <60s | >5min | Force refresh |
| Rate Limit Buffer | >100 | <50 | Reduce API calls |

### Dashboard Queries

```sql
-- Alpha Vantage API performance
SELECT 
    endpoint,
    COUNT(*) as calls,
    AVG(response_time_ms) as avg_response,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::float / COUNT(*) as cache_hit_rate
FROM av_api_metrics
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY endpoint
ORDER BY calls DESC;

-- Greeks tracking
SELECT 
    symbol,
    AVG(entry_delta) as avg_delta,
    AVG(entry_gamma) as avg_gamma,
    AVG(entry_theta) as avg_theta,
    AVG(entry_vega) as avg_vega,
    COUNT(*) as positions
FROM trades
WHERE DATE(timestamp) = CURRENT_DATE
    AND entry_delta IS NOT NULL  -- Has AV Greeks
GROUP BY symbol;

-- API efficiency
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as api_calls,
    COUNT(DISTINCT symbol) as unique_symbols,
    COUNT(*) / COUNT(DISTINCT symbol) as calls_per_symbol
FROM av_api_metrics
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;
```

---

## CONFIGURATION REFERENCE

### config.yaml with Alpha Vantage
```yaml
# Data source configuration
data_sources:
  ibkr:
    host: 127.0.0.1
    port: 7497  # 7496 for live
    client_id: 1
    
  alpha_vantage:
    api_key: ${AV_API_KEY}  # Premium key
    rate_limit: 600  # Calls per minute
    timeout: 30  # Seconds
    retry_count: 3
    
    # Cache TTLs by data type (seconds)
    cache_ttl:
      options: 60  # 1 minute for real-time options
      historical_options: 3600  # 1 hour for historical
      indicators: 300  # 5 minutes for technical indicators
      sentiment: 900  # 15 minutes for news
      fundamentals: 86400  # 1 day for fundamentals
      
    # Endpoints to use
    enabled_apis:
      - REALTIME_OPTIONS
      - HISTORICAL_OPTIONS
      - RSI
      - MACD
      - BBANDS
      - ATR
      - ADX
      - NEWS_SENTIMENT
      - ANALYTICS_FIXED_WINDOW
      - TOP_GAINERS_LOSERS

# Trading configuration
trading:
  mode: paper  # paper/live
  symbols: [SPY, QQQ, IWM]
  
  # Data source routing
  data_routing:
    quotes: ibkr  # Real-time quotes from IBKR
    bars: ibkr  # Price bars from IBKR
    execution: ibkr  # Order execution through IBKR
    options: alpha_vantage  # Options chains from AV
    greeks: alpha_vantage  # Greeks from AV (not calculated!)
    indicators: alpha_vantage  # Technical indicators from AV
    sentiment: alpha_vantage  # News/sentiment from AV

# Risk management with AV Greeks
risk:
  max_positions: 5
  max_position_size: 10000
  daily_loss_limit: 1000
  
  # Greeks limits (using AV data)
  greeks:
    delta: [-0.3, 0.3]
    gamma: [-0.5, 0.5]
    vega: [-500, 500]
    theta: [-200, null]
    
  # Greeks source
  greeks_provider: alpha_vantage  # Not calculated locally!

# ML configuration
ml:
  model_path: models/xgboost_v1.pkl
  confidence_threshold: 0.6
  
  # Training data sources
  training_data:
    options_history: alpha_vantage  # 20 years available!
    price_data: ibkr
    
  # Feature sources
  features:
    price_action: ibkr
    technical_indicators: alpha_vantage
    options_metrics: alpha_vantage
    sentiment: alpha_vantage

# Monitoring
monitoring:
  log_level: INFO
  
  # Alpha Vantage specific monitoring
  av_monitoring:
    track_rate_limit: true
    track_response_times: true
    track_cache_hits: true
    alert_on_rate_limit: 550  # Alert at 550/600 calls
    alert_on_slow_response: 2000  # Alert if >2 seconds
    
  # Health checks
  health_checks:
    - name: ibkr_connection
      interval: 60
    - name: av_api_health
      interval: 300
    - name: av_rate_limit
      interval: 30
```

---

## COMMANDS CHEAT SHEET

```bash
# System startup
export AV_API_KEY="your_premium_key"
./scripts/start_all.sh

# Check Alpha Vantage status
python scripts/av_status.py

# Monitor AV rate limit
python scripts/av_rate_monitor.py

# Test AV endpoints
python scripts/test_av_endpoints.py

# Clear AV cache
python scripts/clear_av_cache.py

# Check Greeks for position
python scripts/check_greeks.py SPY

# Refresh options from AV
python scripts/refresh_av_options.py SPY

# Analyze AV usage
python scripts/av_usage_report.py

# Check both data sources
python scripts/check_data_sources.py

# View positions with AV Greeks
python scripts/show_positions_with_greeks.py

# Daily report with AV analytics
python scripts/daily_report_av.py

# Monitor real-time (includes AV)
python scripts/monitor.py

# Emergency AV fallback
python scripts/av_fallback_mode.py
```

---

## DATA SOURCE QUICK REFERENCE

| Data Type | Source | Frequency | Cache TTL |
|-----------|---------|-----------|-----------|
| Spot Prices | IBKR | Real-time | None |
| 5-sec Bars | IBKR | Real-time | None |
| Options Chains | Alpha Vantage | 1 min | 60s |
| Greeks | Alpha Vantage | With options | 60s |
| Historical Options | Alpha Vantage | On demand | 1 hour |
| Technical Indicators | Alpha Vantage | 5 min | 300s |
| News Sentiment | Alpha Vantage | 15 min | 900s |
| Order Execution | IBKR | Real-time | None |
| Position Management | IBKR | Real-time | None |

---

## SUPPORT & TROUBLESHOOTING

### Alpha Vantage Issues
- API Documentation: https://www.alphavantage.co/documentation/
- Check API status: https://www.alphavantage.co/support/
- Rate limit: 600 calls/minute (premium)
- Support email: support@alphavantage.co

### IBKR Issues
- TWS Help → Contact Support
- Gateway issues: Restart gateway
- API docs: interactivebrokers.github.io

### System Issues
1. Check logs: `logs/trading.log`, `logs/av_api.log`
2. Verify data sources: `python scripts/check_data_sources.py`
3. Database health: `psql -U postgres -d alphatrader`
4. Redis status: `redis-cli ping`

---

END OF OPERATIONS MANUAL v2.0