# AlphaTrader Operations Manual
## Daily Running Guide for Single Developer

---

## QUICK START CHECKLIST

### First Time Setup (30 minutes)
```bash
# 1. Create Python environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install ib_insync xgboost pandas numpy scipy psycopg2-binary redis discord.py ta-lib

# 3. Create database
psql -U postgres
CREATE DATABASE alphatrader;
\q

# 4. Set up configuration
cp config.template.yaml config.yaml
# Edit config.yaml with your settings

# 5. Test IBKR connection
python tests/test_ibkr_connection.py

# Ready to go!
```

---

## DAILY OPERATIONS

### Pre-Market Routine (8:30 AM - 9:30 AM)

#### 1. System Startup (8:30 AM)
```bash
# Terminal 1: Start market data
cd ~/AlphaTrader
source venv/bin/activate
python scripts/start_market_data.py

# Terminal 2: Start paper trader (Phase 2+)
python scripts/start_paper_trader.py

# Terminal 3: Start Discord bot (Phase 2+)
python scripts/start_discord_bot.py
```

#### 2. Health Checks (8:45 AM)
```python
# scripts/morning_checks.py
import asyncio
from src.data.market_data import market_data
from src.data.options_data import options_data

async def morning_checks():
    print("🔍 Running morning checks...")
    
    # Check IBKR connection
    await market_data.connect()
    print("✅ IBKR connected")
    
    # Check market data
    await market_data.subscribe_symbols(['SPY'])
    await asyncio.sleep(5)
    price = market_data.get_latest_price('SPY')
    print(f"✅ SPY price: ${price:.2f}")
    
    # Check options chains
    chain = await options_data.fetch_option_chain('SPY')
    print(f"✅ Options chain: {len(chain)} expirations")
    
    # Check database
    with db.get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = CURRENT_DATE")
        trades_today = cur.fetchone()[0]
        print(f"✅ Database: {trades_today} trades today")
        
    # Check risk limits
    print(f"✅ Risk limits: {risk_manager.positions} / {risk_manager.max_positions} positions")
    print(f"✅ Daily P&L: ${risk_manager.daily_pnl:.2f} / -${risk_manager.daily_loss_limit}")
    
    print("\n🎯 System ready for trading!")

asyncio.run(morning_checks())
```

#### 3. Review Positions (9:00 AM)
```python
# scripts/review_positions.py
def review_positions():
    with db.get_db() as conn:
        cur = conn.cursor()
        
        # Get current positions
        cur.execute("""
            SELECT symbol, option_type, strike, expiry, quantity, 
                   price, (current_price - price) * quantity * 100 as unrealized_pnl
            FROM positions
            WHERE quantity != 0
        """)
        
        positions = cur.fetchall()
        
        print("\n📊 CURRENT POSITIONS")
        print("-" * 60)
        
        total_pnl = 0
        for pos in positions:
            print(f"{pos['symbol']} {pos['option_type']} ${pos['strike']} exp:{pos['expiry']}")
            print(f"  Qty: {pos['quantity']} | Entry: ${pos['price']:.2f} | P&L: ${pos['unrealized_pnl']:.2f}")
            total_pnl += pos['unrealized_pnl']
            
        print(f"\nTotal Unrealized P&L: ${total_pnl:.2f}")
        
        # Check for 0DTE positions
        cur.execute("""
            SELECT * FROM positions 
            WHERE expiry = CURRENT_DATE AND quantity != 0
        """)
        
        expiring = cur.fetchall()
        if expiring:
            print(f"\n⚠️  WARNING: {len(expiring)} positions expiring today!")
            print("These will be auto-closed at 3:59 PM")
```

### Market Hours Operations (9:30 AM - 4:00 PM)

#### Monitoring Dashboard
Create a simple monitoring script that runs continuously:

```python
# scripts/monitor.py
import time
import os
from datetime import datetime

def clear_screen():
    os.system('clear')

async def monitor_loop():
    while True:
        clear_screen()
        
        print("=" * 60)
        print(f"ALPHATRADER MONITOR - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Market Data
        print("\n📈 MARKET DATA")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            price = market_data.get_latest_price(symbol)
            print(f"{symbol}: ${price:.2f}")
            
        # Positions
        print(f"\n💼 POSITIONS: {len(risk_manager.positions)} / {risk_manager.max_positions}")
        for symbol, pos in risk_manager.positions.items():
            print(f"  {symbol}: {pos['quantity']} contracts")
            
        # Greeks
        print(f"\n🎯 PORTFOLIO GREEKS")
        for greek, value in risk_manager.portfolio_greeks.items():
            limit_min, limit_max = risk_manager.greeks_limits[greek]
            status = "✅" if limit_min <= value <= limit_max else "⚠️"
            print(f"  {greek}: {value:.3f} {status}")
            
        # P&L
        print(f"\n💰 DAILY P&L: ${risk_manager.daily_pnl:.2f}")
        if risk_manager.daily_pnl < 0:
            pct_of_limit = abs(risk_manager.daily_pnl / risk_manager.daily_loss_limit * 100)
            print(f"  Loss: {pct_of_limit:.1f}% of daily limit")
            
        # Recent Signals
        print(f"\n📡 RECENT SIGNALS")
        recent = signal_generator.signals_today[-3:] if signal_generator.signals_today else []
        for sig in recent:
            print(f"  {sig['timestamp'].strftime('%H:%M')} - {sig['symbol']} {sig['signal_type']} ({sig['confidence']:.2f})")
            
        # System Status
        print(f"\n⚙️  SYSTEM STATUS")
        print(f"  IBKR: {'🟢 Connected' if market_data.connected else '🔴 Disconnected'}")
        print(f"  Discord: {'🟢 Online' if bot.is_ready() else '🔴 Offline'}")
        print(f"  Mode: {config.mode.upper()}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

asyncio.run(monitor_loop())
```

#### Manual Interventions

**Close Position Manually:**
```python
# scripts/close_position.py
async def close_position(symbol: str):
    position = risk_manager.positions.get(symbol)
    
    if not position:
        print(f"No position in {symbol}")
        return
        
    # Create closing order
    contract = Option(
        symbol,
        position['expiry'].strftime('%Y%m%d'),
        position['strike'],
        position['option_type'],
        'SMART'
    )
    
    order = MarketOrder('SELL', position['quantity'])
    
    if config.mode == 'live':
        trade = market_data.ib.placeOrder(contract, order)
        # Wait for fill...
    else:
        print(f"Paper close: {symbol} {position['quantity']} contracts")
        
    # Update database
    # Remove from risk manager
    del risk_manager.positions[symbol]
```

**Halt Trading:**
```python
# scripts/halt_trading.py
def halt_trading(reason: str):
    print(f"⛔ HALTING TRADING: {reason}")
    
    # Set flag in database
    with db.get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO system_status (key, value, timestamp)
            VALUES ('trading_halted', %s, NOW())
        """, (reason,))
        conn.commit()
        
    # Signal generator will check this flag
    signal_generator.enabled = False
    
    print("Trading halted. Run 'python scripts/resume_trading.py' to resume")
```

### End of Day Operations (3:30 PM - 5:00 PM)

#### 3:30 PM - Pre-Close Checks
```python
# scripts/pre_close.py
def pre_close_checks():
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
                print(f"  {pos['symbol']} {pos['option_type']} ${pos['strike']}")
            print("\n  These will be closed at 3:59 PM")
```

#### 3:59 PM - Auto-Close 0DTE
```python
# This runs automatically in the system
async def close_expiring_positions():
    """Automatically close all 0DTE positions"""
    for symbol, position in risk_manager.positions.items():
        if position['expiry'] == datetime.now().date():
            print(f"🚨 Closing 0DTE: {symbol}")
            await close_position(symbol)
```

#### 4:15 PM - Daily Report
```python
# scripts/daily_report.py
def generate_daily_report():
    print("\n" + "="*60)
    print(f"DAILY REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    
    with db.get_db() as conn:
        cur = conn.cursor()
        
        # Today's trades
        cur.execute("""
            SELECT COUNT(*), SUM(pnl) 
            FROM trades 
            WHERE DATE(timestamp) = CURRENT_DATE
        """)
        trades_count, total_pnl = cur.fetchone()
        
        # Win rate
        cur.execute("""
            SELECT 
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as losses
            FROM trades 
            WHERE DATE(timestamp) = CURRENT_DATE
        """)
        wins, losses = cur.fetchone()
        
        print(f"\n📊 TRADING SUMMARY")
        print(f"  Total Trades: {trades_count}")
        print(f"  Wins: {wins} | Losses: {losses}")
        print(f"  Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A")
        print(f"  Total P&L: ${total_pnl:.2f}")
        
        # Best and worst trades
        cur.execute("""
            SELECT symbol, option_type, strike, pnl
            FROM trades
            WHERE DATE(timestamp) = CURRENT_DATE
            ORDER BY pnl DESC
            LIMIT 1
        """)
        best = cur.fetchone()
        
        cur.execute("""
            SELECT symbol, option_type, strike, pnl
            FROM trades
            WHERE DATE(timestamp) = CURRENT_DATE
            ORDER BY pnl ASC
            LIMIT 1
        """)
        worst = cur.fetchone()
        
        if best:
            print(f"\n🏆 Best Trade: {best['symbol']} {best['option_type']} ${best['strike']} = ${best['pnl']:.2f}")
        if worst:
            print(f"💥 Worst Trade: {worst['symbol']} {worst['option_type']} ${worst['strike']} = ${worst['pnl']:.2f}")
            
        # Save to file
        with open(f"reports/daily_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
            f.write(report_text)
            
        print("\nReport saved to reports/")
```

---

## TROUBLESHOOTING

### Common Issues and Solutions

#### 1. IBKR Connection Lost
```python
# Auto-reconnect is built in, but if needed:
async def reconnect_ibkr():
    market_data.ib.disconnect()
    await asyncio.sleep(5)
    await market_data.connect()
    await market_data.subscribe_symbols(config.symbols)
```

#### 2. Discord Bot Offline
```bash
# Restart bot
pkill -f discord_bot.py
python scripts/start_discord_bot.py &
```

#### 3. High Memory Usage
```python
# Clear caches
options_data.greeks_cache.clear()
db.redis.flushdb()
```

#### 4. Database Issues
```bash
# Check database
psql -U postgres -d alphatrader -c "SELECT COUNT(*) FROM trades;"

# Vacuum database (weekly)
psql -U postgres -d alphatrader -c "VACUUM ANALYZE;"
```

#### 5. ML Model Not Performing
```python
# Retrain model with recent data
python scripts/retrain_model.py --days 30
```

---

## WEEKLY MAINTENANCE

### Sunday Tasks (Market Closed)

#### 1. Backup Database
```bash
pg_dump -U postgres alphatrader > backups/alphatrader_$(date +%Y%m%d).sql
```

#### 2. Review Weekly Performance
```python
# scripts/weekly_review.py
def weekly_review():
    # Generate comprehensive statistics
    # Identify patterns in winning/losing trades
    # Adjust parameters if needed
```

#### 3. Update ML Model
```python
# Retrain with past week's data
async def weekly_model_update():
    # Get past week's data
    data = await get_week_data()
    
    # Retrain model
    ml_model.train(data)
    
    # Backtest new model
    results = backtest_model(ml_model)
    
    if results['sharpe'] > current_sharpe:
        print("✅ New model is better, deploying")
        ml_model.save()
    else:
        print("❌ New model worse, keeping current")
```

#### 4. Clean Up Logs
```bash
# Archive old logs
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;
find logs/ -name "*.gz" -mtime +30 -delete
```

---

## SCALING OPERATIONS

### Phase Transitions

#### Paper → Small Live (Week 9)
1. Change config.yaml: `mode: live`
2. Reduce position sizes: `max_position_size: 2000`
3. Tighten risk limits: `daily_loss_limit: 500`
4. Run paper and live in parallel for 1 week
5. Compare results daily

#### Small Live → Full Production (Week 11)
1. Gradually increase position sizes
2. Add more symbols one at a time
3. Increase risk limits slowly
4. Monitor Greeks more frequently
5. Set up alerts for anomalies

#### Adding Community Tiers (Week 12)
1. Create Discord roles: Free, Premium, VIP
2. Set up payment processing (Stripe/PayPal)
3. Implement delay system for signals
4. Create onboarding documentation
5. Set up customer support channel

---

## EMERGENCY PROCEDURES

### Market Crash / High Volatility
```python
# scripts/emergency_close_all.py
async def emergency_close_all():
    print("🚨 EMERGENCY: CLOSING ALL POSITIONS")
    
    for symbol in list(risk_manager.positions.keys()):
        try:
            await close_position(symbol)
        except Exception as e:
            print(f"Failed to close {symbol}: {e}")
            
    print("All positions closed. Trading halted.")
    halt_trading("Emergency market conditions")
```

### System Failure
1. **Stop all Python processes**
2. **Check IBKR Gateway** - restart if needed
3. **Check database** - restart PostgreSQL
4. **Review error logs** - identify root cause
5. **Restart components** one by one
6. **Verify positions** match database

### Large Loss Event
If daily loss exceeds 50% of limit:
1. System auto-halts (built-in)
2. Review all trades for errors
3. Check if ML model is malfunctioning
4. Verify market data is correct
5. Consider switching to paper mode
6. Analyze and adjust before resuming

---

## PERFORMANCE MONITORING

### Key Metrics to Track Daily

| Metric | Target | Alert Level | Action |
|--------|--------|-------------|--------|
| Win Rate | >55% | <45% | Review model |
| Avg Win/Loss | >1.2 | <1.0 | Adjust stops |
| Daily Trades | 10-30 | >50 | Check for overtrading |
| Latency | <200ms | >500ms | Optimize code |
| Greeks Breach | 0 | >2/day | Tighten limits |
| Discord Delay | <5s | >10s | Check bot |

### Dashboard Queries

```sql
-- Today's performance
SELECT 
    COUNT(*) as trades,
    SUM(CASE WHEN pnl > 0 THEN 1 END)::float / COUNT(*) as win_rate,
    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
    SUM(pnl) as total_pnl
FROM trades 
WHERE DATE(timestamp) = CURRENT_DATE;

-- Position exposure
SELECT 
    symbol,
    SUM(quantity) as total_contracts,
    SUM(quantity * price * 100) as total_exposure
FROM positions
WHERE quantity != 0
GROUP BY symbol;

-- Signal accuracy
SELECT 
    signal_type,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN executed THEN 1 END)::float / COUNT(*) as execution_rate
FROM signals
WHERE DATE(timestamp) = CURRENT_DATE
GROUP BY signal_type;
```

---

## CONFIGURATION REFERENCE

### config.yaml Structure
```yaml
# Mode settings
mode: paper  # paper/live

# Symbols to trade
symbols: [SPY, QQQ, IWM]

# Risk management
risk:
  max_positions: 5  # Start small
  max_position_size: 10000  # Per position
  daily_loss_limit: 1000  # Stop trading if hit
  
  # Greeks limits (portfolio level)
  greeks:
    delta: [-0.3, 0.3]  # Stay delta neutral
    gamma: [-0.5, 0.5]  # Control gamma risk
    vega: [-500, 500]  # Limit vega exposure
    theta: [-200, null]  # Max theta burn

# ML settings
ml:
  model_path: models/xgboost_v1.pkl
  confidence_threshold: 0.6  # Min confidence to trade
  retrain_interval: 7  # Days

# Execution
execution:
  contracts_per_trade: 5  # Start with 5
  slippage_ticks: 1  # Expected slippage
  
# Community (Phase 2+)
community:
  discord_token: ${DISCORD_TOKEN}
  publish_paper: true
  publish_live: false  # Enable when ready
  
  delays:  # Seconds
    free: 300  # 5 minutes
    premium: 30
    vip: 0
```

---

## COMMANDS CHEAT SHEET

```bash
# Start system
./scripts/start_all.sh

# Stop system
./scripts/stop_all.sh

# Check status
python scripts/system_status.py

# View positions
python scripts/show_positions.py

# Close position
python scripts/close_position.py SPY

# Halt trading
python scripts/halt_trading.py "Reason"

# Resume trading
python scripts/resume_trading.py

# Generate report
python scripts/daily_report.py

# Backup database
./scripts/backup_db.sh

# Monitor live
python scripts/monitor.py

# Check logs
tail -f logs/trading.log

# Run tests
pytest tests/

# Retrain model
python scripts/retrain_model.py
```

---

## CONTACT & SUPPORT

### System Issues
- Check logs first: `logs/trading.log`
- Check database: `psql -U postgres -d alphatrader`
- Check IBKR Gateway status
- Restart components if needed

### IBKR Support
- TWS: Help → Contact Support
- Gateway issues: Restart gateway
- API docs: interactivebrokers.github.io

### Community
- Discord server for your subscribers
- Keep a private admin channel
- Document common questions

---

END OF OPERATIONS MANUAL