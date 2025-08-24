# Trading System Operations Manual
## Direct Pipeline Architecture

**Version:** 3.0  
**Architecture:** Direct Pipeline with Async Logging  
**Last Updated:** 2025-01-10  
**Emergency Hotline:** +1-XXX-XXX-XXXX (24/7)

---

# PART 1: SYSTEM OVERVIEW

## 1.1 Architecture Summary

### Core Design
The system uses a **Direct Pipeline Architecture** where market data flows through sequential processing stages via direct function calls. This eliminates message bus latency and provides sub-50ms execution times.

### Key Components

| Component | Purpose | Latency Target | Criticality |
|-----------|---------|---------------|-------------|
| Data Manager | Handles IBKR & Alpha Vantage feeds | <5ms | CRITICAL |
| Feature Engine | Calculates indicators & Greeks | <15ms | CRITICAL |
| Model Server | Generates predictions | <10ms | HIGH |
| Risk Manager | Validates positions & limits | <5ms | CRITICAL |
| Executor | Places orders via IBKR | <15ms | CRITICAL |
| Event Store | Async logging (non-blocking) | N/A | MEDIUM |

### Data Flow
```
IBKR (5-sec bars) → Feature Engine → Model Server → Risk Manager → Executor
                            ↑
Alpha Vantage (36 APIs) → Cache → [Used next cycle]
```

## 1.2 System Specifications

### Performance Requirements
- **Critical Path Latency**: <50ms total (IBKR bar to order)
- **Alpha Vantage Updates**: Every 30 seconds (critical), 5 minutes (full)
- **Risk Calculations**: Real-time with every position change
- **Database Writes**: Async within 1 second

### API Limits & Priorities
- **Alpha Vantage**: 500 calls/minute across 36 endpoints
- **IBKR**: No hard limit, but throttle at 50 messages/second
- **Critical APIs**: REALTIME_OPTIONS, RSI, MACD, VPIN, Greeks
- **Secondary APIs**: Other technical indicators updated every 5 minutes

### Risk Limits
- **Position Limits**: Max 20 positions, $50K per position
- **Portfolio Greeks**: Delta ±0.3, Gamma ±0.75, Vega ±1000, Theta > -500
- **Daily Loss Limit**: $10,000 (hard stop)
- **VPIN Threshold**: 0.7 (toxicity indicator)

---

# PART 2: DAILY OPERATIONS

## 2.1 Market Day Timeline

### Pre-Market (8:00 - 9:30 AM ET)

#### 8:00 AM - System Startup
```bash
# Start core services
./scripts/start_trading_system.sh

# Verify all components
./scripts/health_check.sh
```

Expected output:
- PostgreSQL: CONNECTED
- Redis: CONNECTED  
- IBKR Gateway: CONNECTED
- Alpha Vantage: RESPONDING (latency <500ms)

#### 8:30 AM - Pre-Market Validation

**Checklist:**
1. **Check Positions**
   - Run: `python ops/check_positions.py`
   - Verify all positions from yesterday are accounted for
   - Confirm no unexpected overnight changes
   - Review current Greeks against limits

2. **Verify Data Feeds**
   - IBKR connection status (must show "SYNCHRONIZED")
   - Alpha Vantage rate limit status (<400 calls used in last minute)
   - Cache hit rate (should be >80% for active symbols)

3. **Review Market Conditions**
   - Check economic calendar for major events
   - Review pre-market movers
   - Note any earnings for held positions

4. **System Resources**
   - CPU usage <30%
   - Memory usage <60%
   - Disk space >20GB free
   - Network latency <10ms to IBKR

#### 9:00 AM - Final Checks
- Enable paper trading mode first (if Monday)
- Verify risk limits are set correctly
- Confirm emergency stop script is accessible
- Test notification system

### Market Hours (9:30 AM - 4:00 PM ET)

#### Opening Bell Procedures (9:30 - 9:45 AM)
1. **9:28 AM**: Enable pre-market data collection
2. **9:30 AM**: System automatically begins trading
3. **9:31 AM**: Verify first IBKR bar processed
4. **9:35 AM**: Check dashboard for initial signals
5. **9:45 AM**: Review opening positions

#### Continuous Monitoring

**Every 5 minutes check:**
- Portfolio P&L
- Position count vs limit (max 20)
- VPIN level (must be <0.7)
- Active orders status

**Every 30 minutes check:**
- Greeks dashboard (all within limits)
- API rate limiting status
- System resource usage
- Error log for warnings

**Every hour:**
- Daily P&L vs stop loss ($10K limit)
- Compare predicted vs actual fills
- Review rejected signals log

#### MOC Window (3:40 - 4:00 PM)
1. **3:40 PM**: Begin monitoring MOC imbalances
2. **3:45 PM**: System calculates imbalance signals
3. **3:50 PM**: Review any pending MOC orders
4. **3:55 PM**: Last opportunity for MOC submissions
5. **3:59 PM**: Close all 0DTE positions
6. **4:00 PM**: Market close - system enters post-market mode

### Post-Market (4:00 - 5:00 PM ET)

#### Immediate Tasks (4:00 - 4:15 PM)
1. Generate daily summary report
2. Archive today's event log
3. Calculate final P&L
4. Backup critical data

#### Analysis Tasks (4:15 - 5:00 PM)
1. Review all trades with win/loss analysis
2. Identify any system errors or delays
3. Check for model prediction accuracy
4. Document any unusual market behavior
5. Prepare handoff notes for next operator

---

# PART 3: OPERATIONAL PROCEDURES

## 3.1 Starting the System

### Full System Startup
```bash
# 1. Start infrastructure
docker-compose up -d postgres redis

# 2. Wait for databases
./scripts/wait_for_db.sh

# 3. Start IBKR Gateway
./scripts/start_ibkr_gateway.sh

# 4. Start trading system
python -m trading_system.main \
  --config config/production.yaml \
  --mode live

# 5. Verify startup
curl http://localhost:8080/health
```

### Component Health Verification
After startup, verify each component:

1. **Data Manager**: Should show "READY" with both IBKR and AV connected
2. **Feature Engine**: Should have cached features for all active symbols  
3. **Model Server**: Should load model and show confidence >0.6
4. **Risk Manager**: Should display current limits and positions
5. **Executor**: Should show IBKR connection active

## 3.2 Monitoring & Alerts

### Critical Metrics Dashboard

Access: `http://monitoring:3000/dashboard/trading`

**Key Panels:**
- **System Health**: All components green/yellow/red status
- **P&L Tracker**: Real-time P&L with stop loss indicator
- **Position Monitor**: Current positions with Greeks
- **VPIN Chart**: Toxicity level (alert if >0.6)
- **API Usage**: Alpha Vantage rate limit consumption
- **Execution Metrics**: Fill rate, slippage, rejections

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Daily Loss | -$7,500 | -$9,000 | Reduce position size → Stop trading |
| VPIN | 0.6 | 0.7 | Reduce exposure → Close all positions |
| Position Count | 18 | 20 | Slow down → Stop new positions |
| API Rate | 450/min | 490/min | Throttle → Pause non-critical |
| Latency | 75ms | 100ms | Investigate → Reduce load |
| Model Confidence | <0.5 | <0.4 | Use fallback → Stop ML trading |

### Notification Channels
- **Slack**: #trading-alerts (all warnings)
- **PagerDuty**: Critical alerts only
- **Email**: Daily summaries and reports
- **Dashboard**: Real-time visual alerts

## 3.3 Risk Management

### Position Limits Enforcement

The system enforces these limits automatically:
- **Max Positions**: 20 concurrent
- **Max Position Size**: $50,000
- **Max Daily Loss**: $10,000
- **Greeks Limits**: Checked every 30 seconds

### Manual Risk Override

If you need to override risk limits:

```bash
# Temporary override (expires in 1 hour)
python ops/risk_override.py \
  --parameter max_positions \
  --value 25 \
  --duration 3600 \
  --reason "Special event trading"
```

**Required approvals:**
- Limits +20%: Team lead
- Limits +50%: Risk manager
- Disable limits: C-level executive

### Circuit Breakers

**Automatic Triggers:**
1. **Daily Loss**: -$10,000 → Full stop
2. **VPIN > 0.7**: Toxic flow → Close all
3. **5 Consecutive Losses**: → Pause 30 minutes
4. **Model Accuracy <30%**: → Switch to rules

**Manual Circuit Breaker:**
```bash
# Emergency stop - halts ALL trading immediately
./scripts/EMERGENCY_STOP.sh
```

## 3.4 Common Operations

### Adding a Symbol

```bash
python ops/manage_symbols.py add AAPL \
  --strategy momentum \
  --size 100 \
  --options yes
```

This will:
1. Add symbol to active list
2. Start fetching Alpha Vantage data
3. Subscribe to IBKR feed
4. Initialize feature calculations

### Checking System State

```bash
# Overall health
python ops/system_status.py

# Specific component
python ops/system_status.py --component risk_manager

# Current positions
python ops/show_positions.py --format table

# Today's trades
python ops/show_trades.py --date today
```

### Adjusting Parameters

```bash
# Change model confidence threshold
python ops/adjust_param.py model.confidence_threshold 0.65

# Update position limit
python ops/adjust_param.py risk.max_positions 15

# Modify VPIN threshold
python ops/adjust_param.py risk.vpin_threshold 0.65
```

---

# PART 4: TROUBLESHOOTING

## 4.1 Common Issues

### Issue: IBKR Connection Lost

**Symptoms:** No new bars, orders not filling

**Resolution:**
1. Check IBKR Gateway status
2. Restart gateway: `./scripts/restart_ibkr.sh`
3. Verify network connectivity
4. Check for IBKR maintenance windows
5. If persistent, use backup connection

### Issue: High Latency (>100ms)

**Symptoms:** Slow order execution, missed opportunities

**Resolution:**
1. Check CPU/Memory usage
2. Review active symbol count (reduce if >50)
3. Clear Redis cache: `redis-cli FLUSHDB`
4. Restart feature engine
5. Disable non-critical indicators

### Issue: Alpha Vantage Rate Limit

**Symptoms:** 429 errors, missing data

**Resolution:**
1. Check current usage: `python ops/check_av_rate.py`
2. Reduce update frequency
3. Prioritize critical symbols only
4. Use cached data for non-critical
5. Consider upgrading API plan

### Issue: Model Predictions Inconsistent

**Symptoms:** Low confidence, poor accuracy

**Resolution:**
1. Check feature quality: `python ops/check_features.py`
2. Verify data feed quality
3. Switch to fallback model
4. Review recent market conditions
5. Schedule model retraining

## 4.2 Emergency Procedures

### Complete System Failure

1. **Execute emergency stop:** `./scripts/EMERGENCY_STOP.sh`
2. **Close all positions manually via IBKR TWS**
3. **Document system state:** `python ops/export_state.py`
4. **Notify:** Call emergency hotline
5. **Begin recovery:** Follow disaster recovery runbook

### Database Failure

1. **System auto-switches to cache-only mode**
2. **Verify Redis is functioning**
3. **Start database recovery**
4. **Once recovered, replay events from cache**
5. **Verify data integrity**

### Network Partition

1. **System enters safe mode automatically**
2. **No new positions allowed**
3. **Existing positions managed locally**
4. **Monitor for recovery**
5. **Manual intervention if >5 minutes**

---

# PART 5: MAINTENANCE

## 5.1 Daily Maintenance

### End of Day (Automated at 5 PM)
- Database backup
- Log rotation
- Cache cleanup
- Performance metrics calculation
- Report generation

### Manual Checks
- Review error logs
- Check disk space
- Verify backup completion
- Update symbol lists
- Review tomorrow's calendar

## 5.2 Weekly Maintenance

### Saturday Window (2-4 AM ET)

```bash
# Run maintenance script
./scripts/weekly_maintenance.sh
```

This performs:
- Database vacuum and reindex
- Clear expired cache entries
- Archive old events (>30 days)
- Update dependencies
- Run system diagnostics
- Generate weekly report

### Sunday Tasks
- Review weekly performance
- Analyze losing trades
- Update model if needed
- Plan for upcoming week
- Test disaster recovery

## 5.3 Monthly Tasks

### First Saturday of Month
- Full system backup
- Rotate API keys
- Update documentation
- Review and adjust risk limits
- Analyze strategy performance
- Security patches

### Performance Review
- Calculate monthly Sharpe ratio
- Review all risk breaches
- Analyze system latency trends
- Update feature importance
- Plan optimization projects

---

# PART 6: DEPLOYMENT

## 6.1 Production Deployment

### Pre-Deployment Checklist
- [ ] All tests passing
- [ ] Paper trading for 5+ days
- [ ] Sharpe ratio >1.0 in paper
- [ ] Risk manager approval
- [ ] Rollback plan ready
- [ ] Monitoring alerts configured

### Deployment Process

```bash
# 1. Backup current state
./scripts/backup_all.sh

# 2. Deploy new version
git pull origin main
python -m pip install -r requirements.txt

# 3. Run migrations
python ops/migrate_db.py

# 4. Restart with new version
./scripts/restart_system.sh --gradual

# 5. Monitor for 30 minutes
python ops/monitor_deployment.py --duration 1800
```

### Rollback Procedure

If issues detected:
```bash
# Immediate rollback
./scripts/rollback.sh --to-version previous

# Verify system stable
python ops/verify_stable.py
```

## 6.2 Configuration Management

### Environment Files

```yaml
# config/production.yaml
system:
  mode: production
  log_level: INFO
  
trading:
  enabled: true
  max_positions: 20
  position_size: 10000
  
risk:
  daily_loss_limit: 10000
  vpin_threshold: 0.7
  
apis:
  alpha_vantage:
    rate_limit: 500
  ibkr:
    account: U1234567
```

### Secret Management
- API keys in environment variables
- Rotate monthly
- Never commit to git
- Use secure storage (Vault/AWS Secrets)

---

# PART 7: PERFORMANCE TUNING

## 7.1 Optimization Targets

### Latency Optimization
- **Target**: <50ms critical path
- **Current**: Monitor p50, p95, p99
- **Optimize**: Feature calculation, model inference
- **Cache**: Everything possible

### Resource Optimization
- **CPU**: Keep below 70% peak
- **Memory**: Below 80% usage
- **Network**: <10ms to IBKR
- **Disk I/O**: <100 IOPS average

## 7.2 Scaling Guidelines

### When to Scale

**Add Resources When:**
- Latency p95 >75ms consistently
- CPU usage >80% for 5+ minutes
- Active symbols >75
- Daily volume >1000 trades

**Scaling Options:**
1. Vertical: Upgrade instance (quick fix)
2. Horizontal: Distribute symbols across instances
3. Optimize: Profile and improve code
4. Reduce: Lower symbol count or features

---

# APPENDIX A: QUICK REFERENCE

## Critical Commands

```bash
# Emergency
./scripts/EMERGENCY_STOP.sh           # Halt everything
python ops/close_all_positions.py     # Exit all positions

# Status
python ops/system_status.py           # Full health check
python ops/show_positions.py          # Current positions
python ops/show_pnl.py                # Today's P&L

# Control
python ops/pause_trading.py           # Temporary halt
python ops/resume_trading.py          # Resume after pause
python ops/adjust_param.py [key] [val] # Change parameter

# Monitoring
tail -f logs/trading.log              # Live log stream
python ops/monitor_realtime.py        # Real-time metrics
```

## Key Files & Locations

- **Config**: `/config/production.yaml`
- **Logs**: `/var/log/trading/`
- **Models**: `/models/`
- **Database**: PostgreSQL port 5432
- **Redis**: Port 6379
- **IBKR Gateway**: Port 7496 (live), 7497 (paper)
- **Monitoring**: http://localhost:3000
- **API**: http://localhost:8080

## Contact Information

- **Trading Desk**: ext. 1001
- **DevOps On-Call**: +1-XXX-XXX-XXXX
- **Risk Management**: risk@company.com
- **Emergency**: +1-XXX-XXX-XXXX (24/7)

---

END OF OPERATIONS MANUAL

Version 3.0 | Direct Pipeline Architecture | Last Updated: 2025-01-10