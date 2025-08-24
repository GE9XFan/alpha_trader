# System Coherence Guide
## AlphaTrader Trading System Documentation Integration

---

## Document Overview & Relationships

This guide ensures comprehensive coherence between the three core system documents:

1. **Trading System Technical Specification** (`trading_system_tech_spec.md`)
   - System architecture and component design
   - Data flow and processing pipelines
   - Community engagement features
   - Performance specifications

2. **Trading System Operations Manual** (`trading-ops-manual.md`)
   - Daily operational procedures
   - System monitoring and maintenance
   - Troubleshooting guides
   - Emergency procedures

3. **Alpha Vantage API Reference** (`av_api_reference.py`)
   - Complete list of 36 Alpha Vantage APIs
   - API parameters and usage examples
   - Integration patterns

---

## Key System Constants & Coherence Points

### 1. Performance Metrics (Consistent Across All Documents)

| Metric | Value | Referenced In |
|--------|-------|---------------|
| Critical Path Latency | <50ms | Tech Spec 1.3, Ops Manual 1.2 |
| IBKR Data Cycle | 5 seconds | Tech Spec 2.3, Ops Manual 1.1 |
| Alpha Vantage Rate Limit | 500 calls/minute | Tech Spec 3.1.1, Ops Manual 1.2, API Reference |
| Max Concurrent Positions | 20 | Tech Spec 3.4, Ops Manual 1.2 |
| Max Position Size | $50,000 | Tech Spec 3.4, Ops Manual 1.2 |
| Daily Loss Limit | $10,000 | Tech Spec 3.4, Ops Manual 1.2 |
| VPIN Threshold | 0.7 | Tech Spec 3.4, Ops Manual 1.2 |

### 2. API Integration Points

**Alpha Vantage APIs (36 Total):**

| Category | Count | Update Frequency | Priority |
|----------|-------|------------------|----------|
| Options | 2 | 30 seconds | CRITICAL |
| Technical Indicators | 16 | 30 sec (critical), 5 min (others) | HIGH |
| Analytics | 2 | 5 minutes | MEDIUM |
| Sentiment | 3 | 5 minutes | MEDIUM |
| Fundamentals | 8 | 15 minutes | LOW |
| Economic | 5 | 15 minutes | LOW |

**IBKR Connection:**
- Live Trading Port: 7496
- Paper Trading Port: 7497
- Heartbeat Interval: 30 seconds

### 3. Component Latency Budget

Total Critical Path: 50ms
- IBKR Data Receipt: 5ms
- Feature Calculation: 15ms
- Model Inference: 10ms
- Risk Check: 5ms
- Order Execution: 15ms

### 4. Risk Management Thresholds

**Position Limits:**
- Portfolio Delta: ±0.3
- Portfolio Gamma: ±0.75
- Portfolio Vega: ±1000
- Portfolio Theta: > -500

**Circuit Breakers:**
- Daily Loss: -$10,000 → Full stop
- VPIN > 0.7: Toxic flow → Close all
- 5 Consecutive Losses: → Pause 30 minutes
- Model Accuracy <30%: → Switch to rules

### 5. Community Tiers

| Tier | Price | Signal Delay | Features |
|------|-------|--------------|----------|
| FREE | $0 | 5 minutes | Delayed signals, daily recap |
| PREMIUM | $99 | 30 seconds | Real-time signals, market analysis |
| VIP | $499 | 0 seconds | Instant signals, all features, risk metrics |

---

## Operational Workflows

### Daily Trading Cycle

**Pre-Market (8:00 - 9:30 AM ET)**
1. System startup and health checks
2. Verify all 36 Alpha Vantage APIs responding
3. Confirm IBKR connection on port 7496
4. Check risk limits and positions
5. Review economic calendar

**Market Hours (9:30 AM - 4:00 PM ET)**
1. Process IBKR 5-second bars
2. Update Alpha Vantage data per schedule
3. Execute trades with <50ms latency
4. Broadcast to Discord within 2 seconds
5. Monitor VPIN and risk metrics

**Post-Market (4:00 - 5:00 PM ET)**
1. Generate daily recap
2. Update community leaderboards
3. Archive trade data
4. Backup critical systems

### Data Flow Integration

```
IBKR (5-sec) + Alpha Vantage (36 APIs) 
    ↓
Feature Engine (147 features)
    ↓
Model Server (XGBoost + Fallback)
    ↓
Risk Manager (Greeks + VPIN)
    ↓
Executor → IBKR Orders
    ↓
Discord Bot → Community Channels
```

---

## System Dependencies

### Software Stack
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose
- IBKR Gateway/TWS
- Discord.py
- XGBoost

### Hardware Requirements
- CPU: 16+ cores
- RAM: 64GB minimum
- Storage: 1TB NVMe SSD
- Network: 10Gbps connection

---

## Monitoring & Alerting Coherence

### Critical Metrics Dashboard
All three documents reference these key metrics:
- System Health (component status)
- P&L Tracker (vs $10K daily limit)
- Position Monitor (max 20 positions)
- VPIN Chart (threshold 0.7)
- API Usage (500/min limit)
- Execution Metrics (target <50ms)

### Alert Escalation
1. **Warning Level:** Slack notification
   - Daily Loss > $7,500
   - VPIN > 0.6
   - Latency > 75ms

2. **Critical Level:** PagerDuty alert
   - Daily Loss > $9,000
   - VPIN > 0.7
   - Latency > 100ms

3. **Emergency:** Automatic shutdown
   - Daily Loss = $10,000
   - System failure
   - Network partition > 5 min

---

## Command Quick Reference

### System Control
```bash
# Start system
./scripts/start_trading_system.sh

# Emergency stop
./scripts/EMERGENCY_STOP.sh

# Check status
python ops/system_status.py

# View positions
python ops/show_positions.py
```

### Alpha Vantage Testing
```bash
# Test specific API
python av_api_reference.py --commands

# Check rate limit
python ops/check_av_rate.py
```

### Risk Management
```bash
# Override limits (requires approval)
python ops/risk_override.py --parameter max_positions --value 25

# Close all positions
python ops/close_all_positions.py
```

---

## Compliance & Validation

### Daily Checklist
- [ ] All 36 Alpha Vantage APIs functional
- [ ] IBKR connection stable (port 7496)
- [ ] Risk limits configured correctly
- [ ] Model confidence > 0.6
- [ ] Database backup completed
- [ ] Discord bot online
- [ ] Whop webhooks active

### Performance Validation
- [ ] Critical path latency < 50ms
- [ ] API calls < 500/minute
- [ ] Cache hit rate > 80%
- [ ] Fill rate > 95%
- [ ] Discord broadcast < 2 seconds

---

## Error Code Cross-Reference

| Code | System | Description | Action |
|------|--------|-------------|--------|
| E001 | IBKR | Connection lost | Auto-reconnect |
| E002 | Alpha Vantage | Rate limit hit | Throttle to critical APIs |
| E003 | Model | Low confidence | Use rule-based fallback |
| E004 | Risk | Limit breached | Halt trading |
| E005 | Database | Connection failed | Cache-only mode |
| E006 | Discord | API error | Retry with backoff |
| E007 | Trading | Insufficient funds | Reduce position size |
| E008 | Market | Symbol halted | Cancel orders |
| E009 | Network | Partition detected | Safe mode |
| E010 | Model | Inference timeout | Skip signal |

---

## Version Control & Updates

**Current Version:** 3.0
**Architecture:** Direct Pipeline with Community Integration
**Last Updated:** 2025-01-10

### Document Maintenance Schedule
- Technical Specification: Updated with major releases
- Operations Manual: Updated weekly with operational insights
- API Reference: Updated when Alpha Vantage adds/changes APIs
- This Guide: Updated whenever documents change

---

## Critical Contacts

- **Trading Desk:** ext. 1001
- **DevOps On-Call:** +1-XXX-XXX-XXXX
- **Risk Management:** risk@company.com
- **Emergency Hotline:** +1-XXX-XXX-XXXX (24/7)

---

END OF COHERENCE GUIDE