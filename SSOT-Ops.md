# SSOT-Ops.md – Operational Specification
**Version:** 3.0 (Updated for Batch Implementation & Institutional Analytics)  
**Last Updated:** Current  
**Audience:** Operations, PM, On-call Staff  
**Purpose:** Defines **what** the production options trading system must do, **when**, and under what conditions.  
**Scope:** All operational policies, runbooks, performance gates, risk limits, scheduling, and acceptance criteria.  
**Relation to SSOT-Tech:** SSOT-Tech.md defines **how** each requirement here is implemented.

---

## **1. Introduction & Scope**

### **1.1 System Type & Platform**
- **System Type:** Institutional-Grade Automated Options Trading System – Real Money  
- **Platform:** MacBook Pro (macOS) – Single Instance  
- **Development Approach:** Batch Implementation, API-Driven, Configuration-Based
- **Analytics Level:** Hedge Fund Quality with Microstructure & Quant Features

### **1.2 Version 3.0 Changes**
- **Batch API Implementation:** ALL 41 Alpha Vantage APIs implemented together (not incrementally)
- **IBKR 5-Second Bars Only:** All other timeframes mathematically aggregated
- **Institutional Analytics Added:** VPIN, GEX, microstructure, market profile
- **200+ ML Features:** Comprehensive quant-style feature engineering
- **Advanced ML Models:** LSTM/GRU/Transformer architectures included
- **Professional Backtesting:** Walk-forward analysis, purged CV standard
- **Compressed Timeline:** 87 days to production (vs 107 in v2.0)

### **1.3 MVP Scope**
- **Tier A Symbols:** `SPY, QQQ, IWM, SPX`  
- **Tier B Symbols:** `AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA`  
- **Tier C Symbols:** Rotating watchlist, limited by rate/subscription caps  
- **Strategies:**  
  - 0DTE  
  - 1DTE  
  - 14DTE Swing  
  - MOC Imbalance  
- **Publishing:** Discord webhooks, Dashboard API, Educational content (10+ pieces daily)
- **Analytics:** Institutional-grade with VPIN, GEX, microstructure analysis
- **Capital & Risk:** Fully configurable via YAML files

### **1.4 Out of Scope (Phase 2)**
- Complex ML model retraining (models are walk-forward tested but not retrained live)
- Twitter/X automation
- WHOP integration
- Multi-account support
- Pairs trading
- Crypto integration
- Additional brokers

### **1.5 Scope Creep Prevention**
Do **NOT** add:
- New strategies without full test cycle
- Additional symbols beyond initial set
- Complex UI features beyond dashboard
- Automated model retraining in production
- High-frequency trading features (sub-second)
- Additional data sources
- Hardcoded configuration values

---

## **2. System Overview & Data Architecture**

### **2.1 Data Source Division (CRITICAL UPDATE)**
<!-- START: DO NOT EDIT -->
**IBKR Provides:**
- **5-Second Bars ONLY:** All other timeframes (1m, 5m, 10m, 15m, 30m, 1h) are mathematically aggregated
- Real-time quotes: bid/ask/last with sizes
- MOC Imbalance feed: NYSE/NASDAQ, 15:40–15:55 ET
- Trade execution via TWS API
- Position monitoring and fill confirmation
- Portfolio state and P&L tracking

**Alpha Vantage Provides (41 APIs):**
- Real-time options chains with Greeks (Δ, Γ, Θ, Vega, Rho) - **PRIMARY GREEKS SOURCE**
- 16 technical indicators (RSI, MACD, BBANDS, ATR, ADX, VWAP, and 10 more)
- Advanced analytics: `ANALYTICS_FIXED_WINDOW`, `ANALYTICS_SLIDING_WINDOW`
- Historical options data for backtesting
- Fundamentals (10 APIs): Overview, Earnings, Statements, Dividends, Splits
- Economic indicators (5 APIs): Treasury Yield, Fed Funds, CPI, Inflation, GDP
- News sentiment analysis (3 APIs): News Sentiment, Top Gainers/Losers, Insider Transactions
<!-- END: DO NOT EDIT -->

### **2.2 Core System Flow with Institutional Analytics**
<!-- START: DO NOT EDIT -->
[IBKR 5-sec Bars] → [Bar Aggregator] → [All Timeframes]
                            ↓
[Alpha Vantage Greeks + Indicators] → [Data Ingestion & Validation]
                            ↓
                    [Institutional Analytics Engine]
                    - VPIN Calculation
                    - GEX Analysis
                    - Microstructure Metrics
                    - Market Profile
                            ↓
                    [ML Feature Engineering (200+ features)]
                            ↓
                    [ML Models (LSTM/GRU/XGBoost)]
                            ↓
                    [Decision Engine] ← [Strategy Rules from Config]
                            ↓
                    [Risk Management (VaR/CVaR)]
                            ↓
                    [IBKR Execution]
                            ↓
                    [Position Monitoring]
                            ↓
                    [Community Publishing & Education]
<!-- END: DO NOT EDIT -->

### **2.3 Configuration-Driven Architecture**
All operational parameters are externalized to configuration files:
- **System Settings:** Database, Redis, logging, paths
- **API Settings:** Endpoints, rate limits, retry policies
- **Data Management:** Symbols, schedules, validation rules
- **Strategy Parameters:** Entry/exit rules, confidence thresholds
- **Risk Limits:** Position and portfolio constraints with VaR/CVaR
- **ML Settings:** Model paths, feature specs, thresholds, backtesting params
- **Execution Rules:** Trading hours, order types, slippage
- **Monitoring:** Alert thresholds, webhook URLs, drift detection
- **Analytics:** VPIN params, GEX calculations, microstructure settings

---

## **3. Development & Implementation Procedures**

### **3.1 Batch Implementation Process (NEW)**
<!-- START: DO NOT EDIT -->
**Phase 0 (Days 1-2): Foundation**
1. Create complete project structure
2. Install all dependencies
3. Set up configuration system
4. Initialize database and Redis
5. Create base logger

**Phase 1 (Days 3-8): ALL 41 Alpha Vantage APIs**
1. Test ALL APIs with comprehensive script
2. Document ALL response structures
3. Design complete database schema
4. Create ALL tables at once
5. Implement ALL client methods
6. Build complete ingestion pipeline
7. Test entire AV ecosystem

**Phase 2 (Days 9-14): Complete IBKR Implementation**
1. Test 5-second bar feed
2. Implement bar aggregation to all timeframes
3. Create IBKR database schema
4. Build real-time data pipeline
5. Test MOC imbalance feed
6. Validate aggregation accuracy
<!-- END: DO NOT EDIT -->

### **3.2 API Testing Protocol (Batch Approach)**
<!-- START: DO NOT EDIT -->
For ALL 41 Alpha Vantage APIs simultaneously:

**Discovery Phase:**
1. Create comprehensive test script for all APIs
2. Make test calls with multiple symbols
3. Save ALL responses to JSON files
4. Analyze response structures collectively
5. Document rate limits and quirks
6. Design unified database schema

**Implementation Phase:**
1. Create ALL API methods in av_client.py
2. Configure ALL endpoints in config/apis/
3. Implement unified error handling
4. Test with multiple symbols concurrently

**Schema Creation:**
1. Design complete schema based on ALL responses
2. Create ALL tables in single migration
3. Build comprehensive indexes
4. Optimize for expected query patterns

**Ingestion Phase:**
1. Implement ALL ingestion methods
2. Add unified validation framework
3. Test complete pipeline end-to-end
4. Verify data persistence for all APIs
<!-- END: DO NOT EDIT -->

### **3.3 Daily Operational Procedures**

#### **3.3.1 Pre-Market Checklist (Before 9:00 AM ET)**
- Verify IBKR connection active
- Check 5-second bar aggregation working
- Confirm Alpha Vantage API key valid
- Review overnight position changes
- Check earnings calendar for holdings
- Verify risk parameters loaded from config
- Confirm VPIN/GEX calculations ready
- Clear stale cache entries
- Backup database
- Start monitoring dashboard
- Review any overnight alerts
- Check ML model drift metrics

#### **3.3.2 Market Open Procedures (9:30 AM ET)**
- Enable trading for all strategies
- Verify data feeds operational
- Check initial Greeks validation
- Confirm bar aggregation accurate
- Monitor first VPIN calculation
- Update GEX levels
- Monitor first decision cycle
- Verify Discord webhook active
- Check educational content scheduled

#### **3.3.3 Intraday Monitoring**
- **Every 5 minutes:**
  - Check portfolio Greeks
  - Verify API usage < 500/min
  - Monitor cache hit rate
  - Update VPIN metrics
  - Calculate current GEX
- **Every 30 minutes:**
  - Review decision quality scores
  - Check ML model confidence
  - Monitor position P&L
  - Update VaR calculations
  - Review microstructure metrics
- **3:40 PM:** Begin MOC imbalance monitoring
- **3:50 PM:** Final MOC decision window

#### **3.3.4 Market Close Procedures (4:00 PM ET)**
- Close all 0DTE positions by 3:30 PM
- Final MOC execution by 3:55 PM
- Archive trade data with full context
- Generate daily performance report with SHAP values
- Calculate final VaR/CVaR metrics
- Check tomorrow's earnings calendar
- Run database backup
- Review error logs
- Update configuration if needed
- Generate educational content for distribution

### **3.4 Emergency Procedures**

#### **3.4.1 Market Crisis Response**
1. **Detection:** VIX > 40 or SPY down > 3% or VPIN > threshold
2. **Actions:**
   - Hit emergency stop button
   - Close all positions immediately
   - Disable new trade generation
   - Set system to monitor-only mode
   - Review GEX levels for support
   - Manually manage remaining positions
3. **Documentation:** Log all actions taken
4. **Recovery:** Only resume after volatility normalizes and VPIN < threshold

#### **3.4.2 System Failure Response**
1. **Detection:** Any critical module failure
2. **Actions:**
   - Note all open positions
   - Log into IBKR TWS directly
   - Manually manage positions
   - Diagnose issue offline
   - Check bar aggregation status
3. **Recovery:** Only restart after fix verified in paper mode

#### **3.4.3 API Failure Response**
1. **Alpha Vantage Down:**
   - Use cached Greeks (max 30 seconds old)
   - Halt new trades if cache expired
   - Continue monitoring existing positions
2. **IBKR Connection Lost:**
   - Attempt reconnection with exponential backoff
   - Alert immediately via Discord
   - Switch to manual trading if down > 5 minutes
3. **Bar Aggregation Failure:**
   - Log missing bars
   - Use last valid aggregated bars
   - Alert if gap > 1 minute

---

## **4. Scheduling & Rate Management**

### **4.1 API Rate Management**
- **Alpha Vantage:** Target < 500/min, hard limit 600/min
- **Token Bucket:** 10 tokens/second refill rate
- **Burst Capacity:** 20 tokens maximum
- **IBKR Subscriptions:** Maximum 50 concurrent
- **Priority Order:** Positions > Greeks > Tier A > Tier B > MOC > Tier C

### **4.2 Data Collection Schedules**
All schedules defined in `config/data/schedules.yaml`:

**IBKR Real-Time (Continuous):**
- 5-second bars: Streaming for all active symbols
- Aggregation: Real-time to 1m, 5m, 10m, 15m, 30m, 1h
- Quotes: Tick-by-tick for positions
- MOC Imbalance: 5-second updates (3:40-3:55 PM)

**Alpha Vantage - Tier A (SPY, QQQ, IWM, SPX):**
- Options with Greeks: Every 30 seconds
- RSI, MACD, BBANDS, VWAP: Every 60 seconds
- ATR, ADX: Every 5 minutes
- Analytics: Every 5 minutes

**Alpha Vantage - Tier B (MAG7 Stocks):**
- Options with Greeks: Every 45 seconds
- Core indicators: Every 5 minutes
- Analytics: Every 15 minutes

**Alpha Vantage - Tier C (Watchlist):**
- Options with Greeks: Every 3 minutes
- Indicator bundle: Every 10 minutes

### **4.3 API Call Budget**
```
IBKR: Unlimited 5-second bars (subscription based)
Alpha Vantage Budget:
  Tier A: ~240 calls/minute (4 symbols × 60 calls)
  Tier B: ~105 calls/minute (7 symbols × 15 calls)
  Tier C: ~30 calls/minute (variable)
  Other: ~25 calls/minute (news, fundamentals, economic)
  Total: ~400 calls/minute (200 call buffer remaining)
```

---

## **5. Risk & Performance Gates**

### **5.1 Institutional Risk Metrics**
All limits defined in `config/risk/` directory:

**Position-Level Limits:**
- Max Delta: 0.80 (configurable)
- Min Delta: 0.20 (configurable)
- Max Gamma: 0.20 (configurable)
- Max Vega: 200 (configurable)
- Min Theta Ratio: 0.02 (configurable)
- Max position VaR (95%): 5% of portfolio

**Portfolio-Level Limits:**
- Max Net Delta: 0.30
- Max Net Gamma: 0.75
- Max Net Vega: 1000
- Max Net Theta: -500
- Max Capital at Risk: 20%
- Portfolio VaR (95%): 10%
- Portfolio CVaR (95%): 15%

**Microstructure Limits:**
- Max VPIN: 0.6 (high toxicity threshold)
- Min liquidity score: 0.4
- Max bid-ask spread: 0.5% (for entry)

**Circuit Breakers:**
- Daily Loss: 2% (triggers halt)
- Weekly Loss: 5% (triggers review)
- Max Drawdown: 10% (triggers shutdown)
- VPIN > 0.7: Emergency close all
- Model confidence < 0.3: Halt new trades

### **5.2 Performance Targets**

#### **5.2.1 System Performance**
| Metric | Target | Maximum | Test Method |
|--------|--------|---------|-------------|
| 5-sec bar latency | < 100ms | 200ms | Time from IBKR to storage |
| Bar aggregation | < 50ms | 100ms | 5-sec to 1-min conversion |
| Decision Latency | < 1s | 2s | Time from data to decision |
| Order Submission | < 500ms | 1s | Decision to IBKR |
| Greeks Validation | < 100ms | 200ms | Validation cycle time |
| VPIN Calculation | < 200ms | 500ms | Per symbol calculation |
| GEX Calculation | < 500ms | 1s | Full chain analysis |
| Feature Generation | < 500ms | 1s | 200+ features per symbol |
| ML Prediction | < 200ms | 500ms | Model inference time |
| Database Query | < 100ms | 500ms | Complex query benchmark |
| Cache Retrieval | < 10ms | 50ms | Redis GET operation |

#### **5.2.2 Trading Performance (Paper Mode Minimum)**
| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Win Rate | 45% | 55% | 65% |
| Profit Factor | 1.2 | 1.5 | 2.0 |
| Sharpe Ratio | 1.0 | 1.5 | 2.0 |
| Sortino Ratio | 1.5 | 2.0 | 3.0 |
| Calmar Ratio | 1.0 | 1.5 | 2.5 |
| Max Drawdown | 15% | 10% | 5% |
| Daily VaR (95%) | 3% | 2% | 1% |
| Information Ratio | 0.5 | 1.0 | 1.5 |

#### **5.2.3 ML Model Performance**
| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Prediction Accuracy | 55% | 60% | 65% |
| Precision | 60% | 65% | 70% |
| F1 Score | 0.55 | 0.60 | 0.65 |
| AUC-ROC | 0.60 | 0.65 | 0.70 |
| Feature Importance Stability | 70% | 80% | 90% |
| Model Confidence Average | 0.6 | 0.7 | 0.8 |

### **5.3 Go/No-Go Production Criteria**
Must **ALL** be true before production deployment:

**Data Infrastructure:**
- [ ] All 41 Alpha Vantage APIs operational
- [ ] IBKR 5-second bars streaming reliably
- [ ] Bar aggregation accurate to all timeframes
- [ ] Complete database schema implemented
- [ ] Rate limiting never exceeded in testing
- [ ] All configurations externalized to YAML

**Analytics Requirements:**
- [ ] VPIN calculations validated
- [ ] GEX analysis operational
- [ ] Market profile generation working
- [ ] 200+ features generated per symbol
- [ ] Feature importance analysis complete
- [ ] Microstructure metrics accurate

**ML Requirements:**
- [ ] All models (XGBoost, LSTM, GRU) operational
- [ ] Walk-forward backtesting complete
- [ ] SHAP values generating correctly
- [ ] Model confidence scores calibrated
- [ ] Prediction accuracy > 55%
- [ ] Model drift detection active

**Operational Requirements:**
- [ ] 5+ consecutive days successful paper trading
- [ ] Win rate consistently > 45%
- [ ] All circuit breakers tested and working
- [ ] Emergency stop procedure validated
- [ ] Backup and recovery procedures tested
- [ ] Educational content pipeline active
- [ ] Documentation complete and current

**Risk Management:**
- [ ] Greeks validation catching bad data
- [ ] Position limits enforced correctly
- [ ] Portfolio limits working
- [ ] VaR/CVaR calculations accurate
- [ ] Stop losses executing properly
- [ ] Daily loss breaker tested
- [ ] VPIN threshold triggers tested
- [ ] Manual override procedures working

---

## **6. Failure Detection & Recovery**

### **6.1 Connection Failures**
| Failure | Detection Method | Recovery Action | Testing Method |
|---------|-----------------|-----------------|----------------|
| IBKR Disconnect | No 5-sec bars for 10s | Reconnect with exponential backoff | Kill TWS process |
| Bar Aggregation Fail | Missing timeframes | Use last valid bars, alert | Stop aggregator |
| AV API Down | HTTP 500/503 errors | Use cached data (max 30s old) | Block API endpoint |
| Database Down | Connection timeout | Queue writes to Redis | Stop PostgreSQL |
| Redis Down | Connection refused | Continue without cache | Stop Redis service |
| Network Loss | Multiple failures | Switch to emergency mode | Disconnect network |

### **6.2 Data Quality Failures**
| Failure | Detection Method | Recovery Action | Testing Method |
|---------|-----------------|-----------------|----------------|
| Stale Greeks | Age > 30 seconds | Halt new trades, use last valid | Delay API response |
| Bar Gap | Missing 5-sec bars | Interpolate or use last valid | Skip bars in stream |
| Price Divergence | IBKR/cached > 0.5% | Use IBKR, log for investigation | Inject divergent prices |
| Missing Indicators | Null values in response | Skip signal, retry next cycle | Delete from cache |
| Invalid Greeks | Outside valid bounds | Reject entire chain | Inject invalid values |
| VPIN Spike | VPIN > 0.7 | Emergency close positions | Inject toxic flow |
| Feature NaN | Missing ML features | Use feature median, alert | Corrupt feature data |

### **6.3 Execution Failures**
| Failure | Detection Method | Recovery Action | Testing Method |
|---------|-----------------|-----------------|----------------|
| Order Rejected | IBKR error code | Log, alert, retry with modifications | Submit invalid order |
| Partial Fill | Fill < requested | Adjust monitoring for actual size | Limit order liquidity |
| Stop Loss Failed | Price through stop | Market order immediately | Simulate gap down |
| Position Mismatch | DB != IBKR | Reconcile with IBKR as source | Modify DB directly |
| Execution Timeout | No fill in 30s | Cancel and retry | Delay fill confirmation |

---

## **7. Strategy Operational Specs with Institutional Analytics**

### **7.1 0DTE Strategy Operations**
**Entry Window:** 09:45 - 14:00 ET  
**Auto-Close:** 15:30 ET  
**Max Positions:** 3 concurrent  
**ML Confidence Required:** ≥ 0.75  
**VPIN Threshold:** < 0.5  

**Entry Requirements (from config):**
- RSI between 30-70
- Delta between 0.25-0.75
- Gamma < 0.20
- Theta/Price ratio ≥ 0.03
- IV percentile > 20
- Volume ≥ 0.5× average
- Bid-ask spread < 0.10
- GEX support/resistance identified
- Market profile value area check
- Microstructure score > 0.6

**Monitoring Requirements:**
- Check Greeks every 30 seconds
- Update VPIN every minute
- Monitor GEX levels continuously
- Update stop loss if profitable
- Close if ML confidence drops below 0.60
- Emergency exit if VPIN > 0.6

### **7.2 1DTE Strategy Operations**
**Entry Window:** 09:45 - 15:00 ET  
**Hold Overnight:** Yes  
**Max Positions:** 5 concurrent  
**ML Confidence Required:** ≥ 0.70  
**VPIN Threshold:** < 0.55  

**Operational Differences from 0DTE:**
- Wider delta range: 0.20-0.80
- Lower theta ratio acceptable: 0.02
- Can hold overnight with hedging
- Check overnight GEX shifts
- Review next day's economic calendar

### **7.3 14DTE Swing Operations**
**Entry:** Any time during market hours  
**Hold Period:** 1-14 days  
**Max Positions:** 10 concurrent  
**ML Confidence Required:** ≥ 0.65  
**Regime Alignment:** Required  

**Daily Review Requirements:**
- Check fundamentals for changes
- Review upcoming earnings
- Adjust stops based on volatility
- Monitor regime shifts
- Update position VaR
- Consider rolling if profitable

### **7.4 MOC Imbalance Operations**
**Active Window:** 15:40 - 15:55 ET  
**Decision Time:** By 15:50 ET  
**Execution:** 15:50 - 15:55 ET  
**Min Imbalance:** $10M normalized  
**VPIN Override:** Can trade if VPIN < 0.7  

**Operational Process:**
1. Begin monitoring at 15:40
2. Calculate normalized imbalance
3. Check option setup feasibility
4. Review closing auction liquidity
5. Calculate expected slippage
6. Score opportunity with ML
7. Execute if score ≥ 0.70
8. Use straddle if IV low, directional if high

---

## **8. Testing & Verification Procedures**

### **8.1 Batch API Testing Protocol**
ALL 41 APIs must pass these tests simultaneously:

1. **Functional Testing:**
   - Successful calls for all APIs
   - Correct response parsing for all
   - Unified error handling working
   - Retry logic functioning

2. **Rate Limit Testing:**
   - Combined load stays < 500/min
   - Token bucket prevents bursts
   - Priority system working
   - Graceful degradation

3. **Data Quality Testing:**
   - All responses match schemas
   - Data types correct across APIs
   - Values within reasonable ranges
   - Timestamps synchronized

4. **Persistence Testing:**
   - All data saves correctly
   - Indexes perform well
   - Queries optimized
   - Cache operations efficient

### **8.2 Bar Aggregation Testing**

**Accuracy Testing:**
- 5-sec to 1-min aggregation exact
- OHLC values correct
- Volume sums accurate
- VWAP calculations verified

**Performance Testing:**
- Aggregation < 50ms per symbol
- No data loss under load
- Memory usage stable
- CPU usage acceptable

### **8.3 Institutional Analytics Testing**

**VPIN Testing:**
- Calculate on historical data
- Compare to academic benchmarks
- Verify toxicity detection
- Test threshold triggers

**GEX Testing:**
- Validate gamma calculations
- Verify aggregation across strikes
- Test support/resistance identification
- Compare to known services

**ML Model Testing:**
- Walk-forward validation
- Out-of-sample performance
- Feature importance stability
- Prediction confidence calibration

### **8.4 Paper Trading Validation (Days 52-59)**

**Week 1: Complete System Test**
- All data feeds working
- All analytics calculating
- All strategies triggering
- Risk limits enforcing

**Week 2: Performance Validation**
- Win rate tracking
- Sharpe ratio calculation
- VaR accuracy check
- Educational content generation

---

## **9. Configuration Management**

### **9.1 Configuration Structure for Batch Implementation**
```
config/
├── .env                           # API keys and secrets
├── system/
│   ├── database.yaml              # Database connections
│   ├── redis.yaml                 # Cache configuration
│   ├── logging.yaml               # Logging settings
│   └── paths.yaml                 # Directory paths
├── apis/
│   ├── alpha_vantage.yaml         # ALL 41 endpoints configured
│   ├── ibkr.yaml                  # IBKR settings with aggregation
│   └── rate_limits.yaml           # Unified rate limiting
├── data/
│   ├── symbols.yaml               # Symbol tiers
│   ├── schedules.yaml             # Polling schedules for all APIs
│   └── validation.yaml            # Data validation rules
├── analytics/
│   ├── vpin.yaml                  # VPIN parameters
│   ├── gex.yaml                   # GEX calculation settings
│   ├── microstructure.yaml        # Microstructure params
│   └── market_profile.yaml        # Market profile settings
├── strategies/
│   ├── 0dte.yaml                  # 0DTE parameters
│   ├── 1dte.yaml                  # 1DTE parameters
│   ├── swing_14d.yaml             # Swing parameters
│   └── moc_imbalance.yaml         # MOC parameters
├── risk/
│   ├── position_limits.yaml       # Position limits
│   ├── portfolio_limits.yaml      # Portfolio limits with VaR
│   ├── circuit_breakers.yaml      # Emergency stops
│   └── sizing.yaml                # Position sizing
├── ml/
│   ├── models.yaml                # Model paths and weights
│   ├── features.yaml              # 200+ feature specifications
│   ├── thresholds.yaml            # ML thresholds
│   └── backtesting.yaml           # Backtest parameters
├── execution/
│   ├── trading_hours.yaml         # Market hours
│   └── order_types.yaml           # Order settings
├── monitoring/
│   ├── alerts.yaml                # Alert rules
│   ├── discord.yaml               # Discord settings
│   └── dashboard.yaml             # Dashboard config
└── environments/
    ├── development.yaml           # Dev overrides
    ├── paper.yaml                 # Paper trading settings
    └── production.yaml            # Production settings
```

### **9.2 Configuration Backup Strategy**
- **Frequency:** Every market close + before any changes
- **Retention:** 90 days of configs (increased from 30)
- **Location:** Git with tagged releases
- **Testing:** Weekly restore drill

---

## **10. Maintenance & Monitoring**

### **10.1 Daily Maintenance Tasks**
- **Pre-Market:** 
  - Verify bar aggregation working
  - Clear cache of stale entries
  - Check ML model drift scores
  - Verify configurations loaded
  - Test VPIN calculation
- **Post-Market:**
  - Archive trade data with context
  - Calculate daily VaR metrics
  - Generate SHAP explanations
  - Backup database
  - Review error logs
  - Update educational content

### **10.2 Weekly Maintenance Tasks**
- Review ML model performance
- Analyze feature importance changes
- Check VPIN/GEX accuracy
- Review and tune strategy parameters
- Analyze losing trades for patterns
- Performance optimization review
- Update risk limits if needed

### **10.3 Monthly Maintenance Tasks**
- Full system backup and restore test
- Walk-forward model retraining
- Review feature engineering pipeline
- Analyze strategy effectiveness
- Update symbol lists
- Capacity planning review
- Review institutional metrics accuracy

### **10.4 Real-Time Monitoring Dashboard**

**Critical Metrics:**
- IBKR 5-sec bar streaming status
- Bar aggregation accuracy
- API call rate (< 500/min)
- Current VPIN levels
- GEX support/resistance
- ML model confidence
- Portfolio Greeks
- Position P&L with VaR
- System resources

**Analytics Metrics:**
- Microstructure scores
- Market regime indicators
- Feature quality scores
- Model drift metrics
- Prediction accuracy (real-time)

---

## **11. Phase Deliverables (Version 3.0)**

### **11.1 Development Phase Deliverables - Batch Implementation**

| Phase | Days | Deliverables | Success Criteria |
|-------|------|--------------|------------------|
| **0: Foundation** | 1-2 | - Complete project structure<br>- All dependencies installed<br>- Configuration system ready<br>- Database/Redis connected | All components initialize |
| **1: AV Batch** | 3-8 | - ALL 41 APIs tested & documented<br>- Complete schema created<br>- All ingestion methods working<br>- Scheduler configured for all | 41 APIs storing data |
| **2: IBKR Complete** | 9-14 | - 5-sec bars streaming<br>- Aggregation to all timeframes<br>- MOC feed operational<br>- Complete IBKR pipeline | Accurate bar aggregation |
| **3: Integration** | 15-17 | - All data sources validated<br>- Performance optimized<br>- Monitoring active | < 500 API calls/min maintained |
| **4: Analytics** | 18-24 | - VPIN/GEX operational<br>- Microstructure metrics<br>- Market profile working<br>- All indicators calculating | Institutional metrics accurate |
| **5: ML Features** | 25-28 | - 200+ features per symbol<br>- Feature importance analysis<br>- Real-time generation<br>- Quality monitoring | SHAP values generating |
| **6: ML Models** | 29-35 | - XGBoost/LSTM/GRU deployed<br>- Walk-forward tested<br>- Backtesting complete<br>- Drift detection active | Accuracy > 55% |
| **7: Strategies** | 36-43 | - All 4 strategies implemented<br>- ML integration complete<br>- Config-driven rules | Signals generating correctly |
| **8: Risk** | 44-47 | - VaR/CVaR implemented<br>- Position sizing working<br>- Limits enforced<br>- Circuit breakers tested | Risk metrics accurate |
| **9: Execution** | 48-51 | - Orders executing (paper)<br>- Fill monitoring active<br>- Slippage tracked | Paper trades successful |
| **10: Paper Trading** | 52-59 | - 5+ days paper trading<br>- Performance validated<br>- Win rate > 45%<br>- Educational content started | All strategies profitable |
| **11: Publishing** | 60-66 | - Discord active<br>- Dashboard operational<br>- Reports automated<br>- WebSocket streaming | Real-time updates working |
| **12: Education** | 67-73 | - Market analysis automated<br>- Educational engine active<br>- 10+ daily pieces<br>- Community engaged | Content pipeline complete |
| **13: Testing** | 74-80 | - Full integration tested<br>- Stress testing complete<br>- Performance optimized<br>- Documentation current | All tests passing |
| **14: Production** | 81-87 | - Go/no-go complete<br>- Production environment ready<br>- Team trained<br>- Rollback plan tested | Ready for live trading |

### **11.2 Operational Readiness Checklist**

**Before Production Launch:**
- [ ] All 41 Alpha Vantage APIs batch implemented
- [ ] IBKR 5-second bar aggregation accurate
- [ ] VPIN/GEX calculations validated
- [ ] 200+ ML features generating
- [ ] Models achieving > 55% accuracy
- [ ] 5+ days successful paper trading
- [ ] Performance targets met
- [ ] Risk management validated
- [ ] Emergency procedures tested
- [ ] Educational platform operational
- [ ] Documentation complete
- [ ] Team trained on procedures
- [ ] Rollback plan ready
- [ ] Monitoring dashboard operational

---

## **12. Critical Success Factors**

### **12.1 Technical Success Factors**
- **Data Completeness:** All 41 APIs + IBKR operational before analytics
- **Aggregation Accuracy:** 5-sec bars correctly aggregated to all timeframes
- **Greeks Quality:** Always validated, never stale (< 30 seconds)
- **Rate Limiting:** Never exceed 600 calls/minute
- **Latency:** Decisions made within 2 seconds
- **ML Confidence:** Models calibrated and interpretable

### **12.2 Operational Success Factors**
- **Batch Implementation:** Complete data layer before building on top
- **Configuration:** No hardcoded values anywhere
- **Testing:** Institutional-grade backtesting and validation
- **Documentation:** Every component thoroughly documented
- **Monitoring:** Real-time visibility with institutional metrics
- **Recovery:** Can recover from any failure mode

### **12.3 Business Success Factors**
- **Performance:** Win rate > 45% with Sharpe > 1.0
- **Risk:** VaR/CVaR limits never exceeded
- **Analytics:** Institutional-grade metrics providing edge
- **Education:** 10+ quality pieces daily building community
- **Scalability:** Can add strategies/symbols via config
- **Compliance:** All trades logged with full context

---

## **Appendix A: Institutional Metrics Reference**

### **VPIN (Volume-Synchronized Probability of Informed Trading)**
- **Purpose:** Detect toxic order flow
- **Calculation:** Volume-bucketed price changes
- **Threshold:** > 0.6 indicates high toxicity
- **Action:** Reduce position size or halt trading

### **GEX (Gamma Exposure)**
- **Purpose:** Identify support/resistance from options
- **Calculation:** Net gamma across all strikes
- **Use:** Predict dealer hedging flows
- **Interpretation:** Negative GEX = volatile, Positive = stable

### **Market Profile**
- **Purpose:** Identify value areas and POC
- **TPO:** Time Price Opportunity
- **Value Area:** 70% of volume traded
- **Use:** Support/resistance identification

### **Microstructure Metrics**
- **Kyle's Lambda:** Price impact coefficient
- **Amihud Illiquidity:** Price change per dollar volume
- **Roll's Spread:** Effective spread from serial covariance
- **Use:** Assess execution costs and market quality

---

## **Appendix B: Implementation Timeline Summary**

**Week 1 (Days 1-7):** Foundation + Start API Testing  
**Week 2 (Days 8-14):** Complete Data Foundation (AV + IBKR)  
**Week 3 (Days 15-21):** Integration + Start Analytics  
**Week 4 (Days 22-28):** Complete Analytics + ML Features  
**Week 5 (Days 29-35):** ML Models + Backtesting  
**Week 6 (Days 36-42):** Strategy Implementation  
**Week 7 (Days 43-49):** Risk + Start Execution  
**Week 8 (Days 50-56):** Paper Trading Week 1  
**Week 9 (Days 57-63):** Paper Trading Week 2 + Publishing  
**Week 10 (Days 64-70):** Educational Platform  
**Week 11 (Days 71-77):** Full Integration Testing  
**Week 12 (Days 78-84):** Production Preparation  
**Week 13 (Days 85-87):** Final Validation  

**Total: 87 days to production with institutional-grade analytics**

---

## **END OF SSOT-OPS.MD v3.0**