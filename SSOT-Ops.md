# SSOT-Ops.md – Operational Specification
**Version:** 2.0 (Updated for Skeleton-First, API-Driven Development)  
**Last Updated:** Current  
**Audience:** Operations, PM, On-call Staff  
**Purpose:** Defines **what** the production options trading system must do, **when**, and under what conditions.  
**Scope:** All operational policies, runbooks, performance gates, risk limits, scheduling, and acceptance criteria.  
**Relation to SSOT-Tech:** SSOT-Tech.md defines **how** each requirement here is implemented.

---

## **1. Introduction & Scope**

### **1.1 System Type & Platform**
- **System Type:** Automated Options Trading System – Real Money  
- **Platform:** MacBook Pro (macOS) – Single Instance  
- **Development Approach:** Skeleton-First, API-Driven, Configuration-Based

### **1.2 Version 2.0 Changes**
- **IBKR provides ALL intraday pricing** (not Alpha Vantage)
- **Skeleton-first development** for clear dependencies
- **API-driven schema evolution** based on actual responses
- **Configuration-driven architecture** with no hardcoded values
- **Incremental testing** of each API before proceeding

### **1.3 MVP Scope**
- **Tier A Symbols:** `SPY, QQQ, IWM, SPX`  
- **Tier B Symbols:** `AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA`  
- **Tier C Symbols:** Rotating watchlist, limited by rate/subscription caps  
- **Strategies:**  
  - 0DTE  
  - 1DTE  
  - 14DTE Swing  
  - MOC Imbalance  
- **Publishing:** Discord webhooks only (Twitter/X, WHOP stubs for future)  
- **Capital & Risk Defaults:** Fully configurable via YAML files

### **1.4 Out of Scope (Phase 2)**
- Complex ML model training
- Twitter/X automation
- WHOP integration
- Advanced dashboard visualizations
- Multi-account support
- Pairs trading
- Crypto integration

### **1.5 Scope Creep Prevention**
Do **NOT** add:
- New strategies without full test cycle
- Additional symbols beyond initial set
- Complex UI features
- Automated model retraining
- High-frequency trading features
- Additional brokers
- Hardcoded configuration values

---

## **2. System Overview & Data Architecture**

### **2.1 Data Source Division (UPDATED)**
<!-- START: DO NOT EDIT -->
**IBKR Provides:**
- **ALL Real-time Pricing:** 1s, 5s, 1m, 5m, 15m, 30m, 1h bars
- Real-time quotes: bid/ask/last with sizes
- MOC Imbalance feed: NYSE/NASDAQ, 15:40–15:55 ET
- Trade execution via TWS API
- Position monitoring and fill confirmation
- Portfolio state and P&L tracking

**Alpha Vantage Provides (43 APIs):**
- Real-time options chains with Greeks (Δ, Γ, Θ, Vega, Rho) - **PRIMARY GREEKS SOURCE**
- All technical indicators (RSI, MACD, BBANDS, ATR, ADX, VWAP, and 10 more)
- Advanced analytics: `ANALYTICS_FIXED_WINDOW`, `ANALYTICS_SLIDING_WINDOW`
- Historical options data for backtesting
- Fundamentals (11 APIs): Overview, Earnings, Statements, Dividends, Splits
- Economic indicators (5 APIs): Treasury Yield, Fed Funds, CPI, Inflation, GDP
- News sentiment analysis (3 APIs): News Sentiment, Top Gainers/Losers, Insider Transactions
<!-- END: DO NOT EDIT -->

### **2.2 Core System Flow (UPDATED)**
<!-- START: DO NOT EDIT -->
[IBKR Real-time Pricing] + [Alpha Vantage Greeks] + [Alpha Vantage Indicators]
                            ↓
                    [Data Ingestion & Validation]
                            ↓
                    [ML Feature Engineering]
                            ↓
                    [ML Models Prediction]
                            ↓
                    [Decision Engine] ← [Strategy Rules from Config]
                            ↓
                    [Risk Management Check]
                            ↓
                    [IBKR Execution]
                            ↓
                    [Position Monitoring]
                            ↓
                    [Community Publishing]
<!-- END: DO NOT EDIT -->

### **2.3 Configuration-Driven Architecture**
All operational parameters are externalized to configuration files:
- **System Settings:** Database, Redis, logging, paths
- **API Settings:** Endpoints, rate limits, retry policies
- **Data Management:** Symbols, schedules, validation rules
- **Strategy Parameters:** Entry/exit rules, confidence thresholds
- **Risk Limits:** Position and portfolio constraints
- **ML Settings:** Model paths, feature specs, thresholds
- **Execution Rules:** Trading hours, order types, slippage
- **Monitoring:** Alert thresholds, webhook URLs

---

## **3. Development & Implementation Procedures**

### **3.1 Skeleton-First Development Process**
<!-- START: DO NOT EDIT -->
1. Create all module files with empty classes/methods
2. Establish import structure and dependencies
3. Verify no circular dependencies exist
4. Implement base classes for inheritance
5. Add configuration loading to each module
6. Test module initialization without functionality
7. Document interfaces between modules
8. Commit skeleton to version control
<!-- END: DO NOT EDIT -->

### **3.2 API Implementation Process (Per API)**
<!-- START: DO NOT EDIT -->
For each of the 43 Alpha Vantage APIs and IBKR data feeds:

**Discovery Phase:**
1. Read API documentation thoroughly
2. Make test call with real symbol (e.g., SPY)
3. Save complete response to JSON file
4. Analyze response structure and data types
5. Document actual rate limits observed
6. Note any special parameters or quirks

**Implementation Phase:**
1. Create API method in appropriate client
2. Add configuration to config/apis/
3. Implement error handling and retry logic
4. Test with multiple symbols and edge cases

**Schema Evolution:**
1. Design table schema based on actual response
2. Create migration script for new table
3. Run migration to create table
4. Create indexes for expected query patterns

**Ingestion Phase:**
1. Implement ingestion method for this API
2. Add data validation rules
3. Implement normalization if needed
4. Test data persistence to database
5. Verify data retrieval works correctly

**Integration Phase:**
1. Add to scheduler if recurring calls needed
2. Configure rate limiting parameters
3. Update documentation with findings
4. Commit code with comprehensive tests
<!-- END: DO NOT EDIT -->

### **3.3 Daily Operational Procedures**

#### **3.3.1 Pre-Market Checklist (Before 9:00 AM ET)**
- Verify IBKR connection active
- Check Alpha Vantage API key valid
- Review overnight position changes
- Check earnings calendar for holdings
- Verify risk parameters loaded from config
- Clear stale cache entries
- Backup database
- Start monitoring dashboard
- Review any overnight alerts

#### **3.3.2 Market Open Procedures (9:30 AM ET)**
- Enable trading for all strategies
- Verify data feeds operational
- Check initial Greeks validation
- Monitor first decision cycle
- Verify Discord webhook active

#### **3.3.3 Intraday Monitoring**
- **Every 5 minutes:**
  - Check portfolio Greeks
  - Verify API usage < 500/min
  - Monitor cache hit rate
- **Every 30 minutes:**
  - Review decision quality scores
  - Check for rejected trades
  - Monitor position P&L
- **3:40 PM:** Begin MOC imbalance monitoring
- **3:50 PM:** Final MOC decision window

#### **3.3.4 Market Close Procedures (4:00 PM ET)**
- Close all 0DTE positions by 3:30 PM
- Final MOC execution by 3:55 PM
- Archive trade data
- Generate daily performance report
- Check tomorrow's earnings calendar
- Run database backup
- Review error logs
- Update configuration if needed

### **3.4 Emergency Procedures**

#### **3.4.1 Market Crisis Response**
1. **Detection:** VIX > 40 or SPY down > 3%
2. **Actions:**
   - Hit emergency stop button
   - Close all positions immediately
   - Disable new trade generation
   - Set system to monitor-only mode
   - Manually manage remaining positions
3. **Documentation:** Log all actions taken
4. **Recovery:** Only resume after volatility normalizes

#### **3.4.2 System Failure Response**
1. **Detection:** Any critical module failure
2. **Actions:**
   - Note all open positions
   - Log into IBKR TWS directly
   - Manually manage positions
   - Diagnose issue offline
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

---

## **4. Scheduling & Rate Management**

### **4.1 API Rate Management**
- **Alpha Vantage:** Target < 500/min, hard limit 600/min
- **Token Bucket:** 10 tokens/second refill rate
- **Burst Capacity:** 20 tokens maximum
- **IBKR Subscriptions:** Maximum 50 concurrent
- **Priority Order:** Positions > Tier A > Tier B > MOC > Tier C

### **4.2 Polling Schedules (Configurable)**
All schedules defined in `config/data/schedules.yaml`:

**Tier A (SPY, QQQ, IWM, SPX):**
- Options with Greeks: Every 12 seconds
- RSI, MACD, BBANDS, VWAP: Every 60 seconds
- ATR, ADX: Every 5 minutes
- Analytics: Every 5 minutes

**Tier B (MAG7 Stocks):**
- Options with Greeks: Every 45 seconds
- Core indicators: Every 5 minutes
- Analytics: Every 15 minutes

**Tier C (Watchlist):**
- Options with Greeks: Every 3 minutes
- Indicator bundle: Every 10 minutes

**MOC Window (3:40-3:55 PM):**
- Elevate relevant symbols to 5-second updates
- Increase IBKR subscription priority

### **4.3 API Call Budget**
```
Tier A: ~240 calls/minute (4 symbols × 60 calls)
Tier B: ~105 calls/minute (7 symbols × 15 calls)
Tier C: ~30 calls/minute (variable)
Other: ~25 calls/minute (news, fundamentals, economic)
Total: ~400 calls/minute (200 call buffer remaining)
```

---

## **5. Risk & Performance Gates**

### **5.1 Configuration-Based Risk Limits**
All limits defined in `config/risk/` directory:

**Position-Level Limits:**
- Max Delta: 0.80 (configurable)
- Min Delta: 0.20 (configurable)
- Max Gamma: 0.20 (configurable)
- Max Vega: 200 (configurable)
- Min Theta Ratio: 0.02 (configurable)

**Portfolio-Level Limits:**
- Max Net Delta: 0.30
- Max Net Gamma: 0.75
- Max Net Vega: 1000
- Max Net Theta: -500
- Max Capital at Risk: 20%

**Circuit Breakers:**
- Daily Loss: 2% (triggers halt)
- Weekly Loss: 5% (triggers review)
- Max Drawdown: 10% (triggers shutdown)

### **5.2 Performance Targets**

#### **5.2.1 System Performance**
| Metric | Target | Maximum | Test Method |
|--------|--------|---------|-------------|
| Decision Latency | < 1s | 2s | Time from data to decision |
| Order Submission | < 500ms | 1s | Decision to IBKR |
| Greeks Validation | < 100ms | 200ms | Validation cycle time |
| Database Query | < 100ms | 500ms | Complex query benchmark |
| Cache Retrieval | < 10ms | 50ms | Redis GET operation |
| API Response | < 2s | 5s | Alpha Vantage call |

#### **5.2.2 Trading Performance (Paper Mode)**
| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Win Rate | 45% | 55% | 65% |
| Profit Factor | 1.2 | 1.5 | 2.0 |
| Sharpe Ratio | 1.0 | 1.5 | 2.0 |
| Max Drawdown | 15% | 10% | 5% |
| Daily VaR (95%) | 3% | 2% | 1% |

### **5.3 Go/No-Go Production Criteria**
Must **ALL** be true before production deployment:

**Technical Requirements:**
- [ ] All module skeletons implemented and tested
- [ ] All 43 Alpha Vantage APIs working
- [ ] IBKR real-time data stable for 5 days
- [ ] Rate limiting never exceeded in testing
- [ ] All configurations externalized to YAML
- [ ] Schema matches all API responses exactly

**Operational Requirements:**
- [ ] 5+ consecutive days successful paper trading
- [ ] Win rate consistently > 45%
- [ ] All circuit breakers tested and working
- [ ] Emergency stop procedure validated
- [ ] Backup and recovery procedures tested
- [ ] Documentation complete and current

**Risk Management:**
- [ ] Greeks validation catching bad data
- [ ] Position limits enforced correctly
- [ ] Portfolio limits working
- [ ] Stop losses executing properly
- [ ] Daily loss breaker tested
- [ ] Manual override procedures working

---

## **6. Failure Detection & Recovery**

### **6.1 Connection Failures**
| Failure | Detection Method | Recovery Action | Testing Method |
|---------|-----------------|-----------------|----------------|
| IBKR Disconnect | No heartbeat 10s | Reconnect with exponential backoff | Kill TWS process |
| AV API Down | HTTP 500/503 errors | Use cached data (max 30s old) | Block API endpoint |
| Database Down | Connection timeout | Queue writes to Redis | Stop PostgreSQL |
| Redis Down | Connection refused | Continue without cache | Stop Redis service |
| Network Loss | Multiple failures | Switch to emergency mode | Disconnect network |

### **6.2 Data Quality Failures**
| Failure | Detection Method | Recovery Action | Testing Method |
|---------|-----------------|-----------------|----------------|
| Stale Greeks | Age > 30 seconds | Halt new trades, use last valid | Delay API response |
| Price Divergence | IBKR/cached > 0.5% | Use IBKR, log for investigation | Inject divergent prices |
| Missing Indicators | Null values in response | Skip signal, retry next cycle | Delete from cache |
| Invalid Greeks | Outside valid bounds | Reject entire chain | Inject invalid values |
| Incomplete Data | Missing required fields | Use previous valid data | Corrupt API response |

### **6.3 Execution Failures**
| Failure | Detection Method | Recovery Action | Testing Method |
|---------|-----------------|-----------------|----------------|
| Order Rejected | IBKR error code | Log, alert, retry with modifications | Submit invalid order |
| Partial Fill | Fill < requested | Adjust monitoring for actual size | Limit order liquidity |
| Stop Loss Failed | Price through stop | Market order immediately | Simulate gap down |
| Position Mismatch | DB != IBKR | Reconcile with IBKR as source | Modify DB directly |
| Execution Timeout | No fill in 30s | Cancel and retry | Delay fill confirmation |

---

## **7. Strategy Operational Specs**

### **7.1 0DTE Strategy Operations**
**Entry Window:** 09:45 - 14:00 ET  
**Auto-Close:** 15:30 ET  
**Max Positions:** 3 concurrent  
**Confidence Required:** ≥ 0.75  

**Operational Rules (from config):**
- RSI between 30-70
- Delta between 0.25-0.75
- Gamma < 0.20
- Theta/Price ratio ≥ 0.03
- IV percentile > 20
- Volume ≥ 0.5× average
- Bid-ask spread < 0.10

**Monitoring Requirements:**
- Check Greeks every 30 seconds
- Update stop loss if profitable
- Close if confidence drops below 0.60

### **7.2 1DTE Strategy Operations**
**Entry Window:** 09:45 - 15:00 ET  
**Hold Overnight:** Yes  
**Max Positions:** 5 concurrent  
**Confidence Required:** ≥ 0.70  

**Operational Differences from 0DTE:**
- Wider delta range: 0.20-0.80
- Lower theta ratio acceptable: 0.02
- Can hold overnight with hedging

### **7.3 14DTE Swing Operations**
**Entry:** Any time during market hours  
**Hold Period:** 1-14 days  
**Max Positions:** 10 concurrent  
**Confidence Required:** ≥ 0.65  

**Daily Review Requirements:**
- Check fundamentals for changes
- Review upcoming earnings
- Adjust stops based on volatility
- Consider rolling if profitable

### **7.4 MOC Imbalance Operations**
**Active Window:** 15:40 - 15:55 ET  
**Decision Time:** By 15:50 ET  
**Execution:** 15:50 - 15:55 ET  
**Min Imbalance:** $10M normalized  

**Operational Process:**
1. Begin monitoring at 15:40
2. Calculate normalized imbalance
3. Check option setup feasibility
4. Score opportunity
5. Execute if score ≥ 0.70
6. Use straddle if IV low, directional if high

---

## **8. Testing & Verification Procedures**

### **8.1 API Testing Protocol**
Each API must pass these tests before integration:

1. **Functional Testing:**
   - Successful call with valid symbol
   - Correct response parsing
   - Proper error handling
   - Retry logic working

2. **Rate Limit Testing:**
   - Stays within limits under load
   - Token bucket prevents bursts
   - Graceful degradation

3. **Data Quality Testing:**
   - Response matches expected schema
   - Data types are correct
   - Values within reasonable ranges
   - Timestamps are valid

4. **Persistence Testing:**
   - Data saves to correct table
   - Indexes work efficiently
   - Retrieval queries perform well
   - Cache operations work

### **8.2 Integration Testing Requirements**

**Phase 1: Skeleton Testing**
- All modules initialize without errors
- Dependencies resolve correctly
- Configuration loads properly
- No circular imports

**Phase 2: API Integration Testing**
- Each API integrates with ingestion
- Data flows to database correctly
- Cache layer works as expected
- Scheduler triggers appropriately

**Phase 3: End-to-End Testing**
- Complete decision cycle works
- Risk checks enforce limits
- Orders route to paper account
- Monitoring captures all events

### **8.3 Paper Trading Validation**

**Week 1: Basic Functionality**
- All data feeds working
- Decisions being made
- Orders executing
- Positions tracked

**Week 2: Strategy Validation**
- Each strategy triggers correctly
- Rules apply as configured
- Exits work properly
- P&L calculated accurately

**Week 3: Risk Management**
- Position limits enforced
- Portfolio Greeks tracked
- Stop losses trigger
- Circuit breakers work

**Week 4: Stress Testing**
- High volatility scenarios
- Multiple concurrent positions
- Rapid market moves
- API failures handled

**Week 5: Performance Validation**
- Meet win rate targets
- Achieve profit factor goals
- Stay within drawdown limits
- Complete stability test

---

## **9. Configuration Management**

### **9.1 Configuration Change Process**
1. **Propose Change:** Document what and why
2. **Test in Dev:** Validate in development environment
3. **Test in Paper:** Run in paper trading for 24 hours
4. **Review Results:** Analyze impact on performance
5. **Deploy to Production:** Apply during market close
6. **Monitor:** Watch closely for first trading day
7. **Rollback Plan:** Keep previous config for quick revert

### **9.2 Configuration Backup**
- **Frequency:** Daily after market close
- **Retention:** 30 days of configs
- **Location:** Version controlled in Git
- **Testing:** Monthly restore drill

### **9.3 Environment-Specific Configs**
```
development.yaml: Liberal limits, verbose logging
paper.yaml: Production-like with safety margins
production.yaml: Conservative limits, strict validation
```

---

## **10. Maintenance & Monitoring**

### **10.1 Daily Maintenance Tasks**
- **Pre-Market:** 
  - Clear cache of stale entries
  - Verify configurations loaded
  - Check API connectivity
- **Post-Market:**
  - Archive trade data
  - Backup database
  - Review error logs
  - Update documentation

### **10.2 Weekly Maintenance Tasks**
- Review and tune strategy parameters
- Analyze losing trades for patterns
- Check for API changes or updates
- Performance optimization review
- Update risk limits if needed

### **10.3 Monthly Maintenance Tasks**
- Full system backup and restore test
- Review ML model performance
- Analyze strategy effectiveness
- Update symbol lists
- Capacity planning review

### **10.4 Monitoring Requirements**

**Real-Time Monitoring:**
- API call rate (must stay < 500/min)
- Decision latency (must stay < 2s)
- Portfolio Greeks (within limits)
- Position P&L (stop loss levels)
- System resources (CPU, memory, disk)

**Daily Reports:**
- Trades taken and outcomes
- Win rate and profit factor
- API usage statistics
- Error summary
- Configuration changes

**Weekly Reports:**
- Strategy performance breakdown
- Risk metrics analysis
- System stability metrics
- Optimization opportunities

---

## **11. Phase Deliverables (Updated)**

### **11.1 Development Phase Deliverables**

| Phase | Deliverables | Success Criteria |
|-------|--------------|------------------|
| **0: Infrastructure** | - Complete module skeleton<br>- Config management system<br>- Base classes implemented<br>- Git repository initialized | All modules import without errors |
| **1: Connections** | - IBKR connection working<br>- AV client with rate limiting<br>- Connection recovery logic<br>- Test harness for APIs | Can fetch data from both sources |
| **2: API Implementation** | - All 43 AV APIs tested<br>- IBKR data feeds working<br>- Schema evolved per API<br>- Ingestion pipeline complete | Each API stores data correctly |
| **3: Data Management** | - Scheduler operational<br>- Cache layer working<br>- Data validation active<br>- Rate limits enforced | Never exceeds 600 calls/min |
| **4: Analytics** | - Indicator processing<br>- Greeks validation<br>- Analytics engine<br>- Signal generation | Accurate calculations verified |
| **5: ML Integration** | - Feature builder working<br>- Models loaded<br>- Predictions generated<br>- Confidence scores calculated | Deterministic outputs |
| **6: Decision Engine** | - All strategies implemented<br>- Decision logic complete<br>- Configuration-driven rules<br>- Decision logging | Correct decisions made |
| **7: Risk & Execution** | - Risk limits enforced<br>- Position sizing working<br>- Orders execute (paper)<br>- Trade monitoring active | Paper trades successful |
| **8: Output Layer** | - Discord publishing<br>- Dashboard API<br>- Monitoring active<br>- Alerts working | Real-time updates working |
| **9: Integration** | - 5-day paper trading<br>- Performance validated<br>- All tests passing<br>- Documentation complete | Ready for production |

### **11.2 Operational Readiness Checklist**

**Before Production Launch:**
- [ ] All APIs implemented and tested
- [ ] Configuration fully externalized
- [ ] 5+ days successful paper trading
- [ ] Performance targets met
- [ ] Risk management validated
- [ ] Emergency procedures tested
- [ ] Documentation complete
- [ ] Team trained on procedures
- [ ] Rollback plan ready
- [ ] Monitoring dashboard operational

---

## **12. Critical Success Factors**

### **12.1 Technical Success Factors**
- **Data Quality:** Greeks always validated before use
- **Rate Limiting:** Never exceed 600 calls/minute
- **Latency:** Decisions made within 2 seconds
- **Reliability:** 99.9% uptime during market hours
- **Accuracy:** Schema matches API responses exactly

### **12.2 Operational Success Factors**
- **Configuration:** No hardcoded values anywhere
- **Testing:** Each component tested in isolation
- **Documentation:** Every API response documented
- **Monitoring:** Real-time visibility into system state
- **Recovery:** Can recover from any failure mode

### **12.3 Business Success Factors**
- **Performance:** Win rate > 45% consistently
- **Risk:** Never exceed position or portfolio limits
- **Scalability:** Can add new strategies via config
- **Flexibility:** Can adjust parameters without code changes
- **Compliance:** All trades logged and auditable

---

## **Appendix A: Configuration File Locations**

```
config/
├── .env                           # API keys and secrets
├── system/database.yaml           # Database connections
├── system/redis.yaml              # Cache configuration
├── system/logging.yaml            # Logging settings
├── system/paths.yaml              # Directory paths
├── apis/alpha_vantage.yaml        # AV API settings
├── apis/ibkr.yaml                 # IBKR settings
├── apis/rate_limits.yaml          # Rate limiting
├── data/symbols.yaml              # Symbol tiers
├── data/schedules.yaml            # Polling schedules
├── data/validation.yaml           # Validation rules
├── strategies/0dte.yaml           # 0DTE parameters
├── strategies/1dte.yaml           # 1DTE parameters
├── strategies/swing_14d.yaml      # Swing parameters
├── strategies/moc_imbalance.yaml  # MOC parameters
├── risk/position_limits.yaml      # Position limits
├── risk/portfolio_limits.yaml     # Portfolio limits
├── risk/circuit_breakers.yaml     # Emergency stops
├── risk/sizing.yaml               # Position sizing
├── ml/models.yaml                 # Model paths
├── ml/features.yaml               # Feature specs
├── ml/thresholds.yaml             # ML thresholds
├── execution/trading_hours.yaml   # Market hours
├── execution/order_types.yaml     # Order settings
├── monitoring/alerts.yaml         # Alert rules
├── monitoring/discord.yaml        # Discord settings
└── environments/*.yaml            # Environment overrides
```

---

## **Appendix B: API Implementation Order**

Priority 1 (Week 1-2):
1. IBKR Quotes
2. IBKR 1-min bars
3. IBKR 5-min bars
4. AV REALTIME_OPTIONS (Greeks)

Priority 2 (Week 3-4):
5. AV RSI
6. AV MACD
7. AV BBANDS
8. AV VWAP
9. AV ATR
10. AV ADX

Priority 3 (Week 5-6):
[Continue through all 43 APIs...]

---

## **END OF SSOT-OPS.MD v2.0**