# SSOT-Ops.md - Operational Specification v2.0
**Version:** 2.0 (Plugin Architecture)  
**Last Updated:** Current  
**Purpose:** Operational procedures for plugin-based options trading system  
**Scope:** All operational policies, deployment procedures, monitoring, and acceptance criteria

---

## 1. System Overview

### 1.1 Production Environment

**System Type:** Event-Driven Automated Options Trading System  
**Architecture:** Plugin-based with central message bus  
**Platform:** MacBook Pro (development) → Linux VPS (production)  
**APIs:** 36 Alpha Vantage endpoints + IBKR real-time data  
**Database:** PostgreSQL 15+ with event sourcing  
**Cache:** Redis 7+ for hot data  
**Deployment:** Docker containers with docker-compose  

### 1.2 Core Components

| Component | Purpose | Criticality | Restart Policy |
|-----------|---------|------------|----------------|
| Message Bus | Event distribution | CRITICAL | Always |
| Event Store | Data persistence | CRITICAL | Always |
| Plugin Manager | Component orchestration | CRITICAL | Always |
| AlphaVantage Plugin | Market data | HIGH | On-failure |
| IBKR Plugin | Execution & real-time data | CRITICAL | Always |
| Feature Engine | ML features | MEDIUM | On-failure |
| Model Server | Predictions | MEDIUM | On-failure |
| Risk Manager | Position safety | CRITICAL | Always |
| Executor | Order placement | CRITICAL | Always |

### 1.3 Data Flow Architecture

```
Alpha Vantage (36 APIs) → AlphaVantage Plugin → Message Bus → Event Store
                                                      ↓
IBKR (5-sec bars) → IBKR Plugin → Message Bus → Feature Engine
                                          ↓
                                    ML Models → Predictions
                                          ↓
                                    Risk Manager → Validated Signals
                                          ↓
                                    Executor → IBKR Orders
```

---

## 2. Development & Deployment Procedures

### 2.1 Local Development Setup

```bash
# 1. Clone repository
git clone <repository>
cd alphatrader

# 2. Create Python environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup local services
docker-compose -f docker-compose.dev.yml up -d postgres redis

# 5. Initialize database
python scripts/init_database.py

# 6. Configure environment
cp .env.template .env
# Edit .env with your API keys

# 7. Run plugin manager
python -m core.plugin_manager --config config/development.yaml

# 8. In another terminal, start specific plugins
python -m plugins.datasources.alpha_vantage
python -m plugins.datasources.ibkr
```

### 2.2 Testing Procedures

#### Unit Testing (Per Plugin)
```bash
# Test individual plugin
pytest tests/plugins/test_alpha_vantage.py -v

# Test with coverage
pytest tests/plugins/test_feature_engine.py --cov=plugins.ml.feature_engine

# Test all plugins
pytest tests/plugins/ -v
```

#### Integration Testing
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ -v

# Test specific data flow
pytest tests/integration/test_data_pipeline.py::test_api_to_features -v
```

#### Paper Trading Validation
```bash
# Start paper trading mode
python -m core.main --mode paper --config config/paper_trading.yaml

# Monitor performance
python scripts/monitor_paper_trading.py --days 5
```

### 2.3 Deployment Process

#### Phase 1: Build & Test
```bash
# 1. Run full test suite
make test-all

# 2. Build Docker images
docker build -t alphatrader:latest .

# 3. Run local integration test
docker-compose up -d
python scripts/integration_test.py
docker-compose down
```

#### Phase 2: Deploy to Production
```bash
# 1. Tag release
git tag -a v1.0.0 -m "Production release"
git push origin v1.0.0

# 2. Deploy to production server
ssh production-server
cd /opt/alphatrader

# 3. Pull latest
git pull origin main

# 4. Backup database
pg_dump alphatrader > backup_$(date +%Y%m%d_%H%M%S).sql

# 5. Deploy with zero downtime
docker-compose pull
docker-compose up -d --no-deps --build message_bus
# Wait for message bus
docker-compose up -d --no-deps --build plugin_manager
# Deploy plugins one by one
docker-compose up -d --no-deps --build alphatrader_av
docker-compose up -d --no-deps --build alphatrader_ibkr
# Continue for all plugins...
```

---

## 3. Daily Operational Procedures

### 3.1 Pre-Market Checklist (8:30 AM ET)

```yaml
pre_market_tasks:
  - name: System Health Check
    command: python scripts/health_check.py
    expected: All components GREEN
    
  - name: Verify Data Feeds
    command: python scripts/verify_feeds.py
    checks:
      - Alpha Vantage API responding
      - IBKR connection active
      - Redis cache operational
      - PostgreSQL responsive
      
  - name: Check Overnight Positions
    command: python scripts/check_positions.py
    verify:
      - All positions accounted for
      - No unexpected changes
      - Greeks within limits
      
  - name: Review Economic Calendar
    command: python scripts/economic_calendar.py
    check_for:
      - Major announcements
      - Earnings for holdings
      - Fed announcements
      
  - name: Verify Risk Limits
    command: python scripts/verify_risk_limits.py
    ensure:
      - Daily loss limit set
      - Position limits configured
      - Circuit breakers armed
```

### 3.2 Market Hours Operations (9:30 AM - 4:00 PM ET)

#### Opening Bell (9:30 AM)
```python
# Automated startup sequence
09:25: Enable pre-market data collection
09:28: Final systems check
09:30: Enable trading for all strategies
09:31: Verify first bar aggregation
09:35: Check initial signals
```

#### Continuous Monitoring
```yaml
monitoring_intervals:
  every_5_seconds:
    - IBKR bar aggregation
    - Order status updates
    
  every_30_seconds:
    - Options Greeks validation
    - VPIN calculation
    
  every_1_minute:
    - Position P&L update
    - Risk metrics calculation
    
  every_5_minutes:
    - Alpha Vantage data refresh
    - Feature recalculation
    - ML predictions update
    
  every_15_minutes:
    - Performance metrics
    - System resource check
    - Rate limit status
```

#### MOC Window (3:40 PM - 3:55 PM)
```python
# Special MOC handling
15:40: Begin MOC imbalance monitoring
15:45: Calculate imbalance signals
15:50: Final MOC decision window
15:55: Last MOC order submission
15:59: Close all 0DTE positions
```

### 3.3 Post-Market Procedures (4:00 PM ET)

```yaml
post_market_tasks:
  - name: Generate Daily Report
    command: python scripts/daily_report.py
    outputs:
      - P&L summary
      - Trade analysis
      - Risk metrics
      - Performance statistics
      
  - name: Backup Critical Data
    command: python scripts/backup_data.py
    includes:
      - Database dump
      - Event store export
      - Configuration snapshot
      - Model weights
      
  - name: Analyze Failed Trades
    command: python scripts/analyze_failures.py
    review:
      - Rejected signals
      - Failed executions
      - Risk limit breaches
      
  - name: Update Feature Importance
    command: python scripts/update_features.py
    tasks:
      - Calculate SHAP values
      - Rank feature importance
      - Flag degraded features
```

### 3.4 Weekend Maintenance

```bash
# Saturday maintenance window
#!/bin/bash

# 1. Full system backup
./scripts/full_backup.sh

# 2. Database maintenance
psql alphatrader -c "VACUUM ANALYZE;"
psql alphatrader -c "REINDEX DATABASE alphatrader;"

# 3. Clear old events (keep 90 days)
python scripts/archive_events.py --days 90

# 4. Update dependencies
pip list --outdated
pip install --upgrade -r requirements.txt

# 5. Run model retraining
python scripts/retrain_models.py --method walk_forward

# 6. Generate weekly report
python scripts/weekly_report.py
```

---

## 4. Plugin Management Operations

### 4.1 Adding New Plugins

```python
# 1. Create plugin file
# plugins/[category]/[name].py

class NewPlugin(Plugin):
    def start(self):
        # Subscribe to events
        self.bus.subscribe('event.type', self.handler)
        
    def handler(self, message):
        # Process message
        result = self.process(message.data)
        # Publish result
        self.publish('new.event', result)

# 2. Create configuration
# config/plugins/[name].yaml
new_plugin:
  enabled: true
  parameters:
    setting1: value1
    setting2: value2

# 3. Deploy without restart
curl -X POST http://localhost:8080/plugins/reload

# 4. Verify plugin loaded
curl http://localhost:8080/plugins/status
```

### 4.2 Plugin Monitoring

```yaml
# config/monitoring/plugins.yaml
health_checks:
  alpha_vantage:
    interval: 60
    timeout: 5
    checks:
      - api_calls_per_minute: "<500"
      - error_rate: "<0.01"
      - latency_p95: "<1000ms"
      
  feature_engine:
    interval: 30
    checks:
      - features_calculated: ">0"
      - null_features: "<0.05"
      - calculation_time: "<500ms"
      
  model_server:
    interval: 30
    checks:
      - predictions_per_minute: ">10"
      - confidence_avg: ">0.6"
      - model_drift: "<0.1"
```

### 4.3 Plugin Failure Recovery

```python
# Automatic recovery procedures
recovery_procedures:
  alpha_vantage:
    on_failure:
      - action: restart
        max_attempts: 3
        backoff: exponential
      - action: switch_to_backup
        if_restarts_fail: true
      - action: alert_ops
        severity: high
        
  ibkr:
    on_failure:
      - action: reconnect
        max_attempts: 5
        delay: 5
      - action: halt_trading
        if_reconnect_fails: true
      - action: page_oncall
        severity: critical
        
  model_server:
    on_failure:
      - action: fallback_model
        model: simple_rules
      - action: restart
        max_attempts: 2
      - action: alert_ops
        severity: medium
```

---

## 5. Risk Management Operations

### 5.1 Risk Monitoring Dashboard

```yaml
risk_dashboard:
  real_time_metrics:
    - current_positions:
        display: table
        refresh: 5s
        alerts:
          - position_size > max_size
          - unrealized_loss > stop_loss
          
    - portfolio_greeks:
        display: gauges
        refresh: 30s
        limits:
          delta: [-0.3, 0.3]
          gamma: [-0.75, 0.75]
          vega: [-1000, 1000]
          theta: [-500, 0]
          
    - vpin_monitor:
        display: time_series
        refresh: 60s
        alerts:
          - vpin > 0.6: warning
          - vpin > 0.7: critical
          
    - var_metrics:
        display: bar_chart
        refresh: 5m
        show:
          - var_95: $10,000
          - cvar_95: $15,000
          - max_loss_today: $2,000
```

### 5.2 Circuit Breaker Procedures

```python
# Circuit breaker configuration
circuit_breakers:
  daily_loss:
    trigger: daily_pnl < -10000
    actions:
      - halt_new_trades: true
      - close_risky_positions: true
      - notify: ["ops", "risk_team"]
      - require_manual_reset: true
      
  position_limit:
    trigger: position_count > 20
    actions:
      - prevent_new_positions: true
      - force_position_reduction: true
      
  vpin_toxic:
    trigger: vpin > 0.7
    actions:
      - close_all_positions: true
      - halt_trading: 30_minutes
      - alert_immediate: true
      
  correlation_breakdown:
    trigger: model_correlation < 0.3
    actions:
      - disable_ml_signals: true
      - switch_to_simple_rules: true
      - alert_ml_team: true
```

### 5.3 Emergency Procedures

```bash
#!/bin/bash
# Emergency stop script

# 1. IMMEDIATE: Stop all trading
curl -X POST http://localhost:8080/emergency/stop

# 2. Close all positions
python scripts/close_all_positions.py --confirm

# 3. Disable plugins
curl -X POST http://localhost:8080/plugins/disable/all

# 4. Export current state
python scripts/export_state.py --output emergency_$(date +%s).json

# 5. Notify team
python scripts/notify_team.py --severity CRITICAL --message "Emergency stop activated"
```

---

## 6. Performance Monitoring

### 6.1 Key Performance Indicators

```yaml
kpis:
  trading:
    - win_rate:
        target: ">55%"
        minimum: ">45%"
        calculation: wins / total_trades
        
    - profit_factor:
        target: ">1.5"
        minimum: ">1.2"
        calculation: gross_profit / gross_loss
        
    - sharpe_ratio:
        target: ">1.5"
        minimum: ">1.0"
        calculation: (returns - risk_free) / std_dev
        
  system:
    - api_latency_p99:
        target: "<1s"
        maximum: "<2s"
        
    - prediction_accuracy:
        target: ">60%"
        minimum: ">55%"
        
    - feature_calculation_time:
        target: "<500ms"
        maximum: "<1s"
```

### 6.2 Performance Reports

```python
# Daily performance email
daily_report:
  recipients: ["ops@company.com", "trading@company.com"]
  schedule: "17:00 ET"
  sections:
    - summary:
        - total_trades
        - win_rate
        - pnl
        - sharpe_ratio
    - top_performers:
        limit: 5
        sort_by: pnl
    - worst_performers:
        limit: 5
        sort_by: pnl
    - risk_metrics:
        - max_drawdown
        - var_95
        - largest_position
    - system_health:
        - uptime
        - api_calls_used
        - errors_count
```

### 6.3 Performance Optimization

```yaml
optimization_schedule:
  daily:
    - cache_cleanup:
        time: "02:00"
        action: "Clear expired keys"
        
  weekly:
    - database_optimization:
        day: "Saturday"
        time: "03:00"
        actions:
          - "VACUUM ANALYZE"
          - "REINDEX"
          - "Update statistics"
          
    - model_performance_review:
        day: "Sunday"
        actions:
          - "Calculate prediction accuracy"
          - "Identify degraded features"
          - "Generate retraining recommendations"
          
  monthly:
    - full_system_review:
        actions:
          - "Analyze all trades"
          - "Review strategy performance"
          - "Update risk parameters"
          - "Optimize plugin configurations"
```

---

## 7. Data Management Operations

### 7.1 Data Retention Policy

```yaml
retention_policy:
  event_store:
    hot_storage: 7_days      # In PostgreSQL
    warm_storage: 90_days    # In PostgreSQL archived tables
    cold_storage: 2_years    # In S3/Object storage
    
  market_data:
    tick_data: 30_days
    minute_bars: 1_year
    daily_bars: unlimited
    
  ml_features:
    calculated_features: 30_days
    feature_importance: 90_days
    model_predictions: 90_days
    
  trade_data:
    executions: unlimited
    order_history: unlimited
    pnl_records: unlimited
```

### 7.2 Backup Procedures

```bash
#!/bin/bash
# Backup script - runs daily at 2 AM

# 1. Database backup
pg_dump alphatrader | gzip > /backups/db/alphatrader_$(date +%Y%m%d).sql.gz

# 2. Event store backup
python scripts/backup_events.py --format parquet --output /backups/events/

# 3. Configuration backup
tar -czf /backups/config/config_$(date +%Y%m%d).tar.gz config/

# 4. Model backup
tar -czf /backups/models/models_$(date +%Y%m%d).tar.gz models/

# 5. Upload to S3
aws s3 sync /backups/ s3://alphatrader-backups/

# 6. Cleanup old local backups (keep 7 days)
find /backups/ -type f -mtime +7 -delete
```

### 7.3 Data Quality Monitoring

```python
# Data quality checks
quality_checks:
  market_data:
    - missing_bars:
        threshold: "<0.1%"
        action: "Backfill from provider"
        
    - price_spikes:
        threshold: ">10% in 1 minute"
        action: "Flag and investigate"
        
    - stale_data:
        threshold: ">60 seconds"
        action: "Alert and reconnect"
        
  features:
    - null_values:
        threshold: "<5%"
        action: "Use median imputation"
        
    - out_of_range:
        threshold: ">3 std dev"
        action: "Cap at limits"
        
  predictions:
    - confidence_distribution:
        check: "Should be normal"
        action: "Retrain if skewed"
```

---

## 8. Incident Response

### 8.1 Incident Severity Levels

```yaml
severity_levels:
  SEV1_CRITICAL:
    description: "Trading halted or major loss event"
    response_time: "Immediate"
    escalation: ["CTO", "Head of Trading"]
    examples:
      - "IBKR connection lost"
      - "Unauthorized position opened"
      - "Daily loss > $50,000"
      
  SEV2_HIGH:
    description: "Degraded performance or partial outage"
    response_time: "15 minutes"
    escalation: ["Tech Lead", "Risk Manager"]
    examples:
      - "ML predictions unavailable"
      - "High API error rate"
      - "Risk limits approaching"
      
  SEV3_MEDIUM:
    description: "Non-critical issue affecting operations"
    response_time: "1 hour"
    escalation: ["On-call Engineer"]
    examples:
      - "Delayed data feed"
      - "Dashboard unavailable"
      - "Non-critical plugin failure"
      
  SEV4_LOW:
    description: "Minor issue with no immediate impact"
    response_time: "Next business day"
    escalation: ["Team backlog"]
    examples:
      - "Logging errors"
      - "Performance degradation"
      - "UI issues"
```

### 8.2 Incident Response Runbook

```python
# Incident response automation
incident_response:
  detect:
    - monitoring_alert
    - automated_detection
    - manual_report
    
  assess:
    - determine_severity
    - identify_scope
    - estimate_impact
    
  respond:
    SEV1:
      - immediately_halt_affected_systems
      - page_on_call_immediately
      - open_war_room_channel
      - begin_recording_timeline
      
    SEV2:
      - isolate_affected_components
      - notify_on_call
      - prepare_rollback_plan
      
  recover:
    - implement_fix_or_rollback
    - verify_system_stable
    - monitor_closely_30_minutes
    
  review:
    - conduct_post_mortem
    - document_lessons_learned
    - update_runbooks
    - implement_preventive_measures
```

---

## 9. Compliance & Audit

### 9.1 Audit Logging

```python
# All trades must be logged
audit_requirements:
  trade_log:
    required_fields:
      - timestamp
      - symbol
      - action
      - size
      - price
      - strategy
      - signal_metadata
      - risk_metrics
      - execution_venue
      
  decision_log:
    required_fields:
      - timestamp
      - strategy
      - signals_evaluated
      - signals_rejected
      - rejection_reasons
      - ml_predictions
      - risk_overrides
      
  system_log:
    required_fields:
      - configuration_changes
      - plugin_enable_disable
      - risk_limit_changes
      - manual_interventions
      - emergency_stops
```

### 9.2 Compliance Reports

```yaml
compliance_reports:
  daily:
    - best_execution_report:
        compare: "execution_price vs NBBO"
        flag: "Slippage > 0.1%"
        
  monthly:
    - trade_analysis:
        include:
          - "All trades with rationale"
          - "P&L attribution"
          - "Risk metrics adherence"
          
    - system_changes:
        include:
          - "Configuration modifications"
          - "Model updates"
          - "Strategy changes"
          
  quarterly:
    - full_audit:
        performed_by: "External auditor"
        includes:
          - "Code review"
          - "Trade verification"
          - "Risk assessment"
          - "Compliance attestation"
```

---

## 10. Production Readiness Checklist

### 10.1 Pre-Production Validation

```yaml
validation_requirements:
  paper_trading:
    duration: "Minimum 14 days"
    requirements:
      - win_rate: ">45%"
      - sharpe_ratio: ">1.0"
      - max_drawdown: "<10%"
      - consecutive_profitable_days: ">5"
      
  system_testing:
    - load_testing:
        concurrent_symbols: 50
        messages_per_second: 10000
        duration: 1_hour
        
    - failure_testing:
        test_cases:
          - "Database failure"
          - "Redis failure"
          - "API rate limit"
          - "Network partition"
          - "Plugin crash"
          
    - recovery_testing:
        verify:
          - "Automatic recovery"
          - "No data loss"
          - "Position consistency"
          
  documentation:
    required:
      - "Operations manual"
      - "Incident runbooks"
      - "Architecture diagrams"
      - "API documentation"
      - "Configuration guide"
```

### 10.2 Production Launch

```bash
#!/bin/bash
# Production launch script

# 1. Final validation
echo "Running final validation..."
python scripts/validate_production_ready.py || exit 1

# 2. Create production backup
echo "Creating pre-launch backup..."
./scripts/full_backup.sh

# 3. Deploy configuration
echo "Deploying production configuration..."
cp config/production.yaml config/active.yaml

# 4. Start core services
echo "Starting core services..."
docker-compose up -d postgres redis message_bus

# 5. Wait for services
echo "Waiting for services to be healthy..."
./scripts/wait_for_healthy.sh

# 6. Start plugins
echo "Starting plugins..."
docker-compose up -d plugin_manager

# 7. Enable trading (with small size first)
echo "Enabling trading with reduced size..."
python scripts/enable_trading.py --size_multiplier 0.1

# 8. Monitor closely
echo "Starting intensive monitoring..."
python scripts/monitor_launch.py --duration 60

echo "Production launch complete!"
```

---

## 11. Key Metrics & Alerts

### 11.1 Critical Alerts

```yaml
alerts:
  trading:
    - daily_loss_approaching:
        condition: "daily_pnl < -8000"
        severity: "HIGH"
        action: "Reduce position sizes"
        
    - position_limit_breach:
        condition: "position_count > max_positions"
        severity: "CRITICAL"
        action: "Halt new positions"
        
    - vpin_critical:
        condition: "vpin > 0.7"
        severity: "CRITICAL"
        action: "Close all positions"
        
  system:
    - api_rate_limit:
        condition: "api_calls > 550/min"
        severity: "HIGH"
        action: "Throttle requests"
        
    - database_connection_pool:
        condition: "active_connections > 90%"
        severity: "MEDIUM"
        action: "Scale connection pool"
        
    - plugin_failure:
        condition: "plugin_health != healthy"
        severity: "varies_by_plugin"
        action: "Restart or failover"
```

### 11.2 Monitoring Dashboards

```yaml
dashboards:
  main_trading:
    url: "http://monitoring:3000/d/trading"
    panels:
      - "Real-time P&L"
      - "Active Positions"
      - "Win Rate (Rolling)"
      - "Risk Metrics"
      - "System Health"
      
  technical:
    url: "http://monitoring:3000/d/technical"
    panels:
      - "API Latencies"
      - "Message Bus Throughput"
      - "Plugin Health Matrix"
      - "Database Performance"
      - "Cache Hit Rates"
      
  risk:
    url: "http://monitoring:3000/d/risk"
    panels:
      - "Portfolio Greeks"
      - "VaR/CVaR"
      - "VPIN History"
      - "Position Concentration"
      - "Correlation Matrix"
```

---

## 12. Team Responsibilities

### 12.1 Role Definitions

```yaml
roles:
  trading_operator:
    responsibilities:
      - "Monitor trading dashboard"
      - "Review daily positions"
      - "Validate signals"
      - "Execute emergency stops"
    required_knowledge:
      - "Options trading"
      - "Risk management"
      - "System operations"
      
  system_engineer:
    responsibilities:
      - "Maintain infrastructure"
      - "Deploy updates"
      - "Monitor system health"
      - "Respond to incidents"
    required_knowledge:
      - "Python"
      - "Docker"
      - "PostgreSQL"
      - "Message queuing"
      
  data_scientist:
    responsibilities:
      - "Monitor model performance"
      - "Update features"
      - "Retrain models"
      - "Analyze predictions"
    required_knowledge:
      - "Machine learning"
      - "Statistics"
      - "Python"
      - "Financial markets"
```

### 12.2 On-Call Rotation

```yaml
on_call:
  schedule:
    primary: "Weekly rotation"
    secondary: "Monthly rotation"
    
  responsibilities:
    primary:
      - "First response to all alerts"
      - "Execute runbooks"
      - "Escalate if needed"
      
    secondary:
      - "Backup for primary"
      - "Major incident response"
      - "Architecture decisions"
      
  escalation_path:
    - level_1: "Primary on-call"
    - level_2: "Secondary on-call"
    - level_3: "Tech Lead"
    - level_4: "CTO"
```

---

## END OF OPERATIONAL SPECIFICATION

This operations document ensures:
1. **Clear Procedures**: Step-by-step for all operations
2. **Plugin Management**: Add/remove without downtime
3. **Risk Controls**: Multiple layers of protection
4. **Incident Response**: Clear escalation paths
5. **Monitoring**: Comprehensive observability