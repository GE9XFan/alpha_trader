# Quick Command Reference

## System Control

### Starting the System
```bash
# Start in paper trading mode (SAFE - for testing)
python -m trading_system.main --mode paper --config config/paper_trading.yaml

# Start in live trading mode (DANGEROUS - real money)
python -m trading_system.main --mode live --config config/production.yaml

# Start with specific symbols
python -m trading_system.main --symbols "SPY,QQQ,AAPL" --enable-options

# Start community features only (no trading)
python -m trading_system.main --community-only
```

### Emergency Controls
```bash
# EMERGENCY STOP - Closes all positions immediately
./scripts/EMERGENCY_STOP.sh

# Close all 0DTE positions (run at 3:59 PM)
python ops/close_0dte_positions.py --force

# Close all positions with high VPIN
python ops/emergency_close.py --reason "VPIN_EXCEEDED" --threshold 0.7

# Flatten all Greeks
python ops/flatten_greeks.py --target-delta 0 --target-gamma 0

# Pause trading but keep monitoring
python ops/pause_trading.py --duration 3600  # 1 hour

# Resume trading after pause
python ops/resume_trading.py
```

## Options Trading Operations

### Greeks Management
```bash
# Check current portfolio Greeks
python ops/show_greeks.py --format table

# Calculate Greeks for a specific position
python ops/calculate_greeks.py --symbol AAPL --strike 195 --expiry 2024-01-19 --type CALL

# Monitor Greeks in real-time
python ops/monitor_greeks.py --alert-on-breach --update-interval 5

# Rebalance portfolio Greeks
python ops/rebalance_greeks.py --target-delta 0 --max-trades 10
```

### Options Chain Analysis
```bash
# Fetch option chain
python ops/fetch_chain.py --symbol SPY --expiries all --strikes 20

# Analyze volatility surface
python ops/analyze_vol_surface.py --symbol SPY --output vol_surface.html

# Find mispriced options
python ops/find_arbitrage.py --symbols "SPY,QQQ" --min-edge 0.10

# Scan for unusual options activity
python ops/scan_unusual_options.py --min-volume 1000 --min-oi-change 500
```

### VPIN Monitoring
```bash
# Check current VPIN
python ops/check_vpin.py --symbol SPY

# Historical VPIN analysis
python ops/analyze_vpin.py --symbol SPY --period 1d --plot

# Set VPIN alerts
python ops/set_vpin_alert.py --threshold 0.6 --action notify
python ops/set_vpin_alert.py --threshold 0.7 --action close_all
```

## Risk Management

### Position Limits
```bash
# Check current positions against limits
python ops/check_limits.py --verbose

# Override risk limits (requires approval)
python ops/override_limits.py --parameter max_positions --value 25 --duration 3600 --reason "Special event"

# Show risk metrics dashboard
python ops/risk_dashboard.py --port 8080
```

### Daily Loss Management
```bash
# Check daily P&L
python ops/show_pnl.py --date today --breakdown

# Set daily loss alert
python ops/set_loss_alert.py --threshold -7500 --action notify

# Review losing trades
python ops/analyze_losses.py --date today --min-loss 100
```

## Community Platform

### Discord Bot Management
```bash
# Start Discord bot
python community/start_discord_bot.py --token $DISCORD_BOT_TOKEN

# Test signal broadcast
python community/test_signal.py --type entry --symbol AAPL --strike 195 --dry-run

# Send manual announcement
python community/send_announcement.py --channel vip --message "Market update..."

# Generate daily recap
python community/generate_recap.py --date today --send

# Update leaderboard
python community/update_leaderboard.py --period daily
```

### Whop Integration
```bash
# Sync Whop subscriptions
python community/sync_whop.py --full-sync

# Process webhook
python community/process_webhook.py --payload webhook.json

# Check user subscription
python community/check_subscription.py --user-id 12345

# Grant manual access
python community/grant_access.py --user-id 12345 --tier VIP --duration 30
```

### Competition Management
```bash
# Start paper trading competition
python community/start_competition.py --type weekly --prize-pool 1000

# Get competition standings
python community/show_standings.py --competition-id abc123

# Process competition trades
python community/process_comp_trades.py --competition-id abc123

# Announce winners
python community/announce_winners.py --competition-id abc123
```

## Data Management

### Alpha Vantage Operations
```bash
# Test Alpha Vantage connection
python ops/test_av.py --function REALTIME_OPTIONS --symbol SPY

# Check API usage
python ops/check_av_usage.py --show-remaining

# Fetch historical options data
python ops/fetch_historical_options.py --symbol SPY --start 2024-01-01 --end 2024-01-19

# Update all indicators
python ops/update_indicators.py --symbols "SPY,QQQ" --priority CRITICAL
```

### IBKR Operations
```bash
# Test IBKR connection
python ops/test_ibkr.py --port 7497  # Paper trading

# Subscribe to market data
python ops/subscribe_market_data.py --symbols "SPY,QQQ,AAPL" --data-type "TRADES,BID_ASK,OPTIONS"

# Check account status
python ops/check_account.py --show-positions --show-orders --show-balance

# Download executions
python ops/download_executions.py --date today --format csv
```

## Monitoring & Analytics

### Performance Monitoring
```bash
# Check system latency
python ops/check_latency.py --component all --alert-threshold 50

# Profile critical path
python ops/profile_critical_path.py --iterations 1000 --output profile.html

# Monitor resource usage
python ops/monitor_resources.py --duration 3600 --plot

# Generate performance report
python ops/performance_report.py --period daily --send-email
```

### Trade Analytics
```bash
# Analyze today's trades
python ops/analyze_trades.py --date today --metrics all

# Calculate Sharpe ratio
python ops/calculate_sharpe.py --period 30d

# Review slippage
python ops/analyze_slippage.py --min-slippage 0.01

# Options P&L attribution
python ops/options_pnl_attribution.py --breakdown greeks
```

## Database Operations

### Backup & Restore
```bash
# Backup database
python ops/backup_db.py --output /backups/trading_$(date +%Y%m%d).sql

# Restore database
python ops/restore_db.py --input /backups/trading_20240119.sql

# Archive old data
python ops/archive_data.py --older-than 90d --tables "trades,signals"

# Vacuum and reindex
python ops/maintain_db.py --vacuum --reindex
```

### Cache Management
```bash
# Clear Redis cache
redis-cli FLUSHDB

# Check cache stats
python ops/cache_stats.py

# Warm up cache
python ops/warmup_cache.py --data "greeks,features,chains"
```

## Testing

### Strategy Testing
```bash
# Backtest strategy
python tests/backtest.py --strategy momentum --start 2023-01-01 --end 2024-01-01

# Paper trade validation
python tests/validate_paper.py --days 5 --min-sharpe 1.0

# Run integration tests
pytest tests/integration/ -v

# Run performance benchmarks
pytest tests/performance/ --benchmark-only
```

### System Testing
```bash
# Run all tests
pytest tests/ --cov=trading_system --cov-report=html

# Test risk limits
python tests/test_risk_limits.py --scenario extreme_market

# Test circuit breakers
python tests/test_circuit_breakers.py --trigger vpin_breach

# Load testing
python tests/load_test.py --users 10000 --duration 3600
```

## Deployment

### Production Deployment
```bash
# Deploy new version
./deploy/deploy_production.sh --version 3.0.0 --rollback-enabled

# Rollback to previous version
./deploy/rollback.sh --to-version 2.9.0

# Health check
./deploy/health_check.sh --comprehensive

# Smoke tests
./deploy/smoke_tests.sh
```

## Utilities

### Log Management
```bash
# Tail trading logs
tail -f /var/log/trading/trading.log

# Search for errors
grep ERROR /var/log/trading/*.log | tail -50

# Analyze log patterns
python ops/analyze_logs.py --pattern "E00[1-9]" --period 1h

# Rotate logs
logrotate -f /etc/logrotate.d/trading
```

### Configuration
```bash
# Validate configuration
python ops/validate_config.py --config config/production.yaml

# Update configuration
python ops/update_config.py --key risk.max_positions --value 25

# Show current configuration
python ops/show_config.py --format yaml

# Diff configurations
python ops/diff_configs.py config/production.yaml config/staging.yaml
```

## Claude Code Specific

### Working with Claude Code
```bash
# Ask Claude to implement a feature
claude-code "Implement a volatility arbitrage scanner for SPY options"

# Ask Claude to debug an issue
claude-code "Debug why Greeks calculation is taking >5ms"

# Ask Claude to optimize performance
claude-code "Optimize the feature calculation to meet 15ms target"

# Ask Claude to write tests
claude-code "Write comprehensive tests for the VPIN calculator"

# Ask Claude to document code
claude-code "Document the options pricing module with examples"
```

## Keyboard Shortcuts (When Using Monitor)

- `Ctrl+C`: Emergency stop
- `F1`: Show help
- `F2`: Toggle paper/live mode
- `F5`: Refresh all data
- `P`: Pause trading
- `R`: Resume trading
- `G`: Show Greeks
- `V`: Show VPIN
- `L`: Show positions
- `T`: Show recent trades
- `Q`: Quit (with confirmation)