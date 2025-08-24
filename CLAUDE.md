# Options Trading System with Community Platform

## System Identity
High-frequency OPTIONS and equity trading system with real-time Greeks-based risk management, integrated with Discord/Whop for community monetization. Processes 5-second bars with sub-50ms execution latency.

## Core Capabilities

### Options Trading Engine
- **Real-time Greeks**: Delta, Gamma, Vega, Theta, Rho calculation
- **VPIN Monitoring**: Options flow toxicity (threshold 0.7)
- **Options Strategies**: Single legs, spreads, complex strategies
- **0DTE Management**: Automatic closure at 3:59 PM
- **MOC Trading**: Special 3:40-4:00 PM window processing
- **Chain Processing**: Handle 1000+ option contracts per symbol

### Data Integration
- **IBKR**: Real-time options chains, 5-second bars, execution
- **Alpha Vantage**: 36 APIs including REALTIME_OPTIONS, HISTORICAL_OPTIONS
- **Priority System**: Options data = CRITICAL, updated every 30 seconds

### Risk Management
**Portfolio Greeks Limits (ENFORCED REAL-TIME):**
- Delta: ±0.3
- Gamma: ±0.75  
- Vega: ±1000
- Theta: > -500

**Position Limits:**
- Max 20 concurrent positions
- Max $50,000 per position
- Daily loss limit: $10,000
- VPIN threshold: 0.7

### Community Platform
**Discord Integration:**
- Real-time signal broadcasting (<2 second latency)
- Tiered channels (FREE/PREMIUM/VIP)
- Automated daily recaps (4:30 PM ET)
- Morning analysis (8:30 AM ET)

**Subscription Tiers:**
| Tier | Price | Signal Delay | Max Signals/Day |
|------|-------|--------------|-----------------|
| FREE | $0 | 5 minutes | 5 |
| PREMIUM | $99 | 30 seconds | 20 |
| VIP | $499 | Instant | Unlimited |

**Community Features:**
- Leaderboards (Daily/Weekly/Monthly)
- Paper trading competitions
- Signal copying (mirror/proportional/fixed)
- Performance transparency

## Architecture

### Trading Pipeline (<50ms total)
```
IBKR Options Data (5ms) → 
Feature Engine + Greeks (15ms) → 
XGBoost Model (10ms) → 
Risk Check (5ms) → 
Order Execution (15ms)
```

### Community Pipeline (<2 seconds)
```
Trade Event → 
Event Broadcaster (<100ms) → 
Message Queue (<1s) → 
Discord Bot (<500ms) → 
Subscriber Channels (<500ms)
```

## Critical Performance Metrics

| Component | Target | Critical Path |
|-----------|--------|---------------|
| Options Greeks Calc | <5ms | YES |
| VPIN Calculation | <3ms | YES |
| Feature Calculation | <15ms | YES |
| Model Inference | <10ms | YES |
| Risk Validation | <5ms | YES |
| Order Execution | <15ms | YES |
| **TOTAL** | **<50ms** | **CRITICAL** |

## Database Schema Key Tables

### Options-Specific
- `options_data`: Real-time chains with Greeks
- `options_trades`: Executed options trades
- `options_positions`: Current positions with Greeks
- `vpin_metrics`: Flow toxicity measurements
- `expiry_calendar`: Expiration management

### Community
- `community_members`: User subscriptions
- `signal_broadcasts`: Distribution tracking
- `leaderboard_stats`: Performance rankings
- `paper_trades`: Competition trades

## API Rate Limits

### Alpha Vantage (500 calls/minute)
- REALTIME_OPTIONS: Every 30 seconds (CRITICAL)
- HISTORICAL_OPTIONS: Every 30 seconds (CRITICAL)
- Technical Indicators: 30 sec (critical) / 5 min (others)
- Other endpoints: 5-15 minute updates

### IBKR
- No hard limit but throttle at 50 messages/second
- Options chain updates: Real-time
- Greeks calculation: With every tick

## Development Priorities

1. **Options Data Pipeline** - Must handle high-volume chains efficiently
2. **Greeks Calculation** - Real-time with <5ms latency
3. **VPIN Monitor** - Critical for toxicity detection
4. **Risk Enforcement** - Never bypass Greeks limits
5. **Community Broadcasting** - <2 second signal delivery
6. **0DTE Management** - Automated end-of-day handling

## Error Codes

| Code | Description | Action |
|------|-------------|--------|
| E001 | IBKR connection lost | Reconnect with backoff |
| E002 | Alpha Vantage rate limit | Throttle to critical APIs |
| E003 | Model confidence low | Use fallback model |
| E004 | Greeks limit breached | Halt trading immediately |
| E005 | Database connection failed | Cache-only mode |
| E006 | Discord API error | Retry with backoff |
| E007 | Insufficient funds | Reduce position size |
| E008 | Symbol halted | Cancel pending orders |
| E009 | VPIN > 0.7 | Close all positions |
| E010 | 0DTE not closed | Force closure at 3:59 PM |

## Testing Requirements

- Options pricing accuracy validation
- Greeks calculation verification
- VPIN threshold testing
- 0DTE closure automation
- Community broadcast latency
- Load testing for 10,000+ subscribers
- Paper trading validation (5+ days minimum)

## Compliance & Regulations

- Options market regulations
- Pattern day trader rules
- Options settlement (T+1)
- Exercise/assignment handling
- Margin requirements
- Community: GDPR, payment processing