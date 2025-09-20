# QuantiCity Capital – Implementation Plan

## Phase Scoreboard
| Phase | Status | Delivered Scope Summary | Verification Evidence | Remaining Gaps |
|-------|--------|-------------------------|-----------------------|----------------|
| **1 – Schema & Configuration** | ✅ Complete | Canonical Redis key helpers, unified module toggles, cadence/TTL controls, bootstrap wiring in `main.py` | `redis_keys.py`, `config/config.yaml`, config load path in `main.py` | Expand schema docs, `.env` templates, automated config validation |
| **2 – Dealer-Flow Analytics** | ✅ Complete | DealerFlowCalculator, metrics aggregation into symbol/sector/portfolio snapshots, hedging elasticity metrics, analytics scheduler | `src/dealer_flow_calculator.py`, `src/analytics_engine.py`, dashboards TTL checks | Regression coverage for edge cases, performance profiling |
| **3 – Flow Clustering & Volatility Regimes** | ✅ Complete | FlowClusterModel, VolatilityMetrics, VIX1D regime tagging, analytics integration | `src/flow_clustering.py`, `src/volatility_metrics.py`, scheduler logs | Validate clustering on expanded universes, regime calibration tooling |
| **4 – Signal Integration & Scoring** | ✅ Complete | Feature reader refresh, 0/1/14DTE + MOC playbooks, exposure-aware dedupe, execution acknowledgements, observability metrics | `src/signal_generator.py`, `src/dte_strategies.py`, `src/moc_strategy.py`, `src/signal_deduplication.py` | Strategy unit/backtest harness, tuning guidance |
| **5 – Backfills & Monitoring** | ✅ Complete | Replay utilities for analytics history, monitoring scaffolds, regression coverage for replay paths | Replay scripts, monitoring heartbeats in Redis | Update legacy tests, automate replay validation, alerting for stalled jobs |
| **6 – Execution & Distribution Hardening** | ✅ Complete | Notional-aware sizing, commission normalization, bracket/trailing rebuilds, executed-only distribution, daily P&L reconciliation | `src/execution_manager.py`, `src/position_manager.py`, `src/signal_distributor.py`, `reconcile_daily_pnl.py` | Automated reconciliation checks, expanded risk regression coverage, execution dashboards |
| **7 – Community Publishing & Automation** | ⚠️ In Planning | Scaffolds for social bots, dashboards, morning automation, reporting wired to config + Redis | `src/discord_bot.py`, `src/telegram_bot.py`, `src/twitter_bot.py`, `src/morning_scanner.py`, `src/report_generator.py` | Implement publishing logic, provision credentials, observability, integration tests |

## Delivered Implementation Detail

### Schema, Configuration & Bootstrapping (Phase 1)
- **Unified keyspace helpers**: `redis_keys.py` formalizes namespaces for market data, analytics, signals, risk, monitoring, and system status; every module now uses typed helper functions (`market_book_key`, `analytics_symbol_key`, etc.).
- **Configuration surface**: `config/config.yaml` exposes toggles (`modules.*`), cadences (`analytics.cadence`, `modules.data_ingestion.store_ttls`), thresholds (risk, trailing stops), and distribution tier delays. YAML schema aligns with `.env` expectations.
- **Bootstrap flow**: `main.py` loads config, instantiates Redis clients, and wires module factories so individual services can be enabled/disabled without code edits.
- **Operational artifacts**: Logging defaults defined in `logging_utils.py`, CLI utilities (`clear_distribution_queues.py`, `provision_discord.py`) consume the same config to avoid drift.

### Dealer-Flow Analytics (Phase 2)
- **DealerFlowCalculator** (`src/dealer_flow_calculator.py`): Computes Vanna, Charm, hedging elasticity, skew, and maintains rolling statistics stored under `analytics:{symbol}:{metric}`.
- **Aggregation layer** (`src/analytics_engine.py`): Schedules calculator execution, consolidates contract-level readings into symbol, sector, and portfolio snapshots, and enforces TTLs outlined in config.
- **Cache hygiene**: Analytics outputs respect `modules.data_ingestion.store_ttls.analytics`, ensuring downstream modules (signals, dashboards) consume consistent payloads.
- **Documentation**: Architecture, cadences, and Redis touchpoints for analytics are captured in `docs/analytics_module.md` (paired with the ingestion deep dive).
- **Monitoring**: Heartbeats and state flags updated via `analytics_engine` for operations observability.
- **Participant flow metrics** (`src/pattern_analyzer.py`): Flow toxicity pipeline now classifies institutional vs retail flow heuristically and stores `analytics:{symbol}:institutional_flow` / `retail_flow` payloads for downstream scoring.

### Flow Clustering & Volatility Regimes (Phase 3)
- **FlowClusterModel** (`src/flow_clustering.py`): Consumes trade prints to categorize flow (momentum, dealer hedging, reversion) using KMeans; publishes distribution histograms.
- **VolatilityMetrics** (`src/volatility_metrics.py`): Maintains VIX1D history, computes regimes, and tracks transitions; surfaces regime metadata for signals and dashboards.
- **Integration**: Analytics engine coordinates cadence, gating within RTH, and merges clustering/regime outputs into the same symbol snapshots consumed by signal scoring.

### Signal Integration & Scoring (Phase 4)
- **Feature reader** (`src/signal_generator.py`): Hydrates option chains from the normalized `by_contract` map, merges dealer-flow, participant flow splits, flow clustering, volatility regimes, order imbalance, and unusual activity metrics into strategy-ready payloads with sane defaults.
- **Strategies**: 0/1/14 DTE (`src/dte_strategies.py`) and MOC (`src/moc_strategy.py`) re-tuned to new analytics, including contract hysteresis, strike memory, and liquidity filters.
- **Deduplication** (`src/signal_deduplication.py`): Normalizes contract fingerprints, avoids duplicate emissions, and writes to both `signals:pending:*` (distribution) and `signals:execution:*` (execution loop) queues.
- **Backpressure handling**: Signal generator implements exponential backoff on repeated failures to protect downstream services.
- **Exposure & observability**: Live-locks (`signals:live:*`), exposure caps, veto hashes, and acknowledgement metrics (`signals:ack:*`, `metrics:signals:*`) give operators visibility into blocked signals and pending orders. Details captured in `docs/signal_engine_module.md`.

### Backfills & Monitoring (Phase 5)
- **Replay utilities**: Backfill scripts rehydrate dealer-flow, flow-cluster, and VIX1D histories to accelerate warm-up for new symbols or cold starts.
- **Monitoring scaffolds**: Redis heartbeats (`heartbeat:*`), metrics hashes (`monitoring:analytics:*`, `monitoring:data:*`), and alert queues seeded for future dashboards (Phase 7) with TTL governance.
- **Regression coverage**: Legacy replay paths ported to new module interfaces to ensure analytics stay consistent after refactors.

### Execution, Risk & Distribution (Phase 6)
- **ExecutionManager** (`src/execution_manager.py`): Implements notional-aware sizing, risk gating (25% buying power guardrail), bracket placement, trailing stop maintenance, manual fill backfill, and automated archive alerts for missing executions.
- **PositionManager & RiskManager**: Trailing-stop orchestration, circuit breaker for consecutive losses, timezone-aware daily loss streak resets.
- **Distribution** (`src/signal_distributor.py`): Routes executed signals to premium/basic/free queues, enforces tiered delays, and records delivery metrics; only emits after confirmed fills to prevent leakage.
- **Reconciliation** (`reconcile_daily_pnl.py`): Normalizes commissions, syncs realized P&L with IBKR statements, stores results for downstream reporting.

### Community Publishing & Automation Scaffolds (Phase 7)
- **Social bots** (`src/discord_bot.py`, `src/twitter_bot.py`, `src/telegram_bot.py`): Constructor wiring for shared config/Redis, TODO placeholders for credential loading, tiered messaging, retry logic, and metrics.
- **Dashboards** (`src/dashboard_server.py`, `src/dashboard_routes.py`, `src/dashboard_websocket.py`): FastAPI skeletons, route placeholders, websocket scaffolding, logging hooks.
- **Morning automation** (`src/morning_scanner.py`, `src/news_analyzer.py`): GPT-driven analysis workflow outlines, data ingestion placeholders, queue publishing stubs.
- **Archival/reporting** (`src/report_generator.py`): Stubs for retention management, historical exports, compliance-ready reporting.

## Remaining Implementation Backlog

### 1. Schema & Configuration
- Publish `.env` templates per optional module (social, dashboards, automation) with documented variable descriptions.
- Implement config validation CLI to detect missing toggles/thresholds before deployment.
- Generate schema reference doc from `redis_keys.py` to keep documentation synchronized with code.

### 2. Dealer-Flow Analytics
- Expand unit tests for edge cases (illiquid strikes, empty greeks, missing chain snapshots).
- Benchmark calculator performance with expanded symbol universe (sector ETFs, single names) and adjust cadence if required.
- Add analytics drift detection (alert if Vanna/Charm deviates beyond configurable z-score thresholds).

### 3. Flow Clustering & Volatility Regimes
- Validate clustering accuracy on new datasets; consider adaptive clustering (online updates) if symbol mix changes intraday.
- Surface regime transition alerts into monitoring system with operator notifications.
- Build calibration tool for regime thresholds with historical Monte Carlo simulations.

### 4. Signal Engine & Strategies
- Develop automated backtest harness covering 0/1/14 DTE and MOC strategies with historical replay datasets.
- Add per-strategy unit tests for gating, weighting, and fallback logic.
- Document tuning guidelines (confidence weight sliders, imbalance thresholds, liquidity filters) and include operator playbooks.
- Implement real-time feature drift detection to trigger safe-mode reductions when inputs degrade.

### 5. Backfills & Monitoring
- Migrate remaining legacy tests importing deprecated modules (e.g., `tests/test_day6.py`) to new interfaces.
- Automate replay validation in CI (seed Redis with fixtures, run backfill, assert key outputs).
- Add alerts for stalled backfill jobs and heartbeat monitors for replay workers.
- Integrate monitoring metrics into forthcoming dashboards (Phase 7) with charts/thresholds.

### 6. Execution & Risk
- Automate daily reconciliation checks with diff reports and Slack/alert notifications for discrepancies.
- Expand regression coverage for risk scenarios (loss streak breaker, exposure guardrail, manual override flows).
- Instrument execution metrics (fill latency, slippage, failure rates) for dashboards and weekly reports.
- Implement safe shutdown/restart procedure tooling to snapshot pending orders/positions before maintenance windows.

### 7. Distribution & Client Channels
- Build entitlement-aware tier management (premium/basic/free) with retry and dead-letter handling for each queue.
- Implement delivery metrics dashboard (per channel success/fail counts, retry rates, subscriber analytics).
- Add content templating system for distribution modules to ensure consistent messaging formats.

### 8. Community Publishing & Automation (Phase 7 Execution)
- **DiscordBot**: Implement webhook rotation, message templating, rate-limit backoff, delivery metrics, SLA alerts.
- **TelegramBot**: Stripe subscription integration, entitlement store, tiered broadcast scheduling, command handlers.
- **TwitterBot**: Credential management, tweet scheduling, media handling, compliance logging.
- **Dashboard**: Complete FastAPI endpoints, authentication (JWT or OAuth), WebSocket streaming for real-time metrics, frontend integration (if applicable).
- **Morning Automation**: Data ingestion (macro, futures, options positioning), GPT orchestration with prompt templating, premium/public queue publication, compliance review logging.
- **Reporting**: Historical archives, retention policies, compliance exports, weekly/monthly performance packages, S3 (or equivalent) storage integration.
- **Observability**: Add heartbeats, metrics, and alerts for all new services before toggling them on.

### 9. Infrastructure & Tooling
- Provision optional dependency installation extras (`.[social]`, `.[dashboard]`, etc.) with documentation.
- Establish credentials management process (Vault/SSM) for API keys and webhook secrets.
- Extend CI pipeline to run linting, unit/integration tests, and packaging checks across optional modules.
- Plan observability stack integration (Grafana/Loki/Prometheus) once dashboards are live.

## Upcoming Milestones (Detailed)
| Target | Description | Owner(s) | Dependencies | Exit Criteria | Status |
|--------|-------------|----------|--------------|---------------|--------|
| **Sprint +1** | Discord channel live (premium + basic tiers) with retries & metrics | Distribution Squad | Discord credentials, config toggles, monitoring hooks | Messages flow from `signals:distribution:*` to Discord webhooks with <1% failure, metrics populated | Planned |
| **Sprint +1** | Regression modernization (analytics + signal + execution tests) | QA & Core Eng | Replay fixtures, CI pipeline | Legacy tests replaced, coverage ≥70% for critical modules, CI green | In progress |
| **Sprint +2** | Dashboard MVP (REST auth, WebSocket telemetry, operator views) | Platform Squad | FastAPI infra, auth decision, monitoring metrics | Operator dashboard shows ingestion/execution metrics, auth enforced, heartbeat alerts visible | Scoped |
| **Sprint +2** | Morning automation (GPT summaries + queue delivery) | Research & Automation | GPT access, data feeds, distribution queues | Pre-market report published to premium queue by 08:30 ET with fallback if GPT unavailable | Scoped |
| **Sprint +3** | Reporting & archival pipeline | Platform & Compliance | Storage backend, reconciliation outputs | Daily/weekly exports stored, retention policy automated, manual trigger CLI available | Backlog |

## Risk Register
| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|-----------|-------|
| IBKR/Alpha Vantage entitlement or rate-limit changes | Loss of market/analytics feeds | Medium | Monitor `monitoring:ibkr:metrics` & `monitoring:alpha_vantage:metrics`, auto-escalate when tokens <10, maintain fallback data sources | Ingestion Lead |
| Optional modules misconfigured when enabled | Signal leakage, failed deliveries | Medium | `.env` templates, config validation CLI, smoke tests for disabled modules, staged rollout | Platform Lead |
| Monitoring coverage gaps post Phase 7 | Blind spots for new services | High | Instrument heartbeats + delivery metrics before enabling toggles, integrate with dashboard alerts | SRE |
| GPT dependency for morning automation | Blocked content generation | Medium | Implement cached fallback summaries, monitor usage quotas, maintain manual override | Research Lead |
| Credential sprawl for social integrations | Security audit risk | Medium | Adopt centralized secret management (Vault/SSM), audit access quarterly | Security Officer |

## Dependency & Resource Tracker
- **Credentials**: Discord webhooks, Telegram bot tokens, Stripe API keys, OpenAI API keys, Twitter/X API credentials.
- **Infrastructure**: FastAPI/Uvicorn deployment target, persistent storage (S3 or equivalent) for archives, observability stack integration.
- **Data feeds**: Additional macro/economic data vendors for morning automation (TBD).
- **Tooling**: CI enhancements (containers for optional deps), monitoring dashboards (Grafana), alert routing (PagerDuty/Slack).
- **Human resources**: Distribution squad, Platform squad, SRE, Research & Automation, Compliance liaison.

## Reference Surface
- **Ingestion stack**: `src/ibkr_ingestion.py`, `src/ibkr_processor.py`, `src/av_ingestion.py`, `src/av_options.py`, `src/av_sentiment.py`
- **Analytics suite**: `src/analytics_engine.py`, `src/dealer_flow_calculator.py`, `src/flow_clustering.py`, `src/volatility_metrics.py`, `src/gex_dex_calculator.py`, `src/vpin_calculator.py`
- **Signal pipeline**: `src/signal_generator.py`, `src/dte_strategies.py`, `src/moc_strategy.py`, `src/signal_deduplication.py`
- **Execution & risk**: `src/execution_manager.py`, `src/position_manager.py`, `src/risk_manager.py`, `src/emergency_manager.py`, `reconcile_daily_pnl.py`
- **Distribution & comms**: `src/signal_distributor.py`, `src/discord_bot.py`, `src/telegram_bot.py`, `src/twitter_bot.py`, `src/morning_scanner.py`, `src/news_analyzer.py`, `src/report_generator.py`
- **Documentation**: `docs/ingestion_module.md`, `docs/analytics_module.md`
