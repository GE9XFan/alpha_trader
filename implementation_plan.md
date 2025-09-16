# AlphaTrader Pro – Implementation Plan

## Refactor Snapshot
- **Architecture**: 30 production modules now sit behind a Redis-centric message fabric while `main.py` orchestrates configuration, logging, and shared connection pools.【F:src/main.py†L27-L192】
- **Data integrity**: The canonical Redis schema now publishes typed helper functions (`get_market_key`, `market_bars_key`, `analytics_vpin_key`, `get_system_key`, etc.) that replaced the legacy `Keys` class, so ingestion, analytics, and signal modules all construct Redis namespaces through validated helpers while sharing TTL policies.【F:src/redis_keys.py†L25-L304】【F:src/ibkr_ingestion.py†L560-L748】【F:src/gex_dex_calculator.py†L29-L520】【F:src/signal_generator.py†L520-L606】
- **Analytics coordination**: `AnalyticsEngine` seeds calculators on cadence, publishes sector/portfolio snapshots, and idles gracefully outside configured market hours via the new aggregator pipeline.【F:src/analytics_engine.py†L35-L445】
- **Exposure telemetry**: `GEXDEXCalculator` now records Redis-backed rolling histories and z-scores for exposures while `PatternAnalyzer` lengthens sweep persistence to 30 seconds so toxicity features survive bursty flow.【F:src/gex_dex_calculator.py†L47-L105】【F:src/gex_dex_calculator.py†L370-L594】【F:src/pattern_analyzer.py†L293-L354】
- **Signal pipeline**: `SignalGenerator` now injects strategy handlers/feature readers, adds loop backoff, and delegates idempotency to `SignalDeduplication`; DTE and MOC strategies share typed feature normalization, contradiction gating, and contract hysteresis while the Lua helper emits into dedicated `signals:execution` queues for the execution stack.【F:src/signal_generator.py†L48-L209】【F:src/dte_strategies.py†L13-L218】【F:src/moc_strategy.py†L61-L304】【F:src/signal_deduplication.py†L224-L262】
- **Options-first contract handling**: DTE and MOC selectors now emit normalized, option-only payloads with IBKR-formatted expiries sourced from the shared helpers; `option_utils` centralizes trading-day-aware expiry math while execution and signal modules default to options contracts, validate strikes/expiries, and derive TTLs from normalized metadata.【F:src/dte_strategies.py†L13-L218】【F:src/moc_strategy.py†L61-L304】【F:src/option_utils.py†L1-L104】【F:src/signal_generator.py†L167-L274】【F:src/execution_manager.py†L256-L372】
- **Analytics feature alignment**: The default feature reader consumes VPIN analytics from the canonical `value` field and prioritizes OBI `level1_imbalance` readings (with fallbacks for `level5_imbalance`/legacy aliases), removing the zeroed metrics that blocked signal emission and bringing the Redis schema into sync with analytics producers.【F:src/signal_generator.py†L624-L658】【F:tests/test_signal_generation_enhancements.py†L164-L180】
- **Execution & risk controls**: `ExecutionManager` caches a shared `RiskManager`, reconciles live IBKR positions, strips signal metadata from persisted orders, auto-builds stop/target brackets, and escalates partial fills/timeouts while `PositionManager` cancels bracket legs on exits and `EmergencyManager` drains both pending and execution queues with async Redis primitives.【F:src/execution_manager.py†L59-L972】【F:src/position_manager.py†L70-L799】【F:src/emergency_manager.py†L80-L279】
- **Distribution & telemetry**: `SignalDistributor` enforces validator guardrails, persistent tier scheduling, and dead-letter handling while `PerformanceTracker` captures equity curves, profit factor, Sharpe, and daily PnL for reviews.【F:src/signal_distributor.py†L27-L236】【F:src/signal_performance.py†L25-L181】
- **Testing coverage**: Async suites now cover analytics cadence, signal flow, and consolidated execution/position/risk/emergency scenarios with a lightweight Redis stub validating risk-manager caching, sanitized orders, Greek aggregation, and mass-cancel flows; these suites import the new helper functions so coverage exercises the functional Redis interface end-to-end.【F:tests/test_analytics_engine.py†L1-L216】【F:tests/test_signal_generation_enhancements.py†L1-L120】【F:tests/test_managers.py†L1-L253】
- **Focus areas**: Market ingestion, analytics, signal generation, and risk/execution are production-grade; social, dashboard, and morning-automation services remain scaffolding pending API integrations.

## Module Completion Matrix
| Group | Module | Lines | Status | Notes |
| --- | --- | --- | --- | --- |
| Data Ingestion | `ibkr_ingestion.py` | 1,070 | 100% | Processor-driven normalization with keyed metrics, TTL-managed monitoring, and clean shutdown handling.【F:src/ibkr_ingestion.py†L49-L920】 |
|  | `ibkr_processor.py` | 303 | 100% | Depth aggregation, trade buffering, and quote synthesis extracted into reusable helpers.【F:src/ibkr_processor.py†L31-L219】 |
|  | `av_ingestion.py` | 502 | 100% | Long-lived processors, circuit-breaker retries, and telemetry-rich metrics publishing in place.【F:src/av_ingestion.py†L45-L493】 |
|  | `av_options.py` | 334 | 100% | Chain callbacks, monitoring metrics, and schema-documented payloads delivered.【F:src/av_options.py†L29-L322】 |
|  | `av_sentiment.py` | 429 | 100% | Configurable ETF exclusions, sentiment quality metrics, and enriched technical metadata stored.【F:src/av_sentiment.py†L25-L335】 |
| Analytics | `analytics_engine.py` | 585 | 100% | Cadence-aware scheduler drives calculators, honors market hours, and publishes portfolio/sector/correlation artifacts via the shared Redis schema.【F:src/analytics_engine.py†L35-L584】 |
|  | `vpin_calculator.py` | 317 | 100% | VPIN logic migrated intact with async Redis use.【F:src/vpin_calculator.py†L31-L116】 |
|  | `gex_dex_calculator.py` | 482 | 100% | Normalized chain support now publishes rolling histories and z-score context for GEX/DEX so downstream scorers can detect extremes without recomputing history.【F:src/gex_dex_calculator.py†L47-L105】【F:src/gex_dex_calculator.py†L370-L594】 |
|  | `pattern_analyzer.py` | 477 | 100% | Toxicity/OBI/sweep metrics publish through shared helpers with sweep detections persisted 30 seconds to preserve velocity context for signals.【F:src/pattern_analyzer.py†L293-L354】 |
|  | `parameter_discovery.py` | 828 | 90% | Automated tuning pipeline operational with configurable schedules.【F:src/parameter_discovery.py†L31-L246】 |
| Shared Utilities | `option_utils.py` | 112 | 100% | Centralizes trading-day calendars, DTE-to-expiry math, and expiry normalization for option-only contract selection across strategies and execution.【F:src/option_utils.py†L1-L104】 |
| Signal Generation | `signal_generator.py` | 698 | 98% | Dependency-injected strategies, feature normalization, guardrails, and circuit-breaker backoff in place; VPIN/OBI field mismatch resolved and dedupe now emits to dedicated execution queues for IBKR routing—next audit other analytics payloads for naming drift.【F:src/signal_generator.py†L48-L274】【F:src/signal_deduplication.py†L224-L262】 |
|  | `dte_strategies.py` | 1,016 | 98% | `DTEFeatureSet` normalizes payloads and shared hysteresis keeps strikes stable across 0/1/14DTE plays.【F:src/dte_strategies.py†L13-L218】【F:src/dte_strategies.py†L300-L498】 |
|  | `moc_strategy.py` | 436 | 95% | Contradiction-aware scoring, contract selection, and dedupe-backed memory live; expand alt-contract heuristics next.【F:src/moc_strategy.py†L61-L304】 |
|  | `signal_deduplication.py` | 321 | 100% | Contract-centric dedupe reused by generator/distributor with metrics instrumentation; atomic script now fans out to `signals:execution:{symbol}` for decoupled execution loops.【F:src/signal_deduplication.py†L21-L133】【F:src/signal_deduplication.py†L224-L262】 |
|  | `signal_distributor.py` | 410 | 90% | Validator guardrails, persistent tier scheduling, and dead-letter/backoff metrics ready; downstream connectors still TODO.【F:src/signal_distributor.py†L27-L236】【F:src/signal_distributor.py†L289-L400】 |
|  | `signal_performance.py` | 201 | 90% | Tracks profit factor, Sharpe, drawdown, and daily PnL; pending integration with reporting/dashboard flows.【F:src/signal_performance.py†L25-L181】 |
| Execution & Risk | `execution_manager.py` | 1,118 | 95% | Shared risk-manager cache, IBKR reconciliation, sanitized Redis order snapshots, partial-fill/timeout telemetry, and automatic stop/target bracket placement; remaining work focuses on richer cancel paths.【F:src/execution_manager.py†L59-L972】 |
|  | `position_manager.py` | 1,049 | 95% | Injects contract resolvers, mirrors live IBKR positions at startup, enforces strategy-aware trailing stops, and tears down bracket legs on manual closes while recording exit telemetry.【F:src/position_manager.py†L47-L799】 |
|  | `risk_manager.py` | 921 | 95% | Dataclass-backed VaR/drawdown snapshots with portfolio Greek aggregation and correlation breaker enforcement; future tuning targets real-market calibration.【F:src/risk_manager.py†L57-L350】 |
|  | `emergency_manager.py` | 713 | 85% | Async Redis pipeline drains pending/execution signal queues and orders, archives cancel metadata, and toggles breaker state; alert/webhook integrations remain outstanding.【F:src/emergency_manager.py†L80-L279】 |
| Social & Customer | `twitter_bot.py` | 290 | 25% | Async skeleton with extensive TODO blocks for API wiring and content logic.【F:src/twitter_bot.py†L35-L133】 |
|  | `telegram_bot.py` | 272 | 25% | Subscription tiers and billing placeholders pending implementation.【F:src/telegram_bot.py†L43-L214】 |
|  | `discord_bot.py` | 144 | 30% | Basic message routing scaffold; real channel workflows not yet added.【F:src/discord_bot.py†L25-L140】 |
| Dashboard & APIs | `dashboard_server.py` | 390 | 20% | FastAPI scaffolding with numerous TODO markers for routes/templates.【F:src/dashboard_server.py†L83-L223】 |
|  | `dashboard_routes.py` | 240 | 20% | Planned REST handlers documented but not yet implemented.【F:src/dashboard_routes.py†L25-L200】 |
|  | `dashboard_websocket.py` | 65 | 40% | WebSocket manager drafted; requires integration with server module.【F:src/dashboard_websocket.py†L25-L62】 |
| Morning Automation | `morning_scanner.py` | 267 | 30% | GPT-based analysis framework outlined with many TODOs before production use.【F:src/morning_scanner.py†L85-L206】 |
|  | `news_analyzer.py` | 318 | 30% | Scheduler blueprint awaiting real news pipelines and archival plumbing.【F:src/news_analyzer.py†L65-L254】 |
|  | `report_generator.py` | 97 | 50% | Data archival routines largely implemented; reporting templates can be expanded.【F:src/report_generator.py†L25-L189】 |

## Recent Deliverables – Redis Schema Alignment
- Replaced the monolithic `Keys` helper with `redis_keys.py` function exports so ingestion, analytics, and signal modules build Redis namespaces through validated helpers, eliminating runtime `AttributeError`s and keeping pipelines aligned with the refactored schema.【F:src/redis_keys.py†L25-L304】【F:src/ibkr_ingestion.py†L560-L748】【F:src/gex_dex_calculator.py†L29-L520】【F:src/signal_generator.py†L520-L606】
- Updated ingestion, analytics, and signal pipelines to call the new helpers when persisting market data, analytics outputs, monitoring heartbeats, and deduped payloads so key construction is centralized and easier to audit.【F:src/ibkr_ingestion.py†L560-L748】【F:src/analytics_engine.py†L35-L205】【F:src/signal_generator.py†L520-L606】
- Refreshed analytics and signal test suites to import the helper functions directly, exercising the functional interface end-to-end without depending on the removed class API and covering the VPIN/OBI schema expectations to prevent future drift.【F:tests/test_analytics_engine.py†L1-L216】【F:tests/test_signal_generation_enhancements.py†L1-L180】
- Corrected VPIN and OBI field mismatches in the default feature reader so Redis analytics produced by `vpin_calculator` and `pattern_analyzer` drive confidence scoring again; added regression coverage to lock the behavior.【F:src/signal_generator.py†L624-L658】【F:tests/test_signal_generation_enhancements.py†L164-L180】
- Enriched option-exposure telemetry with Redis-backed rolling histories/z-scores and extended sweep detections to persist for 30 seconds, giving downstream scorers stronger context without recomputing history.【F:src/gex_dex_calculator.py†L47-L105】【F:src/gex_dex_calculator.py†L370-L594】【F:src/pattern_analyzer.py†L293-L354】

## Recent Deliverables – Execution & Risk Hardening
- Cached `RiskManager` instances behind an async factory in `ExecutionManager`, reconciled live IBKR positions on startup, sanitized Redis order snapshots, and expanded partial-fill plus timeout telemetry for operational visibility.【F:src/execution_manager.py†L59-L755】
- Upgraded `PositionManager` with injected contract resolvers, startup reconciliation, strategy-aware trailing stops, and configurable end-of-day reduction/close rules persisted in Redis.【F:src/position_manager.py†L47-L372】
- Structured VaR and drawdown reporting through dataclass-backed snapshots while aggregating portfolio Greeks and correlation breakers in `RiskManager`.【F:src/risk_manager.py†L57-L350】
- Modernized `EmergencyManager` to use async Redis, drain signal queues alongside orders, archive cancellations, and flip breaker state before broadcasting emergency alerts.【F:src/emergency_manager.py†L80-L260】
- Added `tests/test_managers.py` to validate execution, position, risk, and emergency workflows end-to-end with a FakeRedis stub, covering risk-manager caching, sanitized orders, Greek aggregation, and mass-cancel logic.【F:tests/test_managers.py†L1-L253】

## Recent Deliverables – Signal & Execution Decoupling
- Updated the atomic dedupe Lua pipeline to enqueue accepted signals into dedicated `signals:execution:{symbol}` lists so IBKR routing continues even when distribution queues are congested.【F:src/signal_deduplication.py†L224-L262】【F:src/redis_keys.py†L47-L54】
- Pointed `ExecutionManager` at the new execution queues and layered bracket stop/target placement for every fill, capturing target order IDs for lifecycle tracking.【F:src/execution_manager.py†L256-L292】【F:src/execution_manager.py†L828-L972】
- Taught `PositionManager` to cancel bracket legs during manual or automated closes while persisting exit stats, and extended the emergency drain to clear both pending and execution queues to avoid stale orders lingering after halts.【F:src/position_manager.py†L741-L799】【F:src/emergency_manager.py†L262-L274】

## Recent Deliverables – Options-First Trading
- Hardened DTE and MOC contract selectors to emit option-only payloads with preserved OCC symbols, normalized strike metadata, and IBKR-formatted expiries so downstream dedupe, execution, and risk stacks never receive equity orders.【F:src/dte_strategies.py†L13-L218】【F:src/moc_strategy.py†L61-L304】
- Introduced `option_utils` to centralize trading-day calendars, DTE-to-expiry math, and expiry normalization; strategies and execution modules now share these helpers to avoid weekend/holiday skew in expiry selection.【F:src/option_utils.py†L1-L104】【F:src/dte_strategies.py†L158-L204】【F:src/moc_strategy.py†L228-L274】
- Updated `SignalGenerator` and `ExecutionManager` to treat options as the default contract type, validate expiry/strike fields, and derive TTL/expiry hysteresis from normalized metadata before emitting orders to IBKR.【F:src/signal_generator.py†L167-L274】【F:src/execution_manager.py†L256-L372】

## Key Backlog Items by Domain
- **Eliminate runtime imports** – With strategy handlers injected into `SignalGenerator`, focus on execution/risk modules to consume Redis artifacts instead of runtime imports before scaling workers.【F:src/signal_generator.py†L48-L209】【F:src/execution_manager.py†L33-L120】
- **Complete social channels** – Implement API clients, subscription management, and Redis lookups for Twitter, Telegram, and Discord bots to move them from scaffolding to production-ready modules.【F:src/twitter_bot.py†L35-L133】【F:src/telegram_bot.py†L43-L214】【F:src/discord_bot.py†L25-L140】
- **Finish dashboard UX** – Build out FastAPI routes, WebSocket streaming, and front-end assets to surface analytics, risk, and alert feeds.【F:src/dashboard_server.py†L83-L223】【F:src/dashboard_routes.py†L25-L200】
- **Harden emergency controls** – Address TODOs in `EmergencyManager` around alert integrations and circuit-breaker analytics before enabling auto-halt pathways.【F:src/emergency_manager.py†L208-L424】
- **Backfill execution queue coverage** – Extend the manager suite to assert bracket metadata, target cancellations, and execution queue drains so the new order lifecycle remains regression-tested.【F:tests/test_managers.py†L1-L253】【F:src/execution_manager.py†L828-L972】
- **Clarify ingest processor roadmap** – Determine whether `ibkr_processor.py` will host transformation pipelines or be removed after confirming inline processing is sufficient.【F:src/ibkr_processor.py†L2-L33】
- **Surface performance telemetry** – Expose `PerformanceTracker` outputs (profit factor, Sharpe, drawdown, daily PnL) via dashboards or reporting once UX scaffolding is ready.【F:src/signal_performance.py†L143-L181】【F:src/dashboard_server.py†L83-L223】

## Architecture & Redis Conventions
- **Redis-only messaging** keeps modules isolated; new features must register keys through the `redis_keys.py` function exports to maintain a single schema source of truth and leverage the validation baked into the helpers.【F:src/redis_keys.py†L1-L304】
- **Configuration-first toggles**: `config/config.yaml` enables/disables module groups and encodes strategy thresholds, allowing staged rollouts without code changes.【F:config/config.yaml†L58-L240】
- **Logging uniformity**: `main.py` configures rotating file and console handlers; modules should continue using `logging.getLogger(__name__)` for consistent traceability.【F:src/main.py†L27-L66】【F:src/signal_generator.py†L49-L65】

## Development Roadmap
1. **Q1 – Stabilize orchestration**
   - Remove remaining dynamic imports by exposing Redis-driven interfaces for calculators and risk checks.
   - Update automated tests to import `analytics_engine`, `signal_generator`, `execution_manager`, etc., directly instead of the legacy aggregate files.【F:tests/test_day6.py†L21-L39】【F:tests/test_day8.py†L18-L30】
2. **Q2 – Deliver customer-facing features**
   - Implement social channel messaging, subscription tiers, and metrics dashboards.
   - Build dashboard APIs and WebSocket streams to visualize `analytics:*`, `risk:*`, and `alerts:*` feeds.【F:src/dashboard_server.py†L83-L223】
3. **Q3 – Automation & reporting**
   - Integrate GPT/OpenAI workflows for morning analysis and automated report generation once credentials are provisioned.【F:src/morning_scanner.py†L85-L206】【F:src/report_generator.py†L25-L189】
4. **Ongoing – Risk & compliance**
   - Tune correlation gates, VaR windows, and circuit-breaker telemetry based on live trading feedback.【F:src/risk_manager.py†L200-L401】【F:src/emergency_manager.py†L208-L424】

## Testing Strategy
- **Analytics cadence tests**: `tests/test_analytics_engine.py` covers aggregator TTLs, correlation fallbacks, and market-hours gating to guard the refreshed scheduler contract.【F:tests/test_analytics_engine.py†L1-L216】【F:src/analytics_engine.py†L35-L445】
- **Signal pipeline tests**: `tests/test_signal_generation_enhancements.py` verifies TTL rollover, DTE contract hysteresis, contradiction-aware MOC direction, validator guardrails, and the VPIN/OBI schema alignment using lightweight Redis doubles; expand toward distribution/backoff scenarios next.【F:tests/test_signal_generation_enhancements.py†L1-L180】【F:src/signal_distributor.py†L27-L236】
- **Module-level tests**: Grow the consolidated manager suite alongside targeted coverage for `signal_deduplication` and `signal_distributor`, expanding FakeRedis-backed checks for execution edge cases, bracket stop/target flows, and emergency drains across the new execution queues.【F:tests/test_managers.py†L1-L253】【F:src/signal_deduplication.py†L21-L133】【F:src/execution_manager.py†L828-L972】
- **Integration suites**: Migrate existing day-based regression tests to the modular imports once analytics/execution decoupling is complete.【F:tests/test_day6.py†L21-L39】【F:tests/test_day8.py†L18-L30】
- **Operational smoke tests**: Continue using scripts like `tests/verify_signals.py` to monitor live Redis keys, heartbeats, and signal freshness during deployments.【F:tests/verify_signals.py†L1-L60】
- **Dependency validation**: Optional modules require extra packages (`tweepy`, `stripe`, `fastapi`, `openai`, etc.); document and install these before enabling their services to avoid import failures.【510fcf†L1-L27】【F:requirements.txt†L40-L64】

## Dependencies & Environment
- Core runtime dependencies are captured in `requirements.txt`; optional extras remain commented until corresponding modules are activated.【F:requirements.txt†L1-L64】
- Environment variables (e.g., `ALPHA_VANTAGE_API_KEY`) are resolved by `main.py` during configuration loading, so new services should follow the same pattern for secrets management.【F:config/config.yaml†L41-L50】【F:src/main.py†L43-L82】

## Documentation Links
- **System overview**: See `README.md` for module descriptions, redis architecture, and deduplication design.
- **Historical hardening notes**: Contract-centric deduplication details are preserved in `DEDUPLICATION_CHANGES.md.archived` for reference.【F:DEDUPLICATION_CHANGES.md.archived†L1-L120】
- **Analytics approach**: Pattern-based toxicity guidance remains in `toxicity_approach.md.archived` until fully productized.【F:toxicity_approach.md.archived†L1-L78】
