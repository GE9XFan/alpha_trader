# AlphaTrader Pro – Implementation Plan

## Refactor Snapshot
- **Architecture**: 29 production modules now sit behind a Redis-centric message fabric while `main.py` orchestrates configuration, logging, and shared connection pools.【F:src/main.py†L27-L192】
- **Data integrity**: The canonical Redis schema now enumerates the `analytics:*` portfolio/sector keys and TTL helpers that calculators use when persisting outputs, keeping Redis namespaces and expirations consistent across modules.【F:src/redis_keys.py†L25-L205】
- **Analytics coordination**: `AnalyticsEngine` seeds calculators on cadence, publishes sector/portfolio snapshots, and idles gracefully outside configured market hours via the new aggregator pipeline.【F:src/analytics_engine.py†L35-L445】
- **Signal pipeline**: `SignalGenerator` now injects strategy handlers/feature readers, adds loop backoff, and delegates idempotency to `SignalDeduplication`; DTE and MOC strategies share typed feature normalization, contradiction gating, and contract hysteresis.【F:src/signal_generator.py†L48-L209】【F:src/dte_strategies.py†L13-L218】【F:src/moc_strategy.py†L61-L304】
- **Execution & risk controls**: `ExecutionManager` caches a shared `RiskManager`, reconciles live IBKR positions, strips signal metadata from persisted orders, and escalates partial fills/timeouts while `PositionManager` enforces trailing-stop/EOD rules and `EmergencyManager` drains queues with async Redis primitives.【F:src/execution_manager.py†L59-L755】【F:src/position_manager.py†L70-L372】【F:src/emergency_manager.py†L80-L260】
- **Distribution & telemetry**: `SignalDistributor` enforces validator guardrails, persistent tier scheduling, and dead-letter handling while `PerformanceTracker` captures equity curves, profit factor, Sharpe, and daily PnL for reviews.【F:src/signal_distributor.py†L27-L236】【F:src/signal_performance.py†L25-L181】
- **Testing coverage**: Async suites now cover analytics cadence, signal flow, and consolidated execution/position/risk/emergency scenarios with a lightweight Redis stub validating risk-manager caching, sanitized orders, Greek aggregation, and mass-cancel flows.【F:tests/test_analytics_engine.py†L1-L216】【F:tests/test_signal_generation_enhancements.py†L1-L120】【F:tests/test_managers.py†L1-L253】
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
|  | `gex_dex_calculator.py` | 482 | 100% | Normalized chain support, canonical analytics writes, and TTL policies aligned with the engine.【F:src/gex_dex_calculator.py†L29-L520】 |
|  | `pattern_analyzer.py` | 477 | 100% | Toxicity/OBI/sweep metrics publish through shared helpers with velocity tracking TTLs.【F:src/pattern_analyzer.py†L32-L360】 |
|  | `parameter_discovery.py` | 828 | 90% | Automated tuning pipeline operational with configurable schedules.【F:src/parameter_discovery.py†L31-L246】 |
| Signal Generation | `signal_generator.py` | 698 | 98% | Dependency-injected strategies, feature normalization, guardrails, and circuit-breaker backoff in place; consider streaming analytics to reduce Redis round-trips.【F:src/signal_generator.py†L48-L274】 |
|  | `dte_strategies.py` | 1,016 | 98% | `DTEFeatureSet` normalizes payloads and shared hysteresis keeps strikes stable across 0/1/14DTE plays.【F:src/dte_strategies.py†L13-L218】【F:src/dte_strategies.py†L300-L498】 |
|  | `moc_strategy.py` | 436 | 95% | Contradiction-aware scoring, contract selection, and dedupe-backed memory live; expand alt-contract heuristics next.【F:src/moc_strategy.py†L61-L304】 |
|  | `signal_deduplication.py` | 321 | 100% | Contract-centric dedupe reused by generator/distributor with metrics instrumentation.【F:src/signal_deduplication.py†L21-L133】 |
|  | `signal_distributor.py` | 410 | 90% | Validator guardrails, persistent tier scheduling, and dead-letter/backoff metrics ready; downstream connectors still TODO.【F:src/signal_distributor.py†L27-L236】【F:src/signal_distributor.py†L289-L400】 |
|  | `signal_performance.py` | 201 | 90% | Tracks profit factor, Sharpe, drawdown, and daily PnL; pending integration with reporting/dashboard flows.【F:src/signal_performance.py†L25-L181】 |
| Execution & Risk | `execution_manager.py` | 1,118 | 95% | Shared risk-manager cache, IBKR reconciliation, sanitized Redis order snapshots, and partial-fill/timeout telemetry ready; remaining work focuses on expanding automated cancel paths.【F:src/execution_manager.py†L59-L755】 |
|  | `position_manager.py` | 1,049 | 95% | Injects contract resolvers, mirrors live IBKR positions at startup, enforces strategy-aware trailing stops, and applies configurable EOD reductions via Redis state.【F:src/position_manager.py†L47-L372】 |
|  | `risk_manager.py` | 921 | 95% | Dataclass-backed VaR/drawdown snapshots with portfolio Greek aggregation and correlation breaker enforcement; future tuning targets real-market calibration.【F:src/risk_manager.py†L57-L350】 |
|  | `emergency_manager.py` | 713 | 85% | Async Redis pipeline drains signal/order queues, archives cancel metadata, and toggles breaker state; alert/webhook integrations remain outstanding.【F:src/emergency_manager.py†L80-L260】 |
| Social & Customer | `twitter_bot.py` | 290 | 25% | Async skeleton with extensive TODO blocks for API wiring and content logic.【F:src/twitter_bot.py†L35-L133】 |
|  | `telegram_bot.py` | 272 | 25% | Subscription tiers and billing placeholders pending implementation.【F:src/telegram_bot.py†L43-L214】 |
|  | `discord_bot.py` | 144 | 30% | Basic message routing scaffold; real channel workflows not yet added.【F:src/discord_bot.py†L25-L140】 |
| Dashboard & APIs | `dashboard_server.py` | 390 | 20% | FastAPI scaffolding with numerous TODO markers for routes/templates.【F:src/dashboard_server.py†L83-L223】 |
|  | `dashboard_routes.py` | 240 | 20% | Planned REST handlers documented but not yet implemented.【F:src/dashboard_routes.py†L25-L200】 |
|  | `dashboard_websocket.py` | 65 | 40% | WebSocket manager drafted; requires integration with server module.【F:src/dashboard_websocket.py†L25-L62】 |
| Morning Automation | `morning_scanner.py` | 267 | 30% | GPT-based analysis framework outlined with many TODOs before production use.【F:src/morning_scanner.py†L85-L206】 |
|  | `news_analyzer.py` | 318 | 30% | Scheduler blueprint awaiting real news pipelines and archival plumbing.【F:src/news_analyzer.py†L65-L254】 |
|  | `report_generator.py` | 97 | 50% | Data archival routines largely implemented; reporting templates can be expanded.【F:src/report_generator.py†L25-L189】 |

## Recent Deliverables – Execution & Risk Hardening
- Cached `RiskManager` instances behind an async factory in `ExecutionManager`, reconciled live IBKR positions on startup, sanitized Redis order snapshots, and expanded partial-fill plus timeout telemetry for operational visibility.【F:src/execution_manager.py†L59-L755】
- Upgraded `PositionManager` with injected contract resolvers, startup reconciliation, strategy-aware trailing stops, and configurable end-of-day reduction/close rules persisted in Redis.【F:src/position_manager.py†L47-L372】
- Structured VaR and drawdown reporting through dataclass-backed snapshots while aggregating portfolio Greeks and correlation breakers in `RiskManager`.【F:src/risk_manager.py†L57-L350】
- Modernized `EmergencyManager` to use async Redis, drain signal queues alongside orders, archive cancellations, and flip breaker state before broadcasting emergency alerts.【F:src/emergency_manager.py†L80-L260】
- Added `tests/test_managers.py` to validate execution, position, risk, and emergency workflows end-to-end with a FakeRedis stub, covering risk-manager caching, sanitized orders, Greek aggregation, and mass-cancel logic.【F:tests/test_managers.py†L1-L253】

## Key Backlog Items by Domain
- **Eliminate runtime imports** – With strategy handlers injected into `SignalGenerator`, focus on execution/risk modules to consume Redis artifacts instead of runtime imports before scaling workers.【F:src/signal_generator.py†L48-L209】【F:src/execution_manager.py†L33-L120】
- **Complete social channels** – Implement API clients, subscription management, and Redis lookups for Twitter, Telegram, and Discord bots to move them from scaffolding to production-ready modules.【F:src/twitter_bot.py†L35-L133】【F:src/telegram_bot.py†L43-L214】【F:src/discord_bot.py†L25-L140】
- **Finish dashboard UX** – Build out FastAPI routes, WebSocket streaming, and front-end assets to surface analytics, risk, and alert feeds.【F:src/dashboard_server.py†L83-L223】【F:src/dashboard_routes.py†L25-L200】
- **Harden emergency controls** – Address TODOs in `EmergencyManager` around alert integrations and circuit-breaker analytics before enabling auto-halt pathways.【F:src/emergency_manager.py†L208-L424】
- **Clarify ingest processor roadmap** – Determine whether `ibkr_processor.py` will host transformation pipelines or be removed after confirming inline processing is sufficient.【F:src/ibkr_processor.py†L2-L33】
- **Surface performance telemetry** – Expose `PerformanceTracker` outputs (profit factor, Sharpe, drawdown, daily PnL) via dashboards or reporting once UX scaffolding is ready.【F:src/signal_performance.py†L143-L181】【F:src/dashboard_server.py†L83-L223】

## Architecture & Redis Conventions
- **Redis-only messaging** keeps modules isolated; new features must register keys through `redis_keys.py` to maintain a single schema source of truth.【F:src/redis_keys.py†L1-L205】
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
- **Signal pipeline tests**: `tests/test_signal_generation_enhancements.py` verifies TTL rollover, DTE contract hysteresis, contradiction-aware MOC direction, and validator guardrails using lightweight Redis doubles; expand toward distribution/backoff scenarios next.【F:tests/test_signal_generation_enhancements.py†L1-L135】【F:src/signal_distributor.py†L27-L236】
- **Module-level tests**: Grow the consolidated manager suite alongside targeted coverage for `signal_deduplication` and `signal_distributor`, expanding FakeRedis-backed checks for execution edge cases, partial fills, and emergency drains.【F:tests/test_managers.py†L1-L253】【F:src/signal_deduplication.py†L21-L133】【F:src/signal_distributor.py†L27-L236】
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
