# AlphaTrader Pro – Implementation Plan

## Refactor Snapshot
- **Architecture**: 29 production modules now sit behind a Redis-centric message fabric while `main.py` orchestrates configuration, logging, and shared connection pools.【F:src/main.py†L27-L192】
- **Data integrity**: The canonical Redis schema now enumerates the `analytics:*` portfolio/sector keys and TTL helpers that calculators use when persisting outputs, keeping Redis namespaces and expirations consistent across modules.【F:src/redis_keys.py†L25-L205】
- **Analytics coordination**: `AnalyticsEngine` seeds calculators on cadence, publishes sector/portfolio snapshots, and idles gracefully outside configured market hours via the new aggregator pipeline.【F:src/analytics_engine.py†L35-L445】
- **Testing coverage**: Async unit tests validate the aggregator TTLs, correlation sourcing, and the RTH scheduler guardrails to prevent regressions as cadence rules evolve.【F:tests/test_analytics_engine.py†L1-L216】
- **Focus areas**: Market ingestion, analytics, signal generation, and risk/execution are production-grade; social, dashboard, and morning-automation services remain scaffolding pending API integrations.

## Module Completion Matrix
| Group | Module | Lines | Status | Notes |
| --- | --- | --- | --- | --- |
| Data Ingestion | `ibkr_ingestion.py` | 1,070 | 100% | Processor-driven normalization with keyed metrics, TTL-managed monitoring, and clean shutdown handling.【F:src/ibkr_ingestion.py†L49-L920】 |
|  | `ibkr_processor.py` | 215 | 100% | Depth aggregation, trade buffering, and quote synthesis extracted into reusable helpers.【F:src/ibkr_processor.py†L1-L247】 |
|  | `av_ingestion.py` | 365 | 100% | Long-lived processors, circuit-breaker retries, and telemetry-rich metrics publishing in place.【F:src/av_ingestion.py†L33-L458】 |
|  | `av_options.py` | 185 | 100% | Chain callbacks, monitoring metrics, and schema-documented payloads delivered.【F:src/av_options.py†L1-L223】 |
|  | `av_sentiment.py` | 365 | 100% | Configurable ETF exclusions, sentiment quality metrics, and enriched technical metadata stored.【F:src/av_sentiment.py†L1-L365】 |
| Analytics | `analytics_engine.py` | 585 | 100% | Cadence-aware scheduler drives calculators, honors market hours, and publishes portfolio/sector/correlation artifacts via the shared Redis schema.【F:src/analytics_engine.py†L35-L584】 |
|  | `vpin_calculator.py` | 317 | 100% | VPIN logic migrated intact with async Redis use.【F:src/vpin_calculator.py†L31-L116】 |
|  | `gex_dex_calculator.py` | 482 | 100% | Normalized chain support, canonical analytics writes, and TTL policies aligned with the engine.【F:src/gex_dex_calculator.py†L29-L520】 |
|  | `pattern_analyzer.py` | 477 | 100% | Toxicity/OBI/sweep metrics publish through shared helpers with velocity tracking TTLs.【F:src/pattern_analyzer.py†L32-L360】 |
|  | `parameter_discovery.py` | 828 | 90% | Automated tuning pipeline operational with configurable schedules.【F:src/parameter_discovery.py†L31-L246】 |
| Signal Generation | `signal_generator.py` | 512 | 95% | Production guardrails and audits running; keep an eye on runtime imports for future cleanup.【F:src/signal_generator.py†L33-L420】 |
|  | `dte_strategies.py` | 822 | 95% | Strategy evaluation extracted cleanly with shared interfaces.【F:src/dte_strategies.py†L30-L300】 |
|  | `moc_strategy.py` | 321 | 90% | MOC logic functional with Redis hysteresis keys.【F:src/moc_strategy.py†L25-L223】 |
|  | `signal_deduplication.py` | 322 | 100% | Contract-centric dedupe hardened with atomic Lua script and metrics.【F:src/signal_deduplication.py†L21-L133】 |
|  | `signal_distributor.py` | 333 | 85% | Tiered queues implemented; contains TODOs for downstream integrations.【F:src/signal_distributor.py†L25-L274】 |
|  | `signal_performance.py` | 122 | 85% | Metrics tracker operational; future expansion hooks noted.【F:src/signal_performance.py†L25-L219】 |
| Execution & Risk | `execution_manager.py` | 870 | 90% | Order lifecycle, fills, and telemetry implemented; relies on dynamic risk import for now.【F:src/execution_manager.py†L33-L533】 |
|  | `position_manager.py` | 780 | 90% | Manages P&L, scaling, and stop logic with Redis persistence.【F:src/position_manager.py†L30-L459】 |
|  | `risk_manager.py` | 803 | 90% | VaR, drawdown, and breaker gates active; future correlation tuning planned.【F:src/risk_manager.py†L31-L401】 |
|  | `emergency_manager.py` | 698 | 80% | Emergency pathways coded with TODOs for alert integrations and breaker analytics.【F:src/emergency_manager.py†L35-L424】 |
| Social & Customer | `twitter_bot.py` | 290 | 25% | Async skeleton with extensive TODO blocks for API wiring and content logic.【F:src/twitter_bot.py†L35-L133】 |
|  | `telegram_bot.py` | 272 | 25% | Subscription tiers and billing placeholders pending implementation.【F:src/telegram_bot.py†L43-L214】 |
|  | `discord_bot.py` | 144 | 30% | Basic message routing scaffold; real channel workflows not yet added.【F:src/discord_bot.py†L25-L140】 |
| Dashboard & APIs | `dashboard_server.py` | 390 | 20% | FastAPI scaffolding with numerous TODO markers for routes/templates.【F:src/dashboard_server.py†L83-L223】 |
|  | `dashboard_routes.py` | 240 | 20% | Planned REST handlers documented but not yet implemented.【F:src/dashboard_routes.py†L25-L200】 |
|  | `dashboard_websocket.py` | 65 | 40% | WebSocket manager drafted; requires integration with server module.【F:src/dashboard_websocket.py†L25-L62】 |
| Morning Automation | `morning_scanner.py` | 267 | 30% | GPT-based analysis framework outlined with many TODOs before production use.【F:src/morning_scanner.py†L85-L206】 |
|  | `news_analyzer.py` | 318 | 30% | Scheduler blueprint awaiting real news pipelines and archival plumbing.【F:src/news_analyzer.py†L65-L254】 |
|  | `report_generator.py` | 97 | 50% | Data archival routines largely implemented; reporting templates can be expanded.【F:src/report_generator.py†L25-L189】 |

## Recent Deliverables – Analytics Refresh
- Completed the cadence-aware scheduler and aggregator in `AnalyticsEngine`, emitting symbol, sector, and portfolio payloads solely through the canonical Redis helpers while respecting RTH gating and graceful shutdowns.【F:src/analytics_engine.py†L35-L445】
- Extended `redis_keys.py` with dedicated helpers for portfolio analytics keys plus differentiated TTL configuration so calculators, aggregators, and downstream consumers share one schema source of truth.【F:src/redis_keys.py†L25-L205】
- Promoted configuration-driven cadences and sector mappings under `config.yaml`, enabling operations to tune TTLs, refresh intervals, and sector groupings without code edits.【F:config/config.yaml†L84-L115】
- Backfilled async unit coverage for aggregator math, correlation sourcing, and market-hours gating to prevent regressions as additional calculators come online.【F:tests/test_analytics_engine.py†L1-L216】

## Key Backlog Items by Domain
- **Eliminate runtime imports** – Continue refactoring signal generation and execution managers to consume results via Redis rather than importing peer modules during runtime.【F:src/signal_generator.py†L132-L196】【F:src/execution_manager.py†L33-L120】
- **Complete social channels** – Implement API clients, subscription management, and Redis lookups for Twitter, Telegram, and Discord bots to move them from scaffolding to production-ready modules.【F:src/twitter_bot.py†L35-L133】【F:src/telegram_bot.py†L43-L214】【F:src/discord_bot.py†L25-L140】
- **Finish dashboard UX** – Build out FastAPI routes, WebSocket streaming, and front-end assets to surface analytics, risk, and alert feeds.【F:src/dashboard_server.py†L83-L223】【F:src/dashboard_routes.py†L25-L200】
- **Harden emergency controls** – Address TODOs in `EmergencyManager` around alert integrations and circuit-breaker analytics before enabling auto-halt pathways.【F:src/emergency_manager.py†L208-L424】
- **Clarify ingest processor roadmap** – Determine whether `ibkr_processor.py` will host transformation pipelines or be removed after confirming inline processing is sufficient.【F:src/ibkr_processor.py†L2-L33】

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
- **Module-level tests**: Expand async unit tests around `signal_deduplication`, `dte_strategies`, and `execution_manager` to validate Redis interactions without the monolithic shim.【F:src/signal_deduplication.py†L21-L133】【F:src/dte_strategies.py†L30-L300】【F:src/execution_manager.py†L232-L533】
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
