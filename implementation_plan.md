# QuantiCity Capital – Implementation Plan

## Document Map
- [Overview](#overview)
- [Phase Summaries](#phase-summaries)
- [Detailed Phase Notes](#detailed-phase-notes)
- [Open Work](#open-work)
- [Pending Integrations](#pending-integrations)
- [Operational Notes](#operational-notes)

## Overview
This plan tracks the delivery status for every major platform phase, highlighting completed milestones, pending work, and supporting modules.

## Phase Summaries
- **Phase 1 – Schema & configuration** (Complete): Redis key helpers now expose dedicated namespaces for dealer-flow metrics, flow clusters, VIX1D data, and the newly introduced MOC/unusual scorecards, while `config/config.yaml` surfaces cadence, TTL, weighting, and interval controls for the expanded analytics pipeline.【F:src/redis_keys.py†L37-L141】【F:config/config.yaml†L84-L210】
- **Phase 2 – Dealer-flow analytics** (Complete): DealerFlowCalculator derives Vanna, Charm, skew, and hedging elasticity from the normalized option chain and persists rolling histories that the analytics engine aggregates into symbol, sector, and portfolio snapshots.【F:src/dealer_flow_calculator.py†L52-L206】【F:src/analytics_engine.py†L73-L220】【F:src/analytics_engine.py†L500-L660】
- **Phase 3 – Flow clustering & volatility regimes** (Complete): FlowClusterModel classifies recent trade prints into participant archetypes while VolatilityMetrics maintains VIX1D history and regime tags, both orchestrated by the analytics scheduler and folded into downstream aggregates.【F:src/flow_clustering.py†L30-L199】【F:src/volatility_metrics.py†L19-L190】【F:src/analytics_engine.py†L500-L660】
- **Phase 4 – Signal integration & scoring** (Complete): The default feature reader and DTE/MOC playbooks ingest dealer-flow, clustering, VIX1D, imbalance, and unusual-activity features and rebalance confidence scoring weights so emissions align with the new analytics payloads.【F:src/signal_generator.py†L542-L803】【F:src/dte_strategies.py†L68-L209】【F:src/moc_strategy.py†L55-L214】【F:config/config.yaml†L143-L250】
- **Phase 5 – Backfills, monitoring, and regression coverage** (Complete): Backfill utilities replay dealer-flow, flow-cluster, and VIX1D series through dedicated jobs with regression suites covering replay paths; monitoring dashboards for these analytics have been shifted into Phase 7 alongside the broader operator UI work. Migration of legacy tests onto the refactored module surfaces remains a follow-up item.
- **Phase 6 – Execution & distribution hardening** (Complete): Risk gating, commission normalization, bracket maintenance, and the executed-only distribution pipeline are live. Social publishing modules will reuse this pipeline once implemented, but remain disabled until the Phase 7 backlog is delivered.【F:src/execution_manager.py†L828-L1593】【F:src/signal_deduplication.py†L200-L280】【F:src/signal_distributor.py†L71-L229】
- **Phase 7 – Community publishing & automation** (In Planning): Build the social bots, dashboard services, morning automation, and archival/reporting utilities so executed trades, analytics, and pre-market briefs reach every external channel with engagement telemetry fed back into Redis.【F:src/twitter_bot.py†L25-L206】【F:src/discord_bot.py†L25-L141】【F:src/telegram_bot.py†L25-L218】【F:src/dashboard_server.py†L25-L220】【F:src/dashboard_routes.py†L25-L184】【F:src/morning_scanner.py†L25-L220】【F:src/report_generator.py†L25-L120】

## Detailed Phase Notes

### Phase 1 – Schema & Configuration
The Redis schema now dedicates stable helpers for dealer-flow buckets (`analytics:vanna`, `analytics:charm`, `analytics:hedging`, `analytics:skew`), flow-cluster payloads, and VIX1D volatility state to keep all producers and consumers aligned on key formats.【F:src/redis_keys.py†L37-L141】 Configuration exposes cadence, TTLs, and weighting knobs for each analytics family—including cluster windows, VIX1D thresholds, and dealer-flow history depth—allowing operators to retune behaviour without code changes.【F:config/config.yaml†L84-L141】 These settings are loaded during bootstrap alongside existing module toggles so every service shares a single source of truth.【F:main.py†L100-L176】

### Phase 2 – Dealer-Flow Analytics
`DealerFlowCalculator` walks the normalized options chain, filters liquid near-dated contracts, and computes per-contract Greeks before aggregating Vanna, Charm, 0DTE skew, and hedging elasticity into Redis payloads with rolling history/z-score stats that downstream consumers can query.【F:src/dealer_flow_calculator.py†L75-L206】 The analytics engine schedules the calculator on a configurable cadence, writing results into symbol snapshots and portfolio/sector rollups so execution, monitoring, and research services share the same dealer-flow view without recomputing histories.【F:src/analytics_engine.py†L73-L220】【F:src/analytics_engine.py†L500-L660】

### Phase 3 – Flow Clustering & Volatility Regimes
`FlowClusterModel` ingests the latest trade prints, assembles feature vectors (size, direction, sweep velocity, notional, gap), and runs a three-cluster KMeans to infer participant and strategy distributions before persisting them to Redis.【F:src/flow_clustering.py†L30-L199】 The TTL is now tied to the configured cadence so downstream rollups retain probabilities between refreshes instead of aging out mid-interval.【F:src/flow_clustering.py†L37-L78】【F:config/config.yaml†L139-L141】 `VolatilityMetrics` promotes a multi-source VIX1D pipeline that prefers CBOE’s CSV feed, caches history, falls back to IBKR, and only hits Yahoo once backoff gates allow, keeping regime classifications visible even during vendor outages.【F:src/volatility_metrics.py†L24-L222】【F:config/config.yaml†L127-L138】 AnalyticsEngine triggers both calculators through its interval map and drops the outputs directly into the symbol snapshots consumed by portfolio and sector aggregations.【F:src/analytics_engine.py†L500-L660】

### Phase 4 – Signal Integration & Scoring
The default feature reader batches Redis fetches for dealer-flow, hedging, skew, flow-cluster, VIX1D, and now the synthesized MOC and unusual-activity metrics, normalizing each payload and attaching them to the feature map consumed by the strategies.【F:src/signal_generator.py†L542-L803】 The 0DTE/1DTE playbooks incorporate the new metrics into their confidence gates—e.g., requiring minimum Vanna/Charm balance, flow-momentum thresholds, and volatility regime alignment—while the MOC strategy consumes the freshly published imbalance projections to score auction opportunities.【F:src/dte_strategies.py†L391-L540】【F:src/moc_strategy.py†L55-L214】 Configuration holds the weighting tables and per-strategy thresholds, making tuning quick and consistent across deployments.【F:config/config.yaml†L143-L250】 Signals therefore reflect the broader dealer, volatility, imbalance, and options-flow backdrop instead of relying solely on VPIN and GEX.

### Phase 5 – Completed Deliverables
- **Historical backfill jobs** – Delivered via the `backfill` package and CLI, enabling dealer-flow, flow clustering, and VIX1D replays with resumable checkpoints and regression coverage that verifies replay behaviour and time-window filtering.【F:src/backfill/cli.py†L14-L162】【F:tests/test_backfill_jobs.py†L48-L210】
- **Regression suite expansion** – Added targeted pytest coverage for backfill workflows, checkpoint resume, time-window bounds, and sparse flow scenarios to guard the new analytics integration paths.【F:tests/test_backfill_jobs.py†L48-L210】

### Phase 6 – Execution & Distribution Hardening
- **Notional-aware sizing** – `ExecutionManager.calculate_order_size` stores the computed dollar exposure on each signal and reruns `passes_risk_checks` after sizing so the 25 % buying-power guardrail applies to the true post-sizing notional. The notional is persisted with pending orders and positions for downstream reconciliation.【F:src/execution_manager.py†L840-L1593】
- **Commission-normalized P&L** – All fills convert IBKR’s negative commissions into absolute costs before updating positions, metrics, and reconciliation outputs. Daily P&L scripts mirror the same convention so realized totals match broker statements exactly.【F:src/execution_manager.py†L115-L2065】【F:reconcile_daily_pnl.py†L85-L138】
- **Resilient bracket workflow** – Trailing-stop updates tear down and recreate the entire OCA group, scale-outs pause active protection while partial fills route, and reduced positions immediately receive right-sized stops and targets, eliminating IBKR’s dual-side rejection.【F:src/position_manager.py†L70-L1140】【F:src/execution_manager.py†L828-L1593】
- **Executed-only distribution** – Fills enqueue an enriched payload on `signals:distribution:pending`, and the distributor only fans out premium/basic/free messages once execution status is `FILLED`, preserving the 0s/60s/300s tier delays without leaking rejected signals.【F:src/execution_manager.py†L1333-L1593】【F:src/signal_distributor.py†L71-L229】【F:src/redis_keys.py†L55-L64】

### Phase 7 – Community Publishing & Automation (In Planning)
The communication and automation layer ships with scaffolding but still depends on extensive TODO lists before production launch. Constructor signatures across these modules already accept the shared config dict and Redis handle, so they can be wired into the live queues without refactoring when their TODOs are addressed.

- **TwitterBot & SocialMediaAnalytics** – Implement credential loading, Tweepy client initialization, and the posting/analytics loops so winning trades, teasers, daily summaries, and morning previews leave `signals:distribution:pending` and write engagement metrics back to Redis.【F:src/twitter_bot.py†L25-L206】
- **DiscordBot** – Tier workers drain premium/basic/free queues, render the updated embed templates, enforce webhook retries, and persist delivery/dead-letter metrics for dashboards. Summaries/analysis/performance posts remain outstanding for the next milestone.【F:src/discord_bot.py†L200-L579】
- **TelegramBot** – Stand up the command handlers, Stripe subscription flow, tier-aware formatting, and delayed distribution to match the premium/basic/free cadence in Redis.【F:src/telegram_bot.py†L25-L218】
- **Dashboard services & analytics monitoring** – Flesh out the FastAPI routes/WebSocket broadcasting, log aggregation, alert management, performance chart builders, and the newly scoped backfill/analytics dashboards so operators (and community members, if desired) have a real-time lens on health, replay progress, and risk.【F:src/dashboard_server.py†L25-L220】【F:src/dashboard_routes.py†L25-L184】【F:src/dashboard_websocket.py†L1-L70】
- **MorningScanner** – Complete the GPT-4 powered morning analysis workflow (data gathering, key levels, options positioning, distribution) so premium channels receive pre-open briefs and social queues get teasers automatically.【F:src/morning_scanner.py†L25-L220】
- **ReportGenerator & MetricsCollector** – Promote the archival, cleanup, and metrics collection TODOs into working services so historical exports, retention policies, and dashboard metrics stay up to date.【F:src/report_generator.py†L25-L120】【F:src/dashboard_server.py†L166-L220】


## Open Work

| Area | Status | Next Actions |
| --- | --- | --- |
| Social publishing (`twitter_bot`, `discord_bot`, `telegram_bot`) | In progress | Discord relay implemented (tier embeds + simulator); next wire Twitter/Telegram credentials and add summary/analysis/performance drops for Discord. |
| Operator dashboard (`dashboard_server`, `dashboard_routes`, `dashboard_websocket`) | Scaffolding only | Build FastAPI app, auth, REST + WebSocket endpoints, metrics aggregation, alert routing, and front-end payloads before enabling module flag. |
| Morning automation (`morning_scanner`, `news_analyzer`) | Scaffolding only | Gather overnight data, compute key levels/options positioning, orchestrate GPT-4 prompts, persist premium/public previews, and schedule distribution jobs. |
| Archival & reporting (`report_generator`) | Scaffolding only | Implement daily exports, retention cleanup, long-horizon metrics, and operator-triggered report generation. |
| Regression coverage | Needs update | Shift pytest suites to import refactored modules, add integration tests for execution/distribution, and validate Phase 7 services once implemented. |
| Deployment hygiene | In progress | Split optional requirements into extras, document full credential matrix, and extend monitoring/alerting for new queues and dashboard services. |

## Pending Integrations
- [`src/twitter_bot.py`](src/twitter_bot.py)
- [`src/discord_bot.py`](src/discord_bot.py)
- [`src/telegram_bot.py`](src/telegram_bot.py)
- [`src/dashboard_server.py`](src/dashboard_server.py), [`src/dashboard_routes.py`](src/dashboard_routes.py), [`src/dashboard_websocket.py`](src/dashboard_websocket.py)
- [`src/morning_scanner.py`](src/morning_scanner.py)
- [`src/news_analyzer.py`](src/news_analyzer.py)
- [`src/report_generator.py`](src/report_generator.py)


### September 2025 Enhancements
- Added neutral VPIN fallbacks so toxicity and sector summaries always have data, even during low-volume windows.【F:src/vpin_calculator.py†L52-L150】
- Updated analytics viewers to reflect the canonical Redis schema, surfacing dealer-flow totals, toxicity levels, and order-book imbalance without manual reconciliation.
- Swapped the correlation matrix loader to consume the same 1-minute bars written by IBKR ingestion, eliminating empty matrices caused by timeframe suffixes.【F:src/analytics_engine.py†L391-L420】【F:src/ibkr_ingestion.py†L583-L589】
- Introduced synthetic MOC imbalance and unusual-options activity scoring inside AnalyticsEngine so signal modules have dependable inputs when exchanges or venues lack dedicated feeds.【F:src/analytics_engine.py†L676-L1010】
- Expanded the ingestion universe to include PLTR, ensuring 14DTE strategies evaluate fresh analytics for every configured symbol.【F:config/config.yaml†L60-L66】

## Operational Notes

### Operational Orchestration
`main.py` validates the environment, connects to Redis, and instantiates the analytics engine alongside ingestion, parameter discovery, and optional signal/execution stacks. During startup it schedules the analytics engine asynchronously so DealerFlowCalculator, FlowClusterModel, and VolatilityMetrics run continuously without blocking other services.【F:main.py†L111-L326】 Health monitoring keeps the analytics heartbeat fresh in Redis, ensuring operator dashboards can confirm the new calculators are live.【F:main.py†L327-L438】
