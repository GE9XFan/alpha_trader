# AlphaTrader Pro

## Document Map
- [Overview](#overview)
- [Current Capabilities](#current-capabilities)
  - [Platform Highlights](#platform-highlights)
  - [Redis Architecture & Data Flow](#redis-architecture--data-flow)
  - [Analytics Cadence & Aggregation](#analytics-cadence--aggregation)
  - [Logging](#logging)
  - [Setup & Operation](#setup--operation)
  - [Repository Highlights](#repository-highlights)
  - [Testing & Verification](#testing--verification)
  - [Contributing](#contributing)
- [Open Work](#open-work)
  - [Community Publishing & Automation (In Planning)](#community-publishing--automation-in-planning)
  - [Pending Integrations](#pending-integrations)

## Overview

AlphaTrader Pro is a Redis-centric institutional options trading platform that ingests real-time market structure, performs high-frequency analytics, generates contract-level signals, and coordinates execution, risk controls, and downstream distribution. The platform never routes equity orders—option selectors, signal emitters, and execution flows enforce normalized option payloads with validated strikes/expiries so the system remains strictly derivatives-only. Every production module exchanges state exclusively through Redis, allowing services to scale and recover independently while sharing a consistent data fabric.【F:src/main.py†L97-L192】【F:src/redis_keys.py†L1-L141】 The latest iteration decouples signal distribution from order intake by writing deduped payloads into dedicated `signals:execution` queues and wiring bracket-style stop/target management into the execution stack while keeping the emergency breaker fully async-safe.【F:src/signal_deduplication.py†L224-L262】【F:src/execution_manager.py†L256-L292】【F:src/execution_manager.py†L828-L972】【F:src/position_manager.py†L741-L773】【F:src/emergency_manager.py†L211-L279】

## Current Capabilities

### Platform Highlights
- **Real-time market ingestion** – IBKR and Alpha Vantage feeds stream depth, trade prints, option chains, and sentiment with rate-limited retries and freshness telemetry written through the shared Redis schema.【F:src/ibkr_ingestion.py†L49-L748】【F:src/av_ingestion.py†L45-L493】【F:src/redis_keys.py†L8-L158】
- **Dealer-flow analytics fabric** – `AnalyticsEngine` orchestrates VPIN, GEX/DEX, DealerFlowCalculator, FlowClusterModel, and VIX1D updates so Vanna/Charm hedging elasticity, trade-cluster labels, and volatility regimes land in the same Redis snapshots as legacy toxicity metrics and portfolio aggregates.【F:src/analytics_engine.py†L73-L220】【F:src/analytics_engine.py†L500-L660】【F:src/dealer_flow_calculator.py†L52-L206】【F:src/flow_clustering.py†L30-L199】【F:src/volatility_metrics.py†L19-L190】
- **Dealer-aware signal scoring** – The default feature reader and DTE playbooks pull Vanna, Charm, hedging impact, skew, flow-cluster distributions, and VIX1D regimes into per-strategy confidence engines so contracts respect dealer balance and volatility state before emitting orders.【F:src/signal_generator.py†L542-L797】【F:src/dte_strategies.py†L68-L209】【F:src/dte_strategies.py†L391-L540】
- **Contract-aware signal pipeline** – `SignalGenerator` fans out scored opportunities through the hardened dedupe helper, which now emits to both distribution and execution queues to isolate order flow from customer delivery paths.【F:src/signal_generator.py†L48-L274】【F:src/signal_deduplication.py†L224-L262】【F:src/redis_keys.py†L47-L54】
- **Options-first contract handling** – DTE and MOC selectors emit normalized option payloads with IBKR-formatted expiries derived from shared utilities; execution and signal stacks treat options as the default contract type and validate strikes/expiries before routing orders.【F:src/dte_strategies.py†L13-L218】【F:src/moc_strategy.py†L61-L304】【F:src/option_utils.py†L1-L104】【F:src/execution_manager.py†L256-L372】
- **Bracketed execution & lifecycle automation** – Executed fills automatically register stop-loss and profit targets under a shared OCA group, trailing-stop revisions reissue the entire bracket with the original OCA metadata, and scale-outs rebuild reduced-size protection so IBKR never reports crossed exposure.【F:src/execution_manager.py†L828-L1593】【F:src/position_manager.py†L70-L1140】
- **Execution-driven distribution** – Signal distribution now consumes a dedicated executed-trade queue populated after IBKR confirms fills, keeping premium/basic/free broadcasts aligned with real positions and enforcing the 0s/60s/300s tier delays only for live trades.【F:src/execution_manager.py†L1333-L1593】【F:src/signal_distributor.py†L71-L229】
- **Daily P&L reconciliation** – `reconcile_daily_pnl.py` imports the broker’s fills, writes realized/commission totals back into Redis, and rewrites `positions:closed:<date>:<account>:<conId>` snapshots so dashboards and risk services mirror TWS instantly. Closed-position views automatically filter reconciled placeholders, render Eastern timestamps, and display option contract identifiers for operator reviews.【F:reconcile_daily_pnl.py†L1-L259】【F:redis_data_viewer.py†L140-L370】
- **Position lifecycle hygiene** – `PositionManager` skips zero-quantity IBKR reconciliations, prunes cached snapshots when Redis keys vanish, and limits P&L publication to live open quantities so zombie positions no longer repopulate and `ExecutionManager`'s realized/risk totals remain authoritative.【F:src/position_manager.py†L165-L427】【F:src/execution_manager.py†L781-L807】
- **Strategy-normalized sizing** – Order sizing, market/limit routing, and stop heuristics now normalize strategy codes, ensuring `0dte`/`1dte` signals respect the intended contract caps and default risk parameters regardless of payload casing.【F:src/execution_manager.py†L808-L1452】
- **Resilient emergency tooling** – Mass-cancel workflows now clear both pending and execution signal queues, persist cancellation metadata, and broadcast breaker state transitions through Redis pub/sub for operational awareness.【F:src/emergency_manager.py†L211-L319】
- **Extended pattern detections** – Sweep alerts persist for 30 seconds and maintain velocity metadata so signal models can react to short-lived toxicity without re-deriving heuristics on every tick.【F:src/pattern_analyzer.py†L347-L354】

### Recent Analytics Updates *(September 2025)*
- **Multi-source VIX1D ingestion** – `VolatilityMetrics` now prioritizes the CBOE daily CSV, caches history, falls back to IBKR snapshots, and throttles Yahoo retries with exponential backoff so volatility regimes remain populated even under rate limits.【F:src/volatility_metrics.py†L24-L222】【F:config/config.yaml†L127-L138】
- **Correlation data parity** – Portfolio correlations pull the same 1-minute bar stream published by IBKR ingestion, preventing empty matrices when timeframe-suffixed keys are the only data available.【F:src/analytics_engine.py†L391-L420】【F:src/ibkr_ingestion.py†L583-L589】
- **Dashboards aligned with Redis schema** – `analytics_data_viewer.py` now renders the dealer-flow totals, toxicity labels, and order-book stats actually stored in Redis, eliminating blank columns caused by legacy field names.
- **VPIN resilience under sparse flow** – `VPINCalculator` writes a neutral snapshot whenever trade volume is insufficient or errors occur, ensuring toxicity pipelines, sector averages, and dashboards always have a baseline value.【F:src/vpin_calculator.py†L52-L150】
- **Longer-lived flow clustering metrics** – Flow-cluster TTLs scale with the configured interval so sector aggregates retain momentum/hedging probabilities between refreshes without data gaps.【F:src/flow_clustering.py†L37-L78】【F:config/config.yaml†L139-L141】

### Dealer Flow & Regime Analytics
`AnalyticsEngine` now runs on a unified cadence that loads DealerFlowCalculator, FlowClusterModel, and VolatilityMetrics alongside legacy VPIN/GEX calculators so every symbol snapshot carries Vanna- and Charm-derived hedging pressure, 0DTE skew, participant mix, and VIX1D context before the aggregator rolls them into sector and portfolio summaries.【F:src/analytics_engine.py†L73-L220】【F:src/analytics_engine.py†L500-L660】【F:src/dealer_flow_calculator.py†L52-L206】【F:src/flow_clustering.py†L30-L199】【F:src/volatility_metrics.py†L19-L190】 These metrics are persisted under dedicated Redis namespaces (`analytics:dealer:*`, `analytics:flow_clusters:*`, `analytics:volatility:vix1d`) defined in `redis_keys.py` and governed by configurable TTLs in `config.yaml` so downstream services consume a single, normalized schema.【F:src/redis_keys.py†L37-L127】【F:config/config.yaml†L84-L141】

DealerFlowCalculator computes option-chain Vanna, Charm, hedging elasticity, and 0DTE skew with rolling z-scores so strategies can judge how aggressively dealers must hedge near-dated strikes.【F:src/dealer_flow_calculator.py†L75-L206】 FlowClusterModel ingests the latest trade prints, clusters flow into momentum/mean-reversion/hedging archetypes, and publishes participant distributions for each symbol.【F:src/flow_clustering.py†L48-L199】 VolatilityMetrics fetches VIX1D directly from Yahoo Finance, maintains rolling history, and classifies regimes (`NORMAL`, `ELEVATED`, `SHOCK`, etc.) that feed both analytics snapshots and strategy scoring.【F:src/volatility_metrics.py†L45-L190】 The default feature reader and DTE confidence engines consume these payloads to weight dealer congestion, volatility regime, and flow bias before emitting orders, ensuring execution only fires when the broader dealer and volatility backdrop cooperates.【F:src/signal_generator.py†L542-L797】【F:src/dte_strategies.py†L391-L540】

### Quick Start
1. Install dependencies with a Python 3.10+ environment: `pip install -r requirements.txt` (optional extras remain commented).【F:requirements.txt†L1-L64】
2. Configure Redis, IBKR, and module toggles in `config/config.yaml`, replacing `${VAR}` placeholders through environment variables or a `.env` file consumed by the orchestrator.【F:config/config.yaml†L1-L160】【F:src/main.py†L43-L109】
3. Ensure Redis and IBKR TWS/Gateway are running, then launch the coordinator via `python main.py` to start enabled modules in dependency order.【F:src/main.py†L140-L206】
4. Use `pytest -k managers` or targeted suites once a test Redis instance is available; full test runs require Redis and optional IBKR/Alpha Vantage fakes to satisfy external dependencies.【F:tests/test_managers.py†L1-L253】【F:tests/test_analytics_engine.py†L1-L216】
5. Reference `implementation_plan.md` for module readiness, roadmap priorities, and outstanding TODO inventories before enabling optional services.【F:implementation_plan.md†L1-L100】

### Module Topology
The original monolith has been reorganized into 30 focused modules grouped by responsibility. Line counts reflect the current implementation.

### Data Ingestion
| Module | Lines | Primary Classes | Responsibilities | Redis Touchpoints |
| --- | --- | --- | --- | --- |
| `ibkr_ingestion.py` | 1,070 | `IBKRIngestion` | Maintains IBKR Level 2, trade tape, and 1-minute bars for configured symbols.【F:src/ibkr_ingestion.py†L49-L748】 | `market:{symbol}:book`, `market:{symbol}:trades`, `market:{symbol}:bars`, `heartbeat:ibkr_ingestion`【F:src/ibkr_ingestion.py†L2-L63】 |
| `ibkr_processor.py` | 303 | `IBKRDataProcessor` | Normalizes depth, trade, and quote events; caches aggregated top-of-book payloads; and keeps reusable buffers for ingestion loops and tests.【F:src/ibkr_processor.py†L31-L219】 | _Helper for ingestion (no direct Redis writes)_ |
| `av_ingestion.py` | 502 | `RollingRateLimiter`, `AlphaVantageIngestion` | Coordinates long-lived options and sentiment processors behind a rolling-window limiter, applies per-feed circuit breakers, and publishes telemetry for downstream consumers.【F:src/av_ingestion.py†L45-L493】 | `monitoring:alpha_vantage:metrics`, `heartbeat:alpha_vantage`【F:src/av_ingestion.py†L447-L493】 |
| `av_options.py` | 334 | `OptionsProcessor` | Emits normalized chains with per-contract metadata, fires registered callbacks, and records monitoring payloads alongside raw responses.【F:src/av_options.py†L29-L322】 | `options:{symbol}:calls`, `options:{symbol}:puts`, `options:{symbol}:chain`, `options:{symbol}:{contractID}:greeks`, `monitoring:options:{symbol}`【F:src/av_options.py†L214-L309】 |
| `av_sentiment.py` | 429 | `SentimentProcessor` | Streams news sentiment with configurable ETF exclusions, parallel technical fetches, and enriched monitoring metadata for dashboards.【F:src/av_sentiment.py†L25-L335】 | `sentiment:{symbol}`, `technicals:{symbol}:{indicator}`, `monitoring:sentiment:{symbol}`【F:src/av_sentiment.py†L74-L339】 |

### Analytics
| Module | Lines | Primary Classes | Responsibilities | Redis Touchpoints |
| --- | --- | --- | --- | --- |
| `analytics_engine.py` | 585 | `AnalyticsEngine`, `MetricsAggregator` | Coordinates analytics cadence, seeds calculators, and publishes portfolio metrics and heartbeats.【F:src/analytics_engine.py†L35-L584】 | `analytics:{symbol}:*`, `analytics:portfolio:*`, `discovered:*`, `module:heartbeat:analytics`【F:src/analytics_engine.py†L2-L13】 |
| `vpin_calculator.py` | 317 | `VPINCalculator` | Performs volume-synchronized toxicity calculations using legacy bucket logic now encapsulated here.【F:src/vpin_calculator.py†L31-L116】 | `market:{symbol}:trades`, `analytics:{symbol}:vpin`, `discovered:vpin_bucket_size`【F:src/vpin_calculator.py†L2-L12】 |
| `gex_dex_calculator.py` | 482 | `GEXDEXCalculator` | Computes gamma/delta exposure from option greeks and maintains Redis-backed rolling statistics with z-scores for volatility-aware consumers.【F:src/gex_dex_calculator.py†L47-L105】【F:src/gex_dex_calculator.py†L370-L594】 | `options:{symbol}:*:greeks`, `analytics:{symbol}:gex`, `analytics:{symbol}:dex`, `analytics:{symbol}:gex:history`【F:src/gex_dex_calculator.py†L2-L12】【F:src/gex_dex_calculator.py†L47-L70】 |
| `pattern_analyzer.py` | 477 | `PatternAnalyzer` | Detects sweep activity, hidden orders, and order-book imbalance while persisting sweep detections for 30 seconds to maintain velocity context.【F:src/pattern_analyzer.py†L293-L354】 | `market:{symbol}:book`, `market:{symbol}:trades`, `analytics:{symbol}:obi`, `analytics:{symbol}:hidden`, `analytics:{symbol}:sweep`【F:src/pattern_analyzer.py†L2-L14】【F:src/pattern_analyzer.py†L347-L354】 |
| `parameter_discovery.py` | 828 | `ParameterDiscovery` | Automates VPIN bucket tuning, temporal lookbacks, and correlation matrix generation using configurable schedules.【F:src/parameter_discovery.py†L31-L246】 | `market:{symbol}:*`, `discovered:*`【F:src/parameter_discovery.py†L2-L15】 |

### Shared Utilities
| Module | Lines | Primary Classes | Responsibilities | Redis Touchpoints |
| --- | --- | --- | --- | --- |
| `option_utils.py` | 112 | Utility functions | Centralizes trading-day calendars, DTE-to-expiry math, and expiry normalization for option selectors and execution workflows.【F:src/option_utils.py†L1-L104】 | _N/A (pure helper reused by strategies/execution)_ |

### Signal Generation
| Module | Lines | Primary Classes | Responsibilities | Redis Touchpoints |
| --- | --- | --- | --- | --- |
| `signal_generator.py` | 698 | `SignalGenerator` | Orchestrates injected strategy handlers, normalizes features, enforces guardrails/backoff, and emits deduped payloads into pending/execution queues via the shared helper.【F:src/signal_generator.py†L48-L274】【F:src/signal_deduplication.py†L224-L262】 | `analytics:{symbol}:*`, `signals:pending:{symbol}`, `signals:execution:{symbol}`, `signals:latest:{symbol}`, `health:signals:heartbeat`【F:src/signal_generator.py†L138-L403】【F:src/redis_keys.py†L47-L54】 |
| `dte_strategies.py` | 1,016 | `DTEStrategies`, `DTEFeatureSet` | Centralizes feature normalization, shared contract hysteresis, and evaluation logic for 0/1/14DTE strategies.【F:src/dte_strategies.py†L13-L218】【F:src/dte_strategies.py†L300-L498】 | Consumes analytics metrics; persists strike memory via the deduper context. |
| `moc_strategy.py` | 436 | `MOCStrategy` | Scores imbalance plays with contradiction checks, time-window weighting, and contract selection backed by dedupe hysteresis.【F:src/moc_strategy.py†L61-L304】 | `signals:last_contract:{symbol}:moc:{side}:{dte_band}`【F:src/moc_strategy.py†L16-L40】 |
| `signal_deduplication.py` | 321 | `SignalDeduplication` | Provides contract fingerprints, trading-day buckets, and atomic Lua dedupe/cooldown enforcement that now fans out to dedicated execution queues alongside distribution.【F:src/signal_deduplication.py†L21-L133】【F:src/signal_deduplication.py†L224-L262】 | `signals:emitted:{signal_id}`, `signals:cooldown:{contract_fp}`, `signals:pending:{symbol}`, `signals:execution:{symbol}`, `signals:audit:{contract_fp}`, `metrics:signals:*`【F:src/signal_deduplication.py†L1-L74】【F:src/redis_keys.py†L47-L54】 |
| `signal_distributor.py` | 410 | `SignalDistributor`, `SignalValidator` | Applies spread/liquidity/risk guardrails, persistent tier scheduling, dead-letter handling, and adaptive backoff for downstream queues.【F:src/signal_distributor.py†L27-L236】【F:src/signal_distributor.py†L289-L400】 | `distribution:*`, `signals:pending`, `metrics:signals:*`, `distribution:dead_letter`【F:src/signal_distributor.py†L1-L25】 |
| `signal_performance.py` | 201 | `PerformanceTracker` | Captures equity series, aggregates avg win/loss, profit factor, Sharpe, drawdown, and daily PnL for each strategy.【F:src/signal_performance.py†L25-L181】 | `performance:strategy:*`, `performance:signal:*`, `performance:strategy:*:pnl_series`【F:src/signal_performance.py†L2-L40】 |

### Execution & Risk
| Module | Lines | Primary Classes | Responsibilities | Redis Touchpoints |
| --- | --- | --- | --- | --- |
| `execution_manager.py` | 1,118 | `ExecutionManager` | Caches a shared risk manager, reconciles live IB positions into Redis, sanitizes persisted pending orders, tracks partial fills, and automatically builds stop/target brackets for every fill.【F:src/execution_manager.py†L59-L755】【F:src/execution_manager.py†L828-L972】 | `signals:execution:{symbol}`, `orders:pending:{order_id}`, `positions:open:*`, `health:execution:heartbeat`, `alerts:critical`【F:src/execution_manager.py†L1-L200】【F:src/redis_keys.py†L47-L54】 |
| `position_manager.py` | 1,049 | `PositionManager` | Injects a pluggable contract resolver, mirrors IB positions on startup, enforces notional trailing stops, and cancels bracket legs while recording exit telemetry on manual closes.【F:src/position_manager.py†L47-L372】【F:src/position_manager.py†L741-L799】 | `positions:open:*`, `positions:closed:*`, `positions:pnl:*`, `risk:daily_pnl`, `positions:by_symbol:{symbol}`【F:src/position_manager.py†L1-L218】 |
| `risk_manager.py` | 921 | `RiskManager` | Emits dataclass-backed VaR and drawdown snapshots, aggregates portfolio Greeks, and aligns correlation halts with circuit-breaker state.【F:src/risk_manager.py†L57-L350】 | `risk:*`, `circuit_breakers:*`, `alerts:critical`, `system:halt`, `metrics:risk:*`【F:src/risk_manager.py†L1-L214】 |
| `emergency_manager.py` | 713 | `EmergencyManager`, `CircuitBreakers` | Uses async Redis to drain queues (pending and execution), mass-cancel signals/orders, persist emergency state, and coordinate breaker resets.【F:src/emergency_manager.py†L66-L279】 | `risk:emergency:*`, `orders:emergency:*`, `signals:execution:*`, `alerts:emergency`, `metrics:emergency:*`, `circuit_breakers:*`【F:src/emergency_manager.py†L1-L210】【F:src/emergency_manager.py†L262-L274】 |

### Social & Customer Channels
These modules are scaffolding-heavy and require additional API credentials.

| Module | Lines | Primary Classes | Notes |
| --- | --- | --- | --- |
| `twitter_bot.py` | 290 | `TwitterBot`, `SocialMediaAnalytics` | Async loop outlined; API integration and Redis queries marked with TODOs for future implementation.【F:src/twitter_bot.py†L25-L133】 |
| `telegram_bot.py` | 272 | `TelegramBot` | Placeholder handlers for tiered distribution and subscription billing; extensive TODO markers remain.【F:src/telegram_bot.py†L25-L180】 |
| `discord_bot.py` | 144 | `DiscordBot` | Provides skeleton for Discord messaging; awaits implementation of specific channel workflows.【F:src/discord_bot.py†L25-L140】 |

### Dashboard & APIs
| Module | Lines | Primary Classes | Responsibilities |
| --- | --- | --- | --- |
| `dashboard_server.py` | 390 | `Dashboard`, `MetricsCollector` | FastAPI scaffolding for dashboards and REST endpoints, pending completion of routes and templates.【F:src/dashboard_server.py†L33-L223】 |
| `dashboard_routes.py` | 240 | `LogAggregator`, `AlertManager`, `PerformanceDashboard` | Defines planned REST handlers for logs, alerts, and performance data with Redis-backed caches.【F:src/dashboard_routes.py†L25-L200】 |
| `dashboard_websocket.py` | 65 | `WebSocketManager` | Outlines websocket lifecycle management for live dashboards.【F:src/dashboard_websocket.py†L25-L62】 |

### Morning Automation
| Module | Lines | Primary Classes | Responsibilities |
| --- | --- | --- | --- |
| `morning_scanner.py` | 267 | `MarketAnalysisGenerator` | Generates GPT-driven morning analysis and publishes preview/levels queues (implementation TODOs remain).【F:src/morning_scanner.py†L25-L206】 |
| `news_analyzer.py` | 318 | `ScheduledTasks` | Coordinates scheduled jobs, news summaries, and archival tasks; placeholders for real integrations.【F:src/news_analyzer.py†L25-L260】 |
| `report_generator.py` | 97 | `DataArchiver` | Archives historical data sets and produces daily reports using Redis snapshots.【F:src/report_generator.py†L25-L189】 |

### Redis Architecture & Data Flow
All modules share a single Redis schema defined in `redis_keys.py`, which standardizes market data, analytics metrics, portfolio/sector summaries, order/position keys, and TTL policies.【F:src/redis_keys.py†L25-L205】 Typed helper functions (`get_market_key`, `get_options_key`, `get_signal_key`, `get_system_key`, `get_portfolio_key`, etc.) now replace the legacy `Keys` class, so ingestion, analytics, and signal pipelines construct canonical keys through validated call sites while preserving compatibility with existing Redis namespaces.【F:src/redis_keys.py†L25-L205】【F:src/ibkr_ingestion.py†L623-L698】【F:src/signal_generator.py†L138-L403】

```
           +---------------------+       +-----------------------+
           |  Market Data Feeds  |       |  Alt Data (Alpha V.)  |
           +----------+----------+       +-----------+-----------+
                      |                              |
                      v                              v
        +--------------------------+   +---------------------------+
        |   IBKR / AV Ingestion    |   |   Options / Sentiment     |
        +--------------------------+   +---------------------------+
                      |                              |
                      +--------------+---------------+
                                     v
                           +--------------------+
                           |      Redis        |
                           |  (market, metrics,|
                           |   signals, risk)  |
                           +--------------------+
            /--------------------|---------------------\
           v                     v                     v
+----------------+    +----------------------+   +-----------------------+
|  Analytics     |    |  Signal Engine &     |   | Execution / Risk /    |
| (VPIN, GEX/DEX)|    |  Distribution        |   | Emergency Controls    |
+----------------+    +----------------------+   +-----------------------+
                                                        |
                                                        v
                                   +----------------------------------+
                                   | Dashboard / Social / Reporting   |
                                   +----------------------------------+
```

Each box consumes and publishes Redis keys only; orchestration (`main.py`) wires modules together by passing shared Redis connections, keeping runtime coupling minimal.【F:src/main.py†L97-L192】

Signals published by the dedupe helper now land in both `signals:pending:{symbol}` and `signals:execution:{symbol}` queues so downstream distribution and execution loops progress independently; `SignalDistributor` drains the pending lists for customer delivery while `ExecutionManager` polls the execution lists to stage bracket orders without waiting on distribution throughput.【F:src/signal_deduplication.py†L224-L262】【F:src/execution_manager.py†L256-L292】


### Analytics Cadence & Aggregation
- **Cadence-aware scheduler** – `AnalyticsEngine` coordinates VPIN, GEX/DEX, and pattern analyzers on configurable intervals, gating execution outside RTH windows and surfacing idle/running state via module heartbeats.【F:src/analytics_engine.py†L400-L556】
- **Portfolio & sector rollups** – `MetricsAggregator` composes symbol snapshots into `analytics:portfolio:summary`, `analytics:sector:{sector}`, and correlation matrices with TTLs enforced through the schema helpers.【F:src/analytics_engine.py†L35-L205】【F:src/redis_keys.py†L25-L304】
- **Config-driven controls** – `config/config.yaml` exposes analytics TTLs, update cadences, and sector mappings so operators can tune schedules without code changes.【F:config/config.yaml†L84-L115】
- **Canonical calculator outputs** – VPIN, GEX/DEX, and pattern analyzers now publish into the canonical `analytics:{symbol}:*` keys via the shared helper layer, aligning all downstream consumers.【F:src/vpin_calculator.py†L31-L144】【F:src/gex_dex_calculator.py†L1-L514】【F:src/pattern_analyzer.py†L1-L460】

### Logging
- **Structured JSON output** – All modules now emit structured records through `logging_utils.StructuredFormatter`, adding `component`, `subsystem`, and action metadata to every line in `logs/alphatrader.log` for easier searching and ingestion.【F:src/logging_utils.py†L18-L189】【F:main.py†L29-L53】【F:src/analytics_engine.py†L31-L117】【F:src/ibkr_ingestion.py†L40-L215】
- **Console-friendly view** – Terminal output now renders a compact, color-free key/value line per event while the rotating file handler keeps full JSON payloads for downstream ingestion.【F:src/logging_utils.py†L95-L189】
- **Centralised configuration** – Logging options (level, rotation, console mirroring) continue to live under the `logging` block in `config/config.yaml`; updating the file and restarting applies without code edits.【F:config/config.yaml†L393-L402】【F:src/logging_utils.py†L120-L189】
- **Context adapters** – `get_logger` wraps the standard library logger with a context adapter so subsystems automatically annotate events without repeating boilerplate everywhere.【F:src/logging_utils.py†L170-L189】【F:src/signal_generator.py†L46-L181】【F:src/av_ingestion.py†L32-L237】

### Signal Engine Enhancements
- **Dependency-injected strategies** – `SignalGenerator` resolves 0DTE/1DTE/14DTE and MOC handlers through injectable factories, shares a pluggable feature reader, and backs off automatically on repeated failures to protect downstream services.【F:src/signal_generator.py†L51-L209】
- **Normalized feature payloads** – `DTEFeatureSet` guarantees typed defaults (price, VPIN, gamma, imbalance, flows) before strategies score opportunities, avoiding missing-key crashes and aligning analytics expectations.【F:src/dte_strategies.py†L13-L218】
- **Options-only contract normalization** – 0/1/14DTE and MOC selectors emit OCC symbols, validated strikes, and IBKR-formatted expiries using the shared option helpers so downstream dedupe/execution never receive stock orders.【F:src/dte_strategies.py†L13-L218】【F:src/moc_strategy.py†L61-L304】【F:src/option_utils.py†L1-L104】
- **Analytics feature alignment** – `default_feature_reader` now pulls VPIN toxicity from the canonical `value` field and prefers order-book imbalance readings from `level1_imbalance`/`level5_imbalance`, falling back to historical aliases to stay compatible with older payloads and restoring confidence scores that were previously capped by zeros.【F:src/signal_generator.py†L624-L658】
- **Contract hysteresis** – Both DTE and MOC strategies reuse the shared deduper to persist strike memory, roll contracts only when spot/imbalance thresholds are met, and return cached selections when conditions are stable.【F:src/dte_strategies.py†L300-L498】【F:src/moc_strategy.py†L225-L304】
- **Contradiction-aware MOC signals** – MOC direction logic weighs gamma pins, order-book imbalance, indicative price, and recent bars to flatten signals when cues disagree, while time-window scoring emphasizes the final auction minutes.【F:src/moc_strategy.py†L102-L224】
- **Distribution guardrails** – `SignalDistributor` layers validator checks for spreads, halts, and liquidity, applies persistent tier scheduling, and captures dead-letter/backoff metrics for operator visibility.【F:src/signal_distributor.py†L27-L236】【F:src/signal_distributor.py†L289-L400】
- **Execution queue fan-out** – Deduped emits now land in both `signals:pending:{symbol}` (for distribution) and `signals:execution:{symbol}` so IBKR routing remains responsive even when downstream customer queues are backlogged.【F:src/signal_deduplication.py†L224-L262】【F:src/execution_manager.py†L256-L292】【F:src/redis_keys.py†L47-L54】
- **Performance telemetry** – `PerformanceTracker` tallies win/loss counts, equity curves, drawdowns, and daily PnL so strategies can be reviewed without exporting raw fills.【F:src/signal_performance.py†L25-L181】

### Contract-Centric Deduplication
Signal deduplication was hardened to guarantee idempotency and multi-worker safety:
- **Stable fingerprints**: `contract_fingerprint` hashes symbol, strategy, expiry, strike, multiplier, and venue into a deterministic key (`sigfp:<hash>`).【F:src/signal_deduplication.py†L21-L47】
- **Trading-day buckets**: `trading_day_bucket` aligns IDs to the NYSE session instead of UTC to survive overnight trading.【F:src/signal_deduplication.py†L49-L63】
- **Atomic Lua pipeline**: `LUA_ATOMIC_EMIT` performs SETNX, cooldown enforcement, and enqueue in one Redis script, returning 1/0/-1 for success, duplicate, or cooldown blocks.【F:src/signal_deduplication.py†L76-L116】
- **Observability**: Blocked reasons are tallied under `metrics:signals:*` and full audits are stored per contract fingerprint.【F:src/signal_deduplication.py†L118-L133】
- **Performance**: The Day-6 hardening achieved 95.2% duplicate suppression with zero race-condition duplicates in backtesting.【F:DEDUPLICATION_CHANGES.md.archived†L1-L85】
- **Dedicated execution queues**: The Lua script pushes accepted payloads into both pending and execution lists so distribution, dedupe, and IBKR consumers remain decoupled while honoring cooldown semantics.【F:src/signal_deduplication.py†L224-L262】【F:src/execution_manager.py†L256-L292】

### Pattern-Based Toxicity Analytics
Because IBKR does not expose market-maker identities, toxicity detection blends VPIN, venue heuristics, trade-size patterns, and order-book imbalance volatility. Suggested scoring weights and venue toxicity tables are documented for further tuning.【F:toxicity_approach.md.archived†L1-L78】 Parameter discovery applies these heuristics during pattern analysis, leveraging odd-lot ratios, sweep detection, and book variance to label flow without requiring dark-pool attribution.【F:src/parameter_discovery.py†L200-L330】【F:toxicity_approach.md.archived†L34-L58】 Sweep detections now persist for 30 seconds in `analytics:{symbol}:sweep`, allowing downstream strategies to respond to bursts of aggressive flow without recomputing heuristics every tick.【F:src/pattern_analyzer.py†L293-L354】

### Installation & Setup
1. **Python environment** – Use Python 3.10+ and install core dependencies: `pip install -r requirements.txt`. Optional extras (FastAPI, Discord, Tweepy, OpenAI, etc.) are listed but commented for feature-based installation.【F:requirements.txt†L1-L40】【F:requirements.txt†L40-L64】
2. **Redis** – Configure Redis using `config/redis.conf` or a managed instance, then update host/port in `config/config.yaml` if needed.【F:config/config.yaml†L12-L21】
3. **Credentials** – Provide environment variables such as `ALPHA_VANTAGE_API_KEY` before launching; `main.py` loads `.env` and substitutes `${VAR}` placeholders automatically.【F:config/config.yaml†L41-L50】【F:src/main.py†L43-L82】
4. **Run the orchestrator** – Execute `python main.py` to initialize modules based on configuration flags. Modules marked disabled in `config/config.yaml` will be skipped until ready.【F:config/config.yaml†L58-L160】【F:src/main.py†L97-L192】

### Market Data Enhancements
- **IBKR normalization & freshness telemetry** – `IBKRIngestion` now routes every depth, trade, and quote update through `IBKRDataProcessor`, which caches DOM snapshots, aggregated top-of-book payloads, and trade buffers for reuse in downstream writers and tests. Staleness and metrics loops surface gaps via Redis so operators can alert before analytics drift.【F:src/ibkr_processor.py†L53-L219】【F:src/ibkr_ingestion.py†L68-L210】【F:src/ibkr_ingestion.py†L560-L748】
- **Alpha Vantage resilience** – The ingestion loop reuses long-lived option and sentiment processors, wraps fetches in jittered exponential retries, and applies per-symbol/per-feed circuit breakers that back off noisy endpoints while continuing healthy ones. Rolling-window rate limiter stats are exposed for real-time capacity tracking.【F:src/av_ingestion.py†L76-L493】
- **Option-chain instrumentation** – `OptionsProcessor` emits normalized payloads with expiration summaries, per-contract metadata, and monitoring counters before fanning out optional callbacks over the configurable pub/sub channel (default `events:options:chain`).【F:src/av_options.py†L64-L321】【F:config/config.yaml†L41-L92】
- **Sentiment & technical metadata** – `SentimentProcessor` respects configurable ETF skip lists, fetches RSI/MACD/BBANDS/ATR in parallel, and stores rich monitoring payloads that capture freshness, article mix, and technical readings for dashboards.【F:src/av_sentiment.py†L29-L339】

### Execution & Risk Enhancements
- **Shared risk manager cache** – `ExecutionManager` lazily instantiates a shared `RiskManager` via async factory injection so correlation checks reuse a single instance while falling back to on-demand construction when needed.【F:src/execution_manager.py†L59-L312】
- **Notional-aware sizing & guards** – Order sizing records the computed dollar exposure on each signal, reruns risk checks post-sizing, and persists the notional alongside fills so daily risk and reconciliation reports stay in sync.【F:src/execution_manager.py†L840-L1593】
- **Commission-normalized P&L** – All fills convert the broker’s negative commission values into absolute costs, ensuring on-ledger P&L and daily reconciliation figures remain accurate to the cent across entry, stop, and scale-out events.【F:src/execution_manager.py†L115-2065】【F:reconcile_daily_pnl.py†L85-L138】
- **Options-only order validation** – Execution flows default to option contracts, validate normalized strikes/expiries, and derive TTL/expiry hysteresis from shared helpers before routing any IBKR orders, preventing stray equity trades.【F:src/execution_manager.py†L256-L372】【F:src/option_utils.py†L1-L104】【F:src/signal_generator.py†L167-L274】
- **Redis reconciliation & sanitized order state** – Startup reconciliation mirrors live IBKR positions into canonical Redis keys and strips signal metadata before persisting pending orders, keeping operators aligned without leaking analytics payloads.【F:src/execution_manager.py†L187-L678】【F:src/position_manager.py†L93-L218】
- **Partial-fill and timeout telemetry** – Trade synchronization records aggregate fill prices, publishes partial-fill alerts once per order, increments timeout counters, and escalates slow orders with cancel-plus-alert flows for operations visibility.【F:src/execution_manager.py†L632-L755】
- **Automatic bracket management** – Filled orders spawn linked stop-loss and profit targets with shared OCA groups, and manual closes tear down every bracket leg before marking the position complete to prevent orphaned exposure.【F:src/execution_manager.py†L828-L972】【F:src/position_manager.py†L741-L773】
- **Durable trailing stops & EOD trims** – `PositionManager` maintains strategy-specific trailing stop heuristics, updates high-water marks against real-time prices, and enforces configurable end-of-day reductions or closures using Redis-backed state.【F:src/position_manager.py†L70-L372】
- **Structured risk snapshots** – Dataclass-backed VaR and drawdown snapshots flow through `RiskManager`, which aggregates portfolio Greeks, enforces correlation halts, and emits summarized metrics for dashboards and consumers.【F:src/risk_manager.py†L57-L350】
- **Emergency drain coverage** – `EmergencyManager` drains pending and execution signal queues alongside orders, archives cancel metadata, and toggles breaker state via async Redis primitives before triggering alerts and halts.【F:src/emergency_manager.py†L80-L279】【F:src/emergency_manager.py†L262-L274】

### Telemetry & Monitoring
- `monitoring:ibkr:metrics` – Aggregated IBKR update counts, reconnect attempts, and stale symbol flags refreshed every 10 seconds in lockstep with the ingestion heartbeat.【F:src/ibkr_ingestion.py†L718-L766】
- `monitoring:data:stale` – Symbol→age hash populated when market data exceeds freshness thresholds, cleared outside market hours or once feeds recover.【F:src/ibkr_ingestion.py†L588-L645】
- `monitoring:alpha_vantage:metrics` – RollingRateLimiter token levels, average wait times, and circuit-breaker suspensions exported for Alpha Vantage fetch loops.【F:src/av_ingestion.py†L440-L493】
- `monitoring:options:{symbol}` – Contract counts, skipped contracts, and missing Greeks for each chain alongside the canonical `options:{symbol}:chain` document.【F:src/av_options.py†L253-L287】
- `monitoring:sentiment:{symbol}` – Sentiment averages, article age, and source counts mirroring the stored news feed to simplify health dashboards.【F:src/av_sentiment.py†L312-L339】
- `analytics:{symbol}:gex` / `analytics:{symbol}:dex` – Per-symbol exposure snapshots now include rolling z-scores, rolling means, and sample counts so strategies can quickly detect extremes without replaying history.【F:src/gex_dex_calculator.py†L47-L105】【F:src/gex_dex_calculator.py†L370-L594】
- `analytics:{symbol}:sweep` – Binary sweep detections persist for 30 seconds to maintain velocity context for short-lived toxicity events and are emitted alongside detailed logs for operator review.【F:src/pattern_analyzer.py†L293-L354】
- `analytics:{symbol}:moc` – Synthetic closing-imbalance projections combining order-book skew, projected volume, and gamma posture so the MOC strategy has dependable inputs even without exchange feeds.【F:src/analytics_engine.py†L676-L820】【F:src/moc_strategy.py†L55-L214】
- `analytics:{symbol}:unusual` – Unusual-options activity scores summarizing volume/open-interest ratios, notional bursts, and call/put dominance that feed directly into 14DTE confidence gates.【F:src/analytics_engine.py†L822-L1010】【F:src/dte_strategies.py†L748-L1009】
- `events:options:chain` – Optional pub/sub broadcast for consumers that want to react immediately to refreshed chains; override `alpha_vantage.options_pubsub_channel` to use an alternate topic.【F:src/av_ingestion.py†L125-L210】【F:config/config.yaml†L41-L97】

- **Redis schema** – Use the helpers in `redis_keys.py` when adding new keys to guarantee consistency and TTL discipline. The legacy `Keys` class has been retired in favor of explicit helper functions (e.g., `market_bars_key`, `analytics_vpin_key`, `get_system_key`) so call sites validate namespaces at import time while analytics calculators and ingestion loops stay aligned on shared naming conventions.【F:src/redis_keys.py†L25-L304】【F:src/analytics_engine.py†L35-L205】【F:src/ibkr_ingestion.py†L560-L748】 The latest analytics additions reuse these helpers (`analytics_metric_key(symbol, 'moc'/'unusual')`) so signal consumers can rely on consistent key derivation.【F:src/analytics_engine.py†L676-L1010】【F:src/signal_generator.py†L564-L803】
- **Option utilities** – `option_utils.py` centralizes trading-day calendars and expiry normalization; reuse `compute_expiry_from_dte`/`normalize_expiry` when introducing new strategies or execution paths to keep option routing consistent.【F:src/option_utils.py†L1-L104】【F:src/dte_strategies.py†L158-L204】
- **Modular testing** – Each major service exposes async `start()` and `stop()` methods (or synchronous equivalents) to support targeted integration tests and rehearsal in isolation.【F:src/analytics_engine.py†L228-L341】【F:src/signal_generator.py†L121-L199】
- **Configuration-first design** – Module toggles and thresholds live under `config/config.yaml` (`modules`, `signals.strategies`, `risk_management`, etc.), making it straightforward to stage or disable features during development.【F:config/config.yaml†L58-L288】
- **Optional services** – Social, dashboard, and morning-analysis modules contain extensive TODO markers describing pending work; treat them as blueprints until the required APIs and dependencies are provisioned.【F:src/twitter_bot.py†L35-L133】【F:src/dashboard_server.py†L83-L223】【F:src/morning_scanner.py†L85-L206】

- **Testing update** – `pytest` remains the primary harness (`python3 -m pytest`), but older day-based suites still import legacy packages such as `src.signals`. Run targeted modules (e.g., `tests/test_analytics_engine.py`) or provide the missing legacy modules when executing the full suite to avoid collection failures like the `ModuleNotFoundError: src.signals` observed on `tests/test_day6.py`.【F:tests/test_day6.py†L18-L36】

## Open Work

### Community Publishing & Automation (In Planning)
Social distribution, dashboards, and morning automation each ship with scaffolded classes that already accept the shared configuration and Redis handle, but their TODO blocks must be completed before public launch:

- **TwitterBot & SocialMediaAnalytics** – Wire up credential loading, Tweepy clients, posting loops, and engagement tracking so winning trades, teasers, and daily summaries reach external audiences while analytics land back in Redis.【F:src/twitter_bot.py†L25-L206】
- **DiscordBot** – Finish the webhook session, embed formatting, and queue draining to mirror the premium/basic tiers with delivery/error metrics tracked for dashboards.【F:src/discord_bot.py†L25-L141】
- **TelegramBot** – Implement command handlers, Stripe subscription flow, tier-aware formatting, and delayed distribution across premium/basic/free channels.【F:src/telegram_bot.py†L25-L218】
- **Dashboard services** – Populate FastAPI routes, WebSocket broadcasting, log aggregation, alert management, and performance chart generation so operators (and community members) get a real-time control surface.【F:src/dashboard_server.py†L25-L220】【F:src/dashboard_routes.py†L25-L184】【F:src/dashboard_websocket.py†L1-L70】
- **MorningScanner** – Complete the GPT-4 workflow that gathers overnight data, options positioning, economic events, and distributes the resulting analysis to premium feeds and social teasers.【F:src/morning_scanner.py†L25-L220】
- **ReportGenerator & MetricsCollector** – Turn the archival, cleanup, and monitoring TODOs into live services so historical exports and dashboard metrics stay fresh.【F:src/report_generator.py†L25-L120】【F:src/dashboard_server.py†L166-L220】

### Pending Integrations
- [`src/twitter_bot.py`](src/twitter_bot.py)
- [`src/discord_bot.py`](src/discord_bot.py)
- [`src/telegram_bot.py`](src/telegram_bot.py)
- [`src/dashboard_server.py`](src/dashboard_server.py), [`src/dashboard_routes.py`](src/dashboard_routes.py), [`src/dashboard_websocket.py`](src/dashboard_websocket.py)
- [`src/morning_scanner.py`](src/morning_scanner.py)
- [`src/news_analyzer.py`](src/news_analyzer.py)
- [`src/report_generator.py`](src/report_generator.py)

### Roadmap Snapshot
Implementation priorities are tracked in `implementation_plan.md`, including module completion percentages, outstanding TODO counts, and integration milestones across execution, social, and dashboard pillars.【F:implementation_plan.md†L1-L40】 Refer to that document before beginning new work to understand dependencies and sequencing.
