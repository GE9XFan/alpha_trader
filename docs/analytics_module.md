# Analytics Module Deep Dive

## Mission Statement
Transform raw market structure, options positioning, and volatility telemetry in Redis into actionable analytics artifacts with predictable cadences, consistent schemas, and resilient fallbacks so signal generation, risk, and portfolio monitoring can operate on a unified, timely state.

## Component Matrix
| Module | Responsibilities | Core Loops / Handlers | Redis Touchpoints |
|--------|------------------|-----------------------|-------------------|
| `src/analytics_engine.py` (`AnalyticsEngine`, `MetricsAggregator`) | Orchestrate per-symbol calculators, maintain heartbeats, aggregate cross-symbol metrics | `start`, `_calculation_loop`, `_run_calculation`, `_gather_symbol_snapshots`, `calculate_portfolio_metrics`, `calculate_sector_flows`, `calculate_cross_asset_correlations` | `analytics:{symbol}:*`, `analytics:portfolio:*`, `market:{symbol}:*`, `discovered:*`, `heartbeat:analytics` |
| `src/vpin_calculator.py` (`VPINCalculator`) | Compute volume-synchronized probability of informed trading with Lee-Ready classification and bucket history | `calculate_vpin`, `_classify_trades`, `_create_volume_buckets`, `_calculate_vpin_from_buckets` | `analytics:{symbol}:vpin`, `market:{symbol}:ticker`, `market:{symbol}:trades`, `discovered:vpin_bucket_size` |
| `src/gex_dex_calculator.py` (`GEXDEXCalculator`) | Derive gamma/delta exposure heatmaps using normalized option chains or legacy greeks keys, record z-scores | `calculate_gex`, `calculate_dex`, `_collect_option_keys`, `_update_metric_statistics` | `analytics:{symbol}:gex`, `analytics:{symbol}:dex`, `options:{symbol}:*`, `market:{symbol}:ticker`, `analytics:{symbol}:{metric}:history` |
| `src/pattern_analyzer.py` (`PatternAnalyzer`) | Assess flow toxicity, classify institutional vs retail flow, compute order book imbalance, sweep/hidden activity | `analyze_flow_toxicity`, `_compute_participant_flows`, `calculate_order_book_imbalance`, `detect_sweeps`, `detect_hidden_orders` | `analytics:{symbol}:toxicity`, `analytics:{symbol}:institutional_flow`, `analytics:{symbol}:retail_flow`, `analytics:{symbol}:obi`, `analytics:{symbol}:metric:*`, `market:{symbol}:book`, `market:{symbol}:trades`, `analytics:{symbol}:vpin` |
| `src/dealer_flow_calculator.py` (`DealerFlowCalculator`) | Compute vanna, charm, skew, and hedging elasticity across expiries; maintain historical z-scores | `calculate_dealer_metrics`, `_aggregate_metrics`, `_compute_skew`, `_update_history` | `analytics:{symbol}:vanna`, `analytics:{symbol}:charm`, `analytics:{symbol}:hedging_impact`, `analytics:{symbol}:skew`, `options:{symbol}:chain`, `market:{symbol}:ticker` |
| `src/flow_clustering.py` (`FlowClusterModel`) | Cluster recent trades into strategy archetypes (momentum/mean-reversion/hedging) and participant mix | `classify_flows`, `_load_trades`, `_cluster`, `_compute_cluster_stats`, `_build_payload` | `analytics:{symbol}:flow_clusters`, `market:{symbol}:trades` |
| `src/volatility_metrics.py` (`VolatilityMetrics`) | Ingest VIX1D from multiple sources with caching/backoff; compute regime, z-score, percentile | `update_vix1d`, `_fetch_vix1d_*`, `_update_history`, `_classify_regime`, `ingest_manual`, `close` | `analytics:vix1d`, `volatility:vix1d:*`, `discovered:*` |

## Data Flow Overview
```
Redis market namespace ─┐
                        │               ┌──────────────┐
Options chains / Greeks ├─▶ Analytics Engine ─▶ Calculators ─▶ analytics:{symbol}:*
Volatility feeds ───────┘               │              │
                                       └─▶ Metrics Aggregator ─▶ analytics:portfolio:*
```
The engine coordinates asynchronous calculators and persists their outputs back to Redis. Downstream consumers (signal generator, dashboard, monitoring) read from the analytics keys exclusively, decoupling them from raw market ingestion schemas.

## Analytics Engine
### Startup & Scheduling
1. `AnalyticsEngine(config, redis_conn)` loads symbol lists, TTLs (`get_ttl` overrides), update intervals, and market hour preferences (`analytics_rth_only`, `market.extended_hours`).
2. `start()` instantiates calculator dependencies (`VPINCalculator`, `GEXDEXCalculator`, `PatternAnalyzer`, `DealerFlowCalculator`, `FlowClusterModel`, `VolatilityMetrics`) and enters `_calculation_loop()`.
3. `_calculation_loop()` runs at `cadence_hz` (default 2 Hz). For each configured interval (`vpin`, `gex_dex`, `flow_toxicity`, `order_book`, `sweep_detection`, `dealer_flows`, `flow_clustering`, `volatility`, `portfolio`, `sectors`, `correlation`, `moc`, `unusual_activity`) it checks elapsed wall-clock time and triggers `_run_calculation(calc_type)`. Market-hours gating pauses symbol-level work and only publishes idle heartbeats when outside RTH (unless extended hours enabled).
4. After each loop iteration the engine writes a heartbeat (`heartbeat:analytics`) containing cadence, running flag, last-update map, and per-job error counters.

### Execution Semantics
- `_run_calculation` fans out async tasks (one per symbol/calculator) using `asyncio.gather`. Failures increment `error_counts[calc_type]` and are logged; success updates `last_updates` timestamps.
- `dealer_flows`, `flow_clustering`, `moc`, and `unusual_activity` rely on calculators defined below. The `volatility` job simply calls `VolatilityMetrics.update_vix1d()`.
- Portfolio/sectors/correlation jobs run synchronously because they need consolidated state. They call into `MetricsAggregator`.

## Metrics Aggregator
### Snapshot Gathering
`_gather_symbol_snapshots()` fetches per-symbol analytics and market context. The implementation batches Redis GETs via a pipeline (`vpin`, `gex`, `dex`, `vanna`, `charm`, `hedging`, `skew`, `flow_clusters`, `market ticker`) to reduce round-trip overhead. Each snapshot includes:
- Latest VPIN, GEX, DEX, vanna/charm/hedging/skew metrics (with z-scores when present)
- Flow cluster distributions
- Volume (from ticker feed)
- Cached `analytics:vix1d` value for embedding volatility context

### Aggregations
- `calculate_portfolio_metrics()` builds cross-symbol aggregates (counts, average VPIN, gamma/delta totals, vanna/charm/hedging notionals, toxic flow counts, sector summaries) and stores them at `analytics:portfolio:summary` with TTL `store_ttls.portfolio`.
- `calculate_sector_flows()` reuses the last snapshots (or re-fetches on cache miss) to emit per-sector flow payloads keyed by `analytics:portfolio:sector:{sector}`.
- `calculate_cross_asset_correlations()` prefers a cached discovered matrix (`discovered:correlation_matrix`). If absent it computes correlations from Redis bars (`market:{symbol}:bars:1min`). The payload includes matrix, pair counts, and high-correlation pairs flagged against a risk threshold.

## Calculator Highlights
### VPIN Calculator (`src/vpin_calculator.py`)
- Pulls trade history (last 1000 events) from `market:{symbol}:trades`, classifies buys/sells using Lee-Ready with midpoint fallback, then builds volume buckets (`default_bucket_size` or discovered override).
- Computes VPIN either from bucket imbalances or, when bucket count is low, a weighted simple imbalance.
- Stores results at `analytics:{symbol}:vpin` with TTL `store_ttls.analytics`.
- Guards against malformed ticker payloads (`json.loads` returning `None`) and throttles missing-bucket-size warnings.

### GEX/DEX Calculator (`src/gex_dex_calculator.py`)
- Prefers normalized chain (`options:{symbol}:chain`) but can fall back to scanning `options:{symbol}:*:greeks` using `SCAN` (non-blocking).
- For each contract it validates gamma/open-interest ranges, enforces positive strikes, and now treats puts with the same sign as calls when only counterparty OI is known—preventing artificial cancellation.
- Aggregates per-strike GEX and DEX, filters by minimum OI, identifies key strikes (max GEX, zero-gamma) and directional bias, and records z-score history in Redis lists (`analytics:{symbol}:{metric}:history`).
- Persists payloads at `analytics:{symbol}:gex` and `analytics:{symbol}:dex`.

### Pattern Analyzer (`src/pattern_analyzer.py`)
- `analyze_flow_toxicity` reads VPIN and classifies flow into `low/moderate/high`, storing at `analytics:{symbol}:toxicity`.
- `calculate_order_book_imbalance` computes L1/L5 imbalance, pressure ratio, micro-price, and book velocity, persisting to `analytics:{symbol}:obi` and maintaining a short-lived history key for velocity.
- `detect_sweeps` scans last 50 trades, enforces 1-second window and directional consistency, and now stores a structured payload (value, timestamps, counts) at `analytics:{symbol}:metric:sweep`.
- `detect_hidden_orders` loads trades robustly (skipping malformed JSON) to infer dark trades, iceberg refills, and volume-depth anomalies.

### Dealer Flow Calculator (`src/dealer_flow_calculator.py`)
- Normalizes each option contract (type, strike, expiry, implied vol) via `normalize_expiry`, filters out far expiries and low OI, and computes Greeks using either provided values or Black-Scholes estimations.
- Aggregates total vanna/charm/gamma contributions (shares and notional) per symbol and per expiry bucket, and computes skew by pairing delta ~0.25 call/put contracts.
- Updates rolling history for vanna/charm/skew using pipeline operations and stores metrics with TTL `store_ttls.analytics`.

### Flow Clustering (`src/flow_clustering.py`)
- Loads recent trades (`window_trades`, default 150), engineering features (log size, direction, volatility-scaled price change, log notional, sweep flag, log time gap).
- Runs KMeans(k=3) with fallback for low-variance samples (assigns all trades to one cluster while keeping centroid arrays consistent).
- Deterministically maps clusters to `momentum`, `hedging`, `mean_reversion` without reusing indices, and computes participant distribution based on average trade size vs `institutional_size` threshold.
- Stores payload at `analytics:{symbol}:flow_clusters` with TTL derived from update interval.

### Flow Toxicity Heuristics (`PatternAnalyzer.analyze_flow_toxicity`)
- Pulls the latest VPIN value, recent trades (`market:{symbol}:trades`), and top-of-book quotes to rerun a lightweight Lee–Ready classification without relying on venue identifiers.
- Computes aggressor imbalance, spread-cross volume, and large-trade ratios (configurable via `modules.analytics.toxicity_heuristics`) to infer institutional participation and venue proxies.
- Derives participant segmentation heuristics: notional thresholds, dark-pool venues, time-clustered executions, and passive fills contribute to an institutional score; small-lot aggressive prints bias toward retail classification.
- Blends VPIN with the heuristics (default weights 50/30/20 for VPIN/aggressor/large trades) to publish an adjusted toxicity score, level, confidence, and derived venue mix at `analytics:{symbol}:toxicity`.
- Persists complementary participant metrics at `analytics:{symbol}:institutional_flow` / `analytics:{symbol}:retail_flow`, each containing notional share, trade counts, heuristic hits, and aggressor ratios so downstream scoring can weight institutional confirmation directly.
- Downstream modules consume the adjusted score (`toxicity_adjusted`), component ratios (`aggressor_ratio`, `large_trade_ratio`, `institutional_score`), and participant payloads via the feature reader.

### Volatility Metrics (`src/volatility_metrics.py`)
- Attempts multiple sources in order (`cboe`, `ibkr`, `yahoo`) with caching to avoid rate limits. Results get cached in Redis (e.g., `volatility:vix1d:cboe:history`).
- Maintains exponential backoff state (`_failure_count`, `_next_retry_ts`); if all sources fail, raises an error and leaves the previous payload untouched instead of overwriting with `{error: ...}`.
- On success, computes rolling z-score, percentile, and regime classification (BENIGN/NORMAL/ELEVATED/SHOCK) and writes to `analytics:vix1d` + history list.

## Redis Key Conventions
| Namespace | Description | Producers |
|-----------|-------------|-----------|
| `analytics:{symbol}:vpin` | VPIN payload with bucket metadata | `VPINCalculator` |
| `analytics:{symbol}:gex` / `dex` | Exposure summaries and z-scores | `GEXDEXCalculator` |
| `analytics:{symbol}:vanna` / `charm` / `hedging_impact` / `skew` | Dealer positioning metrics | `DealerFlowCalculator` |
| `analytics:{symbol}:obi` | Order book imbalance metrics | `PatternAnalyzer` |
| `analytics:{symbol}:metric:*` | Sweep/hidden/unusual etc. analytics | `PatternAnalyzer`, `_calculate_unusual_activity`, `_calculate_moc_metrics` |
| `analytics:{symbol}:institutional_flow` / `retail_flow` | Participant flow ratios and heuristic hits | `PatternAnalyzer._compute_participant_flows` |
| `analytics:{symbol}:flow_clusters` | Trade clustering payload | `FlowClusterModel` |
| `analytics:portfolio:summary` | Portfolio-wide aggregates | `MetricsAggregator.calculate_portfolio_metrics` |
| `analytics:portfolio:sector:{sector}` | Sector-level flow summaries | `MetricsAggregator.calculate_sector_flows` |
| `analytics:portfolio:correlation` | Cross-asset correlation matrix | `MetricsAggregator.calculate_cross_asset_correlations` |
| `analytics:vix1d` | VIX1D regime snapshot | `VolatilityMetrics` |
| `heartbeat:analytics` | Engine heartbeat document | `AnalyticsEngine._update_heartbeat` |

## Operational Considerations
- **Cadence Control**: Intervals are configurable under `modules.analytics.update_intervals`. Ensure they respect calculator execution time; the pipeline refactor reduces per-symbol latency but heavy options scans can still stretch the loop.
- **Redis TTLs**: Defaults are derived from `redis_keys.get_ttl` but can be overridden with `modules.analytics.store_ttls`. Keep analytics TTL >= update interval to avoid gaps.
- **Market Hours**: With `analytics_rth_only=true`, symbol calculations pause outside 9:30–16:00 ET (configurable extended hours). Heartbeats remain live so monitoring knows the module is idle, not dead.
- **Backoff & Errors**: The engine increments `error_counts` per calc_type and exposes them in the heartbeat. Consumers should monitor this map and alert if counts grow consecutively.
- **Dependency Ordering**: Some calculators rely on others (e.g., toxicity requires VPIN). Ensure their intervals are compatible so dependent metrics don’t consistently read stale data.

## Source Reference
- Analytics Orchestrator: `src/analytics_engine.py`
- Portfolio Aggregator: `src/analytics_engine.py` (`MetricsAggregator`)
- VPIN Calculator: `src/vpin_calculator.py`
- GEX/DEX Calculator: `src/gex_dex_calculator.py`
- Pattern Analyzer: `src/pattern_analyzer.py`
- Dealer Flow Metrics: `src/dealer_flow_calculator.py`
- Flow Clustering: `src/flow_clustering.py`
- Volatility Metrics: `src/volatility_metrics.py`
