# Signal Engine Deep Dive

## Mission Statement
Coordinate multi-strategy options signals that are grounded in real-time analytics, enforce institutional-grade guardrails (freshness, exposure, dedupe, acknowledgements), and hand execution-ready payloads to the downstream order manager and distribution tiers without emitting noise or duplicate contracts.

## Component Matrix
| Module | Responsibilities | Core Routines | Redis Touchpoints |
|--------|------------------|---------------|-------------------|
| `src/signal_generator.py` (`SignalGenerator`) | Scheduler, feature ingestion, strategy routing, exposure & veto enforcement, dedupe integration | `start`, `_evaluate_symbol`, `_process_valid_signal`, `_handle_emit_result`, `_gather_exposure_state` | `health:signals:heartbeat`, `analytics:*`, `signals:pending:{symbol}`, `signals:execution:{symbol}`, `metrics:signals:*` |
| `src/dte_strategies.py` (`DTEStrategies`) | 0DTE/1DTE/14DTE scoring, hard veto checks, contract selection hysteresis | `evaluate`, `evaluate_0dte_conditions`, `evaluate_14dte_conditions`, `select_contract` | Consumes feature dicts; emits reason lists stored in signal payload |
| `src/moc_strategy.py` (`MOCStrategy`) | Market-on-close imbalance analytics, contract selection, hysteresis | `evaluate_moc_conditions`, `select_contract` | `analytics:{symbol}:moc*`, `signals:last_contract:*` |
| `src/signal_deduplication.py` (`SignalDeduplication`) | Fingerprinting, TTL-aware idempotency, live-lock gating, audit logging, update channel | `contract_fingerprint`, `generate_signal_id`, `atomic_emit`, `publish_update`, `check_material_change` | `signals:emitted:*`, `signals:cooldown:*`, `signals:live:*`, `signals:audit:*`, `signals:updates` |
| `src/execution_manager.py` (`ExecutionManager`) | Signal consumption, order placement, stop/target orchestration, acknowledgement publishing | `process_pending_signals`, `execute_signal`, `_acknowledge_signal`, `handle_fill` | `signals:execution:{symbol}`, `orders:pending:*`, `signals:ack:*`, `metrics:signals:acks` |
| `src/signal_distributor.py` (`SignalDistributor`) | Tiered fan-out, delayed delivery, validator enforcement | `_poll_pending`, `_fan_out`, `_publish_webhooks` | `signals:pending:{symbol}`, `distribution:*`, `metrics:signals:distributed:*` |

## Signal Flow Overview
```
analytics:{symbol}:*  ──┐
                       │ feature_reader()
                       ▼
                 ┌─────────────┐     contract_fp / emit id     ┌────────────┐
                 │ Signal Gen. │ ─────────────────────────────▶│ Dedup/Emit │─┐
                 └──────┬──────┘                                └────┬───────┘ │
                        │ veto/exposure checks                       │         │
                        ▼                                            ▼         ▼
                  signals:pending                  signals:execution          audit/updates
                        │                                            │
                        ▼                                            ▼
              ┌─────────────────┐                       ┌──────────────────┐
              │ Distributor     │                       │ ExecutionManager │
              └─────────────────┘                       └────────┬─────────┘
                                                                ack/pubsub
```

## Engine Lifecycle & Scheduling
1. `SignalGenerator.start()` validates the module is enabled and spins the main loop at `modules.signals.loop_interval_s` (default 0.5s) with random jitter.
2. Each iteration updates `health:signals:heartbeat`, then walks enabled strategies and their symbol lists.
3. Strategies are only evaluated inside configured time windows (`strategies.*.time_window`) and respect `_consecutive_errors` circuit-breaking (exponential backoff after repeated exceptions).
4. Per-symbol throttling is enforced via `min_refresh_s`; stale or schema-incomplete features trigger metrics increments (`metrics:signals:blocked:stale_features`, `metrics:signals:blocked:no_features`).

## Feature Ingestion Pipeline
- `default_feature_reader` batches Redis fetches for market ticks, bars, VPIN, OBI, GEX/DEX, dealer flows, toxicity, flow clusters, participant flow splits, and the normalized options chain. When the Alpha Vantage payload only exposes `by_contract`, the reader now hydrates the list directly so every strategy receives canonical strike/expiry/greek data without duplicating storage. Synthetic top-of-book books published by ingestion ensure OBI values stay live even for symbols without native L2 access, so strategy vetoes no longer oscillate between neutral and stale for the equity universe.
- Freshness is validated using the embedded timestamp/age; breaches feed the veto metrics.
- Toxicity payloads now contain: raw and adjusted scores, aggressor ratios, large-trade ratios, institutional score, spread-cross ratios, derived venue mix, and confidence estimation. Complementary participant heuristics are persisted at `analytics:{symbol}:institutional_flow` / `analytics:{symbol}:retail_flow`; the reader surfaces their `value` ratios while preserving the raw payload for diagnostics.
- New feature fields surfaced to strategies include `toxicity_adjusted`, `aggressor_ratio`, `large_trade_ratio`, `institutional_flow`, `retail_flow`, `gamma_pin_proximity`, `gamma_pull_dir`, dealer-flow notional/z-scores, and the fully normalized `options_chain` list.
- Flow clustering outputs are refreshed for every symbol participating in any enabled strategy, so `flow_momentum`, `flow_mean_reversion`, and participant split metrics remain populated for the 14 DTE equity universe instead of only the index complex.

## Strategy Stack
### 0DTE
- Hard vetoes (hedging elasticity, low toxicity, neutral VPIN/OBI, VIX shock, stale analytics) are enforced before scoring; each veto updates `metrics:signals:blocked_by_veto`.
- Scoring weights VPIN, OBI, gamma squeeze proximity, sweeps, dealer flows (Vanna/Charm), skew, flow clustering, hedging elasticity, and VIX regime adjustments.
- Institutional confirmation requires ≥2 distinct signals among sweep/vanna/charm/flow momentum; otherwise the trade is rejected.
- Direction is chosen via multi-factor voting (`_determine_0dte_direction`).

### 1DTE
- Focuses on end-of-day momentum, VPIN/OBI dynamics, delta exposure build, toxicity, and power-hour adjustments. Shares contract selection logic with other DTE variants.

### 14DTE
- Rebalanced weights to include Vanna/Charm pressure, flow momentum, and relaxed thresholds (`min_confidence=75`, `unusual_min=0.5`, `dex_min=3e9`).
- Institutional score from toxicity heuristics boosts confidence; retail dominance trims it.
- Directional voting blends delta exposure trends, unusual activity skew, price momentum, sweep direction, hidden orders, and institutional vs retail tallies.

### Market-On-Close (MOC)
- Evaluates imbalance size/ratio, gamma pin alignment, indicative price divergence, timing window, and liquidity guardrails.
- Selects contracts with hysteresis to avoid churn and uses dedicated audit keys.

### Contract Selection & Hysteresis
- `DTEStrategies.select_contract` and `MOCStrategy.select_contract` normalise expiry/right/strike, refine directly from the hydrated option chain provided by the feature reader, and persist last-used contracts via `signals:last_contract:*` to avoid thrashing unless roll criteria are met.

## Deduplication & Emission Guardrails
- `contract_fingerprint` canonicalises strike, expiry, right, multiplier, and exchange to prevent floating-point drift (`660` vs `660.00`).
- `generate_signal_id` hashes fingerprint + version + expiry, extending idempotency across contract life (TTL derived from DTE: 24h for 0DTE, 48h for 1DTE, ≥1 week for swing).
- `atomic_emit` Lua script enforces four gates in order: existing live lock (`signals:live:{symbol}:{fp}`), idempotency (`signals:emitted:{id}`), cooldown (`signals:cooldown:{fp}`), then queue push to both pending (distribution) and execution pipelines. Successful pushes set the live lock.
- `check_material_change` prevents thin confidence bumps; when exposure constraints are already active it reroutes updates onto `signals:updates` rather than re-emitting.
- Audit entries live under `signals:audit:{fp}` for observability.

## Exposure, Cooldown, & Veto Metrics
- Exposure caps (configurable per strategy/symbol) check live locks, open positions (`positions:by_symbol:*`), and pending orders (`orders:pending:*`). Breaches increment `metrics:signals:blocked:exposure`, publish audit entries, and emit updates.
- Additional block counters: `metrics:signals:blocked:no_features`, `metrics:signals:blocked:stale_features`, `metrics:signals:blocked_by_reason` (hash), and `metrics:signals:blocked_by_veto` (hash per veto code).
- Successful emits increment `metrics:signals:emitted`; duplicates and cooldown hits increment their respective counters.

## Execution Feedback & Live-Lock Release
- Execution manager consumes `signals:execution:{symbol}` and places IBKR orders. On fill/cancel/reject/timeout it calls `_acknowledge_signal`, which:
  - Deletes `signals:live:{symbol}:{fp}` so the generator can re-consider the contract.
  - Stores `signals:ack:{signal_id}` with `{status, filled, avg_price, timestamp}` for 24h.
  - Publishes the same payload on `signals:acknowledged` and increments `metrics:signals:acks`.
- Consumers (risk, dashboards) can monitor ack stream latency relative to `metrics:signals:emitted` to detect stuck orders.

## Distribution & Update Channels
- Distribution tier reads from `signals:pending:{symbol}` and handles premium/basic/free queues with programmable delays.
- Strategy rejections or exposure suppressions broadcast via `signals:updates` for observability (e.g., UI display of rejected reasons).

## Redis Key Reference
| Key | Purpose |
|-----|---------|
| `health:signals:heartbeat` | Heartbeat with timestamp + dry_run flag |
| `signals:pending:{symbol}` | Queue consumed by distribution tier |
| `signals:execution:{symbol}` | Queue consumed by execution manager |
| `signals:live:{symbol}:{contract_fp}` | Live-lock preventing duplicate exposure |
| `signals:emitted:{signal_id}` | Idempotency guard |
| `signals:cooldown:{contract_fp}` | Contract-level cooldown clock |
| `signals:audit:{contract_fp}` | Rolling audit trail (last 50 events) |
| `signals:updates` | Pub/sub channel for blocked/thin-update notifications |
| `signals:ack:{signal_id}` | Execution acknowledgement payload |
| `metrics:signals:*` | Counter family for loop errors, duplicates, veto reasons, exposures, acknowledgements |

## Observability Checklist
- Track block/veto counters in Grafana to highlight strategy gating and stale feature spikes.
- Alert when `signals:acknowledged` latency exceeds 10 seconds relative to `metrics:signals:emitted` or when live locks persist longer than the contract TTL.
- Monitor `signals:audit:*` sample logs during debugging; entries contain action, reason, confidence, and TTL metadata.

## Testing & Regression
- Unit regression `tests/test_signal_governance.py` exercises exposure-cap gating and metric increments with a stub Redis client.
- Existing DTE strategy logic can be validated via historical replays by feeding cached `analytics:*` snapshots into the feature reader and capturing emitted decisions.
- Integration smoke tests should verify the full loop: feature fetch → signal emit → execution ack → live lock cleared.

## Reference Surface
- Core engine: `src/signal_generator.py`
- Strategies: `src/dte_strategies.py`, `src/moc_strategy.py`
- Deduplication: `src/signal_deduplication.py`
- Execution acknowledgements: `src/execution_manager.py`
- Distribution tier: `src/signal_distributor.py`
- Config knobs: `config/config.yaml` (`modules.signals`, `modules.analytics.toxicity_heuristics`)
