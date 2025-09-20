# Execution Router Rewrite – Detailed Plan (Paper Trading Dev Mode)

## Objective
Replace the existing `ExecutionManager`/`PositionManager` complex with a thin, event-driven execution router that treats Interactive Brokers (IBKR) as the single source of truth for orders, positions, and account telemetry while preserving the platform's risk gating, signal acknowledgement, and downstream distribution contracts.

## Immediate Next Actions (Kickoff Checklist)
1. **Branch/Checkpoint**: Create a working branch for router development; snapshot current paper-trading config.
2. **Bug Fix Tickets**: Open tasks for trailing-stop hotfix, ticker cache namespacing, `positions:count` sync, and divergence logging (Phase 0 items).
3. **Paper Account Reset Script**: Automate state reset (cancel orders, close positions) to ensure deterministic integration tests.
4. **Event Schema Draft**: Draft JSON schema for lifecycle events (`OrderLifecycleEvent`, `ExecutionFill`, `PositionSnapshot`, `PnLSnapshot`).
5. **Design Review Schedule**: Book architecture review for end of Phase 1; circulate agenda including IBKR API usage plan and rollback strategy.
6. **Telemetry Baseline**: Capture current execution latency metrics to compare against router performance targets.

## Guiding Principles
- **IBKR-first**: Leverage native IBKR APIs for bracket orders, adjustable stops, open-order tracking, executions, positions, account summaries, and PnL streams.
- **Stateless core**: Router owns no long-lived execution state; Redis is used exclusively for ingress (signal queue) and egress (notification bus) with payloads regenerable from IB events.
- **Deterministic orchestration**: Signals flow through risk/sizing → normalized `OrderRequest` → IBKR order submission, with all subsequent lifecycle updates sourced from IBKR callbacks.
- **Auditability**: Every placed order, modification, and event is logged with correlation IDs, preserving replayability and compliance requirements.
- **Progressive migration**: Maintain production-equivalent reliability in paper trading via phased deployment, dual-run validation, and explicit rollback toggles.

## Phase Breakdown

### Phase 0 – Stabilize Legacy Engine (0.5 week)
| Task | Owner | Deliverable | Acceptance |
|------|-------|-------------|------------|
| Patch trailing stop update bug (missing `action`/`qty`) | Platform | Hotfix merged, regression test | Paper fills demonstrate stop updates succeed |
| Fix symbol-level ticker cache leakage | Platform | Cached by contract fingerprint | Multi-option subscription validated |
| Sync `positions:count` with IB position events | Platform | Counter increments only on new IB positions | Risk gate prevents over-allocation |
| Instrument divergence logging | Platform | Structured logs for Redis vs IB snapshots | No untracked drift during soak |

### Phase 1 – Requirements & Contracts (1 week)
- Catalogue business logic retained from legacy engine (Kelly sizing, correlation checks, lifecycle payload schema, ack semantics).
- Inventory Redis inputs/outputs; tag each as **core**, **derived**, or **deprecated**.
- Define canonical DTOs:
  - `OrderRequest`, `OrderLifecycleEvent`, `ExecutionFill`, `PositionSnapshot`, `PnLSnapshot`.
- Update `implementation_plan.md` scoreboard with new execution rewrite phase.

### Phase 2 – Architecture Blueprint (1 week overlap with Phase 1)
- Sequence diagram of signal → router → IBKR events → downstream modules.
- Module boundaries:
  - `sizing.risk_gateway` (existing components reused)
  - `execution.router` (new)
  - `execution.events` (IB callbacks → DTOs)
  - `messaging.dispatch` (Redis publish/ack)
- Integration points with: RiskManager, KellySizer, SignalDistributor, Analytics.
- Deliverable: design review doc + approval.

### Phase 3 – Execution Router Implementation (2–3 weeks)
1. **Scaffolding**
   - Create `src/execution_router/` package with `__init__.py`, router interface, dependency injection (IB client, risk gateway, publisher).
   - Build synchronous + async wrappers for ib_insync to mock during tests.
2. **Contract Builder**
   - Implement contract normalization (stock/option) referencing existing `create_ib_contract` logic but returning pure data objects.
   - Unit tests covering OCC parsing, expiry normalization, error handling.
3. **Order Submission Layer**
   - `build_bracket_orders` using `ib.bracketOrder` per [IBKR docs](https://interactivebrokers.github.io/tws-api/bracket_order.html).
   - Support adjustable stops: trailing %, step, triggered stop-limit (https://interactivebrokers.github.io/tws-api/adjustable_stops.html).
   - OCA coordination, order modification & cancellation paths (https://interactivebrokers.github.io/tws-api/modifying_orders.html, https://interactivebrokers.github.io/tws-api/cancel_order.html).
   - Comprehensive unit tests with mocked IB client verifying order wiring and error escalation.
4. **Event Stream Bridge**
   - Subscribe to `orderStatusEvent`, `execDetailsEvent`, `commissionReport`, `positionEvent`, `pnlEvent`, `accountSummaryEvent`, `errorEvent`.
   - Map each to normalized DTOs and publish to Redis topics (e.g. `signals:distribution:pending`, `positions:stream`, `pnl:stream`).
   - Guarantee idempotence: replays on reconnect (via `reqOpenOrders`, `reqExecutions`, `reqPositions`, `reqAccountSummary`, `reqPnL`).
5. **Risk & Sizing Integration**
   - Wrap existing KellySizer/RiskManager to produce `OrderRequest` (quantity, confidence, stop/target config) before router submission.
   - Expose instrumentation for gating decisions.
6. **Testing**
   - Unit: ~90% coverage across contract builder, routing, event transforms.
   - Integration: paper account scenarios (full fill, partial, stop-out, manual cancel, reconnect, bracket cascade); automated script resets account to baseline after each scenario.
   - Load: simulate burst of fills to ensure router handles event storms without lag.

### Phase 4 – Dual-Run & Migration (1–2 weeks)
- Feature flag enabling router signal consumption alongside legacy manager.
- Mirror signals: old path executes paper orders; router runs dry, emitting diagnostic events for comparison.
- Establish parity dashboard (latency, fill prices, lifecycle events, ack timings).
- Once parity thresholds met, flip to router for actual order placement with legacy manager in monitoring-only mode; maintain emergency rollback.

### Phase 5 – Decommission Legacy Manager (1 week)
- Migrate consumers off legacy Redis schemas; provide translation adapters during transition.
- Remove or archive `ExecutionManager`, `PositionManager`, and stop-engine integrations not required.
- Update documentation (`docs/signal_engine_module.md`, `README.md`, new router docs).
- Clean up tests/CI to focus on router.

### Phase 6 – Hardening & Enhancements (ongoing)
- Alerting on event anomalies (timeouts, error codes, disconnects).
- Optional IB Algos support (https://interactivebrokers.github.io/tws-api/ibalgos.html) for advanced routing.
- Evaluate move from Redis to streaming platform (Kafka/ZeroMQ) for downstream if throughput demands increase.
- Continuous monitoring: latency histograms, success rate dashboards, auto-recovery on disconnects.

## Dependencies & Tooling
- **IBKR Paper Account** with reproducible state reset script.
- **ib_insync** mocking harness for unit/integration tests.
- **Logging/Observability** stack updates (structured event logs, Grafana dashboards).
- **Secrets Management** for IB credentials (align with security policy).
- **CI/CD** enhancements to run router unit tests and targeted integration checks.

## Risk Register
| Risk | Mitigation |
|------|------------|
| IBKR rate limits during replay | Throttle reconnect flows; stagger requests; monitor API usage |
| Event sequencing inconsistencies | Implement deterministic ordering rules, store event offsets, add reconciliation job |
| Downstream contract drift | Provide adapters and advance warning; version event payloads |
| Paper vs live behaviour deltas | Maintain documented differences; add live-sim validation checklist before production go-live |

## Success Criteria
- **Functional**: Router handles all execution scenarios currently supported, including trailing stops, partial fills, manual adjustments.
- **Operational**: No manual reconciliation required; restarts rebuild state solely from IBKR streams; Redis contains only ephemeral queues/notifications.
- **Performance**: Signal-to-order latency < 150 ms average (paper), event processing lag < 50 ms under load.
- **Observability**: Real-time dashboards for orders, fills, PnL; automated alerts for anomalies.
- **Documentation**: Updated runbooks, architecture diagram, and developer onboarding materials.
