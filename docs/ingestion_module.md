# Ingestion Module Deep Dive

## Mission Statement
Maintain low-latency, loss-aware ingestion of market structure, options analytics, and sentiment data while enforcing schema consistency across Redis so downstream analytics, signals, and execution modules can rely on fresh, normalized payloads.

## Component Matrix
| Module | Responsibilities | Core Loops / Handlers | Redis Touchpoints |
|--------|------------------|-----------------------|-------------------|
| `src/ibkr_ingestion.py` (`IBKRIngestion`) | Stream Level 2 depth, trades, and 5-second bars; aggregate 1-minute candles; monitor freshness and connection state | `start`, `_connect_with_retry`, `_on_depth_update_async`, `_on_ticker_update_async`, `_metrics_loop`, `_freshness_check_loop`, `_status_logger_loop` | `market:{symbol}:book`, `market:{symbol}:ticker`, `market:{symbol}:bars:{tf}`, `market:{symbol}:trades`, `monitoring:ibkr:metrics`, `heartbeat:ibkr_ingestion` |
| `src/ibkr_processor.py` (`IBKRDataProcessor`) | Normalize DOM levels, compute aggregated nbbo, prepare trade payloads, record bar buffers | `update_depth_book`, `compute_aggregated_tob`, `prepare_trade_storage`, `build_quote_ticker`, `record_bar` | In-memory caches feeding `IBKRIngestion` Redis writes |
| `src/av_ingestion.py` (`AlphaVantageIngestion`) | Poll options chains, sentiment feed, and technical indicators with rolling rate limiting and failure circuit breaking | `_update_loop`, `_fetch_with_retry`, `fetch_options_chain`, `fetch_sentiment`, `fetch_technicals`, `_metrics_loop` | `options:{symbol}:*`, `sentiment:{symbol}`, `technicals:{symbol}:{indicator}`, `monitoring:alpha_vantage:metrics`, `heartbeat:alpha_vantage` |
| `src/av_options.py` (`OptionsProcessor`) | Normalize Alpha Vantage option chains, persist calls/puts/greeks, publish monitoring payloads, trigger callbacks | `fetch_options_chain`, `_store_options_data`, `_normalize_contract`, `_notify_chain` | `options:{symbol}:calls`, `options:{symbol}:puts`, `options:{symbol}:chain`, `options:{symbol}:{contractId}:greeks`, `monitoring:options:{symbol}` |
| `src/av_sentiment.py` (`SentimentProcessor`) | Persist sentiment feeds with article metadata, compute aggregates, store technical indicators | `fetch_sentiment`, `_store_sentiment_data`, `fetch_technicals`, `_fetch_single_indicator`, `_store_technical_data` | `sentiment:{symbol}`, `monitoring:sentiment:{symbol}`, `technicals:{symbol}:{indicator}` |

## Data Flow Overview
```
IBKR Gateway ──▶ IBKRIngestion ──┐
                                 │                           ┌─────────────┐
Alpha Vantage API ─▶ AVIngestion ─┼──▶ Normalizers ──▶ Redis │ market/*    │
                                 │                           │ options/*   │
Sentiment/Technicals ────────────┘                           │ analytics/* │
                                                             └─────────────┘
```
Downstream consumers (analytics engine, signal generator, execution manager, distribution workers) subscribe strictly through Redis namespaces maintained here.

## IBKR Ingestion
### Startup Sequence
1. Instantiate `IBKRIngestion(config, redis_conn)`; preload symbol lists, TTLs, rate limits, market calendars, and helper processor.
2. Call `start()`:
   - Run `_connect_with_retry()` with exponential backoff; set `ibkr:connected=1` once logged in.
   - Register event handlers (`pendingTickersEvent`, `barUpdateEvent`, `errorEvent`, `disconnectedEvent`).
   - Launch subscriptions via `_setup_subscriptions()` splitting Level 2 vs standard symbols.
   - Spawn background loops: `_metrics_loop`, `_freshness_check_loop`, `_status_logger_loop`.
3. Main loop idles while `running` and `connected` flags remain true; graceful shutdown triggers `_cleanup()`.

### Event Handling Map
| Event Source | Handler | Notes |
|--------------|---------|-------|
| `pendingTickersEvent` | `_on_ticker_update_async` | Routes depth tickers to `_on_depth_update_async`; processes trades (venue tagging) and TOB updates |
| `barUpdateEvent` | `_on_bar_update_async` | Converts 5-second bars to raw storage + minute aggregates |
| `dom` callbacks (per exchange) | `_on_depth_update_async` | Normalizes DOM via `IBKRDataProcessor`, stores per-exchange book + aggregated NBBO |
| Disconnection | `_on_disconnect` → `_reconnect` | Marks `ibkr:connected=0`, attempts resubscription if `running` |
| Errors (309/2152/etc.) | `_on_error` | Handles entitlement/L2 limits, toggles fallback to TOB |

### Redis Writes
| Key | Payload | Source Method |
|-----|---------|---------------|
| `market:{symbol}:{exchange}:book` | Venue-specific depth snapshot | `_on_depth_update_async` |
| `market:{symbol}:ticker` | Aggregated bid/ask/last/mid/spread | `_on_depth_update_async`, `_on_tob_update_async`, `_update_trade_data_redis` |
| `market:{symbol}:trades` | Rolling list (≤1000) of trades with venue/time | `_update_trade_data_redis` |
| `market:{symbol}:bars:5s` | 5-second raw bars | `_store_raw_bar` |
| `market:{symbol}:bars:1min` | Aggregated minute candles | `_flush_minute_bar` |
| `monitoring:ibkr:metrics` | Connection + throughput metrics | `_metrics_loop` |
| `heartbeat:ibkr_ingestion` | Timestamp + connection metadata | `_metrics_loop` |
| `monitoring:data:stale` | Symbols exceeding freshness threshold | `_check_data_freshness` |

### Failure & Recovery
- **Connection loss**: `_on_disconnect` toggles flags and launches `_reconnect`; reconnect success triggers `_setup_subscriptions()` to restore feeds.
- **Depth entitlement issues**: `_request_exchange_depth` catches code 2152 and adds symbol/exchange to `l2_fallback_exchanges`, forcing TOB-only flow.
- **Depth cap (309)**: Logs warning and falls back to standard subscriptions when IBKR limit (3) is reached.
- **Shutdown**: `_cleanup` cancels background tasks, flushes minute bars, cancels market data, disconnects IB, and clears caches before setting `ibkr:connected=0`.

## IBKR Data Processor
- **Order Book Normalization**: `update_depth_book` converts DOM objects into ordered lists with venue codes mapped via `VENUE_MAP`; cached per symbol/exchange.
- **Aggregated TOB**: `compute_aggregated_tob` scans cached books to produce NBBO with spread/spread_bps, enforcing monotonic timestamps via `last_tob_ts`.
- **Trade Payloads**: `prepare_trade_storage` maintains last trade price, builds `market:{symbol}:last` + ticker payloads with mid/spread when quotes available.
- **Quote Sanitization**: `build_quote_ticker` filters NaN/inf values, ensuring downstream consumers never see invalid floats.
- **Bar Buffers**: `record_bar` and `bars_buffer` enable deterministic tests and future replay tooling.

## Alpha Vantage Ingestion
### Core Flow
1. `AlphaVantageIngestion(config, redis_conn)` validates API key, composes symbol roster (Level 2 + standard), builds rolling `RollingRateLimiter`, and instantiates `OptionsProcessor` + `SentimentProcessor`.
2. `start()` spawns `_update_loop` and `_metrics_loop`.
3. `_update_loop` (within aiohttp session):
   - For each symbol, evaluate jittered intervals for options, sentiment, and technicals.
   - Skip types under circuit break (`_is_suspended`).
   - Submit `_run_fetch` tasks; each wraps target fetcher with `_fetch_with_retry` for exponential backoff and jitter.
4. `_metrics_loop` emits rate limiter stats, failure counts, suspended feeds, and heartbeat entries every 10 seconds.

### Options Processor Highlights
- `fetch_options_chain` executes the REST call after acquiring tokens, handles soft limits (`Note`/`Information`) with sleep-based backoff, and forwards to `_store_options_data`.
- `_store_options_data`:
  - Writes raw call/put arrays.
  - Builds normalized chain payload (`by_contract`, expiration summary, metrics).
  - Persists per-contract greeks under `options:{symbol}:{contractId}:greeks` when available.
  - Publishes monitoring summary & fires registered callbacks (e.g., pub/sub notifications).
- Empty chains clear cached keys and publish status payloads.

### Sentiment Processor Highlights
- `fetch_sentiment` skips configured ETFs (default: SPY/QQQ/IWM/VXX), handles soft limits, and triggers `_store_sentiment_data`.
- `_store_sentiment_data` retains the full article feed with authors, summary, source, sentiment scores, and per-ticker metrics. Calculates avg sentiment/relevance, sentiment distribution, article age stats, and monitoring payload.
- `fetch_technicals` spawns parallel indicator requests (RSI, MACD, BBANDS, ATR); `_store_technical_data` normalizes both single and multi-value indicators and logs key readings.

### Rate Limiting & Circuit Breaking
- `RollingRateLimiter.acquire` implements a token-bucket with jittered sleeps; stats feed metrics.
- `_record_failure` increments counters per symbol/data-type; once threshold reached, applies exponential backoff up to `failure_backoff_max` and logs suspension notices.
- `_metrics_loop` exposes suspended feeds list via `monitoring:alpha_vantage:metrics` for observability.

## Configuration Inputs
| Config Path | Purpose | Default Highlights |
|-------------|---------|--------------------|
| `config['ibkr']` | Host/port/clientId, reconnect strategy, L2 depth rows | Host `127.0.0.1`, port `7497`, max reconnect attempts `10`, base delay `2` |
| `config['symbols']` | Level 2 vs standard symbol lists shared by IBKR + Alpha Vantage | Level 2 default: SPY/QQQ/IWM |
| `config['modules']['data_ingestion']` | TTL policies, data freshness thresholds, monitoring timeouts | TTLs per surface (`market_data`, `order_book`, `bars`, `greeks`) |
| `config['alpha_vantage']` | API key, base URL, rate limit, safety buffer, update intervals, failure backoff | Calls/min 600, safety buffer 10, options interval 10s, sentiment 300s, technicals 60s |
| `config['market']` | After-hours staleness behavior | `check_staleness_after_hours` toggle |

## Metrics & Monitoring
- **Heartbeats**: `heartbeat:ibkr_ingestion`, `heartbeat:alpha_vantage` – consumed by ops dashboards.
- **Performance Metrics**: `monitoring:ibkr:metrics` (depth/trade/bar counts, max lag, reconnect attempts); `monitoring:alpha_vantage:metrics` (token stats, suspended feeds, failure totals).
- **Staleness Watch**: `monitoring:data:stale` hash enumerates symbols exceeding freshness thresholds.
- **Options/Sentiment Monitoring**: `monitoring:options:{symbol}` tracks chain size + greeks coverage; `monitoring:sentiment:{symbol}` captures article counts, avg sentiment, source distribution.

## Operational Playbook
| Scenario | Action |
|----------|--------|
| **Restart IBKR stream** | Kill worker gracefully (`stop()`), confirm `ibkr:connected=0`, rerun `python -m src.ibkr_ingestion`, watch heartbeat refresh |
| **Rate-limit spike** | Inspect `monitoring:alpha_vantage:metrics` tokens; consider increasing safety buffer or staggering intervals; check for suspended feeds list |
| **Chain missing greeks** | Review `monitoring:options:{symbol}` `missing_greeks` metric; escalate to vendor if persistent |
| **Stale market data** | Examine `monitoring:data:stale`; verify IBKR connection + entitlement; consider resubscribing specific symbol |
| **Graceful shutdown** | Ensure `_cancel_background_tasks` completes, `_flush_all_minute_bars` empties buffers, and market data subscriptions are cancelled to avoid IBKR rate penalties |

## Testing Guidance
- `tests/unit/test_ibkr_processor.py`: Validate depth normalization, nbbo aggregation, and ticker payloads.
- `tests/integration/` (planned): Replay recorded websockets to ensure Redis keys update with expected TTLs.
- Suggested new tests: simulate rate-limit responses for Alpha Vantage, verify circuit breaker suspends/resumes correctly, confirm minute bar aggregation flushes at boundary transitions.

## Source Reference
- IBKR Ingestion: `src/ibkr_ingestion.py`
- IBKR Normalizer: `src/ibkr_processor.py`
- Alpha Vantage Coordinator: `src/av_ingestion.py`
- Options Processor: `src/av_options.py`
- Sentiment & Technicals Processor: `src/av_sentiment.py`

