"""End-to-end validation for dealer-flow analytics and signal feature wiring."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytest

from pathlib import Path
import sys
import types

if "holidays" not in sys.modules:
    class _StubHolidays(dict):
        def __contains__(self, item) -> bool:  # noqa: D401
            """Pretend no additional holiday exclusions."""

            return False

    def _us_factory(*_args, **_kwargs):  # noqa: D401
        """Return a stub holiday calendar."""

        return _StubHolidays()

    sys.modules["holidays"] = types.SimpleNamespace(US=_us_factory)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.analytics_engine import AnalyticsEngine
from src.dealer_flow_calculator import DealerFlowCalculator
from src.flow_clustering import FlowClusterModel
from src.signal_generator import default_feature_reader
from src.volatility_metrics import VolatilityMetrics
import src.redis_keys as rkeys


class AsyncFakeRedis:
    """Minimal async Redis stub with pipeline support for integration tests."""

    def __init__(self) -> None:
        self._kv: Dict[str, Any] = {}
        self._ttl: Dict[str, int] = {}
        self._lists: Dict[str, List[Any]] = {}
        self._hashes: Dict[str, Dict[str, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        return self._kv.get(key)

    async def set(self, key: str, value: Any) -> bool:
        self._kv[key] = self._coerce(value)
        return True

    async def setex(self, key: str, ttl: int, value: Any) -> bool:
        self._kv[key] = self._coerce(value)
        self._ttl[key] = ttl
        return True

    async def ttl(self, key: str) -> int:
        return self._ttl.get(key, -1)

    async def lpush(self, key: str, value: Any) -> int:
        bucket = self._lists.setdefault(key, [])
        bucket.insert(0, self._coerce(value))
        return len(bucket)

    async def ltrim(self, key: str, start: int, stop: int) -> bool:
        bucket = self._lists.get(key, [])
        length = len(bucket)
        start_idx = start if start >= 0 else max(length + start, 0)
        stop_idx = stop if stop >= 0 else length + stop
        stop_idx = min(stop_idx, length - 1)
        if stop_idx < start_idx or length == 0:
            self._lists[key] = []
        else:
            self._lists[key] = bucket[start_idx:stop_idx + 1]
        return True

    async def lrange(self, key: str, start: int, stop: int) -> List[Any]:
        bucket = self._lists.get(key, [])
        length = len(bucket)
        if length == 0:
            return []
        start_idx = start if start >= 0 else max(length + start, 0)
        stop_idx = stop if stop >= 0 else length + stop
        stop_idx = min(stop_idx, length - 1)
        if stop_idx < start_idx:
            return []
        return bucket[start_idx:stop_idx + 1]

    async def expire(self, key: str, ttl: int) -> bool:
        self._ttl[key] = ttl
        return True

    async def hgetall(self, key: str) -> Dict[str, Any]:
        return dict(self._hashes.get(key, {}))

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:  # pragma: no cover - helper symmetry
        self._kv.clear()
        self._ttl.clear()
        self._lists.clear()
        self._hashes.clear()

    def pipeline(self, transaction: bool = False) -> "AsyncFakePipeline":  # noqa: ARG002
        return AsyncFakePipeline(self)

    @staticmethod
    def _coerce(value: Any) -> Any:
        if isinstance(value, (str, bytes)):
            return value
        return json.dumps(value)


class PipelineCommand:
    """Allow optional awaiting of queued pipeline operations."""

    def __await__(self):  # pragma: no cover - trivial awaitable
        async def _noop() -> None:
            return None

        return _noop().__await__()


class AsyncFakePipeline:
    """Collects operations and replays them sequentially on execution."""

    def __init__(self, redis: AsyncFakeRedis) -> None:
        self._redis = redis
        self._operations: List[Tuple[Any, Tuple[Any, ...], Dict[str, Any]]] = []

    async def __aenter__(self) -> "AsyncFakePipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, ANN201
        self._operations.clear()

    def _enqueue(self, func, *args, **kwargs) -> PipelineCommand:
        self._operations.append((func, args, kwargs))
        return PipelineCommand()

    def get(self, key: str) -> PipelineCommand:
        return self._enqueue(self._redis.get, key)

    def setex(self, key: str, ttl: int, value: Any) -> PipelineCommand:
        return self._enqueue(self._redis.setex, key, ttl, value)

    def lpush(self, key: str, value: Any) -> PipelineCommand:
        return self._enqueue(self._redis.lpush, key, value)

    def ltrim(self, key: str, start: int, stop: int) -> PipelineCommand:
        return self._enqueue(self._redis.ltrim, key, start, stop)

    def expire(self, key: str, ttl: int) -> PipelineCommand:
        return self._enqueue(self._redis.expire, key, ttl)

    def lrange(self, key: str, start: int, stop: int) -> PipelineCommand:
        return self._enqueue(self._redis.lrange, key, start, stop)

    async def execute(self) -> List[Any]:
        results: List[Any] = []
        for func, args, kwargs in self._operations:
            result = await func(*args, **kwargs)
            results.append(result)
        self._operations.clear()
        return results


def _option_contracts(today_eastern: datetime) -> List[Dict[str, Any]]:
    expiry_today = today_eastern.strftime("%Y-%m-%d")
    expiry_next = (today_eastern + timedelta(days=2)).strftime("%Y-%m-%d")
    return [
        {
            "type": "call",
            "strike": 425.0,
            "expiration": expiry_today,
            "implied_volatility": 0.24,
            "delta": 0.24,
            "gamma": 0.018,
            "vega": 0.11,
            "open_interest": 650,
            "volume": 220,
        },
        {
            "type": "put",
            "strike": 415.0,
            "expiration": expiry_today,
            "implied_volatility": 0.29,
            "delta": -0.26,
            "gamma": 0.017,
            "vega": 0.12,
            "open_interest": 610,
            "volume": 240,
        },
        {
            "type": "call",
            "strike": 430.0,
            "expiration": expiry_next,
            "implied_volatility": 0.23,
            "delta": 0.35,
            "gamma": 0.015,
            "vega": 0.09,
            "open_interest": 420,
            "volume": 180,
        },
    ]


def _build_trade_samples(price: float, count: int) -> List[str]:
    trades: List[str] = []
    base_ts = int(time.time()) - count
    for idx in range(count):
        trade = {
            "price": price + math.sin(idx / 5) * 0.5,
            "size": 150 + (idx % 10) * 25,
            "timestamp": base_ts + idx,
            "side": "buy" if idx % 3 else "sell",
            "sweep": 1 if idx % 7 == 0 else 0,
        }
        trades.append(json.dumps(trade))
    return trades


@pytest.mark.asyncio
async def test_dealer_flow_pipeline_populates_redis(monkeypatch):
    """Dealer flow, clustering, and volatility metrics should persist to Redis and feed signals."""

    redis = AsyncFakeRedis()
    symbol = "SPY"

    today_eastern = datetime.now().astimezone()
    option_chain = {"contracts": _option_contracts(today_eastern)}

    # Seed base analytics and market context consumed by the aggregator and feature reader.
    await redis.setex(rkeys.options_chain_key(symbol), 300, option_chain)
    await redis.setex(
        rkeys.market_ticker_key(symbol),
        60,
        {
            "last": 420.0,
            "timestamp": int(time.time() * 1000),
            "volume": 1_750_000,
        },
    )
    await redis.setex(rkeys.analytics_vpin_key(symbol), 60, {"value": 0.48})
    await redis.setex(rkeys.analytics_gex_key(symbol), 60, {"total_gex": 1.9e9})
    await redis.setex(
        rkeys.analytics_dex_key(symbol),
        60,
        {"total_dex": 2.1e8, "call_dex": 2.9e8, "put_dex": -7.9e7},
    )

    # Populate trade history for flow clustering (append to Redis list semantics).
    trade_key = rkeys.market_trades_key(symbol)
    redis._lists[trade_key] = _build_trade_samples(420.0, 60)

    config = {
        "symbols": {
            "standard": [symbol],
            "level2": [],
        },
        "modules": {
            "analytics": {
                "enabled": True,
                "store_ttls": {
                    "analytics": 120,
                    "portfolio": 180,
                    "sector": 180,
                    "correlation": 300,
                    "heartbeat": 15,
                },
                "update_intervals": {
                    "dealer_flows": 5,
                    "flow_clustering": 5,
                    "volatility": 5,
                    "portfolio": 5,
                    "sectors": 5,
                },
                "sector_map": {symbol: "INDEX"},
                "dealer_flow": {
                    "history_window": 32,
                    "max_expiry_days": 45,
                    "min_samples": 5,
                },
                "flow_clustering": {
                    "window_trades": 200,
                    "min_trades": 40,
                    "random_state": 42,
                },
                "volatility": {
                    "history_window": 64,
                    "vix1d_thresholds": {
                        "shock": 35,
                        "elevated": 25,
                        "benign": 18,
                    },
                    "change_thresholds": {
                        "shock": 5.0,
                        "elevated": 2.0,
                    },
                },
            }
        },
        "parameter_discovery": {
            "toxicity_detection": {"vpin_thresholds": {"toxic": 0.7}}
        },
    }

    engine = AnalyticsEngine(config, redis)
    engine.dealer_flow_calculator = DealerFlowCalculator(config, redis)
    engine.flow_cluster_model = FlowClusterModel(config, redis)

    async def _fetch_vix1d_stub(self) -> float:  # noqa: D401
        """Return a deterministic VIX1D sample."""

        return 32.0

    monkeypatch.setattr(VolatilityMetrics, "_fetch_vix1d", _fetch_vix1d_stub, raising=False)
    engine.volatility_metrics = VolatilityMetrics(config, redis)

    # Execute the calculators through the engine helpers.
    assert await engine._run_calculation("dealer_flows") is True
    assert await engine._run_calculation("flow_clustering") is True
    assert await engine._run_calculation("volatility") is True
    assert await engine._run_calculation("portfolio") is True
    assert await engine._run_calculation("sectors") is True

    # Dealer-flow metrics should now be persisted under their canonical keys.
    vanna_payload = json.loads(await redis.get(rkeys.analytics_vanna_key(symbol)))
    charm_payload = json.loads(await redis.get(rkeys.analytics_charm_key(symbol)))
    hedging_payload = json.loads(await redis.get(rkeys.analytics_hedging_impact_key(symbol)))

    assert abs(vanna_payload["total_vanna_notional_per_pct_vol"]) > 0
    assert abs(charm_payload["total_charm_notional_per_day"]) > 0
    assert hedging_payload["notional_per_pct_move"] != 0

    flow_clusters_payload = json.loads(await redis.get(rkeys.analytics_flow_clusters_key(symbol)))
    assert flow_clusters_payload["strategy_distribution"]["momentum"] >= 0
    assert flow_clusters_payload["participant_distribution"]["institutional"] >= 0

    vix1d_payload = json.loads(await redis.get(rkeys.analytics_vix1d_key()))
    assert vix1d_payload["value"] == pytest.approx(32.0, rel=1e-6)
    assert vix1d_payload["regime"] == "ELEVATED"

    portfolio_summary = json.loads(await redis.get(rkeys.analytics_portfolio_summary_key()))
    assert portfolio_summary["total_vanna_notional"] > 0
    assert portfolio_summary["vix1d"]["regime"] == "ELEVATED"

    sector_summary = json.loads(await redis.get(rkeys.analytics_sector_key("INDEX")))
    assert sector_summary["total_vanna_notional"] > 0
    assert sector_summary["symbol_count"] == 1

    # Feature reader must expose the enriched analytics payloads to strategies.
    features = await default_feature_reader(redis, symbol)

    assert features["vanna_notional"] == pytest.approx(vanna_payload["total_vanna_notional_per_pct_vol"], rel=1e-3)
    assert features["charm_notional"] == pytest.approx(charm_payload["total_charm_notional_per_day"], rel=1e-3)
    assert features["hedging_notional_per_pct"] == pytest.approx(hedging_payload["notional_per_pct_move"], rel=1e-3)
    assert features["flow_momentum"] == pytest.approx(flow_clusters_payload["strategy_distribution"]["momentum"], rel=1e-3)
    assert features["vix1d_regime"] == "ELEVATED"
