from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
import types

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

if 'holidays' not in sys.modules:  # Provide stub to satisfy option utilities
    class _StubHolidays(dict):
        def __contains__(self, item) -> bool:  # noqa: D401
            return False

    def _us_factory(*_args, **_kwargs):
        return _StubHolidays()

    sys.modules['holidays'] = types.SimpleNamespace(US=_us_factory)

from backfill.jobs import DealerFlowBackfillJob, FlowClusterBackfillJob, VixBackfillJob
from backfill.models import DealerFlowSnapshot, FlowSlice, VixSnapshot
from backfill.providers import (
    InMemoryDealerFlowProvider,
    InMemoryFlowSliceProvider,
    InMemoryVixProvider,
)
from backfill.runner import BackfillRunner
from backfill.state import InMemoryCheckpointStore
from dealer_flow_calculator import DealerFlowCalculator
from flow_clustering import FlowClusterModel
from volatility_metrics import VolatilityMetrics

from tests.test_dealer_flow_pipeline import AsyncFakeRedis  # noqa: E402,F401

import redis_keys as rkeys


@pytest.mark.asyncio
async def test_backfill_pipeline_generates_metrics():
    redis = AsyncFakeRedis()

    with open('config/config.yaml', 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    dealer_calc = DealerFlowCalculator(config, redis)
    flow_model = FlowClusterModel(config, redis)
    vol_metrics = VolatilityMetrics(config, redis)

    symbol = 'SPY'
    start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    dealer_snapshots = [
        _build_dealer_snapshot(symbol, start_ts + timedelta(minutes=idx))
        for idx in range(3)
    ]

    flow_slices = [
        FlowSlice(
            symbol=symbol,
            timestamp=start_ts + timedelta(minutes=idx),
            trades=_build_trades(start_ts + timedelta(minutes=idx)),
        )
        for idx in range(3)
    ]

    vix_snapshots = [
        VixSnapshot(
            timestamp=start_ts + timedelta(minutes=idx),
            value=14.0 + idx,
            source='historical',
        )
        for idx in range(3)
    ]

    runner = BackfillRunner([
        DealerFlowBackfillJob(
            redis_conn=redis,
            calculator=dealer_calc,
            provider=InMemoryDealerFlowProvider(dealer_snapshots),
        ),
        FlowClusterBackfillJob(
            redis_conn=redis,
            model=flow_model,
            provider=InMemoryFlowSliceProvider(flow_slices),
        ),
        VixBackfillJob(
            redis_conn=redis,
            metrics=vol_metrics,
            provider=InMemoryVixProvider(vix_snapshots),
        ),
    ])

    results = await runner.run([symbol])

    assert len(results) == 3
    for result in results:
        assert result.snapshots_processed >= 3
        assert result.metrics_written >= 1
        assert not result.errors

    vanna_raw = await redis.get(rkeys.analytics_vanna_key(symbol))
    assert vanna_raw is not None
    vanna_payload = json.loads(vanna_raw)
    assert vanna_payload.get('samples', 0) > 0

    clusters_raw = await redis.get(rkeys.analytics_flow_clusters_key(symbol))
    assert clusters_raw is not None
    clusters_payload = json.loads(clusters_raw)
    assert clusters_payload.get('samples', 0) > 0

    vix_raw = await redis.get(rkeys.analytics_vix1d_key())
    assert vix_raw is not None
    vix_payload = json.loads(vix_raw)
    assert pytest.approx(vix_payload['value'], rel=1e-6) == vix_snapshots[-1].value
    assert vix_payload.get('source') == 'historical'


@pytest.mark.asyncio
async def test_dealer_flow_backfill_respects_checkpoint():
    redis = AsyncFakeRedis()
    with open('config/config.yaml', 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    dealer_calc = DealerFlowCalculator(config, redis)
    symbol = 'SPY'
    start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    snapshots = [
        _build_dealer_snapshot(symbol, start_ts + timedelta(minutes=idx))
        for idx in range(2)
    ]

    checkpoint = InMemoryCheckpointStore()
    await checkpoint.save('dealer_flow', symbol, snapshots[0].timestamp)

    job = DealerFlowBackfillJob(
        redis_conn=redis,
        calculator=dealer_calc,
        provider=InMemoryDealerFlowProvider(snapshots),
        checkpoint_store=checkpoint,
    )

    result = await job.run([symbol])

    assert result.snapshots_processed == 1
    assert result.metrics_written == 1
    latest = await redis.get(rkeys.analytics_vanna_key(symbol))
    assert latest is not None


def _build_dealer_snapshot(symbol: str, timestamp: datetime) -> DealerFlowSnapshot:
    expiry_today = timestamp.astimezone(timezone.utc).date().isoformat()
    expiry_later = (timestamp + timedelta(days=2)).astimezone(timezone.utc).date().isoformat()

    contracts = []
    strikes = [420.0 + idx * 2.0 for idx in range(-6, 6)]
    for idx, strike in enumerate(strikes, start=1):
        vol = 0.20 + 0.01 * (idx % 5)
        gamma = 0.015 + 0.001 * (idx % 4)
        vega = 0.09 + 0.005 * (idx % 6)
        open_interest = 500 + idx * 15
        volume = 180 + idx * 7
        expiry = expiry_today if idx % 3 else expiry_later

        contracts.append(
            {
                'type': 'call',
                'strike': round(strike, 2),
                'expiration': expiry,
                'implied_volatility': vol,
                'delta': 0.15 + 0.01 * idx,
                'gamma': gamma,
                'vega': vega,
                'open_interest': open_interest,
                'volume': volume,
            }
        )
        contracts.append(
            {
                'type': 'put',
                'strike': round(strike, 2),
                'expiration': expiry,
                'implied_volatility': vol + 0.03,
                'delta': -0.15 - 0.01 * idx,
                'gamma': gamma,
                'vega': vega + 0.01,
                'open_interest': open_interest + 20,
                'volume': volume + 15,
            }
        )

    chain = {
        'as_of': timestamp.isoformat(),
        'contracts': contracts,
    }

    ticker = {
        'symbol': symbol,
        'last': 430.0,
        'bid': 429.9,
        'ask': 430.1,
        'timestamp': int(timestamp.timestamp()),
    }

    return DealerFlowSnapshot(
        symbol=symbol,
        timestamp=timestamp,
        chain=chain,
        ticker=ticker,
    )


def _build_trades(timestamp: datetime) -> List[dict]:
    base_ts = int(timestamp.timestamp())
    trades: List[dict] = []
    for idx in range(60):
        trades.append(
            {
                'price': 430.0 + (idx % 5) * 0.05,
                'size': 100 + (idx % 3) * 25,
                'timestamp': base_ts + idx,
                'side': 'buy' if idx % 2 == 0 else 'sell',
                'sweep': 1 if idx % 7 == 0 else 0,
            }
        )
    return trades
