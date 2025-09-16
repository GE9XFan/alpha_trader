"""Unit tests for analytics engine aggregator and scheduler behavior."""

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from src.analytics_engine import AnalyticsEngine, MetricsAggregator
import src.redis_keys as rkeys


class FakeRedis:
    """Minimal async Redis stub for unit testing."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttl_store: dict[str, int] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value
        self.ttl_store[key] = ttl

    async def ttl(self, key: str) -> int:
        return self.ttl_store.get(key, -1)

    async def aclose(self):
        self.store.clear()
        self.ttl_store.clear()


@pytest.mark.asyncio
async def test_metrics_aggregator_portfolio_and_sectors():
    """Aggregator should publish portfolio summary and sector snapshots with configured TTLs."""

    config = {
        'symbols': {
            'standard': ['SPY', 'QQQ'],
            'level2': []
        },
        'modules': {
            'analytics': {
                'store_ttls': {
                    'analytics': 30,
                    'portfolio': 45,
                    'sector': 90,
                    'correlation': 120,
                },
                'sector_map': {
                    'SPY': 'INDEX',
                    'QQQ': 'TECH',
                }
            }
        },
        'parameter_discovery': {
            'toxicity_detection': {
                'vpin_thresholds': {
                    'toxic': 0.7,
                }
            }
        }
    }

    redis = FakeRedis()

    await redis.setex(rkeys.analytics_vpin_key('SPY'), 60, json.dumps({'value': 0.45}))
    await redis.setex(rkeys.analytics_vpin_key('QQQ'), 60, json.dumps({'value': 0.82}))

    await redis.setex(rkeys.analytics_gex_key('SPY'), 60, json.dumps({'total_gex': 1.5e9}))
    await redis.setex(rkeys.analytics_gex_key('QQQ'), 60, json.dumps({'total_gex': -2.5e9}))

    await redis.setex(
        rkeys.analytics_dex_key('SPY'),
        60,
        json.dumps({'total_dex': 5.0e8, 'call_dex': 6.0e8, 'put_dex': -1.0e8})
    )
    await redis.setex(
        rkeys.analytics_dex_key('QQQ'),
        60,
        json.dumps({'total_dex': -3.0e8, 'call_dex': 1.0e8, 'put_dex': -4.0e8})
    )

    await redis.setex(rkeys.market_ticker_key('SPY'), 60, json.dumps({'volume': 1_200_000}))
    await redis.setex(rkeys.market_ticker_key('QQQ'), 60, json.dumps({'volume': 950_000}))

    aggregator = MetricsAggregator(config, redis)

    metrics = await aggregator.calculate_portfolio_metrics()
    stored_summary = json.loads(redis.store[rkeys.analytics_portfolio_summary_key()])

    assert metrics['avg_vpin'] == pytest.approx((0.45 + 0.82) / 2, rel=1e-3)
    assert metrics['max_vpin_symbol'] == 'QQQ'
    assert metrics['total_dex'] == pytest.approx(2.0e8)
    assert stored_summary['sector_flows']['INDEX']['net_flow'] == pytest.approx(5.0e8)
    assert stored_summary['sector_flows']['TECH']['total_volume'] == pytest.approx(950_000)
    assert redis.ttl_store[rkeys.analytics_portfolio_summary_key()] == 45

    sector_payload = await aggregator.calculate_sector_flows()
    assert sector_payload['INDEX']['avg_vpin'] == pytest.approx(0.45, rel=1e-3)
    assert sector_payload['TECH']['toxic_symbols'] == ['QQQ']
    assert redis.ttl_store[rkeys.analytics_sector_key('INDEX')] == 90


@pytest.mark.asyncio
async def test_metrics_aggregator_correlation_sources():
    """Aggregator should hydrate correlations from discovery or compute them from bars."""

    config = {
        'symbols': {
            'standard': ['SPY', 'QQQ'],
            'level2': []
        },
        'modules': {
            'analytics': {
                'store_ttls': {
                    'correlation': 300
                }
            }
        },
        'parameter_discovery': {
            'correlation': {
                'calculation_window': 5,
                'min_correlation_symbols': 2,
            }
        },
        'risk_management': {
            'correlation_limit': 0.6,
        }
    }

    redis = FakeRedis()
    aggregator = MetricsAggregator(config, redis)

    discovered_matrix = {
        'SPY': {'QQQ': 0.88, 'SPY': 1.0},
        'QQQ': {'SPY': 0.88, 'QQQ': 1.0},
    }
    redis.store['discovered:correlation_matrix'] = json.dumps(discovered_matrix)

    payload = await aggregator.calculate_cross_asset_correlations()
    assert payload['source'] == 'discovered'
    assert payload['high_pairs'][0]['pair'] == ['QQQ', 'SPY']
    assert redis.ttl_store[rkeys.analytics_portfolio_correlation_key()] == 300

    redis.store.pop('discovered:correlation_matrix')

    spy_bars = [{'close': price} for price in [100, 101, 102, 103, 104]]
    qqq_bars = [{'close': price} for price in [200, 201, 202, 203, 204]]
    await redis.setex(rkeys.market_bars_key('SPY'), 60, json.dumps(spy_bars))
    await redis.setex(rkeys.market_bars_key('QQQ'), 60, json.dumps(qqq_bars))

    computed = await aggregator.calculate_cross_asset_correlations()
    assert computed['source'] == 'computed'
    assert computed['pair_count'] == 1
    assert computed['high_pairs'][0]['correlation'] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.asyncio
async def test_analytics_engine_respects_market_hours(monkeypatch):
    """Engine should idle when configured for RTH-only operation outside market hours."""

    class DummyVPIN:
        instances: list['DummyVPIN'] = []

        def __init__(self, *_args, **_kwargs):
            self.calls: list[str] = []
            DummyVPIN.instances.append(self)

        async def calculate_vpin(self, symbol: str):
            self.calls.append(symbol)
            return 0.5

    class DummyGEXDEX:
        instances: list['DummyGEXDEX'] = []

        def __init__(self, *_args, **_kwargs):
            self.calls: list[str] = []
            DummyGEXDEX.instances.append(self)

        async def calculate_gex(self, symbol: str):
            self.calls.append(f"gex:{symbol}")
            return {}

        async def calculate_dex(self, symbol: str):
            self.calls.append(f"dex:{symbol}")
            return {}

    class DummyPattern:
        instances: list['DummyPattern'] = []

        def __init__(self, *_args, **_kwargs):
            self.calls: list[str] = []
            DummyPattern.instances.append(self)

        async def analyze_flow_toxicity(self, symbol: str):
            self.calls.append(f"toxicity:{symbol}")

        async def calculate_order_book_imbalance(self, symbol: str):
            self.calls.append(f"obi:{symbol}")

        async def detect_sweeps(self, symbol: str):
            self.calls.append(f"sweep:{symbol}")

    class DummyAggregator:
        instances: list['DummyAggregator'] = []

        def __init__(self, *_args, **_kwargs):
            self.calls = defaultdict(int)
            DummyAggregator.instances.append(self)

        async def calculate_portfolio_metrics(self):
            self.calls['portfolio'] += 1

        async def calculate_sector_flows(self):
            self.calls['sectors'] += 1

        async def calculate_cross_asset_correlations(self):
            self.calls['correlation'] += 1

    monkeypatch.setattr('src.analytics_engine.VPINCalculator', DummyVPIN)
    monkeypatch.setattr('src.analytics_engine.GEXDEXCalculator', DummyGEXDEX)
    monkeypatch.setattr('src.analytics_engine.PatternAnalyzer', DummyPattern)
    monkeypatch.setattr('src.analytics_engine.MetricsAggregator', DummyAggregator)

    config = {
        'symbols': {
            'standard': ['SPY'],
            'level2': []
        },
        'market': {
            'extended_hours': False
        },
        'modules': {
            'analytics': {
                'cadence_hz': 10,
                'analytics_rth_only': True,
                'store_ttls': {
                    'heartbeat': 5
                },
                'update_intervals': {
                    'vpin': 0.05,
                    'gex_dex': 0.05,
                    'flow_toxicity': 0.05,
                    'order_book': 0.05,
                    'sweep_detection': 0.05,
                    'portfolio': 0.05,
                    'sectors': 0.05,
                    'correlation': 0.05,
                },
                'sector_map': {
                    'SPY': 'INDEX'
                }
            }
        }
    }

    redis = FakeRedis()
    engine = AnalyticsEngine(config, redis)
    engine.is_market_hours = lambda _extended=False: False

    task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.2)
    await engine.stop()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 0.5)

    assert DummyVPIN.instances and DummyVPIN.instances[0].calls == []
    assert DummyGEXDEX.instances[0].calls == []
    assert dict(DummyAggregator.instances[0].calls) == {}
    heartbeat = json.loads(redis.store[rkeys.heartbeat_key('analytics')])
    assert heartbeat['idle'] is True
    assert heartbeat['running'] is False
    assert rkeys.analytics_portfolio_summary_key() not in redis.store
