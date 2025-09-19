import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

try:
    from signal_generator import SignalGenerator
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(f"SignalGenerator dependencies unavailable: {exc}", allow_module_level=True)


class DummyRedis:
    def __init__(self):
        self.exists_map = {}
        self.scard_map = {}
        self.get_map = {}
        self.scan_map = {}
        self.incr_counters = {}
        self.hincrby_map = {}

    async def exists(self, key):
        return 1 if self.exists_map.get(key) else 0

    async def scard(self, key):
        return self.scard_map.get(key, 0)

    async def get(self, key):
        return self.get_map.get(key)

    async def scan_iter(self, match=None):
        for key in self.scan_map.get(match, []):
            yield key

    async def incr(self, key):
        self.incr_counters[key] = self.incr_counters.get(key, 0) + 1

    async def hincrby(self, key, field, amount):
        bucket = self.hincrby_map.setdefault(key, {})
        bucket[field] = bucket.get(field, 0) + amount

    async def setex(self, *args, **kwargs):  # pragma: no cover - not used in test
        return

    async def publish(self, *args, **kwargs):  # pragma: no cover
        return


@pytest.mark.asyncio
async def test_exposure_caps_block_and_metrics(monkeypatch):
    redis = DummyRedis()
    redis.exists_map['signals:live:SPY:sigfp:test'] = True
    redis.scan_map['signals:live:SPY:*'] = ['signals:live:SPY:sigfp:test', 'signals:live:SPY:fp2']
    redis.scan_map['orders:pending:*'] = ['orders:pending:1']
    redis.get_map['orders:pending:1'] = json.dumps({'symbol': 'SPY'})
    redis.scard_map['positions:by_symbol:SPY'] = 2

    config = {
        'modules': {
            'signals': {
                'ttl_seconds': 300,
                'exposure_caps': {
                    'strategies': {
                        '0dte': {
                            'max_positions': 1,
                            'max_pending_orders': 0,
                            'max_live_signals': 1,
                        }
                    }
                }
            }
        }
    }

    generator = SignalGenerator(config, redis)
    # Inject stub deduper with async no-op methods
    class StubDeduper:
        async def add_audit_entry(self, *args, **kwargs):
            return

        async def publish_update(self, *args, **kwargs):
            return

    generator._deduper = StubDeduper()

    exposure_state = await generator._gather_exposure_state('SPY', 'sigfp:test')

    blocked = await generator._check_exposure_caps(
        'SPY',
        '0dte',
        'sigfp:test',
        exposure_state,
        {'id': 'test-signal'},
    )

    assert blocked is True
    assert redis.incr_counters.get('metrics:signals:blocked:exposure') == 1
