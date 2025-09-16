import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytz

from src.execution_manager import ExecutionManager
from src.position_manager import PositionManager
from src.risk_manager import RiskManager
from src.emergency_manager import EmergencyManager


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.hash_store = {}
        self.set_store = {}
        self.list_store = {}
        self.published = []

    async def set(self, key, value):
        self.store[key] = value

    async def setex(self, key, ttl, value):
        await self.set(key, value)

    async def get(self, key):
        if key in self.store:
            return self.store[key]
        if key in self.hash_store:
            return self.hash_store[key]
        if key in self.set_store:
            return self.set_store[key]
        if key in self.list_store:
            return self.list_store[key]
        return None

    async def delete(self, key):
        removed = 0
        for store in (self.store, self.hash_store, self.set_store, self.list_store):
            if key in store:
                del store[key]
                removed += 1
        return removed

    async def exists(self, key):
        return 1 if key in self.store or key in self.hash_store or key in self.set_store else 0

    async def keys(self, pattern):
        from fnmatch import fnmatch

        keys = set(self.store.keys()) | set(self.hash_store.keys()) | set(self.set_store.keys()) | set(self.list_store.keys())
        return [key for key in keys if fnmatch(key, pattern)]

    async def incr(self, key):
        return await self.incrby(key, 1)

    async def incrby(self, key, amount):
        value = int(self.store.get(key, 0)) + amount
        self.store[key] = value
        return value

    async def incrbyfloat(self, key, amount):
        value = float(self.store.get(key, 0)) + amount
        self.store[key] = value
        return value

    async def hincrby(self, key, field, amount):
        hash_map = self.hash_store.setdefault(key, {})
        hash_map[field] = int(hash_map.get(field, 0)) + amount
        return hash_map[field]

    async def sadd(self, key, member):
        self.set_store.setdefault(key, set()).add(member)

    async def srem(self, key, member):
        if key in self.set_store:
            self.set_store[key].discard(member)

    async def lpush(self, key, value):
        self.list_store.setdefault(key, [])
        self.list_store[key].insert(0, value)

    async def ltrim(self, key, start, end):
        if key in self.list_store:
            self.list_store[key] = self.list_store[key][start : end + 1]

    async def lrange(self, key, start, end):
        values = self.list_store.get(key, [])
        return values[start : end + 1]

    async def publish(self, channel, message):
        self.published.append((channel, message))

    async def rpop(self, key):
        if key in self.list_store and self.list_store[key]:
            return self.list_store[key].pop()
        return None


@pytest.mark.asyncio
async def test_execution_manager_reuses_risk_manager():
    redis = FakeRedis()
    await redis.set('risk:new_positions_allowed', 'true')
    await redis.set('account:buying_power', '100000')
    await redis.set('risk:daily_pnl', '0')

    factory_calls = {'count': 0}

    class StubRiskManager:
        def __init__(self):
            factory_calls['count'] += 1

        async def check_correlations(self, symbol, side):
            return True

    manager = ExecutionManager(
        config={'trading': {'max_positions': 5, 'max_per_symbol': 2}},
        redis_conn=redis,
        risk_manager_factory=lambda: StubRiskManager(),
    )

    signal = {'symbol': 'AAPL', 'side': 'LONG', 'contract': {}}

    assert await manager.passes_risk_checks(signal)
    assert await manager.passes_risk_checks(signal)
    assert factory_calls['count'] == 1


class DummyTrade:
    def __init__(self, order_id, status, filled, total_quantity, fills):
        self.order = SimpleNamespace(orderId=order_id, orderType='LMT', totalQuantity=total_quantity)
        self.orderStatus = SimpleNamespace(status=status, filled=filled, remaining=total_quantity - filled)
        self.fills = fills

    def isDone(self):
        return self.orderStatus.status in {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive', 'Rejected'}


def make_fill(shares, price, commission=0.0):
    execution = SimpleNamespace(shares=shares, price=price)
    commission_report = SimpleNamespace(commission=commission)
    return SimpleNamespace(execution=execution, commissionReport=commission_report, time=datetime.utcnow())


@pytest.mark.asyncio
async def test_execution_manager_sync_trade_state_sanitises_payload():
    redis = FakeRedis()
    manager = ExecutionManager(config={'trading': {}}, redis_conn=redis)

    manager.pending_orders[1] = {
        'order_id': 1,
        'symbol': 'AAPL',
        'size': 10,
        'status': 'PENDING',
        'placed_at': time.time(),
        'signal': {'id': 'sig1'},
    }

    trade = DummyTrade(order_id=1, status='Submitted', filled=4, total_quantity=10, fills=[make_fill(4, 5.0, 1.0)])

    snapshot = await manager._sync_trade_state(1, trade, manager.pending_orders[1]['signal'])

    assert snapshot['filled'] == 4
    assert snapshot['remaining'] == 6
    stored = json.loads(redis.store['orders:pending:1'])
    assert 'signal' not in stored


@pytest.mark.asyncio
async def test_risk_manager_aggregates_greeks_and_var_snapshot():
    redis = FakeRedis()
    await redis.set('risk:var:portfolio', json.dumps({'var_95': 2000, 'confidence': 0.95, 'timestamp': time.time()}))
    await redis.set('risk:current_drawdown', json.dumps({'drawdown_pct': 4.0}))
    await redis.set('account:value', '100000')
    await redis.set('positions:open:AAPL:pos1', json.dumps({
        'id': 'pos1',
        'symbol': 'AAPL',
        'contract': {'type': 'option'},
        'market_value': 5000,
    }))
    await redis.set('positions:open:MSFT:pos2', json.dumps({
        'id': 'pos2',
        'symbol': 'MSFT',
        'contract': {'type': 'stock'},
        'market_value': 8000,
    }))
    await redis.set('positions:greeks:pos1', json.dumps({'delta': 10, 'gamma': 1, 'theta': -5, 'vega': 2}))

    risk_manager = RiskManager(config={'risk_management': {}}, redis_conn=redis)
    await risk_manager.update_risk_metrics()

    metrics = json.loads(redis.store['risk:metrics:summary'])
    assert metrics['greeks'] == {'delta': 10.0, 'gamma': 1.0, 'theta': -5.0, 'vega': 2.0}
    assert metrics['concentration']['position_count'] == 2
    assert metrics['risk_score']['total'] > 0


@pytest.mark.asyncio
async def test_emergency_manager_cancels_all_signal_queues():
    redis = FakeRedis()
    await redis.set('orders:pending:1', json.dumps({'order_id': '1'}))
    await redis.set('orders:working:2', json.dumps({'order_id': '2'}))
    await redis.set('signals:pending:AAPL', json.dumps({'id': 'sig'}))
    await redis.set('signals:pending', json.dumps({'id': 'old'}))

    manager = EmergencyManager(config={}, redis_conn=redis)
    await manager.cancel_all_orders()

    assert 'orders:pending:1' not in redis.store
    assert 'orders:working:2' not in redis.store
    assert 'signals:pending:AAPL' not in redis.store
    assert redis.store.get('signals:pending') is None
    assert int(redis.store.get('metrics:orders:cancelled:total')) == 2


@pytest.mark.asyncio
async def test_position_manager_handle_eod_rules(monkeypatch):
    redis = FakeRedis()
    manager = PositionManager(config={}, redis_conn=redis)
    manager.eod_rules = [{'strategy': 'EOD', 'action': 'reduce', 'reduction': 0.5, 'reason': 'Trim risk'}]
    position = {
        'id': 'pos1',
        'symbol': 'AAPL',
        'quantity': 4,
        'current_price': 10,
        'entry_price': 8,
        'strategy': 'EOD',
        'contract': {'type': 'stock'},
        'side': 'LONG',
    }
    manager.positions = {'pos1': position}

    async_reduce = AsyncMock()
    monkeypatch.setattr(manager, '_reduce_position', async_reduce)
    monkeypatch.setattr(PositionManager, '_calculate_dte', lambda self, position: 0)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2024, 1, 2, 15, 50)
            if tz:
                return pytz.timezone('US/Eastern').localize(base)
            return base

    monkeypatch.setattr('src.position_manager.datetime', FakeDateTime)

    await manager.handle_eod_positions()

    async_reduce.assert_awaited_once()
