import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import redis_keys as rkeys
from signal_generator import SignalGenerator, default_feature_reader
from dte_strategies import DTEStrategies
from moc_strategy import MOCStrategy
from signal_distributor import SignalValidator


class DummyRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def incr(self, key):
        self.store[key] = int(self.store.get(key, 0)) + 1

    async def hset(self, key, mapping):
        self.store.setdefault(key, {}).update(mapping)

    async def expire(self, key, ttl):
        return True

    async def hincrbyfloat(self, key, field, amount):
        self.store.setdefault(key, {})
        self.store[key][field] = float(self.store[key].get(field, 0.0)) + amount

    async def hincrby(self, key, field, amount):
        self.store.setdefault(key, {})
        self.store[key][field] = int(self.store[key].get(field, 0)) + amount

    async def lpush(self, key, value):
        self.store.setdefault(key, [])
        self.store[key].insert(0, value)

    async def ltrim(self, key, start, end):
        if key in self.store and isinstance(self.store[key], list):
            self.store[key] = self.store[key][start : end + 1]

    async def zadd(self, *args, **kwargs):
        return 1

    def pipeline(self):
        return AsyncPipeline(self)

    async def hgetall(self, key):
        return self.store.get(key, {})

    async def lrange(self, key, start, end):
        if key not in self.store:
            return []
        return self.store[key][start : end + 1 if end != -1 else None]


class AsyncPipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # aioredis pipelines do not auto-execute; keep compatibility with
        # callers that explicitly await execute().
        return False

    def hincrbyfloat(self, key, field, amount):
        self.commands.append(self.redis.hincrbyfloat(key, field, amount))

    def hincrby(self, key, field, amount):
        self.commands.append(self.redis.hincrby(key, field, amount))

    def lpush(self, key, value):
        self.commands.append(self.redis.lpush(key, value))

    def ltrim(self, key, start, end):
        self.commands.append(self.redis.ltrim(key, start, end))

    def expire(self, key, ttl):
        self.commands.append(self.redis.expire(key, ttl))

    def get(self, key):
        self.commands.append(self.redis.get(key))
        return self

    def lrange(self, key, start, end):
        self.commands.append(self.redis.lrange(key, start, end))
        return self

    def execute(self):
        result = asyncio.gather(*self.commands)
        self.commands = []
        return result


@pytest.mark.asyncio
async def test_signal_generator_ttl_rolls_to_next_session():
    config = {'modules': {'signals': {'enabled': True}}}
    generator = SignalGenerator(config, DummyRedis())
    now = datetime(2024, 1, 2, 17, 0)  # After close
    ttl = generator.calculate_dynamic_ttl({'expiry': '0DTE'}, now=now)
    assert ttl >= 60


@pytest.mark.asyncio
async def test_dte_contract_hysteresis_respects_band():
    config = {'modules': {'signals': {'strategies': {}, 'ttl_seconds': 300}}}
    redis = DummyRedis()
    strategies = DTEStrategies(config, redis)
    features = {'price': 430, 'timestamp': 1, 'vpin': 0.5, 'obi': 0.5}

    first = await strategies.select_contract('SPY', '0dte', 'LONG', 430.2, [])
    assert first['dte_band'] == '0'

    # Spot has not moved enough to roll, expect same strike
    second = await strategies.select_contract('SPY', '0dte', 'LONG', 430.3, [])
    assert second['strike'] == first['strike']


@pytest.mark.asyncio
async def test_moc_direction_flat_on_contradiction():
    config = {'modules': {'signals': {'strategies': {'moc': {'thresholds': {}, 'confidence_weights': {}}}}}}
    strategy = MOCStrategy(config, DummyRedis())
    features = {
        'imbalance_side': 'BUY',
        'imbalance_total': 4e9,
        'gamma_pin_proximity': 0.8,
        'indicative_price': 395,
        'price': 390,
        'obi': 0.3,
        'bars': [{'close': 390}, {'close': 388}, {'close': 386}, {'close': 384}, {'close': 382}],
    }
    direction = strategy._determine_moc_direction(features, 'BUY', 4e9, 0.8, 395, 390)
    assert direction == 'FLAT'


@pytest.mark.asyncio
async def test_signal_validator_confidence_scaling():
    config = {'modules': {'signals': {'min_confidence': 0.6}}}
    validator = SignalValidator(config, DummyRedis())
    valid = validator.validate_signal({'confidence': 65, 'strategy': '0dte'})
    invalid = validator.validate_signal({'confidence': 40, 'strategy': '0dte'})
    assert valid is True
    assert invalid is False


@pytest.mark.asyncio
async def test_default_feature_reader_uses_vpin_and_obi_fields():
    redis = DummyRedis()
    symbol = 'SPY'
    redis.store[rkeys.analytics_vpin_key(symbol)] = json.dumps({'value': 0.9794})
    redis.store[rkeys.analytics_obi_key(symbol)] = json.dumps(
        {
            'level1_imbalance': 0.6364,
            'level5_imbalance': 0.6433,
            'pressure_ratio': 1.93,
        }
    )

    features = await default_feature_reader(redis, symbol)

    assert features['vpin'] == pytest.approx(0.9794)
    assert features['obi'] == pytest.approx(0.6364)
