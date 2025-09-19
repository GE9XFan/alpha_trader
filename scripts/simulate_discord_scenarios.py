#!/usr/bin/env python3
"""Simulate Discord signal cascades for premium/basic/free tiers.

Usage examples
--------------

List available scenarios:
    python scripts/simulate_discord_scenarios.py --list

Preview payloads (no Redis writes):
    python scripts/simulate_discord_scenarios.py --scenario 0dte_win

Replay through the full pipeline (writes to ``signals:distribution:pending``):
    python scripts/simulate_discord_scenarios.py --scenario moc_imbalance --push
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import redis
import yaml


CONFIG_PATH = Path("config/config.yaml")


@dataclass
class ScenarioEvent:
    name: str
    payload: Dict[str, object]


@dataclass
class Scenario:
    key: str
    description: str
    expectations: Dict[str, str]
    events: List[ScenarioEvent]


def _now_ts() -> int:
    return int(time.time() * 1000)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _redis_client(config: dict) -> redis.Redis:
    redis_cfg = config.get('redis', {})
    return redis.Redis(
        host=redis_cfg.get('host', '127.0.0.1'),
        port=int(redis_cfg.get('port', 6379)),
        db=int(redis_cfg.get('db', 0)),
        password=redis_cfg.get('password'),
        decode_responses=True,
    )


def _base_entry(
    *,
    signal_id: str,
    symbol: str,
    strategy: str,
    side: str,
    entry: float,
    stop: float,
    target: float,
    confidence: float,
    reasons: Iterable[str],
    qty: int,
    notional: float = None,
) -> Dict[str, object]:
    ts = _now_ts()
    position_notional = notional if notional is not None else round(entry * qty * 100, 2)
    return {
        'id': signal_id,
        'symbol': symbol,
        'side': side,
        'strategy': strategy,
        'confidence': confidence,
        'reasons': list(reasons),
        'entry': entry,
        'stop': stop,
        'targets': [target],
        'position_notional': position_notional,
        'contract': {
            'type': 'option',
            'expiry': '20240426',
            'strike': 455 if symbol == 'SPY' else 400,
            'right': 'C' if side == 'LONG' else 'P',
            'symbol': symbol,
        },
        'action_type': 'ENTRY',
        'ts': ts,
        'execution': {
            'status': 'FILLED',
            'avg_fill_price': entry,
            'filled_quantity': qty,
            'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }


SCENARIOS: Dict[str, Scenario] = {}


def _register_scenario(scenario: Scenario) -> None:
    SCENARIOS[scenario.key] = scenario


_register_scenario(
    Scenario(
        key='0dte_win',
        description='0DTE entry, partial scale, and profitable exit.',
        expectations={
            'premium': 'Full contract, execution, notional, scale rationale, realized P&L + return % on exit.',
            'basic': 'Contract, execution (price/size), full risk plan, confidence band, drivers, upgrade CTA.',
            'free': 'Contract, fill price only, entry/stop risk band, upgrade CTA.',
        },
        events=[
            ScenarioEvent(
                'entry',
                _base_entry(
                    signal_id='sim-0dte',
                    symbol='SPY',
                    strategy='0DTE',
                    side='LONG',
                    entry=4.25,
                    stop=4.05,
                    target=6.45,
                    confidence=94,
                    reasons=['Flow imbalance', 'Dealer unwind'],
                    qty=5,
                    notional=12500,
                ),
            ),
            ScenarioEvent(
                'scale_out',
                {
                    'id': 'sim-0dte',
                    'symbol': 'SPY',
                    'side': 'LONG',
                    'strategy': '0DTE',
                    'confidence': 94,
                    'entry': 4.25,
                    'stop': 4.05,
                    'targets': [6.45],
                    'position_notional': 12500,
                    'action_type': 'SCALE_OUT',
                    'ts': _now_ts(),
                    'execution': {
                        'status': 'PARTIAL',
                        'avg_fill_price': 5.90,
                        'filled_quantity': 2,
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    },
                    'lifecycle': {
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'fill_price': 5.90,
                        'quantity': 2,
                        'remaining_quantity': 3,
                        'reason': 'Target 1 reached',
                        'result': 'SCALE_OUT',
                    },
                },
            ),
            ScenarioEvent(
                'exit',
                {
                    'id': 'sim-0dte',
                    'symbol': 'SPY',
                    'side': 'LONG',
                    'strategy': '0DTE',
                    'confidence': 94,
                    'entry': 4.25,
                    'stop': 4.05,
                    'targets': [6.45],
                    'position_notional': 12500,
                    'action_type': 'EXIT',
                    'ts': _now_ts(),
                    'execution': {
                        'status': 'CLOSED',
                        'avg_fill_price': 6.40,
                        'filled_quantity': 3,
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    },
                    'lifecycle': {
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'exit_price': 6.40,
                        'quantity': 3,
                        'realized_pnl': 1850.0,
                        'return_pct': 0.148,
                        'holding_period_minutes': 43.0,
                        'reason': 'Target 2 reached',
                        'result': 'WIN',
                        'remaining_quantity': 0,
                    },
                },
            ),
        ],
    )
)


_register_scenario(
    Scenario(
        key='1dte_loss',
        description='1DTE entry followed by stop-loss exit.',
        expectations={
            'premium': 'Shows stop trigger with negative P&L and return %, holding time, exit reason.',
            'basic': 'Outcome = LOSS, holding duration, upgrade CTA.',
            'free': 'Fill price + entry/stop reminder with upgrade CTA.',
        },
        events=[
            ScenarioEvent(
                'entry',
                _base_entry(
                    signal_id='sim-1dte',
                    symbol='QQQ',
                    strategy='1DTE',
                    side='LONG',
                    entry=2.10,
                    stop=2.00,
                    target=2.80,
                    confidence=93,
                    reasons=['Volatility regime shift', 'Order book imbalance'],
                    qty=8,
                ),
            ),
            ScenarioEvent(
                'exit',
                {
                    'id': 'sim-1dte',
                    'symbol': 'QQQ',
                    'side': 'LONG',
                    'strategy': '1DTE',
                    'confidence': 93,
                    'entry': 2.10,
                    'stop': 2.00,
                    'targets': [2.80],
                    'position_notional': 1680,
                    'action_type': 'EXIT',
                    'ts': _now_ts(),
                    'execution': {
                        'status': 'CLOSED',
                        'avg_fill_price': 1.95,
                        'filled_quantity': 8,
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    },
                    'lifecycle': {
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'exit_price': 1.95,
                        'quantity': 8,
                        'realized_pnl': -120.0,
                        'return_pct': -0.071,
                        'holding_period_minutes': 25.0,
                        'reason': 'Stop loss hit',
                        'result': 'LOSS',
                        'remaining_quantity': 0,
                    },
                },
            ),
        ],
    )
)


_register_scenario(
    Scenario(
        key='14dte_breakeven',
        description='14+DTE swing with scale-out and breakeven exit.',
        expectations={
            'premium': 'Shows scale ledger, breakeven exit with realized P&L ≈ 0 and long holding period.',
            'basic': 'Highlights partial reduction, remaining size, outcome = BREAKEVEN.',
            'free': 'Teaser keeps entry/stop while hiding size/notional.',
        },
        events=[
            ScenarioEvent(
                'entry',
                _base_entry(
                    signal_id='sim-14dte',
                    symbol='AAPL',
                    strategy='14DTE',
                    side='LONG',
                    entry=5.60,
                    stop=5.30,
                    target=6.80,
                    confidence=95,
                    reasons=['Institutional sweep cluster', 'Dealer skew decompressing'],
                    qty=10,
                ),
            ),
            ScenarioEvent(
                'scale_out',
                {
                    'id': 'sim-14dte',
                    'symbol': 'AAPL',
                    'side': 'LONG',
                    'strategy': '14DTE',
                    'confidence': 95,
                    'entry': 5.60,
                    'stop': 5.30,
                    'targets': [6.80],
                    'position_notional': 5600,
                    'action_type': 'SCALE_OUT',
                    'ts': _now_ts(),
                    'execution': {
                        'status': 'PARTIAL',
                        'avg_fill_price': 6.40,
                        'filled_quantity': 4,
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    },
                    'lifecycle': {
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'fill_price': 6.40,
                        'quantity': 4,
                        'remaining_quantity': 6,
                        'reason': 'Target 1 scaled',
                        'result': 'SCALE_OUT',
                    },
                },
            ),
            ScenarioEvent(
                'exit',
                {
                    'id': 'sim-14dte',
                    'symbol': 'AAPL',
                    'side': 'LONG',
                    'strategy': '14DTE',
                    'confidence': 95,
                    'entry': 5.60,
                    'stop': 5.30,
                    'targets': [6.80],
                    'position_notional': 5600,
                    'action_type': 'EXIT',
                    'ts': _now_ts(),
                    'execution': {
                        'status': 'CLOSED',
                        'avg_fill_price': 5.62,
                        'filled_quantity': 6,
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    },
                    'lifecycle': {
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'exit_price': 5.62,
                        'quantity': 6,
                        'realized_pnl': 48.0,
                        'return_pct': 0.008,
                        'holding_period_minutes': 7200.0,
                        'reason': 'Time-based exit',
                        'result': 'BREAKEVEN',
                        'remaining_quantity': 0,
                    },
                },
            ),
        ],
    )
)


_register_scenario(
    Scenario(
        key='moc_imbalance',
        description='MOC imbalance entry and profitable exit into the close.',
        expectations={
            'premium': 'Highlights imbalance strength context, execution near close, realized profit.',
            'basic': 'Shows contract, fill/size, confidence band, drivers (imbalance, gamma, OBI).',
            'free': 'Entry/stop teaser with upgrade CTA.',
        },
        events=[
            ScenarioEvent(
                'entry',
                _base_entry(
                    signal_id='sim-moc',
                    symbol='SPY',
                    strategy='MOC',
                    side='LONG',
                    entry=3.40,
                    stop=3.20,
                    target=3.95,
                    confidence=96,
                    reasons=['Imbalance strength 3.1B', 'Gamma pull aligned'],
                    qty=20,
                ),
            ),
            ScenarioEvent(
                'exit',
                {
                    'id': 'sim-moc',
                    'symbol': 'SPY',
                    'side': 'LONG',
                    'strategy': 'MOC',
                    'confidence': 96,
                    'entry': 3.40,
                    'stop': 3.20,
                    'targets': [3.95],
                    'position_notional': 6800,
                    'action_type': 'EXIT',
                    'ts': _now_ts(),
                    'execution': {
                        'status': 'CLOSED',
                        'avg_fill_price': 3.92,
                        'filled_quantity': 20,
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    },
                    'lifecycle': {
                        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'exit_price': 3.92,
                        'quantity': 20,
                        'realized_pnl': 1040.0,
                        'return_pct': 0.153,
                        'holding_period_minutes': 8.0,
                        'reason': 'MOC print',
                        'result': 'WIN',
                        'remaining_quantity': 0,
                    },
                },
            ),
        ],
    )
)


def iter_scenarios() -> Iterable[Scenario]:
    for key in sorted(SCENARIOS):
        yield SCENARIOS[key]


def push_events(events: List[ScenarioEvent], client: redis.Redis) -> None:
    run_suffix = str(int(time.time()))
    id_map: Dict[str, str] = {}
    for event in events:
        payload = event.payload.copy()
        payload.setdefault('ts', _now_ts())
        base_id = str(payload.get('id'))
        if base_id not in id_map:
            id_map[base_id] = f"{base_id}-{run_suffix}"
        payload['id'] = id_map[base_id]
        client.lpush('signals:distribution:pending', json.dumps(payload))
        print(f"Enqueued {payload['id']}:{payload['action_type']} -> signals:distribution:pending")


def print_scenario(scenario: Scenario) -> None:
    print(f"Scenario: {scenario.key}\nDescription: {scenario.description}\n")
    print("Tier expectations:")
    for tier, expectation in scenario.expectations.items():
        print(f"  - {tier.capitalize()}: {expectation}")
    print("\nEvents:")
    for event in scenario.events:
        print(f"  • {event.name} -> action={event.payload['action_type']} id={event.payload['id']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Discord signal cascades")
    parser.add_argument('--list', action='store_true', help='List available scenarios and exit')
    parser.add_argument('--scenario', choices=sorted(SCENARIOS.keys()), help='Scenario key to replay')
    parser.add_argument('--push', action='store_true', help='Push events into Redis')
    args = parser.parse_args()

    if args.list or not args.scenario:
        print("Available scenarios:\n")
        for scenario in iter_scenarios():
            print(f"- {scenario.key}: {scenario.description}")
        if not args.scenario:
            return
        print()

    scenario = SCENARIOS[args.scenario]
    print_scenario(scenario)

    if args.push:
        config = _load_config(CONFIG_PATH)
        client = _redis_client(config)
        push_events(scenario.events, client)
    else:
        print("(dry run — no Redis writes)\n")


if __name__ == '__main__':  # pragma: no cover - CLI entry point
    main()
