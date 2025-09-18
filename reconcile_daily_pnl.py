#!/usr/bin/env python3
"""
Reconcile IBKR trades for the current trading day and push totals into Redis.

Usage:
    python reconcile_daily_pnl.py            # dry-run (prints results)
    python reconcile_daily_pnl.py --apply    # also updates Redis metrics

Requirements:
    - ib_insync
    - redis
    - pyyaml
    - pytz
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timezone
from typing import Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo

import pytz
import redis
import yaml
from ib_insync import IB, Contract, ExecutionFilter, Fill


@dataclass
class ExecutionSummary:
    account: str
    symbol: str
    local_symbol: str
    con_id: int
    multiplier: float
    realized: float = 0.0
    commission: float = 0.0
    last_fill_time: Optional[datetime] = None
    last_price: Optional[float] = None
    events: List[Dict[str, Optional[float]]] = field(default_factory=list)


def load_config() -> Dict:
    with open('config/config.yaml', 'r') as fh:
        return yaml.safe_load(fh)


def eastern_trading_day_bounds(now_et: datetime) -> Tuple[datetime, datetime]:
    start = datetime.combine(now_et.date(), dt_time(0, 0), tzinfo=now_et.tzinfo)
    end = datetime.combine(now_et.date(), dt_time(23, 59, 59), tzinfo=now_et.tzinfo)
    return start, end


async def fetch_fills(ib: IB, start_utc: datetime) -> List[Fill]:
    filt = ExecutionFilter(time=start_utc.strftime('%Y%m%d %H:%M:%S'))
    await ib.reqExecutionsAsync(filt)
    return list(ib.fills())


async def load_contracts(ib: IB, fills: List[Fill]) -> Dict[int, Contract]:
    contracts: Dict[int, Contract] = {}
    pending: List[Tuple[int, asyncio.Future]] = []

    for fill in fills:
        contract = getattr(fill, 'contract', None)
        con_id = getattr(contract, 'conId', None)
        if con_id in contracts or con_id is None:
            continue
        pending.append((con_id, ib.reqContractDetailsAsync(Contract(conId=con_id))))

    if not pending:
        return contracts

    results = await asyncio.gather(*(item[1] for item in pending), return_exceptions=True)
    for (con_id, _), details in zip(pending, results):
        if isinstance(details, Exception):
            continue
        if details:
            contracts[con_id] = details[0].contract

    return contracts


def build_summaries(fills: List[Fill], contracts: Dict[int, Contract]) -> Tuple[List[ExecutionSummary], float]:
    summaries: Dict[Tuple[str, int], ExecutionSummary] = {}

    fills_sorted = sorted(fills, key=lambda f: f.execution.time)

    for fill in fills_sorted:
        contract = getattr(fill, 'contract', None)
        con_id = getattr(contract, 'conId', None)
        ib_contract = contracts.get(con_id)
        if ib_contract is None:
            continue

        execu = fill.execution
        account = getattr(execu, 'acctNumber', '') or ''
        key = (account, con_id)
        summary = summaries.setdefault(
            key,
            ExecutionSummary(
                account=account,
                symbol=ib_contract.symbol,
                local_symbol=getattr(ib_contract, 'localSymbol', ib_contract.symbol),
                con_id=con_id,
                multiplier=100.0 if ib_contract.secType == 'OPT' else 1.0,
            ),
        )

        rs = fill.commissionReport or None
        realized = float(getattr(rs, 'realizedPNL', 0.0) or 0.0)
        commission = abs(float(getattr(rs, 'commission', 0.0) or 0.0))

        summary.realized += realized
        summary.commission += commission
        fill_time = getattr(execu, 'time', None)
        if fill_time is not None and fill_time.tzinfo is None:
            fill_time = fill_time.replace(tzinfo=timezone.utc)
        if fill_time is not None:
            if summary.last_fill_time is None or fill_time > summary.last_fill_time:
                summary.last_fill_time = fill_time
        summary.last_price = getattr(execu, 'price', summary.last_price)
        summary.events.append(
            {
                'time': (fill_time or datetime.utcnow().replace(tzinfo=timezone.utc)).isoformat(),
                'side': getattr(execu, 'side', None),
                'shares': getattr(execu, 'shares', None),
                'price': getattr(execu, 'price', None),
                'realized': realized,
                'commission': commission,
            }
        )

    total_realized = sum(s.realized for s in summaries.values())
    return list(summaries.values()), total_realized


def update_redis(r: redis.Redis, summaries: List[ExecutionSummary], total_realized: float) -> None:
    r.set('risk:daily_pnl', f'{total_realized:.2f}')
    r.set('risk:daily_pnl:reconciled_at', datetime.utcnow().isoformat())
    symbol_totals: Dict[str, float] = {}
    for summary in summaries:
        symbol_totals[summary.symbol] = symbol_totals.get(summary.symbol, 0.0) + summary.realized
    r.hset(
        'risk:daily_pnl:by_symbol',
        mapping={symbol: f'{value:.2f}' for symbol, value in symbol_totals.items()},
    )
    r.set('positions:pnl:realized:total', f'{total_realized:.2f}')


def rewrite_closed_positions(
    r: redis.Redis,
    summaries: List[ExecutionSummary],
    trading_day: str,
) -> int:
    updated = 0
    for summary in summaries:
        account = summary.account or ''
        if not account:
            continue

        redis_key = f'positions:closed:{trading_day}:{account}:{summary.con_id}'
        current_payload = r.get(redis_key)
        if not current_payload:
            continue

        position = json.loads(current_payload)
        position['realized_pnl'] = round(summary.realized, 2)
        position['commission'] = round(summary.commission, 2)
        if summary.last_price is not None:
            position['exit_price'] = summary.last_price
        if summary.last_fill_time is not None:
            position['exit_time'] = summary.last_fill_time.isoformat()

        buy_events = [
            ev for ev in summary.events
            if str(ev.get('side', '')).upper() == 'BOT' and ev.get('price') is not None
        ]
        if buy_events:
            first_buy = min(
                buy_events,
                key=lambda ev: datetime.fromisoformat(ev['time'].replace('Z', '+00:00')),
            )
            position['entry_price'] = first_buy['price']
            position['entry_time'] = first_buy['time']

        contract = dict(position.get('contract') or {})
        contract['symbol'] = summary.symbol
        contract['localSymbol'] = summary.local_symbol
        contract['conId'] = summary.con_id
        position['contract'] = contract

        if summary.events:
            position['realized_events'] = summary.events
        position.pop('realized_unposted', None)

        ttl = r.ttl(redis_key)
        payload = json.dumps(position)
        if ttl and ttl > 0:
            r.setex(redis_key, ttl, payload)
        else:
            r.set(redis_key, payload)
        updated += 1

    if updated:
        # Nudge aggregate totals to align with reconciled values
        r.set('positions:pnl:realized:total', f'{sum(s.realized for s in summaries):.2f}')
    return updated


def print_report(summaries: List[ExecutionSummary], total_realized: float) -> None:
    if not summaries:
        print('No executions found for today.')
        return

    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    rows = []
    for s in summaries:
        rows.append(
            [
                s.account or 'N/A',
                s.symbol,
                s.local_symbol,
                s.con_id,
                f'{s.multiplier:.0f}',
                f'{s.realized:.2f}',
                f'{s.commission:.2f}',
                f'{s.realized:.2f}',
            ]
        )

    print()
    if tabulate:
        print(tabulate(
            rows,
            headers=['Account', 'Symbol', 'Local', 'ConId', 'Mult', 'Realized P&L', 'Commission', 'Net P&L'],
            tablefmt='github',
        ))
    else:
        header = ['Account', 'Symbol', 'Local', 'ConId', 'Mult', 'Realized', 'Commission', 'Net']
        print(' | '.join(header))
        for row in rows:
            print(' | '.join(str(col) for col in row))
    print()
    total_commission = sum(s.commission for s in summaries)
    print(f'Total realized (IB net): ${total_realized:.2f}')
    print(f'Total commissions:       ${total_commission:.2f}')


async def main(apply_updates: bool) -> None:
    config = load_config()

    ib_conf = config.get('ibkr', {})
    redis_conf = config.get('redis', {})

    host = ib_conf.get('host', '127.0.0.1')
    port = ib_conf.get('port', 7497)
    client_id = ib_conf.get('client_id', 1) + 500  # use offset to avoid clashes

    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id, timeout=ib_conf.get('timeout', 10))

    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    start_et, _ = eastern_trading_day_bounds(now_et)
    start_utc = start_et.astimezone(pytz.utc)

    fills = await fetch_fills(ib, start_utc)
    if not fills:
        print('No executions returned by IBKR for today.')
        ib.disconnect()
        return

    contracts = await load_contracts(ib, fills)
    ib.disconnect()

    summaries, total_realized = build_summaries(fills, contracts)
    print_report(summaries, total_realized)

    if apply_updates:
        r = redis.Redis(
            host=redis_conf.get('host', '127.0.0.1'),
            port=redis_conf.get('port', 6379),
            db=redis_conf.get('db', 0),
            password=redis_conf.get('password'),
            decode_responses=True,
        )
        update_redis(r, summaries, total_realized)
        rewrote = rewrite_closed_positions(r, summaries, now_et.strftime('%Y%m%d'))
        print('\nRedis metrics updated.')
        if rewrote:
            print(f'Rewrote {rewrote} closed position snapshots.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconcile IBKR executions and update Redis P&L metrics.')
    parser.add_argument('--apply', action='store_true', help='Persist reconciled totals back into Redis.')
    args = parser.parse_args()

    asyncio.run(main(args.apply))
