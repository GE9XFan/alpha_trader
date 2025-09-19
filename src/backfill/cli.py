"""Command-line interface for replaying analytics backfills."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import redis.asyncio as aioredis
import yaml

from .jobs import DealerFlowBackfillJob, FlowClusterBackfillJob, VixBackfillJob
from .providers import (
    JsonlDealerFlowProvider,
    JsonlFlowSliceProvider,
    JsonlVixProvider,
)
from .runner import BackfillRunner
from .state import RedisCheckpointStore
from .utils import ensure_utc, parse_timestamp
from dealer_flow_calculator import DealerFlowCalculator
from flow_clustering import FlowClusterModel
from volatility_metrics import VolatilityMetrics

DEFAULT_CONFIG_PATH = Path('config/config.yaml')


async def run_backfill_async(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    redis_url = args.redis_url or _build_redis_url(config.get('redis', {}))
    redis = aioredis.from_url(redis_url, decode_responses=False)

    checkpoint_store = RedisCheckpointStore(redis) if args.checkpoint else None

    jobs = []

    dealer_flow_provider = JsonlDealerFlowProvider(Path(args.dealer_flow_jsonl)) if args.dealer_flow_jsonl else None
    flow_slice_provider = JsonlFlowSliceProvider(Path(args.flow_jsonl)) if args.flow_jsonl else None
    vix_provider = JsonlVixProvider(Path(args.vix_jsonl)) if args.vix_jsonl else None

    if dealer_flow_provider:
        dealer_calc = DealerFlowCalculator(config, redis)
        jobs.append(
            DealerFlowBackfillJob(
                redis_conn=redis,
                calculator=dealer_calc,
                provider=dealer_flow_provider,
                checkpoint_store=checkpoint_store,
                sleep_interval=args.sleep,
            )
        )

    if flow_slice_provider:
        flow_model = FlowClusterModel(config, redis)
        jobs.append(
            FlowClusterBackfillJob(
                redis_conn=redis,
                model=flow_model,
                provider=flow_slice_provider,
                checkpoint_store=checkpoint_store,
                sleep_interval=args.sleep,
            )
        )

    if vix_provider:
        vol_metrics = VolatilityMetrics(config, redis)
        jobs.append(
            VixBackfillJob(
                redis_conn=redis,
                metrics=vol_metrics,
                provider=vix_provider,
                checkpoint_store=checkpoint_store,
                sleep_interval=args.sleep,
            )
        )

    if not jobs:
        raise SystemExit('No backfill jobs configured. Provide at least one data source.')

    runner = BackfillRunner(jobs)

    symbols = _parse_symbols(args.symbols)
    start_ts = parse_timestamp(args.start) if args.start else None
    end_ts = parse_timestamp(args.end) if args.end else None

    def _progress(job_name: str, symbol: str, ts: datetime) -> None:
        if args.quiet:
            return
        print(f"[{job_name}] {symbol} -> {ensure_utc(ts).isoformat()}")

    results = await runner.run(
        symbols,
        start=start_ts,
        end=end_ts,
        progress_callback=_progress,
    )

    for result in results:
        _emit_summary(result, quiet=args.quiet)

    await redis.aclose()
    return 0


def _emit_summary(result, *, quiet: bool) -> None:
    if quiet:
        return
    earliest = result.earliest_snapshot.isoformat() if result.earliest_snapshot else 'n/a'
    latest = result.latest_snapshot.isoformat() if result.latest_snapshot else 'n/a'
    errors = f", errors={len(result.errors)}" if result.errors else ''
    print(
        f"Job={result.job} symbols={','.join(result.symbols)} snapshots={result.snapshots_processed} "
        f"metrics={result.metrics_written} window=({earliest} -> {latest}){errors}"
    )
    if result.errors:
        for line in result.errors:
            print(f"  ! {line}")


def _build_redis_url(config: dict) -> str:
    host = config.get('host', '127.0.0.1')
    port = int(config.get('port', 6379))
    db = int(config.get('db', 0))
    password = config.get('password')
    auth = f":{password}@" if password else ''
    return f"redis://{auth}{host}:{port}/{db}"


def _parse_symbols(raw: Optional[str]) -> Sequence[str]:
    if raw is None:
        return []
    return [token.strip().upper() for token in raw.split(',') if token.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='QuantiCity Capital Phase 5 backfill runner')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='Path to config.yaml')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbol list (optional for VIX only)')
    parser.add_argument('--start', type=str, help='Inclusive ISO timestamp for backfill window')
    parser.add_argument('--end', type=str, help='Inclusive ISO timestamp for backfill window')
    parser.add_argument('--dealer-flow-jsonl', type=str, help='NDJSON file containing dealer-flow snapshots')
    parser.add_argument('--flow-jsonl', type=str, help='NDJSON file containing flow clustering trade slices')
    parser.add_argument('--vix-jsonl', type=str, help='NDJSON file containing VIX observations')
    parser.add_argument('--redis-url', type=str, help='Redis URL override (redis://user:pass@host:port/db)')
    parser.add_argument('--checkpoint', action='store_true', help='Persist checkpoints to Redis for resuming')
    parser.add_argument('--sleep', type=float, default=0.0, help='Sleep interval between snapshots (seconds)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress and summary output')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run_backfill_async(args))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        return 130


if __name__ == '__main__':  # pragma: no cover - CLI guard
    sys.exit(main())
