#!/usr/bin/env python3
"""Utility to clear Discord distribution queues prior to validation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import redis
import yaml


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_redis(config: dict) -> redis.Redis:
    redis_cfg = config.get('redis', {})
    return redis.Redis(
        host=redis_cfg.get('host', '127.0.0.1'),
        port=int(redis_cfg.get('port', 6379)),
        db=int(redis_cfg.get('db', 0)),
        password=redis_cfg.get('password'),
        decode_responses=True,
    )


def queue_lengths(client: redis.Redis, queues: Iterable[str]) -> List[tuple[str, int]]:
    stats = []
    for name in queues:
        try:
            stats.append((name, client.llen(name)))
        except redis.RedisError:
            stats.append((name, -1))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear Discord distribution queues")
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH, help='Path to config.yaml')
    parser.add_argument('--confirm', action='store_true', help='Acknowledge destructive queue clear')
    parser.add_argument('--include-pending', action='store_true', help='Also clear signals:distribution:pending queue')
    args = parser.parse_args()

    if not args.confirm:
        print("⚠️  No action taken. Re-run with --confirm to clear queues.")
        return 1

    config = load_config(args.config)
    client = resolve_redis(config)

    queues = [
        'distribution:premium:queue',
        'distribution:basic:queue',
        'distribution:free:queue',
        'distribution:scheduled:basic',
        'distribution:scheduled:free',
    ]
    if args.include_pending:
        queues.append('signals:distribution:pending')

    stats_before = queue_lengths(client, queues)
    for queue, length in stats_before:
        state = f"{length} items" if length >= 0 else "unavailable"
        print(f"{queue:35s} {state}")

    client.delete(*queues)

    stats_after = queue_lengths(client, queues)
    print("\nPost-clear state:")
    for queue, length in stats_after:
        state = f"{length} items" if length >= 0 else "unavailable"
        print(f"{queue:35s} {state}")

    return 0


if __name__ == '__main__':  # pragma: no cover - utility entry point
    raise SystemExit(main())

