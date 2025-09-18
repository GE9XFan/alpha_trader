"""Checkpoint stores for long-running backfill jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Optional

import redis.asyncio as aioredis

from .utils import ensure_utc, parse_timestamp


class CheckpointStore(ABC):
    """Persistence abstraction for backfill progress markers."""

    @abstractmethod
    async def load(self, job: str, symbol: str) -> Optional[datetime]:
        """Return the last processed timestamp or ``None``."""

    @abstractmethod
    async def save(self, job: str, symbol: str, timestamp: datetime) -> None:
        """Persist the latest processed timestamp."""


class InMemoryCheckpointStore(CheckpointStore):
    """Checkpoint store backed by an in-memory dictionary (tests)."""

    def __init__(self) -> None:
        self._state: Dict[tuple[str, str], datetime] = {}

    async def load(self, job: str, symbol: str) -> Optional[datetime]:
        return self._state.get((job, symbol))

    async def save(self, job: str, symbol: str, timestamp: datetime) -> None:
        self._state[(job, symbol)] = ensure_utc(timestamp)


class RedisCheckpointStore(CheckpointStore):
    """Checkpoint store that persists markers in Redis."""

    def __init__(self, redis_conn: aioredis.Redis, *, ttl_seconds: int = 7 * 24 * 3600) -> None:
        self.redis = redis_conn
        self.ttl = ttl_seconds

    async def load(self, job: str, symbol: str) -> Optional[datetime]:
        key = self._key(job, symbol)
        raw = await self.redis.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
        try:
            return parse_timestamp(raw)
        except Exception:
            return None

    async def save(self, job: str, symbol: str, timestamp: datetime) -> None:
        key = self._key(job, symbol)
        payload = ensure_utc(timestamp).isoformat()
        await self.redis.setex(key, self.ttl, payload)

    def _key(self, job: str, symbol: str) -> str:
        symbol_slug = symbol.replace(':', '_')
        return f"backfill:checkpoint:{job}:{symbol_slug}"


__all__ = [
    'CheckpointStore',
    'InMemoryCheckpointStore',
    'RedisCheckpointStore',
]
