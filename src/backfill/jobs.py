"""Core backfill jobs for dealer-flow, flow clustering, and VIX1D metrics."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Callable, Optional, Sequence

import redis.asyncio as aioredis

from logging_utils import get_logger

import redis_keys as rkeys
from redis_keys import get_ttl

from .models import BackfillResult, DealerFlowSnapshot, FlowSlice, VixSnapshot
from .providers import (
    AbstractDealerFlowProvider,
    AbstractFlowSliceProvider,
    AbstractVixProvider,
)
from .state import CheckpointStore
from .utils import ensure_utc

from dealer_flow_calculator import DealerFlowCalculator
from flow_clustering import FlowClusterModel
from volatility_metrics import VolatilityMetrics

ProgressCallback = Callable[[str, str, datetime], None]


class BaseBackfillJob(ABC):
    """Common scaffolding for all backfill jobs."""

    name: str

    def __init__(
        self,
        *,
        redis_conn: aioredis.Redis,
        checkpoint_store: Optional[CheckpointStore] = None,
        sleep_interval: float = 0.0,
    ) -> None:
        self.redis = redis_conn
        self.checkpoint_store = checkpoint_store
        self.sleep_interval = sleep_interval
        self.logger = get_logger(__name__, component="backfill", subsystem=self.name)

    @abstractmethod
    async def _iterate(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> AsyncIterator[object]:
        """Yield historical samples for ``symbol``."""

    @abstractmethod
    async def _process_sample(self, symbol: str, sample: object) -> bool:
        """Persist ``sample`` into Redis and update analytics, returning success."""

    async def run(
        self,
        symbols: Sequence[str],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> BackfillResult:
        start_dt = ensure_utc(start) if start else None
        end_dt = ensure_utc(end) if end else None

        result = BackfillResult(job=self.name, symbols=list(symbols))

        for symbol in symbols:
            resume_after: Optional[datetime] = None
            iter_start = start_dt

            if self.checkpoint_store:
                checkpoint = await self.checkpoint_store.load(self.name, symbol)
                if checkpoint is not None:
                    resume_after = ensure_utc(checkpoint)
                    if iter_start is None or resume_after > iter_start:
                        iter_start = resume_after

            async for sample in self._iterate(symbol, iter_start, end_dt):
                timestamp = ensure_utc(self._extract_timestamp(sample))
                if resume_after and timestamp <= resume_after:
                    continue
                try:
                    success = await self._process_sample(symbol, sample)
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.exception("Failed to process backfill sample", extra={
                        'job': self.name,
                        'symbol': symbol,
                        'timestamp': timestamp.isoformat(),
                    })
                    result.record_error(f"{symbol}:{timestamp.isoformat()} -> {exc}")
                    continue

                result.record_snapshot(timestamp)
                if success:
                    result.record_metric()

                if self.checkpoint_store:
                    await self.checkpoint_store.save(self.name, symbol, timestamp)

                if progress:
                    progress(self.name, symbol, timestamp)

                if self.sleep_interval:
                    await asyncio.sleep(self.sleep_interval)

        return result

    @staticmethod
    def _extract_timestamp(sample: object) -> datetime:
        if isinstance(sample, DealerFlowSnapshot):
            return sample.timestamp
        if isinstance(sample, FlowSlice):
            return sample.timestamp
        if isinstance(sample, VixSnapshot):
            return sample.timestamp
        raise TypeError(f"Unsupported backfill sample type: {type(sample)!r}")


class DealerFlowBackfillJob(BaseBackfillJob):
    """Replays historical option chains to regenerate dealer-flow analytics."""

    name = 'dealer_flow'

    def __init__(
        self,
        *,
        redis_conn: aioredis.Redis,
        calculator: DealerFlowCalculator,
        provider: AbstractDealerFlowProvider,
        checkpoint_store: Optional[CheckpointStore] = None,
        sleep_interval: float = 0.0,
        chain_ttl: Optional[int] = None,
        ticker_ttl: Optional[int] = None,
    ) -> None:
        super().__init__(
            redis_conn=redis_conn,
            checkpoint_store=checkpoint_store,
            sleep_interval=sleep_interval,
        )
        self.calculator = calculator
        self.provider = provider
        self.chain_ttl = int(chain_ttl if chain_ttl is not None else get_ttl('options_chain'))
        self.ticker_ttl = int(ticker_ttl if ticker_ttl is not None else get_ttl('market_ticker'))

    async def _iterate(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> AsyncIterator[DealerFlowSnapshot]:
        async for snapshot in self.provider.iter_snapshots(symbol, start, end):
            yield snapshot

    async def _process_sample(self, symbol: str, sample: object) -> bool:  # type: ignore[override]
        assert isinstance(sample, DealerFlowSnapshot)

        ticker_payload = dict(sample.ticker)
        ticker_payload.setdefault('timestamp', int(sample.timestamp.timestamp()))

        chain_payload = sample.chain
        if isinstance(chain_payload, bytes):
            chain_blob = chain_payload
        elif isinstance(chain_payload, str):
            chain_blob = chain_payload.encode('utf-8')
        else:
            chain_blob = json.dumps(chain_payload).encode('utf-8')

        ticker_blob = json.dumps(ticker_payload).encode('utf-8')

        chain_key = rkeys.options_chain_key(symbol)
        ticker_key = rkeys.market_ticker_key(symbol)

        async with self.redis.pipeline(transaction=False) as pipe:
            await pipe.setex(chain_key, self.chain_ttl, chain_blob)
            await pipe.setex(ticker_key, self.ticker_ttl, ticker_blob)
            await pipe.execute()

        metrics = await self.calculator.calculate_dealer_metrics(symbol)
        if isinstance(metrics, dict) and 'error' in metrics:
            self.logger.debug(
                "Dealer-flow calculator returned error during backfill",
                extra={'symbol': symbol, 'error': metrics.get('error')},
            )
            return False
        return True


class FlowClusterBackfillJob(BaseBackfillJob):
    """Replays historical trade slices to regenerate flow clusters."""

    name = 'flow_clusters'

    def __init__(
        self,
        *,
        redis_conn: aioredis.Redis,
        model: FlowClusterModel,
        provider: AbstractFlowSliceProvider,
        checkpoint_store: Optional[CheckpointStore] = None,
        sleep_interval: float = 0.0,
        trades_ttl: Optional[int] = None,
    ) -> None:
        super().__init__(
            redis_conn=redis_conn,
            checkpoint_store=checkpoint_store,
            sleep_interval=sleep_interval,
        )
        self.model = model
        self.provider = provider
        self.trades_ttl = int(trades_ttl if trades_ttl is not None else get_ttl('market_trades'))

    async def _iterate(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> AsyncIterator[FlowSlice]:
        async for slice_ in self.provider.iter_slices(symbol, start, end):
            yield slice_

    async def _process_sample(self, symbol: str, sample: object) -> bool:  # type: ignore[override]
        assert isinstance(sample, FlowSlice)

        trades_key = rkeys.market_trades_key(symbol)
        encoded_trades = [
            json.dumps(trade).encode('utf-8')
            for trade in sample.trades
        ]

        async with self.redis.pipeline(transaction=False) as pipe:
            await pipe.ltrim(trades_key, 1, 0)  # Clear existing window
            for payload in reversed(encoded_trades):
                await pipe.lpush(trades_key, payload)
            await pipe.expire(trades_key, self.trades_ttl)
            await pipe.execute()

        payload = await self.model.classify_flows(symbol)
        if isinstance(payload, dict) and payload.get('samples', 0) > 0:
            return True
        return False


class VixBackfillJob(BaseBackfillJob):
    """Replays historical VIX1D observations."""

    name = 'vix1d'

    def __init__(
        self,
        *,
        redis_conn: aioredis.Redis,
        metrics: VolatilityMetrics,
        provider: AbstractVixProvider,
        checkpoint_store: Optional[CheckpointStore] = None,
        sleep_interval: float = 0.0,
    ) -> None:
        super().__init__(
            redis_conn=redis_conn,
            checkpoint_store=checkpoint_store,
            sleep_interval=sleep_interval,
        )
        self.metrics = metrics
        self.provider = provider

    async def _iterate(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> AsyncIterator[VixSnapshot]:  # symbol unused
        async for snapshot in self.provider.iter_snapshots(start, end):
            yield snapshot

    async def run(
        self,
        symbols: Sequence[str],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> BackfillResult:
        # Treat VIX as global; ignore symbol enumeration but keep API parity.
        result = BackfillResult(job=self.name, symbols=list(symbols) or ['VIX1D'])
        start_dt = ensure_utc(start) if start else None
        end_dt = ensure_utc(end) if end else None

        async for snapshot in self._iterate('VIX1D', start_dt, end_dt):
            timestamp = ensure_utc(snapshot.timestamp)
            payload = await self.metrics.ingest_manual(
                snapshot.value,
                as_of=timestamp,
                source=snapshot.source,
            )
            result.record_snapshot(timestamp)
            if payload and 'error' not in payload:
                result.record_metric()
            if progress:
                progress(self.name, 'VIX1D', timestamp)
            if self.sleep_interval:
                await asyncio.sleep(self.sleep_interval)

        return result

    async def _process_sample(self, symbol: str, sample: object) -> bool:  # pragma: no cover - not used
        raise NotImplementedError("VixBackfillJob overrides run()")


__all__ = [
    'BackfillResult',
    'BaseBackfillJob',
    'DealerFlowBackfillJob',
    'FlowClusterBackfillJob',
    'VixBackfillJob',
]
