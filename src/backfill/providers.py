"""Historical data providers feeding the backfill jobs."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Optional, Sequence

from .models import DealerFlowSnapshot, FlowSlice, VixSnapshot
from .utils import in_range, parse_timestamp


class AbstractDealerFlowProvider(ABC):
    """Source of historical dealer-flow snapshots."""

    @abstractmethod
    async def iter_snapshots(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[DealerFlowSnapshot]:
        """Yield snapshots for ``symbol`` within ``[start, end]``."""


class AbstractFlowSliceProvider(ABC):
    """Source of historical trade slices for flow clustering."""

    @abstractmethod
    async def iter_slices(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[FlowSlice]:
        """Yield trade windows for ``symbol`` within ``[start, end]``."""


class AbstractVixProvider(ABC):
    """Source of historical VIX1D observations."""

    @abstractmethod
    async def iter_snapshots(
        self,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[VixSnapshot]:
        """Yield VIX observations within ``[start, end]``."""


class InMemoryDealerFlowProvider(AbstractDealerFlowProvider):
    """Simple provider backed by an in-memory sequence."""

    def __init__(self, snapshots: Iterable[DealerFlowSnapshot]):
        self._snapshots: List[DealerFlowSnapshot] = sorted(
            snapshots,
            key=lambda snap: (snap.symbol, snap.timestamp),
        )

    async def iter_snapshots(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[DealerFlowSnapshot]:
        start_dt = parse_timestamp(start) if start is not None else None
        end_dt = parse_timestamp(end) if end is not None else None
        for snapshot in self._snapshots:
            if snapshot.symbol != symbol:
                continue
            if not in_range(snapshot.timestamp, start_dt, end_dt):
                continue
            yield snapshot


class InMemoryFlowSliceProvider(AbstractFlowSliceProvider):
    """In-memory provider for flow clustering backfills."""

    def __init__(self, slices: Iterable[FlowSlice]):
        self._slices: List[FlowSlice] = sorted(
            slices,
            key=lambda item: (item.symbol, item.timestamp),
        )

    async def iter_slices(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[FlowSlice]:
        start_dt = parse_timestamp(start) if start is not None else None
        end_dt = parse_timestamp(end) if end is not None else None
        for slice_ in self._slices:
            if slice_.symbol != symbol:
                continue
            if not in_range(slice_.timestamp, start_dt, end_dt):
                continue
            yield slice_


class InMemoryVixProvider(AbstractVixProvider):
    """In-memory provider for VIX backfills."""

    def __init__(self, snapshots: Iterable[VixSnapshot]):
        self._snapshots: List[VixSnapshot] = sorted(
            snapshots,
            key=lambda snap: snap.timestamp,
        )

    async def iter_snapshots(
        self,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[VixSnapshot]:
        start_dt = parse_timestamp(start) if start is not None else None
        end_dt = parse_timestamp(end) if end is not None else None
        for snapshot in self._snapshots:
            if not in_range(snapshot.timestamp, start_dt, end_dt):
                continue
            yield snapshot


class _JsonlLoader:
    """Utility mixin for reading newline-delimited JSON files asynchronously."""

    def __init__(self, path: Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)

    async def _load(self) -> List[Dict[str, object]]:
        def _read() -> List[Dict[str, object]]:
            records: List[Dict[str, object]] = []
            with self._path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records

        return await asyncio.to_thread(_read)


class JsonlDealerFlowProvider(_JsonlLoader, AbstractDealerFlowProvider):
    """Dealer-flow provider backed by an NDJSON file."""

    def __init__(self, path: Path):
        super().__init__(path)

    async def iter_snapshots(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[DealerFlowSnapshot]:
        start_dt = parse_timestamp(start) if start is not None else None
        end_dt = parse_timestamp(end) if end is not None else None
        records = await self._load()
        for record in records:
            record_symbol = record.get('symbol')
            if record_symbol != symbol:
                continue
            timestamp = parse_timestamp(record['timestamp'])
            if not in_range(timestamp, start_dt, end_dt):
                continue
            chain = record.get('chain')
            ticker = record.get('ticker') or {}
            extras = {
                key: value
                for key, value in record.items()
                if key not in {'symbol', 'timestamp', 'chain', 'ticker'}
            }
            yield DealerFlowSnapshot(
                symbol=record_symbol,
                timestamp=timestamp,
                chain=chain,
                ticker=ticker,
                metadata=extras,
            )


class JsonlFlowSliceProvider(_JsonlLoader, AbstractFlowSliceProvider):
    """Flow clustering provider backed by an NDJSON file."""

    def __init__(self, path: Path):
        super().__init__(path)

    async def iter_slices(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[FlowSlice]:
        start_dt = parse_timestamp(start) if start is not None else None
        end_dt = parse_timestamp(end) if end is not None else None
        records = await self._load()
        for record in records:
            record_symbol = record.get('symbol')
            if record_symbol != symbol:
                continue
            timestamp = parse_timestamp(record['timestamp'])
            if not in_range(timestamp, start_dt, end_dt):
                continue
            trades = record.get('trades') or []
            extras = {
                key: value
                for key, value in record.items()
                if key not in {'symbol', 'timestamp', 'trades'}
            }
            yield FlowSlice(
                symbol=record_symbol,
                timestamp=timestamp,
                trades=list(trades),
                metadata=extras,
            )


class JsonlVixProvider(_JsonlLoader, AbstractVixProvider):
    """VIX history provider backed by an NDJSON file."""

    def __init__(self, path: Path):
        super().__init__(path)

    async def iter_snapshots(
        self,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> AsyncIterator[VixSnapshot]:
        start_dt = parse_timestamp(start) if start is not None else None
        end_dt = parse_timestamp(end) if end is not None else None
        records = await self._load()
        for record in records:
            timestamp = parse_timestamp(record['timestamp'])
            if not in_range(timestamp, start_dt, end_dt):
                continue
            value = float(record['value'])
            source = str(record.get('source', 'manual'))
            extras = {
                key: value
                for key, value in record.items()
                if key not in {'timestamp', 'value', 'source'}
            }
            yield VixSnapshot(
                timestamp=timestamp,
                value=value,
                source=source,
                metadata=extras,
            )


__all__ = [
    'AbstractDealerFlowProvider',
    'AbstractFlowSliceProvider',
    'AbstractVixProvider',
    'InMemoryDealerFlowProvider',
    'InMemoryFlowSliceProvider',
    'InMemoryVixProvider',
    'JsonlDealerFlowProvider',
    'JsonlFlowSliceProvider',
    'JsonlVixProvider',
]
