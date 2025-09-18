"""Data structures used by the backfill subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class DealerFlowSnapshot:
    """Normalized payload required to backfill dealer-flow analytics."""

    symbol: str
    timestamp: datetime
    chain: Any
    ticker: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowSlice:
    """Collection of recent trades required to backfill flow clustering."""

    symbol: str
    timestamp: datetime
    trades: Sequence[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VixSnapshot:
    """Single VIX1D observation used in volatility backfills."""

    timestamp: datetime
    value: float
    source: str = 'manual'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackfillResult:
    """Execution metadata returned by every backfill job."""

    job: str
    symbols: Sequence[str]
    snapshots_processed: int = 0
    metrics_written: int = 0
    earliest_snapshot: Optional[datetime] = None
    latest_snapshot: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    def record_snapshot(self, timestamp: datetime) -> None:
        if self.earliest_snapshot is None or timestamp < self.earliest_snapshot:
            self.earliest_snapshot = timestamp
        if self.latest_snapshot is None or timestamp > self.latest_snapshot:
            self.latest_snapshot = timestamp
        self.snapshots_processed += 1

    def record_metric(self) -> None:
        self.metrics_written += 1

    def record_error(self, message: str) -> None:
        self.errors.append(message)
