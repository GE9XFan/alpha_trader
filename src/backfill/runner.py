"""Orchestration helpers for coordinating multiple backfill jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Sequence

from .jobs import BaseBackfillJob
from .models import BackfillResult


class BackfillRunner:
    """Execute a collection of backfill jobs in sequence."""

    def __init__(self, jobs: Iterable[BaseBackfillJob]) -> None:
        self.jobs: List[BaseBackfillJob] = list(jobs)

    async def run(
        self,
        symbols: Sequence[str],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        progress_callback=None,
    ) -> List[BackfillResult]:
        results: List[BackfillResult] = []
        for job in self.jobs:
            result = await job.run(
                symbols,
                start=start,
                end=end,
                progress=progress_callback,
            )
            results.append(result)
        return results


__all__ = ['BackfillRunner']
