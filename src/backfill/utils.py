"""Utility helpers for backfill workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union


def ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_timestamp(value: Union[str, int, float, datetime]) -> datetime:
    """Parse an ISO8601/epoch timestamp into a UTC datetime."""

    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if not isinstance(value, str):
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    text = value.strip()
    if text.endswith('Z'):
        text = text[:-1]
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=timezone.utc)

    dt = datetime.fromisoformat(text)
    return ensure_utc(dt)


def in_range(ts: datetime, start: Optional[datetime], end: Optional[datetime]) -> bool:
    """Return ``True`` if ``ts`` falls within ``[start, end]`` (both inclusive)."""

    if start is not None and ts < start:
        return False
    if end is not None and ts > end:
        return False
    return True
