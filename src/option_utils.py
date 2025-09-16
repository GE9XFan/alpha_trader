"""Utility helpers for working with option contract metadata."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Union

import holidays
import pytz

EASTERN = pytz.timezone("US/Eastern")
_US_HOLIDAYS = holidays.US(observed=True, expand=True)


def is_trading_day(day: date) -> bool:
    """Return True when *day* is a regular US equity trading session."""

    return day.weekday() < 5 and day not in _US_HOLIDAYS


def ensure_trading_day(day: date) -> date:
    """Advance forward until the result falls on a trading day."""

    current = day
    while not is_trading_day(current):
        current += timedelta(days=1)
    return current


def advance_trading_days(day: date, days: int) -> date:
    """Advance *day* by ``days`` trading sessions (skipping weekends/holidays)."""

    current = ensure_trading_day(day)
    remaining = max(days, 0)
    while remaining > 0:
        current += timedelta(days=1)
        current = ensure_trading_day(current)
        remaining -= 1
    return current


def compute_expiry_from_dte(dte: int, *, now: Optional[datetime] = None) -> str:
    """Compute an IBKR-compatible expiry string (YYYYMMDD) for the given *dte*."""

    current = now.astimezone(EASTERN) if now else datetime.now(EASTERN)
    anchor = ensure_trading_day(current.date())
    market_close = current.replace(hour=16, minute=0, second=0, microsecond=0)
    if current >= market_close:
        anchor = advance_trading_days(anchor, 1)

    if dte <= 0:
        expiry = anchor
    else:
        expiry = advance_trading_days(anchor, dte)

    return expiry.strftime("%Y%m%d")


def normalize_expiry(
    value: Optional[Union[str, int, float, datetime, date]],
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Normalize arbitrary expiry inputs to IBKR's ``YYYYMMDD`` format."""

    if value is None:
        return fallback

    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")

    if isinstance(value, date):
        return value.strftime("%Y%m%d")

    if isinstance(value, (int, float)):
        value = str(int(value))

    value_str = str(value).strip()
    if not value_str:
        return fallback

    candidates = {value_str, value_str.replace("-", ""), value_str.replace("/", "")}

    for candidate in candidates:
        if len(candidate) == 8 and candidate.isdigit():
            return candidate
        if len(candidate) == 6 and candidate.isdigit():
            return f"20{candidate}"

    try:
        parsed = datetime.fromisoformat(value_str)
        return parsed.strftime("%Y%m%d")
    except ValueError:
        pass

    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%b %d %Y"):
        try:
            parsed = datetime.strptime(value_str, fmt)
            return parsed.strftime("%Y%m%d")
        except ValueError:
            continue

    return fallback


__all__ = [
    "advance_trading_days",
    "compute_expiry_from_dte",
    "ensure_trading_day",
    "is_trading_day",
    "normalize_expiry",
]
