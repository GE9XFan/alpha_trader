#!/usr/bin/env python3
"""Utilities for normalizing IBKR market data before it hits Redis."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, DefaultDict, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class DepthLevel:
    """Light-weight view of a DOM level used by :class:`IBKRDataProcessor`."""

    price: float
    size: int
    venue: str


VENUE_MAP: Dict[str, str] = {
    "NSDQ": "NSDQ",
    "NASDAQ": "NSDQ",
    "BATS": "BATS",
    "EDGX": "EDGX",
    "EDGEA": "EDGEA",
    "DRCTEDGE": "DRCTEDGE",
    "BYX": "BYX",
    "ARCA": "ARCA",
    "NYSE": "NYSE",
    "AMEX": "AMEX",
    "NYSENAT": "NYSENAT",
    "PSX": "PSX",
    "IEX": "IEX",
    "LTSE": "LTSE",
    "CHX": "CHX",
    "ISE": "ISE",
    "PEARL": "PEARL",
    "MEMX": "MEMX",
    "BEX": "BEX",
    "IBEOS": "IBEOS",
    "IBKRATS": "IBKRATS",
    "OVERNIGHT": "OVERNIGHT",
    "ISLAND": "NSDQ",
    "SMART": "SMART",
    "UNKNOWN": "UNKNOWN",
    "": "UNKNOWN",
}


class IBKRDataProcessor:
    """Shared normalization helpers for :mod:`ibkr_ingestion`.

    The ingestion loop orchestrates subscriptions and Redis writes, while the
    processor keeps the heavy numerical manipulation here so we can test it in
    isolation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        ingestion_config = config.get("modules", {}).get("data_ingestion", {})
        self.ttls: Dict[str, int] = ingestion_config.get("store_ttls", {})

        self.order_books: Dict[str, Dict[str, Any]] = {}
        self.aggregated_tob: Dict[str, Dict[str, Any]] = {}
        self.trades_buffer: DefaultDict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=5000)
        )
        self.bars_buffer: DefaultDict[str, DefaultDict[str, Deque[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=500))
        )
        self.last_trade: Dict[str, float] = {}
        self.last_tob_ts: Dict[str, int] = {}

        self.logger.debug("IBKRDataProcessor ready (ttls=%s)", self.ttls)

    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------
    def update_depth_book(self, symbol: str, exchange: str, ticker: Any) -> Optional[Dict[str, Any]]:
        """Return the normalized order book for ``symbol``/``exchange``.

        The method stores the processed DOM snapshot for reuse in subsequent
        aggregation calls. ``ticker`` is an :class:`ib_insync.Ticker` but we only
        rely on duck-typed attributes so the processor remains test friendly.
        """

        book_key = f"{symbol}:{exchange}"
        book = {
            "bids": self._extract_depth_levels(getattr(ticker, "domBids", [])),
            "asks": self._extract_depth_levels(getattr(ticker, "domAsks", [])),
            "timestamp": int(time.time() * 1000),
            "exchange": exchange,
        }

        if not book["bids"] and not book["asks"]:
            return None

        self.order_books[book_key] = book
        return book

    def compute_aggregated_tob(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return a best bid/ask view by scanning all cached books for ``symbol``."""

        related_books = [
            book
            for key, book in self.order_books.items()
            if key.startswith(f"{symbol}:")
        ]

        if not related_books:
            return None

        best_bid = 0.0
        best_ask = math.inf
        best_bid_exchange = ""
        best_ask_exchange = ""

        for book in related_books:
            if book["bids"]:
                top_bid = book["bids"][0]
                price = float(top_bid["price"])
                if price > best_bid:
                    best_bid = price
                    best_bid_exchange = top_bid.get("venue") or top_bid.get("mm") or book.get("exchange", "")

            if book["asks"]:
                top_ask = book["asks"][0]
                price = float(top_ask["price"])
                if price < best_ask:
                    best_ask = price
                    best_ask_exchange = top_ask.get("venue") or top_ask.get("mm") or book.get("exchange", "")

        if best_bid <= 0 or not math.isfinite(best_ask):
            return None

        now_ms = int(time.time() * 1000)
        last_ts = self.last_tob_ts.get(symbol, 0)
        if now_ms <= last_ts:
            return None

        last_price = self.last_trade.get(symbol)
        bid = round(best_bid, 6)
        ask = round(best_ask, 6)
        mid = round((bid + ask) / 2, 6)
        spread = round(ask - bid, 6)

        spread_bps: Optional[float] = None
        if bid > 0:
            spread_bps = round((spread / bid) * 10000, 2)

        numeric_values = [v for v in [bid, ask, mid, spread, spread_bps, last_price] if v is not None]
        if any(not math.isfinite(v) for v in numeric_values):
            self.logger.debug("Skipping aggregated TOB for %s due to non-finite values", symbol)
            return None

        payload = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last_price,
            "mid": mid,
            "spread": spread,
            "spread_bps": spread_bps,
            "bid_exchange": best_bid_exchange,
            "ask_exchange": best_ask_exchange,
            "timestamp": now_ms,
        }

        self.aggregated_tob[symbol] = payload
        self.last_tob_ts[symbol] = now_ms
        return payload

    # ------------------------------------------------------------------
    # Trade helpers
    # ------------------------------------------------------------------
    def prepare_trade_storage(self, symbol: str, trade: Dict[str, Any], ticker: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (last_price_payload, ticker_payload) for a trade update."""

        self.last_trade[symbol] = float(trade["price"])
        self.trades_buffer[symbol].append(trade)

        last_price_payload = {"price": trade["price"], "ts": trade["time"]}

        bid = self._safe_round(getattr(ticker, "bid", None))
        ask = self._safe_round(getattr(ticker, "ask", None))
        last_price = round(float(trade["price"]), 6)

        ticker_payload: Dict[str, Any] = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last_price,
            "volume": self._safe_int(getattr(ticker, "volume", 0)),
            "vwap": self._safe_round(getattr(ticker, "vwap", None)),
            "timestamp": trade["time"],
        }

        if bid is not None and ask is not None:
            mid = round((bid + ask) / 2, 6)
            spread = round(ask - bid, 6)
            spread_bps = round((spread / bid) * 10000, 2) if bid > 0 else None
            ticker_payload.update({"mid": mid, "spread": spread, "spread_bps": spread_bps})

        return last_price_payload, ticker_payload

    def build_quote_ticker(self, symbol: str, ticker: Any) -> Optional[Dict[str, Any]]:
        """Sanitize a top-of-book quote into the canonical ticker payload."""

        bid = self._safe_round(getattr(ticker, "bid", None))
        ask = self._safe_round(getattr(ticker, "ask", None))

        if bid is None or ask is None:
            return None

        last_price = self._safe_round(getattr(ticker, "last", None))
        if last_price is not None:
            self.last_trade[symbol] = last_price
        else:
            last_price = self.last_trade.get(symbol)

        mid = round((bid + ask) / 2, 6)
        spread = round(ask - bid, 6)
        spread_bps = round((spread / bid) * 10000, 2) if bid > 0 else None

        numeric_values = [v for v in [bid, ask, mid, spread, spread_bps, last_price] if v is not None]
        if any(not math.isfinite(v) for v in numeric_values):
            return None

        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last_price,
            "mid": mid,
            "spread": spread,
            "spread_bps": spread_bps,
            "timestamp": int(time.time() * 1000),
        }

    def record_bar(self, symbol: str, bar_data: Dict[str, Any], timeframe: str = "5s") -> None:
        """Append a bar sample to the local buffer for reuse in tests."""

        self.bars_buffer[symbol][timeframe].append(bar_data)

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear cached state, used during shutdown or tests."""

        self.order_books.clear()
        self.aggregated_tob.clear()
        self.trades_buffer.clear()
        self.bars_buffer.clear()
        self.last_trade.clear()
        self.last_tob_ts.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _extract_depth_levels(self, levels: Iterable[Any]) -> List[Dict[str, Any]]:
        extracted: List[Dict[str, Any]] = []
        for level in levels or []:
            price = self._safe_round(getattr(level, "price", None))
            size = self._safe_int(getattr(level, "size", 0))
            raw_venue = getattr(level, "marketMaker", "") or "UNKNOWN"
            venue = VENUE_MAP.get(raw_venue, "UNKNOWN")

            if price is None or size is None:
                continue

            extracted.append({"price": price, "size": size, "venue": venue, "mm": venue})

        return extracted

    @staticmethod
    def _safe_round(value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(number) or math.isinf(number):
            return None

        return round(number, 6)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
        return number
