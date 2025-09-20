"""
Execution Manager Module

Manages order execution through IBKR (Interactive Brokers). Handles order
placement, monitoring, fills, and stop loss management.

Redis Keys Used:
    Read:
        - risk:halt:status (trading halt status)
        - account:value (account net liquidation)
        - account:buying_power (available buying power)
        - risk:daily_pnl (daily P&L for risk checks)
        - risk:new_positions_allowed (risk manager gate)
        - risk:position_size_multiplier (position sizing adjustment)
        - signals:execution:{symbol} (execution signals queue)
        - positions:open:{symbol}:* (open position tracking)
    Write:
        - health:execution:heartbeat (health monitoring)
        - execution:connection:status (IBKR connection status)
        - orders:pending:{order_id} (pending order tracking)
        - positions:open:{symbol}:{position_id} (new positions)
        - positions:by_symbol:{symbol} (symbol index)
        - execution:fills:total (fill counter)
        - execution:fills:daily (daily fill counter)
        - execution:commission:total (commission tracking)
        - orders:fills:{date} (fill history)
        - execution:rejections:* (rejection tracking)
        - alerts:critical (critical alerts)
        - contracts:qualified:* (qualified contract cache)

Author: QuantiCity Capital
Version: 3.0.0
"""

import asyncio
import inspect
import json
import math
import time
import uuid
import re
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
import redis.asyncio as aioredis
from ib_insync import (
    IB,
    Stock,
    Option,
    MarketOrder,
    LimitOrder,
    StopOrder,
    Order,
    ExecutionFilter,
    util,
)

from signal_deduplication import contract_fingerprint
from src.option_utils import normalize_expiry
from src.stop_engine import StopEngine, StopState
from src.kelly_sizer import KellySizer
from logging_utils import get_logger


class RiskManager:
    """Forward declaration for RiskManager - imported at runtime to avoid circular imports."""
    pass


class ExecutionManager:
    """
    Manage order execution through IBKR.
    Handles order placement, monitoring, and fills.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        redis_conn: aioredis.Redis,
        risk_manager: Optional["RiskManager"] = None,
        risk_manager_factory: Optional[Callable[[], Awaitable["RiskManager"]]] = None,
    ):
        """
        Initialize execution manager with configuration.
        Uses separate IBKR connection for execution (different from data connection).
        """
        self.config = config
        self.redis = redis_conn
        self.ib = IB()

        # Risk manager injection/caching
        self._risk_manager: Optional["RiskManager"] = risk_manager
        self._risk_manager_factory = risk_manager_factory
        self._risk_manager_lock = asyncio.Lock()

        # IBKR configuration for execution
        self.ibkr_config = config.get('ibkr', {})
        self.host = self.ibkr_config.get('host', '127.0.0.1')
        self.port = self.ibkr_config.get('port', 7497)  # Paper trading port
        self.client_id = self.ibkr_config.get('client_id', 1) + 100  # Different client ID for execution

        # Position limits from config (read from risk_management section)
        risk_config = config.get('risk_management', {})
        self.max_positions = risk_config.get('max_positions', 10)
        self.max_per_symbol = risk_config.get('max_per_symbol', 5)
        self.max_0dte_contracts = risk_config.get('max_0dte_contracts', 50)
        self.max_other_contracts = risk_config.get('max_other_contracts', 100)
        self.stop_engine = StopEngine.from_risk_config(risk_config)

        # Order management
        self.pending_orders: Dict[int, Dict[str, Any]] = {}  # order_id -> order details
        self.stop_orders: Dict[str, Any] = {}  # position_id -> stop trade/order
        self.target_orders: Dict[str, List[Any]] = {}  # position_id -> list of target trades
        self.active_trades: Dict[int, Dict[str, Any]] = {}  # order_id -> {'trade': trade, 'signal': signal}
        self.bracket_registry: Dict[int, Dict[str, Any]] = {}
        self.position_brackets: Dict[str, Dict[str, Any]] = {}
        self._order_watchers: Dict[int, asyncio.Task] = {}

        # Connection state
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        self.logger = get_logger(__name__, component="execution", subsystem="manager")

        # Tracking for manual fills to detect missing execution events
        self._manual_fill_positions: Set[str] = set()
        self._missing_manual_fill_alerted: Set[str] = set()

        # Live account / P&L telemetry
        self.account_state: Dict[str, Any] = {
            'account': None,
            'net_liquidation': 0.0,
            'buying_power': 0.0,
            'excess_liquidity': 0.0,
            'total_cash_value': 0.0,
        }
        self.pnl_state: Dict[str, float] = {
            'daily': 0.0,
            'realized': 0.0,
            'unrealized': 0.0,
        }
        self._pnl_req_id: Optional[int] = None
        self._pnl_single_req_ids: Set[int] = set()
        self._pnl_single_handles: Dict[int, int] = {}

        # Kelly sizing helper
        self.kelly_sizer = KellySizer(redis_conn, config)

        # Market data cache for trailing logic
        self.position_tickers: Dict[str, Any] = {}
        self._price_cache: Dict[str, float] = {}

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._events_registered = False

    async def _scan_keys(self, pattern: str, *, count: int = 200) -> List[str]:
        """Return all keys matching *pattern* using non-blocking SCAN."""
        cursor = 0
        results: List[str] = []
        while True:
            cursor, batch = await self.redis.scan(cursor=cursor, match=pattern, count=count)
            if batch:
                # aioredis returns str when decode_responses=True
                results.extend(batch)
            if cursor == 0:
                break
        return results

    @staticmethod
    def _normalize_strategy(strategy: Optional[str]) -> str:
        """Return a normalized uppercase strategy code for consistent checks."""
        if strategy is None:
            return ''
        return str(strategy).strip().upper()

    @staticmethod
    def _extract_commission(fill) -> float:
        """Return the absolute commission cost for a fill."""
        commission_report = getattr(fill, 'commissionReport', None)
        raw = getattr(commission_report, 'commission', 0) if commission_report else 0
        try:
            return abs(float(raw or 0.0))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Internal helpers for position reconciliation
    # ------------------------------------------------------------------
    def _canonical_contract_key(
        self,
        symbol: Optional[str],
        contract: Any,
        side: Optional[str] = None,
    ) -> tuple:
        """Return a normalized key so we can match IB and Redis positions."""

        contract_type: Optional[str]
        expiry: Optional[str]
        strike: Optional[float]
        right: Optional[str]

        if isinstance(contract, dict):
            contract_type = contract.get('type') or contract.get('secType')
            expiry = contract.get('expiry') or contract.get('lastTradeDateOrContractMonth')
            strike = contract.get('strike')
            right = contract.get('right')
        else:
            contract_type = getattr(contract, 'secType', None)
            expiry = getattr(contract, 'lastTradeDateOrContractMonth', None)
            strike = getattr(contract, 'strike', None)
            right = getattr(contract, 'right', None)

        normalized_symbol = (symbol or '').upper()
        normalized_type = (str(contract_type).lower() if contract_type else '')
        normalized_expiry = str(expiry) if expiry else ''
        normalized_right = str(right).upper() if right else ''

        normalized_strike: Optional[float]
        try:
            normalized_strike = float(strike) if strike not in (None, '') else None
        except (TypeError, ValueError):
            normalized_strike = None

        normalized_side = (side or '').upper()

        return (
            normalized_symbol,
            normalized_type,
            normalized_expiry,
            normalized_strike,
            normalized_right,
            normalized_side,
        )

    def _contract_payload_from_ib(self, contract: Any) -> Dict[str, Any]:
        """Convert an IB contract object to the Redis-friendly payload."""

        payload = {
            'symbol': getattr(contract, 'symbol', None),
            'secType': getattr(contract, 'secType', None),
            'exchange': getattr(contract, 'exchange', None),
            'currency': getattr(contract, 'currency', None),
        }

        sec_type = getattr(contract, 'secType', None)
        if sec_type == 'OPT':
            payload.update(
                {
                    'type': 'option',
                    'expiry': getattr(contract, 'lastTradeDateOrContractMonth', None),
                    'strike': getattr(contract, 'strike', None),
                    'right': getattr(contract, 'right', None),
                }
            )
        else:
            payload['type'] = str(sec_type or 'STK').lower()

        return payload

    def _normalize_avg_cost(self, contract: Any, avg_cost: float) -> float:
        """Normalize IB average cost (options are reported in cents)."""

        sec_type = getattr(contract, 'secType', None)
        if sec_type == 'OPT':
            return avg_cost / 100.0
        return avg_cost

    @staticmethod
    def _safe_price(value: Any) -> Optional[float]:
        """Return a finite positive float price when available."""

        try:
            price = float(value)
        except (TypeError, ValueError):
            return None

        if price <= 0 or math.isnan(price) or math.isinf(price):
            return None

        return price

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            result = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result

    def _create_position_snapshot_from_ib(
        self,
        position_id: str,
        contract: Any,
        size: float,
        avg_cost: float,
        account: str,
        existing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build or update a Redis position snapshot from live IB data."""

        symbol = getattr(contract, 'symbol', None)
        side = 'LONG' if size >= 0 else 'SHORT'
        entry_price = self._normalize_avg_cost(contract, avg_cost)

        if existing is not None:
            snapshot = existing
        else:
            snapshot = {
                'id': position_id,
                'symbol': symbol,
                'commission': 0.0,
                'stop_loss': None,
                'targets': [],
                'current_price': entry_price,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'status': 'OPEN',
                'strategy': 'SYNCED',
            }

        snapshot['id'] = position_id
        snapshot['symbol'] = symbol
        snapshot['contract'] = self._contract_payload_from_ib(contract)
        snapshot['side'] = side
        snapshot['quantity'] = abs(size)
        snapshot['entry_price'] = entry_price
        snapshot.setdefault('entry_time', datetime.utcnow().isoformat())
        snapshot.setdefault('current_price', entry_price)
        snapshot.setdefault('unrealized_pnl', 0.0)
        snapshot.setdefault('realized_pnl', 0.0)
        snapshot['reconciled'] = True
        snapshot['ib_account'] = account
        snapshot['ib_con_id'] = getattr(contract, 'conId', None)

        return snapshot

    def _merge_position_metadata(
        self,
        source: Dict[str, Any],
        destination: Dict[str, Any],
    ) -> None:
        """Preserve stop/target metadata when consolidating duplicate entries."""

        fields_to_merge = [
            'stop_order_id',
            'target_order_ids',
            'oca_group',
            'targets',
            'stop_loss',
            'strategy',
            'order_id',
            'signal_id',
            'commission',
            'entry_time',
            'current_price',
            'unrealized_pnl',
            'realized_pnl',
            'status',
        ]

        for field in fields_to_merge:
            if field not in source:
                continue

            if field == 'target_order_ids':
                existing = destination.get('target_order_ids') or []
                merged = existing + [oid for oid in source.get('target_order_ids') or [] if oid not in existing]
                if merged:
                    destination['target_order_ids'] = merged
                continue

            if field == 'targets':
                existing_targets = destination.get('targets') or []
                if not existing_targets and source.get('targets'):
                    destination['targets'] = source['targets']
                continue

            if destination.get(field) in (None, '', [], {}):
                destination[field] = source[field]

    async def _remove_duplicate_position(self, symbol: str, position_id: str) -> None:
        """Delete a duplicate Redis position record and clean indexes."""

        await self.redis.delete(f'positions:open:{symbol}:{position_id}')
        removed = await self.redis.srem(f'positions:by_symbol:{symbol}', position_id)
        if removed:
            await self._decrement_position_count(symbol, removed)

        self.stop_orders.pop(position_id, None)
        self.target_orders.pop(position_id, None)

    async def _increment_position_count(self, amount: int = 1) -> None:
        """Increase the cached open-position count."""
        if amount <= 0:
            return
        await self.redis.incrby('positions:count', amount)

    async def _decrement_position_count(self, symbol: str, amount: int = 1) -> None:
        """Decrease the cached open-position count and clean empty symbol sets."""
        if amount <= 0:
            return

        new_value = await self.redis.decrby('positions:count', amount)
        if new_value < 0:
            await self.redis.set('positions:count', 0)

        members = await self.redis.scard(f'positions:by_symbol:{symbol}')
        if members == 0:
            await self.redis.delete(f'positions:by_symbol:{symbol}')

    # ------------------------------------------------------------------
    # Position lifecycle helpers
    # ------------------------------------------------------------------
    def _position_multiplier(self, position: Dict[str, Any]) -> int:
        contract_type = position.get('contract', {}).get('type', '').lower()
        return 100 if contract_type == 'option' else 1

    def _weighted_entry_price(
        self,
        current_qty: float,
        current_price: float,
        fill_qty: float,
        fill_price: float,
    ) -> float:
        try:
            total_qty = current_qty + fill_qty
            if total_qty <= 0:
                return fill_price
            weighted = ((current_qty * current_price) + (fill_qty * fill_price)) / total_qty
            return round(weighted, 6)
        except Exception:
            return fill_price

    @staticmethod
    def _compute_holding_minutes(entry_time: Optional[str], exit_time: Optional[str]) -> Optional[float]:
        if not entry_time or not exit_time:
            return None

        try:
            start = datetime.fromisoformat(entry_time)
            end = datetime.fromisoformat(exit_time)
        except ValueError:
            return None

        delta_seconds = (end - start).total_seconds()
        if delta_seconds < 0:
            return None
        return round(delta_seconds / 60.0, 2)

    @staticmethod
    def _classify_result(realized_pnl: Optional[float]) -> str:
        if realized_pnl is None:
            return 'FLAT'
        if realized_pnl > 0:
            return 'WIN'
        if realized_pnl < 0:
            return 'LOSS'
        return 'FLAT'

    def _calculate_return_pct(
        self,
        position: Dict[str, Any],
        realized_pnl: Optional[float],
        closed_quantity: float,
    ) -> Optional[float]:
        try:
            base_notional = float(position.get('position_notional') or 0.0)
        except (TypeError, ValueError):
            base_notional = 0.0

        if base_notional <= 0:
            entry_price = 0.0
            try:
                entry_price = float(position.get('entry_price') or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            multiplier = self._position_multiplier(position)
            base_notional = abs(entry_price) * abs(closed_quantity) * multiplier

        if base_notional <= 0:
            return None

        try:
            pnl_value = float(realized_pnl or 0.0)
        except (TypeError, ValueError):
            return None

        return pnl_value / base_notional

    @staticmethod
    def _map_event_to_status(event_type: str) -> str:
        mapping = {
            'EXIT': 'CLOSED',
            'SCALE_OUT': 'PARTIAL',
            'ENTRY': 'FILLED',
        }
        key = (event_type or '').upper()
        return mapping.get(key, key)

    async def _publish_lifecycle_event(
        self,
        position: Dict[str, Any],
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        if not self.config.get('modules', {}).get('signals', {}).get('enabled', False):
            return

        signal_id = position.get('signal_id') or position.get('id')
        if not signal_id:
            return

        lifecycle = dict(details or {})
        execution_snapshot = {
            'status': self._map_event_to_status(event_type),
            'avg_fill_price': lifecycle.get('fill_price') or lifecycle.get('exit_price'),
            'filled_quantity': lifecycle.get('quantity'),
            'executed_at': lifecycle.get('executed_at') or lifecycle.get('timestamp'),
        }
        execution_snapshot = {
            key: value for key, value in execution_snapshot.items() if value is not None
        }

        payload = {
            'id': signal_id,
            'symbol': position.get('symbol'),
            'side': position.get('side'),
            'strategy': self._normalize_strategy(position.get('strategy')),
            'action_type': event_type,
            'ts': int(time.time() * 1000),
            'position_id': position.get('id'),
            'position_notional': position.get('position_notional'),
            'entry': position.get('entry_price'),
            'stop': position.get('stop_loss'),
            'targets': position.get('targets', []),
            'contract': position.get('contract'),
            'lifecycle': lifecycle,
        }

        if execution_snapshot:
            payload['execution'] = execution_snapshot

        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            await self.redis.lpush('signals:distribution:pending', json.dumps(payload))
        except Exception as exc:
            self.logger.error(
                "Failed to enqueue execution lifecycle payload",
                extra={'position_id': position.get('id'), 'event': event_type, 'error': str(exc)}
            )

    async def _publish_exit_distribution(
        self,
        position: Dict[str, Any],
        exit_price: Optional[float],
        closed_quantity: float,
        reason: str,
    ) -> None:
        symbol = position.get('symbol')
        position_id = position.get('id')
        if not symbol or not position_id:
            return

        payload = {
            'id': position.get('signal_id') or position_id,
            'symbol': symbol,
            'side': position.get('side'),
            'strategy': self._normalize_strategy(position.get('strategy')),
            'action_type': 'EXIT',
            'ts': int(time.time() * 1000),
            'position_id': position_id,
            'position_notional': position.get('position_notional'),
            'entry': position.get('entry_price'),
            'stop': position.get('stop_loss'),
            'targets': position.get('targets', []),
            'contract': position.get('contract'),
        }

        exit_time = position.get('exit_time') or datetime.utcnow().isoformat()
        realized_total = position.get('realized_pnl', 0.0)
        return_pct = self._calculate_return_pct(position, realized_total, closed_quantity)
        holding_minutes = self._compute_holding_minutes(position.get('entry_time'), exit_time)

        lifecycle = {
            'executed_at': exit_time,
            'exit_price': round(float(exit_price), 4) if exit_price is not None else None,
            'quantity': closed_quantity,
            'realized_pnl': round(float(realized_total or 0.0), 2),
            'return_pct': return_pct,
            'holding_period_minutes': holding_minutes,
            'reason': reason,
            'result': self._classify_result(realized_total),
            'remaining_quantity': position.get('quantity', 0),
        }

        execution_snapshot = {
            'status': 'FILLED',
            'avg_fill_price': round(float(exit_price), 4) if exit_price is not None else None,
            'filled_quantity': closed_quantity,
            'executed_at': exit_time,
            'order_id': position.get('order_id'),
            'position_id': position_id,
        }

        lifecycle = {k: v for k, v in lifecycle.items() if v is not None}
        execution_snapshot = {k: v for k, v in execution_snapshot.items() if v is not None}
        if lifecycle:
            payload['lifecycle'] = lifecycle
        if execution_snapshot:
            payload['execution'] = execution_snapshot

        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            self.logger.info(
                "exit_distribution_enqueued",
                extra={
                    "action": "exit_distribution",
                    "symbol": symbol,
                    "position_id": position_id,
                    "reason": reason,
                    "quantity": closed_quantity,
                    "realized": lifecycle.get('realized_pnl'),
                },
            )
            await self.redis.lpush('signals:distribution:pending', json.dumps(payload))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to enqueue exit distribution payload",
                extra={'position_id': position_id, 'error': str(exc), 'symbol': symbol}
            )

    async def _publish_realized_metrics(self, amount: float) -> None:
        """Update realized PnL aggregates when we recognize ``amount``."""

        if not amount:
            return

        try:
            await self.redis.incrbyfloat('execution:pnl:realized:delta', amount)
            await self.redis.incrbyfloat('execution:pnl:daily_delta', amount)
        except Exception as exc:
            self.logger.error(
                "realized_metrics_update_failed",
                extra={"action": "realized_metrics", "amount": amount, "error": str(exc)},
            )
            raise

    async def _register_realized_event(
        self,
        position: Dict[str, Any],
        realized_delta: float,
    ) -> None:
        """Accumulate realized PnL on ``position`` and publish metrics."""

        if realized_delta in (None, 0):
            return

        prior_total = float(position.get('realized_pnl') or 0.0)
        new_total = prior_total + float(realized_delta)
        position['realized_pnl'] = new_total

        posted_total = float(position.get('realized_posted', 0.0) or 0.0)
        unposted_total = float(position.get('realized_unposted', 0.0) or 0.0)

        try:
            await self._publish_realized_metrics(float(realized_delta))
        except Exception:
            position['realized_unposted'] = unposted_total + float(realized_delta)
        else:
            position['realized_posted'] = posted_total + float(realized_delta)
            # Clear any previously unposted balance that we just recognised
            remaining_unposted = unposted_total - float(realized_delta)
            position['realized_unposted'] = remaining_unposted if remaining_unposted > 0 else 0.0

    def _build_pending_signal_stub(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct a minimal signal payload for pending orders after restarts."""

        return {
            'id': order_data.get('signal_id'),
            'symbol': order_data.get('symbol'),
            'strategy': order_data.get('strategy'),
            'side': order_data.get('side'),
            'contract': order_data.get('contract'),
            'position_size': order_data.get('position_notional'),
            'targets': order_data.get('targets') or [],
            'stop': order_data.get('stop_loss'),
            'confidence': order_data.get('confidence'),
        }

    async def _load_pending_orders_from_redis(self) -> None:
        """Warm ``self.pending_orders`` from persisted Redis snapshots."""

        keys = await self._scan_keys('orders:pending:*')
        for key in keys:
            payload = await self.redis.get(key)
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                self.logger.error(
                    "pending_order_load_failed",
                    extra={"action": "load_pending_order", "key": key},
                )
                continue

            order_identifier = record.get('order_id') or record.get('orderId')
            if order_identifier in (None, ''):
                continue

            try:
                order_id = int(order_identifier)
            except (TypeError, ValueError):
                continue

            record['order_id'] = order_id
            record.setdefault('placed_at', time.time())
            self.pending_orders[order_id] = record

    async def _hydrate_open_orders(self) -> None:
        """Reconcile IB open orders with local caches after (re)connect."""

        if not self.ib.isConnected():
            return

        try:
            await self.ib.reqOpenOrdersAsync()
        except Exception as exc:
            self.logger.warning(
                "open_orders_hydration_failed",
                extra={"action": "hydrate_open_orders", "error": str(exc)},
            )
            return

        trade_lookup: Dict[int, Any] = {}
        for trade in self.ib.trades():
            order = getattr(trade, 'order', None)
            if not order or order.orderId is None:
                continue
            trade_lookup[order.orderId] = trade

        # Rebuild pending order watchers and registry
        for order_id, order_data in list(self.pending_orders.items()):
            stub = self._build_pending_signal_stub(order_data)
            order_data.setdefault('signal', stub)
            trade = trade_lookup.get(order_id)

            if trade:
                self.active_trades[order_id] = {'trade': trade, 'signal': stub}
                if order_id not in self._order_watchers:
                    watcher = asyncio.create_task(self.monitor_order(trade, stub))
                    self._order_watchers[order_id] = watcher
                    watcher.add_done_callback(
                        lambda task, oid=order_id: self._order_watchers.pop(oid, None)
                    )
            else:
                self.logger.debug(
                    "pending_order_trade_missing",
                    extra={"action": "hydrate_pending_order", "order_id": order_id},
                )

            stop_id = order_data.get('stop_order_id')
            target_ids = order_data.get('target_order_ids') or []
            stop_trade = trade_lookup.get(int(stop_id)) if stop_id not in (None, '') else None
            if stop_trade and getattr(stop_trade, 'isDone', lambda: True)():
                stop_trade = None

            target_trades = []
            for target_id in target_ids:
                try:
                    target_int = int(target_id)
                except (TypeError, ValueError):
                    continue
                trade_obj = trade_lookup.get(target_int)
                if trade_obj and not getattr(trade_obj, 'isDone', lambda: True)():
                    target_trades.append(trade_obj)

            self.bracket_registry[order_id] = {
                'parent_trade': trade,
                'stop_trade': stop_trade,
                'target_trades': target_trades,
                'stop_order_id': stop_id,
                'target_order_ids': target_ids,
                'stop_engine_state': order_data.get('stop_engine_state'),
                'plan_records': order_data.get('stop_engine_plan', []),
                'initial_stop': order_data.get('stop_loss'),
                'symbol': order_data.get('symbol'),
                'side': order_data.get('side'),
                'strategy': self._normalize_strategy(order_data.get('strategy')),
                'contract': order_data.get('contract'),
                'quantity': int(self._safe_float(order_data.get('size'))),
                'entry_reference': order_data.get('entry_target'),
                'basis_price': order_data.get('entry_target'),
                'oca_group': order_data.get('oca_group'),
            }

        # Rehydrate protection orders for open positions
        position_keys = await self._scan_keys('positions:open:*')
        for key in position_keys:
            payload = await self.redis.get(key)
            if not payload:
                continue
            try:
                position = json.loads(payload)
            except json.JSONDecodeError:
                continue

            position_id = position.get('id')
            if not position_id:
                continue

            stop_id = position.get('stop_order_id')
            target_ids = position.get('target_order_ids') or []

            stop_trade = None
            if stop_id not in (None, ''):
                try:
                    stop_trade = trade_lookup.get(int(stop_id))
                except (TypeError, ValueError):
                    stop_trade = None
            if stop_trade and getattr(stop_trade, 'isDone', lambda: True)():
                stop_trade = None

            target_trades = []
            for target_id in target_ids:
                try:
                    target_int = int(target_id)
                except (TypeError, ValueError):
                    continue
                trade_obj = trade_lookup.get(target_int)
                if trade_obj and not getattr(trade_obj, 'isDone', lambda: True)():
                    target_trades.append(trade_obj)

            if stop_trade:
                self.stop_orders[position_id] = stop_trade
            else:
                self.stop_orders.pop(position_id, None)

            if target_trades:
                self.target_orders[position_id] = target_trades
            else:
                self.target_orders.pop(position_id, None)

            bracket_entry = {
                'parent_trade': None,
                'stop_trade': stop_trade,
                'target_trades': target_trades,
                'stop_order_id': stop_id,
                'target_order_ids': target_ids,
                'stop_engine_state': position.get('stop_engine_state'),
                'plan_records': position.get('stop_engine_plan', []),
                'initial_stop': position.get('stop_loss'),
                'symbol': position.get('symbol'),
                'side': position.get('side'),
                'strategy': self._normalize_strategy(position.get('strategy')),
                'contract': position.get('contract'),
                'quantity': int(self._safe_float(position.get('quantity'))),
                'entry_reference': position.get('basis_price') or position.get('entry_price'),
                'basis_price': position.get('basis_price') or position.get('entry_price'),
                'oca_group': position.get('oca_group'),
            }
            self.position_brackets[position_id] = bracket_entry

            await self._ensure_position_protection(position, trade_lookup)

    async def _ensure_position_protection(
        self,
        position: Dict[str, Any],
        trade_lookup: Optional[Dict[int, Any]] = None,
    ) -> None:
        """Ensure ``position`` has an active protective bracket."""

        position_id = position.get('id')
        if not position_id:
            return

        quantity = self._safe_float(position.get('quantity'))
        if quantity <= 0:
            return

        stop_trade = self.stop_orders.get(position_id)
        if not stop_trade:
            stop_id = position.get('stop_order_id')
            if trade_lookup and stop_id not in (None, ''):
                try:
                    stop_trade = trade_lookup.get(int(stop_id))
                except (TypeError, ValueError):
                    stop_trade = None
            if stop_trade and getattr(stop_trade, 'isDone', lambda: True)():
                stop_trade = None

        if stop_trade:
            # Cache refresh for active stop orders
            self.stop_orders[position_id] = stop_trade
            return

        if not self.ib.isConnected():
            return

        try:
            position_copy = dict(position)
            await self._rebuild_bracket(position_copy)
        except Exception as exc:
            self.logger.warning(
                "position_protection_rebuild_failed",
                extra={
                    "action": "rebuild_bracket",
                    "position_id": position_id,
                    "symbol": position.get('symbol'),
                    "error": str(exc),
                },
            )

    async def _prime_execution_stream(self) -> None:
        """Ensure IBKR sends execution events for manual trades."""

        if not self.ib.isConnected():
            return

        try:
            executions = await self.ib.reqExecutionsAsync(ExecutionFilter())
            count = len(executions) if executions is not None else 0
            self.logger.info(
                "execution_stream_primed",
                extra={"action": "prime_executions", "count": count},
            )
        except Exception as exc:
            self.logger.warning(
                "execution_stream_prime_failed",
                extra={"action": "prime_executions", "error": str(exc)},
            )

    def _register_event_streams(self) -> None:
        """Register IB event callbacks once per connection."""

        if self._events_registered:
            return

        self.ib.accountSummaryEvent += self._on_account_summary
        self.ib.pnlEvent += self._on_pnl_update
        self.ib.pnlSingleEvent += self._on_pnl_single_update
        self.ib.positionEvent += self._on_position_event
        self._events_registered = True

    async def _subscribe_account_summary(self, account_code: str) -> None:
        """Subscribe to ongoing account summary updates."""
        try:
            await self.ib.reqAccountSummaryAsync()
        except Exception as exc:
            self.logger.warning(
                "account_summary_subscription_failed",
                extra={"action": "account_summary", "error": str(exc)},
            )

    async def _subscribe_pnl_streams(self, account_code: str) -> None:
        """Subscribe to account-level P&L streams."""
        try:
            pnl = self.ib.reqPnL(account_code, '')
            self._pnl_req_id = getattr(pnl, 'reqId', None)
        except Exception as exc:
            self.logger.warning(
                "pnl_subscription_failed",
                extra={"action": "pnl", "error": str(exc)},
            )

    def _on_account_summary(self, account: str, tag: str, value: str, currency: str) -> None:
        """Process account summary events and propagate to Redis."""
        tag = tag or ''
        value_float = self._safe_float(value)
        if tag == 'NetLiquidation':
            self.account_state['net_liquidation'] = value_float
            asyncio.create_task(self.redis.set('account:value', value))
        elif tag == 'BuyingPower':
            self.account_state['buying_power'] = value_float
            asyncio.create_task(self.redis.set('account:buying_power', value))
        elif tag == 'ExcessLiquidity':
            self.account_state['excess_liquidity'] = value_float
        elif tag == 'TotalCashValue':
            self.account_state['total_cash_value'] = value_float

        if account and not self.account_state.get('account'):
            self.account_state['account'] = account

    def _on_pnl_update(self, req_id: int, daily_pnl: float, unrealized_pnl: float, realized_pnl: float) -> None:
        """Handle aggregate P&L updates."""
        self.pnl_state.update(
            {
                'daily': daily_pnl or 0.0,
                'unrealized': unrealized_pnl or 0.0,
                'realized': realized_pnl or 0.0,
            }
        )

        async def _update_redis():
            await self.redis.set('risk:daily_pnl', f"{self.pnl_state['daily']:.2f}")
            await self.redis.set('positions:pnl:unrealized', f"{self.pnl_state['unrealized']:.2f}")
            await self.redis.set('positions:pnl:realized:total', f"{self.pnl_state['realized']:.2f}")

        asyncio.create_task(_update_redis())

    def _on_pnl_single_update(self, pnl_single) -> None:
        """Update Redis with contract-level PnL snapshots."""
        try:
            symbol = getattr(pnl_single, 'symbol', None)
            unrealized = getattr(pnl_single, 'unrealizedPnl', None)
            realized = getattr(pnl_single, 'realizedPnl', None)
            con_id = getattr(pnl_single, 'conId', None)
        except Exception:
            return

        if not symbol:
            return

        async def _update_redis():
            key = f'positions:pnl:by_symbol:{symbol}'
            payload = {
                'unrealized': unrealized,
                'realized': realized,
                'updated_at': datetime.utcnow().isoformat(),
            }
            await self.redis.set(key, json.dumps(payload))

        asyncio.create_task(_update_redis())

    def _ensure_pnl_single_subscription(self, account: Optional[str], con_id: Optional[int]) -> None:
        if not account or con_id in (None, ''):
            return
        try:
            con_id_int = int(con_id)
        except (TypeError, ValueError):
            return

        if con_id_int in self._pnl_single_handles:
            return

        try:
            pnl_single = self.ib.reqPnLSingle(account, '', con_id_int)
        except Exception as exc:
            self.logger.debug(
                "pnl_single_subscription_failed",
                extra={"action": "pnl_single", "account": account, "con_id": con_id_int, "error": str(exc)},
            )
            return

        req_id = getattr(pnl_single, 'reqId', None)
        if req_id is not None:
            self._pnl_single_handles[con_id_int] = req_id

    def _cancel_pnl_single_subscription(self, con_id: Optional[int]) -> None:
        if con_id in (None, ''):
            return
        try:
            con_id_int = int(con_id)
        except (TypeError, ValueError):
            return

        req_id = self._pnl_single_handles.pop(con_id_int, None)
        if req_id is None:
            return
        try:
            self.ib.cancelPnLSingle(req_id)
        except Exception as exc:
            self.logger.debug(
                "pnl_single_cancel_failed",
                extra={"action": "pnl_single_cancel", "con_id": con_id_int, "error": str(exc)},
            )

    def _on_position_event(self, account: str, contract: Any, position: float, avg_cost: float) -> None:
        """Synchronize Redis with live IB position updates."""
        if contract is None or not contract.conId:
            return

        symbol = getattr(contract, 'symbol', None)
        if not symbol:
            return

        position_id = f"{account}:{contract.conId}"
        snapshot = self._create_position_snapshot_from_ib(
            position_id,
            contract,
            position,
            avg_cost,
            account,
        )

        async def _update():
            redis_key = f'positions:open:{symbol}:{position_id}'
            await self.redis.set(redis_key, json.dumps(snapshot))
            await self.redis.sadd(f'positions:by_symbol:{symbol}', position_id)

        asyncio.create_task(_update())

        if abs(position) > 0:
            self._ensure_pnl_single_subscription(account, contract.conId)
        else:
            self._cancel_pnl_single_subscription(contract.conId)

    def _ensure_monitor_loop(self) -> None:
        """Spawn trailing stop monitor loop if not already running."""
        if self._monitor_task and not self._monitor_task.done():
            return
        self._monitor_task = asyncio.create_task(self._trailing_monitor_loop())

    async def _trailing_monitor_loop(self) -> None:
        while True:
            try:
                await self._evaluate_trailing_stops()
            except Exception as exc:
                self.logger.error(
                    "trailing_monitor_error",
                    extra={"action": "trailing_monitor", "error": str(exc)},
                )
            await asyncio.sleep(2.0)

    async def _evaluate_trailing_stops(self) -> None:
        position_keys = await self._scan_keys('positions:open:*')
        for key in position_keys:
            payload = await self.redis.get(key)
            if not payload:
                continue
            try:
                position = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if str(position.get('status', 'OPEN')).upper() != 'OPEN':
                continue

            quantity = self._safe_float(position.get('quantity'))
            if quantity <= 0:
                continue

            symbol = position.get('symbol')
            if not symbol:
                continue

            contract_info = position.get('contract') or {}
            await self._ensure_market_data_subscription(symbol, contract_info)

            current_price = self._get_market_price(symbol, position)
            if current_price is None:
                continue

            entry_price = self._safe_float(position.get('entry_price'))
            if entry_price <= 0:
                continue

            side = (position.get('side') or '').upper()
            contract_type = contract_info.get('type') or contract_info.get('secType') or 'stock'
            state = StopState.from_dict(position.get('stop_engine_state'))
            current_stop = position.get('stop_loss')
            current_stop = self._safe_float(current_stop) if current_stop is not None else None

            decision = self.stop_engine.evaluate(
                contract_type,
                side,
                entry_price,
                current_price,
                current_stop=current_stop,
                state=state,
            )

            if decision is None:
                continue

            if not decision.should_update or decision.stop_price is None:
                continue

            success = await self._update_trailing_stop(position, decision.stop_price, decision.state)
            if not success:
                continue

            position['stop_loss'] = decision.stop_price
            position['stop_engine_state'] = decision.state.to_dict()
            position['stop_engine_plan'] = position.get('stop_engine_plan', [])
            position['last_update'] = time.time()
            position['current_price'] = current_price

            await self.redis.set(key, json.dumps(position))

    async def _ensure_market_data_subscription(self, symbol: str, contract_info: Dict[str, Any]) -> None:
        if symbol in self.position_tickers:
            return

        if not self.ib.isConnected():
            return

        try:
            contract = await self.create_ib_contract({**contract_info, 'symbol': symbol})
        except Exception:
            contract = None
        if not contract:
            return

        try:
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.position_tickers[symbol] = ticker
        except Exception as exc:
            self.logger.debug(f"Failed to subscribe market data for {symbol}: {exc}")

    def _get_market_price(self, symbol: str, position: Dict[str, Any]) -> Optional[float]:
        ticker = self.position_tickers.get(symbol)
        price: Optional[float] = None
        if ticker is not None:
            for attr in ('last', 'midpoint', 'marketPrice', 'close'):
                candidate = getattr(ticker, attr, None)
                if candidate:
                    price = float(candidate)
                    break
            if not price:
                bid = getattr(ticker, 'bid', None)
                ask = getattr(ticker, 'ask', None)
                if bid and ask:
                    price = (float(bid) + float(ask)) / 2

        if price:
            self._price_cache[symbol] = price
            return price

        cached = self._price_cache.get(symbol)
        if cached:
            return cached

        try:
            return float(position.get('current_price') or 0.0) or None
        except (TypeError, ValueError):
            return None

    def _drop_market_data_subscription(self, symbol: str) -> None:
        ticker = self.position_tickers.pop(symbol, None)
        if ticker is not None:
            try:
                self.ib.cancelMktData(ticker)
            except Exception:
                pass
        self._price_cache.pop(symbol, None)

    async def _maybe_release_market_data(self, symbol: Optional[str]) -> None:
        if not symbol:
            return
        try:
            open_count = await self.redis.scard(f'positions:by_symbol:{symbol}')
        except Exception:
            open_count = 0
        if open_count and open_count > 0:
            return

        for pending in self.pending_orders.values():
            if isinstance(pending, dict) and pending.get('symbol') == symbol:
                return

        self._drop_market_data_subscription(symbol)

    async def _update_trailing_stop(
        self,
        position: Dict[str, Any],
        stop_price: float,
        state: StopState,
    ) -> bool:
        position_id = position.get('id')
        symbol = position.get('symbol')
        if not position_id or not symbol:
            return False

        stop_trade = self.stop_orders.get(position_id)
        if not stop_trade:
            return False

        close_action = 'SELL' if (position.get('side') or '').upper() == 'LONG' else 'BUY'
        quantity = int(round(self._safe_float(position.get('quantity'))))
        if quantity <= 0:
            return False

        contract_info = position.get('contract') or {}
        try:
            ib_contract = await self.create_ib_contract({**contract_info, 'symbol': symbol})
        except Exception:
            ib_contract = None
        if not ib_contract:
            return False

        existing_order = getattr(stop_trade, 'order', None)
        if existing_order is None or existing_order.orderId is None:
            return False

        order_id = existing_order.orderId
        stop_price = round(float(stop_price), 4)

        if state.last_stop_type == 'TRAIL_PERCENT':
            order = Order()
            order.orderType = 'TRAIL'
            order.trailingPercent = round(state.last_stop_value or 0.0, 6)
            order.trailStopPrice = stop_price
        else:
            order = StopOrder(close_action, quantity, stop_price)

        order.action = close_action
        order.totalQuantity = quantity
        order.orderId = order_id
        order.parentId = getattr(existing_order, 'parentId', None)
        order.ocaGroup = position.get('oca_group')
        order.ocaType = getattr(existing_order, 'ocaType', 1)
        order.tif = 'GTC'
        order.outsideRth = getattr(existing_order, 'outsideRth', True)
        order.transmit = True

        try:
            updated_trade = self.ib.placeOrder(ib_contract, order)
            self.stop_orders[position_id] = updated_trade
            bracket = self.position_brackets.get(position_id)
            if bracket is not None:
                bracket['stop_trade'] = updated_trade
                bracket['stop_order_id'] = getattr(updated_trade.order, 'orderId', order_id)
                bracket['stop_engine_state'] = state.to_dict()
                bracket['initial_stop'] = stop_price
                bracket['quantity'] = quantity
            position['stop_order_id'] = getattr(updated_trade.order, 'orderId', order_id)
            position['stop_engine_state'] = state.to_dict()
            return True
        except Exception as exc:
            self.logger.warning(
                "trailing_stop_update_failed",
                extra={
                    "action": "stop_update",
                    "position_id": position_id,
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            return False

    def _build_protection_plan(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        contract_info: Dict[str, Any],
        strategy_code: str,
        signal_targets: Optional[List[float]],
        state_override: Optional[StopState] = None,
    ) -> Tuple[float, StopState, List[Dict[str, Any]], str]:
        instrument_type = contract_info.get('type') or contract_info.get('secType') or 'default'
        state = state_override if state_override else StopState.from_dict(None)

        stop_price, state = self.stop_engine.initial_stop(
            instrument_type,
            side,
            entry_price,
            state=state,
        )

        allocations = self.stop_engine.plan_targets(
            instrument_type,
            side,
            entry_price,
            int(quantity),
            filled_targets=state.filled_targets,
        )

        plan_records: List[Dict[str, Any]] = []
        if allocations:
            for alloc in allocations:
                plan_records.append(
                    {
                        'profit_pct': alloc.profit_pct,
                        'price': round(float(alloc.price), 4),
                        'fraction': alloc.fraction,
                        'quantity': int(alloc.quantity),
                        'status': 'pending',
                        'order_id': None,
                    }
                )
        else:
            raw_targets = [
                round(float(target), 4)
                for target in (signal_targets or [])
                if self._safe_float(target) > 0
            ]

            if raw_targets and quantity > 0:
                target_count = len(raw_targets)
                base = quantity // target_count
                remainder = quantity % target_count
                allocations = [base] * target_count
                for idx in range(remainder):
                    allocations[idx] += 1

                for price, qty in zip(raw_targets, allocations):
                    if qty <= 0:
                        continue
                    plan_records.append(
                        {
                            'profit_pct': None,
                            'price': price,
                            'fraction': qty / max(quantity, 1),
                            'quantity': qty,
                            'status': 'pending',
                            'order_id': None,
                        }
                    )

            if not plan_records:
                fallback_raw = self.stop_engine.target_price(side, entry_price, 30.0)
                fallback_rounded = self.stop_engine.quantize_price(
                    instrument_type,
                    side,
                    fallback_raw,
                )
                plan_records.append(
                    {
                        'profit_pct': 30.0,
                        'price': round(float(fallback_rounded), 4),
                        'fraction': 1.0,
                        'quantity': int(quantity),
                        'status': 'pending',
                        'order_id': None,
                    }
                )

        base_group = f"BRK_{symbol}_{strategy_code or 'GEN'}"
        sanitized_group = re.sub(r'[^A-Za-z0-9_]+', '_', base_group)
        oca_group = f"{sanitized_group[:58]}_{int(time.time())}"

        return stop_price, state, plan_records, oca_group

    async def _submit_bracket(
        self,
        *,
        ib_contract: Any,
        action: str,
        order_size: int,
        order_type: str,
        limit_price: Optional[float],
        stop_price: float,
        stop_state: StopState,
        plan_records: List[Dict[str, Any]],
        oca_group: str,
    ) -> Dict[str, Any]:
        parent_order: Order
        if order_type == 'MARKET':
            parent_order = MarketOrder(action, order_size)
        else:
            parent_order = LimitOrder(action, order_size, limit_price)
        parent_order.transmit = False
        parent_order.outsideRth = True

        parent_trade = self.ib.placeOrder(ib_contract, parent_order)
        parent_id = getattr(parent_trade.order, 'orderId', None)
        if parent_id is None:
            raise RuntimeError('Parent order ID not allocated by IBKR')

        close_action = 'SELL' if action == 'BUY' else 'BUY'
        target_trades: List[Any] = []
        target_order_ids: List[int] = []

        for plan in plan_records:
            qty = int(plan.get('quantity') or 0)
            price = float(plan.get('price') or 0)
            if qty <= 0 or price <= 0:
                continue

            limit_order = LimitOrder(close_action, qty, price)
            limit_order.parentId = parent_id
            limit_order.ocaGroup = oca_group
            limit_order.ocaType = 1
            limit_order.tif = 'GTC'
            limit_order.transmit = False
            trade = self.ib.placeOrder(ib_contract, limit_order)
            target_trades.append(trade)
            plan['order_id'] = getattr(trade.order, 'orderId', None)
            plan['status'] = 'working'
            if trade.order.orderId is not None:
                target_order_ids.append(trade.order.orderId)

        stop_price = round(float(stop_price), 4)
        if stop_state.last_stop_type == 'TRAIL_PERCENT':
            stop_order = Order()
            stop_order.orderType = 'TRAIL'
            stop_order.trailingPercent = round(stop_state.last_stop_value or 0.0, 6)
            stop_order.trailStopPrice = stop_price
        else:
            stop_order = StopOrder(close_action, order_size, stop_price)

        stop_order.parentId = parent_id
        stop_order.ocaGroup = oca_group
        stop_order.ocaType = 1
        stop_order.tif = 'GTC'
        stop_order.outsideRth = True
        stop_order.transmit = True

        stop_trade = self.ib.placeOrder(ib_contract, stop_order)
        stop_order_id = getattr(stop_trade.order, 'orderId', None)

        return {
            'parent_trade': parent_trade,
            'stop_trade': stop_trade,
            'target_trades': target_trades,
            'parent_order_id': parent_id,
            'stop_order_id': stop_order_id,
            'target_order_ids': target_order_ids,
            'plan_records': plan_records,
            'stop_state': stop_state.to_dict(),
            'initial_stop': stop_price,
            'oca_group': oca_group,
        }

    async def _rebuild_bracket(self, position: Dict[str, Any]) -> None:
        """Recreate protective stop/target orders for an open position."""

        symbol = position.get('symbol')
        position_id = position.get('id')
        if not symbol or not position_id:
            return

        quantity = int(round(self._safe_float(position.get('quantity'))))
        if quantity <= 0:
            return

        if not self.ib.isConnected():
            return

        contract_info = dict(position.get('contract') or {})
        contract_info['symbol'] = symbol

        ib_contract = await self.create_ib_contract(contract_info)
        if not ib_contract:
            self.logger.warning(f"Unable to rebuild bracket for {symbol} – contract lookup failed")
            return

        previous_bracket = self.position_brackets.get(position_id)
        previous_stop_trade = self.stop_orders.get(position_id)
        previous_target_trades = list(self.target_orders.get(position_id, []))
        previous_stop_order_id = position.get('stop_order_id')
        previous_target_order_ids = list(position.get('target_order_ids') or [])

        entry_price = self._safe_float(position.get('basis_price'))
        if entry_price <= 0:
            entry_price = self._safe_float(position.get('last_fill_price'))
        if entry_price <= 0:
            entry_price = self._safe_float(position.get('entry_price'))
        if entry_price <= 0:
            entry_price = self._safe_float(position.get('current_price')) or 1.0

        strategy_code = self._normalize_strategy(position.get('strategy'))
        side = (position.get('side') or '').upper()
        existing_state = StopState.from_dict(position.get('stop_engine_state'))

        stop_price, state, plan_records, oca_group = self._build_protection_plan(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            contract_info=contract_info,
            strategy_code=strategy_code,
            signal_targets=position.get('targets', []),
            state_override=existing_state,
        )

        # Ensure total allocation does not exceed available quantity
        allocated = sum(int(plan.get('quantity') or 0) for plan in plan_records)
        if allocated > quantity:
            excess = allocated - quantity
            for plan in reversed(plan_records):
                qty = int(plan.get('quantity') or 0)
                if qty <= 0:
                    continue
                deduction = min(qty, excess)
                plan['quantity'] = qty - deduction
                excess -= deduction
                if plan['quantity'] <= 0:
                    plan['status'] = 'skipped'
                if excess <= 0:
                    break
            plan_records = [plan for plan in plan_records if int(plan.get('quantity') or 0) > 0]

        close_action = 'SELL' if side == 'LONG' else 'BUY'

        target_trades: List[Any] = []
        target_order_ids: List[int] = []

        try:
            for plan in plan_records:
                qty = int(plan.get('quantity') or 0)
                price = float(plan.get('price') or 0)
                if qty <= 0 or price <= 0:
                    continue
                limit_order = LimitOrder(close_action, qty, price)
                limit_order.ocaGroup = oca_group
                limit_order.ocaType = 1
                limit_order.tif = 'GTC'
                limit_order.outsideRth = True
                limit_order.transmit = False
                trade = self.ib.placeOrder(ib_contract, limit_order)
                plan['order_id'] = getattr(trade.order, 'orderId', None)
                plan['status'] = 'working'
                target_trades.append(trade)
                if plan['order_id'] is not None:
                    target_order_ids.append(plan['order_id'])

            rounded_stop = round(float(stop_price), 4)
            if state.last_stop_type == 'TRAIL_PERCENT':
                stop_order = Order()
                stop_order.orderType = 'TRAIL'
                stop_order.trailingPercent = round(state.last_stop_value or 0.0, 6)
                stop_order.trailStopPrice = rounded_stop
            else:
                stop_order = StopOrder(close_action, quantity, rounded_stop)

            stop_order.ocaGroup = oca_group
            stop_order.ocaType = 1
            stop_order.tif = 'GTC'
            stop_order.outsideRth = True
            stop_order.transmit = True

            stop_trade = self.ib.placeOrder(ib_contract, stop_order)
            stop_order_id = getattr(stop_trade.order, 'orderId', None)
        except Exception as exc:
            for trade in target_trades:
                await self._cancel_trade_safe(trade, context=f"{position_id} rebuild_target")
            locals_stop_trade = locals().get('stop_trade')
            if locals_stop_trade:
                await self._cancel_trade_safe(locals_stop_trade, context=f"{position_id} rebuild_stop")
            self.logger.warning(
                "protective_orders_rebuild_failed",
                extra={
                    "action": "rebuild_bracket",
                    "symbol": symbol,
                    "position_id": position_id,
                    "error": str(exc),
                },
            )
            raise

        self.stop_orders[position_id] = stop_trade
        if target_trades:
            self.target_orders[position_id] = target_trades
        else:
            self.target_orders.pop(position_id, None)

        position['stop_order_id'] = stop_order_id
        position['target_order_ids'] = target_order_ids
        position['oca_group'] = oca_group
        position['stop_loss'] = rounded_stop
        position['stop_engine_state'] = state.to_dict()
        position['stop_engine_plan'] = plan_records
        position['targets'] = [plan.get('price') for plan in plan_records]
        position['last_update'] = time.time()

        await self.redis.set(f'positions:open:{symbol}:{position_id}', json.dumps(position))

        self.position_brackets[position_id] = {
            'parent_trade': None,
            'stop_trade': stop_trade,
            'target_trades': target_trades,
            'stop_order_id': stop_order_id,
            'target_order_ids': target_order_ids,
            'stop_engine_state': state.to_dict(),
            'plan_records': plan_records,
            'initial_stop': rounded_stop,
            'symbol': symbol,
            'side': side,
            'strategy': strategy_code,
            'contract': contract_info,
            'quantity': quantity,
            'entry_reference': entry_price,
            'oca_group': oca_group,
            'basis_price': entry_price,
        }

        if previous_stop_trade or previous_target_trades or previous_stop_order_id or previous_target_order_ids:
            await self._retire_previous_bracket(
                position_id,
                stop_trade=previous_stop_trade,
                target_trades=previous_target_trades,
                stop_order_id=previous_stop_order_id,
                target_order_ids=previous_target_order_ids,
            )

        self.logger.info(
            "protective_orders_rebuilt",
            extra={
                "action": "rebuild_bracket",
                "symbol": symbol,
                "position_id": position_id,
                "quantity": quantity,
                "stop_price": rounded_stop,
                "targets": position['targets'],
            },
        )



    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        """Best-effort conversion of various datetime representations to ``datetime``."""

        if isinstance(value, datetime):
            return value

        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value))
            except (OverflowError, OSError, ValueError):
                return None

        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            try:
                # Handle ISO format, including "Z" suffix
                return datetime.fromisoformat(normalized.replace('Z', '+00:00'))
            except ValueError:
                try:
                    return util.parseIBDatetime(normalized)
                except Exception:
                    return None

        return None

    async def _backfill_realized_from_ib(
        self,
        position: Dict[str, Any],
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Pull execution history for ``position`` and compute realized P&L."""

        if not self.ib.isConnected():
            return None

        account = position.get('ib_account')
        con_id = position.get('ib_con_id')
        if not account or con_id in (None, ''):
            return None

        try:
            con_id_int = int(con_id)
        except (TypeError, ValueError):
            return None

        filt = ExecutionFilter()
        filt.acctCode = account
        filt.conId = con_id_int

        entry_time = self._parse_datetime(position.get('entry_time'))
        if entry_time:
            filt.time = entry_time.strftime('%Y%m%d %H:%M:%S')

        try:
            executions = await self.ib.reqExecutionsAsync(filt)
        except Exception as exc:
            self.logger.warning(
                "execution_backfill_failed",
                extra={
                    "action": "execution_backfill",
                    "position_id": position.get('id'),
                    "symbol": position.get('symbol'),
                    "error": str(exc),
                },
            )
            return None

        if not executions:
            return None

        side = (position.get('side') or '').upper()
        entry_price = float(position.get('entry_price', 0.0) or 0.0)
        multiplier = self._position_multiplier(position)
        existing_commission = float(position.get('commission') or 0.0)

        total_realized = 0.0
        total_commission = existing_commission
        closed_quantity = 0.0
        weighted_exit = 0.0
        exit_time_obj: Optional[datetime] = None
        events: List[Dict[str, Any]] = []

        for detail in executions:
            contract = getattr(detail, 'contract', None)
            if contract is not None:
                detail_con_id = getattr(contract, 'conId', None)
                if detail_con_id not in (None, con_id_int):
                    continue

            execution = getattr(detail, 'execution', None)
            if execution is None:
                continue

            fill_side = str(getattr(execution, 'side', '')).upper()
            shares = abs(float(getattr(execution, 'shares', 0) or 0.0))
            price = float(getattr(execution, 'price', 0.0) or 0.0)
            if shares <= 0 or price <= 0:
                continue

            is_reduction = (
                (side == 'LONG' and fill_side == 'SLD') or
                (side == 'SHORT' and fill_side == 'BOT')
            )
            if not is_reduction:
                continue

            commission_report = getattr(detail, 'commissionReport', None)
            fill_commission = 0.0
            fill_realized: Optional[float] = None
            if commission_report is not None:
                try:
                    fill_commission = abs(float(getattr(commission_report, 'commission', 0.0) or 0.0))
                except (TypeError, ValueError):
                    fill_commission = 0.0

                realized_value = getattr(commission_report, 'realizedPNL', None)
                if realized_value is not None:
                    try:
                        fill_realized = float(realized_value)
                    except (TypeError, ValueError):
                        fill_realized = None

            if fill_realized is None:
                if side == 'LONG':
                    fill_realized = (price - entry_price) * shares * multiplier
                else:
                    fill_realized = (entry_price - price) * shares * multiplier

            total_realized += fill_realized
            total_commission += fill_commission
            closed_quantity += shares
            weighted_exit += shares * price

            timestamp_str = getattr(execution, 'time', None)
            timestamp_dt = util.parseIBDatetime(timestamp_str) if timestamp_str else None
            if timestamp_dt:
                exit_time_obj = (
                    max(exit_time_obj, timestamp_dt)
                    if exit_time_obj else timestamp_dt
                )

            events.append(
                {
                    'type': f'{reason}_BACKFILL',
                    'quantity': shares,
                    'price': price,
                    'commission': fill_commission,
                    'realized': fill_realized,
                    'timestamp': (timestamp_dt or datetime.utcnow()).isoformat(),
                    'exec_id': getattr(execution, 'execId', None),
                    'side': fill_side,
                }
            )

        if closed_quantity <= 0:
            return None

        exit_price = weighted_exit / closed_quantity if closed_quantity else None

        return {
            'realized': total_realized,
            'commission': total_commission,
            'exit_price': exit_price,
            'exit_time': exit_time_obj.isoformat() if exit_time_obj else None,
            'events': events,
            'closed_quantity': closed_quantity,
        }

    async def _cancel_bracket_orders(self, position: Dict[str, Any]) -> None:
        position_id = position.get('id')
        stop_trade = self.stop_orders.pop(position_id, None)
        if stop_trade and not stop_trade.isDone():
            try:
                self.ib.cancelOrder(stop_trade.order)
            except Exception as exc:
                self.logger.warning(f"Failed to cancel stop for {position_id}: {exc}")

        for trade in self.target_orders.pop(position_id, []):
            try:
                if trade and not trade.isDone():
                    self.ib.cancelOrder(trade.order)
            except Exception as exc:
                self.logger.warning(f"Failed to cancel target for {position_id}: {exc}")

        position['stop_order_id'] = None
        position['target_order_ids'] = []
        position['oca_group'] = None
        position['stop_engine_plan'] = []
        self.position_brackets.pop(position_id, None)

    async def _cancel_trade_safe(self, trade, *, context: str = "") -> None:
        """Cancel an IB trade object if it is still active."""
        if not trade:
            return

        try:
            if not trade.isDone():
                self.ib.cancelOrder(trade.order)
        except Exception as exc:
            detail = f" ({context})" if context else ""
            self.logger.warning(f"Failed to cancel trade{detail}: {exc}")

    async def _cancel_order_id_safe(self, order_id: Optional[int], *, context: str = "") -> None:
        """Attempt to cancel an order by ID even if we don't have the trade handle."""
        if order_id is None:
            return

        trade = None
        for existing in self.ib.trades():
            if getattr(existing.order, 'orderId', None) == order_id:
                trade = existing
                break

        if trade is None:
            try:
                await self.ib.reqOpenOrdersAsync()
            except Exception as exc:
                detail = f" ({context})" if context else ""
                self.logger.debug(
                    "Failed to refresh open orders%s for cancellation of %s: %s",
                    detail,
                    order_id,
                    exc,
                )
            else:
                for existing in self.ib.trades():
                    if getattr(existing.order, 'orderId', None) == order_id:
                        trade = existing
                        break

        if trade is not None:
            await self._cancel_trade_safe(trade, context=context)
        else:
            detail = f" ({context})" if context else ""
            self.logger.debug(f"Unable to locate trade{detail} for cancellation of order {order_id}")

    async def _retire_previous_bracket(
        self,
        position_id: str,
        *,
        stop_trade,
        target_trades,
        stop_order_id,
        target_order_ids,
    ) -> None:
        """Cancel legacy bracket orders after new protection is in place."""

        seen_ids = set()

        if stop_trade:
            order_id = getattr(stop_trade.order, 'orderId', None)
            if order_id is not None:
                seen_ids.add(order_id)
            await self._cancel_trade_safe(stop_trade, context=f"{position_id} stop")

        if stop_order_id is not None and stop_order_id not in seen_ids:
            seen_ids.add(stop_order_id)
            await self._cancel_order_id_safe(stop_order_id, context=f"{position_id} stop")

        for trade in target_trades or []:
            order_id = getattr(trade.order, 'orderId', None)
            if order_id is not None:
                seen_ids.add(order_id)
            await self._cancel_trade_safe(trade, context=f"{position_id} target")

        for order_id in target_order_ids or []:
            if order_id in seen_ids:
                continue
            seen_ids.add(order_id)
            await self._cancel_order_id_safe(order_id, context=f"{position_id} target")

    async def _finalize_position_close(
        self,
        position: Dict[str, Any],
        exit_price: float,
        realized_pnl_delta: float,
        reason: str,
        closed_quantity: float,
    ) -> None:
        symbol = position.get('symbol')
        position_id = position.get('id')

        if not symbol or not position_id:
            return

        await self._cancel_bracket_orders(position)
        self.position_brackets.pop(position_id, None)

        position['quantity'] = 0
        position['status'] = 'CLOSED'
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.utcnow().isoformat()
        position['close_reason'] = reason
        position['stop_engine_plan'] = []

        total_realized = float(position.get('realized_pnl') or 0.0)
        delta = float(realized_pnl_delta or 0.0)
        if total_realized == 0.0 and delta:
            total_realized = delta
            position['realized_pnl'] = total_realized
        position['commission'] = float(position.get('commission') or 0.0)

        redis_key = f'positions:open:{symbol}:{position_id}'
        await self.redis.delete(redis_key)
        removed = await self.redis.srem(f'positions:by_symbol:{symbol}', position_id)
        if removed:
            await self._decrement_position_count(symbol, removed)
            remaining = await self.redis.scard(f'positions:by_symbol:{symbol}')
            if remaining == 0:
                self._drop_market_data_subscription(symbol)

        closed_key = f'positions:closed:{datetime.utcnow().strftime("%Y%m%d")}:{position_id}'
        snapshot = dict(position)
        total_realized = float(position.get('realized_pnl') or 0.0)
        snapshot['realized_pnl'] = round(total_realized, 2)
        snapshot['commission'] = round(position['commission'], 2)
        await self.redis.setex(closed_key, 604800, json.dumps(snapshot))
        ts_now = time.time()
        await self.redis.zadd('positions:closed:index', {closed_key: ts_now})
        lookback_seconds = getattr(self.kelly_sizer, 'lookback_seconds', self.kelly_sizer.lookback_days * 86400)
        await self.redis.zremrangebyscore('positions:closed:index', 0, ts_now - (lookback_seconds * 2))
        self._cancel_pnl_single_subscription(position.get('ib_con_id'))

        await self.redis.incr('positions:closed:total')

        posted_total = float(position.get('realized_posted', 0.0) or 0.0)
        prior_unposted = float(position.pop('realized_unposted', 0.0) or 0.0)
        amount_to_publish = total_realized - posted_total
        if abs(amount_to_publish) < 1e-6 and prior_unposted:
            amount_to_publish = prior_unposted

        if abs(amount_to_publish) > 1e-6:
            try:
                await self._publish_realized_metrics(amount_to_publish)
            except Exception:
                position['realized_unposted'] = prior_unposted + amount_to_publish
            else:
                position['realized_posted'] = posted_total + amount_to_publish
                position['realized_unposted'] = 0.0
        else:
            position['realized_posted'] = posted_total
            position['realized_unposted'] = max(prior_unposted, 0.0)

        self.logger.info(
            "position_closed",
            extra={
                "action": "position_closed",
                "symbol": symbol,
                "position_id": position_id,
                "reason": reason,
                "exit_price": round(exit_price, 4),
                "realized_pnl_delta": round(realized_pnl_delta or 0.0, 4),
            },
        )

        await self._publish_exit_distribution(position, exit_price, closed_quantity, reason)
        await self.kelly_sizer.invalidate()


    async def _handle_target_fill(
        self,
        position: Dict[str, Any],
        fill_price: float,
        filled_qty: float,
        realized_pnl: float,
        commission: float,
        order_id: int,
    ) -> None:
        """Handle partial take-profit fills and rebuild protection for remainder."""

        try:
            symbol = position.get('symbol')
            position_id = position.get('id')
            if not symbol or not position_id:
                return

            multiplier = self._position_multiplier(position)
            side = position.get('side')
            state = StopState.from_dict(position.get('stop_engine_state'))

            current_qty = int(position.get('quantity', 0) or 0)
            filled_units = int(round(filled_qty))
            remaining_qty = max(current_qty - filled_units, 0)
            position['quantity'] = remaining_qty

            await self._register_realized_event(position, realized_pnl)

            if commission:
                position['commission'] = float(position.get('commission', 0.0) or 0.0) + commission

            plan_records = position.get('stop_engine_plan', []) or []
            matched_stage = None
            for stage in plan_records:
                if stage.get('order_id') == order_id:
                    matched_stage = stage
                    stage['status'] = 'filled'
                    stage['filled_quantity'] = stage.get('filled_quantity', 0) + filled_qty
                    stage['fill_price'] = fill_price
                    break

            if matched_stage and matched_stage.get('profit_pct') is not None:
                pct = float(matched_stage['profit_pct'])
                rounded_pct = round(pct, 6)
                existing = {round(ft, 6) for ft in state.filled_targets}
                if rounded_pct not in existing:
                    state.filled_targets.append(pct)

            reductions = position.setdefault('reductions', [])
            reductions.append(
                {
                    'quantity': filled_units,
                    'price': fill_price,
                    'reason': 'TARGET',
                    'timestamp': datetime.utcnow().isoformat(),
                    'multiplier': multiplier,
                    'realized': realized_pnl,
                    'commission': commission,
                }
            )

            position['target_order_ids'] = [
                oid for oid in position.get('target_order_ids', []) if oid != order_id
            ]
            position['stop_order_id'] = None
            position['oca_group'] = None
            position['stop_engine_state'] = state.to_dict()
            position['stop_engine_plan'] = plan_records

            redis_key = f'positions:open:{symbol}:{position_id}'
            await self.redis.set(redis_key, json.dumps(position))

            # Remove cached trade references for the expired bracket
            stop_trade = self.stop_orders.pop(position_id, None)
            if stop_trade and not stop_trade.isDone():
                try:
                    self.ib.cancelOrder(stop_trade.order)
                except Exception:
                    pass

            target_trades = self.target_orders.pop(position_id, None) or []
            for trade in target_trades:
                try:
                    if trade and not trade.isDone():
                        self.ib.cancelOrder(trade.order)
                except Exception:
                    continue

            if remaining_qty <= 0:
                # Fully closed by cascading target fills (rare)
                await self._finalize_position_close(position, fill_price, realized_pnl, 'TARGET', filled_units)
                return

            # Rebuild protective orders for the remaining size
            await self._rebuild_bracket(position)

            lifecycle_details = {
                'executed_at': datetime.utcnow().isoformat(),
                'fill_price': round(float(fill_price), 4),
                'quantity': filled_units,
                'realized_pnl': round(float(realized_pnl), 2),
                'remaining_quantity': remaining_qty,
                'reason': 'TARGET',
                'result': 'SCALE_OUT',
                'side': side,
            }
            await self._publish_lifecycle_event(position, 'SCALE_OUT', lifecycle_details)

        except Exception as exc:
            self.logger.error(f"Error handling target fill: {exc}")

    async def _archive_position(self, position: Dict[str, Any], reason: str) -> None:
        """Move a stale Redis position to the closed bucket."""

        symbol = position.get('symbol')
        position_id = position.get('id')
        if not symbol or not position_id:
            return

        now_iso = datetime.utcnow().isoformat()
        position['status'] = 'CLOSED'
        position['exit_reason'] = reason
        position.setdefault('close_reason', reason)
        position.setdefault('exit_time', now_iso)
        position.setdefault('exit_price', position.get('current_price'))

        pre_close_quantity = float(position.get('quantity') or 0.0)
        closed_quantity_override: Optional[float] = None

        if reason == 'SYNCED_CLOSED':
            if (
                position_id not in self._manual_fill_positions
                and position_id not in self._missing_manual_fill_alerted
            ):
                alert_payload = {
                    'type': 'execution_manual_fill_missing',
                    'symbol': symbol,
                    'position_id': position_id,
                    'reason': reason,
                    'account': position.get('ib_account'),
                    'timestamp': now_iso,
                }
                await self.redis.publish('alerts:critical', json.dumps(alert_payload))
                self._missing_manual_fill_alerted.add(position_id)
                self.logger.error(
                    "Position sync archived %s (%s) without execution event; attempting backfill",
                    symbol,
                    position_id,
                )

            backfill = await self._backfill_realized_from_ib(position, reason)
            if backfill:
                realized_val = backfill.get('realized')
                if realized_val is not None:
                    position['realized_pnl'] = float(realized_val)
                commission_val = backfill.get('commission')
                if commission_val is not None:
                    position['commission'] = float(commission_val)
                exit_price_val = backfill.get('exit_price')
                if exit_price_val is not None:
                    position['exit_price'] = float(exit_price_val)
                exit_time_val = backfill.get('exit_time')
                if exit_time_val:
                    position['exit_time'] = exit_time_val
                events = backfill.get('events') or []
                if events:
                    position.setdefault('realized_events', [])
                    position['realized_events'].extend(events)
                if backfill.get('closed_quantity') is not None:
                    closed_quantity_override = float(backfill['closed_quantity'] or 0.0)

                self.logger.info(
                    "execution_backfill_applied",
                    extra={
                        "action": "execution_backfill",
                        "symbol": symbol,
                        "position_id": position_id,
                        "realized": backfill.get('realized'),
                        "commission": backfill.get('commission'),
                    },
                )
            else:
                self.logger.warning(
                    "execution_backfill_unavailable",
                    extra={
                        "action": "execution_backfill",
                        "symbol": symbol,
                        "position_id": position_id,
                    },
                )

        realized_total = float(position.get('realized_pnl', 0.0) or 0.0)
        position['commission'] = float(position.get('commission') or 0.0)
        posted_total = float(position.get('realized_posted', 0.0) or 0.0)
        prior_unposted = float(position.pop('realized_unposted', 0.0) or 0.0)
        amount_to_publish = realized_total - posted_total
        if abs(amount_to_publish) < 1e-6 and prior_unposted:
            amount_to_publish = prior_unposted

        if abs(amount_to_publish) > 1e-6:
            try:
                await self._publish_realized_metrics(amount_to_publish)
            except Exception:
                position['realized_unposted'] = prior_unposted + amount_to_publish
            else:
                position['realized_posted'] = posted_total + amount_to_publish
                position['realized_unposted'] = 0.0
        else:
            position['realized_posted'] = posted_total
            position['realized_unposted'] = max(prior_unposted, 0.0)

        closed_quantity = (
            closed_quantity_override
            if closed_quantity_override is not None
            else pre_close_quantity
        )
        position['quantity'] = 0

        redis_key = f'positions:open:{symbol}:{position_id}'
        await self.redis.delete(redis_key)
        removed = await self.redis.srem(f'positions:by_symbol:{symbol}', position_id)
        if removed:
            await self._decrement_position_count(symbol, removed)

        closed_key = f'positions:closed:{datetime.utcnow().strftime("%Y%m%d")}:{position_id}'
        snapshot = dict(position)
        snapshot['realized_pnl'] = round(realized_total, 2)
        snapshot['commission'] = round(position['commission'], 2)
        await self.redis.setex(closed_key, 604800, json.dumps(snapshot))

        self.stop_orders.pop(position_id, None)
        self.target_orders.pop(position_id, None)

        self._manual_fill_positions.discard(position_id)
        self._missing_manual_fill_alerted.discard(position_id)

        if reason == 'SYNCED_CLOSED' and realized_total:
            log_msg = (
                f"Position sync: Archived {symbol} position {position_id[:8]} ({reason}); "
                f"realized={realized_total:.2f}"
            )
            self.logger.info(log_msg)
        else:
            self.logger.warning(
                f"Position sync: Archived {symbol} position {position_id[:8]} ({reason}); "
                "realized P&L may be incomplete"
            )

        await self._publish_exit_distribution(position, position.get('exit_price'), closed_quantity, reason)

    async def start(self):
        """
        Main execution loop for processing signals.
        Processing frequency: Every 200ms
        """
        self.logger.info("execution_manager_start", extra={"action": "start"})

        # Connect to IBKR
        await self.connect_ibkr()

        # Rehydrate cached state before processing loop
        await self._load_pending_orders_from_redis()

        # Ensure Redis state reflects live IB positions
        await self.reconcile_positions()

        # Hydrate open orders and protection orders now that Redis is current
        await self._hydrate_open_orders()

        # Set up event handlers (ib_insync specific events)
        self.ib.orderStatusEvent += self.on_order_status
        self.ib.execDetailsEvent += self.on_fill  # execDetailsEvent for fills in ib_insync
        self.ib.errorEvent += self.on_error

        # Start position sync task (runs every 30 seconds)
        asyncio.create_task(self.position_sync_loop())
        self._ensure_monitor_loop()

        while True:
            try:
                # Check if trading is halted
                halt_status = await self.redis.get('risk:halt:status')
                if halt_status == 'true':
                    await asyncio.sleep(1)
                    continue

                # Check connection health
                if not self.ib.isConnected():
                    await self.connect_ibkr()
                    continue

                # Process pending signals
                await self.process_pending_signals()

                # Update pending orders
                await self.update_order_status()

                # Update heartbeat
                await self.redis.setex('health:execution:heartbeat', 15, time.time())

            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")

            await asyncio.sleep(0.2)

    async def connect_ibkr(self):
        """
        Connect to IBKR with retry logic.
        Separate connection from data ingestion (different client ID).
        """
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()

                self.logger.info(
                    "execution_manager_connect",
                    extra={
                        "action": "connect",
                        "host": self.host,
                        "port": self.port,
                        "client_id": self.client_id,
                    },
                )
                await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=10)

                # Verify connection
                if self.ib.isConnected():
                    self.connected = True
                    self.reconnect_attempts = 0
                    self.logger.info(
                        "execution_manager_connected",
                        extra={"action": "connect", "status": "connected"}
                    )

                    # Register telemetry handlers once connected
                    self._register_event_streams()

                    # Get account info (ib_insync doesn't have async version)
                    account_values = self.ib.accountValues()
                    account_code = None
                    for av in account_values:
                        if av.tag == 'NetLiquidation':
                            await self.redis.set('account:value', av.value)
                            self.account_state['net_liquidation'] = self._safe_float(av.value)
                        elif av.tag == 'BuyingPower':
                            await self.redis.set('account:buying_power', av.value)
                            self.account_state['buying_power'] = self._safe_float(av.value)
                        elif av.tag == 'ExcessLiquidity':
                            self.account_state['excess_liquidity'] = self._safe_float(av.value)
                        elif av.tag == 'TotalCashValue':
                            self.account_state['total_cash_value'] = self._safe_float(av.value)
                        if av.account and not account_code:
                            account_code = av.account

                    if account_code:
                        self.account_state['account'] = account_code
                        await self._subscribe_account_summary(account_code)
                        await self._subscribe_pnl_streams(account_code)

                    await self.redis.set('execution:connection:status', 'connected')
                    await self._prime_execution_stream()
                    return True

            except Exception as e:
                self.logger.error(f"IBKR connection failed: {e}")
                self.reconnect_attempts += 1
                await asyncio.sleep(min(2 ** self.reconnect_attempts, 60))

        self.logger.error("Max reconnection attempts reached")
        await self.redis.set('execution:connection:status', 'failed')
        return False

    async def position_sync_loop(self):
        """
        Background task that syncs positions with IBKR every 30 seconds.
        """
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds

                raw_redis_keys = await self._scan_keys('positions:open:*')
                redis_positions: Dict[str, Dict[str, Any]] = {}

                for raw_key in raw_redis_keys:
                    key = raw_key
                    position_data = await self.redis.get(key)
                    if not position_data:
                        continue

                    position = json.loads(position_data)
                    symbol = position.get('symbol')
                    position_id = position.get('id')
                    if not symbol or not position_id:
                        continue

                    quantity = position.get('quantity', 0)
                    try:
                        quantity_val = float(quantity)
                    except (TypeError, ValueError):
                        quantity_val = 0.0

                    if quantity_val <= 0:
                        await self._archive_position(position, 'SYNCED_ZERO_QUANTITY')
                        continue

                    redis_positions[key] = position

                if not self.ib.isConnected():
                    continue

                # Get current positions from IBKR
                ib_positions = await self.ib.reqPositionsAsync()

                ib_position_map: Dict[str, Dict[str, Any]] = {}
                ib_contract_index: Dict[tuple, str] = {}

                for account, contract, size, avg_cost in ib_positions:
                    if abs(size) == 0:
                        continue

                    key = f"{account}:{contract.conId}"
                    ib_position_map[key] = {
                        'id': key,
                        'contract': contract,
                        'size': size,
                        'avg_cost': avg_cost,
                        'account': account,
                    }

                    canonical_key = self._canonical_contract_key(
                        getattr(contract, 'symbol', None),
                        contract,
                        'LONG' if size >= 0 else 'SHORT',
                    )
                    ib_contract_index[canonical_key] = key

                processed_ib_ids: set = set()

                for redis_key, position in list(redis_positions.items()):
                    symbol = position.get('symbol')
                    position_id = position.get('id')

                    if not symbol or not position_id:
                        continue

                    canonical_key = self._canonical_contract_key(
                        symbol,
                        position.get('contract', {}),
                        position.get('side'),
                    )

                    ib_entry = ib_position_map.get(position_id)
                    if ib_entry:
                        updated_snapshot = self._create_position_snapshot_from_ib(
                            position_id,
                            ib_entry['contract'],
                            ib_entry['size'],
                            ib_entry['avg_cost'],
                            ib_entry['account'],
                            existing=position,
                        )
                        await self.redis.set(redis_key, json.dumps(updated_snapshot))
                        await self._ensure_position_protection(updated_snapshot)
                        processed_ib_ids.add(position_id)
                        continue

                    ib_match_id = ib_contract_index.get(canonical_key)
                    if ib_match_id:
                        ib_entry = ib_position_map.get(ib_match_id)
                        if not ib_entry:
                            continue

                        dest_key = f'positions:open:{symbol}:{ib_match_id}'
                        destination = redis_positions.get(dest_key)
                        if destination is None:
                            dest_json = await self.redis.get(dest_key)
                            if dest_json:
                                destination = json.loads(dest_json)
                            else:
                                destination = self._create_position_snapshot_from_ib(
                                    ib_match_id,
                                    ib_entry['contract'],
                                    ib_entry['size'],
                                    ib_entry['avg_cost'],
                                    ib_entry['account'],
                                )

                        self._merge_position_metadata(position, destination)
                        await self.redis.set(dest_key, json.dumps(destination))
                        await self.redis.sadd(f'positions:by_symbol:{symbol}', ib_match_id)

                        if position_id != ib_match_id:
                            await self._remove_duplicate_position(symbol, position_id)
                            redis_positions.pop(redis_key, None)

                        redis_positions[dest_key] = destination
                        await self._ensure_position_protection(destination)
                        processed_ib_ids.add(ib_match_id)
                        continue

                    # No IB match – position has been closed outside the system
                    await self._archive_position(position, 'SYNCED_CLOSED')
                    redis_positions.pop(redis_key, None)

                for ib_id, ib_entry in ib_position_map.items():
                    if ib_id in processed_ib_ids:
                        continue

                    contract = ib_entry['contract']
                    symbol = getattr(contract, 'symbol', None)
                    if not symbol:
                        continue

                    # Skip zero-quantity positions from IB
                    if abs(ib_entry['size']) < 0.0001:
                        self.logger.debug(f"Position sync: Skipping zero-quantity {symbol} position {ib_id}")
                        continue

                    snapshot = self._create_position_snapshot_from_ib(
                        ib_id,
                        contract,
                        ib_entry['size'],
                        ib_entry['avg_cost'],
                        ib_entry['account'],
                    )

                    redis_key = f'positions:open:{symbol}:{ib_id}'
                    await self.redis.set(redis_key, json.dumps(snapshot))
                    added = await self.redis.sadd(f'positions:by_symbol:{symbol}', ib_id)
                    if added:
                        await self._increment_position_count(added)
                    await self._ensure_position_protection(snapshot)

                    self.logger.info(
                        "execution_position_added",
                        extra={
                            "action": "position_sync",
                            "symbol": symbol,
                            "position_id": ib_id,
                        },
                    )

                self.logger.debug("Position sync completed")

            except Exception as e:
                self.logger.error(f"Error in position sync loop: {e}")
                await asyncio.sleep(30)

    async def reconcile_positions(self):
        """Backfill Redis with any live IB positions that are missing locally."""
        if not self.ib.isConnected():
            return

        try:
            positions = await self.ib.reqPositionsAsync()
        except Exception as exc:  # pragma: no cover - IB call guard
            self.logger.warning(f"Unable to reconcile positions from IB: {exc}")
            return

        for account, contract, size, avg_cost in positions:
            try:
                if abs(size) == 0:
                    continue

                symbol = getattr(contract, 'symbol', None)
                if not symbol:
                    continue

                position_id = f"{account}:{contract.conId}"
                redis_key = f'positions:open:{symbol}:{position_id}'

                existing_json = await self.redis.get(redis_key)
                existing = json.loads(existing_json) if existing_json else None

                snapshot = self._create_position_snapshot_from_ib(
                    position_id,
                    contract,
                    size,
                    avg_cost,
                    account,
                    existing=existing,
                )

                await self.redis.set(redis_key, json.dumps(snapshot))
                added = await self.redis.sadd(f'positions:by_symbol:{symbol}', position_id)
                if not existing and added:
                    await self._increment_position_count(added)
                self.logger.info(
                    "execution_position_reconciled",
                    extra={
                        "action": "position_sync",
                        "symbol": symbol,
                        "position_id": position_id,
                    },
                )
                self._ensure_pnl_single_subscription(account, contract.conId)
            except Exception as snapshot_error:  # pragma: no cover - defensive per-position
                self.logger.error(
                    f"Failed to reconcile position for {getattr(contract, 'symbol', 'UNKNOWN')}: {snapshot_error}"
                )

    async def process_pending_signals(self):
        """
        Process signals from the pending queue.
        """
        # Get symbols we should process
        symbols = self.config.get('symbols', {}).get('level2', []) + \
                  self.config.get('symbols', {}).get('standard', [])

        for symbol in symbols:
            # Check position limits
            existing_positions = await self.redis.scard(f'positions:by_symbol:{symbol}')
            if existing_positions and existing_positions >= self.max_per_symbol:
                continue

            # Get pending signal (handle bytes or string)
            signal_data = await self.redis.rpop(f'signals:execution:{symbol}')
            if not signal_data:
                continue

            try:
                # Handle both string and bytes from Redis
                if isinstance(signal_data, bytes):
                    signal_json = signal_data.decode('utf-8')
                else:
                    signal_json = signal_data
                signal = json.loads(signal_json)

                # Check signal freshness (max 5 seconds old)
                age_ms = time.time() * 1000 - signal.get('ts', 0)
                if age_ms > 5000:
                    self.logger.warning(f"Signal too old: {age_ms}ms for {symbol}")
                    await self._acknowledge_signal(
                        signal,
                        status='STALE',
                        reason=f'stale_signal_{int(age_ms)}ms',
                        broadcast=True,
                    )
                    continue

                # Check risk approval
                if await self.passes_risk_checks(signal):
                    await self.execute_signal(signal)
                else:
                    await self._acknowledge_signal(
                        signal,
                        status='RISK_REJECTED',
                        reason='risk_checks_failed',
                        broadcast=True,
                    )

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid signal JSON: {e}")

    async def _acknowledge_signal(
        self,
        signal: Optional[Dict[str, Any]],
        *,
        status: str,
        filled: float = 0.0,
        avg_price: Optional[float] = None,
        reason: Optional[str] = None,
        broadcast: bool = False,
    ) -> None:
        """Publish execution acknowledgement and release signal live-lock."""

        if not signal:
            return

        symbol = signal.get('symbol')
        contract = signal.get('contract') or {}
        strategy = signal.get('strategy') or ''
        side = signal.get('side') or ''
        signal_id = signal.get('id')

        if not (symbol and contract):
            return

        try:
            contract_fp = contract_fingerprint(symbol, strategy, side, contract)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"Failed to compute contract fingerprint for ack: {exc}")
            return

        live_key = f'signals:live:{symbol}:{contract_fp}'
        try:
            await self.redis.delete(live_key)
        except Exception:
            pass

        ack_payload = {
            'id': signal_id,
            'symbol': symbol,
            'strategy': strategy,
            'side': side,
            'status': status,
            'contract_fp': contract_fp,
            'filled': float(filled or 0.0),
            'avg_price': float(avg_price) if avg_price is not None else None,
            'timestamp': time.time(),
        }
        if reason:
            ack_payload['reason'] = reason

        if signal_id:
            try:
                await self.redis.setex(
                    f'signals:ack:{signal_id}',
                    86400,
                    json.dumps(ack_payload),
                )
                await self.redis.publish('signals:acknowledged', json.dumps(ack_payload))
                await self.redis.incr('metrics:signals:acks')
            except Exception as exc:  # pragma: no cover
                self.logger.error(f"Failed to publish signal ack: {exc}")
        else:
            self.logger.debug(
                "signal_ack_missing_id",
                extra={'action': 'signal_ack', 'symbol': symbol, 'contract_fp': contract_fp},
            )

        if broadcast:
            await self._publish_signal_status(
                signal,
                status=status,
                reason=reason,
                filled=filled,
                avg_price=avg_price,
            )

    async def _get_risk_manager(self) -> Optional["RiskManager"]:
        """Return the shared RiskManager instance, creating it if needed."""
        async with self._risk_manager_lock:
            if self._risk_manager is None:
                try:
                    if self._risk_manager_factory:
                        candidate = self._risk_manager_factory()
                        if inspect.isawaitable(candidate):
                            candidate = await candidate
                        self._risk_manager = candidate
                    else:
                        from risk_manager import RiskManager  # Avoid circular import

                        self._risk_manager = RiskManager(self.config, self.redis)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.error(f"Failed to initialize RiskManager: {exc}")
                    self._risk_manager = None
            return self._risk_manager

    async def passes_risk_checks(self, signal: dict, *, position_size_override: Optional[float] = None) -> bool:
        """
        Perform pre-trade risk checks.

        Returns:
            True if all risk checks pass
        """
        try:
            # Check if new positions allowed
            new_positions_allowed = await self.redis.get('risk:new_positions_allowed')
            if new_positions_allowed == 'false':
                self.logger.warning("New positions blocked by risk manager")
                return False

            # Check buying power
            buying_power = float(await self.redis.get('account:buying_power') or 0)
            position_size = position_size_override
            if position_size is None:
                position_size = signal.get('position_size', 0)

            if position_size and position_size > buying_power * 0.25:  # Max 25% of buying power
                self.logger.warning(f"Position size {position_size} exceeds 25% of buying power {buying_power}")
                return False

            # Check daily loss limit
            daily_pnl = float(await self.redis.get('risk:daily_pnl') or 0)
            if daily_pnl < -20000:  # $20,000 daily loss limit (paper)
                self.logger.warning(f"Daily loss limit exceeded: {daily_pnl}")
                return False

            # Check correlation limits (delegated to RiskManager)
            symbol = signal.get('symbol')
            side = signal.get('side')

            risk_manager = await self._get_risk_manager()
            if risk_manager and not await risk_manager.check_correlations(symbol, side):
                self.logger.warning(f"Correlation limit exceeded for {symbol} {side}")
                return False

            # Check total position count
            open_positions = self._safe_float(await self.redis.get('positions:count') or 0.0)
            if open_positions >= self.max_positions:
                self.logger.warning(f"Max positions limit reached: {int(open_positions)}/{self.max_positions}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in risk checks: {e}")
            return False  # Conservative: block on error

    async def execute_signal(self, signal: dict):
        """
        Execute a trading signal through IBKR.
        """
        try:
            symbol = signal.get('symbol')
            side = signal.get('side')  # LONG or SHORT
            confidence = signal.get('confidence', 60)
            strategy = signal.get('strategy')  # 0DTE, 1DTE, 14DTE, MOC
            strategy_code = self._normalize_strategy(strategy)

            self.logger.info(
                "execution_signal_accepted",
                extra={
                    "action": "execute_signal",
                    "symbol": symbol,
                    "side": side,
                    "strategy": strategy,
                    "confidence_pct": confidence,
                },
            )

            # Create IB contract (add symbol to contract info)
            contract_info = signal.get('contract', {})
            contract_info['symbol'] = symbol  # Add symbol to contract dict
            ib_contract = await self.create_ib_contract(contract_info)
            if not ib_contract:
                self.logger.error(f"Failed to create contract for {symbol}")
                await self._acknowledge_signal(
                    signal,
                    status='REJECTED',
                    reason='contract_lookup_failed',
                    broadcast=True,
                )
                await self._maybe_release_market_data(symbol)
                return

            # Get current market data (reuse existing subscription when possible)
            await self._ensure_market_data_subscription(symbol, contract_info)
            ticker = self.position_tickers.get(symbol)
            if ticker is None:
                ticker = self.ib.reqMktData(ib_contract, '', False, False)
                self.position_tickers[symbol] = ticker
            await asyncio.sleep(0.3)  # Allow snapshot to populate

            # Calculate order size
            order_size = await self.calculate_order_size(signal, ticker)
            if order_size == 0:
                self.logger.warning(f"Order size is 0 for {symbol}")
                await self._acknowledge_signal(
                    signal,
                    status='REJECTED',
                    reason='sizing_zero',
                    broadcast=True,
                )
                await self._maybe_release_market_data(symbol)
                return

            # Re-run notional-based risk gate now that we know the exposure
            position_notional = signal.get('position_size')
            if not await self.passes_risk_checks(signal, position_size_override=position_notional):
                self.logger.warning(
                    "Risk checks failed after sizing",
                    extra={
                        "action": "risk_block",
                        "symbol": symbol,
                        "position_notional": position_notional,
                    },
                )
                await self._acknowledge_signal(
                    signal,
                    status='RISK_REJECTED',
                    reason='risk_checks_failed_post_sizing',
                    broadcast=True,
                )
                await self._maybe_release_market_data(symbol)
                return

            # Determine order type and price
            contract_type = str(contract_info.get('type') or '').strip().lower()
            option_types = {'option', 'options', 'opt'}
            side_upper = (side or '').upper()

            if contract_type in option_types:
                action = 'BUY' if side_upper == 'LONG' else 'SELL'
            else:
                action = 'BUY' if side_upper == 'LONG' else 'SELL'

            limit_price = None
            order_type = 'MARKET'
            selected_price = None

            if confidence > 85 and strategy_code != '0DTE':
                order_type = 'MARKET'
            else:
                bid = self._safe_price(getattr(ticker, 'bid', None))
                ask = self._safe_price(getattr(ticker, 'ask', None))
                if bid is not None and ask is not None:
                    spread = max(ask - bid, 0.0)
                    if confidence >= 70:
                        limit_price = (bid + ask) / 2
                    else:
                        limit_price = bid + spread * 0.33 if action == 'BUY' else ask - spread * 0.33
                if limit_price is None:
                    for candidate in (
                        self._safe_price(getattr(ticker, 'last', None)),
                        self._safe_price(signal.get('entry')),
                    ):
                        if candidate is not None:
                            limit_price = candidate
                            break

                if limit_price is not None:
                    limit_price = round(limit_price, 2)
                    order_type = 'LIMIT'
                    selected_price = limit_price
                else:
                    self.logger.warning(
                        "Falling back to market order due to missing price",
                        extra={
                            "symbol": symbol,
                            "reason": "no_valid_limit_price",
                            "strategy": strategy_code,
                        },
                    )
                    order_type = 'MARKET'

            if selected_price is None:
                selected_price = self._safe_price(getattr(ticker, 'last', None))
            if selected_price is None:
                selected_price = self._safe_price(getattr(ticker, 'midpoint', None))
            if selected_price is None:
                selected_price = self._safe_price(signal.get('entry'))
            if selected_price is None:
                self.logger.warning(
                    "Aborting execution due to missing market data",
                    extra={
                        "action": "execute_signal",
                        "symbol": symbol,
                        "reason": "no_market_data",
                    },
                )
                await self._acknowledge_signal(
                    signal,
                    status='NO_MARKET_DATA',
                    reason='price_unavailable',
                    broadcast=True,
                )
                await self._maybe_release_market_data(symbol)
                return

            stop_price, stop_state, plan_records, oca_group = self._build_protection_plan(
                symbol=symbol,
                side=side_upper,
                quantity=order_size,
                entry_price=float(selected_price),
                contract_info=contract_info,
                strategy_code=strategy_code,
                signal_targets=signal.get('targets'),
            )

            bracket = await self._submit_bracket(
                ib_contract=ib_contract,
                action=action,
                order_size=order_size,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                stop_state=stop_state,
                plan_records=plan_records,
                oca_group=oca_group,
            )

            trade = bracket['parent_trade']
            order_id = bracket['parent_order_id']

            # Store pending order in Redis
            order_data = {
                'order_id': order_id,
                'signal_id': signal.get('id'),
                'symbol': symbol,
                'contract': contract_info,
                'side': side,
                'action': action,
                'size': order_size,
                'position_notional': signal.get('position_size'),
                'order_type': order_type,
                'confidence': confidence,
                'strategy': strategy_code or strategy,
                'entry_target': signal.get('entry'),
                'stop_loss': stop_price,
                'targets': [plan.get('price') for plan in plan_records],
                'stop_order_id': bracket.get('stop_order_id'),
                'target_order_ids': bracket.get('target_order_ids'),
                'oca_group': oca_group,
                'placed_at': time.time(),
                'status': 'PENDING',
                'signal': signal,
            }

            self.pending_orders[order_id] = order_data

            await self._persist_pending_order(order_id)

            # Track trade for reconciliation/updates
            self.active_trades[order_id] = {'trade': trade, 'signal': signal}

            self.bracket_registry[order_id] = {
                'parent_trade': trade,
                'stop_trade': bracket['stop_trade'],
                'target_trades': bracket['target_trades'],
                'stop_order_id': bracket.get('stop_order_id'),
                'target_order_ids': bracket.get('target_order_ids'),
                'stop_engine_state': bracket['stop_state'],
                'plan_records': plan_records,
                'initial_stop': stop_price,
                'symbol': symbol,
                'side': side_upper,
                'strategy': strategy_code,
                'contract': contract_info,
                'quantity': order_size,
                'entry_reference': selected_price,
                'basis_price': selected_price,
            }

            watcher = asyncio.create_task(self.monitor_order(trade, signal))
            self._order_watchers[order_id] = watcher
            watcher.add_done_callback(lambda _: self._order_watchers.pop(order_id, None))

            self.logger.info(
                "execution_order_placed",
                extra={
                    "action": "order_placed",
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": action,
                    "size": order_size,
                    "order_type": order_type,
                },
            )

        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            await self.redis.incr('execution:errors:total')
            order_id = locals().get('order_id')
            if order_id is not None:
                await self._cancel_order_id_safe(order_id, context='execute_signal_exception')
                await self.redis.delete(f'orders:pending:{order_id}')
                self.pending_orders.pop(order_id, None)
                self.bracket_registry.pop(order_id, None)
            await self._acknowledge_signal(
                signal,
                status='ERROR',
                reason='execute_signal_exception',
                broadcast=True,
            )
            await self._maybe_release_market_data(signal.get('symbol') if signal else None)

    async def create_ib_contract(self, signal_contract: dict):
        """
        Create IBKR contract object from signal contract.

        Returns:
            IB Contract object
        """
        try:
            raw_type = signal_contract.get('type') or 'option'
            contract_type = str(raw_type).strip().lower()
            symbol = signal_contract.get('symbol')

            if contract_type in {'option', 'options', 'opt'}:
                # Parse OCC symbol if provided
                occ_symbol = signal_contract.get('occ_symbol')
                if occ_symbol:
                    # OCC format: SPYYYMMDDCP00500000
                    # Extract components using regex from analytics.py
                    occ_re = re.compile(r'^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$')
                    match = occ_re.match(occ_symbol)

                    if match:
                        root = match.group('root')
                        date_str = match.group('date')
                        right = 'C' if match.group('cp') == 'C' else 'P'
                        strike = float(match.group('strike')) / 1000  # Convert to actual strike

                        # Convert date to YYYYMMDD format
                        expiry = f"20{date_str}"
                    else:
                        # Fallback to explicit fields
                        root = symbol
                        expiry = signal_contract.get('expiry')
                        strike = signal_contract.get('strike')
                        right = signal_contract.get('right', 'C')
                else:
                    # Use explicit fields
                    root = symbol
                    expiry = signal_contract.get('expiry')
                    strike = signal_contract.get('strike')
                    right = signal_contract.get('right', 'C')

                expiry = normalize_expiry(expiry)
                if not expiry:
                    self.logger.error(f"Invalid or missing expiry for option contract: {signal_contract}")
                    return None

                try:
                    strike = float(strike)
                except (TypeError, ValueError):
                    self.logger.error(f"Invalid strike for option contract: {signal_contract}")
                    return None

                right = str(right or 'C').upper()
                right = 'C' if right.startswith('C') else 'P'

                # Create Option contract
                contract = Option(root, expiry, strike, right, 'SMART', currency='USD')
                contract.multiplier = '100'

            elif contract_type in {'stock', 'equity'}:
                # Stock contract
                contract = Stock(symbol, 'SMART', 'USD')

            else:
                self.logger.warning(
                    "Unknown contract type '%s'; defaulting to option", raw_type
                )
                expiry = normalize_expiry(signal_contract.get('expiry'))
                if not expiry:
                    self.logger.error(f"Invalid or missing expiry for option contract: {signal_contract}")
                    return None
                strike = signal_contract.get('strike')
                try:
                    strike = float(strike)
                except (TypeError, ValueError):
                    self.logger.error(f"Invalid strike for option contract: {signal_contract}")
                    return None
                right = signal_contract.get('right', 'C')
                right = str(right or 'C').upper()
                right = 'C' if right.startswith('C') else 'P'
                contract = Option(symbol, expiry, strike, right, 'SMART', currency='USD')
                contract.multiplier = '100'

            # Qualify contract to ensure it's valid (use async method)
            qualified = await self.ib.qualifyContractsAsync(contract)
            if qualified:
                # Cache qualified contract (convert to dict manually)
                contract_dict = {
                    'symbol': contract.symbol,
                    'secType': contract.secType,
                    'exchange': contract.exchange,
                    'currency': contract.currency
                }
                # Add option-specific fields if present
                if hasattr(contract, 'strike'):
                    contract_dict['strike'] = contract.strike
                    contract_dict['right'] = contract.right
                    contract_dict['lastTradeDateOrContractMonth'] = contract.lastTradeDateOrContractMonth
                    cache_key = f"contracts:qualified:{contract.symbol}:{contract.lastTradeDateOrContractMonth}:{contract.strike}:{contract.right}"
                else:
                    cache_key = f"contracts:qualified:{contract.symbol}"

                await self.redis.setex(cache_key, 300, json.dumps(contract_dict))
                return qualified[0]
            else:
                self.logger.error(f"Failed to qualify contract: {signal_contract}")
                return None

        except Exception as e:
            self.logger.error(f"Error creating IB contract: {e}")
            return None

    async def calculate_order_size(self, signal: dict, ticker) -> int:
        """
        Calculate appropriate order size based on account size and confidence.

        Returns:
            Order size (contracts or shares)
        """
        try:
            account_value = self.account_state.get('net_liquidation') or self._safe_float(
                await self.redis.get('account:value')
            )
            buying_power = self.account_state.get('buying_power') or self._safe_float(
                await self.redis.get('account:buying_power')
            )

            confidence = float(signal.get('confidence', 60))
            strategy_code = self._normalize_strategy(signal.get('strategy'))
            contract_payload = signal.get('contract', {})
            raw_type = str(contract_payload.get('type') or contract_payload.get('secType') or 'option')
            contract_type = raw_type.strip().lower()

            notional, stats = await self.kelly_sizer.suggest_notional(
                strategy=strategy_code,
                confidence=confidence,
                account_value=account_value or 0.0,
                buying_power=buying_power or 0.0,
            )

            if notional <= 0:
                notional = max(1000.0, account_value * 0.01)

            option_price: Optional[float] = None
            stock_price: Optional[float] = None
            if ticker:
                bid = self._safe_price(getattr(ticker, 'bid', None))
                ask = self._safe_price(getattr(ticker, 'ask', None))
                last = self._safe_price(getattr(ticker, 'last', None))
                mid = None
                if bid is not None and ask is not None:
                    mid = (bid + ask) / 2

                midpoint = self._safe_price(getattr(ticker, 'midpoint', None)) or mid

                if midpoint is not None:
                    option_price = midpoint
                    stock_price = midpoint

                if last is not None:
                    option_price = option_price or last
                    stock_price = stock_price or last

            entry_hint = self._safe_price(signal.get('entry'))
            if option_price is None:
                option_price = entry_hint if entry_hint is not None else 1.0
            if stock_price is None:
                stock_price = entry_hint if entry_hint is not None else 50.0

            signal.setdefault('sizing_stats', {})

            if contract_type in {'option', 'options', 'opt'}:
                multiplier = 100.0
                contracts = int(max(1, notional / max(option_price * multiplier, 1)))
                if strategy_code == '0DTE':
                    contracts = min(contracts, self.max_0dte_contracts)
                else:
                    contracts = min(contracts, self.max_other_contracts)
                signal['order_size'] = max(1, contracts)
                position_notional = signal['order_size'] * option_price * multiplier
                signal['position_size'] = round(position_notional, 2)
                signal['sizing_stats'] = {
                    'win_rate': stats.win_rate,
                    'payoff_ratio': stats.payoff_ratio,
                    'kelly_fraction': stats.kelly_fraction,
                    'sample_size': stats.sample_size,
                    'suggested_notional': notional,
                }
                return signal['order_size']

            elif contract_type in {'stock', 'equity'}:
                shares = int(max(1, notional / max(stock_price, 1)))
                signal['order_size'] = max(1, shares)
                position_notional = signal['order_size'] * stock_price
                signal['position_size'] = round(position_notional, 2)
                signal['sizing_stats'] = {
                    'win_rate': stats.win_rate,
                    'payoff_ratio': stats.payoff_ratio,
                    'kelly_fraction': stats.kelly_fraction,
                    'sample_size': stats.sample_size,
                    'suggested_notional': notional,
                }
                return signal['order_size']

            else:
                self.logger.warning(
                    "Unknown contract type '%s'; defaulting to option sizing", raw_type
                )
                multiplier = 100.0
                contracts = int(max(1, notional / max(option_price * multiplier, 1)))
                signal['order_size'] = max(1, contracts)
                position_notional = signal['order_size'] * option_price * multiplier
                signal['position_size'] = round(position_notional, 2)
                signal['sizing_stats'] = {
                    'win_rate': stats.win_rate,
                    'payoff_ratio': stats.payoff_ratio,
                    'kelly_fraction': stats.kelly_fraction,
                    'sample_size': stats.sample_size,
                    'suggested_notional': notional,
                }
                return signal['order_size']

        except Exception as e:
            self.logger.error(f"Error calculating order size: {e}")
            return 1  # Conservative default

    async def _persist_pending_order(self, order_id: int, ttl: int = 300) -> None:
        """Persist a pending order snapshot back to Redis without internal metadata."""
        order_record = self.pending_orders.get(order_id)
        if not order_record:
            return

        payload = {k: v for k, v in order_record.items() if k != 'signal'}
        await self.redis.setex(
            f'orders:pending:{order_id}',
            ttl,
            json.dumps(payload)
        )

    async def _sync_trade_state(self, order_id: int, trade, signal: dict) -> Dict[str, Any]:
        """Synchronise in-memory/Redis snapshots with the latest IB trade state."""
        status = trade.orderStatus.status or 'PendingSubmit'
        total_quantity = getattr(trade.order, 'totalQuantity', 0) or 0
        try:
            total_quantity = int(total_quantity)
        except (TypeError, ValueError):
            total_quantity = int(self.pending_orders.get(order_id, {}).get('size', 0))

        filled = trade.orderStatus.filled or 0
        remaining = trade.orderStatus.remaining if trade.orderStatus.remaining is not None else 0

        if trade.fills:
            fill_qty = sum(abs(getattr(fill.execution, 'shares', 0)) for fill in trade.fills)
            if fill_qty:
                filled = max(filled, fill_qty)
                remaining = max(total_quantity - fill_qty, 0)

        if total_quantity and not remaining:
            remaining = max(total_quantity - filled, 0)

        avg_price = None
        if trade.fills:
            qty_sum = sum(abs(getattr(fill.execution, 'shares', 0)) for fill in trade.fills)
            if qty_sum:
                avg_price = sum(
                    abs(getattr(fill.execution, 'shares', 0)) * getattr(fill.execution, 'price', 0)
                    for fill in trade.fills
                ) / qty_sum

        order_record = self.pending_orders.get(order_id)
        if order_record is not None:
            order_record['status'] = status
            order_record['filled'] = filled
            order_record['remaining'] = remaining
            order_record['avg_fill_price'] = avg_price
            order_record['last_update'] = time.time()

            if filled and remaining and not order_record.get('partial_alerted'):
                self.logger.info(
                    "execution_order_partial",
                    extra={
                        "action": "order_partial",
                        "order_id": order_id,
                        "filled": filled,
                        "total": total_quantity,
                        "symbol": order_record.get('symbol'),
                    },
                )
                order_record['partial_alerted'] = True

            await self._persist_pending_order(order_id)

        return {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avg_price': avg_price,
            'total': total_quantity,
        }

    async def _enqueue_distribution_signal(
        self,
        signal: dict,
        position: Dict[str, Any],
        avg_price: float,
        quantity: int,
        total_commission: float,
        executed_at: str,
        order_id: int,
    ) -> None:
        """Push executed signal details into the downstream distribution queue."""
        try:
            payload = dict(signal)
            payload['ts'] = payload.get('ts') or int(time.time() * 1000)
            payload['position_id'] = position.get('id')
            payload['position_notional'] = signal.get('position_size')
            payload['stop_loss'] = position.get('stop_loss')
            payload['targets'] = position.get('targets', [])
            payload['action_type'] = payload.get('action_type') or 'ENTRY'
            payload['execution'] = {
                'status': 'FILLED',
                'avg_fill_price': round(float(avg_price), 4),
                'filled_quantity': int(quantity),
                'commission': round(float(total_commission), 4),
                'order_id': order_id,
                'position_id': position.get('id'),
                'executed_at': executed_at,
                'notional': signal.get('position_size'),
            }

            await self.redis.lpush('signals:distribution:pending', json.dumps(payload))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"Failed to enqueue distribution payload: {exc}")

    async def _publish_signal_status(
        self,
        signal: Optional[Dict[str, Any]],
        *,
        status: str,
        reason: Optional[str] = None,
        filled: float = 0.0,
        avg_price: Optional[float] = None,
    ) -> None:
        """Emit a lightweight lifecycle event for non-fill outcomes."""
        if not signal:
            return

        if not self.config.get('modules', {}).get('signals', {}).get('enabled', False):
            return

        try:
            payload = {
                'id': signal.get('id'),
                'symbol': signal.get('symbol'),
                'strategy': self._normalize_strategy(signal.get('strategy')),
                'side': signal.get('side'),
                'action_type': 'EXECUTION_STATUS',
                'ts': int(time.time() * 1000),
                'contract': signal.get('contract'),
                'status': status,
                'reason': reason,
            }
            execution_snapshot = {
                'status': status,
                'filled_quantity': filled,
                'avg_fill_price': float(avg_price) if avg_price is not None else None,
                'executed_at': datetime.utcnow().isoformat(),
                'reason': reason,
            }
            payload['execution'] = {k: v for k, v in execution_snapshot.items() if v is not None}
            payload = {k: v for k, v in payload.items() if v is not None}
            await self.redis.lpush('signals:distribution:pending', json.dumps(payload))
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(f"Failed to publish signal status: {exc}")

    async def _escalate_timeout(self, order_id: int, trade, signal: dict, filled: float, timeout_seconds: int) -> None:
        """Alert and cancel an order that exceeded its monitoring window."""
        symbol = signal.get('symbol')
        side = signal.get('side')

        self.logger.warning(
            f"Order {order_id} timed out after {timeout_seconds}s (filled={filled})"
        )

        await self.redis.incr('execution:timeouts:total')
        alert = {
            'type': 'execution_timeout',
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'filled': filled,
            'timeout_seconds': timeout_seconds,
            'timestamp': time.time(),
        }
        await self.redis.publish('alerts:critical', json.dumps(alert))

        try:
            self.ib.cancelOrder(trade.order)
        except Exception as cancel_error:
            self.logger.error(f"Failed to cancel order {order_id} after timeout: {cancel_error}")

        order_snapshot = self.pending_orders.get(order_id)

        await self.redis.delete(f'orders:pending:{order_id}')
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]

        self.bracket_registry.pop(order_id, None)

        await self.handle_order_rejection(trade.order, 'Timeout', order_snapshot)
        await self._maybe_release_market_data(symbol)

    async def monitor_order(self, trade, signal: dict):
        """
        Monitor order until completion.
        """
        order_id = trade.order.orderId
        base_timeout = 60 if trade.order.orderType == 'LMT' else 30
        start_time = time.time()

        try:
            while True:
                await asyncio.sleep(0.2)

                snapshot = await self._sync_trade_state(order_id, trade, signal)
                status = snapshot['status']
                filled = snapshot['filled']

                if status == 'Filled':
                    await self.handle_fill(trade, signal)
                    break

                if status in {'Cancelled', 'ApiCancelled'}:
                    self.logger.warning(f"Order {order_id} cancelled ({status})")
                    await self.redis.delete(f'orders:pending:{order_id}')
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    self.bracket_registry.pop(order_id, None)
                    await self._acknowledge_signal(
                        signal,
                        status=status.upper(),
                        filled=snapshot.get('filled', 0),
                        avg_price=snapshot.get('avg_price'),
                        reason=status.lower(),
                        broadcast=True,
                    )
                    await self._maybe_release_market_data(signal.get('symbol') if signal else None)
                    break

                if status in {'Inactive', 'Rejected'}:
                    self.bracket_registry.pop(order_id, None)
                    snapshot = self.pending_orders.get(order_id)
                    await self.handle_order_rejection(trade.order, status, snapshot)
                    break

                timeout_window = base_timeout * (2 if filled else 1)
                if time.time() - start_time >= timeout_window:
                    await self._escalate_timeout(order_id, trade, signal, filled, timeout_window)
                    break
        except Exception as exc:
            self.logger.error(f"Error monitoring order {order_id}: {exc}")
            await self._acknowledge_signal(
                signal,
                status='ERROR',
                reason='monitor_exception',
                broadcast=True,
            )
            await self._maybe_release_market_data(signal.get('symbol') if signal else None)
        finally:
            self.active_trades.pop(order_id, None)
            self._order_watchers.pop(order_id, None)

    async def handle_fill(self, trade, signal: dict):
        """
        Process filled orders and create position.
        """
        try:
            fills = trade.fills or []
            if not fills:
                self.logger.error("No fill data available")
                return

            order_id = trade.order.orderId
            symbol = signal.get('symbol')

            total_quantity = sum(abs(getattr(fill.execution, 'shares', 0)) for fill in fills)
            if total_quantity == 0:
                self.logger.error(f"No filled quantity for order {order_id}")
                return

            total_cost = sum(
                abs(getattr(fill.execution, 'shares', 0)) * getattr(fill.execution, 'price', 0)
                for fill in fills
            )
            avg_price = total_cost / total_quantity

            total_commission = sum(self._extract_commission(fill) for fill in fills)

            last_fill_time = max((getattr(fill, 'time', None) for fill in fills if getattr(fill, 'time', None)), default=None)
            entry_time = last_fill_time.isoformat() if last_fill_time else datetime.utcnow().isoformat()

            # Create position record
            ib_contract = getattr(trade, 'contract', None)
            account = next(
                (
                    getattr(fill.execution, 'acctNumber', None)
                    for fill in fills
                    if getattr(fill.execution, 'acctNumber', None)
                ),
                None,
            )
            con_id = getattr(ib_contract, 'conId', None)

            if account and con_id:
                position_id = f"{account}:{con_id}"
            else:
                position_id = str(uuid.uuid4())

            contract_payload = dict(signal.get('contract', {}) or {})
            if ib_contract:
                contract_payload.update(self._contract_payload_from_ib(ib_contract))
            contract_payload['symbol'] = symbol

            redis_key = f'positions:open:{symbol}:{position_id}'
            existing_json = await self.redis.get(redis_key)
            existing_position = json.loads(existing_json) if existing_json else None

            if existing_position:
                position = existing_position
                existing_qty = float(position.get('quantity', 0) or 0)
                existing_commission = float(position.get('commission', 0.0) or 0.0)
                existing_entry_price = float(position.get('entry_price', avg_price) or avg_price)
            else:
                existing_qty = 0.0
                existing_commission = 0.0
                existing_entry_price = avg_price
                position = {
                    'id': position_id,
                    'symbol': symbol,
                    'quantity': 0,
                    'commission': 0.0,
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'stop_order_id': None,
                    'target_order_ids': [],
                    'oca_group': None,
                    'status': 'OPEN',
                    'targets': signal.get('targets', []),
                    'stop_loss': signal.get('stop'),
                }

            new_quantity = existing_qty + total_quantity
            position['quantity'] = new_quantity
            position['commission'] = existing_commission + total_commission
            position['entry_price'] = self._weighted_entry_price(
                existing_qty,
                existing_entry_price,
                total_quantity,
                avg_price,
            )

            position['contract'] = contract_payload
            position['side'] = signal.get('side')
            normalized_strategy = self._normalize_strategy(signal.get('strategy'))
            if normalized_strategy:
                position['strategy'] = normalized_strategy
            elif position.get('strategy'):
                position['strategy'] = self._normalize_strategy(position.get('strategy'))
            if not existing_position or not position.get('entry_time'):
                position['entry_time'] = entry_time
            if signal.get('stop') is not None:
                position['stop_loss'] = signal.get('stop')

            targets_from_signal = signal.get('targets')
            if targets_from_signal is not None:
                position['targets'] = targets_from_signal
            position['current_price'] = avg_price
            position['order_id'] = order_id
            position['signal_id'] = signal.get('id')
            position['reconciled'] = False
            position['ib_account'] = account
            position['ib_con_id'] = con_id
            if signal.get('position_size') is not None:
                position['position_notional'] = signal.get('position_size')

            position.setdefault('realized_posted', float(position.get('realized_posted', 0.0) or 0.0))
            position.setdefault('realized_unposted', float(position.get('realized_unposted', 0.0) or 0.0))

            bracket_plan = self.bracket_registry.pop(order_id, None)
            if bracket_plan:
                position['stop_order_id'] = bracket_plan.get('stop_order_id')
                position['target_order_ids'] = bracket_plan.get('target_order_ids') or []
                position['oca_group'] = bracket_plan.get('oca_group')
                position['stop_engine_state'] = bracket_plan.get('stop_engine_state')
                position['stop_engine_plan'] = bracket_plan.get('plan_records')
                position['stop_loss'] = bracket_plan.get('initial_stop')
                position['targets'] = [plan.get('price') for plan in bracket_plan.get('plan_records', [])]
                basis = bracket_plan.get('basis_price')
                if basis:
                    position['basis_price'] = basis
                else:
                    position.setdefault('basis_price', position.get('entry_price'))
                self.stop_orders[position_id] = bracket_plan.get('stop_trade')
                if bracket_plan.get('target_trades'):
                    self.target_orders[position_id] = bracket_plan['target_trades']
                else:
                    self.target_orders.pop(position_id, None)
                self.position_brackets[position_id] = bracket_plan
            else:
                # Legacy fallback for manual/unsupervised fills
                await self._rebuild_bracket(position)

            # Store position in Redis
            await self.redis.set(redis_key, json.dumps(position))

            # Add to symbol index and update open-position count only when new
            added = await self.redis.sadd(f'positions:by_symbol:{symbol}', position_id)
            if not existing_position and added:
                await self._increment_position_count(added)

            # Update metrics
            await self.redis.incr('execution:fills:total')
            await self.redis.incr('execution:fills:daily')

            # Calculate and store commission
            await self.redis.incrbyfloat('execution:commission:total', total_commission)

            await self._ensure_market_data_subscription(symbol, contract_payload)

            # Clean up pending order
            await self.redis.delete(f'orders:pending:{order_id}')
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

            # Log fill
            self.logger.info(
                "execution_position_created",
                extra={
                    "action": "position_created",
                    "symbol": symbol,
                    "side": position['side'],
                    "quantity": position['quantity'],
                    "entry_price": round(position['entry_price'], 4),
                    "commission": round(total_commission, 4),
                    "position_id": position_id,
                },
            )

            await self._enqueue_distribution_signal(
                signal,
                position,
                avg_price,
                total_quantity,
                total_commission,
                entry_time,
                order_id,
            )

            # Store fill record
            fill_record = {
                'order_id': order_id,
                'position_id': position_id,
                'symbol': symbol,
                'side': signal.get('side'),
                'quantity': total_quantity,
                'price': avg_price,
                'commission': total_commission,
                'time': entry_time
            }

            await self.redis.lpush(
                f'orders:fills:{datetime.now().strftime("%Y%m%d")}',
                json.dumps(fill_record)
            )

            await self._acknowledge_signal(
                signal,
                status='FILLED',
                filled=total_quantity,
                avg_price=avg_price,
            )

        except Exception as e:
            self.logger.error(f"Error handling fill: {e}")

    async def place_stop_loss(self, position: dict):
        """
        Legacy wrapper that delegates to the unified bracket builder.
        """
        try:
            await self._rebuild_bracket(position)
        except Exception as e:
            self.logger.error(f"Error placing stop loss: {e}")

    async def handle_order_rejection(
        self,
        order,
        reason: str,
        order_snapshot: Optional[Dict[str, Any]] = None,
    ):
        """
        Handle rejected orders.
        """
        try:
            order_id = getattr(order, 'orderId', None)
            if order_id is None and order_snapshot:
                order_id = order_snapshot.get('order_id') or order_snapshot.get('orderId')

            if order_id is None:
                self.logger.error(f"Unable to determine order ID for rejection ({reason})")
                return

            self.bracket_registry.pop(order_id, None)

            # Log rejection
            self.logger.error(f"Order {order_id} rejected: {reason}")

            # Update metrics
            await self.redis.incr('execution:rejections:total')
            await self.redis.incr(f'execution:rejections:reason:{reason}')

            # Store rejection record
            rejection_record = {
                'order_id': order_id,
                'reason': reason,
                'time': time.time(),
                'order_details': order_snapshot or self.pending_orders.get(order_id, {})
            }

            await self.redis.lpush(
                'execution:rejections:log',
                json.dumps(rejection_record)
            )
            await self.redis.ltrim('execution:rejections:log', 0, 999)  # Keep last 1000

            # Alert if critical
            if reason in ['Insufficient Buying Power', 'Margin Violation']:
                await self.redis.publish(
                    'alerts:critical',
                    json.dumps({
                        'type': 'order_rejection',
                        'severity': 'critical',
                        'reason': reason,
                        'order_id': order_id,
                        'time': time.time()
                    })
                )

            # Clean up pending order
            await self.redis.delete(f'orders:pending:{order_id}')
            cached_signal = None
            if order_id in self.pending_orders:
                cached_signal = self.pending_orders[order_id].get('signal')
                del self.pending_orders[order_id]
            self.active_trades.pop(order_id, None)
            watcher = self._order_watchers.pop(order_id, None)
            if watcher:
                watcher.cancel()

            symbol = None
            if cached_signal and isinstance(cached_signal, dict):
                symbol = cached_signal.get('symbol')
            if not symbol and isinstance(order_snapshot, dict):
                symbol = order_snapshot.get('symbol')
            await self._maybe_release_market_data(symbol)

            signal_payload = cached_signal or (
                (order_snapshot or {}).get('signal')
                if isinstance(order_snapshot, dict)
                else None
            )
            status_payload = (reason or 'REJECTED').upper()
            await self._acknowledge_signal(
                signal_payload,
                status=status_payload,
                filled=(order_snapshot or {}).get('filled', 0) if isinstance(order_snapshot, dict) else 0,
                avg_price=(order_snapshot or {}).get('avg_fill_price') if isinstance(order_snapshot, dict) else None,
                reason=(reason or 'rejected').lower(),
                broadcast=True,
            )

        except Exception as e:
            self.logger.error(f"Error handling rejection: {e}")

    async def get_existing_position_symbols(self) -> list:
        """
        Get list of symbols with open positions.

        Returns:
            List of symbols with positions
        """
        try:
            symbol_sets = await self._scan_keys('positions:by_symbol:*')
            symbols = [key.split(':', 2)[2] for key in symbol_sets if ':' in key]
            return symbols

        except Exception as e:
            self.logger.error(f"Error getting position symbols: {e}")
            return []

    def on_order_status(self, trade):
        """
        Handle order status updates from IBKR.
        """
        try:
            order_id = trade.order.orderId
            status = trade.orderStatus.status

            # Update pending orders
            if order_id in self.pending_orders:
                self.pending_orders[order_id]['status'] = status
                self.pending_orders[order_id]['filled'] = trade.orderStatus.filled
                self.pending_orders[order_id]['remaining'] = trade.orderStatus.remaining

            self.logger.debug(f"Order {order_id} status: {status}")

        except Exception as e:
            self.logger.error(f"Error handling order status: {e}")

    def on_fill(self, trade, fill):
        """
        Handle fill events from IBKR.
        """
        try:
            self.logger.info(
                "execution_fill_received",
                extra={
                    "action": "fill",
                    "symbol": getattr(fill.contract, 'symbol', None),
                    "shares": getattr(fill.execution, 'shares', None),
                    "price": round(getattr(fill.execution, 'price', 0.0), 4),
                    "order_id": getattr(fill.execution, 'orderId', None),
                },
            )

            async def _process_fill():
                handled = await self.handle_bracket_fill(trade, fill)
                if not handled:
                    await self._handle_untracked_fill(fill)
                await self.update_fill_metrics(fill)

            asyncio.create_task(_process_fill())

        except Exception as e:
            self.logger.error(f"Error handling fill: {e}")

    def on_error(self, reqId, errorCode, errorString, contract):
        """
        Handle error events from IBKR.
        """
        try:
            # Log error
            self.logger.error(f"IBKR Error {errorCode}: {errorString} (reqId={reqId})")

            # Handle specific error codes
            if errorCode == 201:  # Order rejected
                asyncio.create_task(self.handle_order_rejection_by_id(reqId, errorString))
            elif errorCode == 202:  # Order cancelled
                self.logger.info(
                    "execution_order_cancelled",
                    extra={"action": "order_cancelled", "order_id": reqId}
                )
            elif errorCode in [2104, 2106, 2158]:  # Connection OK messages
                self.logger.info(
                    "execution_ibkr_info",
                    extra={"action": "ibkr_info", "message": errorString, "code": errorCode}
                )
            elif errorCode == 1100:  # Connection lost
                self.logger.critical("IBKR connection lost!")
                self.connected = False

        except Exception as e:
            self.logger.error(f"Error handling IBKR error: {e}")

    async def handle_bracket_fill(self, trade, fill) -> bool:
        """
        Handle fills for stop loss and target orders.
        """
        try:
            order_id = getattr(fill.execution, 'orderId', None)
            if not order_id:
                return False

            symbol = fill.contract.symbol
            filled_qty = abs(getattr(fill.execution, 'shares', 0))
            fill_price = getattr(fill.execution, 'price', 0)
            commission = 0.0
            realized_override = None
            if fill.commissionReport:
                commission = self._extract_commission(fill)
                realized_val = getattr(fill.commissionReport, 'realizedPNL', None)
                if realized_val is not None:
                    realized_override = float(realized_val)

            # Check if this is a stop or target order by checking all positions
            position_ids = await self.redis.smembers(f'positions:by_symbol:{symbol}')

            for position_id in position_ids or []:
                key = f'positions:open:{symbol}:{position_id}'
                position_data = await self.redis.get(key)
                if not position_data:
                    continue

                position = json.loads(position_data)
                stop_order_id = position.get('stop_order_id')
                target_order_ids = position.get('target_order_ids', [])

                # Check if this fill is for a stop or target order
                if order_id == stop_order_id or order_id in target_order_ids:
                    if realized_override is not None:
                        realized_pnl = realized_override
                    else:
                        multiplier = self._position_multiplier(position)
                        entry_ref = self._safe_float(position.get('entry_price'))
                        if position.get('side') == 'LONG':
                            realized_pnl = (fill_price - entry_ref) * filled_qty * multiplier
                        else:
                            realized_pnl = (entry_ref - fill_price) * filled_qty * multiplier
                        if commission:
                            realized_pnl -= commission

                    reason = 'STOP' if order_id == stop_order_id else 'TARGET'

                    if reason == 'STOP':
                        await self._apply_position_reduction(
                            position,
                            filled_qty,
                            fill_price,
                            reason,
                            commission=commission,
                            realized_override=realized_pnl,
                            execution=getattr(fill, 'execution', None),
                        )
                    else:
                        await self._handle_target_fill(
                            position,
                            fill_price,
                            filled_qty,
                            realized_pnl,
                            commission,
                            order_id,
                        )
                    return True

        except Exception as e:
            self.logger.error(f"Error handling bracket fill: {e}")
        return False

    async def _handle_untracked_fill(self, fill) -> None:
        """Handle fills that are not tied to tracked entry/stop/target orders."""

        order_id = getattr(fill.execution, 'orderId', None)
        if order_id and order_id in self.pending_orders:
            # Entry order – handled by handle_fill
            return

        account = getattr(fill.execution, 'acctNumber', None)
        con_id = getattr(fill.contract, 'conId', None)
        symbol = getattr(fill.contract, 'symbol', None)
        if not (account and con_id and symbol):
            return

        redis_key = f'positions:open:{symbol}:{account}:{con_id}'
        position_json = await self.redis.get(redis_key)
        if not position_json:
            return

        position = json.loads(position_json)
        side = (position.get('side') or '').upper()
        fill_side = (getattr(fill.execution, 'side', '') or '').upper()
        filled_qty = abs(getattr(fill.execution, 'shares', 0))
        fill_price = getattr(fill.execution, 'price', 0.0)

        position_id = position.get('id')
        if position_id:
            self._manual_fill_positions.add(position_id)
            self._missing_manual_fill_alerted.discard(position_id)

        commission = 0.0
        realized_override = None
        if fill.commissionReport:
            commission = self._extract_commission(fill)
            realized_val = getattr(fill.commissionReport, 'realizedPNL', None)
            if realized_val is not None:
                realized_override = float(realized_val)

        if filled_qty == 0 or side not in {'LONG', 'SHORT'} or fill_side not in {'BOT', 'SLD'}:
            return

        if (side == 'LONG' and fill_side == 'SLD') or (side == 'SHORT' and fill_side == 'BOT'):
            # Reducing or closing position
            await self._apply_position_reduction(
                position,
                filled_qty,
                fill_price,
                'MANUAL',
                commission=commission,
                realized_override=realized_override,
                execution=getattr(fill, 'execution', None),
            )
        elif (side == 'LONG' and fill_side == 'BOT') or (side == 'SHORT' and fill_side == 'SLD'):
            # Scaling into existing position
            await self._apply_position_addition(
                position,
                filled_qty,
                fill_price,
                commission=commission,
                execution=getattr(fill, 'execution', None),
            )

    async def _apply_position_addition(
        self,
        position: Dict[str, Any],
        added_qty: float,
        fill_price: float,
        commission: float = 0.0,
        execution: Optional[Any] = None,
    ) -> None:
        symbol = position.get('symbol')
        position_id = position.get('id')
        if not symbol or not position_id:
            return

        current_qty = float(position.get('quantity', 0) or 0.0)
        new_qty = current_qty + added_qty
        if new_qty <= 0:
            return

        position['entry_price'] = self._weighted_entry_price(
            current_qty,
            float(position.get('entry_price', fill_price) or fill_price),
            added_qty,
            fill_price,
        )
        position['quantity'] = new_qty
        position['current_price'] = fill_price
        position['last_fill_price'] = fill_price
        position['basis_price'] = position['entry_price']
        position['last_update'] = time.time()
        if commission:
            position['commission'] = float(position.get('commission', 0.0) or 0.0) + commission
        if execution is not None:
            fills = position.setdefault('fill_history', [])
            fills.append(
                {
                    'type': 'ADD',
                    'quantity': added_qty,
                    'price': fill_price,
                    'commission': commission,
                    'timestamp': datetime.utcnow().isoformat(),
                    'exec_id': getattr(execution, 'execId', None),
                    'side': getattr(execution, 'side', None),
                }
            )

        await self.redis.set(
            f'positions:open:{symbol}:{position_id}',
            json.dumps(position)
        )

        await self._rebuild_bracket(position)

    async def _apply_position_reduction(
        self,
        position: Dict[str, Any],
        closed_qty: float,
        fill_price: float,
        reason: str,
        commission: float = 0.0,
        realized_override: Optional[float] = None,
        execution: Optional[Any] = None,
    ) -> None:
        symbol = position.get('symbol')
        position_id = position.get('id')
        if not symbol or not position_id:
            return

        current_qty = float(position.get('quantity', 0) or 0.0)
        closed_qty = min(closed_qty, current_qty)
        if closed_qty <= 0:
            return

        multiplier = self._position_multiplier(position)
        entry_price = float(position.get('entry_price', 0) or 0.0)
        side = (position.get('side') or '').upper()

        if realized_override is not None:
            realized_delta = float(realized_override)
        else:
            if side == 'LONG':
                realized_delta = (fill_price - entry_price) * closed_qty * multiplier
            else:
                realized_delta = (entry_price - fill_price) * closed_qty * multiplier
            if commission:
                realized_delta -= commission

        new_quantity = max(current_qty - closed_qty, 0)
        position['quantity'] = new_quantity
        await self._register_realized_event(position, realized_delta)
        position['last_fill_price'] = fill_price
        position['basis_price'] = position.get('entry_price')
        position.setdefault('reductions', []).append(
            {
                'quantity': closed_qty,
                'price': fill_price,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'realized': realized_delta,
                'commission': commission,
                'multiplier': multiplier,
            }
        )
        if commission:
            position['commission'] = float(position.get('commission', 0.0) or 0.0) + commission
        if execution is not None:
            events = position.setdefault('realized_events', [])
            events.append(
                {
                    'type': reason,
                    'quantity': closed_qty,
                    'price': fill_price,
                    'commission': commission,
                    'realized': realized_delta,
                    'timestamp': datetime.utcnow().isoformat(),
                    'exec_id': getattr(execution, 'execId', None),
                    'side': getattr(execution, 'side', None),
                    'remaining_quantity': position['quantity'],
                }
            )

        if new_quantity <= 0:
            await self._finalize_position_close(position, fill_price, realized_delta, reason, closed_qty)
        else:
            await self.redis.set(
                f'positions:open:{symbol}:{position_id}',
                json.dumps(position)
            )
            await self._rebuild_bracket(position)

    async def update_fill_metrics(self, fill):
        """
        Update fill metrics in Redis.
        """
        try:
            commission = self._extract_commission(fill)
            await self.redis.incrbyfloat('execution:commission:daily', commission)
            await self.redis.hincrby('execution:fills:by_symbol', fill.contract.symbol, 1)
        except Exception as e:
            self.logger.error(f"Error updating fill metrics: {e}")

    async def handle_order_rejection_by_id(self, order_id: int, reason: str):
        """
        Handle order rejection by order ID.
        """
        order_data = self.pending_orders.get(order_id)
        if not order_data:
            payload = await self.redis.get(f'orders:pending:{order_id}')
            if payload:
                try:
                    order_data = json.loads(payload)
                except json.JSONDecodeError:
                    order_data = None

        await self.handle_order_rejection(None, reason, order_data)

    async def update_order_status(self):
        """
        Update status of all pending orders.
        """
        trades_by_id = {trade.order.orderId: trade for trade in self.ib.trades()}

        for order_id, order_data in list(self.pending_orders.items()):
            trade = trades_by_id.get(order_id)
            signal = order_data.get('signal', {}) if isinstance(order_data, dict) else {}

            if trade:
                snapshot = await self._sync_trade_state(order_id, trade, signal)
                status = snapshot['status']

                if status == 'Filled':
                    await self.handle_fill(trade, signal)
                    self.active_trades.pop(order_id, None)
                elif status in {'Cancelled', 'ApiCancelled'}:
                    self.logger.warning(f"Order {order_id} cancelled during status reconciliation")
                    await self.redis.delete(f'orders:pending:{order_id}')
                    del self.pending_orders[order_id]
                    self.active_trades.pop(order_id, None)
                elif status in {'Inactive', 'Rejected'}:
                    await self.handle_order_rejection(trade.order, status, order_data)
            else:
                if time.time() - order_data.get('placed_at', 0) > 300:
                    self.logger.warning(f"Removing stale order {order_id}")
                    await self.handle_order_rejection(None, 'Timeout', order_data)
                    del self.pending_orders[order_id]
