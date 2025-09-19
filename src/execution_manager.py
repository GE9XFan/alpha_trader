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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
import redis.asyncio as aioredis
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, StopOrder, Order, ExecutionFilter, util

from signal_deduplication import contract_fingerprint
from src.option_utils import normalize_expiry
from src.stop_engine import StopEngine, StopState
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

        # Connection state
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        self.logger = get_logger(__name__, component="execution", subsystem="manager")

        # Tracking for manual fills to detect missing execution events
        self._manual_fill_positions: Set[str] = set()
        self._missing_manual_fill_alerted: Set[str] = set()

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

        position['quantity'] = 0
        position['status'] = 'CLOSED'
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.utcnow().isoformat()
        position['close_reason'] = reason
        position['stop_engine_plan'] = []

        prior_realized = float(position.get('realized_pnl') or 0.0)
        delta = float(realized_pnl_delta or 0.0)
        unposted = float(position.pop('realized_unposted', 0.0) or 0.0)
        realized_total = prior_realized
        if realized_total == 0.0 and (unposted or delta):
            realized_total = unposted if unposted else delta
        position['realized_pnl'] = realized_total
        position['commission'] = float(position.get('commission') or 0.0)

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

        await self.redis.incr('positions:closed:total')
        await self.redis.incrbyfloat('positions:pnl:realized:total', realized_total)
        await self.redis.incrbyfloat('risk:daily_pnl', realized_total if unposted else delta)

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
            position['realized_pnl'] = position.get('realized_pnl', 0.0) + realized_pnl

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
            await self.place_stop_loss(position)

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
        position.pop('realized_unposted', None)
        position['commission'] = float(position.get('commission') or 0.0)
        if realized_total:
            await self.redis.incrbyfloat('positions:pnl:realized:total', realized_total)
            await self.redis.incrbyfloat('risk:daily_pnl', realized_total)

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

        # Ensure Redis state reflects live IB positions
        await self.reconcile_positions()

        # Set up event handlers (ib_insync specific events)
        self.ib.orderStatusEvent += self.on_order_status
        self.ib.execDetailsEvent += self.on_fill  # execDetailsEvent for fills in ib_insync
        self.ib.errorEvent += self.on_error

        # Start position sync task (runs every 30 seconds)
        asyncio.create_task(self.position_sync_loop())

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

                    # Get account info (ib_insync doesn't have async version)
                    account_values = self.ib.accountValues()
                    for av in account_values:
                        if av.tag == 'NetLiquidation':
                            await self.redis.set('account:value', av.value)
                        elif av.tag == 'BuyingPower':
                            await self.redis.set('account:buying_power', av.value)

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

                raw_redis_keys = await self.redis.keys('positions:open:*')
                redis_positions: Dict[str, Dict[str, Any]] = {}

                for raw_key in raw_redis_keys:
                    key = raw_key.decode('utf-8') if isinstance(raw_key, bytes) else raw_key
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
            existing_positions = await self.redis.keys(f'positions:open:{symbol}:*')
            if len(existing_positions) >= self.max_per_symbol:
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
                    continue

                # Check risk approval
                if await self.passes_risk_checks(signal):
                    await self.execute_signal(signal)

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid signal JSON: {e}")

    async def _acknowledge_signal(
        self,
        signal: Optional[Dict[str, Any]],
        *,
        status: str,
        filled: float = 0.0,
        avg_price: Optional[float] = None,
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
            open_positions = await self.redis.keys('positions:open:*')
            if len(open_positions) >= self.max_positions:
                self.logger.warning(f"Max positions limit reached: {len(open_positions)}/{self.max_positions}")
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
                return

            # Get current market data
            ticker = self.ib.reqMktData(ib_contract, '', False, False)
            await asyncio.sleep(0.5)  # Wait for data

            # Calculate order size
            order_size = await self.calculate_order_size(signal, ticker)
            if order_size == 0:
                self.logger.warning(f"Order size is 0 for {symbol}")
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
                return

            # Determine order type and price
            contract_type = str(contract_info.get('type') or '').strip().lower()
            option_types = {'option', 'options', 'opt'}
            side_upper = (side or '').upper()

            if contract_type in option_types:
                # Directional options trades are opened with long premium (buy puts/calls)
                action = 'BUY'
            else:
                action = 'BUY' if side_upper == 'LONG' else 'SELL'

            if confidence > 85 and strategy_code != '0DTE':  # Never use market for 0DTE
                # Market order for high confidence (except 0DTE)
                order = MarketOrder(action, order_size)
                order_type = 'MARKET'
            else:
                # Limit order
                limit_price = None
                bid = self._safe_price(getattr(ticker, 'bid', None))
                ask = self._safe_price(getattr(ticker, 'ask', None))

                if bid is not None and ask is not None:
                    spread = max(ask - bid, 0)
                    if confidence >= 70:
                        limit_price = (bid + ask) / 2
                    else:
                        if action == 'BUY':
                            limit_price = bid + spread * 0.33
                        else:
                            limit_price = ask - spread * 0.33

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

                if limit_price is None:
                    self.logger.warning(
                        "Falling back to market order due to missing price",
                        extra={
                            "symbol": symbol,
                            "reason": "no_valid_limit_price",
                            "strategy": strategy_code,
                        },
                    )
                    order = MarketOrder(action, order_size)
                    order_type = 'MARKET'
                else:
                    order = LimitOrder(action, order_size, limit_price)
                    order_type = 'LIMIT'

            # Place order
            trade = self.ib.placeOrder(ib_contract, order)
            order_id = trade.order.orderId

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
                'stop_loss': signal.get('stop'),
                'targets': signal.get('targets', []),
                'placed_at': time.time(),
                'status': 'PENDING',
                'signal': signal,
            }

            self.pending_orders[order_id] = order_data

            await self._persist_pending_order(order_id)

            # Track trade for reconciliation/updates
            self.active_trades[order_id] = {'trade': trade, 'signal': signal}

            try:
                # Monitor order
                await self.monitor_order(trade, signal)
            finally:
                self.active_trades.pop(order_id, None)

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
            # Get account value and buying power
            buying_power = float(await self.redis.get('account:buying_power') or 100000)
            account_value = float(await self.redis.get('account:value') or 100000)

            # Get risk multiplier if set by risk manager
            risk_multiplier = float(await self.redis.get('risk:position_size_multiplier') or 1.0)

            confidence = signal.get('confidence', 60) / 100.0
            strategy = signal.get('strategy')
            strategy_code = self._normalize_strategy(strategy)
            contract_payload = signal.get('contract', {})
            raw_type = contract_payload.get('type') or 'option'
            contract_type = str(raw_type).strip().lower()

            # Base position size: 2-5% of account based on confidence
            base_position_size = account_value * (0.02 + 0.03 * confidence) * risk_multiplier

            # Cap at 25% of buying power
            max_position_size = buying_power * 0.25
            position_size = min(base_position_size, max_position_size)
            signal['position_size'] = round(position_size, 2)

            if contract_type in {'option', 'options', 'opt'}:
                # For options, calculate number of contracts
                # Use mid price or last price
                if ticker.bid and ticker.ask:
                    option_price = (ticker.bid + ticker.ask) / 2
                elif ticker.last:
                    option_price = ticker.last
                else:
                    # Fallback to estimated price from signal
                    option_price = signal.get('entry', 1.0)

                if option_price > 0:
                    # Calculate contracts (each contract = 100 shares)
                    contracts = int(position_size / (option_price * 100))

                    # Apply strategy-specific limits
                    if strategy_code == '0DTE':
                        max_contracts = min(self.max_0dte_contracts, 50)
                    else:
                        max_contracts = min(self.max_other_contracts, 100)

                    # Apply confidence-based scaling
                    if confidence < 0.7:
                        max_contracts = int(max_contracts * 0.5)
                    elif confidence < 0.85:
                        max_contracts = int(max_contracts * 0.75)

                    # Minimum 1 contract, maximum based on strategy
                    order_size = max(1, min(contracts, max_contracts))
                    signal['order_size'] = order_size
                    return order_size
                else:
                    signal['order_size'] = 1
                    return 1  # Default to 1 contract

            elif contract_type in {'stock', 'equity'}:
                # For stocks, calculate number of shares
                if ticker.last:
                    stock_price = ticker.last
                elif ticker.bid and ticker.ask:
                    stock_price = (ticker.bid + ticker.ask) / 2
                else:
                    stock_price = signal.get('entry', 100)

                if stock_price > 0:
                    shares = int(position_size / stock_price)
                    # Round to nearest 100 for better liquidity
                    shares = max(100, (shares // 100) * 100)
                    signal['order_size'] = shares
                    return shares
                else:
                    signal['order_size'] = 100
                    return 100  # Default to 100 shares
            else:
                self.logger.warning(
                    "Unknown contract type '%s'; defaulting to option sizing", raw_type
                )
                if ticker.bid and ticker.ask:
                    option_price = (ticker.bid + ticker.ask) / 2
                elif ticker.last:
                    option_price = ticker.last
                else:
                    option_price = signal.get('entry', 1.0)

                if option_price > 0:
                    contracts = int(position_size / (option_price * 100))
                    order_size = max(1, contracts)
                    signal['order_size'] = order_size
                    return order_size
                signal['order_size'] = 1
                return 1

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

        await self.handle_order_rejection(trade.order, 'Timeout', order_snapshot)

    async def monitor_order(self, trade, signal: dict):
        """
        Monitor order until completion.
        """
        order_id = trade.order.orderId
        base_timeout = 60 if trade.order.orderType == 'LMT' else 30
        start_time = time.time()

        while True:
            await asyncio.sleep(0.2)

            snapshot = await self._sync_trade_state(order_id, trade, signal)
            status = snapshot['status']
            filled = snapshot['filled']
            remaining = snapshot['remaining']

            if status == 'Filled':
                await self.handle_fill(trade, signal)
                break

            if status in {'Cancelled', 'ApiCancelled'}:
                self.logger.warning(f"Order {order_id} cancelled ({status})")
                await self.redis.delete(f'orders:pending:{order_id}')
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                await self._acknowledge_signal(
                    signal,
                    status=status.upper(),
                    filled=snapshot.get('filled', 0),
                    avg_price=snapshot.get('avg_price'),
                )
                break

            if status in {'Inactive', 'Rejected'}:
                await self.handle_order_rejection(trade.order, status)
                break

            timeout_window = base_timeout * (2 if filled else 1)
            if time.time() - start_time >= timeout_window:
                await self._escalate_timeout(order_id, trade, signal, filled, timeout_window)
                break

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

            # Place stop loss order
            await self.place_stop_loss(position)

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
        Place stop loss order for new position.
        """
        try:
            # Create IB contract (add symbol to contract info)
            contract_info = position.get('contract', {})
            contract_info['symbol'] = position.get('symbol')  # Add symbol to contract dict
            ib_contract = await self.create_ib_contract(contract_info)
            if not ib_contract:
                self.logger.error(f"Failed to create contract for stop loss")
                return

            # Determine stop and target actions (opposite of entry)
            side = position.get('side')
            close_action = 'SELL' if side == 'LONG' else 'BUY'

            quantity = position.get('quantity') or 0
            if quantity <= 0:
                self.logger.warning("Cannot place bracket orders without quantity")
                return

            position_id = position.get('id')
            symbol = position.get('symbol')

            existing_stop_trade = self.stop_orders.get(position_id)
            existing_target_trades = self.target_orders.get(position_id, [])
            existing_stop_id = position.get('stop_order_id')
            existing_target_ids = list(position.get('target_order_ids') or [])

            entry_price = float(position.get('entry_price') or 0)
            strategy = position.get('strategy')
            strategy_code = self._normalize_strategy(strategy)

            instrument_type = contract_info.get('type') or contract_info.get('secType') or 'default'
            state = StopState.from_dict(position.get('stop_engine_state'))

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
            if not allocations:
                raw_targets = position.get('targets', []) or []
                for target in raw_targets:
                    try:
                        plan_records.append(
                            {
                                'profit_pct': None,
                                'price': round(float(target), 4),
                                'fraction': 1.0,
                                'quantity': int(quantity),
                                'status': 'pending',
                                'order_id': None,
                            }
                        )
                    except (TypeError, ValueError):
                        continue

                if not plan_records and entry_price:
                    fallback_raw = self.stop_engine.target_price(side, entry_price, 30.0)
                    fallback_rounded = self.stop_engine.quantize_price(
                        instrument_type,
                        side,
                        fallback_raw,
                    )
                    default_price = round(fallback_rounded, 4)
                    plan_records.append(
                        {
                            'profit_pct': 30.0,
                            'price': default_price,
                            'fraction': 1.0,
                            'quantity': int(quantity),
                            'status': 'pending',
                            'order_id': None,
                        }
                    )
            else:
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

            sanitized_position = re.sub(r'[^A-Za-z0-9]+', '_', str(position_id or 'UNKNOWN'))
            base_group = position.get('oca_group') or f"OCA_{sanitized_position}"
            base_group = re.sub(r'[^A-Za-z0-9_]+', '_', base_group)
            oca_group = f"{base_group}_{int(time.time())}"

            # Create and submit stop order
            if state.last_stop_type == 'TRAIL_PERCENT':
                stop_order = Order()
                stop_order.orderType = 'TRAIL'
                stop_order.trailingPercent = round(state.last_stop_value or 0.0, 6)
                stop_order.trailStopPrice = round(stop_price, 4)
            else:
                stop_order = StopOrder(close_action, quantity, round(stop_price, 4))
            stop_order.action = close_action
            stop_order.totalQuantity = quantity
            stop_order.ocaGroup = oca_group
            stop_order.ocaType = 1
            stop_order.tif = 'GTC'
            stop_order.outsideRth = True
            stop_trade = self.ib.placeOrder(ib_contract, stop_order)
            stop_order_id = stop_trade.order.orderId

            # Create profit-taking orders for each allocation (sequential OCA bucket)
            target_order_ids: List[int] = []
            target_trades: List[Any] = []
            for idx, plan in enumerate(plan_records):
                alloc_qty = int(plan.get('quantity') or 0)
                target_price = float(plan.get('price') or 0)
                if alloc_qty <= 0 or target_price <= 0:
                    continue

                limit_order = LimitOrder(close_action, alloc_qty, target_price)
                limit_order.ocaGroup = oca_group
                limit_order.ocaType = 1
                limit_order.tif = 'GTC'

                target_trade = self.ib.placeOrder(ib_contract, limit_order)
                target_trades.append(target_trade)
                target_order_ids.append(target_trade.order.orderId)
                plan['order_id'] = target_trade.order.orderId
                plan['status'] = 'working'

                self.logger.info(
                    "execution_target_order",
                    extra={
                        "action": "target_order",
                        "position_id": position_id,
                        "side": close_action,
                        "quantity": alloc_qty,
                        "target_price": round(target_price, 4),
                        "stage_index": idx,
                    },
                )

            # Update position with bracket metadata
            position['stop_order_id'] = stop_order_id
            position['target_order_ids'] = target_order_ids
            position['oca_group'] = oca_group
            position['stop_loss'] = stop_price
            position['targets'] = [plan['price'] for plan in plan_records]
            position['stop_engine_state'] = state.to_dict()
            position['stop_engine_plan'] = plan_records

            await self.redis.set(f'positions:open:{symbol}:{position_id}', json.dumps(position))

            # Track active orders
            self.stop_orders[position_id] = stop_trade
            if target_trades:
                self.target_orders[position_id] = target_trades
            else:
                self.target_orders.pop(position_id, None)

            self.logger.info(
                "execution_stop_order",
                extra={
                    "action": "stop_order",
                    "position_id": position_id,
                    "side": close_action,
                    "quantity": quantity,
                    "stop_price": round(stop_price, 4),
                },
            )

            if any((existing_stop_trade, existing_target_trades, existing_stop_id, existing_target_ids)):
                await self._retire_previous_bracket(
                    position_id,
                    stop_trade=existing_stop_trade,
                    target_trades=existing_target_trades,
                    stop_order_id=existing_stop_id,
                    target_order_ids=existing_target_ids,
                )

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

            signal_payload = cached_signal or (
                (order_snapshot or {}).get('signal')
                if isinstance(order_snapshot, dict)
                else None
            )
            await self._acknowledge_signal(
                signal_payload,
                status=reason.upper(),
                filled=(order_snapshot or {}).get('filled', 0) if isinstance(order_snapshot, dict) else 0,
                avg_price=(order_snapshot or {}).get('avg_fill_price') if isinstance(order_snapshot, dict) else None,
            )

        except Exception as e:
            self.logger.error(f"Error handling rejection: {e}")

        pass

    async def get_existing_position_symbols(self) -> list:
        """
        Get list of symbols with open positions.

        Returns:
            List of symbols with positions
        """
        try:
            position_keys = await self.redis.keys('positions:open:*')
            symbols = set()

            for key in position_keys:
                # Extract symbol from key format: positions:open:{symbol}:{position_id}
                parts = key.split(':')
                if len(parts) >= 4:
                    symbols.add(parts[2])

            return list(symbols)

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
            position_keys = await self.redis.keys(f'positions:open:{symbol}:*')

            for key in position_keys:
                position_data = await self.redis.get(key)
                if not position_data:
                    continue

                position = json.loads(position_data)
                stop_order_id = position.get('stop_order_id')
                target_order_ids = position.get('target_order_ids', [])

                # Check if this fill is for a stop or target order
                if order_id == stop_order_id or order_id in target_order_ids:
                    multiplier = self._position_multiplier(position)
                    if realized_override is not None:
                        realized_pnl = realized_override
                    else:
                        if position.get('side') == 'LONG':
                            realized_pnl = (fill_price - position.get('entry_price', 0)) * filled_qty * multiplier
                        else:
                            realized_pnl = (position.get('entry_price', 0) - fill_price) * filled_qty * multiplier
                        if commission:
                            realized_pnl -= commission

                    reason = 'STOP' if order_id == stop_order_id else 'TARGET'

                    position['commission'] = float(position.get('commission', 0.0) or 0.0) + commission
                    events = position.setdefault('realized_events', [])
                    events.append(
                        {
                            'type': reason,
                            'quantity': filled_qty,
                            'price': fill_price,
                            'commission': commission,
                            'realized': realized_pnl,
                            'timestamp': datetime.utcnow().isoformat(),
                            'exec_id': getattr(fill.execution, 'execId', None),
                            'side': getattr(fill.execution, 'side', None),
                        }
                    )

                    if reason == 'STOP':
                        await self._finalize_position_close(position, fill_price, realized_pnl, reason, filled_qty)
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

        # Cancel existing protective orders before resizing them
        await self._cancel_bracket_orders(position)

        position['entry_price'] = self._weighted_entry_price(
            current_qty,
            float(position.get('entry_price', fill_price) or fill_price),
            added_qty,
            fill_price,
        )
        position['quantity'] = new_qty
        position['current_price'] = fill_price
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

        await self._place_bracket_orders(position, quantity=int(new_qty))

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
        prior_realized = float(position.get('realized_pnl') or 0.0)
        position['realized_pnl'] = prior_realized + realized_delta
        unposted = float(position.get('realized_unposted', 0.0) or 0.0) + realized_delta
        position['realized_unposted'] = unposted
        position.setdefault('reductions', []).append(
            {
                'quantity': closed_qty,
                'price': fill_price,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'realized': realized_delta,
                'commission': commission,
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
            await self._cancel_bracket_orders(position)
            await self.redis.set(
                f'positions:open:{symbol}:{position_id}',
                json.dumps(position)
            )
            await self._place_bracket_orders(position, quantity=int(new_quantity))

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
                    del self.pending_orders[order_id]
                    await self.redis.delete(f'orders:pending:{order_id}')
