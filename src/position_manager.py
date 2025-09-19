"""
Position Manager Module

Manages position lifecycle including P&L tracking, stop management, scaling,
and exit conditions. Monitors all open positions and handles trailing stops.

Redis Keys Used:
    Read:
        - positions:open:* (all open positions)
        - market:{symbol}:quote (market data)
        - options:chain:{symbol} (options chain data)
        - positions:greeks:{position_id} (position Greeks)
    Write:
        - health:positions:heartbeat (health monitoring)
        - positions:open:{symbol}:{position_id} (position updates)
        - positions:closed:{date}:{position_id} (closed positions)
        - positions:by_symbol:{symbol} (symbol index)
        - positions:pnl:unrealized (total unrealized P&L)
        - positions:pnl:realized:total (cumulative realized P&L)
        - positions:summary (portfolio summary)
        - positions:closed:total (closed position counter)

Author: QuantiCity Capital
Version: 3.0.0
"""

import asyncio
import inspect
import json
import time
import logging
import math
from datetime import datetime, timezone
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
import pytz
import redis.asyncio as aioredis
from ib_insync import IB, Stock, Option, LimitOrder, MarketOrder, StopOrder, Order

from src.stop_engine import StopEngine, StopState


class PositionManager:
    """
    Manage position lifecycle including P&L tracking and exit management.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        redis_conn: aioredis.Redis,
        ib_connection=None,
        contract_resolver: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None,
    ):
        """
        Initialize position manager with configuration.
        Can share IBKR connection with ExecutionManager or create its own.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Can share IB connection or create new one
        self.ib = ib_connection if ib_connection else IB()
        self._contract_resolver = contract_resolver

        # Position tracking
        self.positions = {}  # position_id -> position data
        self.position_tickers = {}  # symbol -> ticker subscription

        # Stop management
        self.stop_orders = {}  # position_id -> stop order
        self.target_orders = {}  # position_id -> list of target orders
        self.stop_states: Dict[str, StopState] = {}

        # P&L tracking
        self.high_water_marks = {}  # position_id -> max profit
        self._pending_price_updates: Set[str] = set()

        risk_config = self.config.get('risk_management', {})
        self.stop_engine = StopEngine.from_risk_config(risk_config)
        expiry_cfg = risk_config.get('expiry_protection', {}) if isinstance(risk_config, dict) else {}
        self.expiry_alert_levels: List[int] = sorted(
            {int(level) for level in expiry_cfg.get('alert_days', [5, 3, 1]) if level >= 0},
            reverse=True,
        )
        self.expiry_force_close_days: int = int(expiry_cfg.get('force_close_days', 0))
        default_eod_rules = risk_config.get('eod_rules') or [
            {'strategy': '0DTE', 'max_dte': 0, 'action': 'close', 'reason': 'EOD expiry avoidance'},
            {'strategy': '1DTE', 'max_dte': 1, 'action': 'reduce', 'reduction': 0.5, 'reason': 'EOD risk trim'},
        ]
        # Copy rules to avoid mutating shared config structures
        self.eod_rules = [dict(rule) for rule in default_eod_rules]

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

    @staticmethod
    def _normalize_strategy(strategy: Optional[str]) -> str:
        return str(strategy).strip().upper() if strategy else ''

    @staticmethod
    def _compute_holding_minutes(entry_time: Optional[str], exit_time: Optional[str]) -> Optional[float]:
        if not entry_time or not exit_time:
            return None

        try:
            start = datetime.fromisoformat(entry_time)
            end = datetime.fromisoformat(exit_time)
        except ValueError:
            return None

        delta = (end - start).total_seconds()
        if delta < 0:
            return None

        return round(delta / 60.0, 2)

    @staticmethod
    def _calculate_return_pct(realized_pnl: Optional[float], position_notional: Optional[float]) -> Optional[float]:
        if realized_pnl is None or position_notional in (None, 0):
            return None

        try:
            return round(float(realized_pnl) / float(position_notional), 4)
        except (TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def _classify_result(realized_pnl: Optional[float]) -> Optional[str]:
        if realized_pnl is None:
            return None
        if realized_pnl > 0:
            return 'WIN'
        if realized_pnl < 0:
            return 'LOSS'
        return 'BREAKEVEN'

    @staticmethod
    def _map_event_to_status(event_type: str) -> str:
        mapping = {
            'EXIT': 'CLOSED',
            'SCALE_OUT': 'PARTIAL',
            'ENTRY': 'FILLED',
        }
        return mapping.get(event_type.upper(), event_type.upper())

    async def _publish_lifecycle_event(
        self,
        position: Dict[str, Any],
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Emit lifecycle events into the downstream distribution queue."""

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

        # Remove keys with None values to keep payload compact
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            await self.redis.lpush('signals:distribution:pending', json.dumps(payload))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to enqueue lifecycle distribution payload",
                extra={'position_id': position.get('id'), 'event': event_type, 'error': str(exc)}
            )

    def _close_action(self, position: Dict[str, Any]) -> str:
        side = (position.get('side') or '').upper()
        return 'SELL' if side == 'LONG' else 'BUY'

    def _find_trade_by_id(self, order_id: Optional[int]):
        if order_id is None:
            return None

        for trade in self.stop_orders.values():
            if trade and getattr(trade.order, 'orderId', None) == order_id:
                return trade

        for trades in self.target_orders.values():
            for trade in trades:
                if trade and getattr(trade.order, 'orderId', None) == order_id:
                    return trade

        for trade in self.ib.trades():
            if getattr(trade.order, 'orderId', None) == order_id:
                return trade

        return None

    def _get_stop_state(self, position: Dict[str, Any]) -> StopState:
        position_id = position.get('id')
        if not position_id:
            return StopState()

        state = self.stop_states.get(position_id)
        if state:
            return state

        state = StopState.from_dict(position.get('stop_engine_state'))
        self.stop_states[position_id] = state
        return state

    def _ensure_stop_trade_reference(self, position: Dict[str, Any]):
        position_id = position.get('id')
        if not position_id:
            return None

        trade = self.stop_orders.get(position_id)
        if trade and not trade.isDone():
            return trade

        order_id = position.get('stop_order_id')
        if order_id is None:
            return None

        trade = self._find_trade_by_id(order_id)
        if trade and not trade.isDone():
            self.stop_orders[position_id] = trade
            return trade
        return None

    async def _cancel_order_id(self, order_id: Optional[int]) -> None:
        if order_id is None or not self.ib.isConnected():
            return

        trade = self._find_trade_by_id(order_id)
        if not trade:
            return

        if trade.isDone():
            return

        try:
            self.ib.cancelOrder(trade.order)
        except Exception as cancel_error:
            self.logger.error(f"Failed to cancel order {order_id}: {cancel_error}")

    async def _cancel_stop_order(self, position: Dict[str, Any]) -> None:
        position_id = position.get('id')
        stop_order_id = position.get('stop_order_id')
        await self._cancel_order_id(stop_order_id)
        position['stop_order_id'] = None
        if position_id in self.stop_orders:
            self.stop_orders.pop(position_id, None)

    async def _cancel_target_orders(self, position: Dict[str, Any]) -> None:
        position_id = position.get('id')
        target_ids = position.get('target_order_ids') or []
        for order_id in target_ids:
            await self._cancel_order_id(order_id)
        position['target_order_ids'] = []
        position['stop_engine_plan'] = []
        if position_id in self.target_orders:
            self.target_orders.pop(position_id, None)

    async def _submit_stop_order(
        self,
        position: Dict[str, Any],
        *,
        stop_price: float,
        stop_type: Optional[str],
        stop_value: Optional[float],
        quantity: int,
        oca_group: str,
    ) -> Optional[Any]:
        if quantity <= 0 or not self.ib.isConnected():
            return None

        ib_contract = await self._resolve_contract(position.get('contract'))
        if not ib_contract:
            return None

        action = self._close_action(position)
        if (stop_type or '').upper() == 'TRAIL_PERCENT':
            order = Order()
            order.orderType = 'TRAIL'
            order.trailingPercent = round(stop_value or 0.0, 6)
            order.trailStopPrice = round(stop_price, 4)
        else:
            order = StopOrder(action, int(quantity), round(stop_price, 4))

        order.action = action
        order.totalQuantity = int(quantity)
        order.ocaGroup = oca_group
        order.ocaType = 1
        order.tif = 'GTC'
        order.outsideRth = True

        stop_trade = self.ib.placeOrder(ib_contract, order)
        self.stop_orders[position.get('id')] = stop_trade
        return stop_trade

    async def _update_existing_stop_order(
        self,
        position: Dict[str, Any],
        stop_price: float,
        quantity: int,
    ) -> bool:
        await self._place_bracket_orders(position, quantity=quantity)
        return True

    async def _submit_target_orders(
        self,
        position: Dict[str, Any],
        plan_records: List[Dict[str, Any]],
        oca_group: str,
    ) -> List[Any]:
        if not plan_records or not self.ib.isConnected():
            return []

        ib_contract = await self._resolve_contract(position.get('contract'))
        if not ib_contract:
            return []

        action = self._close_action(position)
        trades: List[Any] = []
        for idx, plan in enumerate(plan_records):
            qty_val = plan.get('quantity') or 0
            if isinstance(qty_val, float) and (math.isnan(qty_val) or math.isinf(qty_val)):
                qty_val = 0
            qty = int(qty_val)
            price = float(plan.get('price') or 0)
            if qty <= 0 or price <= 0:
                continue

            limit_order = LimitOrder(action, qty, price)
            limit_order.ocaGroup = oca_group
            limit_order.ocaType = 1
            limit_order.tif = 'GTC'
            trade = self.ib.placeOrder(ib_contract, limit_order)
            trades.append(trade)
            plan['order_id'] = trade.order.orderId
            plan['status'] = 'working'
            plan['stage_index'] = idx

        position_id = position.get('id')
        if trades:
            self.target_orders[position_id] = trades
        else:
            self.target_orders.pop(position_id, None)
        return trades

    async def _place_bracket_orders(
        self,
        position: Dict[str, Any],
        *,
        quantity: Optional[int] = None,
    ) -> None:
        qty = int(quantity if quantity is not None else position.get('quantity', 0) or 0)
        if qty <= 0:
            return

        await self._cancel_stop_order(position)
        await self._cancel_target_orders(position)

        entry_price = float(position.get('entry_price', 0) or 0)
        side = (position.get('side') or '').upper()
        contract_type = position.get('contract', {}).get('type', 'stock')
        state = self._get_stop_state(position)

        stop_price, state = self.stop_engine.initial_stop(
            contract_type,
            side,
            entry_price,
            state=state,
        )
        stop_price = round(float(stop_price), 4)

        raw_group = position.get('oca_group') or f"OCA_{position.get('id')}"
        sanitized_group = re.sub(r'[^A-Za-z0-9_]+', '_', str(raw_group))
        oca_group = (sanitized_group[:60] or 'OCA_DEFAULT') + f"_{int(time.time())}"

        stop_trade = await self._submit_stop_order(
            position,
            stop_price=stop_price,
            stop_type=state.last_stop_type,
            stop_value=state.last_stop_value,
            quantity=qty,
            oca_group=oca_group,
        )
        if not stop_trade:
            return

        position_id = position.get('id')

        allocations = self.stop_engine.plan_targets(
            contract_type,
            side,
            entry_price,
            max(qty, 0),
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
                        'quantity': alloc.quantity,
                        'status': 'pending',
                        'order_id': None,
                    }
                )
        else:
            fallback_price = self.stop_engine.target_price(side, entry_price, 30.0)
            fallback_price = self.stop_engine.quantize_price(contract_type, side, fallback_price)
            plan_records.append(
                {
                    'profit_pct': 30.0,
                    'price': round(float(fallback_price), 4),
                    'fraction': 1.0,
                    'quantity': qty,
                    'status': 'pending',
                    'order_id': None,
                }
            )

        target_trades = await self._submit_target_orders(position, plan_records, oca_group)

        position['oca_group'] = oca_group
        position['stop_order_id'] = stop_trade.order.orderId
        position['stop_loss'] = stop_price
        position['target_order_ids'] = [trade.order.orderId for trade in target_trades]
        position['targets'] = [plan['price'] for plan in plan_records]
        position['stop_engine_plan'] = plan_records
        self.stop_states[position_id] = state
        position['stop_engine_state'] = state.to_dict()

        symbol = position.get('symbol')
        if symbol:
            await self.redis.set(
                f'positions:open:{symbol}:{position_id}',
                json.dumps(position)
            )

    async def start(self):
        """
        Main position management loop.
        Processing frequency: Every 1 second for monitoring, 5 seconds for P&L updates
        """
        self.logger.info("Starting position manager...")

        # Ensure IBKR connection if not shared
        if not self.ib.isConnected():
            await self.connect_ibkr()

        await self.reconcile_positions()

        last_pnl_update = 0

        while True:
            try:
                current_time = time.time()

                # Load all open positions from Redis
                await self.load_positions()

                # Update P&L every 2 seconds (reduced from 5 for faster stop detection)
                if current_time - last_pnl_update >= 2:
                    await self.update_all_positions_pnl()
                    last_pnl_update = current_time

                # Check for exit conditions (backup for IBKR stops)
                await self.check_exit_conditions()

                dirty_ids = await self.refresh_market_prices()
                if dirty_ids:
                    await self.manage_trailing_stops(candidate_ids=dirty_ids)
                else:
                    await self.manage_trailing_stops()

                await self.enforce_stop_invariants()
                await self.enforce_expiry_rules()

                # Check targets for scaling
                await self.check_targets()

                # Handle EOD for 0DTE positions
                await self.handle_eod_positions()

                # Update heartbeat
                await self.redis.setex('health:positions:heartbeat', 15, current_time)

            except Exception as e:
                self.logger.error(f"Error in position management loop: {e}")

            await asyncio.sleep(1)

    async def connect_ibkr(self):
        """
        Connect to IBKR if not using shared connection.
        """
        try:
            ibkr_config = self.config.get('ibkr', {})
            host = ibkr_config.get('host', '127.0.0.1')
            port = ibkr_config.get('port', 7497)
            client_id = ibkr_config.get('client_id', 1) + 200  # Different from execution

            await self.ib.connectAsync(host, port, clientId=client_id, timeout=10)
            self.logger.info(f"Position manager connected to IBKR (client_id={client_id})")

        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")

    async def reconcile_positions(self):
        """Ensure Redis mirrors the live IBKR positions on startup."""
        if not self.ib.isConnected():
            return

        try:
            positions = await self.ib.reqPositionsAsync()
        except Exception as exc:  # pragma: no cover - network guard
            self.logger.warning(f"Unable to reconcile IB positions: {exc}")
            return

        for account, contract, size, avg_cost in positions:
            try:
                symbol = contract.symbol
                position_id = f"{account}:{contract.conId}"
                redis_key = f'positions:open:{symbol}:{position_id}'

                if await self.redis.exists(redis_key):
                    continue

                try:
                    quantity = abs(float(size))
                except (TypeError, ValueError):
                    self.logger.warning(
                        "Skipping IBKR position with unparseable size %s for %s",
                        size,
                        symbol,
                    )
                    continue

                if quantity <= 0:
                    # Ignore reconciling flat snapshots
                    continue

                contract_payload = {
                    'symbol': contract.symbol,
                    'secType': contract.secType,
                    'exchange': contract.exchange,
                    'currency': contract.currency,
                }
                if getattr(contract, 'lastTradeDateOrContractMonth', None):
                    contract_payload['expiry'] = contract.lastTradeDateOrContractMonth
                if getattr(contract, 'strike', None) is not None:
                    contract_payload['strike'] = contract.strike
                    contract_payload['right'] = getattr(contract, 'right', None)
                    contract_payload['type'] = 'option'
                else:
                    contract_payload['type'] = contract_payload.get('secType', 'STK').lower()

                if isinstance(quantity, float) and quantity.is_integer():
                    quantity = int(quantity)

                side = 'LONG' if size >= 0 else 'SHORT'

                position_snapshot = {
                    'id': position_id,
                    'symbol': symbol,
                    'contract': contract_payload,
                    'side': side,
                    'strategy': 'RECONCILED',
                    'quantity': quantity,
                    'entry_price': avg_cost,
                    'entry_time': datetime.utcnow().isoformat(),
                    'commission': 0.0,
                    'stop_loss': None,
                    'targets': [],
                    'current_price': avg_cost,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'stop_order_id': None,
                    'status': 'OPEN',
                    'order_id': None,
                    'signal_id': None,
                    'reconciled': True,
                }

                await self.redis.set(redis_key, json.dumps(position_snapshot))
                added = await self.redis.sadd(f'positions:by_symbol:{symbol}', position_id)
                if added:
                    await self._increment_position_count(added)
                self.positions[position_id] = position_snapshot
            except Exception as snapshot_error:  # pragma: no cover - per-position guard
                self.logger.error(
                    f"Failed to reconcile position for {getattr(contract, 'symbol', 'UNKNOWN')}: {snapshot_error}"
                )

    async def load_positions(self):
        """
        Load all open positions from Redis.
        """
        try:
            position_keys = await self.redis.keys('positions:open:*')

            redis_position_ids = set()

            for key in position_keys:
                position_json = await self.redis.get(key)
                if position_json:
                    position = json.loads(position_json)
                    position_id = position.get('id')

                    if not position_id:
                        continue

                    # Store in memory
                    self.positions[position_id] = position
                    self.stop_states[position_id] = StopState.from_dict(
                        position.get('stop_engine_state')
                    )
                    redis_position_ids.add(position_id)

                    # Subscribe to market data if needed
                    symbol = position.get('symbol')
                    if symbol not in self.position_tickers:
                        await self.subscribe_market_data(symbol, position.get('contract'))

            # Update the position count to match reality
            await self.redis.set('positions:count', len(redis_position_ids))

            # Remove any cached positions that no longer exist in Redis
            stale_ids = set(self.positions.keys()) - redis_position_ids
            for stale_id in stale_ids:
                stale_position = self.positions.pop(stale_id, None)
                self.high_water_marks.pop(stale_id, None)
                self.stop_orders.pop(stale_id, None)
                self.stop_states.pop(stale_id, None)

                if not stale_position:
                    continue

                symbol = stale_position.get('symbol')
                if not symbol:
                    continue

                # Drop ticker subscriptions with no remaining positions
                if all(pos.get('symbol') != symbol for pos in self.positions.values()):
                    ticker = self.position_tickers.pop(symbol, None)
                    if ticker:
                        try:
                            self.ib.cancelMktData(ticker)
                        except Exception as cancel_error:  # pragma: no cover - defensive guard
                            self.logger.debug(
                                "Failed to cancel market data for %s: %s",
                                symbol,
                                cancel_error,
                            )

        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")

    async def subscribe_market_data(self, symbol: str, contract_info: dict):
        """
        Subscribe to market data for position tracking.
        """
        try:
            ib_contract = await self._resolve_contract(contract_info)

            if ib_contract:
                ticker = self.ib.reqMktData(ib_contract, '', False, False)
                self.position_tickers[symbol] = ticker
                self.logger.debug(f"Subscribed to market data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error subscribing to market data: {e}")

    async def _resolve_contract(self, contract_info: Optional[Dict[str, Any]]):
        """Resolve an IB contract using an injected resolver or local builder."""
        if not contract_info:
            return None

        if self._contract_resolver:
            try:
                contract = self._contract_resolver(contract_info)
                if inspect.isawaitable(contract):
                    contract = await contract
                if contract:
                    return contract
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.error(f"Injected contract resolver failed: {exc}")

        return await self._build_contract(contract_info)

    async def _build_contract(self, contract_info: Dict[str, Any]):
        """Build and qualify an IB contract directly from contract metadata."""
        try:
            contract_type = contract_info.get('type', 'stock')
            symbol = contract_info.get('symbol')

            if contract_type == 'option':
                occ_symbol = contract_info.get('occ_symbol')
                if occ_symbol:
                    occ_re = re.compile(r'^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$')
                    match = occ_re.match(occ_symbol)
                    if match:
                        root = match.group('root')
                        date_str = match.group('date')
                        right = 'C' if match.group('cp') == 'C' else 'P'
                        strike = float(match.group('strike')) / 1000
                        expiry = f"20{date_str}"
                        symbol = root
                    else:
                        expiry = contract_info.get('expiry')
                        strike = contract_info.get('strike')
                        right = contract_info.get('right', 'C')
                else:
                    expiry = contract_info.get('expiry') or contract_info.get('lastTradeDateOrContractMonth')
                    strike = contract_info.get('strike')
                    right = contract_info.get('right', 'C')

                contract = Option(symbol, expiry, strike, right, 'SMART', currency='USD')
                contract.multiplier = '100'
            else:
                contract = Stock(symbol, 'SMART', 'USD')

            qualified = await self.ib.qualifyContractsAsync(contract)
            return qualified[0] if qualified else None
        except Exception as exc:
            self.logger.error(f"Failed to resolve contract for {contract_info.get('symbol')}: {exc}")
            return None

    async def update_all_positions_pnl(self):
        """
        Update P&L for all positions.
        """
        total_unrealized_pnl = 0

        for position_id, position in self.positions.items():
            try:
                quantity_value = position.get('quantity')
                try:
                    quantity = float(quantity_value)
                except (TypeError, ValueError):
                    quantity = 0

                status_value = position.get('status', 'OPEN')
                status = str(status_value).upper() if status_value is not None else 'OPEN'

                if quantity <= 0 or status != 'OPEN':
                    continue

                symbol = position.get('symbol')

                # Get current price from ticker or Redis
                current_price = await self.get_current_price(symbol, position.get('contract'))

                if current_price:
                    # Calculate P&L
                    unrealized_pnl = await self.calculate_unrealized_pnl(position, current_price)

                    # Update position
                    position['current_price'] = current_price
                    position['unrealized_pnl'] = unrealized_pnl
                    position['last_update'] = time.time()

                    # Track high water mark for trailing stops
                    if position_id not in self.high_water_marks:
                        self.high_water_marks[position_id] = unrealized_pnl
                    else:
                        self.high_water_marks[position_id] = max(
                            self.high_water_marks[position_id],
                            unrealized_pnl
                        )

                    # Get Greeks for options from Alpha Vantage data in Redis
                    if position.get('contract', {}).get('type') == 'option':
                        greeks = await self.get_option_greeks(position)
                        if greeks:
                            position['greeks'] = greeks

                    # Update in Redis
                    await self.redis.set(
                        f'positions:open:{symbol}:{position_id}',
                        json.dumps(position)
                    )

                    # Accumulate totals
                    total_unrealized_pnl += unrealized_pnl

            except Exception as e:
                self.logger.error(f"Error updating position {position_id}: {e}")

        # Update account totals
        await self.redis.set('positions:pnl:unrealized', total_unrealized_pnl)

    async def get_current_price(self, symbol: str, contract_info: dict) -> float:
        """
        Get current price from ticker or Redis.
        """
        try:
            # First try ticker if we have it
            if symbol in self.position_tickers:
                ticker = self.position_tickers[symbol]
                if ticker.last:
                    return ticker.last
                elif ticker.bid and ticker.ask:
                    return (ticker.bid + ticker.ask) / 2

            # Fallback to Redis market data
            market_data = await self.redis.get(f'market:{symbol}:quote')
            if market_data:
                data = json.loads(market_data)
                return data.get('last', data.get('mid', 0))

            # For options, try to get from options chain
            if contract_info.get('type') == 'option':
                chain_data = await self.redis.get(f'options:chain:{symbol}')
                if chain_data:
                    chain = json.loads(chain_data)
                    occ_symbol = contract_info.get('occ_symbol')
                    option_data = None

                    if isinstance(chain, dict):
                        by_contract = chain.get('by_contract') if 'by_contract' in chain else None
                        if occ_symbol and isinstance(by_contract, dict):
                            option_data = by_contract.get(occ_symbol)
                        elif occ_symbol:
                            option_data = chain.get(occ_symbol)

                    if option_data:
                        # Prefer mark, then last, then derived mid
                        mark_price = option_data.get('mark') or option_data.get('last_price')
                        if mark_price is not None:
                            return mark_price
                        if option_data.get('last') is not None:
                            return option_data.get('last')
                        bid = option_data.get('bid')
                        ask = option_data.get('ask')
                        if bid is not None and ask is not None:
                            return (bid + ask) / 2

            return None

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def calculate_unrealized_pnl(self, position: dict, current_price: float) -> float:
        """
        Calculate unrealized P&L for position.

        Returns:
            Unrealized P&L in dollars
        """
        try:
            entry_price = position.get('entry_price', 0)
            quantity = position.get('quantity', 0)
            side = position.get('side')
            contract_type = position.get('contract', {}).get('type', 'stock')

            if contract_type == 'option':
                # Options have 100x multiplier
                multiplier = 100
                if side == 'LONG':
                    pnl = (current_price - entry_price) * quantity * multiplier
                else:  # SHORT
                    pnl = (entry_price - current_price) * quantity * multiplier
            else:
                # Stocks
                if side == 'LONG':
                    pnl = (current_price - entry_price) * quantity
                else:  # SHORT
                    pnl = (entry_price - current_price) * quantity

            # Subtract commission
            commission = position.get('commission', 0)
            pnl -= commission

            return round(pnl, 2)

        except Exception as e:
            self.logger.error(f"Error calculating P&L: {e}")
            return 0

    async def get_option_greeks(self, position: dict) -> dict:
        """
        Get option Greeks from Alpha Vantage data stored in Redis.
        """
        try:
            symbol = position.get('symbol')
            contract = position.get('contract', {})
            occ_symbol = contract.get('occ_symbol')

            if not occ_symbol:
                return None

            # Get options chain from Redis (stored by Alpha Vantage)
            chain_key = f'options:chain:{symbol}'
            chain_data = await self.redis.get(chain_key)

            if chain_data:
                chain = json.loads(chain_data)
                option_data = None

                by_contract = chain.get('by_contract') if isinstance(chain, dict) else None
                if occ_symbol and isinstance(by_contract, dict):
                    option_data = by_contract.get(occ_symbol)
                elif occ_symbol and isinstance(chain, dict):
                    option_data = chain.get(occ_symbol)

                if option_data:
                    quantity = position.get('quantity', 0)

                    def scaled(value):
                        return value * quantity * 100 if value is not None else 0

                    greeks = {
                        'delta': scaled(option_data.get('delta')),
                        'gamma': scaled(option_data.get('gamma')),
                        'theta': scaled(option_data.get('theta')),
                        'vega': scaled(option_data.get('vega')),
                        'rho': scaled(option_data.get('rho')),
                        'iv': option_data.get('implied_volatility', 0)
                    }

                    # Store position Greeks
                    await self.redis.setex(
                        f'positions:greeks:{position.get("id")}',
                        60,
                        json.dumps(greeks)
                    )

                    return greeks

            return None

        except Exception as e:
            self.logger.error(f"Error getting Greeks: {e}")
            return None

    async def manage_trailing_stops(self, *, force_refresh: bool = False):
        """Evaluate dynamic stop adjustments for every open position."""
        for position_id, position in self.positions.items():
            try:
                status = str(position.get('status', 'OPEN')).upper()
                if status != 'OPEN':
                    continue

                quantity = int(position.get('quantity', 0) or 0)
                if quantity <= 0:
                    continue

                entry_price_raw = position.get('entry_price')
                current_price_raw = position.get('current_price')
                try:
                    entry_price = float(entry_price_raw)
                    current_price = float(current_price_raw)
                except (TypeError, ValueError):
                    continue

                if entry_price <= 0 or current_price <= 0:
                    continue

                contract_type = position.get('contract', {}).get('type', 'stock')
                side = (position.get('side') or '').upper()
                if side not in {'LONG', 'SHORT'}:
                    continue

                state = self._get_stop_state(position)
                current_stop_raw = position.get('stop_loss')
                try:
                    current_stop = float(current_stop_raw) if current_stop_raw is not None else None
                except (TypeError, ValueError):
                    current_stop = None

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

                trade = self._ensure_stop_trade_reference(position)

                has_active_stop = trade is not None or position.get('stop_order_id') is not None
                if not has_active_stop or force_refresh:
                    self.stop_states[position_id] = decision.state
                    position['stop_engine_state'] = decision.state.to_dict()
                    await self._place_bracket_orders(position)
                    if not has_active_stop:
                        self.logger.warning(
                            "Rebuilt missing stop for %s at %.4f",
                            position_id,
                            decision.stop_price or 0.0,
                        )
                    continue

                if decision.should_update and decision.stop_price is not None:
                    previous_state = StopState.from_dict(
                        self.stop_states[position_id].to_dict()
                    ) if position_id in self.stop_states else None
                    self.stop_states[position_id] = decision.state
                    position['stop_engine_state'] = decision.state.to_dict()
                    success = await self._update_existing_stop_order(position, decision.stop_price, quantity)
                    if success:
                        position['stop_loss'] = decision.stop_price
                        state = decision.state
                        self.stop_states[position_id] = state
                        position['stop_engine_state'] = state.to_dict()
                        symbol = position.get('symbol')
                        if symbol:
                            await self.redis.set(
                                f'positions:open:{symbol}:{position_id}',
                                json.dumps(position)
                            )
                        self.logger.info(
                            "dynamic_stop_update",
                            extra={
                                "position_id": position_id,
                                "symbol": position.get('symbol'),
                                "side": side,
                                "instrument": contract_type,
                                "reason": decision.reason,
                                "profit_pct": round(decision.profit_pct * 100, 2),
                                "stop_price": round(decision.stop_price, 4),
                                "improvement": round(decision.improvement, 4),
                            },
                        )
                    else:
                        if previous_state is not None:
                            self.stop_states[position_id] = previous_state
                            position['stop_engine_state'] = previous_state.to_dict()

            except Exception as e:
                self.logger.error(f"Error managing trailing stop for {position_id}: {e}")

    def _calculate_notional(self, position: Dict[str, Any]) -> float:
        """Calculate the absolute notional exposure of a position."""
        try:
            contract_type = position.get('contract', {}).get('type', 'stock')
            quantity = position.get('quantity', 0)
            entry_price = position.get('entry_price', 0)
            multiplier = 100 if contract_type == 'option' else 1
            return abs(entry_price * quantity * multiplier)
        except Exception:
            return 0.0

    async def update_stop_loss(self, position: dict, new_stop: float):
        """
        Update stop loss order with IBKR.
        """
        try:
            if new_stop is None or not self.ib.isConnected():
                return

            stop_price = round(float(new_stop), 4)

            quantity = int(position.get('quantity', 0) or 0)
            if quantity <= 0:
                return

            success = await self._update_existing_stop_order(position, stop_price, quantity)
            if not success:
                return

            state = self._get_stop_state(position)
            state.last_stop_price = stop_price
            state.last_reason = 'manual_update'
            state.last_update_ts = time.time()

            position_id = position.get('id')
            position['stop_loss'] = stop_price
            position['stop_engine_state'] = state.to_dict()

            symbol = position.get('symbol')
            if symbol and position_id:
                await self.redis.set(
                    f'positions:open:{symbol}:{position_id}',
                    json.dumps(position)
                )

        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")

    async def refresh_market_prices(self) -> Set[str]:
        """Update cached prices from IBKR tickers and return affected positions."""
        dirty: Set[str] = set()
        for symbol, ticker in self.position_tickers.items():
            price = None
            try:
                if getattr(ticker, 'last', None):
                    price = ticker.last
                elif getattr(ticker, 'close', None):
                    price = ticker.close
                elif getattr(ticker, 'marketPrice', None):
                    price = ticker.marketPrice
            except Exception:
                price = None

            if price is None or price <= 0:
                continue

            for position_id, position in self.positions.items():
                if position.get('symbol') != symbol or str(position.get('status', 'OPEN')).upper() != 'OPEN':
                    continue

                try:
                    previous = float(position.get('current_price') or 0.0)
                except (TypeError, ValueError):
                    previous = 0.0

                profile = self.stop_engine.profile_for(position.get('contract', {}).get('type', 'stock'))
                if abs(previous - price) >= profile.min_tick:
                    position['current_price'] = price
                    dirty.add(position_id)
        return dirty

    async def enforce_stop_invariants(self) -> None:
        """Ensure every open position has a live protective stop."""
        for position_id, position in self.positions.items():
            if str(position.get('status', 'OPEN')).upper() != 'OPEN':
                continue
            trade = self._ensure_stop_trade_reference(position)
            if trade or position.get('stop_order_id'):
                continue
            self.logger.warning("Stop invariant rebuild triggered for %s", position_id)
            await self._place_bracket_orders(position)

    async def enforce_expiry_rules(self) -> None:
        """Apply simple expiry-based risk controls for option positions."""
        if not self.expiry_alert_levels and self.expiry_force_close_days <= 0:
            return

        today = datetime.now(timezone.utc).date()

        for position_id, position in list(self.positions.items()):
            contract = position.get('contract') or {}
            if contract.get('type') != 'option':
                continue

            expiry_str = contract.get('expiry') or contract.get('lastTradeDateOrContractMonth')
            if not expiry_str:
                continue

            try:
                expiry_date = datetime.strptime(expiry_str[:8], '%Y%m%d').date()
            except ValueError:
                continue

            days_to_expiry = (expiry_date - today).days

            if self.expiry_force_close_days and days_to_expiry <= self.expiry_force_close_days:
                price = position.get('current_price') or position.get('entry_price') or 0.0
                await self.close_position(position, price, 'EXPIRY_PROTECTION')
                self.logger.warning(
                    "Force closed %s (%s) due to expiry window (%d days)",
                    position.get('symbol'),
                    position_id,
                    days_to_expiry,
                )
                continue

            for level in self.expiry_alert_levels:
                if days_to_expiry == level:
                    self.logger.warning(
                        "Expiry alert for %s (%s): %d days remaining",
                        position.get('symbol'),
                        position_id,
                        days_to_expiry,
                    )
                    break

    async def check_targets(self):
        """
        Check if price targets hit for scaling out.
        """
        for position_id, position in self.positions.items():
            try:
                if position.get('stop_engine_plan'):
                    # Active bracket orders manage profit taking
                    continue
                targets = position.get('targets', [])
                if not targets:
                    continue

                current_price = position.get('current_price', 0)
                current_target_index = position.get('current_target_index', 0)
                side = position.get('side')

                if current_target_index >= len(targets):
                    continue  # All targets hit

                target_price = targets[current_target_index]

                # Check if target hit
                target_hit = False
                if side == 'LONG' and current_price >= target_price:
                    target_hit = True
                elif side == 'SHORT' and current_price <= target_price:
                    target_hit = True

                if target_hit:
                    await self.scale_out(position, target_price, current_target_index)

            except Exception as e:
                self.logger.error(f"Error checking targets for {position_id}: {e}")

    async def scale_out(self, position: dict, target_price: float, target_index: int):
        """
        Scale out of position at target.
        """
        try:
            if position.get('stop_engine_plan'):
                return
            position_id = position.get('id')
            symbol = position.get('symbol')
            current_quantity = int(position.get('quantity', 0) or 0)
            side = position.get('side')

            # Scale percentages for each target
            scale_percentages = [0.33, 0.50, 1.0]
            scale_pct = scale_percentages[min(target_index, 2)]

            # Calculate quantity to close
            if target_index == 0:
                # First target: 33% of original
                close_quantity = max(1, int(current_quantity * scale_pct))
            elif target_index == 1:
                # Second target: 50% of remaining
                close_quantity = max(1, int(current_quantity * scale_pct))
            else:
                # Final target: close all
                close_quantity = current_quantity

            close_quantity = min(close_quantity, current_quantity)
            if close_quantity <= 0 or current_quantity <= 0:
                return

            # Don't scale if less than minimum
            contract_type = position.get('contract', {}).get('type', 'stock')
            if contract_type == 'option' and current_quantity < 3:
                # For options, only scale if we have 3+ contracts
                if target_index == 2:  # Final target, close all
                    close_quantity = current_quantity
                else:
                    return  # Skip scaling

            ib_contract = await self._resolve_contract(position.get('contract'))

            if ib_contract and self.ib.isConnected():
                # Pause existing protective orders while we work the scale-out
                await self._cancel_stop_order(position)
                await self._cancel_target_orders(position)

                action = 'SELL' if side == 'LONG' else 'BUY'
                limit_price = round(float(target_price), 2)
                order = LimitOrder(action, close_quantity, limit_price)
                order.tif = 'GTC'
                trade = self.ib.placeOrder(ib_contract, order)

                fill_deadline = time.time() + 5
                while time.time() < fill_deadline and not trade.isDone():
                    await asyncio.sleep(0.2)

                if trade.isDone() and trade.orderStatus.status == 'Filled':
                    fill_price = trade.fills[-1].execution.price if trade.fills else limit_price

                    position['current_target_index'] = target_index + 1
                    await self._register_partial_close(
                        position,
                        close_quantity,
                        fill_price,
                        f'Target {target_index + 1}'
                    )

                    # Reestablish protective orders for the remaining size
                    if position.get('quantity', 0) > 0:
                        await self._place_bracket_orders(position)
                else:
                    if not trade.isDone():
                        try:
                            self.ib.cancelOrder(trade.order)
                        except Exception as cancel_error:
                            self.logger.error(
                                f"Failed to cancel scale-out order for {position_id}: {cancel_error}"
                            )
                    self.logger.info(
                        f"Scale-out limit order for {symbol} target {target_index + 1} timed out without fill"
                    )
                    # Restore protection since the position size did not change
                    await self._place_bracket_orders(position, quantity=current_quantity)

        except Exception as e:
            self.logger.error(f"Error scaling out position: {e}")

    async def close_position(self, position: dict, exit_price: float, reason: str):
        """
        Close position and update records.
        """
        try:
            position_id = position.get('id')
            symbol = position.get('symbol')

            # Cancel stop order if exists
            stop_order_id = position.get('stop_order_id')
            if stop_order_id and position_id in self.stop_orders:
                stop_trade = self.stop_orders[position_id]
                if stop_trade and not stop_trade.isDone():
                    self.ib.cancelOrder(stop_trade.order)
                del self.stop_orders[position_id]

            # Cancel any outstanding target/limit orders associated with brackets
            target_order_ids = position.get('target_order_ids') or []
            if target_order_ids:
                for order_id in target_order_ids:
                    try:
                        existing_trade = next(
                            (trade for trade in self.ib.trades() if trade.order.orderId == order_id),
                            None
                        )
                        if existing_trade and not existing_trade.isDone():
                            self.ib.cancelOrder(existing_trade.order)
                    except Exception as cancel_error:
                        self.logger.error(
                            f"Failed to cancel target order {order_id} for {position_id}: {cancel_error}"
                        )
            position['target_order_ids'] = []
            position['oca_group'] = None
            position['stop_engine_plan'] = []

            # Update position record
            position['status'] = 'CLOSED'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now().isoformat()
            position['exit_reason'] = reason

            # Calculate final P&L if any quantity remains
            remaining_qty = position.get('quantity', 0)
            entry_price = position.get('entry_price', 0)
            contract_type = position.get('contract', {}).get('type', 'stock')
            side = position.get('side')
            final_pnl = 0.0

            if remaining_qty > 0:
                if contract_type == 'option':
                    if side == 'LONG':
                        final_pnl = (exit_price - entry_price) * remaining_qty * 100
                    else:
                        final_pnl = (entry_price - exit_price) * remaining_qty * 100
                else:
                    if side == 'LONG':
                        final_pnl = (exit_price - entry_price) * remaining_qty
                    else:
                        final_pnl = (entry_price - exit_price) * remaining_qty

            position['realized_pnl'] = position.get('realized_pnl', 0) + final_pnl

            closed_quantity = remaining_qty
            position['quantity'] = 0

            # Move to closed positions
            await self.redis.delete(f'positions:open:{symbol}:{position_id}')
            await self.redis.setex(
                f'positions:closed:{datetime.now().strftime("%Y%m%d")}:{position_id}',
                604800,  # 7 days TTL
                json.dumps(position)
            )

            # Update metrics and symbol index
            removed = await self.redis.srem(f'positions:by_symbol:{symbol}', position_id)
            if removed:
                await self._decrement_position_count(symbol, removed)
            await self.redis.incr('positions:closed:total')
            await self.redis.incrbyfloat('positions:pnl:realized:total', position['realized_pnl'])
            await self.redis.incrbyfloat('risk:daily_pnl', position['realized_pnl'])

            # Remove from memory
            if position_id in self.positions:
                del self.positions[position_id]
            if position_id in self.high_water_marks:
                del self.high_water_marks[position_id]

            # Log closure
            self.logger.info(
                f"Position closed: {symbol} {position.get('side')} "
                f"P&L: ${position['realized_pnl']:.2f} "
                f"Reason: {reason}"
            )

            lifecycle_details = {
                'executed_at': position.get('exit_time'),
                'exit_price': round(float(exit_price), 4) if exit_price is not None else None,
                'quantity': closed_quantity,
                'realized_pnl': round(float(position.get('realized_pnl', 0.0)), 2),
                'return_pct': self._calculate_return_pct(position.get('realized_pnl'), position.get('position_notional')),
                'holding_period_minutes': self._compute_holding_minutes(position.get('entry_time'), position.get('exit_time')),
                'reason': reason,
                'result': self._classify_result(position.get('realized_pnl')),
                'remaining_quantity': position.get('quantity', 0),
            }

            await self._publish_lifecycle_event(position, 'EXIT', lifecycle_details)

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    async def check_exit_conditions(self):
        """
        Check for positions that need to be exited.
        Note: Stop losses are primarily handled by IBKR orders.
        This is a backup check for positions without active IBKR stops.
        """
        for position_id, position in list(self.positions.items()):
            try:
                strategy = position.get('strategy')

                # Skip if position has an active IBKR stop order
                if position.get('stop_order_id') and position_id in self.stop_orders:
                    # IBKR is handling the stop - we don't need to check manually
                    continue

                # Manual stop loss check for positions without IBKR stops
                current_price_raw = position.get('current_price')
                stop_loss_raw = position.get('stop_loss')
                side = position.get('side')

                current_price = None
                stop_loss = None

                try:
                    if current_price_raw is not None:
                        current_price = float(current_price_raw)
                except (TypeError, ValueError):
                    self.logger.debug(
                        "Invalid current price for %s: %r", position_id, current_price_raw
                    )

                try:
                    if stop_loss_raw is not None:
                        stop_loss = float(stop_loss_raw)
                except (TypeError, ValueError):
                    self.logger.debug(
                        "Invalid stop loss for %s: %r", position_id, stop_loss_raw
                    )

                if (
                    stop_loss is not None
                    and stop_loss > 0
                    and current_price is not None
                    and side in {'LONG', 'SHORT'}
                ):
                    if (
                        side == 'LONG' and current_price <= stop_loss
                    ) or (
                        side == 'SHORT' and current_price >= stop_loss
                    ):
                        await self.close_position(position, current_price, 'Stop loss hit')
                        continue

                # Check time-based exits for options
                if position.get('contract', {}).get('type') == 'option':
                    greeks = position.get('greeks', {}) or {}
                    theta_raw = greeks.get('theta')
                    try:
                        theta = float(theta_raw) if theta_raw is not None else 0.0
                    except (TypeError, ValueError):
                        self.logger.debug(
                            "Invalid theta for %s: %r", position_id, theta_raw
                        )
                        theta = 0.0

                    # Exit if theta decay is too high
                    if (
                        strategy == '0DTE'
                        and current_price is not None
                        and abs(theta) > 50  # $50/day decay
                    ):
                        await self.close_position(position, current_price, 'Theta decay limit')

            except Exception as e:
                self.logger.error(f"Error checking exit conditions: {e}")

    def _calculate_dte(self, position: Dict[str, Any]) -> Optional[int]:
        """Compute days-to-expiry for option positions."""
        contract = position.get('contract', {})
        expiry = contract.get('expiry') or contract.get('lastTradeDateOrContractMonth')
        if not expiry:
            return None

        expiry_dt = None
        for fmt in ('%Y%m%d', '%y%m%d'):
            try:
                expiry_dt = datetime.strptime(expiry, fmt)
                break
            except ValueError:
                continue

        if expiry_dt is None:
            try:
                expiry_dt = datetime.fromisoformat(expiry)
            except ValueError:
                return None

        eastern_today = datetime.now(pytz.timezone('US/Eastern')).date()
        return max((expiry_dt.date() - eastern_today).days, 0)

    async def _register_partial_close(
        self,
        position: dict,
        close_quantity: int,
        fill_price: float,
        reason: str,
    ):
        """Record a partial close, updating state, Redis, and metrics."""
        position_id = position.get('id')
        symbol = position.get('symbol')
        side = position.get('side')
        contract_type = position.get('contract', {}).get('type', 'stock')
        entry_price = position.get('entry_price', 0)
        multiplier = 100 if contract_type == 'option' else 1

        if side == 'LONG':
            realized_pnl = (fill_price - entry_price) * close_quantity * multiplier
        else:
            realized_pnl = (entry_price - fill_price) * close_quantity * multiplier

        position['quantity'] = max(position.get('quantity', 0) - close_quantity, 0)
        position['realized_pnl'] = position.get('realized_pnl', 0) + realized_pnl

        reductions = position.setdefault('reductions', [])
        reductions.append({
            'quantity': close_quantity,
            'price': fill_price,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })

        self.logger.info(
            f"Reduced {symbol} by {close_quantity} @ ${fill_price:.2f} ({reason})"
        )

        lifecycle_details = {
            'executed_at': datetime.now().isoformat(),
            'fill_price': round(float(fill_price), 4),
            'quantity': close_quantity,
            'realized_pnl': round(float(realized_pnl), 2),
            'remaining_quantity': position['quantity'],
            'reason': reason,
            'result': 'SCALE_OUT',
        }

        if position['quantity'] <= 0:
            await self.close_position(position, fill_price, reason)
        else:
            await self.redis.set(
                f'positions:open:{symbol}:{position_id}',
                json.dumps(position)
            )
            await self._publish_lifecycle_event(position, 'SCALE_OUT', lifecycle_details)

    async def _reduce_position(self, position: dict, reduction: float, reason: str):
        """Reduce a position by a percentage at market."""
        current_qty = position.get('quantity', 0)
        if current_qty <= 0:
            return

        reduction = max(0.0, min(reduction, 1.0))
        close_quantity = int(max(1, current_qty * reduction))

        if close_quantity >= current_qty:
            await self.close_position(position, position.get('current_price', 0), reason)
            return

        ib_contract = await self._resolve_contract(position.get('contract'))
        if not (ib_contract and self.ib.isConnected()):
            return

        action = 'SELL' if position.get('side') == 'LONG' else 'BUY'
        order = MarketOrder(action, close_quantity)
        trade = self.ib.placeOrder(ib_contract, order)
        await asyncio.sleep(0.5)

        if trade.isDone() and trade.orderStatus.status == 'Filled':
            fill_price = trade.fills[-1].execution.price if trade.fills else position.get('current_price', 0)
            await self._register_partial_close(position, close_quantity, fill_price, reason)

    async def handle_eod_positions(self):
        """
        Handle end-of-day position management.
        """
        try:
            current_time = datetime.now(pytz.timezone('US/Eastern'))

            if current_time.hour < 15 or (current_time.hour == 15 and current_time.minute < 45):
                return

            for position_id, position in list(self.positions.items()):
                strategy = position.get('strategy')
                dte = self._calculate_dte(position)
                risk_level = position.get('risk_level', 'standard')

                for rule in self.eod_rules:
                    if rule.get('strategy') and rule['strategy'] != strategy:
                        continue

                    max_dte = rule.get('max_dte')
                    if max_dte is not None and dte is not None and dte > max_dte:
                        continue

                    allowed_risk = rule.get('risk_levels')
                    if allowed_risk and risk_level not in allowed_risk:
                        continue

                    action = rule.get('action', 'close')
                    reason = rule.get('reason', 'EOD risk rule')

                    if action == 'close':
                        current_price = position.get('current_price', 0)
                        await self.close_position(position, current_price, reason)
                        self.logger.info(f"Closing {strategy} position {position_id[:8]} for EOD rule")
                        break
                    elif action == 'reduce':
                        reduction = rule.get('reduction', 0.5)
                        await self._reduce_position(position, reduction, reason)
                        break
                    elif action == 'hedge':
                        alert = {
                            'type': 'hedge_request',
                            'symbol': position.get('symbol'),
                            'strategy': strategy,
                            'reason': reason,
                            'timestamp': time.time(),
                        }
                        await self.redis.publish('alerts:critical', json.dumps(alert))
                        break

        except Exception as e:
            self.logger.error(f"Error handling EOD positions: {e}")

    async def get_position_summary(self) -> dict:
        """
        Get summary of all open positions.

        Returns:
            Position summary dictionary
        """
        try:
            summary = {
                'total_positions': len(self.positions),
                'total_unrealized_pnl': 0,
                'total_realized_pnl': 0,
                'positions_by_strategy': {},
                'exposure_by_symbol': {},
                'portfolio_greeks': {
                    'delta': 0,
                    'gamma': 0,
                    'theta': 0,
                    'vega': 0
                },
                'positions': []
            }

            for position_id, position in self.positions.items():
                # Accumulate P&L
                summary['total_unrealized_pnl'] += position.get('unrealized_pnl', 0)
                summary['total_realized_pnl'] += position.get('realized_pnl', 0)

                # Count by strategy
                strategy = position.get('strategy', 'unknown')
                summary['positions_by_strategy'][strategy] = \
                    summary['positions_by_strategy'].get(strategy, 0) + 1

                # Calculate exposure
                symbol = position.get('symbol')
                quantity = position.get('quantity', 0)
                current_price = position.get('current_price', 0)

                if position.get('contract', {}).get('type') == 'option':
                    exposure = quantity * 100 * current_price
                else:
                    exposure = quantity * current_price

                summary['exposure_by_symbol'][symbol] = \
                    summary['exposure_by_symbol'].get(symbol, 0) + exposure

                # Aggregate Greeks
                greeks = position.get('greeks', {})
                for greek in ['delta', 'gamma', 'theta', 'vega']:
                    summary['portfolio_greeks'][greek] += greeks.get(greek, 0)

                # Add position detail
                summary['positions'].append({
                    'id': position_id[:8],
                    'symbol': symbol,
                    'side': position.get('side'),
                    'quantity': quantity,
                    'entry_price': position.get('entry_price'),
                    'current_price': current_price,
                    'unrealized_pnl': position.get('unrealized_pnl', 0),
                    'strategy': strategy
                })

            # Store summary in Redis
            await self.redis.setex(
                'positions:summary',
                60,
                json.dumps(summary)
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error getting position summary: {e}")
            return {}
