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
        - positions:pnl:realized (total realized P&L)
        - positions:pnl:realized:total (cumulative realized P&L)
        - risk:daily_pnl (daily P&L for risk manager)
        - positions:summary (portfolio summary)
        - positions:closed:total (closed position counter)

Author: AlphaTrader Pro
Version: 3.0.0
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pytz
import redis.asyncio as aioredis
from ib_insync import IB, Stock, Option, MarketOrder, StopOrder


class ExecutionManager:
    """Forward declaration for ExecutionManager - imported at runtime to avoid circular imports."""
    pass


class PositionManager:
    """
    Manage position lifecycle including P&L tracking and exit management.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis, ib_connection=None):
        """
        Initialize position manager with configuration.
        Can share IBKR connection with ExecutionManager or create its own.
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Can share IB connection or create new one
        self.ib = ib_connection if ib_connection else IB()

        # Position tracking
        self.positions = {}  # position_id -> position data
        self.position_tickers = {}  # symbol -> ticker subscription

        # Stop management
        self.stop_orders = {}  # position_id -> stop order
        self.trailing_stops = {}  # position_id -> trailing stop config

        # P&L tracking
        self.high_water_marks = {}  # position_id -> max profit

        # Configuration
        self.trail_config = {
            '0DTE': {'profit_trigger': 0.5, 'trail_percent': 0.5},  # Trail at 50% profit, keep 50%
            '1DTE': {'profit_trigger': 0.75, 'trail_percent': 0.75},  # Trail at 75% profit, keep 75%
            '14DTE': {'profit_trigger': 0.25, 'trail_percent': 0.25},  # Trail at 25% profit, keep 25%
            'default': {'profit_trigger': 0.3, 'trail_percent': 0.5}
        }

    async def start(self):
        """
        Main position management loop.
        Processing frequency: Every 1 second for monitoring, 5 seconds for P&L updates
        """
        self.logger.info("Starting position manager...")

        # Ensure IBKR connection if not shared
        if not self.ib.isConnected():
            await self.connect_ibkr()

        last_pnl_update = 0

        while True:
            try:
                current_time = time.time()

                # Load all open positions from Redis
                await self.load_positions()

                # Update P&L every 5 seconds
                if current_time - last_pnl_update >= 5:
                    await self.update_all_positions_pnl()
                    last_pnl_update = current_time

                # Check for exit conditions
                await self.check_exit_conditions()

                # Manage trailing stops
                await self.manage_trailing_stops()

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

    async def load_positions(self):
        """
        Load all open positions from Redis.
        """
        try:
            position_keys = await self.redis.keys('positions:open:*')

            for key in position_keys:
                position_json = await self.redis.get(key)
                if position_json:
                    position = json.loads(position_json)
                    position_id = position.get('id')

                    # Store in memory
                    self.positions[position_id] = position

                    # Subscribe to market data if needed
                    symbol = position.get('symbol')
                    if symbol not in self.position_tickers:
                        await self.subscribe_market_data(symbol, position.get('contract'))

        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")

    async def subscribe_market_data(self, symbol: str, contract_info: dict):
        """
        Subscribe to market data for position tracking.
        """
        try:
            # Import ExecutionManager here to avoid circular import
            from execution_manager import ExecutionManager
            # Create IB contract
            exec_mgr = ExecutionManager(self.config, self.redis)
            ib_contract = await exec_mgr.create_ib_contract(contract_info)

            if ib_contract:
                ticker = self.ib.reqMktData(ib_contract, '', False, False)
                self.position_tickers[symbol] = ticker
                self.logger.debug(f"Subscribed to market data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error subscribing to market data: {e}")

    async def update_all_positions_pnl(self):
        """
        Update P&L for all positions.
        """
        total_unrealized_pnl = 0
        total_realized_pnl = 0

        for position_id, position in self.positions.items():
            try:
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
                    await self.redis.setex(
                        f'positions:open:{symbol}:{position_id}',
                        86400,
                        json.dumps(position)
                    )

                    # Accumulate totals
                    total_unrealized_pnl += unrealized_pnl
                    total_realized_pnl += position.get('realized_pnl', 0)

            except Exception as e:
                self.logger.error(f"Error updating position {position_id}: {e}")

        # Update account totals
        await self.redis.set('positions:pnl:unrealized', total_unrealized_pnl)
        await self.redis.set('positions:pnl:realized', total_realized_pnl)
        await self.redis.set('risk:daily_pnl', total_realized_pnl)  # For risk manager

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

    async def manage_trailing_stops(self):
        """
        Manage trailing stops for all profitable positions.
        """
        for position_id, position in self.positions.items():
            try:
                unrealized_pnl = position.get('unrealized_pnl', 0)
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', 0)
                strategy = position.get('strategy', 'default')
                side = position.get('side')

                if unrealized_pnl <= 0:
                    continue  # Only trail profitable positions

                # Get trail configuration for strategy
                trail_config = self.trail_config.get(strategy, self.trail_config['default'])
                profit_trigger = trail_config['profit_trigger']
                trail_percent = trail_config['trail_percent']

                # Calculate profit percentage
                profit_pct = unrealized_pnl / (entry_price * position.get('quantity', 1) * 100)

                if profit_pct >= profit_trigger:
                    # Calculate new stop price
                    if side == 'LONG':
                        # Trail stop up
                        profit_to_lock = (current_price - entry_price) * trail_percent
                        new_stop = entry_price + profit_to_lock
                    else:  # SHORT
                        # Trail stop down
                        profit_to_lock = (entry_price - current_price) * trail_percent
                        new_stop = entry_price - profit_to_lock

                    # Round stop price
                    new_stop = round(new_stop, 2)

                    # Check if new stop is better than current
                    current_stop = position.get('stop_loss', 0)

                    should_update = False
                    if side == 'LONG' and new_stop > current_stop:
                        should_update = True
                    elif side == 'SHORT' and new_stop < current_stop:
                        should_update = True

                    if should_update:
                        await self.update_stop_loss(position, new_stop)
                        self.logger.info(
                            f"Trailing stop for {position_id[:8]}: "
                            f"${current_stop:.2f} → ${new_stop:.2f} "
                            f"(profit: {profit_pct:.1%})"
                        )

            except Exception as e:
                self.logger.error(f"Error managing trailing stop for {position_id}: {e}")

    async def update_stop_loss(self, position: dict, new_stop: float):
        """
        Update stop loss order with IBKR.
        """
        try:
            position_id = position.get('id')
            stop_order_id = position.get('stop_order_id')

            if stop_order_id and self.ib.isConnected():
                # Cancel old stop order
                old_order = self.stop_orders.get(position_id)
                if old_order:
                    self.ib.cancelOrder(old_order.order)

                # Import ExecutionManager here to avoid circular import
                from execution_manager import ExecutionManager
                # Place new stop order
                exec_mgr = ExecutionManager(self.config, self.redis)
                contract_info = position.get('contract')
                ib_contract = await exec_mgr.create_ib_contract(contract_info)

                if ib_contract:
                    side = position.get('side')
                    action = 'SELL' if side == 'LONG' else 'BUY'
                    quantity = position.get('quantity')

                    stop_order = StopOrder(action, quantity, new_stop)
                    stop_trade = self.ib.placeOrder(ib_contract, stop_order)

                    # Update tracking
                    self.stop_orders[position_id] = stop_trade
                    position['stop_loss'] = new_stop
                    position['stop_order_id'] = stop_trade.order.orderId

                    # Update in Redis
                    symbol = position.get('symbol')
                    await self.redis.setex(
                        f'positions:open:{symbol}:{position_id}',
                        86400,
                        json.dumps(position)
                    )

        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")

    async def check_targets(self):
        """
        Check if price targets hit for scaling out.
        """
        for position_id, position in self.positions.items():
            try:
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
            position_id = position.get('id')
            symbol = position.get('symbol')
            current_quantity = position.get('quantity', 0)
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

            # Don't scale if less than minimum
            contract_type = position.get('contract', {}).get('type', 'stock')
            if contract_type == 'option' and current_quantity < 3:
                # For options, only scale if we have 3+ contracts
                if target_index == 2:  # Final target, close all
                    close_quantity = current_quantity
                else:
                    return  # Skip scaling

            # Import ExecutionManager here to avoid circular import
            from execution_manager import ExecutionManager
            # Create closing order
            exec_mgr = ExecutionManager(self.config, self.redis)
            ib_contract = await exec_mgr.create_ib_contract(position.get('contract'))

            if ib_contract and self.ib.isConnected():
                action = 'SELL' if side == 'LONG' else 'BUY'
                order = MarketOrder(action, close_quantity)
                trade = self.ib.placeOrder(ib_contract, order)

                # Wait for fill
                await asyncio.sleep(1)

                if trade.isDone() and trade.orderStatus.status == 'Filled':
                    fill_price = trade.fills[-1].execution.price if trade.fills else target_price

                    # Calculate realized P&L for this portion
                    entry_price = position.get('entry_price', 0)
                    if contract_type == 'option':
                        if side == 'LONG':
                            realized_pnl = (fill_price - entry_price) * close_quantity * 100
                        else:
                            realized_pnl = (entry_price - fill_price) * close_quantity * 100
                    else:
                        if side == 'LONG':
                            realized_pnl = (fill_price - entry_price) * close_quantity
                        else:
                            realized_pnl = (entry_price - fill_price) * close_quantity

                    # Update position
                    position['quantity'] = current_quantity - close_quantity
                    position['realized_pnl'] = position.get('realized_pnl', 0) + realized_pnl
                    position['current_target_index'] = target_index + 1

                    # Log scale out
                    self.logger.info(
                        f"Scaled out {close_quantity} {symbol} @ ${fill_price:.2f} "
                        f"(target {target_index + 1}, P&L: ${realized_pnl:.2f})"
                    )

                    # Check if fully closed
                    if position['quantity'] <= 0:
                        await self.close_position(position, fill_price, 'All targets hit')
                    else:
                        # Update in Redis
                        await self.redis.setex(
                            f'positions:open:{symbol}:{position_id}',
                            86400,
                            json.dumps(position)
                        )

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

            # Update position record
            position['status'] = 'CLOSED'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now().isoformat()
            position['exit_reason'] = reason

            # Calculate final P&L if any quantity remains
            remaining_qty = position.get('quantity', 0)
            if remaining_qty > 0:
                entry_price = position.get('entry_price', 0)
                contract_type = position.get('contract', {}).get('type', 'stock')
                side = position.get('side')

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

            # Move to closed positions
            await self.redis.delete(f'positions:open:{symbol}:{position_id}')
            await self.redis.setex(
                f'positions:closed:{datetime.now().strftime("%Y%m%d")}:{position_id}',
                604800,  # 7 days TTL
                json.dumps(position)
            )

            # Update metrics
            await self.redis.srem(f'positions:by_symbol:{symbol}', position_id)
            await self.redis.incr('positions:closed:total')
            await self.redis.incrbyfloat('positions:pnl:realized:total', position['realized_pnl'])

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

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    async def check_exit_conditions(self):
        """
        Check for positions that need to be exited.
        """
        for position_id, position in list(self.positions.items()):
            try:
                strategy = position.get('strategy')

                # Check stop loss hit
                current_price = position.get('current_price', 0)
                stop_loss = position.get('stop_loss', 0)
                side = position.get('side')

                if stop_loss > 0:
                    if (side == 'LONG' and current_price <= stop_loss) or \
                       (side == 'SHORT' and current_price >= stop_loss):
                        await self.close_position(position, current_price, 'Stop loss hit')
                        continue

                # Check time-based exits for options
                if position.get('contract', {}).get('type') == 'option':
                    greeks = position.get('greeks', {})
                    theta = greeks.get('theta', 0)

                    # Exit if theta decay is too high
                    if strategy == '0DTE' and abs(theta) > 50:  # $50/day decay
                        await self.close_position(position, current_price, 'Theta decay limit')

            except Exception as e:
                self.logger.error(f"Error checking exit conditions: {e}")

    async def handle_eod_positions(self):
        """
        Handle end-of-day position management.
        """
        try:
            current_time = datetime.now(pytz.timezone('US/Eastern'))

            # Check if near market close (3:45 PM ET)
            if current_time.hour == 15 and current_time.minute >= 45:
                for position_id, position in list(self.positions.items()):
                    strategy = position.get('strategy')

                    # Close all 0DTE positions before expiry
                    if strategy == '0DTE':
                        current_price = position.get('current_price', 0)
                        await self.close_position(position, current_price, 'EOD 0DTE expiry avoidance')
                        self.logger.info(f"Closing 0DTE position {position_id[:8]} before expiry")

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
