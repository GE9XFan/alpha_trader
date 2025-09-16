"""
Emergency Manager Module

Handles emergency situations and provides close-all capabilities. Includes
automated circuit breakers for risk control and emergency state preservation.

Redis Keys Used:
    Read:
        - account:value (account value)
        - account:cash (cash balance)
        - risk:daily_pnl (daily P&L)
        - risk:consecutive_losses (consecutive losses)
        - risk:drawdown:current (current drawdown)
        - market:volatility:spike (volatility spike)
        - system:errors:count (system errors)
        - positions:open:* (open positions)
        - orders:pending:* (pending orders)
        - orders:working:* (working orders)
        - orders:stop:* (stop orders)
        - orders:active:* (active orders)
        - risk:* (all risk metrics)
        - risk:high_water_mark (HWM)
        - circuit_breakers:* (circuit breaker states)
    Write:
        - system:halt (system halt status)
        - risk:halt:status (halt flag)
        - risk:emergency:active (emergency flag)
        - risk:emergency:timestamp (emergency timestamp)
        - risk:permanent_halt (permanent halt flag)
        - risk:permanent_halt:reason (halt reason)
        - orders:emergency:{order_id} (emergency orders)
        - orders:cancelled:{order_id} (cancelled orders)
        - positions:closing:{position_id} (closing positions)
        - alerts:emergency (emergency alerts)
        - alerts:critical (critical alerts)
        - alerts:emergency:history (alert history)
        - system:emergency:active (emergency active flag)
        - system:emergency:reason (emergency reason)
        - system:emergency:timestamp (emergency timestamp)
        - metrics:emergency:closes (emergency close counter)
        - metrics:orders:cancelled:total (cancelled order counter)
        - metrics:orders:last_mass_cancel (last mass cancel time)
        - circuit_breakers:{breaker}:* (breaker states)
        - circuit_breakers:trips:history (trip history)
        - circuit_breakers:reset:history (reset history)
        - circuit_breakers:status (overall status)
        - circuit_breakers:last_reset (last reset info)
        - metrics:circuit_breakers:{breaker}:trips (trip counters)

Author: AlphaTrader Pro
Version: 3.0.0
"""

import asyncio
import json
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import redis


class EmergencyManager:
    """
    Handle emergency situations and provide close-all capabilities.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize emergency manager with configuration.

        TODO: Load configuration from config.yaml
        TODO: Set up Redis connection
        TODO: Initialize IBKR connection
        """
        self.config = config  # Loaded from config.yaml
        self.redis = redis_conn
        self.ib = None  # Would initialize IB() here
        self.logger = logging.getLogger(__name__)

    async def emergency_close_all(self):
        """
        Emergency close all positions immediately.
        This is the nuclear option - use carefully!
        """
        self.logger.critical("EMERGENCY: Closing all positions!")

        try:
            # Set emergency halt status immediately
            await self.redis.set('system:halt', 'EMERGENCY_CLOSE_ALL')
            await self.redis.set('risk:halt:status', 'true')
            await self.redis.set('risk:emergency:active', 'true')
            await self.redis.set('risk:emergency:timestamp', time.time())

            # Save system state for post-mortem
            await self.save_emergency_state()

            # Cancel all pending orders first
            await self.cancel_all_orders()

            # Get all open positions
            position_keys = await self.redis.keys('positions:open:*')
            self.logger.info(f"Found {len(position_keys)} positions to close")

            # Close each position at market
            close_tasks = []
            for key in position_keys:
                pos_data = await self.redis.get(key)
                if pos_data:
                    position = json.loads(pos_data)
                    close_tasks.append(self.close_position_emergency(position))

            # Execute all closes in parallel
            if close_tasks:
                results = await asyncio.gather(*close_tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Failed to close position {i}: {result}")

            # Send emergency alerts
            await self.trigger_emergency_alerts("EMERGENCY_CLOSE_ALL activated")

            # Set permanent halt
            await self.redis.set('risk:permanent_halt', 'true')
            await self.redis.set('risk:permanent_halt:reason', 'Emergency close all positions')
            await self.redis.set('risk:permanent_halt:timestamp', time.time())

            self.logger.critical("Emergency close all completed. Trading permanently halted.")

        except Exception as e:
            self.logger.error(f"Critical error in emergency_close_all: {e}")
            # Try to at least halt trading
            try:
                await self.redis.set('system:halt', 'EMERGENCY_ERROR')
                await self.redis.set('risk:halt:status', 'true')
            except:
                pass

    async def close_position_emergency(self, position: dict):
        """
        Close a single position at market.
        Emergency close uses market orders only.
        """
        try:
            symbol = position.get('symbol')
            side = position.get('side')
            quantity = position.get('quantity', 0)
            position_id = position.get('position_id')

            if not all([symbol, side, quantity]):
                self.logger.error(f"Invalid position data for emergency close: {position}")
                return

            # Determine close direction (opposite of position)
            close_side = 'SELL' if side == 'BUY' else 'BUY'

            # Create emergency close order
            close_order = {
                'symbol': symbol,
                'side': close_side,
                'quantity': abs(quantity),
                'order_type': 'MARKET',
                'time_in_force': 'IOC',  # Immediate or cancel
                'emergency': True,
                'position_id': position_id,
                'reason': 'EMERGENCY_CLOSE'
            }

            # Store emergency close order
            order_id = f"EMERGENCY_{position_id}_{int(time.time()*1000)}"
            await self.redis.setex(
                f'orders:emergency:{order_id}',
                3600,
                json.dumps(close_order)
            )

            # Would submit to IBKR here
            # For now, mark position as closing
            position['status'] = 'EMERGENCY_CLOSING'
            position['close_timestamp'] = time.time()
            position['close_reason'] = 'EMERGENCY'

            await self.redis.setex(
                f'positions:closing:{position_id}',
                300,
                json.dumps(position)
            )

            # Remove from open positions
            await self.redis.delete(f'positions:open:{position_id}')

            self.logger.warning(f"Emergency close initiated for {symbol} {side} x{quantity}")

            # Track emergency close
            await self.redis.incr('metrics:emergency:closes')

        except Exception as e:
            self.logger.error(f"Failed to emergency close position: {e}")

    async def cancel_all_orders(self):
        """
        Cancel all pending orders.
        Includes working orders, stop orders, and pending orders.
        """
        try:
            cancelled_count = 0

            # Get all order keys from Redis
            order_patterns = [
                'orders:pending:*',
                'orders:working:*',
                'orders:stop:*',
                'orders:active:*'
            ]

            for pattern in order_patterns:
                order_keys = await self.redis.keys(pattern)

                for key in order_keys:
                    try:
                        # Get order data
                        order_data = await self.redis.get(key)
                        if order_data:
                            order = json.loads(order_data)
                            order_id = order.get('order_id') or key.split(':')[-1]

                            # Mark as cancelled
                            order['status'] = 'CANCELLED'
                            order['cancel_time'] = time.time()
                            order['cancel_reason'] = 'EMERGENCY_CANCEL_ALL'

                            # Store cancelled order
                            await self.redis.setex(
                                f'orders:cancelled:{order_id}',
                                86400,  # Keep for 24 hours
                                json.dumps(order)
                            )

                            # Remove from active lists
                            await self.redis.delete(key)

                            cancelled_count += 1
                            self.logger.info(f"Cancelled order {order_id}")

                    except Exception as e:
                        self.logger.error(f"Error cancelling order {key}: {e}")

            # Clear order queues
            await self.redis.delete('orders:queue')
            await self.redis.delete('signals:pending')

            self.logger.warning(f"Cancelled {cancelled_count} orders")

            # Update metrics
            await self.redis.incr('metrics:orders:cancelled:total', cancelled_count)
            await self.redis.setex('metrics:orders:last_mass_cancel', 3600, time.time())

        except Exception as e:
            self.logger.error(f"Error in cancel_all_orders: {e}")

    async def trigger_emergency_alerts(self, reason: str):
        """
        Send emergency alerts to all monitoring channels.
        """
        try:
            alert = {
                'type': 'EMERGENCY',
                'reason': reason,
                'timestamp': time.time(),
                'timestamp_iso': datetime.now().isoformat(),
                'account_value': float(await self.redis.get('account:value') or 0),
                'daily_pnl': float(await self.redis.get('risk:daily_pnl') or 0),
                'open_positions': len(await self.redis.keys('positions:open:*')),
                'pending_orders': len(await self.redis.keys('orders:pending:*'))
            }

            # Publish to Redis pub/sub channels
            await self.redis.publish('alerts:emergency', json.dumps(alert))
            await self.redis.publish('alerts:critical', json.dumps(alert))

            # Store in alert history
            await self.redis.lpush('alerts:emergency:history', json.dumps(alert))
            await self.redis.ltrim('alerts:emergency:history', 0, 999)  # Keep last 1000

            # Log to file
            self.logger.critical(f"EMERGENCY ALERT: {reason}")
            self.logger.critical(f"Alert details: {json.dumps(alert, indent=2)}")

            # Set emergency status flags
            await self.redis.set('system:emergency:active', 'true')
            await self.redis.set('system:emergency:reason', reason)
            await self.redis.set('system:emergency:timestamp', time.time())

        except Exception as e:
            self.logger.error(f"Failed to send emergency alerts: {e}")

    async def save_emergency_state(self):
        """
        Save system state for post-mortem analysis.
        Saved to: data/emergency/[timestamp]/
        """
        try:
            # Create emergency directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            emergency_dir = Path(f'data/emergency/{timestamp}')
            emergency_dir.mkdir(parents=True, exist_ok=True)

            # Save positions
            positions = {}
            for key in await self.redis.keys('positions:*'):
                data = await self.redis.get(key)
                if data:
                    positions[key.decode() if isinstance(key, bytes) else key] = json.loads(data)

            with open(emergency_dir / 'positions.json', 'w') as f:
                json.dump(positions, f, indent=2)

            # Save orders
            orders = {}
            for key in await self.redis.keys('orders:*'):
                data = await self.redis.get(key)
                if data:
                    orders[key.decode() if isinstance(key, bytes) else key] = json.loads(data)

            with open(emergency_dir / 'orders.json', 'w') as f:
                json.dump(orders, f, indent=2)

            # Save risk metrics
            risk_metrics = {}
            for key in await self.redis.keys('risk:*'):
                data = await self.redis.get(key)
                if data:
                    try:
                        risk_metrics[key.decode() if isinstance(key, bytes) else key] = json.loads(data)
                    except:
                        risk_metrics[key.decode() if isinstance(key, bytes) else key] = data.decode() if isinstance(data, bytes) else str(data)

            with open(emergency_dir / 'risk_metrics.json', 'w') as f:
                json.dump(risk_metrics, f, indent=2)

            # Save account info
            account_info = {
                'account_value': float(await self.redis.get('account:value') or 0),
                'daily_pnl': float(await self.redis.get('risk:daily_pnl') or 0),
                'high_water_mark': float(await self.redis.get('risk:high_water_mark') or 0),
                'timestamp': timestamp,
                'emergency_timestamp': time.time()
            }

            with open(emergency_dir / 'account_info.json', 'w') as f:
                json.dump(account_info, f, indent=2)

            self.logger.info(f"Emergency state saved to {emergency_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save emergency state: {e}")


class CircuitBreakers:
    """
    Automated circuit breakers for risk control.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize circuit breakers with configuration.
        """
        self.config = config
        self.redis = redis_conn

        # Load risk management config
        risk_config = config.get('risk_management', {})
        circuit_config = risk_config.get('circuit_breakers', {})

        # Initialize breakers with config values
        self.breakers = {
            'daily_loss': {
                'enabled': circuit_config.get('daily_loss', True),
                'limit': risk_config.get('max_daily_loss_pct', 2.0),
                'current': 0,
                'triggered': False,
                'last_triggered': None,
                'cooldown': 3600  # 1 hour cooldown
            },
            'consecutive_losses': {
                'enabled': circuit_config.get('consecutive_losses', True),
                'limit': risk_config.get('consecutive_loss_limit', 3),
                'current': 0,
                'triggered': False,
                'last_triggered': None,
                'cooldown': 1800  # 30 min cooldown
            },
            'drawdown': {
                'enabled': circuit_config.get('max_drawdown', True),
                'limit': risk_config.get('max_drawdown_pct', 10.0),
                'current': 0,
                'triggered': False,
                'last_triggered': None,
                'cooldown': 7200  # 2 hour cooldown
            },
            'volatility_spike': {
                'enabled': circuit_config.get('volatility_spike', True),
                'limit': 3.0,  # 3 sigma event
                'current': 0,
                'triggered': False,
                'last_triggered': None,
                'cooldown': 900  # 15 min cooldown
            },
            'system_errors': {
                'enabled': circuit_config.get('system_errors', True),
                'limit': 10,  # 10 errors in monitoring window
                'current': 0,
                'triggered': False,
                'last_triggered': None,
                'cooldown': 600  # 10 min cooldown
            },
            'correlation': {
                'enabled': True,
                'limit': risk_config.get('correlation_limit', 0.7),
                'current': 0,
                'triggered': False,
                'last_triggered': None,
                'cooldown': 1800
            },
            'position_limit': {
                'enabled': True,
                'limit': risk_config.get('max_positions', 5),
                'current': 0,
                'triggered': False,
                'last_triggered': None,
                'cooldown': 300
            }
        }

        self.logger = logging.getLogger(__name__)
        self.last_reset = time.time()

    def check_breaker(self, breaker_name: str, current_value: float) -> bool:
        """
        Check if circuit breaker should trip.

        Returns:
            True if breaker should trip
        """
        try:
            if breaker_name not in self.breakers:
                self.logger.warning(f"Unknown breaker: {breaker_name}")
                return False

            breaker = self.breakers[breaker_name]

            # Skip if disabled
            if not breaker.get('enabled', True):
                return False

            # Check if in cooldown
            if breaker['triggered'] and breaker['last_triggered']:
                cooldown_remaining = (breaker['last_triggered'] + breaker['cooldown']) - time.time()
                if cooldown_remaining > 0:
                    self.logger.debug(f"Breaker {breaker_name} in cooldown for {cooldown_remaining:.0f}s")
                    return True  # Still tripped
                else:
                    # Reset after cooldown
                    breaker['triggered'] = False
                    breaker['last_triggered'] = None

            # Update current value
            breaker['current'] = current_value

            # Store current value in Redis
            self.redis.setex(
                f'circuit_breakers:{breaker_name}:current',
                60,
                json.dumps({
                    'value': current_value,
                    'limit': breaker['limit'],
                    'timestamp': time.time()
                })
            )

            # Check if limit exceeded
            limit_exceeded = False

            if breaker_name in ['daily_loss', 'drawdown', 'volatility_spike', 'correlation']:
                # For percentage/ratio limits
                limit_exceeded = current_value >= breaker['limit']
            else:
                # For count limits (consecutive_losses, system_errors, position_limit)
                limit_exceeded = current_value >= breaker['limit']

            if limit_exceeded and not breaker['triggered']:
                # Trip the breaker
                breaker['triggered'] = True
                breaker['last_triggered'] = time.time()

                # Log critical event
                self.logger.critical(
                    f"Circuit breaker '{breaker_name}' TRIPPED! "
                    f"Current: {current_value:.2f}, Limit: {breaker['limit']:.2f}"
                )

                # Store trip event
                trip_event = {
                    'breaker': breaker_name,
                    'current_value': current_value,
                    'limit': breaker['limit'],
                    'timestamp': time.time(),
                    'timestamp_iso': datetime.now().isoformat()
                }

                self.redis.lpush('circuit_breakers:trips:history', json.dumps(trip_event))
                self.redis.ltrim('circuit_breakers:trips:history', 0, 99)

                # Set breaker status
                self.redis.set(f'circuit_breakers:{breaker_name}:tripped', 'true')
                self.redis.setex(f'circuit_breakers:{breaker_name}:trip_time', 3600, time.time())

                # Increment metrics
                self.redis.incr(f'metrics:circuit_breakers:{breaker_name}:trips')

                return True

            # Log warnings when approaching limits
            elif not breaker['triggered']:
                threshold = 0.8  # Warn at 80% of limit
                if breaker['limit'] > 0:
                    ratio = current_value / breaker['limit']
                    if ratio >= threshold:
                        self.logger.warning(
                            f"Circuit breaker '{breaker_name}' approaching limit: "
                            f"{current_value:.2f}/{breaker['limit']:.2f} ({ratio*100:.0f}%)"
                        )

            return breaker['triggered']

        except Exception as e:
            self.logger.error(f"Error checking breaker {breaker_name}: {e}")
            # Fail safe - trip on error
            return True

    def reset_daily_breakers(self):
        """
        Reset daily circuit breakers.
        Called at market open each day.
        """
        try:
            self.logger.info("Resetting daily circuit breakers")

            # Reset daily loss
            if 'daily_loss' in self.breakers:
                self.breakers['daily_loss']['current'] = 0
                self.breakers['daily_loss']['triggered'] = False
                self.breakers['daily_loss']['last_triggered'] = None

            # Reset consecutive losses
            if 'consecutive_losses' in self.breakers:
                self.breakers['consecutive_losses']['current'] = 0
                self.breakers['consecutive_losses']['triggered'] = False
                self.breakers['consecutive_losses']['last_triggered'] = None

            # Reset system errors counter
            if 'system_errors' in self.breakers:
                self.breakers['system_errors']['current'] = 0
                self.breakers['system_errors']['triggered'] = False
                self.breakers['system_errors']['last_triggered'] = None

            # Clear Redis flags
            self.redis.delete('circuit_breakers:daily_loss:tripped')
            self.redis.delete('circuit_breakers:consecutive_losses:tripped')
            self.redis.delete('circuit_breakers:system_errors:tripped')

            # Reset Redis counters
            self.redis.set('risk:consecutive_losses', '0')
            self.redis.set('system:errors:count', '0')

            # Update reset timestamp
            self.last_reset = time.time()
            reset_event = {
                'type': 'daily_reset',
                'timestamp': self.last_reset,
                'timestamp_iso': datetime.now().isoformat(),
                'breakers_reset': ['daily_loss', 'consecutive_losses', 'system_errors']
            }

            self.redis.setex('circuit_breakers:last_reset', 86400, json.dumps(reset_event))

            # Log reset event
            self.redis.lpush('circuit_breakers:reset:history', json.dumps(reset_event))
            self.redis.ltrim('circuit_breakers:reset:history', 0, 29)  # Keep last 30 resets

            self.logger.info("Daily circuit breakers reset complete")

        except Exception as e:
            self.logger.error(f"Error resetting daily breakers: {e}")

    def get_breaker_status(self) -> dict:
        """
        Get current status of all breakers.

        Returns:
            Breaker status dictionary
        """
        try:
            status = {
                'timestamp': time.time(),
                'timestamp_iso': datetime.now().isoformat(),
                'last_reset': self.last_reset,
                'breakers': {},
                'any_tripped': False,
                'total_tripped': 0
            }

            for name, breaker in self.breakers.items():
                # Update current values from Redis where applicable
                if name == 'daily_loss':
                    daily_pnl = self.redis.get('risk:daily_pnl')
                    account_value = self.redis.get('account:value')
                    if daily_pnl and account_value:
                        loss_pct = abs(min(0, float(daily_pnl))) / float(account_value) * 100
                        breaker['current'] = loss_pct

                elif name == 'consecutive_losses':
                    losses = self.redis.get('risk:consecutive_losses')
                    if losses:
                        breaker['current'] = int(losses)

                elif name == 'drawdown':
                    drawdown_pct = self.redis.get('risk:drawdown:current')
                    if drawdown_pct:
                        breaker['current'] = float(drawdown_pct)

                elif name == 'volatility_spike':
                    vol_spike = self.redis.get('market:volatility:spike')
                    if vol_spike:
                        breaker['current'] = float(vol_spike)

                elif name == 'system_errors':
                    errors = self.redis.get('system:errors:count')
                    if errors:
                        breaker['current'] = int(errors)

                elif name == 'position_limit':
                    positions = len(self.redis.keys('positions:open:*'))
                    breaker['current'] = positions

                # Calculate status
                current = breaker['current']
                limit = breaker['limit']

                breaker_status = {
                    'enabled': breaker.get('enabled', True),
                    'current': current,
                    'limit': limit,
                    'triggered': breaker['triggered'],
                    'last_triggered': breaker.get('last_triggered'),
                    'cooldown': breaker.get('cooldown', 0),
                    'percentage': (current / limit * 100) if limit > 0 else 0,
                    'distance_to_limit': limit - current if limit > 0 else 0
                }

                # Check if in cooldown
                if breaker['triggered'] and breaker.get('last_triggered'):
                    cooldown_remaining = max(0,
                        (breaker['last_triggered'] + breaker['cooldown']) - time.time()
                    )
                    breaker_status['cooldown_remaining'] = cooldown_remaining
                    breaker_status['in_cooldown'] = cooldown_remaining > 0

                # Determine health status
                if breaker['triggered']:
                    breaker_status['health'] = 'TRIPPED'
                    status['any_tripped'] = True
                    status['total_tripped'] += 1
                elif breaker_status['percentage'] >= 90:
                    breaker_status['health'] = 'CRITICAL'
                elif breaker_status['percentage'] >= 80:
                    breaker_status['health'] = 'WARNING'
                elif breaker_status['percentage'] >= 60:
                    breaker_status['health'] = 'CAUTION'
                else:
                    breaker_status['health'] = 'HEALTHY'

                status['breakers'][name] = breaker_status

            # Store status in Redis
            self.redis.setex('circuit_breakers:status', 60, json.dumps(status))

            return status

        except Exception as e:
            self.logger.error(f"Error getting breaker status: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
                'any_tripped': False
            }