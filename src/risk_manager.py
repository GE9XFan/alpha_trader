"""
Risk Manager Module

Monitors risk metrics and enforces trading limits through circuit breakers.
Production-ready risk management with multiple safety layers including VaR
calculation, drawdown monitoring, and correlation limits.

Redis Keys Used:
    Read:
        - account:value (account net liquidation)
        - risk:daily_pnl (daily P&L)
        - risk:consecutive_losses (consecutive loss counter)
        - market:volatility:spike (volatility detection)
        - system:errors:count (system error counter)
        - discovered:correlation_matrix (correlation data)
        - positions:open:* (open positions)
        - risk:high_water_mark (HWM for drawdown)
        - risk:pnl:history (historical P&L)
        - analytics:{symbol}:volatility (symbol volatility)
    Write:
        - risk:circuit_breaker:status (circuit breaker status)
        - risk:halt:status (trading halt flag)
        - risk:halt:reason (halt reason)
        - risk:halt:timestamp (halt timestamp)
        - risk:halt:history (halt event history)
        - risk:circuit_breakers:status (breaker details)
        - risk:position_size_multiplier (position sizing)
        - risk:stop_multiplier (stop adjustment)
        - risk:current_drawdown (drawdown metrics)
        - risk:drawdown:history (drawdown history)
        - risk:var:portfolio (VaR metrics)
        - risk:var:detailed (detailed VaR)
        - risk:high_var_flag (high VaR flag)
        - risk:new_positions_allowed (position gate)
        - risk:daily:status (daily P&L status)
        - risk:consecutive_losing_days (losing day counter)
        - risk:metrics:summary (risk metrics summary)
        - risk:correlation:blocked:{symbol} (correlation blocks)
        - metrics:risk:* (various risk metrics)
        - alerts:critical (critical alerts)

Author: QuantiCity Capital
Version: 3.0.0
"""

import asyncio
import json
import time
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, List, Any, Optional
import redis.asyncio as aioredis
import pytz


@dataclass
class VaRSnapshot:
    var_95: float
    confidence: float
    timestamp: float
    historical_var: Optional[float] = None
    parametric_var: Optional[float] = None
    mean_return: Optional[float] = None
    std_return: Optional[float] = None
    data_points: Optional[int] = None

    def to_dict(self, include_details: bool = False) -> Dict[str, Any]:
        base = {
            'var_95': self.var_95,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
        }
        if include_details:
            if self.historical_var is not None:
                base['historical_var'] = self.historical_var
            if self.parametric_var is not None:
                base['parametric_var'] = self.parametric_var
            if self.mean_return is not None:
                base['mean_return'] = self.mean_return
            if self.std_return is not None:
                base['std_return'] = self.std_return
            if self.data_points is not None:
                base['data_points'] = self.data_points
        return base

    def to_json(self, include_details: bool = False) -> str:
        return json.dumps(self.to_dict(include_details=include_details))

    @classmethod
    def from_redis(cls, payload: Any) -> Optional['VaRSnapshot']:
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            try:
                value = float(payload)
            except ValueError:
                return None
            return cls(var_95=value, confidence=0.95, timestamp=time.time())

        return cls(
            var_95=float(data.get('var_95', 0)),
            confidence=float(data.get('confidence', 0.95)),
            timestamp=float(data.get('timestamp', time.time())),
            historical_var=data.get('historical_var'),
            parametric_var=data.get('parametric_var'),
            mean_return=data.get('mean_return'),
            std_return=data.get('std_return'),
            data_points=data.get('data_points'),
        )


@dataclass
class DrawdownSnapshot:
    drawdown_pct: float
    high_water_mark: Optional[float] = None
    timestamp: Optional[float] = None

    @classmethod
    def from_redis(cls, payload: Any) -> Optional['DrawdownSnapshot']:
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            try:
                return cls(drawdown_pct=float(payload))
            except ValueError:
                return None
        return cls(
            drawdown_pct=float(data.get('drawdown_pct', 0)),
            high_water_mark=data.get('high_water_mark'),
            timestamp=data.get('timestamp'),
        )


class RiskManager:
    """
    Monitor risk metrics and enforce trading limits through circuit breakers.
    Production-ready risk management with multiple safety layers.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """
        Initialize risk manager with configuration.
        Note: redis_conn can be sync or async Redis depending on context.
        """
        self.config = config
        self.redis = redis_conn

        # Load risk limits from config
        risk_config = config.get('risk_management', {})
        self.max_daily_loss_pct = risk_config.get('max_daily_loss_pct', 2.0)
        self.max_position_loss_pct = risk_config.get('max_position_loss_pct', 1.0)
        self.consecutive_loss_limit = risk_config.get('consecutive_loss_limit', 3)
        self.correlation_limit = risk_config.get('correlation_limit', 0.7)
        self.margin_buffer = risk_config.get('margin_buffer', 1.25)
        self.max_drawdown_pct = risk_config.get('max_drawdown_pct', 10.0)

        # Circuit breaker states
        self.circuit_breakers_tripped = set()
        self.halt_reason = None
        self.last_check_time = 0

        # Track consecutive losses
        self.consecutive_losses = 0

        # VaR parameters
        self.var_confidence = 0.95  # 95% confidence level
        self.var_lookback_days = 30

        self.logger = logging.getLogger(__name__)

    async def start(self):
        """
        Continuous risk monitoring loop.
        Checks all risk metrics and enforces limits.

        Processing frequency: Every 1 second
        """
        self.logger.info("Starting risk manager...")

        # Check if we need to reset daily metrics (only at market open, not on restart)
        await self.check_and_reset_daily_metrics()

        while True:
            try:
                # Check if we should run checks (throttle to every 1 second)
                current_time = time.time()
                if current_time - self.last_check_time < 1:
                    await asyncio.sleep(0.1)
                    continue

                self.last_check_time = current_time

                # Run all risk checks
                await self.check_circuit_breakers()
                await self.monitor_drawdown()
                await self.check_daily_limits()
                await self.update_risk_metrics()

                # Calculate and store VaR
                var_snapshot = await self.calculate_var()
                if var_snapshot:
                    await self.redis.setex(
                        'risk:var:portfolio',
                        300,
                        var_snapshot.to_json()
                    )

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(1)

    async def check_circuit_breakers(self):
        """
        Check all circuit breaker conditions and trigger halts if needed.
        Production-ready with multiple safety layers.
        """
        try:
            breakers_status = {}

            # 1. Check daily loss limit
            daily_pnl = await self.redis.get('risk:daily_pnl')
            if daily_pnl:
                daily_pnl = float(daily_pnl)
                account_value = float(await self.redis.get('account:value') or 100000)
                daily_loss_pct = abs(min(0, daily_pnl)) / account_value * 100

                breakers_status['daily_loss'] = {
                    'current': daily_loss_pct,
                    'limit': self.max_daily_loss_pct,
                    'triggered': daily_loss_pct >= self.max_daily_loss_pct
                }

                if daily_loss_pct >= self.max_daily_loss_pct:
                    await self.halt_trading(f"Daily loss limit exceeded: {daily_loss_pct:.1f}%")
                    self.circuit_breakers_tripped.add('daily_loss')

            # 2. Check consecutive losses
            consecutive = await self.redis.get('risk:consecutive_losses')
            if consecutive:
                consecutive = int(consecutive)
                breakers_status['consecutive_losses'] = {
                    'current': consecutive,
                    'limit': self.consecutive_loss_limit,
                    'triggered': consecutive >= self.consecutive_loss_limit
                }

                if consecutive >= self.consecutive_loss_limit:
                    await self.halt_trading(f"Consecutive loss limit hit: {consecutive} losses")
                    self.circuit_breakers_tripped.add('consecutive_losses')

            # 3. Check volatility spike
            market_vol = await self.redis.get('market:volatility:spike')
            if market_vol:
                vol_spike = float(market_vol)
                vol_threshold = 3.0  # 3 sigma event

                breakers_status['volatility_spike'] = {
                    'current': vol_spike,
                    'limit': vol_threshold,
                    'triggered': vol_spike >= vol_threshold
                }

                if vol_spike >= vol_threshold:
                    await self.halt_trading(f"Volatility spike detected: {vol_spike:.1f} sigma")
                    self.circuit_breakers_tripped.add('volatility')

            # 4. Check system errors
            error_count = await self.redis.get('system:errors:count')
            if error_count:
                errors = int(error_count)
                error_threshold = 10  # 10 errors in monitoring window

                breakers_status['system_errors'] = {
                    'current': errors,
                    'limit': error_threshold,
                    'triggered': errors >= error_threshold
                }

                if errors >= error_threshold:
                    await self.halt_trading(f"System error threshold exceeded: {errors} errors")
                    self.circuit_breakers_tripped.add('system_errors')

            # 5. Store circuit breaker status
            await self.redis.setex('risk:circuit_breakers:status', 60,
                                  json.dumps(breakers_status))

            # Log if any breakers are close to triggering
            for breaker, status in breakers_status.items():
                if not status.get('triggered', False):
                    current = status.get('current', 0)
                    limit = status.get('limit', 0)
                    if limit > 0 and current / limit > 0.8:  # Within 80% of limit
                        self.logger.warning(
                            f"Circuit breaker '{breaker}' approaching limit: "
                            f"{current:.1f}/{limit:.1f} ({current/limit*100:.0f}%)"
                        )

        except Exception as e:
            self.logger.error(f"Error checking circuit breakers: {e}")
            # On error, fail safe and halt
            await self.halt_trading(f"Circuit breaker check failed: {e}")

    async def halt_trading(self, reason: str):
        """
        Halt all trading activity immediately.
        This is a critical safety function.
        """
        try:
            self.logger.critical(f"TRADING HALT TRIGGERED: {reason}")

            # Set halt flag (highest priority)
            await self.redis.set('risk:circuit_breaker:status', 'HALTED')
            await self.redis.set('risk:halt:status', 'true')
            await self.redis.set('risk:halt:reason', reason)
            await self.redis.set('risk:halt:timestamp', time.time())

            # Cancel all pending orders
            pending_orders = await self.redis.keys('orders:pending:*')
            for order_key in pending_orders:
                await self.redis.delete(order_key)
                self.logger.info(f"Cancelled pending order: {order_key}")

            # Store halt event in history
            halt_event = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'circuit_breakers': list(self.circuit_breakers_tripped),
                'account_value': float(await self.redis.get('account:value') or 0),
                'daily_pnl': float(await self.redis.get('risk:daily_pnl') or 0),
                'open_positions': len(await self.redis.keys('positions:open:*'))
            }

            await self.redis.lpush('risk:halt:history', json.dumps(halt_event))
            await self.redis.ltrim('risk:halt:history', 0, 99)  # Keep last 100 halts

            # Send alerts (would integrate with monitoring system)
            await self.redis.publish('alerts:critical', json.dumps({
                'type': 'TRADING_HALT',
                'reason': reason,
                'timestamp': time.time()
            }))

            # Update monitoring metrics
            await self.redis.incr('metrics:risk:halts:total')
            await self.redis.setex('metrics:risk:halts:latest', 3600, reason)

            self.halt_reason = reason

        except Exception as e:
            self.logger.error(f"Error halting trading: {e}")
            # Try simpler halt as fallback
            try:
                await self.redis.set('risk:halt:status', 'true')
            except:
                pass

    async def check_correlations(self, symbol: str, side: str) -> bool:
        """
        Check if adding position would create excessive correlation.
        Prevents concentration risk from correlated positions.

        Returns:
            True if position is allowed, False if correlation too high
        """
        try:
            # Get correlation matrix
            corr_data = await self.redis.get('discovered:correlation_matrix')
            if not corr_data:
                self.logger.warning("No correlation matrix available")
                return True  # Allow if no data

            correlation_matrix = json.loads(corr_data)

            # Get current open positions
            position_keys = await self.redis.keys('positions:open:*')
            if not position_keys:
                return True  # No positions, correlation not an issue

            existing_positions = {}
            for key in position_keys:
                pos_data = await self.redis.get(key)
                if pos_data:
                    pos = json.loads(pos_data)
                    pos_symbol = pos.get('symbol')
                    pos_side = pos.get('side')
                    if pos_symbol and pos_side:
                        existing_positions[pos_symbol] = pos_side

            # Calculate correlations with existing positions
            high_correlations = []

            for pos_symbol, pos_side in existing_positions.items():
                if pos_symbol == symbol:
                    # Same symbol
                    if pos_side == side:
                        # Adding to existing position is ok
                        continue
                    else:
                        # Opposite direction on same symbol not allowed
                        self.logger.warning(f"Blocking opposite position on {symbol}")
                        return False

                # Check correlation
                corr_key = f"{symbol}:{pos_symbol}"
                alt_key = f"{pos_symbol}:{symbol}"

                correlation = correlation_matrix.get(corr_key) or correlation_matrix.get(alt_key)

                if correlation is not None:
                    abs_corr = abs(float(correlation))

                    # Check if positions would compound risk
                    if abs_corr > self.correlation_limit:
                        if (correlation > 0 and pos_side == side) or \
                           (correlation < 0 and pos_side != side):
                            # High correlation in same direction
                            high_correlations.append({
                                'symbol': pos_symbol,
                                'correlation': correlation,
                                'side': pos_side
                            })

            if high_correlations:
                # Check total correlated exposure
                avg_correlation = sum(abs(hc['correlation']) for hc in high_correlations) / len(high_correlations)

                if avg_correlation > self.correlation_limit:
                    self.logger.warning(
                        f"Position {symbol} {side} blocked due to high correlation: "
                        f"{avg_correlation:.2f} with {[hc['symbol'] for hc in high_correlations]}"
                    )

                    # Store correlation block event
                    await self.redis.setex(
                        f'risk:correlation:blocked:{symbol}',
                        60,
                        json.dumps({
                            'symbol': symbol,
                            'side': side,
                            'avg_correlation': avg_correlation,
                            'correlated_with': high_correlations,
                            'timestamp': time.time()
                        })
                    )

                    return False
                elif avg_correlation > self.correlation_limit * 0.8:
                    # Warning zone
                    self.logger.warning(
                        f"Position {symbol} approaching correlation limit: {avg_correlation:.2f}"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Error checking correlations: {e}")
            # On error, be conservative and block
            return False

    async def monitor_drawdown(self):
        """
        Monitor drawdown from high water mark and trigger risk reduction if needed.
        Critical for capital preservation.
        """
        try:
            # Get current account value
            account_value = await self.redis.get('account:value')
            if not account_value:
                # Try to calculate from positions
                account_value = await self._calculate_account_value()
                await self.redis.setex('account:value', 60, account_value)
            else:
                account_value = float(account_value)

            # Get or initialize high water mark
            hwm = await self.redis.get('risk:high_water_mark')
            if not hwm:
                # Initialize HWM
                hwm = account_value
                await self.redis.set('risk:high_water_mark', hwm)
            else:
                hwm = float(hwm)

            # Update HWM if new high
            if account_value > hwm:
                hwm = account_value
                await self.redis.set('risk:high_water_mark', hwm)
                await self.redis.set('risk:hwm:timestamp', time.time())
                self.logger.info(f"New high water mark: ${hwm:,.2f}")

            # Calculate drawdown
            if hwm > 0:
                drawdown_pct = ((hwm - account_value) / hwm) * 100
            else:
                drawdown_pct = 0

            # Store current drawdown
            await self.redis.setex('risk:current_drawdown', 60, json.dumps({
                'drawdown_pct': drawdown_pct,
                'current_value': account_value,
                'high_water_mark': hwm,
                'timestamp': time.time()
            }))

            # Check drawdown thresholds
            if drawdown_pct >= self.max_drawdown_pct:
                # Critical drawdown - halt trading
                await self.halt_trading(f"Maximum drawdown exceeded: {drawdown_pct:.1f}%")
                self.circuit_breakers_tripped.add('max_drawdown')

            elif drawdown_pct >= self.max_drawdown_pct * 0.8:  # 80% of max
                # Warning zone - reduce position sizes
                await self.redis.set('risk:position_size_multiplier', 0.5)
                self.logger.warning(f"Drawdown warning: {drawdown_pct:.1f}% - reducing position sizes")

            elif drawdown_pct >= self.max_drawdown_pct * 0.6:  # 60% of max
                # Caution zone - tighten stops
                await self.redis.set('risk:stop_multiplier', 0.75)
                self.logger.warning(f"Drawdown caution: {drawdown_pct:.1f}% - tightening stops")

            # Track drawdown history
            await self.redis.lpush('risk:drawdown:history', json.dumps({
                'timestamp': datetime.now().isoformat(),
                'drawdown_pct': drawdown_pct,
                'account_value': account_value,
                'hwm': hwm
            }))
            await self.redis.ltrim('risk:drawdown:history', 0, 1439)  # Keep 24 hours at 1min intervals

            # Update metrics
            await self.redis.setex('metrics:risk:drawdown:current', 60, drawdown_pct)
            await self.redis.setex('metrics:risk:drawdown:max_today', 3600,
                                  max(drawdown_pct, float(await self.redis.get('metrics:risk:drawdown:max_today') or 0)))

        except Exception as e:
            self.logger.error(f"Error monitoring drawdown: {e}")

    async def _calculate_account_value(self) -> float:
        """
        Calculate account value from cash + positions.
        """
        try:
            # Get cash balance
            cash = float(await self.redis.get('account:cash') or 100000)

            # Get all open positions
            position_keys = await self.redis.keys('positions:open:*')
            positions_value = 0

            for key in position_keys:
                pos_data = await self.redis.get(key)
                if pos_data:
                    pos = json.loads(pos_data)
                    # Use mark-to-market value
                    positions_value += pos.get('market_value', 0)

            return cash + positions_value

        except Exception as e:
            self.logger.error(f"Error calculating account value: {e}")
            return 100000  # Default fallback

    async def check_daily_limits(self):
        """
        Check and enforce daily loss limits.
        Prevents catastrophic single-day losses.
        """
        try:
            # Get current P&L for the day
            daily_pnl = await self.redis.get('risk:daily_pnl')
            if not daily_pnl:
                daily_pnl = 0
            else:
                daily_pnl = float(daily_pnl)

            # Get account value for percentage calculation
            account_value = float(await self.redis.get('account:value') or 100000)

            # Calculate loss percentage
            if daily_pnl < 0:
                daily_loss_pct = abs(daily_pnl) / account_value * 100
            else:
                daily_loss_pct = 0

            # Store current daily P&L status
            await self.redis.setex('risk:daily:status', 60, json.dumps({
                'pnl': daily_pnl,
                'loss_pct': daily_loss_pct,
                'limit_pct': self.max_daily_loss_pct,
                'account_value': account_value,
                'timestamp': time.time()
            }))

            # Check against limit
            if daily_loss_pct >= self.max_daily_loss_pct:
                # Daily loss limit exceeded - halt trading
                await self.halt_trading(
                    f"Daily loss limit exceeded: ${abs(daily_pnl):,.2f} "
                    f"({daily_loss_pct:.1f}% of account)"
                )
                self.circuit_breakers_tripped.add('daily_loss')

            elif daily_loss_pct >= self.max_daily_loss_pct * 0.75:  # 75% of limit
                # Warning zone - restrict new positions
                await self.redis.set('risk:new_positions_allowed', 'false')
                self.logger.warning(
                    f"Daily loss warning: ${abs(daily_pnl):,.2f} "
                    f"({daily_loss_pct:.1f}%) - new positions blocked"
                )

            elif daily_loss_pct >= self.max_daily_loss_pct * 0.5:  # 50% of limit
                # Caution zone - reduce position sizes
                await self.redis.set('risk:position_size_multiplier', 0.7)
                self.logger.warning(
                    f"Daily loss caution: ${abs(daily_pnl):,.2f} "
                    f"({daily_loss_pct:.1f}%) - reducing position sizes"
                )
            else:
                # Normal operation
                await self.redis.set('risk:new_positions_allowed', 'true')
                await self.redis.set('risk:position_size_multiplier', 1.0)

            # Track consecutive losing days
            if daily_pnl < 0:
                losing_days = await self.redis.incr('risk:consecutive_losing_days')
                if losing_days >= 3:
                    self.logger.warning(f"Alert: {losing_days} consecutive losing days")
                    # Reduce risk after consecutive losses
                    await self.redis.set('risk:position_size_multiplier', 0.5)

            # Update metrics
            await self.redis.setex('metrics:risk:daily_pnl', 60, daily_pnl)
            await self.redis.setex('metrics:risk:daily_loss_pct', 60, daily_loss_pct)

        except Exception as e:
            self.logger.error(f"Error checking daily limits: {e}")
            # On error, be conservative
            await self.redis.set('risk:new_positions_allowed', 'false')

    async def calculate_var(self, confidence: float = 0.95) -> Optional[VaRSnapshot]:
        """
        Calculate Value at Risk using historical simulation method.
        More accurate than parametric VaR for non-normal distributions.

        Returns:
            VaR at specified confidence level (potential loss amount)
        """
        try:
            # Get historical P&L data
            pnl_history = await self.redis.lrange('risk:pnl:history', 0, self.var_lookback_days * 390)  # 390 minutes per trading day

            if not pnl_history or len(pnl_history) < 20:
                # Not enough data for meaningful VaR
                self.logger.warning("Insufficient data for VaR calculation")
                # Fallback to position-based estimate
                fallback = await self._calculate_position_based_var()
                snapshot = VaRSnapshot(var_95=fallback, confidence=confidence, timestamp=time.time())
                await self.redis.setex('risk:var:detailed', 300, snapshot.to_json(include_details=True))
                return snapshot

            # Parse P&L values
            pnl_values = []
            for pnl_json in pnl_history:
                try:
                    pnl_data = json.loads(pnl_json)
                    pnl_values.append(float(pnl_data.get('pnl', 0)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            if len(pnl_values) < 20:
                fallback = await self._calculate_position_based_var()
                snapshot = VaRSnapshot(var_95=fallback, confidence=confidence, timestamp=time.time())
                await self.redis.setex('risk:var:detailed', 300, snapshot.to_json(include_details=True))
                return snapshot

            # Calculate returns (changes in P&L)
            returns = []
            for i in range(1, len(pnl_values)):
                returns.append(pnl_values[i] - pnl_values[i-1])

            # Sort returns for percentile calculation
            returns.sort()

            # Calculate VaR at confidence level
            percentile_index = int(len(returns) * (1 - confidence))
            var_value = abs(returns[percentile_index]) if percentile_index < len(returns) else abs(returns[0])

            # Calculate additional risk metrics
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            # Parametric VaR for comparison (assumes normal distribution)
            z_score = 1.65 if confidence == 0.95 else 2.33 if confidence == 0.99 else 1.65
            parametric_var = abs(mean_return - z_score * std_return)

            # Use the more conservative estimate
            final_var = max(var_value, parametric_var)

            snapshot = VaRSnapshot(
                var_95=final_var,
                confidence=confidence,
                timestamp=time.time(),
                historical_var=var_value,
                parametric_var=parametric_var,
                mean_return=float(mean_return),
                std_return=float(std_return),
                data_points=len(returns),
            )

            # Store VaR metrics
            await self.redis.setex('risk:var:detailed', 300, snapshot.to_json(include_details=True))

            # Check VaR against limits
            account_value = float(await self.redis.get('account:value') or 100000)
            var_pct = (final_var / account_value) * 100

            if var_pct > 5:  # VaR exceeds 5% of account
                self.logger.warning(f"High VaR detected: ${final_var:,.2f} ({var_pct:.1f}% of account)")
                # Reduce position sizes when VaR is high
                await self.redis.set('risk:high_var_flag', 'true')
            else:
                await self.redis.set('risk:high_var_flag', 'false')

            return snapshot

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            # Fallback to position-based estimate
            fallback = await self._calculate_position_based_var()
            snapshot = VaRSnapshot(var_95=fallback, confidence=confidence, timestamp=time.time())
            await self.redis.setex('risk:var:detailed', 300, snapshot.to_json(include_details=True))
            return snapshot

    async def _calculate_position_based_var(self) -> float:
        """
        Fallback VaR calculation based on current positions.
        Uses position sizes and historical volatility.
        """
        try:
            # Get all open positions
            position_keys = await self.redis.keys('positions:open:*')
            total_var = 0

            for key in position_keys:
                pos_data = await self.redis.get(key)
                if pos_data:
                    pos = json.loads(pos_data)
                    symbol = pos.get('symbol')
                    position_value = pos.get('market_value', 0)

                    # Get symbol volatility
                    vol_data = await self.redis.get(f'analytics:{symbol}:volatility')
                    if vol_data:
                        volatility = float(json.loads(vol_data).get('daily_vol', 0.02))  # 2% default
                    else:
                        volatility = 0.02

                    # Position VaR (95% confidence = 1.65 sigma)
                    position_var = abs(position_value) * volatility * 1.65
                    total_var += position_var

            # Apply diversification benefit (square root rule for uncorrelated positions)
            # This is conservative as it assumes some correlation
            diversification_factor = 0.75  # Assumes moderate correlation
            portfolio_var = total_var * diversification_factor

            return portfolio_var

        except Exception as e:
            self.logger.error(f"Error in position-based VaR: {e}")
            # Ultimate fallback: 2% of account value
            account_value = float(await self.redis.get('account:value') or 100000)
            return account_value * 0.02

    async def _load_var_snapshot(self) -> Optional[VaRSnapshot]:
        payload = await self.redis.get('risk:var:portfolio')
        return VaRSnapshot.from_redis(payload)

    async def _load_drawdown_snapshot(self) -> Optional[DrawdownSnapshot]:
        payload = await self.redis.get('risk:current_drawdown')
        return DrawdownSnapshot.from_redis(payload)

    async def update_risk_metrics(self):
        """
        Update comprehensive risk metrics for monitoring and decision-making.
        """
        try:
            metrics = {}

            # 1. Position concentration
            position_keys = await self.redis.keys('positions:open:*')
            position_count = len(position_keys)

            symbol_exposure = {}
            total_exposure = 0

            for key in position_keys:
                pos_data = await self.redis.get(key)
                if pos_data:
                    pos = json.loads(pos_data)
                    symbol = pos.get('symbol')
                    value = abs(pos.get('market_value', 0))
                    symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + value
                    total_exposure += value

            # Calculate concentration metrics
            max_concentration = 0
            if total_exposure > 0:
                max_concentration = max(symbol_exposure.values()) / total_exposure if symbol_exposure else 0
                metrics['concentration'] = {
                    'max_symbol_pct': max_concentration * 100,
                    'position_count': position_count,
                    'total_exposure': total_exposure,
                    'by_symbol': {k: v/total_exposure*100 for k, v in symbol_exposure.items()}
                }
            else:
                metrics['concentration'] = {'max_symbol_pct': 0, 'position_count': 0}

            # 2. Portfolio Greeks (for options)
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0

            for key in position_keys:
                pos_data = await self.redis.get(key)
                if pos_data:
                    pos = json.loads(pos_data)
                    contract_type = pos.get('contract', {}).get('type', pos.get('type'))
                    if contract_type == 'option':
                        greeks_data = await self.redis.get(f"positions:greeks:{pos.get('id')}")
                        greeks = {}
                        if greeks_data:
                            try:
                                greeks = json.loads(greeks_data)
                            except json.JSONDecodeError:
                                greeks = {}
                        if not greeks:
                            greeks = pos.get('greeks', {})

                        total_delta += float(greeks.get('delta', 0))
                        total_gamma += float(greeks.get('gamma', 0))
                        total_theta += float(greeks.get('theta', 0))
                        total_vega += float(greeks.get('vega', 0))

            metrics['greeks'] = {
                'delta': total_delta,
                'gamma': total_gamma,
                'theta': total_theta,
                'vega': total_vega
            }

            # 3. Risk scores
            account_value = float(await self.redis.get('account:value') or 100000)
            var_snapshot = await self._load_var_snapshot()
            var_95 = var_snapshot.var_95 if var_snapshot else 0.0
            drawdown_snapshot = await self._load_drawdown_snapshot()
            drawdown = drawdown_snapshot.drawdown_pct if drawdown_snapshot else 0.0

            # Calculate composite risk score (0-100)
            risk_score = 0
            risk_score += min(30, (var_95 / account_value) * 100 * 10)  # VaR component
            risk_score += min(30, drawdown * 3)  # Drawdown component
            risk_score += min(20, max_concentration * 100)  # Concentration component
            risk_score += min(20, position_count * 4)  # Position count component

            metrics['risk_score'] = {
                'total': min(100, risk_score),
                'rating': 'HIGH' if risk_score > 70 else 'MEDIUM' if risk_score > 40 else 'LOW'
            }

            # 4. Store all metrics
            await self.redis.setex('risk:metrics:summary', 60, json.dumps(metrics))
            await self.redis.setex('risk:metrics:timestamp', 60, time.time())

            # 5. Check for risk warnings
            if metrics['risk_score']['total'] > 70:
                self.logger.warning(f"High risk score: {metrics['risk_score']['total']:.0f}")

            if max_concentration > 0.3:  # 30% in one symbol
                self.logger.warning(f"High concentration risk: {max_concentration:.1%} in single symbol")

            if abs(total_gamma) > 1000:
                self.logger.warning(f"High gamma exposure: {total_gamma:.0f}")

        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")

    async def check_and_reset_daily_metrics(self):
        """
        Reset daily risk metrics only if it's a new trading day.
        """
        try:
            eastern = pytz.timezone('US/Eastern')
            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
            now_et = now_utc.astimezone(eastern)
            today = now_et.date()
            today_str = now_et.strftime('%Y%m%d')

            last_reset_raw = await self.redis.get('risk:daily:reset_time')
            last_reset_dt: Optional[datetime] = None
            if last_reset_raw:
                try:
                    last_reset_dt = datetime.fromtimestamp(float(last_reset_raw), tz=pytz.utc).astimezone(eastern)
                except (ValueError, OSError):
                    last_reset_dt = None

            if last_reset_dt and last_reset_dt.date() == today:
                self.logger.info(f"Daily metrics already set for {today_str}, not resetting")
                return

            if not last_reset_dt:
                daily_pnl = float(await self.redis.get('risk:daily_pnl') or 0.0)
                daily_trades = int(await self.redis.get('risk:daily_trades') or 0)
                open_positions = await self.redis.keys('positions:open:*')
                if abs(daily_pnl) > 1e-6 or daily_trades > 0 or open_positions:
                    self.logger.info(
                        "Daily metrics already active with no reset timestamp; preserving current totals"
                    )
                    return

            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

            if now_et < market_open:
                self.logger.info("Pre-market restart detected; resetting metrics ahead of open")

            await self.reset_daily_metrics()
            self.logger.info(f"Daily metrics reset for {today_str}")

        except Exception as e:
            self.logger.error(f"Error checking daily reset: {e}")

    async def reset_daily_metrics(self):
        """
        Actually reset daily risk metrics.
        """
        try:
            # Reset daily P&L and trades
            await self.redis.set('risk:daily_pnl', 0)
            await self.redis.set('risk:daily_trades', 0)
            await self.redis.set('risk:daily:reset_time', time.time())

            # Reset consecutive losses if profitable day
            yesterday_pnl = await self.redis.get('risk:yesterday_pnl')
            if yesterday_pnl and float(yesterday_pnl) > 0:
                await self.redis.set('risk:consecutive_losses', 0)
                await self.redis.set('risk:consecutive_losing_days', 0)

            # Clear circuit breakers (except system errors)
            self.circuit_breakers_tripped.discard('daily_loss')

            self.logger.info("Daily risk metrics reset")

        except Exception as e:
            self.logger.error(f"Error resetting daily metrics: {e}")
