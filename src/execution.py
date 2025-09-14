#!/usr/bin/env python3
"""
Execution Module - Order Execution, Position Management, Risk Control
Handles IBKR order execution, position lifecycle, risk management, and emergency procedures

Components:
- ExecutionManager: IBKR order placement and monitoring
- PositionManager: P&L tracking, stop management, scaling
- RiskManager: Circuit breakers, correlation limits, drawdown control
- EmergencyManager: Emergency close-all capabilities
"""

import asyncio
import json
import time
import redis
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, StopOrder
import logging


class ExecutionManager:
    """
    Manage order execution through IBKR.
    Handles order placement, monitoring, and fills.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize execution manager with configuration.
        
        TODO: Load configuration from config.yaml
        TODO: Set up Redis connection
        TODO: Initialize IBKR connection
        TODO: Set position limits from config
        """
        self.config = config  # Loaded from config.yaml
        self.redis = redis_conn
        self.ib = IB()
        
        # Position limits from config
        self.max_positions = config.get('trading', {}).get('max_positions', 5)
        self.max_per_symbol = config.get('trading', {}).get('max_per_symbol', 2)
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """
        Main execution loop for processing signals.
        
        TODO: Connect to IBKR Gateway/TWS
        TODO: Monitor for pending signals
        TODO: Check position limits before execution
        TODO: Execute signals that pass risk checks
        TODO: Monitor order status
        TODO: Handle fills and rejections
        
        Processing frequency: Every 200ms
        """
        self.logger.info("Starting execution manager...")
        
        # Connect to IBKR
        # TODO: Use connection params from config
        # TODO: Handle connection failures
        
        while True:
            # Check if trading is halted
            # TODO: Check global:halt flag
            
            # Process pending signals
            # TODO: Check position limits
            # TODO: Execute signals
            
            await asyncio.sleep(0.2)
    
    def passes_risk_checks(self, signal: dict) -> bool:
        """
        Perform pre-trade risk checks.
        
        TODO: Check buying power vs position size
        TODO: Check position correlation limits
        TODO: Verify daily loss limit not exceeded
        TODO: Check symbol exposure limits
        TODO: Validate contract is tradeable
        
        Risk checks:
        - Position size <= 25% of buying power
        - Correlation with existing positions < 0.7
        - Daily loss < $2000 limit
        - Symbol exposure < max_per_symbol
        
        Returns:
            True if all risk checks pass
        """
        pass
    
    async def execute_signal(self, signal: dict):
        """
        Execute a trading signal through IBKR.
        
        TODO: Create IB contract (Stock or Option)
        TODO: Qualify contract with IBKR
        TODO: Get current market data
        TODO: Determine order type (Market vs Limit)
        TODO: Calculate order size
        TODO: Place order through IBKR
        TODO: Store pending order in Redis
        TODO: Monitor order until filled
        
        Order logic:
        - Confidence > 85: Market order
        - Confidence <= 85: Limit order at mid
        
        Redis keys to update:
        - orders:pending:{order_id}
        - orders:working:{order_id}
        """
        pass
    
    def create_ib_contract(self, signal_contract: dict):
        """
        Create IBKR contract object from signal contract.
        
        TODO: Determine if Stock or Option
        TODO: For Options: Create Option with symbol, expiry, strike, right
        TODO: For Stocks: Create Stock with symbol, exchange, currency
        TODO: Set exchange to SMART for smart routing
        
        IBKR Contract types:
        - Stock(symbol, 'SMART', 'USD')
        - Option(symbol, expiry, strike, right, 'SMART')
        
        Returns:
            IB Contract object
        """
        pass
    
    def calculate_order_size(self, signal: dict, ticker) -> int:
        """
        Calculate appropriate order size.
        
        TODO: For options: Calculate contracts from position size and premium
        TODO: For stocks: Calculate shares from position size and price
        TODO: Apply minimum size (1 contract or 1 share)
        TODO: Apply maximum limits (50 contracts, position limits)
        
        Sizing logic:
        - Options: position_size / (premium * 100)
        - Stocks: position_size / stock_price
        
        Returns:
            Order size (contracts or shares)
        """
        pass
    
    async def monitor_order(self, trade, signal: dict):
        """
        Monitor order until completion.
        
        TODO: Wait for order to complete (filled/cancelled)
        TODO: On fill: Create position record
        TODO: On fill: Place stop loss order
        TODO: On rejection: Log and alert
        TODO: Update order status in Redis
        TODO: Clean up pending order record
        
        Position creation includes:
        - Entry price, size, time
        - Stop loss and targets
        - Strategy and signal reference
        """
        pass
    
    async def place_stop_loss(self, position: dict):
        """
        Place stop loss order for new position.
        
        TODO: Create IB contract for position
        TODO: Create StopOrder with opposite direction
        TODO: Set stop price from position data
        TODO: Place order through IBKR
        TODO: Store stop order ID with position
        
        Stop order is:
        - SELL for long positions
        - BUY for short positions
        """
        pass
    
    def handle_order_rejection(self, order, reason: str):
        """
        Handle rejected orders.
        
        TODO: Log rejection reason
        TODO: Alert via monitoring system
        TODO: Update Redis with rejection
        TODO: Consider retry logic for certain errors
        
        Common rejections:
        - Insufficient buying power
        - Contract not found
        - Outside trading hours
        - Position limit exceeded
        """
        pass
    
    def get_existing_position_symbols(self) -> list:
        """
        Get list of symbols with open positions.
        
        TODO: Query Redis for all position keys
        TODO: Extract symbol from each key
        TODO: Return unique symbols list
        
        Used for position limit checks
        
        Returns:
            List of symbols with positions
        """
        pass


class PositionManager:
    """
    Manage position lifecycle including P&L tracking and exit management.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize position manager with configuration.
        
        TODO: Load configuration from config.yaml
        TODO: Set up Redis connection
        TODO: Initialize position tracking
        """
        self.config = config  # Loaded from config.yaml
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """
        Main position management loop.
        
        TODO: Monitor all open positions
        TODO: Update P&L continuously
        TODO: Manage stop losses (trail if profitable)
        TODO: Check targets for scaling out
        TODO: Handle position closes
        
        Processing frequency: Every 1 second
        """
        self.logger.info("Starting position manager...")
        
        while True:
            # Get all open positions
            # TODO: Query Redis for positions
            # TODO: Update each position
            
            await asyncio.sleep(1)
    
    def update_position(self, position: dict):
        """
        Update position P&L and manage stops.
        
        TODO: Get current price from Redis
        TODO: Calculate unrealized P&L
        TODO: For options: Account for multiplier (100)
        TODO: Trail stop if position profitable
        TODO: Check if targets hit for scaling
        TODO: Update position in Redis
        
        P&L Calculation:
        - Long: (current - entry) * size
        - Short: (entry - current) * size
        - Options: Include 100x multiplier
        
        Redis keys to update:
        - positions:{symbol}:{position_id}
        """
        pass
    
    def calculate_unrealized_pnl(self, position: dict, current_price: float) -> float:
        """
        Calculate unrealized P&L for position.
        
        TODO: Determine if options or stock
        TODO: Apply direction (long/short)
        TODO: For options: Apply 100x multiplier
        TODO: Calculate based on current vs entry price
        
        Returns:
            Unrealized P&L in dollars
        """
        pass
    
    def trail_stop(self, position: dict, current_price: float):
        """
        Trail stop loss if position is profitable.
        
        TODO: Calculate profit percentage
        TODO: For longs: Trail stop up to lock in 50% of profit
        TODO: For shorts: Trail stop down to lock in 50% of profit
        TODO: Only trail if new stop is better than current
        TODO: Update stop in position record
        TODO: Place new stop order with IBKR
        
        Trailing logic:
        - Trail to breakeven at 1R profit
        - Trail to lock 50% of profit above 1R
        """
        pass
    
    def check_targets(self, position: dict, current_price: float):
        """
        Check if price targets hit for scaling out.
        
        TODO: Get current target index
        TODO: Check if price reached target
        TODO: Calculate scale-out size (33%, 50%, 100%)
        TODO: Execute scale-out order
        TODO: Update position size and realized P&L
        TODO: Move to next target
        
        Scaling plan:
        - Target 1: Close 33%
        - Target 2: Close 50% of remainder
        - Target 3: Close 100%
        """
        pass
    
    def scale_out(self, position: dict, target_price: float, target_index: int):
        """
        Scale out of position at target.
        
        TODO: Calculate shares/contracts to sell
        TODO: Place market order to close partial position
        TODO: Update position size
        TODO: Calculate and add to realized P&L
        TODO: Update current target index
        TODO: Close position if fully scaled out
        
        Scale percentages: [0.33, 0.50, 1.0]
        """
        pass
    
    def close_position(self, position: dict, exit_price: float, reason: str):
        """
        Close position and update records.
        
        TODO: Mark position as CLOSED
        TODO: Record exit price and time
        TODO: Calculate final realized P&L
        TODO: Update global P&L counters
        TODO: Cancel stop loss order
        TODO: Store exit reason
        
        Redis keys to update:
        - positions:{symbol}:{position_id}
        - global:pnl:realized
        - global:positions:count
        """
        pass
    
    def get_position_summary(self) -> dict:
        """
        Get summary of all open positions.
        
        TODO: Query all position records
        TODO: Calculate total unrealized P&L
        TODO: Count positions by strategy
        TODO: Calculate exposure by symbol
        TODO: Determine portfolio Greeks (if options)
        
        Returns:
            Position summary dictionary
        """
        pass


class RiskManager:
    """
    Monitor risk metrics and enforce trading limits through circuit breakers.
    Production-ready risk management with multiple safety layers.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn):
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
        
        # Reset daily metrics at market open
        await self.reset_daily_metrics()
        
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
                var_95 = await self.calculate_var()
                if var_95:
                    await self.redis.setex('risk:var:portfolio', 300, json.dumps({
                        'var_95': var_95,
                        'confidence': self.var_confidence,
                        'timestamp': current_time
                    }))
                
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
    
    async def calculate_var(self, confidence: float = 0.95) -> float:
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
                return await self._calculate_position_based_var()
            
            # Parse P&L values
            pnl_values = []
            for pnl_json in pnl_history:
                try:
                    pnl_data = json.loads(pnl_json)
                    pnl_values.append(float(pnl_data.get('pnl', 0)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            if len(pnl_values) < 20:
                return await self._calculate_position_based_var()
            
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
            
            # Store VaR metrics
            await self.redis.setex('risk:var:detailed', 300, json.dumps({
                'var_95': final_var,
                'historical_var': var_value,
                'parametric_var': parametric_var,
                'mean_return': mean_return,
                'std_return': std_return,
                'confidence': confidence,
                'data_points': len(returns),
                'timestamp': time.time()
            }))
            
            # Check VaR against limits
            account_value = float(await self.redis.get('account:value') or 100000)
            var_pct = (final_var / account_value) * 100
            
            if var_pct > 5:  # VaR exceeds 5% of account
                self.logger.warning(f"High VaR detected: ${final_var:,.2f} ({var_pct:.1f}% of account)")
                # Reduce position sizes when VaR is high
                await self.redis.set('risk:high_var_flag', 'true')
            else:
                await self.redis.set('risk:high_var_flag', 'false')
            
            return final_var
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            # Fallback to position-based estimate
            return await self._calculate_position_based_var()
    
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
                    if pos.get('type') == 'option':
                        total_delta += pos.get('delta', 0) * pos.get('quantity', 0) * 100
                        total_gamma += pos.get('gamma', 0) * pos.get('quantity', 0) * 100
                        total_theta += pos.get('theta', 0) * pos.get('quantity', 0) * 100
                        total_vega += pos.get('vega', 0) * pos.get('quantity', 0) * 100
            
            metrics['greeks'] = {
                'delta': total_delta,
                'gamma': total_gamma,
                'theta': total_theta,
                'vega': total_vega
            }
            
            # 3. Risk scores
            account_value = float(await self.redis.get('account:value') or 100000)
            var_95 = float(await self.redis.get('risk:var:portfolio') or 0)
            drawdown = float((await self.redis.get('risk:current_drawdown') or '{}') and 
                           json.loads(await self.redis.get('risk:current_drawdown') or '{}').get('drawdown_pct', 0))
            
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
    
    async def reset_daily_metrics(self):
        """
        Reset daily risk metrics at market open.
        """
        try:
            # Reset daily P&L
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
    
    def calculate_position_sizes(self) -> dict:
        """
        Calculate current position sizes and exposures.
        
        TODO: Get all open positions
        TODO: Calculate dollar exposure per position
        TODO: Calculate exposure by symbol
        TODO: Calculate exposure by strategy
        TODO: Check concentration limits
        
        Returns:
            Position size breakdown
        """
        pass


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
        self.ib = IB()
        self.logger = logging.getLogger(__name__)
    
    async def emergency_close_all(self):
        """
        Emergency close all positions immediately.
        
        TODO: Connect to IBKR if not connected
        TODO: Cancel all pending orders
        TODO: Get all open positions
        TODO: Place market orders to close all
        TODO: Halt trading permanently
        TODO: Send emergency alerts
        
        This is the nuclear option - use carefully!
        """
        self.logger.critical("EMERGENCY: Closing all positions!")
        
        # Connect to IBKR
        # TODO: Ensure connection established
        
        # Cancel pending orders
        # TODO: Get all orders and cancel
        
        # Close all positions
        # TODO: Market orders for immediate exit
        
        # Halt trading
        # TODO: Set permanent halt flag
        pass
    
    async def close_position_emergency(self, position: dict):
        """
        Close a single position at market.
        
        TODO: Create IB contract from position
        TODO: Determine close direction (opposite of position)
        TODO: Create market order for full size
        TODO: Place order immediately
        TODO: Update position status
        
        Emergency close uses market orders only
        """
        pass
    
    def cancel_all_orders(self):
        """
        Cancel all pending orders.
        
        TODO: Get list of all open orders from IBKR
        TODO: Cancel each order
        TODO: Clear Redis order records
        TODO: Log cancellations
        
        Includes:
        - Working orders
        - Stop orders
        - Pending orders
        """
        pass
    
    def trigger_emergency_protocol(self, reason: str):
        """
        Initiate emergency protocol.
        
        TODO: Log emergency trigger
        TODO: Send alerts to all channels
        TODO: Initiate position closes
        TODO: Save system state
        TODO: Generate incident report
        
        Emergency triggers:
        - System failure
        - Massive drawdown
        - Market crash
        - Technical malfunction
        """
        pass
    
    def save_emergency_state(self):
        """
        Save system state for post-mortem analysis.
        
        TODO: Dump all Redis data
        TODO: Save position records
        TODO: Save order history
        TODO: Save market data snapshot
        TODO: Create timestamped backup
        
        Saved to: data/emergency/[timestamp]/
        """
        pass


class CircuitBreakers:
    """
    Automated circuit breakers for risk control.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: redis.Redis):
        """
        Initialize circuit breakers.
        
        TODO: Load breaker configurations
        TODO: Set up Redis connection
        """
        self.config = config
        self.redis = redis_conn
        self.breakers = {
            'daily_loss': {'limit': 2000, 'current': 0},
            'correlation': {'limit': 0.8, 'current': 0},
            'drawdown': {'limit': 0.10, 'current': 0},
            'consecutive_losses': {'limit': 3, 'current': 0},
            'position_limit': {'limit': 5, 'current': 0}
        }
        self.logger = logging.getLogger(__name__)
    
    def check_breaker(self, breaker_name: str, current_value: float) -> bool:
        """
        Check if circuit breaker should trip.
        
        TODO: Update current value
        TODO: Compare with limit
        TODO: Return True if limit exceeded
        TODO: Log breaker status
        
        Returns:
            True if breaker should trip
        """
        pass
    
    def reset_daily_breakers(self):
        """
        Reset daily circuit breakers.
        
        TODO: Reset daily loss counter
        TODO: Reset consecutive losses
        TODO: Update reset timestamp
        TODO: Log reset event
        
        Called at market open each day
        """
        pass
    
    def get_breaker_status(self) -> dict:
        """
        Get current status of all breakers.
        
        TODO: Calculate current values
        TODO: Compare with limits
        TODO: Calculate distance to limits
        TODO: Return status summary
        
        Returns:
            Breaker status dictionary
        """
        pass