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
import redis.asyncio as aioredis
import uuid
import numpy as np
import pytz
from datetime import datetime
from typing import Dict, List, Any, Optional
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, StopOrder
import logging


class ExecutionManager:
    """
    Manage order execution through IBKR.
    Handles order placement, monitoring, and fills.
    """
    
    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """
        Initialize execution manager with configuration.
        Uses separate IBKR connection for execution (different from data connection).
        """
        self.config = config
        self.redis = redis_conn
        self.ib = IB()
        
        # IBKR configuration for execution
        self.ibkr_config = config.get('ibkr', {})
        self.host = self.ibkr_config.get('host', '127.0.0.1')
        self.port = self.ibkr_config.get('port', 7497)  # Paper trading port
        self.client_id = self.ibkr_config.get('client_id', 1) + 100  # Different client ID for execution
        
        # Position limits from config
        self.max_positions = config.get('trading', {}).get('max_positions', 5)
        self.max_per_symbol = config.get('trading', {}).get('max_per_symbol', 2)
        self.max_0dte_contracts = config.get('trading', {}).get('max_0dte_contracts', 50)
        self.max_other_contracts = config.get('trading', {}).get('max_other_contracts', 100)
        
        # Order management
        self.pending_orders = {}  # order_id -> order details
        self.active_stops = {}    # position_id -> stop order
        
        # Connection state
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """
        Main execution loop for processing signals.
        Processing frequency: Every 200ms
        """
        self.logger.info("Starting execution manager...")
        
        # Connect to IBKR
        await self.connect_ibkr()
        
        # Set up event handlers (ib_insync specific events)
        self.ib.orderStatusEvent += self.on_order_status
        self.ib.execDetailsEvent += self.on_fill  # execDetailsEvent for fills in ib_insync
        self.ib.errorEvent += self.on_error
        
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
                
                self.logger.info(f"Connecting to IBKR at {self.host}:{self.port} (client_id={self.client_id})")
                await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=10)
                
                # Verify connection
                if self.ib.isConnected():
                    self.connected = True
                    self.reconnect_attempts = 0
                    self.logger.info("IBKR execution connection established")
                    
                    # Get account info (ib_insync doesn't have async version)
                    account_values = self.ib.accountValues()
                    for av in account_values:
                        if av.tag == 'NetLiquidation':
                            await self.redis.set('account:value', av.value)
                        elif av.tag == 'BuyingPower':
                            await self.redis.set('account:buying_power', av.value)
                    
                    await self.redis.set('execution:connection:status', 'connected')
                    return True
                    
            except Exception as e:
                self.logger.error(f"IBKR connection failed: {e}")
                self.reconnect_attempts += 1
                await asyncio.sleep(min(2 ** self.reconnect_attempts, 60))
        
        self.logger.error("Max reconnection attempts reached")
        await self.redis.set('execution:connection:status', 'failed')
        return False
    
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
            signal_data = await self.redis.rpop(f'signals:pending:{symbol}')
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
    
    async def passes_risk_checks(self, signal: dict) -> bool:
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
            position_size = signal.get('position_size', 0)
            
            if position_size > buying_power * 0.25:  # Max 25% of buying power
                self.logger.warning(f"Position size {position_size} exceeds 25% of buying power {buying_power}")
                return False
            
            # Check daily loss limit
            daily_pnl = float(await self.redis.get('risk:daily_pnl') or 0)
            if daily_pnl < -2000:  # $2000 daily loss limit
                self.logger.warning(f"Daily loss limit exceeded: {daily_pnl}")
                return False
            
            # Check correlation limits (delegated to RiskManager)
            symbol = signal.get('symbol')
            side = signal.get('side')
            
            # Use RiskManager's correlation check
            risk_manager = RiskManager(self.config, self.redis)
            if not await risk_manager.check_correlations(symbol, side):
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
            
            self.logger.info(f"Executing signal: {symbol} {side} {strategy} (confidence={confidence}%)")
            
            # Create IB contract
            contract_info = signal.get('contract', {})
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
            
            # Determine order type and price
            action = 'BUY' if side == 'LONG' else 'SELL'
            
            if confidence > 85 and strategy != '0DTE':  # Never use market for 0DTE
                # Market order for high confidence (except 0DTE)
                order = MarketOrder(action, order_size)
                order_type = 'MARKET'
            else:
                # Limit order
                if ticker.bid and ticker.ask:
                    spread = ticker.ask - ticker.bid
                    if confidence >= 70:
                        # Limit at mid
                        limit_price = round((ticker.bid + ticker.ask) / 2, 2)
                    else:
                        # Limit closer to favorable side
                        if action == 'BUY':
                            limit_price = round(ticker.bid + spread * 0.33, 2)
                        else:
                            limit_price = round(ticker.ask - spread * 0.33, 2)
                else:
                    # Fallback to last price
                    limit_price = ticker.last or signal.get('entry', 0)
                
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
                'order_type': order_type,
                'confidence': confidence,
                'strategy': strategy,
                'entry_target': signal.get('entry'),
                'stop_loss': signal.get('stop'),
                'targets': signal.get('targets', []),
                'placed_at': time.time(),
                'status': 'PENDING'
            }
            
            await self.redis.setex(
                f'orders:pending:{order_id}',
                300,  # 5 minute TTL
                json.dumps(order_data)
            )
            
            self.pending_orders[order_id] = order_data
            
            # Monitor order
            await self.monitor_order(trade, signal)
            
            self.logger.info(f"Order {order_id} placed: {action} {order_size} {symbol} @ {order_type}")
            
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
            contract_type = signal_contract.get('type', 'stock')
            symbol = signal_contract.get('symbol')
            
            if contract_type == 'option':
                # Parse OCC symbol if provided
                occ_symbol = signal_contract.get('occ_symbol')
                if occ_symbol:
                    # OCC format: SPYYYMMDDCP00500000
                    # Extract components using regex from analytics.py
                    import re
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
                
                # Create Option contract
                from ib_insync import Option
                contract = Option(root, expiry, strike, right, 'SMART', currency='USD')
                contract.multiplier = '100'
                
            else:
                # Stock contract
                from ib_insync import Stock
                contract = Stock(symbol, 'SMART', 'USD')
            
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
            contract_type = signal.get('contract', {}).get('type', 'stock')
            
            # Base position size: 2-5% of account based on confidence
            base_position_size = account_value * (0.02 + 0.03 * confidence) * risk_multiplier
            
            # Cap at 25% of buying power
            max_position_size = buying_power * 0.25
            position_size = min(base_position_size, max_position_size)
            
            if contract_type == 'option':
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
                    if strategy == '0DTE':
                        max_contracts = min(self.max_0dte_contracts, 50)
                    else:
                        max_contracts = min(self.max_other_contracts, 100)
                    
                    # Apply confidence-based scaling
                    if confidence < 0.7:
                        max_contracts = int(max_contracts * 0.5)
                    elif confidence < 0.85:
                        max_contracts = int(max_contracts * 0.75)
                    
                    # Minimum 1 contract, maximum based on strategy
                    return max(1, min(contracts, max_contracts))
                else:
                    return 1  # Default to 1 contract
                    
            else:
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
                    return shares
                else:
                    return 100  # Default to 100 shares
                    
        except Exception as e:
            self.logger.error(f"Error calculating order size: {e}")
            return 1  # Conservative default
    
    async def monitor_order(self, trade, signal: dict):
        """
        Monitor order until completion.
        """
        order_id = trade.order.orderId
        max_wait_time = 60 if trade.order.orderType == 'LMT' else 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            await asyncio.sleep(0.2)
            
            if trade.isDone():
                if trade.orderStatus.status == 'Filled':
                    # Order filled - create position
                    await self.handle_fill(trade, signal)
                    break
                elif trade.orderStatus.status in ['Cancelled', 'ApiCancelled']:
                    self.logger.warning(f"Order {order_id} cancelled")
                    await self.redis.delete(f'orders:pending:{order_id}')
                    break
                elif trade.orderStatus.status == 'Inactive':
                    # Order rejected
                    self.logger.error(f"Order {order_id} rejected")
                    await self.handle_order_rejection(trade.order, 'Inactive')
                    break
            
            # Update order status in Redis
            if order_id in self.pending_orders:
                self.pending_orders[order_id]['status'] = trade.orderStatus.status
                await self.redis.setex(
                    f'orders:pending:{order_id}',
                    300,
                    json.dumps(self.pending_orders[order_id])
                )
        
        # Timeout - cancel order
        if not trade.isDone():
            self.logger.warning(f"Order {order_id} timed out after {max_wait_time}s")
            self.ib.cancelOrder(trade.order)
            await self.redis.delete(f'orders:pending:{order_id}')
    
    async def handle_fill(self, trade, signal: dict):
        """
        Process filled orders and create position.
        """
        try:
            fill = trade.fills[-1] if trade.fills else None
            if not fill:
                self.logger.error("No fill data available")
                return
            
            order_id = trade.order.orderId
            symbol = signal.get('symbol')
            
            # Create position record
            position_id = str(uuid.uuid4())
            position = {
                'id': position_id,
                'symbol': symbol,
                'contract': signal.get('contract'),
                'side': signal.get('side'),
                'strategy': signal.get('strategy'),
                'quantity': fill.execution.shares,
                'entry_price': fill.execution.price,
                'entry_time': fill.time.isoformat(),
                'commission': fill.commissionReport.commission if fill.commissionReport else 0,
                'stop_loss': signal.get('stop'),
                'targets': signal.get('targets', []),
                'current_price': fill.execution.price,
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'stop_order_id': None,
                'status': 'OPEN',
                'order_id': order_id,
                'signal_id': signal.get('id')
            }
            
            # Store position in Redis
            await self.redis.setex(
                f'positions:open:{symbol}:{position_id}',
                86400,  # 24 hour TTL
                json.dumps(position)
            )
            
            # Add to symbol index
            await self.redis.sadd(f'positions:by_symbol:{symbol}', position_id)
            
            # Update metrics
            await self.redis.incr('execution:fills:total')
            await self.redis.incr('execution:fills:daily')
            
            # Calculate and store commission
            commission = fill.commissionReport.commission if fill.commissionReport else 0
            await self.redis.incrbyfloat('execution:commission:total', commission)
            
            # Place stop loss order
            await self.place_stop_loss(position)
            
            # Clean up pending order
            await self.redis.delete(f'orders:pending:{order_id}')
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            # Log fill
            self.logger.info(
                f"Position created: {symbol} {position['side']} "
                f"{position['quantity']} @ ${position['entry_price']:.2f} "
                f"(commission=${commission:.2f})"
            )
            
            # Store fill record
            fill_record = {
                'order_id': order_id,
                'position_id': position_id,
                'symbol': symbol,
                'side': signal.get('side'),
                'quantity': fill.execution.shares,
                'price': fill.execution.price,
                'commission': commission,
                'time': fill.time.isoformat()
            }
            
            await self.redis.lpush(
                f'orders:fills:{datetime.now().strftime("%Y%m%d")}',
                json.dumps(fill_record)
            )
            
        except Exception as e:
            self.logger.error(f"Error handling fill: {e}")
    
    async def place_stop_loss(self, position: dict):
        """
        Place stop loss order for new position.
        """
        try:
            # Create IB contract
            contract_info = position.get('contract', {})
            ib_contract = await self.create_ib_contract(contract_info)
            if not ib_contract:
                self.logger.error(f"Failed to create contract for stop loss")
                return
            
            # Determine stop action (opposite of entry)
            side = position.get('side')
            stop_action = 'SELL' if side == 'LONG' else 'BUY'
            
            # Get stop price from position
            stop_price = position.get('stop_loss')
            if not stop_price:
                # Calculate default stop based on strategy
                entry_price = position.get('entry_price')
                strategy = position.get('strategy')
                
                if strategy == '0DTE':
                    # 50% stop for 0DTE options
                    stop_price = entry_price * 0.5 if side == 'LONG' else entry_price * 1.5
                elif strategy == '1DTE':
                    # 25% stop for 1DTE
                    stop_price = entry_price * 0.75 if side == 'LONG' else entry_price * 1.25
                else:
                    # 30% stop for longer dated
                    stop_price = entry_price * 0.7 if side == 'LONG' else entry_price * 1.3
            
            # Round stop price
            stop_price = round(stop_price, 2)
            
            # Create stop order
            quantity = position.get('quantity')
            stop_order = StopOrder(stop_action, quantity, stop_price)
            
            # Place stop order
            stop_trade = self.ib.placeOrder(ib_contract, stop_order)
            stop_order_id = stop_trade.order.orderId
            
            # Update position with stop order ID
            position['stop_order_id'] = stop_order_id
            position_id = position.get('id')
            symbol = position.get('symbol')
            
            await self.redis.setex(
                f'positions:open:{symbol}:{position_id}',
                86400,
                json.dumps(position)
            )
            
            # Track active stop
            self.active_stops[position_id] = stop_trade
            
            self.logger.info(
                f"Stop loss placed: {stop_action} {quantity} @ ${stop_price:.2f} "
                f"for position {position_id[:8]}"
            )
            
        except Exception as e:
            self.logger.error(f"Error placing stop loss: {e}")
    
    async def handle_order_rejection(self, order, reason: str):
        """
        Handle rejected orders.
        """
        try:
            order_id = order.orderId
            
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
                'order_details': self.pending_orders.get(order_id, {})
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
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
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
    
    def on_fill(self, fill):
        """
        Handle fill events from IBKR.
        """
        try:
            self.logger.info(
                f"Fill received: {fill.contract.symbol} "
                f"{fill.execution.shares} @ ${fill.execution.price:.2f}"
            )
            
            # Update metrics asynchronously
            asyncio.create_task(self.update_fill_metrics(fill))
            
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
                self.logger.info(f"Order {reqId} cancelled")
            elif errorCode in [2104, 2106, 2158]:  # Connection OK messages
                self.logger.info(f"IBKR Info: {errorString}")
            elif errorCode == 1100:  # Connection lost
                self.logger.critical("IBKR connection lost!")
                self.connected = False
            
        except Exception as e:
            self.logger.error(f"Error handling IBKR error: {e}")
    
    async def update_fill_metrics(self, fill):
        """
        Update fill metrics in Redis.
        """
        try:
            commission = fill.commissionReport.commission if fill.commissionReport else 0
            await self.redis.incrbyfloat('execution:commission:daily', commission)
            await self.redis.hincrby('execution:fills:by_symbol', fill.contract.symbol, 1)
        except Exception as e:
            self.logger.error(f"Error updating fill metrics: {e}")
    
    async def handle_order_rejection_by_id(self, order_id: int, reason: str):
        """
        Handle order rejection by order ID.
        """
        if order_id in self.pending_orders:
            order_data = self.pending_orders[order_id]
            await self.handle_order_rejection(None, reason)
    
    async def update_order_status(self):
        """
        Update status of all pending orders.
        """
        for order_id, order_data in list(self.pending_orders.items()):
            # Check if order is stale
            if time.time() - order_data.get('placed_at', 0) > 300:  # 5 minutes
                self.logger.warning(f"Removing stale order {order_id}")
                del self.pending_orders[order_id]
                await self.redis.delete(f'orders:pending:{order_id}')


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
                    if occ_symbol and occ_symbol in chain:
                        option_data = chain[occ_symbol]
                        return option_data.get('last_price', option_data.get('mark', 0))
            
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
                if occ_symbol in chain:
                    option_data = chain[occ_symbol]
                    
                    # Extract Greeks from Alpha Vantage data
                    quantity = position.get('quantity', 0)
                    greeks = {
                        'delta': option_data.get('delta', 0) * quantity * 100,  # Position delta
                        'gamma': option_data.get('gamma', 0) * quantity * 100,  # Position gamma
                        'theta': option_data.get('theta', 0) * quantity * 100,  # Daily theta
                        'vega': option_data.get('vega', 0) * quantity * 100,    # Vega per 1% IV
                        'rho': option_data.get('rho', 0) * quantity * 100,      # Rho
                        'iv': option_data.get('implied_volatility', 0)          # IV
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