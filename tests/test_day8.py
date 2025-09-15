#!/usr/bin/env python3
"""
Day 8 Test Suite - Execution Manager
Tests for IBKR order execution, fill handling, and stop loss placement
"""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution import ExecutionManager


class TestExecutionManager:
    """Test suite for ExecutionManager class."""
    
    @pytest.fixture
    async def setup(self):
        """Set up test fixtures."""
        # Mock config
        config = {
            'ibkr': {
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 1
            },
            'trading': {
                'max_positions': 5,
                'max_per_symbol': 2,
                'max_0dte_contracts': 50,
                'max_other_contracts': 100
            },
            'symbols': {
                'level2': ['SPY', 'QQQ'],
                'standard': ['AAPL', 'TSLA']
            },
            'risk_management': {
                'correlation_limit': 0.7
            }
        }
        
        # Create mock Redis
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.set = AsyncMock(return_value=True)
        redis_mock.setex = AsyncMock(return_value=True)
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.rpop = AsyncMock(return_value=None)
        redis_mock.incr = AsyncMock(return_value=1)
        redis_mock.incrbyfloat = AsyncMock(return_value=1.0)
        redis_mock.sadd = AsyncMock(return_value=1)
        redis_mock.lpush = AsyncMock(return_value=1)
        redis_mock.ltrim = AsyncMock(return_value=True)
        redis_mock.publish = AsyncMock(return_value=1)
        redis_mock.hincrby = AsyncMock(return_value=1)
        
        # Create ExecutionManager with mocked IBKR
        exec_manager = ExecutionManager(config, redis_mock)
        
        # Mock IBKR connection - use correct Mock types for sync/async methods
        exec_manager.ib = Mock()  # Main IB object is not async
        exec_manager.ib.isConnected = Mock(return_value=True)  # sync
        exec_manager.ib.connectAsync = AsyncMock(return_value=True)  # async
        exec_manager.ib.disconnect = Mock()  # sync
        exec_manager.ib.accountValues = Mock(return_value=[  # sync in ib_insync
            Mock(tag='NetLiquidation', value='100000'),
            Mock(tag='BuyingPower', value='50000')
        ])
        exec_manager.ib.qualifyContracts = Mock(return_value=[Mock()])  # sync in ib_insync
        exec_manager.ib.reqMktData = Mock()  # sync
        exec_manager.ib.placeOrder = Mock()  # sync
        exec_manager.ib.cancelOrder = Mock()  # sync
        # Add event handlers as Mock objects
        exec_manager.ib.orderStatusEvent = Mock()
        exec_manager.ib.execDetailsEvent = Mock()
        exec_manager.ib.errorEvent = Mock()
        
        return exec_manager, redis_mock, config
    
    @pytest.mark.asyncio
    async def test_ibkr_connection(self, setup):
        """Test IBKR connection establishment."""
        exec_manager, redis_mock, config = await setup
        
        # Test connection
        await exec_manager.connect_ibkr()
        
        # Verify connection attempted
        exec_manager.ib.connectAsync.assert_called_once()
        
        # Verify account values fetched (sync method)
        exec_manager.ib.accountValues.assert_called_once()
        
        # Verify Redis updated
        redis_mock.set.assert_any_call('account:value', '100000')
        redis_mock.set.assert_any_call('account:buying_power', '50000')
        redis_mock.set.assert_any_call('execution:connection:status', 'connected')
    
    @pytest.mark.asyncio
    async def test_risk_checks(self, setup):
        """Test pre-trade risk checks."""
        exec_manager, redis_mock, config = await setup
        
        # Setup mock data
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'risk:new_positions_allowed': 'true',
            'account:buying_power': '50000',
            'risk:daily_pnl': '-500'
        }.get(key))
        
        redis_mock.keys = AsyncMock(return_value=['positions:open:1'])
        
        # Create test signal
        signal = {
            'symbol': 'SPY',
            'side': 'LONG',
            'position_size': 5000  # 10% of buying power
        }
        
        # Test risk checks pass
        result = await exec_manager.passes_risk_checks(signal)
        assert result is True
        
        # Test with excessive position size
        signal['position_size'] = 20000  # 40% of buying power
        result = await exec_manager.passes_risk_checks(signal)
        assert result is False
        
        # Test with daily loss limit exceeded
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'risk:new_positions_allowed': 'true',
            'account:buying_power': '50000',
            'risk:daily_pnl': '-2500'  # Exceeds $2000 limit
        }.get(key))
        
        signal['position_size'] = 5000
        result = await exec_manager.passes_risk_checks(signal)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_option_contract(self, setup):
        """Test option contract creation from signal."""
        exec_manager, redis_mock, config = await setup
        
        # Mock qualified contract (qualifyContracts is sync in ib_insync)
        mock_contract = Mock()
        mock_contract.symbol = 'SPY'
        mock_contract.secType = 'OPT'
        mock_contract.exchange = 'SMART'
        mock_contract.currency = 'USD'
        mock_contract.strike = 450
        mock_contract.right = 'C'
        mock_contract.lastTradeDateOrContractMonth = '20240315'
        exec_manager.ib.qualifyContracts = Mock(return_value=[mock_contract])
        
        # Test with OCC symbol
        contract_info = {
            'type': 'option',
            'symbol': 'SPY',
            'occ_symbol': 'SPY240315C00450000'  # SPY Mar 15 2024 450 Call
        }
        
        result = await exec_manager.create_ib_contract(contract_info)
        
        assert result is not None
        exec_manager.ib.qualifyContracts.assert_called_once()
        
        # Verify caching
        redis_mock.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_calculate_order_size_options(self, setup):
        """Test order size calculation for options."""
        exec_manager, redis_mock, config = await setup
        
        # Setup mock data
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'account:buying_power': '50000',
            'account:value': '100000',
            'risk:position_size_multiplier': '1.0'
        }.get(key))
        
        # Create mock ticker
        ticker = Mock()
        ticker.bid = 2.50
        ticker.ask = 2.60
        ticker.last = 2.55
        
        # Test 0DTE signal
        signal = {
            'confidence': 85,
            'strategy': '0DTE',
            'contract': {'type': 'option'},
            'entry': 2.55
        }
        
        size = await exec_manager.calculate_order_size(signal, ticker)
        
        # Should calculate based on account size and confidence
        # Base: 100k * (0.02 + 0.03 * 0.85) = 4550
        # Option contracts: 4550 / (2.55 * 100) = 17 contracts
        # Limited by 0DTE max (50)
        assert size >= 1
        assert size <= 50
    
    @pytest.mark.asyncio
    async def test_execute_signal(self, setup):
        """Test signal execution flow."""
        exec_manager, redis_mock, config = await setup
        
        # Mock contract creation
        mock_contract = Mock()
        exec_manager.create_ib_contract = AsyncMock(return_value=mock_contract)
        
        # Mock ticker
        ticker = Mock()
        ticker.bid = 2.50
        ticker.ask = 2.60
        ticker.last = 2.55
        exec_manager.ib.reqMktData = Mock(return_value=ticker)
        
        # Mock order placement
        mock_trade = Mock()
        mock_trade.order = Mock(orderId=12345)
        mock_trade.orderStatus = Mock(status='Submitted')
        mock_trade.isDone = Mock(return_value=False)
        exec_manager.ib.placeOrder = Mock(return_value=mock_trade)
        
        # Mock monitor_order to avoid waiting
        exec_manager.monitor_order = AsyncMock()
        
        # Create test signal
        signal = {
            'id': 'sig_123',
            'symbol': 'SPY',
            'side': 'LONG',
            'confidence': 75,
            'strategy': '1DTE',
            'contract': {
                'type': 'option',
                'symbol': 'SPY',
                'occ_symbol': 'SPY240316C00450000'
            },
            'entry': 2.55,
            'stop': 1.50,
            'targets': [3.00, 3.50, 4.00]
        }
        
        # Execute signal
        await exec_manager.execute_signal(signal)
        
        # Verify order placed
        exec_manager.ib.placeOrder.assert_called_once()
        
        # Verify order stored in Redis
        redis_mock.setex.assert_called()
        
        # Verify monitoring started
        exec_manager.monitor_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_fill(self, setup):
        """Test fill handling and position creation."""
        exec_manager, redis_mock, config = await setup
        
        # Mock stop loss placement
        exec_manager.place_stop_loss = AsyncMock()
        
        # Create mock trade with fill
        mock_fill = Mock()
        mock_fill.execution = Mock(shares=10, price=2.55)
        mock_fill.time = datetime.now()
        mock_fill.commissionReport = Mock(commission=1.50)
        
        mock_trade = Mock()
        mock_trade.fills = [mock_fill]
        mock_trade.order = Mock(orderId=12345)
        
        # Create signal
        signal = {
            'id': 'sig_123',
            'symbol': 'SPY',
            'side': 'LONG',
            'strategy': '0DTE',
            'contract': {'type': 'option'},
            'stop': 1.50,
            'targets': [3.00, 3.50, 4.00]
        }
        
        # Handle fill
        await exec_manager.handle_fill(mock_trade, signal)
        
        # Verify position created in Redis
        setex_calls = redis_mock.setex.call_args_list
        position_calls = [call for call in setex_calls 
                         if 'positions:open:SPY' in call[0][0]]
        assert len(position_calls) > 0
        
        # Verify position data
        position_data = json.loads(position_calls[0][0][2])
        assert position_data['symbol'] == 'SPY'
        assert position_data['side'] == 'LONG'
        assert position_data['quantity'] == 10
        assert position_data['entry_price'] == 2.55
        assert position_data['commission'] == 1.50
        
        # Verify stop loss placed
        exec_manager.place_stop_loss.assert_called_once()
        
        # Verify metrics updated
        redis_mock.incr.assert_any_call('execution:fills:total')
        redis_mock.incrbyfloat.assert_any_call('execution:commission:total', 1.50)
    
    @pytest.mark.asyncio
    async def test_place_stop_loss(self, setup):
        """Test stop loss order placement."""
        exec_manager, redis_mock, config = await setup
        
        # Mock contract creation
        mock_contract = Mock()
        exec_manager.create_ib_contract = AsyncMock(return_value=mock_contract)
        
        # Mock stop order placement
        mock_stop_trade = Mock()
        mock_stop_trade.order = Mock(orderId=54321)
        exec_manager.ib.placeOrder = Mock(return_value=mock_stop_trade)
        
        # Create position with stop
        position = {
            'id': 'pos_123',
            'symbol': 'SPY',
            'side': 'LONG',
            'strategy': '0DTE',
            'quantity': 10,
            'entry_price': 2.55,
            'stop_loss': 1.50,
            'contract': {'type': 'option'}
        }
        
        # Place stop loss
        await exec_manager.place_stop_loss(position)
        
        # Verify stop order placed
        exec_manager.ib.placeOrder.assert_called_once()
        order_call = exec_manager.ib.placeOrder.call_args[0]
        stop_order = order_call[1]
        
        # Verify stop order details
        assert stop_order.action == 'SELL'  # Opposite of LONG
        assert stop_order.totalQuantity == 10
        assert stop_order.auxPrice == 1.50  # StopOrder uses auxPrice, not stopPrice
        
        # Verify position updated with stop order ID
        assert position['stop_order_id'] == 54321
    
    @pytest.mark.asyncio
    async def test_order_rejection_handling(self, setup):
        """Test order rejection handling."""
        exec_manager, redis_mock, config = await setup
        
        # Create mock order
        mock_order = Mock(orderId=12345)
        
        # Test rejection handling
        await exec_manager.handle_order_rejection(mock_order, 'Insufficient Buying Power')
        
        # Verify metrics updated
        redis_mock.incr.assert_any_call('execution:rejections:total')
        redis_mock.incr.assert_any_call('execution:rejections:reason:Insufficient Buying Power')
        
        # Verify rejection logged
        redis_mock.lpush.assert_called()
        
        # Verify critical alert published
        redis_mock.publish.assert_called_once()
        publish_call = redis_mock.publish.call_args[0]
        assert publish_call[0] == 'alerts:critical'
        alert_data = json.loads(publish_call[1])
        assert alert_data['type'] == 'order_rejection'
        assert alert_data['severity'] == 'critical'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])