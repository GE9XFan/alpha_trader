#!/usr/bin/env python3
"""
Day 9 Test Suite - Position Manager
Tests for P&L tracking, stop trailing, scaling, and position lifecycle
"""

import pytest
import asyncio
import json
import time
import pytz
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution import PositionManager


class TestPositionManager:
    """Test suite for PositionManager class."""
    
    @pytest.fixture
    async def setup(self):
        """Set up test fixtures."""
        # Mock config
        config = {
            'ibkr': {
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 1
            }
        }
        
        # Create mock Redis
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.set = AsyncMock(return_value=True)
        redis_mock.setex = AsyncMock(return_value=True)
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.incr = AsyncMock(return_value=1)
        redis_mock.incrbyfloat = AsyncMock(return_value=1.0)
        redis_mock.srem = AsyncMock(return_value=1)
        
        # Create PositionManager with mocked IBKR
        pos_manager = PositionManager(config, redis_mock)
        
        # Mock IBKR connection
        pos_manager.ib = AsyncMock()
        pos_manager.ib.isConnected = Mock(return_value=True)
        pos_manager.ib.connectAsync = AsyncMock(return_value=True)
        pos_manager.ib.reqMktData = Mock()
        pos_manager.ib.placeOrder = Mock()
        pos_manager.ib.cancelOrder = Mock()
        
        return pos_manager, redis_mock, config
    
    @pytest.mark.asyncio
    async def test_load_positions(self, setup):
        """Test loading positions from Redis."""
        pos_manager, redis_mock, config = await setup
        
        # Mock position data in Redis
        position_data = {
            'id': 'pos_123',
            'symbol': 'SPY',
            'side': 'LONG',
            'quantity': 10,
            'entry_price': 2.55,
            'contract': {'type': 'option', 'occ_symbol': 'SPY240315C00450000'}
        }
        
        redis_mock.keys = AsyncMock(return_value=['positions:open:SPY:pos_123'])
        redis_mock.get = AsyncMock(return_value=json.dumps(position_data))
        
        # Mock market data subscription
        pos_manager.subscribe_market_data = AsyncMock()
        
        # Load positions
        await pos_manager.load_positions()
        
        # Verify position loaded
        assert 'pos_123' in pos_manager.positions
        assert pos_manager.positions['pos_123']['symbol'] == 'SPY'
        
        # Verify market data subscription
        pos_manager.subscribe_market_data.assert_called_once_with('SPY', position_data['contract'])
    
    @pytest.mark.asyncio
    async def test_calculate_unrealized_pnl_options(self, setup):
        """Test P&L calculation for options positions."""
        pos_manager, redis_mock, config = await setup
        
        # Test long call option
        position = {
            'entry_price': 2.50,
            'quantity': 10,
            'side': 'LONG',
            'contract': {'type': 'option'},
            'commission': 15.00
        }
        
        current_price = 3.50
        pnl = await pos_manager.calculate_unrealized_pnl(position, current_price)
        
        # P&L = (3.50 - 2.50) * 10 * 100 - 15 = 985
        assert pnl == 985.00
        
        # Test short put option
        position = {
            'entry_price': 1.80,
            'quantity': 5,
            'side': 'SHORT',
            'contract': {'type': 'option'},
            'commission': 7.50
        }
        
        current_price = 0.90
        pnl = await pos_manager.calculate_unrealized_pnl(position, current_price)
        
        # P&L = (1.80 - 0.90) * 5 * 100 - 7.50 = 442.50
        assert pnl == 442.50
    
    @pytest.mark.asyncio
    async def test_get_option_greeks(self, setup):
        """Test fetching Greeks from Alpha Vantage data in Redis."""
        pos_manager, redis_mock, config = await setup
        
        # Mock options chain data from Alpha Vantage
        chain_data = {
            'SPY240315C00450000': {
                'delta': 0.45,
                'gamma': 0.02,
                'theta': -0.08,
                'vega': 0.15,
                'rho': 0.05,
                'implied_volatility': 0.18
            }
        }
        
        redis_mock.get = AsyncMock(return_value=json.dumps(chain_data))
        
        # Test position
        position = {
            'id': 'pos_123',
            'symbol': 'SPY',
            'quantity': 10,
            'contract': {
                'type': 'option',
                'occ_symbol': 'SPY240315C00450000'
            }
        }
        
        # Get Greeks
        greeks = await pos_manager.get_option_greeks(position)
        
        assert greeks is not None
        assert greeks['delta'] == 0.45 * 10 * 100  # Position delta = 450
        assert greeks['gamma'] == 0.02 * 10 * 100  # Position gamma = 20
        assert greeks['theta'] == -0.08 * 10 * 100  # Daily theta = -80
        assert greeks['vega'] == 0.15 * 10 * 100  # Vega = 150
        assert greeks['iv'] == 0.18
        
        # Verify Greeks stored in Redis
        redis_mock.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_trailing_stop_management(self, setup):
        """Test trailing stop logic for profitable positions."""
        pos_manager, redis_mock, config = await setup
        
        # Mock update_stop_loss
        pos_manager.update_stop_loss = AsyncMock()
        
        # Create profitable 0DTE position
        position = {
            'id': 'pos_123',
            'symbol': 'SPY',
            'side': 'LONG',
            'strategy': '0DTE',
            'entry_price': 2.00,
            'current_price': 3.00,
            'quantity': 10,
            'stop_loss': 1.50,
            'unrealized_pnl': 1000  # $1000 profit
        }
        
        pos_manager.positions = {'pos_123': position}
        
        # Test trailing stop
        await pos_manager.manage_trailing_stops()
        
        # For 0DTE: Trail at 50% profit, keep 50%
        # Profit = $1.00 per contract, lock 50% = $0.50
        # New stop = 2.00 + 0.50 = 2.50
        pos_manager.update_stop_loss.assert_called_once()
        call_args = pos_manager.update_stop_loss.call_args[0]
        assert call_args[1] == 2.50  # New stop price
    
    @pytest.mark.asyncio
    async def test_check_targets(self, setup):
        """Test target checking and scaling logic."""
        pos_manager, redis_mock, config = await setup
        
        # Mock scale_out
        pos_manager.scale_out = AsyncMock()
        
        # Create position with targets
        position = {
            'id': 'pos_123',
            'symbol': 'SPY',
            'side': 'LONG',
            'current_price': 3.00,
            'targets': [2.80, 3.20, 3.50],
            'current_target_index': 0
        }
        
        pos_manager.positions = {'pos_123': position}
        
        # Check targets (first target hit at 3.00)
        await pos_manager.check_targets()
        
        # Verify scale out called
        pos_manager.scale_out.assert_called_once()
        call_args = pos_manager.scale_out.call_args[0]
        assert call_args[1] == 2.80  # Target price
        assert call_args[2] == 0  # Target index
    
    @pytest.mark.asyncio
    async def test_scale_out_options(self, setup):
        """Test scaling out of option positions."""
        pos_manager, redis_mock, config = await setup
        
        # Mock IBKR order execution
        mock_contract = Mock()
        mock_trade = Mock()
        mock_trade.isDone = Mock(return_value=True)
        mock_trade.orderStatus = Mock(status='Filled')
        mock_fill = Mock()
        mock_fill.execution = Mock(price=3.00)
        mock_trade.fills = [mock_fill]
        
        pos_manager.ib.placeOrder = Mock(return_value=mock_trade)
        
        # Mock contract creation
        from unittest.mock import patch
        with patch('src.execution.ExecutionManager') as mock_exec:
            mock_exec_instance = mock_exec.return_value
            mock_exec_instance.create_ib_contract = AsyncMock(return_value=mock_contract)
            
            # Create position with 10 contracts
            position = {
                'id': 'pos_123',
                'symbol': 'SPY',
                'side': 'LONG',
                'quantity': 10,
                'entry_price': 2.00,
                'contract': {'type': 'option'},
                'realized_pnl': 0,
                'current_target_index': 0
            }
            
            # Scale out at first target (33%)
            await pos_manager.scale_out(position, 3.00, 0)
            
            # Should close 3 contracts (33% of 10)
            pos_manager.ib.placeOrder.assert_called_once()
            order_call = pos_manager.ib.placeOrder.call_args[0]
            order = order_call[1]
            assert order.action == 'SELL'
            assert order.totalQuantity == 3
            
            # Verify position updated
            assert position['quantity'] == 7  # 10 - 3
            assert position['current_target_index'] == 1
            # Realized P&L = (3.00 - 2.00) * 3 * 100 = 300
            assert position['realized_pnl'] == 300
    
    @pytest.mark.asyncio
    async def test_close_position(self, setup):
        """Test position closing and cleanup."""
        pos_manager, redis_mock, config = await setup
        
        # Mock stop order cancellation
        mock_stop_trade = Mock()
        mock_stop_trade.isDone = Mock(return_value=False)
        pos_manager.stop_orders = {'pos_123': mock_stop_trade}
        
        # Create position
        position = {
            'id': 'pos_123',
            'symbol': 'SPY',
            'side': 'LONG',
            'quantity': 5,
            'entry_price': 2.50,
            'contract': {'type': 'option'},
            'stop_order_id': 54321,
            'realized_pnl': 100
        }
        
        pos_manager.positions = {'pos_123': position}
        pos_manager.high_water_marks = {'pos_123': 500}
        
        # Close position
        await pos_manager.close_position(position, 3.00, 'Target hit')
        
        # Verify stop cancelled
        pos_manager.ib.cancelOrder.assert_called_once()
        
        # Verify position marked as closed
        assert position['status'] == 'CLOSED'
        assert position['exit_price'] == 3.00
        assert position['exit_reason'] == 'Target hit'
        
        # Final P&L = (3.00 - 2.50) * 5 * 100 + 100 = 350
        assert position['realized_pnl'] == 350
        
        # Verify Redis updates
        redis_mock.delete.assert_called()  # Delete open position
        redis_mock.setex.assert_called()  # Store closed position
        redis_mock.srem.assert_called()  # Remove from symbol index
        redis_mock.incr.assert_called()  # Update metrics
        
        # Verify memory cleanup
        assert 'pos_123' not in pos_manager.positions
        assert 'pos_123' not in pos_manager.high_water_marks
    
    @pytest.mark.asyncio
    async def test_handle_eod_0dte(self, setup):
        """Test end-of-day handling for 0DTE positions."""
        pos_manager, redis_mock, config = await setup
        
        # Mock close_position
        pos_manager.close_position = AsyncMock()
        
        # Create 0DTE position
        position = {
            'id': 'pos_123',
            'symbol': 'SPY',
            'strategy': '0DTE',
            'current_price': 2.75
        }
        
        pos_manager.positions = {'pos_123': position}
        
        # Mock time to be 3:45 PM ET
        with patch('src.execution.datetime') as mock_datetime:
            mock_time = datetime(2024, 3, 15, 15, 45, tzinfo=pytz.timezone('US/Eastern'))
            mock_datetime.now.return_value = mock_time
            
            # Handle EOD
            await pos_manager.handle_eod_positions()
            
            # Verify 0DTE position closed
            pos_manager.close_position.assert_called_once()
            call_args = pos_manager.close_position.call_args[0]
            assert call_args[0] == position
            assert call_args[1] == 2.75  # Current price
            assert 'EOD 0DTE' in call_args[2]  # Reason
    
    @pytest.mark.asyncio
    async def test_position_summary(self, setup):
        """Test position summary generation."""
        pos_manager, redis_mock, config = await setup
        
        # Create multiple positions
        positions = {
            'pos_1': {
                'id': 'pos_1',
                'symbol': 'SPY',
                'side': 'LONG',
                'strategy': '0DTE',
                'quantity': 10,
                'entry_price': 2.50,
                'current_price': 3.00,
                'unrealized_pnl': 500,
                'realized_pnl': 0,
                'contract': {'type': 'option'},
                'greeks': {
                    'delta': 450,
                    'gamma': 20,
                    'theta': -80,
                    'vega': 150
                }
            },
            'pos_2': {
                'id': 'pos_2',
                'symbol': 'QQQ',
                'side': 'SHORT',
                'strategy': '1DTE',
                'quantity': 5,
                'entry_price': 1.80,
                'current_price': 1.50,
                'unrealized_pnl': 150,
                'realized_pnl': 100,
                'contract': {'type': 'option'},
                'greeks': {
                    'delta': -225,
                    'gamma': -10,
                    'theta': 40,
                    'vega': -75
                }
            }
        }
        
        pos_manager.positions = positions
        
        # Get summary
        summary = await pos_manager.get_position_summary()
        
        # Verify summary
        assert summary['total_positions'] == 2
        assert summary['total_unrealized_pnl'] == 650
        assert summary['total_realized_pnl'] == 100
        assert summary['positions_by_strategy']['0DTE'] == 1
        assert summary['positions_by_strategy']['1DTE'] == 1
        
        # Verify portfolio Greeks
        assert summary['portfolio_greeks']['delta'] == 225  # 450 - 225
        assert summary['portfolio_greeks']['gamma'] == 10  # 20 - 10
        assert summary['portfolio_greeks']['theta'] == -40  # -80 + 40
        assert summary['portfolio_greeks']['vega'] == 75  # 150 - 75
        
        # Verify Redis storage
        redis_mock.setex.assert_called_with('positions:summary', 60, json.dumps(summary))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])