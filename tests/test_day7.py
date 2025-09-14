#!/usr/bin/env python3
"""
Day 7 Test Suite - Risk Management
Tests for circuit breakers, correlation checks, drawdown monitoring, and VaR
"""

import pytest
import asyncio
import json
import time
import redis.asyncio as aioredis
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution import RiskManager


class TestRiskManager:
    """Test suite for RiskManager class."""
    
    @pytest.fixture
    async def setup(self):
        """Set up test fixtures."""
        # Mock config
        config = {
            'risk_management': {
                'max_daily_loss_pct': 2.0,
                'max_position_loss_pct': 1.0,
                'consecutive_loss_limit': 3,
                'correlation_limit': 0.7,
                'margin_buffer': 1.25,
                'max_drawdown_pct': 10.0
            }
        }
        
        # Create mock Redis
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.set = AsyncMock(return_value=True)
        redis_mock.setex = AsyncMock(return_value=True)
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.incr = AsyncMock(return_value=1)
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.lpush = AsyncMock(return_value=1)
        redis_mock.ltrim = AsyncMock(return_value=True)
        redis_mock.lrange = AsyncMock(return_value=[])
        redis_mock.publish = AsyncMock(return_value=1)
        
        # Create RiskManager instance
        risk_manager = RiskManager(config, redis_mock)
        
        return risk_manager, redis_mock, config
    
    @pytest.mark.asyncio
    async def test_circuit_breakers_daily_loss(self, setup):
        """Test daily loss circuit breaker."""
        risk_manager, redis_mock, config = await setup
        
        # Setup daily loss exceeding limit
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'risk:daily_pnl': '-2500',  # Loss of $2,500
            'account:value': '100000',   # $100k account
            'risk:consecutive_losses': '0',
            'market:volatility:spike': '1.0',
            'system:errors:count': '0'
        }.get(key))
        
        # Run circuit breaker check
        await risk_manager.check_circuit_breakers()
        
        # Verify halt was triggered
        assert redis_mock.set.call_count >= 2
        halt_calls = [call for call in redis_mock.set.call_args_list 
                     if call[0][0] == 'risk:halt:status']
        assert len(halt_calls) > 0
        assert halt_calls[0][0][1] == 'true'
        
        # Verify circuit breaker was recorded
        assert 'daily_loss' in risk_manager.circuit_breakers_tripped
    
    @pytest.mark.asyncio
    async def test_circuit_breakers_consecutive_losses(self, setup):
        """Test consecutive losses circuit breaker."""
        risk_manager, redis_mock, config = await setup
        
        # Setup consecutive losses at limit
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'risk:daily_pnl': '-500',
            'account:value': '100000',
            'risk:consecutive_losses': '3',  # At limit
            'market:volatility:spike': '1.0',
            'system:errors:count': '0'
        }.get(key))
        
        # Run circuit breaker check
        await risk_manager.check_circuit_breakers()
        
        # Verify halt was triggered
        assert 'consecutive_losses' in risk_manager.circuit_breakers_tripped
    
    @pytest.mark.asyncio
    async def test_circuit_breakers_volatility_spike(self, setup):
        """Test volatility spike circuit breaker."""
        risk_manager, redis_mock, config = await setup
        
        # Setup volatility spike
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'risk:daily_pnl': '0',
            'account:value': '100000',
            'risk:consecutive_losses': '0',
            'market:volatility:spike': '3.5',  # 3.5 sigma event
            'system:errors:count': '0'
        }.get(key))
        
        # Run circuit breaker check
        await risk_manager.check_circuit_breakers()
        
        # Verify halt was triggered
        assert 'volatility' in risk_manager.circuit_breakers_tripped
    
    @pytest.mark.asyncio
    async def test_correlation_check_allows_uncorrelated(self, setup):
        """Test that uncorrelated positions are allowed."""
        risk_manager, redis_mock, config = await setup
        
        # Setup correlation matrix
        correlation_matrix = {
            'AAPL:MSFT': 0.3,  # Low correlation
            'AAPL:TSLA': 0.1,
        }
        
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'discovered:correlation_matrix': json.dumps(correlation_matrix)
        }.get(key))
        
        redis_mock.keys = AsyncMock(return_value=['positions:open:MSFT'])
        
        # Check if AAPL position is allowed
        allowed = await risk_manager.check_correlations('AAPL', 'LONG')
        
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_correlation_check_blocks_high_correlation(self, setup):
        """Test that highly correlated positions are blocked."""
        risk_manager, redis_mock, config = await setup
        
        # Setup correlation matrix with high correlation
        correlation_matrix = {
            'AAPL:MSFT': 0.85,  # High correlation
        }
        
        # Mock existing position
        existing_position = {
            'symbol': 'MSFT',
            'side': 'LONG'
        }
        
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'discovered:correlation_matrix': json.dumps(correlation_matrix),
            'positions:open:MSFT': json.dumps(existing_position)
        }.get(key))
        
        redis_mock.keys = AsyncMock(return_value=['positions:open:MSFT'])
        
        # Check if AAPL LONG position is allowed (same direction, high correlation)
        allowed = await risk_manager.check_correlations('AAPL', 'LONG')
        
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_drawdown_monitoring(self, setup):
        """Test drawdown monitoring and HWM updates."""
        risk_manager, redis_mock, config = await setup
        
        # Test new high water mark
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'account:value': '110000',  # Current value
            'risk:high_water_mark': '100000'  # Previous HWM
        }.get(key))
        
        await risk_manager.monitor_drawdown()
        
        # Verify HWM was updated
        hwm_calls = [call for call in redis_mock.set.call_args_list 
                    if call[0][0] == 'risk:high_water_mark']
        assert len(hwm_calls) > 0
        assert float(hwm_calls[0][0][1]) == 110000
        
        # Test drawdown calculation
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'account:value': '95000',  # 5% drawdown
            'risk:high_water_mark': '100000'
        }.get(key))
        
        await risk_manager.monitor_drawdown()
        
        # Verify drawdown was stored
        setex_calls = redis_mock.setex.call_args_list
        drawdown_calls = [call for call in setex_calls 
                         if 'drawdown' in call[0][0]]
        assert len(drawdown_calls) > 0
    
    @pytest.mark.asyncio
    async def test_drawdown_triggers_halt(self, setup):
        """Test that max drawdown triggers halt."""
        risk_manager, redis_mock, config = await setup
        
        # Setup max drawdown exceeded
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'account:value': '89000',  # 11% drawdown
            'risk:high_water_mark': '100000'
        }.get(key))
        
        await risk_manager.monitor_drawdown()
        
        # Verify halt was triggered
        assert 'max_drawdown' in risk_manager.circuit_breakers_tripped
    
    @pytest.mark.asyncio
    async def test_daily_loss_limits(self, setup):
        """Test daily loss limit enforcement."""
        risk_manager, redis_mock, config = await setup
        
        # Test approaching limit (75% of max)
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'risk:daily_pnl': '-1500',  # $1,500 loss (75% of 2% limit)
            'account:value': '100000'
        }.get(key))
        
        await risk_manager.check_daily_limits()
        
        # Verify new positions blocked
        set_calls = redis_mock.set.call_args_list
        position_calls = [call for call in set_calls 
                         if call[0][0] == 'risk:new_positions_allowed']
        assert len(position_calls) > 0
        assert position_calls[0][0][1] == 'false'
    
    @pytest.mark.asyncio
    async def test_halt_mechanism(self, setup):
        """Test halt trading mechanism."""
        risk_manager, redis_mock, config = await setup
        
        # Mock pending orders
        redis_mock.keys = AsyncMock(return_value=[
            'orders:pending:order1',
            'orders:pending:order2'
        ])
        
        # Trigger halt
        await risk_manager.halt_trading("Test halt reason")
        
        # Verify halt flags were set
        set_calls = redis_mock.set.call_args_list
        halt_status = [call for call in set_calls 
                      if call[0][0] == 'risk:halt:status']
        assert len(halt_status) > 0
        assert halt_status[0][0][1] == 'true'
        
        # Verify orders were cancelled
        assert redis_mock.delete.call_count >= 2
        
        # Verify alert was published
        assert redis_mock.publish.called
    
    @pytest.mark.asyncio
    async def test_var_calculation_historical(self, setup):
        """Test VaR calculation with historical data."""
        risk_manager, redis_mock, config = await setup
        
        # Mock historical P&L data
        pnl_history = [
            json.dumps({'pnl': 100}),
            json.dumps({'pnl': -50}),
            json.dumps({'pnl': 200}),
            json.dumps({'pnl': -150}),
            json.dumps({'pnl': 75}),
            json.dumps({'pnl': -100}),
            json.dumps({'pnl': 50}),
            json.dumps({'pnl': -75}),
            json.dumps({'pnl': 150}),
            json.dumps({'pnl': -200}),
        ] * 5  # 50 data points
        
        redis_mock.lrange = AsyncMock(return_value=pnl_history)
        redis_mock.get = AsyncMock(return_value='100000')
        
        # Calculate VaR
        var_95 = await risk_manager.calculate_var(0.95)
        
        # Verify VaR is reasonable
        assert var_95 > 0
        assert var_95 < 10000  # Should be less than 10% of account
        
        # Verify metrics were stored
        assert redis_mock.setex.called
    
    @pytest.mark.asyncio
    async def test_var_fallback_position_based(self, setup):
        """Test position-based VaR fallback."""
        risk_manager, redis_mock, config = await setup
        
        # No historical data
        redis_mock.lrange = AsyncMock(return_value=[])
        
        # Mock positions
        redis_mock.keys = AsyncMock(return_value=[
            'positions:open:AAPL',
            'positions:open:MSFT'
        ])
        
        position_data = {
            'symbol': 'AAPL',
            'market_value': 10000
        }
        
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'account:value': '100000',
            'positions:open:AAPL': json.dumps(position_data),
            'positions:open:MSFT': json.dumps({'symbol': 'MSFT', 'market_value': 5000}),
            'analytics:AAPL:volatility': json.dumps({'daily_vol': 0.02}),
            'analytics:MSFT:volatility': json.dumps({'daily_vol': 0.025})
        }.get(key))
        
        # Calculate VaR
        var_95 = await risk_manager.calculate_var()
        
        # Verify fallback VaR is calculated
        assert var_95 > 0
        assert var_95 < 5000  # Should be reasonable for positions
    
    @pytest.mark.asyncio
    async def test_risk_metrics_update(self, setup):
        """Test comprehensive risk metrics update."""
        risk_manager, redis_mock, config = await setup
        
        # Mock positions
        redis_mock.keys = AsyncMock(return_value=[
            'positions:open:AAPL',
            'positions:open:MSFT'
        ])
        
        position_aapl = {
            'symbol': 'AAPL',
            'market_value': 50000,
            'type': 'option',
            'delta': 0.5,
            'gamma': 0.02,
            'theta': -0.05,
            'vega': 0.1,
            'quantity': 10
        }
        
        position_msft = {
            'symbol': 'MSFT',
            'market_value': 30000,
            'type': 'stock'
        }
        
        redis_mock.get = AsyncMock(side_effect=lambda key: {
            'positions:open:AAPL': json.dumps(position_aapl),
            'positions:open:MSFT': json.dumps(position_msft),
            'account:value': '100000',
            'risk:var:portfolio': '2000',
            'risk:current_drawdown': json.dumps({'drawdown_pct': 3.5})
        }.get(key))
        
        # Update metrics
        await risk_manager.update_risk_metrics()
        
        # Verify metrics were stored
        setex_calls = redis_mock.setex.call_args_list
        metrics_calls = [call for call in setex_calls 
                        if 'metrics' in call[0][0]]
        assert len(metrics_calls) > 0
        
        # Verify Greeks were calculated
        summary_calls = [call for call in setex_calls 
                        if call[0][0] == 'risk:metrics:summary']
        if summary_calls:
            # setex arguments are: key, ttl, value - so value is at index 2
            metrics = json.loads(summary_calls[0][0][2])
            assert 'greeks' in metrics
            assert metrics['greeks']['delta'] == 500  # 0.5 * 10 * 100
    
    @pytest.mark.asyncio
    async def test_reset_daily_metrics(self, setup):
        """Test daily metrics reset at market open."""
        risk_manager, redis_mock, config = await setup
        
        # Mock yesterday's profitable day
        redis_mock.get = AsyncMock(return_value='1000')  # Profit
        
        # Reset metrics
        await risk_manager.reset_daily_metrics()
        
        # Verify daily P&L was reset
        set_calls = redis_mock.set.call_args_list
        pnl_reset = [call for call in set_calls 
                    if call[0][0] == 'risk:daily_pnl']
        assert len(pnl_reset) > 0
        assert pnl_reset[0][0][1] == 0
        
        # Verify consecutive losses were reset (profitable day)
        consecutive_reset = [call for call in set_calls 
                            if call[0][0] == 'risk:consecutive_losses']
        assert len(consecutive_reset) > 0
        assert consecutive_reset[0][0][1] == 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])