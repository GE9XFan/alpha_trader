#!/usr/bin/env python3
"""
Day 6 Test Suite - Signal Generation and Distribution
Tests for signal generation, guardrails, and tiered distribution
"""

import pytest
import asyncio
import json
import time
import hashlib
import redis.asyncio as aioredis
from datetime import datetime, timedelta, time as datetime_time
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals import SignalGenerator, SignalDistributor, SignalValidator, PerformanceTracker


class TestSignalGenerator:
    """Test suite for SignalGenerator class."""
    
    @pytest.fixture
    async def setup(self):
        """Set up test fixtures."""
        # Mock config
        config = {
            'modules': {
                'signals': {
                    'enabled': True,
                    'dry_run': True,
                    'max_staleness_s': 5,
                    'min_confidence': 0.60,
                    'min_refresh_s': 2,
                    'cooldown_s': 30,
                    'ttl_seconds': 300,
                    'version': 'TEST',
                    'strategies': {
                        '0dte': {
                            'enabled': True,
                            'symbols': ['SPY', 'QQQ'],
                            'time_window': {'start': '09:45', 'end': '15:00'},
                            'confidence_weights': {
                                'vpin': 30,
                                'obi': 25,
                                'gamma_proximity': 30,
                                'sweep': 15
                            },
                            'thresholds': {
                                'min_confidence': 60,
                                'vpin_min': 0.40,
                                'obi_min': 0.30,
                                'gamma_pin_distance': 0.005
                            }
                        },
                        '1dte': {
                            'enabled': True,
                            'symbols': ['SPY', 'QQQ', 'VXX'],
                            'time_window': {'start': '14:00', 'end': '15:30'},
                            'confidence_weights': {
                                'volatility_regime': 20,
                                'obi': 30,
                                'gex': 25,
                                'vpin': 25
                            },
                            'thresholds': {
                                'min_confidence': 65
                            }
                        },
                        '14dte': {
                            'enabled': True,
                            'symbols': ['AAPL', 'TSLA'],
                            'time_window': {'start': '09:30', 'end': '16:00'},
                            'confidence_weights': {
                                'unusual_options': 40,
                                'sweep': 30,
                                'hidden_orders': 20,
                                'dex': 10
                            },
                            'thresholds': {
                                'min_confidence': 70
                            }
                        },
                        'moc': {
                            'enabled': True,
                            'symbols': ['SPY', 'QQQ'],
                            'time_window': {'start': '15:30', 'end': '15:50'},
                            'confidence_weights': {
                                'imbalance_strength': 45,
                                'gamma_pull': 25,
                                'obi': 20,
                                'friday_factor': 10
                            },
                            'thresholds': {
                                'min_confidence': 75,
                                'min_imbalance_notional': 2.0e9,
                                'min_imbalance_ratio': 0.60
                            }
                        }
                    }
                }
            }
        }
        
        # Mock Redis
        redis_mock = AsyncMock()
        
        # Create generator
        generator = SignalGenerator(config, redis_mock)
        
        return generator, redis_mock, config
    
    @pytest.mark.asyncio
    async def test_time_window_active(self, setup):
        """Test strategy time window detection."""
        generator, redis_mock, config = await setup
        
        # Test during 0DTE window (9:45 AM - 3:00 PM)
        with patch('src.signals.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30)  # Monday 10:30 AM
            mock_datetime.now.return_value = mock_datetime.now.return_value.replace(
                tzinfo=generator.eastern
            )
            
            assert generator.is_strategy_active('0dte', mock_datetime.now.return_value)
            assert not generator.is_strategy_active('moc', mock_datetime.now.return_value)
    
    @pytest.mark.asyncio
    async def test_0dte_scoring(self, setup):
        """Test 0DTE strategy scoring logic."""
        generator, redis_mock, config = await setup
        
        # Mock features with conditions that should trigger signal
        features = {
            'vpin': 0.45,  # Above 0.40 threshold -> +30 points
            'obi': 0.35,   # Above 0.30 threshold -> +25 points
            'gamma_pin_proximity': 0.8,  # Near pin -> +24 points (30 * 0.8)
            'sweep': 1.0,  # Sweep detected -> +15 points
            'price': 450.0,
            'vwap': 449.0,
            'bars': [
                {'close': 449.5},
                {'close': 450.0}
            ]
        }
        
        confidence, reasons, side = generator.evaluate_0dte_conditions('SPY', features)
        
        # Should get 30 + 25 + 24 + 15 = 94 confidence
        assert confidence == 94
        assert 'VPIN pressure' in reasons
        assert 'OBI imbalance' in reasons
        assert 'Near gamma pin' in reasons
        assert 'Sweep detected' in reasons
        assert side == 'LONG'  # OBI > 0.5 and price >= VWAP
    
    @pytest.mark.asyncio
    async def test_1dte_scoring(self, setup):
        """Test 1DTE strategy scoring logic."""
        generator, redis_mock, config = await setup
        
        features = {
            'regime': 'HIGH',  # +20 points
            'obi': 0.25,       # +15 points (30 * 0.25/0.5)
            'gex_z': 1.0,      # +12 points (25 * 1.0/2)
            'vpin': 0.50,      # +10 points (25 * (0.50-0.35)/0.35)
            'bars': [
                {'close': 449.0},
                {'close': 450.0}
            ]
        }
        
        confidence, reasons, side = generator.evaluate_1dte_conditions('SPY', features)
        
        assert confidence > 0
        assert 'HIGH volatility regime' in reasons
        assert side == 'LONG'  # Positive return
    
    @pytest.mark.asyncio
    async def test_14dte_scoring(self, setup):
        """Test 14DTE strategy scoring logic."""
        generator, redis_mock, config = await setup
        
        features = {
            'unusual_activity': 0.8,  # +32 points
            'sweep': 1.0,            # +30 points
            'hidden_orders': 0.6,    # +15 points
            'dex_z': 1.0,           # +5 points
            'dex': 100000,          # Positive DEX
            'bars': [
                {'close': 149.0},
                {'close': 150.0}
            ]
        }
        
        confidence, reasons, side = generator.evaluate_14dte_conditions('AAPL', features)
        
        assert confidence >= 70  # Should meet threshold
        assert 'Unusual options activity' in reasons
        assert side == 'LONG'  # Majority vote
    
    @pytest.mark.asyncio
    async def test_moc_scoring(self, setup):
        """Test MOC strategy scoring logic."""
        generator, redis_mock, config = await setup
        
        features = {
            'imbalance_total': 3e9,     # $3B imbalance
            'imbalance_ratio': 0.70,    # 70% ratio
            'imbalance_side': 'BUY',
            'gamma_pull_dir': 'UP',
            'gamma_pin_proximity': 0.8,
            'obi': 0.5,
            'price': 450.0
        }
        
        # Mock Friday
        with patch('src.signals.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 19, 15, 40)  # Friday 3:40 PM
            mock_datetime.now.return_value = mock_datetime.now.return_value.replace(
                tzinfo=generator.eastern
            )
            
            confidence, reasons, side = generator.evaluate_moc_conditions('SPY', features)
            
            assert confidence >= 75  # Should meet threshold
            assert side == 'LONG'  # BUY imbalance -> calls
            assert any('imbalance' in r for r in reasons)
    
    @pytest.mark.asyncio
    async def test_freshness_gate(self, setup):
        """Test data freshness guardrail."""
        generator, redis_mock, config = await setup
        
        # Test stale data
        stale_features = {'age_s': 10}  # 10 seconds old
        assert not generator.check_freshness(stale_features)
        
        # Test fresh data
        fresh_features = {'age_s': 2}  # 2 seconds old
        assert generator.check_freshness(fresh_features)
    
    @pytest.mark.asyncio
    async def test_schema_validation(self, setup):
        """Test schema validation guardrail."""
        generator, redis_mock, config = await setup
        
        # Test missing required fields
        bad_features = {'vpin': 0.5}  # Missing obi, price, timestamp
        assert not generator.check_schema(bad_features)
        
        # Test valid schema
        good_features = {
            'vpin': 0.5,
            'obi': 0.3,
            'price': 450.0,
            'timestamp': time.time() * 1000
        }
        assert generator.check_schema(good_features)
        
        # Test clamping
        features_to_clamp = {
            'vpin': 1.5,  # Should be clamped to 1.0
            'obi': -0.2,  # Should be clamped to 0.0
            'price': 450.0,
            'timestamp': time.time() * 1000
        }
        assert generator.check_schema(features_to_clamp)
        assert features_to_clamp['vpin'] == 1.0
        assert features_to_clamp['obi'] == 0.0
    
    @pytest.mark.asyncio
    async def test_cooldown_mechanism(self, setup):
        """Test cooldown guardrail."""
        generator, redis_mock, config = await setup
        
        # Test no cooldown exists
        redis_mock.exists.return_value = False
        assert await generator.check_cooldown('SPY', 'LONG')
        
        # Test cooldown exists
        redis_mock.exists.return_value = True
        assert not await generator.check_cooldown('SPY', 'LONG')
    
    @pytest.mark.asyncio
    async def test_idempotency(self, setup):
        """Test idempotency guardrail."""
        generator, redis_mock, config = await setup
        
        signal_id = 'test_signal_123'
        
        # First attempt - should succeed
        redis_mock.set.return_value = True
        assert await generator.check_idempotency(signal_id)
        
        # Verify SET NX was called correctly
        redis_mock.set.assert_called_with(
            f'signals:emitted:{signal_id}',
            '1',
            nx=True,
            ex=300
        )
        
        # Second attempt - should fail (already exists)
        redis_mock.set.return_value = None
        assert not await generator.check_idempotency(signal_id)
    
    @pytest.mark.asyncio
    async def test_signal_id_generation(self, setup):
        """Test idempotent signal ID generation."""
        generator, redis_mock, config = await setup
        
        # Same inputs within 5-second window should generate same ID
        with patch('time.time', return_value=1000.0):
            id1 = generator.generate_signal_id('SPY', 'LONG', 450.12)
            id2 = generator.generate_signal_id('SPY', 'LONG', 450.12)
            assert id1 == id2
        
        # Different symbol should generate different ID
        with patch('time.time', return_value=1000.0):
            id3 = generator.generate_signal_id('QQQ', 'LONG', 450.12)
            assert id3 != id1
        
        # Different time bucket should generate different ID
        with patch('time.time', return_value=1006.0):  # 6 seconds later
            id4 = generator.generate_signal_id('SPY', 'LONG', 450.12)
            assert id4 != id1
    
    @pytest.mark.asyncio
    async def test_contract_selection(self, setup):
        """Test options contract selection."""
        generator, redis_mock, config = await setup
        
        # Test 0DTE contract selection
        contract = generator.select_contract('SPY', '0dte', 'LONG', 450.0)
        assert contract['type'] == 'OPT'
        assert contract['right'] == 'C'
        assert contract['expiry'] == '0DTE'
        assert contract['strike'] == 451.0  # Next dollar up
        
        # Test 1DTE contract selection
        contract = generator.select_contract('SPY', '1dte', 'SHORT', 450.0)
        assert contract['right'] == 'P'
        assert contract['expiry'] == '1DTE'
        assert contract['strike'] == 446.0  # 1% OTM down
        
        # Test 14DTE contract selection
        contract = generator.select_contract('AAPL', '14dte', 'LONG', 150.0)
        assert contract['expiry'] == '14DTE'
        assert contract['strike'] == 153.0  # 2% OTM up
    
    @pytest.mark.asyncio
    async def test_position_sizing(self, setup):
        """Test position size calculation."""
        generator, redis_mock, config = await setup
        
        # Test 0DTE sizing (5% max)
        size = generator.calculate_position_size(80, '0dte')
        assert size <= 10000 * 0.05  # Max 5% of base
        
        # Test confidence scaling
        low_conf_size = generator.calculate_position_size(60, '14dte')
        high_conf_size = generator.calculate_position_size(100, '14dte')
        assert high_conf_size > low_conf_size
    
    @pytest.mark.asyncio
    async def test_atr_calculation(self, setup):
        """Test ATR calculation."""
        generator, redis_mock, config = await setup
        
        bars = [
            {'high': 451, 'low': 449, 'close': 450},
            {'high': 452, 'low': 450, 'close': 451},
            {'high': 453, 'low': 451, 'close': 452},
        ]
        
        atr = await generator.calculate_atr(bars)
        assert atr >= 0.5  # Minimum ATR
        assert atr <= 10  # Reasonable maximum


class TestSignalDistributor:
    """Test suite for SignalDistributor class."""
    
    @pytest.fixture
    async def setup(self):
        """Set up test fixtures."""
        config = {
            'modules': {
                'signals': {
                    'enabled': True,
                    'distribution': {
                        'tiers': {
                            'premium': {'delay_seconds': 0, 'include_all_details': True},
                            'basic': {'delay_seconds': 60, 'include_all_details': False},
                            'free': {'delay_seconds': 300, 'include_all_details': False}
                        }
                    }
                }
            }
        }
        
        redis_mock = AsyncMock()
        distributor = SignalDistributor(config, redis_mock)
        
        return distributor, redis_mock
    
    @pytest.mark.asyncio
    async def test_premium_formatting(self, setup):
        """Test premium signal formatting."""
        distributor, redis_mock = await setup
        
        signal = {
            'id': 'test-123',
            'symbol': 'SPY',
            'side': 'LONG',
            'confidence': 85,
            'entry': 450.0,
            'stop': 448.0,
            'targets': [452.0, 454.0, 456.0],
            'contract': {'type': 'OPT', 'strike': 451},
            'reasons': ['Test reason']
        }
        
        premium = distributor.format_premium_signal(signal)
        assert premium == signal  # Premium gets everything
    
    @pytest.mark.asyncio
    async def test_basic_formatting(self, setup):
        """Test basic signal formatting."""
        distributor, redis_mock = await setup
        
        signal = {
            'symbol': 'SPY',
            'side': 'LONG',
            'strategy': '0dte',
            'confidence': 85,
            'entry': 450.0,
            'ts': 123456789
        }
        
        basic = distributor.format_basic_signal(signal)
        assert basic['symbol'] == 'SPY'
        assert basic['side'] == 'LONG'
        assert basic['confidence_band'] == 'HIGH'  # 85 >= 80
        assert 'entry' not in basic  # No specific levels
    
    @pytest.mark.asyncio
    async def test_free_formatting(self, setup):
        """Test free signal formatting."""
        distributor, redis_mock = await setup
        
        signal = {
            'symbol': 'SPY',
            'side': 'LONG',
            'ts': 123456789
        }
        
        free = distributor.format_free_signal(signal)
        assert free['symbol'] == 'SPY'
        assert free['sentiment'] == 'bullish'
        assert 'Upgrade' in free['message']
    
    @pytest.mark.asyncio
    async def test_delayed_publish(self, setup):
        """Test delayed publishing mechanism."""
        distributor, redis_mock = await setup
        
        data = {'test': 'data'}
        
        # Start delayed publish
        task = asyncio.create_task(
            distributor.delayed_publish('test:queue', data, 0.1)
        )
        
        # Should not be published immediately
        redis_mock.lpush.assert_not_called()
        
        # Wait for delay
        await asyncio.sleep(0.15)
        await task
        
        # Should now be published
        redis_mock.lpush.assert_called_once_with('test:queue', json.dumps(data))


class TestSignalValidator:
    """Test suite for SignalValidator class."""
    
    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        config = {
            'modules': {
                'signals': {
                    'min_confidence': 0.60
                }
            },
            'market': {
                'extended_hours': False
            }
        }
        
        redis_mock = AsyncMock()
        validator = SignalValidator(config, redis_mock)
        
        return validator, redis_mock
    
    def test_confidence_validation(self, setup):
        """Test confidence threshold validation."""
        validator, redis_mock = setup
        
        # Low confidence should fail
        bad_signal = {'confidence': 50}
        assert not validator.validate_signal(bad_signal)
        
        # High confidence should pass
        good_signal = {'confidence': 75}
        assert validator.validate_signal(good_signal)
    
    def test_stop_distance_validation(self, setup):
        """Test stop loss distance validation."""
        validator, redis_mock = setup
        
        # Stop too far (>5%)
        bad_signal = {
            'confidence': 75,
            'entry': 100.0,
            'stop': 94.0  # 6% away
        }
        assert not validator.validate_signal(bad_signal)
        
        # Stop reasonable (<5%)
        good_signal = {
            'confidence': 75,
            'entry': 100.0,
            'stop': 97.0  # 3% away
        }
        assert validator.validate_signal(good_signal)
    
    def test_risk_reward_validation(self, setup):
        """Test risk/reward ratio validation."""
        validator, redis_mock = setup
        
        # Poor R/R (<1.5)
        bad_signal = {
            'confidence': 75,
            'entry': 100.0,
            'stop': 98.0,
            'targets': [101.0]  # Risk=2, Reward=1, R/R=0.5
        }
        assert not validator.validate_signal(bad_signal)
        
        # Good R/R (>=1.5)
        good_signal = {
            'confidence': 75,
            'entry': 100.0,
            'stop': 98.0,
            'targets': [103.0]  # Risk=2, Reward=3, R/R=1.5
        }
        assert validator.validate_signal(good_signal)


class TestPerformanceTracker:
    """Test suite for PerformanceTracker class."""
    
    @pytest.fixture
    async def setup(self):
        """Set up test fixtures."""
        config = {}
        redis_mock = AsyncMock()
        tracker = PerformanceTracker(config, redis_mock)
        
        return tracker, redis_mock
    
    @pytest.mark.asyncio
    async def test_track_signal_performance(self, setup):
        """Test tracking individual signal performance."""
        tracker, redis_mock = await setup
        
        outcome = {
            'strategy': '0dte',
            'pnl': 150.0,
            'entry': 450.0,
            'exit': 451.5
        }
        
        await tracker.track_signal_performance('signal-123', outcome)
        
        # Should store performance data
        redis_mock.hset.assert_called_once()
        
        # Should increment win counter (positive P&L)
        redis_mock.incr.assert_called_with('performance:strategy:0dte:wins')
    
    @pytest.mark.asyncio
    async def test_calculate_strategy_metrics(self, setup):
        """Test strategy metrics calculation."""
        tracker, redis_mock = await setup
        
        # Mock win/loss data
        redis_mock.get.side_effect = [
            '15',  # wins
            '10'   # losses
        ]
        
        metrics = await tracker.calculate_strategy_metrics('0dte')
        
        assert metrics['wins'] == 15
        assert metrics['losses'] == 10
        assert metrics['total_trades'] == 25
        assert metrics['win_rate'] == 0.6  # 15/25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])