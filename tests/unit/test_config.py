"""
Comprehensive unit tests for configuration management.

Tests all aspects of configuration loading, validation, and edge cases.
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import (
    TradingConfig,
    ConfigManager,
    TradingMode,
    GreeksLimits,
    RiskLimits,
    IBKRConfig,
    AlphaVantageConfig,
    CommunityConfig,
)
from src.core.exceptions import ConfigurationError


class TestGreeksLimits:
    """Test Greeks limits configuration."""
    
    def test_valid_greeks_limits(self):
        """Test creating valid Greeks limits."""
        greeks = GreeksLimits(
            delta_min=-0.3,
            delta_max=0.3,
            gamma_min=-0.75,
            gamma_max=0.75,
            vega_min=-1000,
            vega_max=1000,
            theta_min=-500,
        )
        assert greeks.delta_min == -0.3
        assert greeks.delta_max == 0.3
        assert greeks.gamma_min == -0.75
        assert greeks.gamma_max == 0.75
    
    def test_invalid_delta_range(self):
        """Test invalid delta range raises error."""
        with pytest.raises(ConfigurationError, match="Invalid delta range"):
            GreeksLimits(delta_min=0.3, delta_max=-0.3)
    
    def test_invalid_gamma_range(self):
        """Test invalid gamma range raises error."""
        with pytest.raises(ConfigurationError, match="Invalid gamma range"):
            GreeksLimits(gamma_min=0.75, gamma_max=-0.75)
    
    def test_invalid_vega_range(self):
        """Test invalid vega range raises error."""
        with pytest.raises(ConfigurationError, match="Invalid vega range"):
            GreeksLimits(vega_min=1000, vega_max=-1000)
    
    def test_positive_theta_min(self):
        """Test positive theta minimum raises error."""
        with pytest.raises(ConfigurationError, match="Theta minimum must be negative"):
            GreeksLimits(theta_min=500)
    
    def test_immutability(self):
        """Test that Greeks limits are immutable."""
        greeks = GreeksLimits()
        with pytest.raises(AttributeError):
            greeks.delta_min = -0.5  # type: ignore


class TestRiskLimits:
    """Test risk limits configuration."""
    
    def test_valid_risk_limits(self):
        """Test creating valid risk limits."""
        risk = RiskLimits(
            max_positions=20,
            max_position_size=50000,
            daily_loss_limit=10000,
            vpin_threshold=0.7,
        )
        assert risk.max_positions == 20
        assert risk.max_position_size == 50000
        assert risk.daily_loss_limit == 10000
        assert risk.vpin_threshold == 0.7
    
    def test_invalid_max_positions(self):
        """Test invalid max positions."""
        with pytest.raises(ConfigurationError, match="Invalid max_positions"):
            RiskLimits(max_positions=0)
        
        with pytest.raises(ConfigurationError, match="Invalid max_positions"):
            RiskLimits(max_positions=101)
    
    def test_invalid_position_size(self):
        """Test invalid position size."""
        with pytest.raises(ConfigurationError, match="Invalid max_position_size"):
            RiskLimits(max_position_size=-1000)
    
    def test_invalid_loss_limit(self):
        """Test invalid daily loss limit."""
        with pytest.raises(ConfigurationError, match="Invalid daily_loss_limit"):
            RiskLimits(daily_loss_limit=0)
    
    def test_invalid_vpin_threshold(self):
        """Test invalid VPIN threshold."""
        with pytest.raises(ConfigurationError, match="VPIN threshold must be between"):
            RiskLimits(vpin_threshold=1.5)
        
        with pytest.raises(ConfigurationError, match="VPIN threshold must be between"):
            RiskLimits(vpin_threshold=0)


class TestIBKRConfig:
    """Test IBKR configuration."""
    
    def test_valid_ibkr_config(self):
        """Test creating valid IBKR config."""
        ibkr = IBKRConfig(
            account="DU123456",
            host="127.0.0.1",
            live_port=7496,
            paper_port=7497,
        )
        assert ibkr.account == "DU123456"
        assert ibkr.live_port == 7496
        assert ibkr.paper_port == 7497
    
    def test_missing_account(self):
        """Test missing account raises error."""
        with pytest.raises(ConfigurationError, match="IBKR account ID is required"):
            IBKRConfig(account="")
    
    def test_same_ports(self):
        """Test same live and paper ports raises error."""
        with pytest.raises(ConfigurationError, match="Live and paper ports must be different"):
            IBKRConfig(account="DU123456", live_port=7496, paper_port=7496)
    
    def test_invalid_heartbeat(self):
        """Test invalid heartbeat interval."""
        with pytest.raises(ConfigurationError, match="Invalid heartbeat interval"):
            IBKRConfig(account="DU123456", heartbeat_interval=0)


class TestAlphaVantageConfig:
    """Test Alpha Vantage configuration."""
    
    def test_valid_av_config(self):
        """Test creating valid Alpha Vantage config."""
        av = AlphaVantageConfig(
            api_key="test_key_123",
            rate_limit=500,
        )
        assert av.api_key == "test_key_123"
        assert av.rate_limit == 500
    
    def test_missing_api_key(self):
        """Test missing API key raises error."""
        with pytest.raises(ConfigurationError, match="Alpha Vantage API key is required"):
            AlphaVantageConfig(api_key="")
    
    def test_invalid_rate_limit(self):
        """Test invalid rate limit."""
        with pytest.raises(ConfigurationError, match="Invalid rate limit"):
            AlphaVantageConfig(api_key="test", rate_limit=600)
        
        with pytest.raises(ConfigurationError, match="Invalid rate limit"):
            AlphaVantageConfig(api_key="test", rate_limit=0)


class TestCommunityConfig:
    """Test community configuration."""
    
    def test_community_disabled(self):
        """Test community features disabled by default."""
        community = CommunityConfig()
        assert not community.is_enabled
    
    def test_community_enabled_discord(self):
        """Test community enabled with Discord."""
        community = CommunityConfig(discord_token="test_token")
        assert community.is_enabled
    
    def test_community_enabled_whop(self):
        """Test community enabled with Whop."""
        community = CommunityConfig(whop_api_key="test_key")
        assert community.is_enabled
    
    def test_broadcast_delays(self):
        """Test broadcast delay configuration."""
        community = CommunityConfig()
        assert community.broadcast_delay_free == 300
        assert community.broadcast_delay_premium == 30
        assert community.broadcast_delay_vip == 0


class TestTradingConfig:
    """Test main trading configuration."""
    
    def test_valid_config(self):
        """Test creating valid trading config."""
        config = TradingConfig(
            mode=TradingMode.DEVELOPMENT,
            environment="development",
            ibkr=IBKRConfig(account="DU123456"),
            alpha_vantage=AlphaVantageConfig(api_key="test_key"),
            greeks_limits=GreeksLimits(),
            risk_limits=RiskLimits(),
            community=CommunityConfig(),
        )
        assert config.mode == TradingMode.DEVELOPMENT
        assert len(config.symbols) > 0
        assert "SPY" in config.symbols
    
    def test_mode_environment_mismatch(self):
        """Test mode and environment mismatch."""
        with pytest.raises(ConfigurationError, match="Mode .* inconsistent with environment"):
            TradingConfig(
                mode=TradingMode.PRODUCTION,
                environment="development",
                ibkr=IBKRConfig(account="DU123456"),
                alpha_vantage=AlphaVantageConfig(api_key="test_key"),
                greeks_limits=GreeksLimits(),
                risk_limits=RiskLimits(),
                community=CommunityConfig(),
            )
    
    def test_empty_symbols(self):
        """Test empty symbols list raises error."""
        with pytest.raises(ConfigurationError, match="At least one trading symbol is required"):
            TradingConfig(
                mode=TradingMode.DEVELOPMENT,
                environment="development",
                ibkr=IBKRConfig(account="DU123456"),
                alpha_vantage=AlphaVantageConfig(api_key="test_key"),
                greeks_limits=GreeksLimits(),
                risk_limits=RiskLimits(),
                community=CommunityConfig(),
                symbols=[],
            )
    
    def test_to_dict(self):
        """Test configuration serialization."""
        config = TradingConfig(
            mode=TradingMode.DEVELOPMENT,
            environment="development",
            ibkr=IBKRConfig(account="DU123456"),
            alpha_vantage=AlphaVantageConfig(api_key="test_key"),
            greeks_limits=GreeksLimits(),
            risk_limits=RiskLimits(),
            community=CommunityConfig(),
        )
        
        data = config.to_dict()
        assert data["mode"] == "development"
        assert data["environment"] == "development"
        assert data["ibkr"]["account"] == "DU123456"
        assert data["alpha_vantage"]["api_key"] == "test_key"
    
    def test_validate_for_production(self):
        """Test production validation."""
        # Test with demo account
        config = TradingConfig(
            mode=TradingMode.PRODUCTION,
            environment="production",
            ibkr=IBKRConfig(account="DUdemo123"),
            alpha_vantage=AlphaVantageConfig(api_key="test_key"),
            greeks_limits=GreeksLimits(),
            risk_limits=RiskLimits(),
            community=CommunityConfig(discord_token="token"),
        )
        
        issues = config.validate_for_production()
        assert any("demo account" in issue for issue in issues)
        
        # Test with localhost database
        config.database_url = "postgresql://localhost/alphatrader"
        issues = config.validate_for_production()
        assert any("localhost database" in issue for issue in issues)
        
        # Test with high loss limit
        config = TradingConfig(
            mode=TradingMode.PRODUCTION,
            environment="production",
            ibkr=IBKRConfig(account="DU123456"),
            alpha_vantage=AlphaVantageConfig(api_key="test_key"),
            greeks_limits=GreeksLimits(),
            risk_limits=RiskLimits(daily_loss_limit=100000),
            community=CommunityConfig(),
        )
        issues = config.validate_for_production()
        assert any("Daily loss limit unusually high" in issue for issue in issues)


class TestConfigManager:
    """Test configuration manager."""
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "IBKR_ACCOUNT": "DU123456",
            "ALPHA_VANTAGE_KEY": "test_key_123",
            "TRADING_MODE": "development",
            "ENVIRONMENT": "development",
            "MAX_POSITIONS": "15",
            "MAX_POSITION_SIZE": "30000",
            "DAILY_LOSS_LIMIT": "5000",
            "VPIN_THRESHOLD": "0.6",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            manager = ConfigManager()
            config = manager.load_from_env()
            
            assert config.ibkr.account == "DU123456"
            assert config.alpha_vantage.api_key == "test_key_123"
            assert config.mode == TradingMode.DEVELOPMENT
            assert config.risk_limits.max_positions == 15
            assert config.risk_limits.max_position_size == 30000
            assert config.risk_limits.daily_loss_limit == 5000
            assert config.risk_limits.vpin_threshold == 0.6
    
    def test_load_from_env_missing_required(self):
        """Test loading fails with missing required variables."""
        env_vars = {
            "TRADING_MODE": "development",
            # Missing IBKR_ACCOUNT and ALPHA_VANTAGE_KEY
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            manager = ConfigManager()
            with pytest.raises(ConfigurationError, match="Required environment variable not set"):
                manager.load_from_env()
    
    def test_load_from_env_with_symbols(self):
        """Test loading symbols from environment."""
        env_vars = {
            "IBKR_ACCOUNT": "DU123456",
            "ALPHA_VANTAGE_KEY": "test_key",
            "TRADING_SYMBOLS": "AAPL,MSFT,GOOGL,TSLA",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            manager = ConfigManager()
            config = manager.load_from_env()
            
            assert config.symbols == ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = {
            "mode": "development",
            "environment": "development",
            "ibkr": {
                "account": "DU123456",
                "host": "127.0.0.1",
                "live_port": 7496,
                "paper_port": 7497,
            },
            "alpha_vantage": {
                "api_key": "test_key",
                "rate_limit": 400,
            },
            "greeks_limits": {
                "delta_min": -0.3,
                "delta_max": 0.3,
                "gamma_min": -0.75,
                "gamma_max": 0.75,
                "vega_min": -1000,
                "vega_max": 1000,
                "theta_min": -500,
            },
            "risk_limits": {
                "max_positions": 15,
                "max_position_size": 30000,
                "daily_loss_limit": 5000,
                "vpin_threshold": 0.6,
            },
            "community": {
                "discord_token": "test_token",
            },
            "symbols": ["SPY", "QQQ", "AAPL"],
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = Path(f.name)
        
        try:
            manager = ConfigManager()
            config = manager.load_from_yaml(temp_file)
            
            assert config.mode == TradingMode.DEVELOPMENT
            assert config.ibkr.account == "DU123456"
            assert config.alpha_vantage.rate_limit == 400
            assert config.risk_limits.max_positions == 15
            assert config.symbols == ["SPY", "QQQ", "AAPL"]
            assert config.community.discord_token == "test_token"
        finally:
            temp_file.unlink()
    
    def test_load_from_yaml_missing_file(self):
        """Test loading from non-existent YAML file."""
        manager = ConfigManager()
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            manager.load_from_yaml(Path("nonexistent.yaml"))
    
    def test_save_to_yaml(self):
        """Test saving configuration to YAML file."""
        config = TradingConfig(
            mode=TradingMode.DEVELOPMENT,
            environment="development",
            ibkr=IBKRConfig(account="DU123456"),
            alpha_vantage=AlphaVantageConfig(api_key="test_key"),
            greeks_limits=GreeksLimits(),
            risk_limits=RiskLimits(),
            community=CommunityConfig(),
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            manager = ConfigManager()
            manager.save_to_yaml(config, temp_file)
            
            # Load it back
            with open(temp_file, 'r') as f:
                data = yaml.safe_load(f)
            
            assert data["mode"] == "development"
            assert data["ibkr"]["account"] == "DU123456"
            assert data["alpha_vantage"]["api_key"] == "test_key"
        finally:
            temp_file.unlink()
    
    def test_get_config_singleton(self):
        """Test config manager maintains singleton."""
        env_vars = {
            "IBKR_ACCOUNT": "DU123456",
            "ALPHA_VANTAGE_KEY": "test_key",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            manager = ConfigManager()
            config1 = manager.get_config()
            config2 = manager.get_config()
            
            assert config1 is config2  # Same instance


class TestConfigPerformance:
    """Test configuration performance requirements."""
    
    def test_config_load_time(self):
        """Test configuration loads within 100ms."""
        import time
        
        env_vars = {
            "IBKR_ACCOUNT": "DU123456",
            "ALPHA_VANTAGE_KEY": "test_key",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            manager = ConfigManager()
            
            start = time.perf_counter()
            config = manager.load_from_env()
            elapsed = (time.perf_counter() - start) * 1000
            
            assert elapsed < 100  # Must load in under 100ms
            assert config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])