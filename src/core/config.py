"""
Configuration Management - Implementation Plan Week 1 Day 1-2
Single source of truth for all configuration
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml
import os
from pathlib import Path


@dataclass
class IBKRConfig:
    """IBKR configuration for quotes/bars/execution"""
    host: str = '127.0.0.1'
    port: int = 7497  # 7496 for live
    client_id: int = 1
    connection_timeout: int = 30
    heartbeat_interval: int = 10


@dataclass
class AlphaVantageConfig:
    """Alpha Vantage configuration - 38 APIs total!"""
    api_key: str = ''
    tier: str = 'premium'  # 600 calls/minute
    rate_limit: int = 600
    rate_window: int = 60
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    concurrent_requests: int = 10
    cache_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enabled_apis: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.api_key = os.getenv('AV_API_KEY', '')
        if not self.cache_config:
            self.cache_config = {
                'options': {'ttl': 60, 'max_size': 1000},
                'historical_options': {'ttl': 3600, 'max_size': 10000},
                'indicators': {'ttl': 300, 'max_size': 500},
                'sentiment': {'ttl': 900, 'max_size': 100},
                'fundamentals': {'ttl': 86400, 'max_size': 100}
            }


@dataclass
class TradingConfig:
    """Trading configuration - Implementation Plan Week 1"""
    mode: str = 'paper'  # paper/live
    symbols: List[str] = field(default_factory=list)
    max_positions: int = 5
    max_position_size: float = 10000
    daily_loss_limit: float = 1000
    weekly_loss_limit: float = 3000
    monthly_loss_limit: float = 5000
    
    # Greeks limits using Alpha Vantage data
    greeks_limits: Dict[str, tuple] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.symbols:
            self.symbols = ['SPY', 'QQQ', 'IWM']
        if not self.greeks_limits:
            self.greeks_limits = {
                'delta': (-0.3, 0.3),
                'gamma': (-0.5, 0.5),
                'vega': (-500, 500),
                'theta': (-200, float('inf'))
            }


class ConfigManager:
    """
    Central configuration manager - REUSED BY ALL COMPONENTS
    Implementation Plan Week 1: "THIS CONFIG IS REUSED BY EVERY COMPONENT"
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = Path(config_path)
        self._raw_config = self._load_config()
        
        # Parse into specific configs
        self.ibkr = self._parse_ibkr_config()
        self.av = self._parse_av_config()
        self.trading = self._parse_trading_config()
        self.database = self._raw_config.get('database', {})
        self.monitoring = self._raw_config.get('monitoring', {})
        self.community = self._raw_config.get('community', {})
        self.ml = self._raw_config.get('ml', {})
        self.risk = self._raw_config.get('risk', {})
    
    def _load_config(self) -> dict:
        """Load YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _parse_ibkr_config(self) -> IBKRConfig:
        """Parse IBKR configuration"""
        ibkr_data = self._raw_config.get('data_sources', {}).get('ibkr', {})
        return IBKRConfig(**{k: v for k, v in ibkr_data.items() 
                            if k in IBKRConfig.__dataclass_fields__})
    
    def _parse_av_config(self) -> AlphaVantageConfig:
        """Parse Alpha Vantage configuration"""
        av_data = self._raw_config.get('data_sources', {}).get('alpha_vantage', {})
        return AlphaVantageConfig(**{k: v for k, v in av_data.items() 
                                    if k in AlphaVantageConfig.__dataclass_fields__})
    
    def _parse_trading_config(self) -> TradingConfig:
        """Parse trading configuration"""
        trading_data = self._raw_config.get('trading', {})
        risk_data = self._raw_config.get('risk', {})
        
        config = TradingConfig(
            mode=trading_data.get('mode', 'paper'),
            symbols=trading_data.get('symbols', ['SPY', 'QQQ', 'IWM']),
            max_positions=risk_data.get('max_positions', 5),
            max_position_size=risk_data.get('max_position_size', 10000),
            daily_loss_limit=risk_data.get('daily_loss_limit', 1000)
        )
        
        # Parse Greeks limits from risk config
        if 'portfolio_greeks' in risk_data:
            greeks = risk_data['portfolio_greeks']
            config.greeks_limits = {
                'delta': (greeks['delta']['min'], greeks['delta']['max']),
                'gamma': (greeks['gamma']['min'], greeks['gamma']['max']),
                'vega': (greeks['vega']['min'], greeks['vega']['max']),
                'theta': (greeks['theta']['min'], greeks['theta']['max'] or float('inf'))
            }
        
        return config


# Global config instance - reused everywhere
config = ConfigManager()
