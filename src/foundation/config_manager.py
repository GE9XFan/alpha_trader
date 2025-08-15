#!/usr/bin/env python3
"""
Configuration Management System
Handles all configuration loading and access
NO HARDCODED VALUES - Everything from config files
"""

import os
import re
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from dotenv import load_dotenv
import logging
from datetime import datetime
from functools import reduce

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Central configuration management - NO HARDCODED VALUES
    All configuration loaded from YAML files and environment variables
    """
    
    def __init__(self, environment: str = None, config_dir: str = 'config'):
        """
        Initialize configuration manager
        
        Args:
            environment: Environment name (development, paper, production)
                        If None, reads from ENVIRONMENT env var or defaults to 'development'
            config_dir: Path to configuration directory
        """
        # Load environment variables first
        load_dotenv()
        
        # Determine environment
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path(config_dir)
        
        # Validate environment
        valid_environments = ['development', 'paper', 'production']
        if self.environment not in valid_environments:
            raise ValueError(f"Invalid environment: {self.environment}. Must be one of {valid_environments}")
        
        # Initialize configuration storage
        self.config = {}
        self.env_vars = {}
        
        # Load all configurations
        self._load_env_variables()
        self._load_yaml_configs()
        self._apply_environment_overrides()
        
        # Validate configuration
        self._validate_required_keys()
        
        logger.info(f"ConfigManager initialized for environment: {self.environment}")
        logger.info(f"Loaded {len(self.config)} configuration sections")
    
    def _load_env_variables(self) -> None:
        """Load environment variables from .env file and system"""
        self.env_vars = {
            # Database
            'DB_HOST': os.getenv('DB_HOST', 'localhost'),
            'DB_PORT': os.getenv('DB_PORT', '5432'),
            'DB_NAME': os.getenv('DB_NAME', 'trading_system'),
            'DB_USER': os.getenv('DB_USER', os.getenv('USER', 'postgres')),
            'DB_PASSWORD': os.getenv('DB_PASSWORD', ''),
            
            # Redis
            'REDIS_HOST': os.getenv('REDIS_HOST', 'localhost'),
            'REDIS_PORT': os.getenv('REDIS_PORT', '6379'),
            'REDIS_PASSWORD': os.getenv('REDIS_PASSWORD', ''),
            
            # API Keys
            'AV_API_KEY': os.getenv('AV_API_KEY', ''),
            'IBKR_USERNAME': os.getenv('IBKR_USERNAME', ''),
            'IBKR_PASSWORD': os.getenv('IBKR_PASSWORD', ''),
            'IBKR_ACCOUNT': os.getenv('IBKR_ACCOUNT', ''),
            
            # Discord
            'DISCORD_WEBHOOK_URL': os.getenv('DISCORD_WEBHOOK_URL', ''),
            
            # Environment
            'ENVIRONMENT': self.environment,
        }
        
        # Store in config for easy access
        self.config['env'] = self.env_vars
    
    def _load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """Load a single YAML file and process environment variables"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Replace environment variables in format ${VAR_NAME}
                def replace_env_var(match):
                    var_name = match.group(1)
                    value = self.env_vars.get(var_name, '')
                    if not value and var_name in ['AV_API_KEY', 'IBKR_USERNAME']:
                        logger.warning(f"Environment variable ${{{var_name}}} not set in {filepath}")
                    return value
                
                content = re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
                
                # Parse YAML
                data = yaml.safe_load(content)
                return data if data else {}
                
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}
    
    def _load_yaml_configs(self) -> None:
        """Load all YAML configuration files from config directories"""
        
        # Define configuration categories and their directories
        config_categories = {
            'system': 'system',
            'apis': 'apis',
            'data': 'data',
            'strategies': 'strategies',
            'risk': 'risk',
            'ml': 'ml',
            'execution': 'execution',
            'monitoring': 'monitoring'
        }
        
        for category, directory in config_categories.items():
            category_path = self.config_dir / directory
            self.config[category] = {}
            
            if category_path.exists():
                # Load all YAML files in the directory
                for yaml_file in category_path.glob('*.yaml'):
                    # Use filename (without extension) as key
                    config_key = yaml_file.stem
                    config_data = self._load_yaml_file(yaml_file)
                    
                    # Store the configuration
                    if config_data:
                        self.config[category][config_key] = config_data
                        logger.debug(f"Loaded {category}/{config_key}")
            else:
                logger.warning(f"Configuration directory not found: {category_path}")
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides"""
        env_file = self.config_dir / 'environments' / f'{self.environment}.yaml'
        
        if env_file.exists():
            overrides = self._load_yaml_file(env_file)
            
            if overrides and 'environment' in overrides:
                env_config = overrides['environment']
                
                # Apply overrides if present
                if 'overrides' in env_config:
                    self._deep_merge(self.config, env_config['overrides'])
                    logger.info(f"Applied {self.environment} environment overrides")
                
                # Store environment metadata
                self.config['environment_meta'] = {
                    'name': env_config.get('name', self.environment),
                    'debug': env_config.get('debug', False),
                    'testing': env_config.get('testing', True)
                }
        else:
            logger.warning(f"No environment override file found: {env_file}")
    
    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """
        Deep merge override dictionary into base dictionary
        Modifies base in place
        """
        for key, value in override.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    self._deep_merge(base[key], value)
                else:
                    base[key] = value
            else:
                base[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation path
        
        Args:
            path: Dot notation path (e.g., 'apis.alpha_vantage.api_key')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        
        Examples:
            config.get('apis.alpha_vantage.api_key')
            config.get('strategies.0dte.confidence.minimum', 0.75)
            config.get('risk.position_limits.max_delta')
        """
        try:
            keys = path.split('.')
            value = reduce(lambda d, key: d.get(key, {}), keys, self.config)
            
            # If we got an empty dict and it wasn't explicitly set, return default
            if value == {} and path not in str(self.config):
                return default
                
            return value if value != {} else default
            
        except (KeyError, TypeError, AttributeError):
            return default
    
    def set(self, path: str, value: Any) -> None:
        """
        Set configuration value by dot notation path
        
        Args:
            path: Dot notation path
            value: Value to set
        """
        keys = path.split('.')
        target = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # Set the value
        target[keys[-1]] = value
        logger.debug(f"Set config {path} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'apis', 'strategies', 'risk')
            
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy
        
        Args:
            strategy_name: Strategy name (e.g., '0dte', '1dte', 'swing_14d')
            
        Returns:
            Strategy configuration
        """
        strategy_map = {
            '0dte': '0dte',
            '1dte': '1dte',
            'swing': 'swing_14d',
            'swing_14d': 'swing_14d',
            'moc': 'moc_imbalance',
            'moc_imbalance': 'moc_imbalance'
        }
        
        config_key = strategy_map.get(strategy_name, strategy_name)
        return self.get(f'strategies.{config_key}', {})
    
    def validate_required_keys(self) -> bool:
        """
        Validate all required configuration keys are present
        
        Returns:
            True if all required keys present, raises exception otherwise
        """
        required_keys = [
            # Critical API keys
            'env.AV_API_KEY',
            'env.IBKR_USERNAME',
            
            # Database configuration
            'env.DB_HOST',
            'env.DB_NAME',
            'env.DB_USER',
            
            # Core system configs
            'apis.alpha_vantage',
            'apis.ibkr',
            'apis.rate_limits',
            
            # Strategy configs
            'strategies.0dte',
            'strategies.1dte',
            'strategies.swing_14d',
            'strategies.moc_imbalance',
            
            # Risk configs
            'risk.position_limits',
            'risk.portfolio_limits',
            'risk.circuit_breakers'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def _validate_required_keys(self) -> None:
        """Internal validation called during initialization"""
        try:
            self.validate_required_keys()
            logger.info("All required configuration keys validated")
        except ValueError as e:
            logger.warning(f"Configuration validation warning: {e}")
            # Don't fail initialization, just warn
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == 'production'
    
    def is_paper(self) -> bool:
        """Check if running in paper trading environment"""
        return self.environment == 'paper'
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == 'development'
    
    def get_trading_mode(self) -> str:
        """Get current trading mode (paper or live)"""
        ibkr_mode = self.get('apis.ibkr.ibkr.trading_mode', 'paper')
        
        # Safety check - never allow live trading in non-production
        if ibkr_mode == 'live' and not self.is_production():
            logger.warning("Live trading requested in non-production environment - forcing paper mode")
            return 'paper'
            
        return ibkr_mode
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'host': self.env_vars.get('DB_HOST'),
            'port': int(self.env_vars.get('DB_PORT', 5432)),
            'database': self.env_vars.get('DB_NAME'),
            'user': self.env_vars.get('DB_USER'),
            'password': self.env_vars.get('DB_PASSWORD'),
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            'host': self.env_vars.get('REDIS_HOST'),
            'port': int(self.env_vars.get('REDIS_PORT', 6379)),
            'password': self.env_vars.get('REDIS_PASSWORD'),
            'decode_responses': True,
        }
    
    def export_config(self, filepath: str = None) -> str:
        """
        Export current configuration to JSON file
        
        Args:
            filepath: Output file path (defaults to config_export_<timestamp>.json)
            
        Returns:
            Path to exported file
        """
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'config_export_{self.environment}_{timestamp}.json'
        
        # Remove sensitive data before export
        export_config = self.config.copy()
        if 'env' in export_config:
            export_config['env'] = {
                k: '***' if 'PASSWORD' in k or 'KEY' in k else v
                for k, v in export_config['env'].items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_config, f, indent=2, default=str)
        
        logger.info(f"Configuration exported to {filepath}")
        return filepath
    
    def print_summary(self) -> None:
        """Print configuration summary"""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Environment: {self.environment}")
        print(f"Trading Mode: {self.get_trading_mode()}")
        print(f"Config Directory: {self.config_dir}")
        print(f"\nLoaded Sections:")
        
        for section in self.config:
            if section != 'env':  # Don't show env vars in summary
                if isinstance(self.config[section], dict):
                    print(f"  • {section}: {len(self.config[section])} items")
                else:
                    print(f"  • {section}")
        
        print(f"\nDatabase: {self.env_vars.get('DB_NAME')} @ {self.env_vars.get('DB_HOST')}")
        print(f"Redis: {self.env_vars.get('REDIS_HOST')}:{self.env_vars.get('REDIS_PORT')}")
        
        # Check critical API keys
        av_key_set = bool(self.env_vars.get('AV_API_KEY') and 
                         self.env_vars['AV_API_KEY'] != 'your_alpha_vantage_key_here')
        ibkr_user_set = bool(self.env_vars.get('IBKR_USERNAME') and 
                            self.env_vars['IBKR_USERNAME'] != 'your_ibkr_username')
        
        print(f"\nAPI Keys:")
        print(f"  • Alpha Vantage: {'✓ Set' if av_key_set else '✗ Not Set'}")
        print(f"  • IBKR: {'✓ Set' if ibkr_user_set else '✗ Not Set'}")
        print("=" * 60)


# Convenience function for quick config access
_global_config = None

def get_config(environment: str = None) -> ConfigManager:
    """
    Get or create global configuration instance
    
    Args:
        environment: Environment to load (uses ENVIRONMENT env var if not specified)
        
    Returns:
        ConfigManager instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(environment)
    return _global_config


# Example usage and testing
if __name__ == "__main__":
    """Test configuration manager"""
    
    # Create config manager
    config = ConfigManager()
    
    # Print summary
    config.print_summary()
    
    # Test getting various configuration values
    print("\n" + "=" * 60)
    print("CONFIGURATION TESTS")
    print("=" * 60)
    
    # Test dot notation access
    tests = [
        ('apis.alpha_vantage.alpha_vantage.base_url', 'Alpha Vantage URL'),
        ('strategies.0dte.strategy.name', '0DTE Strategy Name'),
        ('risk.position_limits.position_limits.max_delta', 'Max Delta Limit'),
        ('data.symbols.symbols.tier_a.symbols', 'Tier A Symbols'),
        ('monitoring.discord.discord.username', 'Discord Username'),
    ]
    
    for path, description in tests:
        value = config.get(path, 'NOT FOUND')
        print(f"{description}: {value}")
    
    # Test strategy config helper
    print("\n" + "=" * 60)
    print("STRATEGY CONFIGURATIONS")
    print("=" * 60)
    
    for strategy in ['0dte', '1dte', 'swing_14d', 'moc_imbalance']:
        strategy_config = config.get_strategy_config(strategy)
        if strategy_config:
            name = strategy_config.get('strategy', {}).get('name', 'Unknown')
            min_confidence = strategy_config.get('confidence', {}).get('minimum', 'N/A')
            print(f"{strategy}: {name} (Min Confidence: {min_confidence})")
    
    # Export configuration (for debugging)
    # export_path = config.export_config()
    # print(f"\nConfiguration exported to: {export_path}")