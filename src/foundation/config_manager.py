#!/usr/bin/env python3
"""
Configuration Manager - Central configuration management for the trading system
Phase 0: Foundation Setup
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import json


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            environment: Override environment (development, paper, production)
        """
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.root_dir / 'config'
        
        # Load environment variables
        self._load_env()
        
        # Set environment
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        
        # Initialize logger
        self._setup_logging()
        
        # Load all configurations
        self.configs = {}
        self._load_all_configs()
        
        # Create convenience properties
        self._setup_properties()
        
        self.logger.info(f"ConfigManager initialized for environment: {self.environment}")
    
    def _load_env(self):
        """Load environment variables from .env file"""
        env_file = self.config_dir / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logging.info(f"Loaded environment from {env_file}")
        else:
            logging.warning(f"No .env file found at {env_file}, using system environment")
    
    def _setup_logging(self):
        """Setup basic logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _substitute_env_vars(self, config: Dict) -> Dict:
        """
        Recursively substitute environment variables in config
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Config with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            value = os.getenv(env_var)
            if value is None:
                self.logger.warning(f"Environment variable {env_var} not found")
                return config
            # Try to parse as JSON for proper type conversion
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        else:
            return config
    
    def _load_yaml(self, file_path: Path) -> Dict:
        """
        Load and parse YAML file with environment variable substitution
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed configuration dictionary
        """
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            return config or {}
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def _load_all_configs(self):
            """Load all configuration files"""
            # System configurations
            self.configs['database'] = self._load_yaml(self.config_dir / 'system' / 'database.yaml')
            self.configs['redis'] = self._load_yaml(self.config_dir / 'system' / 'redis.yaml')
            self.configs['logging'] = self._load_yaml(self.config_dir / 'system' / 'logging.yaml')  # Add this line
            
            # API configurations
            self.configs['alpha_vantage'] = self._load_yaml(self.config_dir / 'apis' / 'alpha_vantage.yaml')
            self.configs['ibkr'] = self._load_yaml(self.config_dir / 'apis' / 'ibkr.yaml')
            
            # Data configurations
            self.configs['schedules'] = self._load_yaml(self.config_dir / 'data' / 'schedules.yaml')
            
            self.logger.info(f"Loaded {len(self.configs)} configuration files")    
            
    def _setup_properties(self):
        """Setup convenience properties for common access patterns"""
        # Database
        self.db_config = self.configs.get('database', {}).get('database', {})
        
        # Redis
        self.redis_config = self.configs.get('redis', {}).get('redis', {})
        
        # Alpha Vantage
        self.av_config = self.configs.get('alpha_vantage', {}).get('alpha_vantage', {})
        self.av_api_key = self.av_config.get('api_key', os.getenv('ALPHA_VANTAGE_API_KEY'))
        
        # IBKR
        self.ibkr_config = self.configs.get('ibkr', {}).get('ibkr', {})
        
        # Schedules
        self.schedules = self.configs.get('schedules', {}).get('schedules', {})
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation path
        
        Args:
            path: Dot-notation path (e.g., 'database.pool_size')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        value = self.configs
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_db_connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['name']}"
        )
    
    def get_redis_connection_params(self) -> Dict:
        """Get Redis connection parameters"""
        return {
            'host': self.redis_config['host'],
            'port': self.redis_config['port'],
            'db': self.redis_config['db'],
            'password': self.redis_config['password'] if self.redis_config.get('password') else None,
            'decode_responses': True,
            'socket_timeout': self.redis_config.get('socket_timeout', 5)
        }
    
    def reload(self):
        """Reload all configurations"""
        self.logger.info("Reloading configurations...")
        self._load_all_configs()
        self._setup_properties()
        self.logger.info("Configuration reload complete")


# Singleton instance
_config_manager = None


def get_config_manager(environment: Optional[str] = None) -> ConfigManager:
    """
    Get or create singleton ConfigManager instance
    
    Args:
        environment: Optional environment override
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(environment)
    return _config_manager