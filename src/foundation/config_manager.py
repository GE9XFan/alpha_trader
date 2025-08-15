"""
Configuration Management System
Handles all configuration loading and access
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Central configuration management - NO HARDCODED VALUES
    All configuration loaded from YAML files and environment variables
    """
    
    def __init__(self, environment: str = 'development'):
        """
        Initialize configuration manager
        
        Args:
            environment: Environment name (development, paper, production)
        """
        self.environment = environment
        self.config: Dict[str, Any] = {}
        self.config_dir = Path('config')
        self._load_env_variables()
        self._load_yaml_configs()
        self._apply_environment_overrides()
        logger.info(f"ConfigManager initialized for environment: {environment}")
    
    def _load_env_variables(self) -> None:
        """Load environment variables from .env file"""
        load_dotenv()
        self.config['env'] = {
            'av_api_key': os.getenv('AV_API_KEY'),
            'ibkr_username': os.getenv('IBKR_USERNAME'),
            'ibkr_password': os.getenv('IBKR_PASSWORD'),
            'ibkr_account': os.getenv('IBKR_ACCOUNT'),
            'db_host': os.getenv('DB_HOST', 'localhost'),
            'db_port': os.getenv('DB_PORT', '5432'),
            'db_name': os.getenv('DB_NAME', 'trading_system'),
            'db_user': os.getenv('DB_USER', 'postgres'),
            'db_password': os.getenv('DB_PASSWORD'),
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': os.getenv('REDIS_PORT', '6379'),
            'redis_password': os.getenv('REDIS_PASSWORD'),
            'discord_webhook_url': os.getenv('DISCORD_WEBHOOK_URL'),
        }
    
    def _load_yaml_configs(self) -> None:
        """Load all YAML configuration files"""
        # Implementation will be completed in Step 0.6
        pass
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides"""
        # Implementation will be completed in Step 0.6
        pass
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation path
        
        Args:
            path: Dot notation path (e.g., 'apis.alpha_vantage.rate_limit')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        # Implementation will be completed in Step 0.6
        pass
    
    def validate_required_keys(self) -> bool:
        """Validate all required configuration keys are present"""
        # Implementation will be completed in Step 0.6
        pass
