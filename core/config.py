"""
Configuration management for the AlphaTrader system.
Handles YAML configuration files with environment variable substitution.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from string import Template

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and manages configuration from YAML files.
    
    Features:
    - Environment variable substitution
    - Hierarchical configuration
    - Default values
    - Configuration validation
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache: Dict[str, Dict] = {}
        
        # Load environment variables
        self._load_env()
        
        logger.info(f"Configuration loader initialized with dir: {config_dir}")
    
    def _load_env(self) -> None:
        """Load environment variables from .env file if it exists."""
        env_file = Path(".env")
        
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv()
                logger.info("Loaded environment variables from .env")
            except ImportError:
                logger.warning("python-dotenv not installed, skipping .env file")
    
    def load(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            use_cache: Whether to use cached configuration
            
        Returns:
            Configuration dictionary
        """
        if use_cache and config_name in self.config_cache:
            return self.config_cache[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} not found")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                # Read file content
                content = f.read()
                
                # Substitute environment variables
                content = self._substitute_env_vars(content)
                
                # Parse YAML
                config = yaml.safe_load(content)
                
                # Cache the configuration
                self.config_cache[config_name] = config
                
                logger.info(f"Loaded configuration: {config_name}")
                return config
                
        except Exception as e:
            logger.error(f"Failed to load configuration {config_name}: {e}")
            return {}
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.
        
        Supports ${VAR_NAME} syntax with optional defaults: ${VAR_NAME:default_value}
        
        Args:
            content: Configuration file content
            
        Returns:
            Content with environment variables substituted
        """
        # Find all environment variable references
        import re
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_expr = match.group(1)
            
            # Check for default value syntax
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name = var_expr
                default_value = None
            
            # Get environment variable value
            value = os.getenv(var_name, default_value)
            
            if value is None:
                logger.warning(f"Environment variable {var_name} not found and no default provided")
                return match.group(0)  # Return original if not found
            
            return value
        
        return re.sub(pattern, replacer, content)
    
    def load_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin configuration dictionary
        """
        # Try to load from plugins directory
        plugin_config_path = self.config_dir / "plugins" / f"{plugin_name.lower()}.yaml"
        
        if plugin_config_path.exists():
            try:
                with open(plugin_config_path, 'r') as f:
                    content = f.read()
                    content = self._substitute_env_vars(content)
                    config = yaml.safe_load(content)
                    return config
            except Exception as e:
                logger.error(f"Failed to load plugin config {plugin_name}: {e}")
        
        return {}
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        Get the main system configuration.
        
        Returns:
            System configuration dictionary
        """
        return self.load("system")
    
    def reload_all(self) -> None:
        """Reload all configurations from disk."""
        self.config_cache.clear()
        logger.info("Configuration cache cleared, will reload on next access")
    
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate a configuration against a schema.
        
        Args:
            config: Configuration to validate
            schema: Schema to validate against
            
        Returns:
            True if valid, False otherwise
        """
        # Simple validation - can be extended with jsonschema or pydantic
        for required_key in schema.get('required', []):
            if required_key not in config:
                logger.error(f"Required configuration key missing: {required_key}")
                return False
        
        return True


class Config:
    """
    Singleton configuration manager for the entire system.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.loader = ConfigLoader()
            self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Dot-separated configuration key (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Load system config
        config = self.loader.get_system_config()
        
        # Navigate through nested keys
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get configuration for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin configuration dictionary
        """
        return self.loader.load_plugin_config(plugin_name)