"""
Configuration management with ZERO hardcoded values
Everything comes from environment or config files
Institutional-grade configuration system
"""
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConfigException(Exception):
    """Configuration-related exception"""
    pass


class ConfigManager:
    """
    All configuration externalized - zero hardcoding
    Singleton pattern for global access
    """
    
    _instance = None
    _configs: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load all configurations from environment - NO DEFAULTS"""
        # EVERYTHING from environment - no defaults allowed
        self.root_dir = self._require_env('APP_ROOT_DIR')
        self.config_dir = self._require_env('CONFIG_DIR')
        self.environment = self._require_env('ENVIRONMENT')
        
        # Convert to Path objects
        self.root_dir = Path(self.root_dir)
        self.config_dir = Path(self.config_dir)
        
        # Validate paths exist
        if not self.root_dir.exists():
            raise ConfigException(
                f"Root directory {self.root_dir} not found. "
                f"Check APP_ROOT_DIR environment variable."
            )
        
        if not self.config_dir.exists():
            raise ConfigException(
                f"Config directory {self.config_dir} not found. "
                f"Check CONFIG_DIR environment variable."
            )
        
        # Load all YAML configs
        self._load_all_configs()
    
    def _require_env(self, key: str) -> str:
        """
        Get environment variable or FAIL - no defaults allowed
        This enforces our zero-hardcoding policy
        """
        value = os.getenv(key)
        if value is None:
            raise ConfigException(
                f"Required environment variable '{key}' not set. "
                f"NO DEFAULTS ALLOWED - all values must be external. "
                f"Please set this in your .env file or environment."
            )
        return value
    
    def _load_all_configs(self):
        """Load all YAML configuration files with env var substitution"""
        for config_file in self.config_dir.rglob("*.yaml"):
            try:
                relative_path = config_file.relative_to(self.config_dir)
                config_key = str(relative_path).replace("/", ".").replace(".yaml", "")
                
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                    # Replace all ${VAR_NAME} with environment variables
                    content = self._substitute_env_vars(content)
                    
                    # Parse YAML
                    config_data = yaml.safe_load(content)
                    
                    # Store in configs
                    self._configs[config_key] = config_data
                    
            except Exception as e:
                raise ConfigException(
                    f"Failed to load config file {config_file}: {e}"
                )
    
    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${VAR_NAME} with environment variable values"""
        import re
        
        def replacer(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ConfigException(
                    f"Environment variable '{var_name}' referenced in config but not set. "
                    f"All configuration values must be provided."
                )
            return value
        
        # Replace ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replacer, content)
    
    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value with dot notation
        Default must be None to enforce externalization
        """
        if default is not None:
            import warnings
            warnings.warn(
                f"Default value provided for '{key}'. "
                f"Consider externalizing this to configuration.",
                UserWarning
            )
        
        # Navigate through nested config with dot notation
        keys = key.split('.')
        value = self._configs
        
        for k in keys:
            if isinstance(value, dict):
                # Check environment-specific config first
                if self.environment in value and isinstance(value[self.environment], dict):
                    env_value = value[self.environment].get(k)
                    if env_value is not None:
                        value = env_value
                        continue
                
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_required(self, key: str) -> Any:
        """Get configuration value that must exist - no default"""
        value = self.get(key)
        if value is None:
            raise ConfigException(
                f"Required configuration key '{key}' not found. "
                f"Check your configuration files and environment variables."
            )
        return value
    
    def reload(self):
        """Reload all configurations - useful for hot reload"""
        self._configs.clear()
        self._load_all_configs()
    
    def get_all(self) -> Dict[str, Any]:
        """Get all loaded configurations"""
        return self._configs.copy()
    
    def validate(self):
        """Validate all required configurations are present"""
        required_keys = [
            'system.database.host',
            'system.database.port',
            'system.database.database',
            'system.database.user',
            'system.database.password',
            'system.redis.host',
            'system.redis.port',
            'system.logging.level',
            'system.foundation.retry.max_attempts',
            'system.foundation.metrics.enabled',
        ]
        
        missing = []
        for key in required_keys:
            if self.get(key) is None:
                missing.append(key)
        
        if missing:
            raise ConfigException(
                f"Missing required configuration keys: {missing}. "
                f"Check your YAML files and environment variables."
            )
        
        return True