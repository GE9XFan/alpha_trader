"""
Logging configuration for AlphaTrader
"""
import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional


class LoggerManager:
    """Centralized logger management"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self):
        """Setup logging configuration"""
        config_path = Path('config/logging.yaml')
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Ensure log directory exists
            Path('logs').mkdir(exist_ok=True)
            
            logging.config.dictConfig(config)
        else:
            # Fallback configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance"""
        return logging.getLogger(name)


# Global logger manager
logger_manager = LoggerManager()

# Convenience function
def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module"""
    return logger_manager.get_logger(name)
