"""
Base Module Abstract Class
All modules inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """
    Abstract base class for all system modules
    Provides consistent interface for initialization, health checks, and shutdown
    """
    
    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initialize base module
        
        Args:
            config: Module configuration dictionary
            name: Module name for logging
        """
        self.config = config
        self.name = name
        self.is_initialized = False
        logger.info(f"Initializing module: {name}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the module
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform module health check
        
        Returns:
            Dictionary with health status and metrics
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Gracefully shutdown the module
        
        Returns:
            True if shutdown successful
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current module status
        
        Returns:
            Status dictionary
        """
        return {
            'name': self.name,
            'initialized': self.is_initialized,
            'health': self.health_check() if self.is_initialized else None
        }
