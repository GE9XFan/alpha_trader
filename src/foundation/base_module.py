"""
Base Module Abstract Class
All modules inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


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
        self.status = ComponentStatus.UNINITIALIZED
        self.health_status = HealthStatus.UNKNOWN
        self.start_time = None
        self.error_count = 0
        self.last_error = None
        self.dependencies = []
        
        # Set up module-specific logger
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.info(f"Creating module: {name}")
    
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
    
    def start(self) -> bool:
        """
        Start the module
        
        Returns:
            True if started successfully
        """
        try:
            self.status = ComponentStatus.INITIALIZING
            self.logger.info(f"Starting module: {self.name}")
            
            # Initialize
            if self.initialize():
                self.status = ComponentStatus.RUNNING
                self.start_time = datetime.now()
                self.logger.info(f"Module started successfully: {self.name}")
                return True
            else:
                self.status = ComponentStatus.ERROR
                self.logger.error(f"Failed to start module: {self.name}")
                return False
                
        except Exception as e:
            self.status = ComponentStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            self.logger.error(f"Error starting module {self.name}: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the module
        
        Returns:
            True if stopped successfully
        """
        try:
            self.logger.info(f"Stopping module: {self.name}")
            
            if self.shutdown():
                self.status = ComponentStatus.SHUTDOWN
                self.logger.info(f"Module stopped successfully: {self.name}")
                return True
            else:
                self.logger.error(f"Failed to stop module cleanly: {self.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping module {self.name}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current module status
        
        Returns:
            Status dictionary
        """
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'name': self.name,
            'status': self.status.value,
            'health': self.health_status.value,
            'uptime_seconds': uptime,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'dependencies': self.dependencies
        }
    
    def add_dependency(self, module_name: str) -> None:
        """Add a module dependency"""
        if module_name not in self.dependencies:
            self.dependencies.append(module_name)
    
    def is_ready(self) -> bool:
        """Check if module is ready for operations"""
        return self.status == ComponentStatus.RUNNING
    
    def is_healthy(self) -> bool:
        """Check if module is healthy"""
        return self.health_status == HealthStatus.HEALTHY