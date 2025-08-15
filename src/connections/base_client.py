"""
Base API Client Abstract Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """
    Abstract base class for API clients
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize API client
        
        Args:
            config: API configuration
        """
        self.config = config
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to API"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from API"""
        pass
    
    @abstractmethod
    def call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check API connection health"""
        pass
