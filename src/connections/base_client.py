"""
Base API Client Abstract Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """
    Abstract base class for API clients
    Provides consistent interface for all external API connections
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "APIClient"):
        """
        Initialize API client
        
        Args:
            config: API configuration
            name: Client name for logging
        """
        self.config = config
        self.name = name
        self.is_connected = False
        self.last_call_time = None
        self.total_calls = 0
        self.error_count = 0
        self.rate_limiter = None
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to API
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from API
        
        Returns:
            True if disconnection successful
        """
        pass
    
    @abstractmethod
    def call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API call
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check API connection health
        
        Returns:
            True if healthy
        """
        pass
    
    def call_with_retry(self, endpoint: str, params: Dict[str, Any], 
                       max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Make API call with retry logic
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            max_retries: Maximum retry attempts
            
        Returns:
            API response or None if all retries failed
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.call(endpoint, params)
                self.total_calls += 1
                self.last_call_time = datetime.now()
                return response
                
            except Exception as e:
                last_error = e
                self.error_count += 1
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                import time
                time.sleep(wait_time)
        
        self.logger.error(f"All retry attempts failed. Last error: {last_error}")
        return None
    
    def set_rate_limiter(self, rate_limiter) -> None:
        """Attach a rate limiter to this client"""
        self.rate_limiter = rate_limiter
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'name': self.name,
            'connected': self.is_connected,
            'total_calls': self.total_calls,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_calls, 1),
            'last_call': self.last_call_time.isoformat() if self.last_call_time else None
        }