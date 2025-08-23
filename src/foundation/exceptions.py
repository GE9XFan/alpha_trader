"""
Custom exception hierarchy with correlation tracking
Institutional-grade error handling
"""
import uuid
import os
from typing import Optional, Dict, Any
from datetime import datetime


class AlphaTraderException(Exception):
    """
    Base exception with correlation ID and metadata
    All custom exceptions inherit from this
    """
    
    def __init__(
        self, 
        message: str, 
        correlation_id: Optional[str] = None,
        **metadata
    ):
        """
        Initialize exception with correlation tracking
        
        Args:
            message: Error message
            correlation_id: Optional correlation ID for tracking
            **metadata: Additional context information
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.metadata = metadata
        self.message = message
        self.timestamp = datetime.utcnow().isoformat()
        
        # Include environment for debugging (from env, not hardcoded)
        self.environment = os.getenv('ENVIRONMENT', 'unknown')
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp,
            'environment': self.environment,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation with correlation ID"""
        return f"{self.__class__.__name__}: {self.message} (correlation_id: {self.correlation_id})"


class ConfigException(AlphaTraderException):
    """Configuration-related exceptions"""
    pass


class DatabaseException(AlphaTraderException):
    """Database operation exceptions"""
    pass


class CacheException(AlphaTraderException):
    """Cache operation exceptions"""
    pass


class ValidationException(AlphaTraderException):
    """Data validation exceptions"""
    pass


class RateLimitException(AlphaTraderException):
    """Rate limiting exceptions"""
    
    def __init__(
        self, 
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize rate limit exception
        
        Args:
            message: Error message
            retry_after: Seconds until retry is allowed
            **kwargs: Additional metadata
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.metadata['retry_after'] = retry_after


class ConnectionException(AlphaTraderException):
    """Connection-related exceptions"""
    pass


class TimeoutException(AlphaTraderException):
    """Timeout exceptions"""
    
    def __init__(
        self,
        message: str,
        timeout_value: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize timeout exception
        
        Args:
            message: Error message
            timeout_value: The timeout value that was exceeded
            **kwargs: Additional metadata
        """
        super().__init__(message, **kwargs)
        if timeout_value:
            self.metadata['timeout_value'] = timeout_value


class AuthenticationException(AlphaTraderException):
    """Authentication/authorization exceptions"""
    pass


class DataIntegrityException(AlphaTraderException):
    """Data integrity and consistency exceptions"""
    pass


class CircuitBreakerException(AlphaTraderException):
    """Circuit breaker triggered exceptions"""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        recovery_time: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize circuit breaker exception
        
        Args:
            message: Error message
            service: Service that triggered the breaker
            recovery_time: Estimated recovery time in seconds
            **kwargs: Additional metadata
        """
        super().__init__(message, **kwargs)
        if service:
            self.metadata['service'] = service
        if recovery_time:
            self.metadata['recovery_time'] = recovery_time


class HealthCheckException(AlphaTraderException):
    """Health check failure exceptions"""
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        status: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize health check exception
        
        Args:
            message: Error message
            component: Component that failed health check
            status: Health status
            **kwargs: Additional metadata
        """
        super().__init__(message, **kwargs)
        if component:
            self.metadata['component'] = component
        if status:
            self.metadata['status'] = status