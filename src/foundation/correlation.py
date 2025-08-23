"""
Correlation tracking for request tracing
Institutional-grade request tracking
"""
import os
import uuid
import threading
from typing import Optional

from .logger import get_logger


class CorrelationContext:
    """
    Thread-local correlation ID management
    All configuration from environment
    """
    
    _thread_local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID from thread-local storage"""
        return getattr(cls._thread_local, 'correlation_id', None)
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID in thread-local storage"""
        cls._thread_local.correlation_id = correlation_id
    
    @classmethod
    def clear_correlation_id(cls):
        """Clear correlation ID from thread-local storage"""
        if hasattr(cls._thread_local, 'correlation_id'):
            delattr(cls._thread_local, 'correlation_id')
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate new correlation ID"""
        return str(uuid.uuid4())
    
    @classmethod
    def ensure_correlation_id(cls) -> str:
        """Ensure correlation ID exists, generate if needed"""
        correlation_id = cls.get_correlation_id()
        if not correlation_id:
            correlation_id = cls.generate_correlation_id()
            cls.set_correlation_id(correlation_id)
        return correlation_id
    
    @classmethod
    def get_header_name(cls) -> str:
        """Get correlation ID header name from environment"""
        return os.environ.get('CORRELATION_ID_HEADER', 'X-Correlation-ID')