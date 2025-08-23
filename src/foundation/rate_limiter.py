"""
Token bucket rate limiter implementation
All configuration from environment - zero hardcoding
"""
import os
import time
import threading
from typing import Optional

from .logger import get_logger
from .exceptions import RateLimitException
from .metrics import MetricsCollector


class RateLimiter:
    """
    Token bucket rate limiter
    All configuration from environment
    """
    
    def __init__(self, name: str):
        """
        Initialize rate limiter
        
        Args:
            name: Name of the rate limiter (used for config lookup)
        """
        self.name = name
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Get configuration from environment - NO DEFAULTS
        self.enabled = os.environ['RATE_LIMIT_ENABLED'].lower() == 'true'
        
        # Try to get specific config for this limiter, fall back to defaults
        env_prefix = f'RATE_LIMIT_{name.upper()}'
        self.bucket_size = int(
            os.environ.get(f'{env_prefix}_BUCKET_SIZE') or 
            os.environ['RATE_LIMIT_DEFAULT_BUCKET_SIZE']
        )
        self.refill_rate = float(
            os.environ.get(f'{env_prefix}_REFILL_RATE') or 
            os.environ['RATE_LIMIT_DEFAULT_REFILL_RATE']
        )
        
        # Initialize token bucket
        self.tokens = self.bucket_size
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        self.logger.info(
            f"Rate limiter initialized",
            name=name,
            bucket_size=self.bucket_size,
            refill_rate=self.refill_rate
        )
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate
        
        # Add tokens up to bucket size
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def allow_request(self, tokens: int = 1) -> bool:
        """
        Check if request is allowed
        
        Args:
            tokens: Number of tokens required
            
        Returns:
            True if request is allowed
        """
        if not self.enabled:
            return True
        
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                
                # Update metrics
                self.metrics.update_rate_limit(
                    self.name,
                    int(self.tokens),
                    'allowed'
                )
                
                return True
            else:
                # Update metrics
                self.metrics.update_rate_limit(
                    self.name,
                    int(self.tokens),
                    'denied'
                )
                
                return False
    
    def consume_token(self, tokens: int = 1):
        """
        Consume tokens or raise exception
        
        Args:
            tokens: Number of tokens to consume
            
        Raises:
            RateLimitException: If not enough tokens
        """
        if not self.allow_request(tokens):
            wait_time = self.get_wait_time(tokens)
            raise RateLimitException(
                f"Rate limit exceeded for {self.name}",
                retry_after=int(wait_time)
            )
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait for tokens to be available
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait
        """
        if not self.enabled:
            return 0
        
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0
            
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            
            return wait_time
    
    def get_available_tokens(self) -> int:
        """Get current available tokens"""
        if not self.enabled:
            return self.bucket_size
        
        with self.lock:
            self._refill()
            return int(self.tokens)
    
    def reset(self):
        """Reset rate limiter to full capacity"""
        with self.lock:
            self.tokens = self.bucket_size
            self.last_refill = time.time()
            
            self.logger.info(f"Rate limiter {self.name} reset")