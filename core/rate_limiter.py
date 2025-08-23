"""
Rate limiting utilities for API calls.
Critical for staying within Alpha Vantage's API limits.
"""

import asyncio
import time
import threading
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket rate limiter for API calls.
    
    This implementation is thread-safe and supports:
    - Configurable capacity (max tokens)
    - Configurable refill rate
    - Burst handling
    - Async and sync interfaces
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize the token bucket.
        
        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Number of tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.RLock()
        
        logger.info(f"Token bucket initialized: capacity={capacity}, rate={refill_rate}/sec")
    
    def _refill(self) -> None:
        """
        Refill tokens based on time elapsed.
        Must be called with lock held.
        """
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate
        
        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens, blocking if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens were acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if self.try_acquire(tokens):
                return True
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            # Sleep briefly before retrying
            time.sleep(0.01)  # 10ms
    
    async def async_acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Async version of acquire.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens were acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if self.try_acquire(tokens):
                return True
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            # Async sleep
            await asyncio.sleep(0.01)  # 10ms
    
    def available_tokens(self) -> float:
        """
        Get the number of available tokens.
        
        Returns:
            Number of tokens available
        """
        with self._lock:
            self._refill()
            return self.tokens
    
    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.time()
            logger.info("Token bucket reset to full capacity")


class MultiLevelRateLimiter:
    """
    Multi-level rate limiter for complex API limits.
    All limits are configuration-driven.
    
    CRITICAL: Alpha Vantage allows 600 calls per MINUTE total.
    This is NOT 10 calls per second sustained - it's a hard limit.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-level rate limiter from configuration.
        
        Args:
            config: Rate limiter configuration dictionary with:
                - calls_per_minute: Max calls per minute (600 for Alpha Vantage)
                - daily_limit: Maximum calls per day
        """
        # CRITICAL: Alpha Vantage calls per minute is a hard limit
        # Get from config - NO HARDCODING
        calls_per_minute = config.get('calls_per_minute')
        if calls_per_minute is None:
            raise ValueError("calls_per_minute MUST be specified in config")
        
        # Standard limit - refill rate is calls_per_minute / 60 seconds
        self.standard_bucket = TokenBucket(
            capacity=calls_per_minute,
            refill_rate=calls_per_minute / 60.0  # This gives us the per-second rate
        )
        
        # Premium limit from config - NO DEFAULTS
        premium_capacity = config.get('premium_capacity')
        if premium_capacity:
            self.premium_bucket = TokenBucket(
                capacity=premium_capacity,
                refill_rate=config.get('premium_rate', premium_capacity / 60.0)
            )
        else:
            self.premium_bucket = None
        
        # Burst bucket from config - OPTIONAL
        burst_capacity = config.get('burst_capacity')
        if burst_capacity:
            self.burst_bucket = TokenBucket(
                capacity=burst_capacity,
                refill_rate=config.get('burst_rate', 1.0)
            )
        else:
            self.burst_bucket = None
        
        # Daily limit from config - REQUIRED
        self.daily_calls = 0
        self.daily_limit = config.get('daily_limit')
        if self.daily_limit is None:
            raise ValueError("daily_limit MUST be specified in config")
        self.last_reset = time.time()
        
        # Timeout configuration - this one can have a default
        self.acquire_timeout = config.get('acquire_timeout', 5.0)
        
        logger.info(f"Multi-level rate limiter initialized with config: {config}")
    
    def _check_daily_reset(self) -> None:
        """Reset daily counter if new day."""
        now = time.time()
        # Reset every 24 hours
        if now - self.last_reset > 86400:
            self.daily_calls = 0
            self.last_reset = now
            logger.info("Daily rate limit counter reset")
    
    async def acquire_standard(self) -> bool:
        """
        Acquire permission for a standard API call.
        
        Returns:
            True if permission granted
        """
        self._check_daily_reset()
        
        # Check daily limit
        if self.daily_limit and self.daily_calls >= self.daily_limit:
            logger.warning(f"Daily limit reached: {self.daily_calls}/{self.daily_limit}")
            return False
        
        # Try burst bucket first for immediate response
        if self.burst_bucket and self.burst_bucket.try_acquire():
            self.daily_calls += 1
            return True
        
        # Fall back to standard bucket
        if await self.standard_bucket.async_acquire(timeout=self.acquire_timeout):
            self.daily_calls += 1
            return True
        
        logger.warning("Failed to acquire rate limit token")
        return False
    
    async def acquire_premium(self) -> bool:
        """
        Acquire permission for a premium API call.
        
        Returns:
            True if permission granted
        """
        self._check_daily_reset()
        
        # Check daily limit
        if self.daily_limit and self.daily_calls >= self.daily_limit:
            logger.warning(f"Daily limit reached: {self.daily_calls}/{self.daily_limit}")
            return False
        
        # Premium endpoints use the premium bucket
        if self.premium_bucket and await self.premium_bucket.async_acquire(timeout=self.acquire_timeout):
            self.daily_calls += 1
            return True
        
        logger.warning("Failed to acquire premium rate limit token")
        return False
    
    def get_status(self) -> dict:
        """
        Get current rate limiter status.
        
        Returns:
            Dictionary with rate limiter metrics
        """
        return {
            'standard_tokens': self.standard_bucket.available_tokens(),
            'premium_tokens': self.premium_bucket.available_tokens() if self.premium_bucket else 0,
            'burst_tokens': self.burst_bucket.available_tokens() if self.burst_bucket else 0,
            'daily_calls': self.daily_calls,
            'daily_limit': self.daily_limit,
            'daily_remaining': max(0, self.daily_limit - self.daily_calls) if self.daily_limit else 0
        }
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'MultiLevelRateLimiter':
        """
        Create rate limiter from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured MultiLevelRateLimiter instance
        """
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        rate_config = config.get('rate_limiter', {})
        return cls(rate_config)