#!/usr/bin/env python3
"""
Rate Limiter - Token bucket algorithm for API rate limiting
Phase 1: Zero hardcoded values - everything configuration-driven
"""

import time
import threading
import logging
from typing import Optional


class RateLimiter:
    """
    Token bucket rate limiter for API calls - Zero hardcoded values
    
    Implements a token bucket algorithm to ensure API calls stay within rate limits
    while allowing for burst capacity when tokens are available.
    All values must be provided via configuration.
    """
    
    def __init__(self, calls_per_minute: int, burst_capacity: int, 
                 refill_rate: float, time_window: int, check_interval: float,
                 initial_tokens: Optional[float] = None, 
                 initial_total_calls: int = 0, initial_rejected_calls: int = 0,
                 initial_window_calls: int = 0):
        """
        Initialize rate limiter - ALL parameters required from configuration
        
        Args:
            calls_per_minute: Maximum calls allowed per time window
            burst_capacity: Maximum tokens that can be stored
            refill_rate: Tokens added per second
            time_window: Time window in seconds (e.g., 60 for per-minute limiting)
            check_interval: How often to check for token refill (seconds)
            initial_tokens: Starting token count (defaults to burst_capacity if None)
            initial_total_calls: Starting total call count
            initial_rejected_calls: Starting rejected call count
            initial_window_calls: Starting window call count
        """
        # Rate limiting configuration
        self.calls_per_minute = calls_per_minute
        self.burst_capacity = burst_capacity
        self.refill_rate = refill_rate
        self.time_window = time_window
        self.check_interval = check_interval
        
        # Token bucket state
        self.tokens = float(initial_tokens if initial_tokens is not None else burst_capacity)
        self.last_refill = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Tracking - all configurable
        self.total_calls = initial_total_calls
        self.rejected_calls = initial_rejected_calls
        self.window_calls = initial_window_calls
        self.window_start = time.time()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"RateLimiter initialized: {calls_per_minute} calls/{time_window}s, "
                        f"burst capacity: {burst_capacity}, refill rate: {refill_rate}/sec, "
                        f"check interval: {check_interval}s")
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
        
        self.last_refill = now
    
    def _reset_window_counter(self):
        """Reset time window counter if window has passed"""
        now = time.time()
        if now - self.window_start >= self.time_window:
            self.window_calls = 0
            self.window_start = now
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens for API calls
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait for tokens (None = wait indefinitely)
            
        Returns:
            True if tokens acquired, False if timeout exceeded
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                self._refill_tokens()
                self._reset_window_counter()
                
                # Check per-window limit
                if self.window_calls >= self.calls_per_minute:
                    if timeout is not None and (time.time() - start_time) >= timeout:
                        self.rejected_calls += 1
                        self.logger.warning(f"Rate limit exceeded: per-{self.time_window}s limit reached")
                        return False
                    
                    # Wait until next window
                    sleep_time = self.time_window - (time.time() - self.window_start)
                    if sleep_time > 0:
                        time.sleep(min(self.check_interval, sleep_time))
                    continue
                
                # Check token availability
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.total_calls += 1
                    self.window_calls += 1
                    
                    self.logger.debug(f"Token acquired. Remaining: {self.tokens:.2f}, "
                                    f"Window calls: {self.window_calls}/{self.calls_per_minute}")
                    return True
                
                # Not enough tokens available
                if timeout is not None and (time.time() - start_time) >= timeout:
                    self.rejected_calls += 1
                    self.logger.warning("Rate limit exceeded: insufficient tokens")
                    return False
            
            # Wait for token refill (outside lock to allow other threads)
            time.sleep(self.check_interval)
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens immediately without waiting
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired immediately, False otherwise
        """
        return self.acquire(tokens, timeout=0)
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics
        
        Returns:
            Dictionary with rate limiter stats
        """
        with self._lock:
            self._refill_tokens()
            self._reset_window_counter()
            
            return {
                'total_calls': self.total_calls,
                'rejected_calls': self.rejected_calls,
                'current_tokens': round(self.tokens, 2),
                'window_calls': self.window_calls,
                'calls_per_window_limit': self.calls_per_minute,
                'time_window': self.time_window,
                'burst_capacity': self.burst_capacity,
                'refill_rate': self.refill_rate,
                'check_interval': self.check_interval,
                'rejection_rate': (self.rejected_calls / max(1, self.total_calls + self.rejected_calls)) * 100
            }
    
    def reset_stats(self, reset_total_calls: int = 0, reset_rejected_calls: int = 0, 
                   reset_window_calls: int = 0):
        """
        Reset statistics counters - values configurable
        
        Args:
            reset_total_calls: Value to reset total_calls to
            reset_rejected_calls: Value to reset rejected_calls to  
            reset_window_calls: Value to reset window_calls to
        """
        with self._lock:
            self.total_calls = reset_total_calls
            self.rejected_calls = reset_rejected_calls
            self.window_calls = reset_window_calls
            self.window_start = time.time()
            self.logger.info(f"Rate limiter statistics reset to: total={reset_total_calls}, "
                           f"rejected={reset_rejected_calls}, window={reset_window_calls}")
    
    def wait_for_capacity(self, required_tokens: int = 1) -> float:
        """
        Calculate how long to wait for required token capacity
        
        Args:
            required_tokens: Number of tokens needed
            
        Returns:
            Time in seconds to wait
        """
        with self._lock:
            self._refill_tokens()
            
            if self.tokens >= required_tokens:
                return 0.0
            
            tokens_needed = required_tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            
            return wait_time
    
    def set_rate_limit(self, calls_per_minute: int, reset_window_calls: int = 0):
        """
        Update the rate limit
        
        Args:
            calls_per_minute: New rate limit
            reset_window_calls: Value to reset window calls to
        """
        with self._lock:
            old_limit = self.calls_per_minute
            self.calls_per_minute = calls_per_minute
            
            # Reset window counter to apply new limit immediately
            self.window_calls = reset_window_calls
            self.window_start = time.time()
            
            self.logger.info(f"Rate limit updated: {old_limit} -> {calls_per_minute} calls/{self.time_window}s")
    
    def reconfigure(self, calls_per_minute: Optional[int] = None, 
                   burst_capacity: Optional[int] = None,
                   refill_rate: Optional[float] = None,
                   time_window: Optional[int] = None,
                   check_interval: Optional[float] = None):
        """
        Reconfigure rate limiter parameters
        
        Args:
            calls_per_minute: New calls per time window limit
            burst_capacity: New burst capacity
            refill_rate: New refill rate
            time_window: New time window
            check_interval: New check interval
        """
        with self._lock:
            if calls_per_minute is not None:
                self.calls_per_minute = calls_per_minute
            if burst_capacity is not None:
                self.burst_capacity = burst_capacity
                # Adjust current tokens if needed
                self.tokens = min(self.burst_capacity, self.tokens)
            if refill_rate is not None:
                self.refill_rate = refill_rate
            if time_window is not None:
                self.time_window = time_window
                # Reset window when changing time window
                self.window_calls = 0
                self.window_start = time.time()
            if check_interval is not None:
                self.check_interval = check_interval
            
            self.logger.info(f"Rate limiter reconfigured: {self.calls_per_minute} calls/{self.time_window}s, "
                           f"burst: {self.burst_capacity}, refill: {self.refill_rate}/s, "
                           f"check: {self.check_interval}s")
    
    def __enter__(self):
        """Context manager entry"""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass  # Nothing to do on exit