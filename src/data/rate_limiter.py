"""Token bucket rate limiter for API calls - Phase 2"""

import time
import threading
from datetime import datetime, timedelta


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for Alpha Vantage API
    - Maximum 600 calls per minute (hard limit)
    - Refill rate: 10 tokens per second
    - Burst capacity: 20 tokens
    """
    
    def __init__(self, 
                 max_per_minute=600,
                 refill_rate=10,  # tokens per second
                 burst_capacity=20):
        
        # Configuration
        self.max_per_minute = max_per_minute
        self.refill_rate = refill_rate
        self.burst_capacity = burst_capacity
        
        # State
        self.tokens = burst_capacity  # Start with full burst capacity
        self.last_refill = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.calls_made = 0
        self.calls_blocked = 0
        self.minute_window_start = time.time()
        self.minute_window_calls = 0
        
        print(f"Rate limiter initialized:")
        print(f"  - Max per minute: {max_per_minute}")
        print(f"  - Refill rate: {refill_rate} tokens/second")
        print(f"  - Burst capacity: {burst_capacity}")
    
    def acquire(self, tokens=1, blocking=True, timeout=30):
        """
        Acquire tokens to make an API call
        
        Args:
            tokens: Number of tokens needed (default 1)
            blocking: Wait if tokens not available (default True)
            timeout: Max seconds to wait (default 30)
        
        Returns:
            bool: True if tokens acquired, False if timeout/non-blocking fail
        """
        deadline = time.time() + timeout if blocking else time.time()
        
        while True:
            with self.lock:
                # Refill tokens based on time elapsed
                self._refill()
                
                # Check minute window
                self._check_minute_window()
                
                # Try to acquire tokens
                if self.tokens >= tokens and self.minute_window_calls < self.max_per_minute:
                    self.tokens -= tokens
                    self.calls_made += 1
                    self.minute_window_calls += 1
                    
                    # Log status every 10 calls
                    if self.calls_made % 10 == 0:
                        self._log_status()
                    
                    return True
                
                # If not blocking or timeout, return False
                if not blocking or time.time() >= deadline:
                    self.calls_blocked += 1
                    print(f"⚠️ Rate limit: {self.tokens:.1f} tokens available, "
                          f"{self.minute_window_calls}/{self.max_per_minute} this minute")
                    return False
            
            # Wait a bit before trying again
            time.sleep(0.1)
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        
        if tokens_to_add > 0:
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def _check_minute_window(self):
        """Reset minute window if needed"""
        now = time.time()
        if now - self.minute_window_start >= 60:
            self.minute_window_start = now
            self.minute_window_calls = 0
    
    def _log_status(self):
        """Log current rate limiter status"""
        print(f"Rate limiter: {self.calls_made} calls made, "
              f"{self.tokens:.1f}/{self.burst_capacity} tokens, "
              f"{self.minute_window_calls}/{self.max_per_minute} this minute")
    
    def get_stats(self):
        """Get rate limiter statistics"""
        with self.lock:
            return {
                'calls_made': self.calls_made,
                'calls_blocked': self.calls_blocked,
                'tokens_available': self.tokens,
                'minute_window_calls': self.minute_window_calls,
                'success_rate': (self.calls_made / (self.calls_made + self.calls_blocked) * 100) 
                                if (self.calls_made + self.calls_blocked) > 0 else 100
            }
    
    def wait_time(self):
        """Get estimated wait time for next token"""
        with self.lock:
            if self.tokens >= 1:
                return 0
            return (1 - self.tokens) / self.refill_rate


# Global rate limiter instance (singleton)
_rate_limiter = None

def get_rate_limiter():
    """Get or create the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = TokenBucketRateLimiter()
    return _rate_limiter