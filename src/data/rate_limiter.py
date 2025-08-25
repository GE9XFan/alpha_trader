"""
Rate Limiter - Tech Spec Section 7.2
Smart rate limiting for 600 calls/minute
"""
import asyncio
import time
from typing import Optional

from src.core.logger import get_logger
from src.core.exceptions import RateLimitException


logger = get_logger(__name__)


class RateLimiter:
    """
    Smart rate limiting for Alpha Vantage 600 calls/minute
    Tech Spec Section 7.2
    """
    
    def __init__(self, calls_per_minute: int = 600, window: int = 60):
        self.calls_per_minute = calls_per_minute
        self.window = window
        self.bucket = calls_per_minute
        self.last_refill = time.time()
        self.remaining = calls_per_minute
        self.reset_time = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self, cost: int = 1):
        """Acquire permission to make API call"""
        async with self._lock:
            # Refill bucket
            now = time.time()
            elapsed = now - self.last_refill
            refill = elapsed * (self.calls_per_minute / self.window)
            self.bucket = min(self.calls_per_minute, self.bucket + refill)
            self.last_refill = now
            
            # Update remaining
            self.remaining = int(self.bucket)
            
            # Check if we can make call
            if self.bucket >= cost:
                self.bucket -= cost
                self.remaining = int(self.bucket)
                return True
            else:
                # Calculate wait time
                wait_time = (cost - self.bucket) / (self.calls_per_minute / self.window)
                self.reset_time = wait_time
                
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                
                # Wait for refill
                await asyncio.sleep(wait_time)
                return await self.acquire(cost)
    
    def check_limit(self) -> bool:
        """Check if we're near the limit"""
        return self.remaining < 50
    
    async def __aenter__(self):
        """Context manager support"""
        await self.acquire()
        return self
    
    async def __aexit__(self, *args):
        pass


# Global rate limiter for Alpha Vantage
av_rate_limiter = RateLimiter(600, 60)
