"""
Cache Manager - Tech Spec Section 7.1
Multi-tier caching for Alpha Vantage data
"""
import asyncio
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import json

from src.core.logger import get_logger
from src.data.database import db


logger = get_logger(__name__)


class CacheManager:
    """
    Multi-tier caching for Alpha Vantage data
    Tech Spec Section 7.1
    """
    
    def __init__(self):
        self.l1_cache = {}  # In-memory (microseconds)
        self.l2_cache = db.redis  # Redis (milliseconds)
        self.l3_cache = db  # Database (for historical)
        
        # Metrics
        self.hits = 0
        self.misses = 0
    
    async def get_with_cache(self, key: str, fetch_func: Callable, 
                            ttl: int = 60) -> Any:
        """Get with multi-tier cache"""
        # L1: Check memory
        if key in self.l1_cache:
            if datetime.now() < self.l1_cache[key]['expires']:
                self.hits += 1
                return self.l1_cache[key]['data']
        
        # L2: Check Redis
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value:
                self.hits += 1
                data = json.loads(value)
                self.l1_cache[key] = {
                    'data': data,
                    'expires': datetime.now() + timedelta(seconds=ttl)
                }
                return data
        
        # L3: Fetch from API
        self.misses += 1
        value = await fetch_func()
        
        # Cache in all tiers
        self.l1_cache[key] = {
            'data': value,
            'expires': datetime.now() + timedelta(seconds=ttl)
        }
        if self.l2_cache:
            self.l2_cache.setex(key, ttl, json.dumps(value))
        
        return value
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # Clear L1
        keys_to_remove = [k for k in self.l1_cache if pattern in k]
        for key in keys_to_remove:
            del self.l1_cache[key]
        
        # Clear L2
        if self.l2_cache:
            for key in self.l2_cache.scan_iter(match=f"*{pattern}*"):
                self.l2_cache.delete(key)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


cache_manager = CacheManager()
