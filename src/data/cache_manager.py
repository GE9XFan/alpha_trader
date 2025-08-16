"""Redis cache manager for market data - Phase 4.1"""

import redis
import json
import os
from typing import Optional, Any
from pathlib import Path
import yaml

class CacheManager:
    """
    Manages Redis cache for market data
    Phase 4: Simple cache with TTL support
    """
    
    def __init__(self):
        # Load Redis URL from environment
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Load cache configuration
        config_path = Path(__file__).parent.parent.parent / 'config' / 'system' / 'redis.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create Redis connection
        self.redis_client = redis.from_url(
            redis_url,
            decode_responses=True  # Get strings instead of bytes
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            print("✓ Redis cache connected")
        except redis.ConnectionError as e:
            print(f"✗ Redis connection failed: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        Returns None if key doesn't exist or expired
        """
        try:
            value = self.redis_client.get(key)
            if value:
                # Try to parse as JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Return as string if not JSON
                    return value
            return None
        except Exception as e:
            print(f"Cache get error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if dict/list)
            ttl: Time to live in seconds (None = no expiry)
        """
        try:
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if ttl:
                return self.redis_client.setex(key, ttl, value)
            else:
                return self.redis_client.set(key, value)
        except Exception as e:
            print(f"Cache set error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            print(f"Cache delete error for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            print(f"Cache exists error for {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key (-1 if no expiry, -2 if doesn't exist)"""
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            print(f"Cache TTL error for {key}: {e}")
            return -2
    
    def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern (e.g., 'options:SPY:*')"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache flush error for pattern {pattern}: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            info = self.redis_client.info('stats')
            return {
                'total_connections': info.get('total_connections_received', 0),
                'commands_processed': info.get('total_commands_processed', 0),
                'keys': self.redis_client.dbsize(),
                'used_memory': self.redis_client.info('memory').get('used_memory_human', 'Unknown')
            }
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {}


# Global cache instance (singleton)
_cache = None

def get_cache():
    """Get or create the global cache instance"""
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache