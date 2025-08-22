#!/usr/bin/env python3
"""
Cache Manager - Redis cache management
Phase 0: Foundation Setup
"""

import json
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta

import redis
from redis.exceptions import RedisError

from src.foundation.config_manager import get_config_manager


class CacheManager:
    """Manages Redis cache operations"""
    
    def __init__(self):
        """Initialize cache manager"""
        self.config = get_config_manager()
        self.redis_config = self.config.redis_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create Redis client
        self.client = self._create_client()
        
        # Test connection
        self._test_connection()
        
        # Load TTL settings
        self.default_ttls = self.redis_config.get('default_ttl', {})
        self.key_prefixes = self.redis_config.get('key_prefixes', {})
        
        self.logger.info("CacheManager initialized successfully")
    
    def _create_client(self) -> redis.Redis:
        """Create Redis client with connection pool"""
        connection_params = self.config.get_redis_connection_params()
        
        # Create connection pool
        pool = redis.ConnectionPool(
            **connection_params,
            max_connections=self.redis_config.get('connection_pool', {}).get('max_connections', 50)
        )
        
        client = redis.Redis(connection_pool=pool)
        
        self.logger.info(f"Created Redis client connected to {connection_params['host']}:{connection_params['port']}")
        return client
    
    def _test_connection(self):
        """Test Redis connection"""
        try:
            self.client.ping()
            self.logger.info("Redis connection test successful")
        except RedisError as e:
            self.logger.error(f"Redis connection test failed: {e}")
            raise
    
    def _get_key(self, key: str, prefix: Optional[str] = None) -> str:
        """
        Get cache key with optional prefix
        
        Args:
            key: Base key
            prefix: Optional prefix (av, ibkr, feature, model, signal)
            
        Returns:
            Prefixed key
        """
        if prefix and prefix in self.key_prefixes:
            return f"{self.key_prefixes[prefix]}{key}"
        return key
    
    def get(self, key: str, prefix: Optional[str] = None) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            prefix: Optional key prefix
            
        Returns:
            Cached value or None
        """
        try:
            full_key = self._get_key(key, prefix)
            value = self.client.get(full_key)
            
            if value:
                # Try to deserialize JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            
            return None
            
        except RedisError as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            prefix: Optional[str] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            prefix: Optional key prefix
            
        Returns:
            True if successful
        """
        try:
            full_key = self._get_key(key, prefix)
            
            # Serialize to JSON if needed
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if ttl:
                return self.client.setex(full_key, ttl, value)
            else:
                return self.client.set(full_key, value)
                
        except RedisError as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str, prefix: Optional[str] = None) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key
            prefix: Optional key prefix
            
        Returns:
            True if key was deleted
        """
        try:
            full_key = self._get_key(key, prefix)
            return bool(self.client.delete(full_key))
        except RedisError as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str, prefix: Optional[str] = None) -> bool:
        """
        Check if key exists
        
        Args:
            key: Cache key
            prefix: Optional key prefix
            
        Returns:
            True if key exists
        """
        try:
            full_key = self._get_key(key, prefix)
            return bool(self.client.exists(full_key))
        except RedisError as e:
            self.logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def get_ttl(self, key: str, prefix: Optional[str] = None) -> int:
        """
        Get remaining TTL for key
        
        Args:
            key: Cache key
            prefix: Optional key prefix
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            full_key = self._get_key(key, prefix)
            return self.client.ttl(full_key)
        except RedisError as e:
            self.logger.error(f"Cache TTL error for key {key}: {e}")
            return -2
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Key pattern (e.g., "av:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except RedisError as e:
            self.logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary of cache stats
        """
        try:
            info = self.client.info('stats')
            memory = self.client.info('memory')
            
            return {
                'total_connections_received': info.get('total_connections_received', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'used_memory_human': memory.get('used_memory_human', 'N/A'),
                'used_memory_peak_human': memory.get('used_memory_peak_human', 'N/A'),
                'connected_clients': self.client.info('clients').get('connected_clients', 0),
                'total_keys': self.client.dbsize()
            }
        except RedisError as e:
            self.logger.error(f"Cache stats error: {e}")
            return {}
    
    def flush(self):
        """Flush all keys from current database"""
        try:
            self.client.flushdb()
            self.logger.warning("Cache flushed - all keys deleted")
        except RedisError as e:
            self.logger.error(f"Cache flush error: {e}")
    
    def close(self):
        """Close Redis connection"""
        self.client.close()
        self.logger.info("Redis connection closed")


# Singleton instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get or create singleton CacheManager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager