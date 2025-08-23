"""
Production Redis cache management
Real connections, real operations, circuit breaker pattern
Zero hardcoded values - all from environment
"""
import os
import json
import time
import redis
from typing import Any, Optional, Dict, List
from datetime import datetime

from .logger import get_logger
from .exceptions import CacheException, CircuitBreakerException


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    All thresholds from environment - no hardcoding
    """
    
    def __init__(self, name: str):
        """Initialize circuit breaker with environment config"""
        self.name = name
        self.failure_threshold = int(os.environ['CB_FAILURE_THRESHOLD'])
        self.recovery_timeout = int(os.environ['CB_RECOVERY_TIMEOUT'])
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        
        self.logger = get_logger(__name__)
    
    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerException: If circuit is open
        """
        # Check if circuit should be reset
        if self.state == 'open':
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = 'half-open'
                self.logger.info(f"Circuit breaker {self.name} moved to half-open")
            else:
                raise CircuitBreakerException(
                    f"Circuit breaker {self.name} is open",
                    service=self.name,
                    recovery_time=self.recovery_timeout
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in half-open
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.name} closed")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                self.logger.error(
                    f"Circuit breaker {self.name} opened",
                    failures=self.failure_count,
                    threshold=self.failure_threshold
                )
            
            raise


class CacheManager:
    """
    Real Redis connection and operations
    All configuration from environment - zero hardcoding
    """
    
    def __init__(self):
        """Initialize Redis client with environment configuration"""
        self.logger = get_logger(__name__)
        self.circuit_breaker = CircuitBreaker('redis')
        
        # Get ALL configuration from environment - NO DEFAULTS
        self.config = {
            'host': os.environ['REDIS_HOST'],
            'port': int(os.environ['REDIS_PORT']),
            'db': int(os.environ['REDIS_DB']),
            'password': os.environ.get('REDIS_PASSWORD') or None,
            'socket_timeout': int(os.environ['REDIS_SOCKET_TIMEOUT']),
            'socket_connect_timeout': int(os.environ['REDIS_SOCKET_CONNECT_TIMEOUT']),
            'socket_keepalive': os.environ['REDIS_SOCKET_KEEPALIVE'].lower() == 'true',
            'max_connections': int(os.environ['REDIS_MAX_CONNECTIONS']),
            'decode_responses': os.environ['REDIS_DECODE_RESPONSES'].lower() == 'true',
            'retry_on_timeout': os.environ['REDIS_RETRY_ON_TIMEOUT'].lower() == 'true',
        }
        
        # Performance tracking from environment
        self.slow_cache_threshold = float(os.environ['PERF_SLOW_CACHE_THRESHOLD_MS'])
        
        self.redis_client = self._create_client()
        self._test_connection()
    
    def _create_client(self) -> redis.Redis:
        """Create REAL Redis connection"""
        self.logger.info(
            "Creating Redis connection",
            host=self.config['host'],
            port=self.config['port'],
            db=self.config['db']
        )
        
        # Create connection pool
        pool = redis.ConnectionPool(
            host=self.config['host'],
            port=self.config['port'],
            db=self.config['db'],
            password=self.config['password'],
            socket_timeout=self.config['socket_timeout'],
            socket_connect_timeout=self.config['socket_connect_timeout'],
            socket_keepalive=self.config['socket_keepalive'],
            max_connections=self.config['max_connections'],
            decode_responses=self.config['decode_responses'],
            retry_on_timeout=self.config['retry_on_timeout']
        )
        
        return redis.Redis(connection_pool=pool)
    
    def _test_connection(self):
        """Test Redis connectivity"""
        try:
            self.redis_client.ping()
            self.logger.info("Redis connection established successfully")
        except redis.RedisError as e:
            raise CacheException(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from REAL Redis instance
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        try:
            start_time = time.time()
            
            # Use circuit breaker
            value = self.circuit_breaker.call(self.redis_client.get, key)
            
            # Track performance
            duration_ms = (time.time() - start_time) * 1000
            
            if value:
                self.logger.debug(f"Cache hit", key=key, duration_ms=duration_ms)
                # Deserialize JSON if we got a string
                if isinstance(value, (str, bytes)):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return value
            else:
                self.logger.debug(f"Cache miss", key=key, duration_ms=duration_ms)
                return None
                
        except CircuitBreakerException:
            # Circuit is open, return None
            self.logger.warning(f"Circuit breaker open, cache get failed", key=key)
            return None
            
        except redis.RedisError as e:
            self.logger.error(f"Cache get failed", key=key, error=str(e))
            raise CacheException(f"Failed to get {key}: {e}")
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in REAL Redis instance
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            start_time = time.time()
            
            # Serialize value to JSON
            serialized = json.dumps(value) if not isinstance(value, (str, bytes)) else value
            
            # Use circuit breaker
            if ttl:
                result = self.circuit_breaker.call(
                    self.redis_client.setex,
                    key, ttl, serialized
                )
            else:
                result = self.circuit_breaker.call(
                    self.redis_client.set,
                    key, serialized
                )
            
            # Track performance
            duration_ms = (time.time() - start_time) * 1000
            
            if duration_ms > self.slow_cache_threshold:
                self.logger.warning(
                    "Slow cache operation",
                    operation="set",
                    key=key,
                    duration_ms=duration_ms
                )
            
            return bool(result)
            
        except CircuitBreakerException:
            self.logger.warning(f"Circuit breaker open, cache set failed", key=key)
            return False
            
        except redis.RedisError as e:
            self.logger.error(f"Cache set failed", key=key, error=str(e))
            raise CacheException(f"Failed to set {key}: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        try:
            result = self.circuit_breaker.call(self.redis_client.delete, key)
            return bool(result)
            
        except CircuitBreakerException:
            self.logger.warning(f"Circuit breaker open, cache delete failed", key=key)
            return False
            
        except redis.RedisError as e:
            self.logger.error(f"Cache delete failed", key=key, error=str(e))
            raise CacheException(f"Failed to delete {key}: {e}")
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        try:
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            
            values = self.circuit_breaker.call(pipe.execute)
            
            results = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        results[key] = json.loads(value) if isinstance(value, (str, bytes)) else value
                    except json.JSONDecodeError:
                        results[key] = value
            
            return results
            
        except CircuitBreakerException:
            self.logger.warning("Circuit breaker open, batch get failed")
            return {}
            
        except redis.RedisError as e:
            self.logger.error(f"Batch get failed", error=str(e))
            raise CacheException(f"Failed to get multiple keys: {e}")
    
    def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple values in cache
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            True if all successful
        """
        try:
            pipe = self.redis_client.pipeline()
            
            for key, value in items.items():
                serialized = json.dumps(value) if not isinstance(value, (str, bytes)) else value
                if ttl:
                    pipe.setex(key, ttl, serialized)
                else:
                    pipe.set(key, serialized)
            
            results = self.circuit_breaker.call(pipe.execute)
            return all(results)
            
        except CircuitBreakerException:
            self.logger.warning("Circuit breaker open, batch set failed")
            return False
            
        except redis.RedisError as e:
            self.logger.error(f"Batch set failed", error=str(e))
            raise CacheException(f"Failed to set multiple keys: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Test REAL Redis connectivity
        
        Returns:
            Health check results
        """
        try:
            start_time = time.time()
            
            # Ping Redis
            result = self.redis_client.ping()
            
            # Get info
            info = self.redis_client.info()
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'healthy': result,
                'response_time_ms': duration_ms,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'circuit_breaker_state': self.circuit_breaker.state,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                'healthy': False,
                'error': str(e),
                'circuit_breaker_state': self.circuit_breaker.state,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            self.logger.info("Redis connection closed")