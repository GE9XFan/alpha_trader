#!/usr/bin/env python3
"""
Redis Cache Manager for Options Trading System
Handles all caching with proper TTL management
"""

import json
import redis
import yaml
import os
from typing import Any, Optional, Dict, List
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CacheManager:
    """Redis cache manager with TTL-based expiration"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize cache manager with config"""
        self.config = self._load_config(config_path)
        self.redis_client = self._connect_redis()
        self.ttl_config = self.config['cache']['ttl']

        # Track cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }

        logger.info(f"Cache manager initialized with Redis at {self.config['cache']['host']}:{self.config['cache']['port']}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Try from parent directory if running from scripts/
            config_file = Path(__file__).parent.parent / config_path
            if not config_file.exists():
                logger.error(f"Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Substitute environment variables
        config = self._substitute_env_vars(config)
        return config

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Parse ${VAR:default} format
            var_expr = config[2:-1]
            if ':' in var_expr:
                var_name, default = var_expr.split(':', 1)
                value = os.getenv(var_name, default)
            else:
                value = os.getenv(var_expr, config)

            # Try to convert to appropriate type
            if value.isdigit():
                return int(value)
            elif value.replace('.', '', 1).isdigit():
                return float(value)
            elif value.lower() in ('true', 'yes'):
                return True
            elif value.lower() in ('false', 'no'):
                return False
            return value
        return config

    def _connect_redis(self) -> redis.Redis:
        """Connect to Redis with configuration"""
        try:
            client = redis.Redis(
                host=self.config['cache']['host'],
                port=self.config['cache']['port'],
                db=self.config['cache']['db'],
                password=self.config['cache'].get('password'),
                decode_responses=True,
                max_connections=self.config['cache'].get('max_connections', 100),
                socket_keepalive=self.config['cache'].get('socket_keepalive', True),
                socket_connect_timeout=self.config['cache'].get('socket_connect_timeout', 5)
            )

            # Test connection
            client.ping()

            # Set max memory if configured
            max_memory = self.config['cache'].get('max_memory')
            if max_memory:
                try:
                    client.config_set('maxmemory', max_memory)
                    client.config_set('maxmemory-policy',
                                     self.config['cache'].get('max_memory_policy', 'volatile-lru'))
                except redis.ResponseError:
                    logger.warning("Could not set Redis memory config (may require admin rights)")

            return client

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _get_ttl(self, data_type: str) -> int:
        """Get TTL for data type from config"""
        return self.ttl_config.get(data_type, 60)  # Default 60 seconds

    def _make_key(self, prefix: str, identifier: str) -> str:
        """Create consistent cache key"""
        return f"{prefix}:{identifier}"

    # Order Book Methods
    def set_order_book(self, symbol: str, order_book: Dict) -> bool:
        """Cache order book with 1 second TTL"""
        try:
            key = self._make_key("orderbook", symbol)
            ttl = self._get_ttl("order_book")

            self.redis_client.setex(
                key,
                ttl,
                json.dumps(order_book)
            )

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to cache order book for {symbol}: {e}")
            return False

    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get cached order book"""
        try:
            key = self._make_key("orderbook", symbol)
            data = self.redis_client.get(key)

            if data:
                self.stats['hits'] += 1
                return json.loads(str(data))  # Ensure string type

            self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None

    # Options Chain Methods
    def set_options_chain(self, symbol: str, chain: Dict) -> bool:
        """Cache options chain with 10 second TTL"""
        try:
            key = self._make_key("options", symbol)
            ttl = self._get_ttl("options_chain")

            self.redis_client.setex(
                key,
                ttl,
                json.dumps(chain)
            )

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to cache options chain for {symbol}: {e}")
            return False

    def get_options_chain(self, symbol: str) -> Optional[Dict]:
        """Get cached options chain"""
        try:
            key = self._make_key("options", symbol)
            data = self.redis_client.get(key)

            if data:
                self.stats['hits'] += 1
                return json.loads(str(data))  # Ensure string type

            self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Failed to get options chain for {symbol}: {e}")
            return None

    # Greeks Methods (separate from chain for granular caching)
    def set_greeks(self, symbol: str, strike: float, expiry: str, greeks: Dict) -> bool:
        """Cache option Greeks with 10 second TTL"""
        try:
            key = self._make_key("greeks", f"{symbol}_{strike}_{expiry}")
            ttl = self._get_ttl("greeks")

            self.redis_client.setex(
                key,
                ttl,
                json.dumps(greeks)
            )

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to cache Greeks for {symbol}: {e}")
            return False

    # Market Metrics Methods
    def set_metrics(self, symbol: str, metrics: Dict) -> bool:
        """Cache calculated metrics with 5 second TTL"""
        try:
            key = self._make_key("metrics", symbol)
            ttl = self._get_ttl("calculated_metrics")

            # Add timestamp
            metrics['cached_at'] = datetime.now().isoformat()

            self.redis_client.setex(
                key,
                ttl,
                json.dumps(metrics)
            )

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to cache metrics for {symbol}: {e}")
            return False

    def get_metrics(self, symbol: str) -> Optional[Dict]:
        """Get cached metrics"""
        try:
            key = self._make_key("metrics", symbol)
            data = self.redis_client.get(key)

            if data:
                self.stats['hits'] += 1
                return json.loads(str(data))  # Ensure string type

            self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Failed to get metrics for {symbol}: {e}")
            return None

    # Technical Indicators Methods
    def set_indicator(self, symbol: str, indicator_name: str, data: Dict) -> bool:
        """Cache technical indicator with 60 second TTL"""
        try:
            key = self._make_key("indicator", f"{symbol}_{indicator_name}")
            ttl = self._get_ttl("technical_indicators")

            self.redis_client.setex(
                key,
                ttl,
                json.dumps(data)
            )

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to cache {indicator_name} for {symbol}: {e}")
            return False

    def get_indicator(self, symbol: str, indicator_name: str) -> Optional[Dict]:
        """Get cached technical indicator"""
        try:
            key = self._make_key("indicator", f"{symbol}_{indicator_name}")
            data = self.redis_client.get(key)

            if data:
                self.stats['hits'] += 1
                return json.loads(str(data))  # Ensure string type

            self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Failed to get {indicator_name} for {symbol}: {e}")
            return None

    # Sentiment Methods
    def set_sentiment(self, symbol: str, sentiment: Dict) -> bool:
        """Cache sentiment data with 5 minute TTL"""
        try:
            key = self._make_key("sentiment", symbol)
            ttl = self._get_ttl("sentiment")

            self.redis_client.setex(
                key,
                ttl,
                json.dumps(sentiment)
            )

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to cache sentiment for {symbol}: {e}")
            return False

    # Trade/Bar Data Methods
    def append_trade(self, symbol: str, trade: Dict) -> bool:
        """Append trade to recent trades list (keep last 1000)"""
        try:
            key = self._make_key("trades", symbol)

            # Add to list
            self.redis_client.lpush(key, json.dumps(trade))

            # Trim to last 1000
            self.redis_client.ltrim(key, 0, 999)

            # Set expiry on the list
            self.redis_client.expire(key, self._get_ttl("trades"))

            return True

        except Exception as e:
            logger.error(f"Failed to append trade for {symbol}: {e}")
            return False

    def get_recent_trades(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get recent trades"""
        try:
            key = self._make_key("trades", symbol)
            trades = self.redis_client.lrange(key, 0, count - 1)

            if trades:
                self.stats['hits'] += 1
                return [json.loads(str(t)) for t in trades]  # type: ignore

            self.stats['misses'] += 1
            return []

        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            return []

    # VPIN Cache (very short TTL)
    def set_vpin(self, symbol: str, vpin: float) -> bool:
        """Cache VPIN score with 1 second TTL"""
        try:
            key = self._make_key("vpin", symbol)
            ttl = self._get_ttl("vpin")

            self.redis_client.setex(
                key,
                ttl,
                str(vpin)
            )

            return True

        except Exception as e:
            logger.error(f"Failed to cache VPIN for {symbol}: {e}")
            return False

    def get_vpin(self, symbol: str) -> Optional[float]:
        """Get cached VPIN score"""
        try:
            key = self._make_key("vpin", symbol)
            data = self.redis_client.get(key)

            if data:
                self.stats['hits'] += 1
                return float(str(data))  # Ensure string type before float conversion

            self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Failed to get VPIN for {symbol}: {e}")
            return None

    # Fundamentals Methods
    def set_fundamentals(self, symbol: str, fundamentals: Dict) -> bool:
        """Cache fundamental data with 1 hour TTL"""
        try:
            key = self._make_key("fundamentals", symbol)
            ttl = self._get_ttl("fundamentals")

            self.redis_client.setex(
                key,
                ttl,
                json.dumps(fundamentals)
            )

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to cache fundamentals for {symbol}: {e}")
            return False

    # Generic Methods
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Generic set with optional TTL"""
        try:
            if ttl:
                self.redis_client.setex(key, ttl, json.dumps(value))
            else:
                self.redis_client.set(key, json.dumps(value))

            self.stats['sets'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to set {key}: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Generic get"""
        try:
            data = self.redis_client.get(key)

            if data:
                self.stats['hits'] += 1
                return json.loads(str(data))  # Ensure string type

            self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Failed to get {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key"""
        try:
            result = self.redis_client.delete(key)
            self.stats['deletes'] += 1
            return bool(result)

        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            # Get keys matching pattern
            keys_result = self.redis_client.keys(pattern)  
            keys = list(keys_result) if keys_result else []  # type: ignore
            
            if keys:
                deleted = self.redis_client.delete(*keys)  
                deleted_count = int(deleted) if deleted else 0  # type: ignore
                self.stats['deletes'] += deleted_count
                return deleted_count
            return 0

        except Exception as e:
            logger.error(f"Failed to flush pattern {pattern}: {e}")
            return 0

    # Statistics Methods
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_ops = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_ops * 100) if total_ops > 0 else 0

        # Get Redis info
        try:
            info_result = self.redis_client.info('memory')  # type: ignore
            # Handle both dict and ResponseT types
            if hasattr(info_result, '__getitem__'):
                memory_used = info_result.get('used_memory_human', 'N/A')  # type: ignore
                memory_peak = info_result.get('used_memory_peak_human', 'N/A')  # type: ignore
            else:
                memory_used = 'N/A'
                memory_peak = 'N/A'
                
            db_size = self.redis_client.dbsize()  # type: ignore
            keys_count = int(db_size) if db_size else 0  # type: ignore
        except:
            memory_used = 'N/A'
            memory_peak = 'N/A'
            keys_count = 0

        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'deletes': self.stats['deletes'],
            'hit_rate': f"{hit_rate:.2f}%",
            'memory_used': memory_used,
            'memory_peak': memory_peak,
            'keys': keys_count
        }

    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            return bool(self.redis_client.ping())  # Ensure bool type
        except:
            return False

    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Cache manager connection closed")
