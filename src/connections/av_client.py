"""Alpha Vantage client with rate limiting and caching - Phase 4.1"""

import requests
import json
from pathlib import Path
import sys
import time
import hashlib

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.foundation.config_manager import ConfigManager
from src.data.rate_limiter import get_rate_limiter
from src.data.cache_manager import get_cache


class AlphaVantageClient:
    def __init__(self):
        self.config = ConfigManager()
        self.api_key = self.config.av_api_key
        self.base_url = self.config.av_config.get('base_url')
        self.timeout = self.config.av_config.get('timeout', 30)
        
        # Get the global rate limiter
        self.rate_limiter = get_rate_limiter()
        
        # Get the global cache instance
        self.cache = get_cache()
        
        # Cache TTL from config
        self.cache_ttl = {
            'realtime_options': self.config.av_config['endpoints'].get('realtime_options', {}).get('cache_ttl', 30),
            'historical_options': self.config.av_config['endpoints'].get('historical_options', {}).get('cache_ttl', 86400),
            'rsi': self.config.av_config['endpoints'].get('rsi', {}).get('cache_ttl', 60)
        }
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        print(f"AV Client initialized with rate limiting and caching")
    
    def _make_cache_key(self, function: str, symbol: str, extra: str = "") -> str:
        """Generate a consistent cache key"""
        # Create a unique key for this API call
        if extra:
            return f"av:{function}:{symbol}:{extra}"
        return f"av:{function}:{symbol}"
    
    def _make_request(self, params, description="API call", cache_key=None, cache_ttl=None):
        """
        Make a rate-limited, cached API request
        Phase 4: Check cache first, then make API call if needed
        """
        # Check cache first if cache_key provided
        if cache_key:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                print(f"✓ Cache hit for {description}")
                return cached_data
            else:
                print(f"Cache miss for {description}, calling API...")
        
        # Acquire token from rate limiter
        wait_time = self.rate_limiter.wait_time()
        if wait_time > 0:
            print(f"Rate limit: waiting {wait_time:.1f}s...")
        
        if not self.rate_limiter.acquire(blocking=True, timeout=30):
            raise Exception(f"Rate limit timeout for {description}")
        
        # Make the actual request
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                # This is the rate limit message from Alpha Vantage
                raise Exception(f"API Rate Limit Hit: {data['Note']}")
            
            # Cache the successful response if cache_key provided
            if cache_key and cache_ttl:
                self.cache.set(cache_key, data, ttl=cache_ttl)
                print(f"✓ Cached {description} for {cache_ttl} seconds")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            raise
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def get_realtime_options(self, symbol, use_cache=True):
            """
            Get real-time options data for a symbol
            Phase 4: Now with caching support
            
            Args:
                symbol: Stock symbol (REQUIRED)
                use_cache: Whether to use cache
            """
            endpoint_config = self.config.av_config['endpoints']['realtime_options']
            
            params = {
                'function': endpoint_config['function'],
                'symbol': symbol,
                'apikey': self.api_key,
                'datatype': endpoint_config.get('datatype', 'json')
            }
            
            if endpoint_config.get('require_greeks'):
                params['require_greeks'] = endpoint_config['require_greeks']
            
            # Generate cache key
            cache_key = None
            cache_ttl = None
            if use_cache:
                cache_key = self._make_cache_key('realtime_options', symbol)
                cache_ttl = self.cache_ttl['realtime_options']
            
            print(f"Calling REALTIME_OPTIONS for {symbol}...")
            data = self._make_request(
                params, 
                f"REALTIME_OPTIONS({symbol})",
                cache_key=cache_key,
                cache_ttl=cache_ttl
            )
            
            if not use_cache or not cache_key:
                print(f"✓ Successfully retrieved options data for {symbol}")
            
            return data
    
    def get_historical_options(self, symbol, date=None, use_cache=True):
        """
        Get historical options data for a symbol
        Phase 4: Now with caching support
        
        Args:
            symbol: Stock symbol (REQUIRED)
            date: Optional date
            use_cache: Whether to use cache
        """
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'apikey': self.api_key,
            'datatype': 'json'
        }
        
        if date:
            params['date'] = date
        
        # Generate cache key
        cache_key = None
        cache_ttl = None
        if use_cache:
            date_str = date if date else 'latest'
            cache_key = self._make_cache_key('historical_options', symbol, date_str)
            cache_ttl = self.cache_ttl['historical_options']
        
        print(f"Calling HISTORICAL_OPTIONS for {symbol}...")
        data = self._make_request(
            params, 
            f"HISTORICAL_OPTIONS({symbol})",
            cache_key=cache_key,
            cache_ttl=cache_ttl
        )
        
        if not use_cache or not cache_key:
            print(f"✓ Successfully retrieved historical options for {symbol}")
        
        return data

    def get_rsi(self, symbol, interval=None, time_period=None, 
                series_type=None, use_cache=True):
        """
        Get RSI (Relative Strength Index) data for a symbol
        Phase 5.1: Technical indicator with caching
        
        Args:
            symbol: Stock symbol (REQUIRED)
            interval: From config if None
            time_period: From config if None  
            series_type: From config if None
            use_cache: Whether to use cache
        """
        # Get RSI configuration
        rsi_config = self.config.av_config['endpoints']['rsi']
        
        # Use config defaults if not specified
        if interval is None:
            interval = rsi_config['default_params']['interval']
        if time_period is None:
            time_period = rsi_config['default_params']['time_period']
        if series_type is None:
            series_type = rsi_config['default_params']['series_type']
        
        params = {
            'function': rsi_config['function'],
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type,
            'apikey': self.api_key,
            'datatype': rsi_config.get('datatype', 'json')
        }
        
        # Generate cache key
        cache_key = None
        cache_ttl = None
        if use_cache:
            cache_key = self._make_cache_key('rsi', symbol, f"{interval}_{time_period}")
            cache_ttl = rsi_config.get('cache_ttl', 60)
        
        print(f"Calling RSI for {symbol} ({interval})...")
        data = self._make_request(
            params, 
            f"RSI({symbol}, {interval})",
            cache_key=cache_key,
            cache_ttl=cache_ttl
        )
        
        if not use_cache or not cache_key:
            print(f"✓ Successfully retrieved RSI for {symbol}")
        
        return data    
    def get_rate_limit_status(self):
        """Get current rate limit statistics"""
        return self.rate_limiter.get_stats()
    
    def get_cache_status(self):
        """Get cache statistics"""
        stats = self.cache.get_stats()
        
        # Count AV-specific keys
        av_keys = len(self.cache.redis_client.keys("av:*"))
        stats['av_keys'] = av_keys
        
        return stats


if __name__ == "__main__":
    # Quick test
    client = AlphaVantageClient()
    print(f"Rate limit status: {client.get_rate_limit_status()}")
    print(f"Cache status: {client.get_cache_status()}")