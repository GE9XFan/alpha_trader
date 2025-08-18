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
            'rsi': self.config.av_config['endpoints'].get('rsi', {}).get('cache_ttl', 60),
            'macd': self.config.av_config['endpoints'].get('macd', {}).get('cache_ttl', 60),
            'bbands': self.config.av_config['endpoints'].get('bbands', {}).get('cache_ttl', 60),
            'vwap': self.config.av_config['endpoints'].get('vwap', {}).get('cache_ttl', 60),
            'atr': self.config.av_config['endpoints'].get('atr', {}).get('cache_ttl', 300)
        }
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        print(f"AV Client initialized with rate limiting and caching")
    
    def _make_cache_key(self, function: str, symbol: str, **kwargs) -> str:
        """Generate a unique cache key including ALL parameters"""
        key_parts = ['av', function.lower(), symbol]
        
        # Sort parameters for consistent ordering
        sorted_params = sorted(kwargs.items())
        
        # Add each non-None parameter to the key
        for param_name, param_value in sorted_params:
            if param_name in ['use_cache', 'self'] or param_value is None:
                continue
            key_parts.append(f"{param_name}={param_value}")
        
        return ':'.join(key_parts)
    
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
            cache_key = self._make_cache_key('historical_options', symbol, date=date_str)
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
            cache_key = self._make_cache_key(
            'rsi', 
            symbol,
            interval=interval,
            time_period=time_period,
            series_type=series_type
        )
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
    
    def get_macd(self, symbol, interval=None, fastperiod=None, slowperiod=None,
             signalperiod=None, series_type=None, use_cache=True):
        """
        Get MACD (Moving Average Convergence Divergence) data for a symbol
        Phase 5.2: Technical indicator with caching
        
        Args:
            symbol: Stock symbol (REQUIRED)
            interval: From config if None
            fastperiod: From config if None  
            slowperiod: From config if None
            signalperiod: From config if None
            series_type: From config if None
            use_cache: Whether to use cache
        """
        # Get MACD configuration
        macd_config = self.config.av_config['endpoints']['macd']
        
        # Use config defaults if not specified
        if interval is None:
            interval = macd_config['default_params']['interval']
        if fastperiod is None:
            fastperiod = macd_config['default_params']['fastperiod']
        if slowperiod is None:
            slowperiod = macd_config['default_params']['slowperiod']
        if signalperiod is None:
            signalperiod = macd_config['default_params']['signalperiod']
        if series_type is None:
            series_type = macd_config['default_params']['series_type']
        
        params = {
            'function': macd_config['function'],
            'symbol': symbol,
            'interval': interval,
            'fastperiod': fastperiod,
            'slowperiod': slowperiod,
            'signalperiod': signalperiod,
            'series_type': series_type,
            'apikey': self.api_key,
            'datatype': macd_config.get('datatype', 'json')
        }
        
        # Generate cache key
        cache_key = None
        cache_ttl = None
        if use_cache:
            cache_key = self._make_cache_key(
            'macd',
            symbol,
            interval=interval,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
            series_type=series_type
        )
            cache_ttl = macd_config.get('cache_ttl', 60)
        
        print(f"Calling MACD for {symbol} ({interval})...")
        data = self._make_request(
            params, 
            f"MACD({symbol}, {interval})",
            cache_key=cache_key,
            cache_ttl=cache_ttl
        )
        
        if not use_cache or not cache_key:
            print(f"✓ Successfully retrieved MACD for {symbol}")
        
        return data

    def get_bbands(self, symbol, interval=None, time_period=None, series_type=None, 
                    nbdevup=None, nbdevdn=None, matype=None, use_cache=True):
        """
        Get Bollinger Bands (BBANDS) data for a symbol
        Phase 5.3: Technical indicator with caching
        
        Args:
            symbol: Stock symbol (REQUIRED)
            interval: Time interval (REQUIRED)
            time_period: Number of data points (REQUIRED)
            series_type: Price type (REQUIRED)
            nbdevup: Upper band std deviations (REQUIRED)
            nbdevdn: Lower band std deviations (REQUIRED)
            matype: Moving average type (REQUIRED)
            use_cache: Whether to use cache
        """
        
        # ADD THIS SECTION - Load defaults from config
        bbands_config = self.config.av_config['endpoints']['bbands']
        
        if interval is None:
            interval = bbands_config['default_params']['interval']
        if time_period is None:
            time_period = bbands_config['default_params']['time_period']
        if series_type is None:
            series_type = bbands_config['default_params']['series_type']
        if nbdevup is None:
            nbdevup = bbands_config['default_params']['nbdevup']
        if nbdevdn is None:
            nbdevdn = bbands_config['default_params']['nbdevdn']
        if matype is None:
            matype = bbands_config['default_params']['matype']

        params = {
            'function': bbands_config['function'],
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type,
            'nbdevup': nbdevup,
            'nbdevdn': nbdevdn,
            'matype': matype,
            'apikey': self.api_key,
            'datatype': bbands_config.get('datatype', 'json')
        }
        
        # Generate cache key
        cache_key = None
        cache_ttl = None
        if use_cache:
            cache_key = self._make_cache_key(
            'bbands',
            symbol,
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype
        )
            cache_ttl = bbands_config.get('cache_ttl', 60)
        
        print(f"Calling BBANDS for {symbol} ({interval})...")
        data = self._make_request(
            params, 
            f"BBANDS({symbol}, {interval})",
            cache_key=cache_key,
            cache_ttl=cache_ttl
        )
        
        if not use_cache or not cache_key:
            print(f"✓ Successfully retrieved BBANDS for {symbol}")
        
        return data

    def get_vwap(self, symbol, interval=None, use_cache=True):
        """Get VWAP data from Alpha Vantage"""
        # Get config - NO HARDCODING
        vwap_config = self.config.av_config['endpoints']['vwap']
        
        # Use config default if not specified
        if interval is None:
            interval = vwap_config['default_params']['interval']
        
        # Build params
        params = {
            'function': vwap_config['function'],
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'datatype': vwap_config['default_params'].get('datatype', 'json')
        }
        
        # Generate cache key using standard pattern
        cache_key = None
        cache_ttl = None
        if use_cache:
            cache_key = self._make_cache_key('vwap', symbol, interval=interval)
            cache_ttl = vwap_config.get('cache_ttl', 60)
        
        # Make request using standard pattern with cache support
        print(f"Fetching VWAP for {symbol} with interval={interval}")
        response = self._make_request(
            params,
            f"VWAP({symbol}, {interval})",
            cache_key=cache_key,
            cache_ttl=cache_ttl
        )
        
        # Keep the existing validation logic
        if response and 'Technical Analysis: VWAP' in response:
            data_points = len(response.get('Technical Analysis: VWAP', {}))
            print(f"Fetched {data_points} VWAP data points for {symbol}")
        
        return response

    def get_adx(self, symbol: str, interval: str = None, time_period: int = None) -> dict:
        """
        Get Average Directional Index (ADX) data - Phase 5.6
        
        ADX measures trend strength regardless of direction.
        Values > 25 indicate strong trend, > 50 very strong.
        
        Args:
            symbol: Stock symbol (from config if not provided)
            interval: Time interval - typically '5min' for ADX
            time_period: Number of periods for ADX calculation (default 14)
        
        Returns:
            API response dict or cached data
        """
        # Get ADX config 
        adx_config = self.config.av_config['endpoints']['adx']
        default_params = adx_config['default_params']
        
        # Use provided values or fall back to config
        if interval is None:
            interval = default_params.get('interval', '5min')
        if time_period is None:
            time_period = default_params.get('time_period', 14)
        
        # Check cache first
        cache_key = self._make_cache_key(
            'adx',
            symbol,
            interval=interval,
            time_period=time_period
        )
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            print(f"  📊 ADX {symbol}: Cache hit (trend strength)")
            return cached_data
        
        # Rate limit check
        self.rate_limiter.acquire()
        
        # Prepare API parameters
        params = {
            'function': adx_config['function'],
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'apikey': self.api_key,
            'datatype': adx_config.get('datatype', 'json')
        }
        
        try:
            # Make API call
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                print(f"  ❌ ADX API Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                print(f"  ⚠️ Rate limit warning: {data['Note']}")
                return None
            
            # Validate we got ADX data
            if 'Technical Analysis: ADX' not in data:
                print(f"  ❌ No ADX data in response for {symbol}")
                return None
            
            # Cache successful response
            cache_ttl = adx_config.get('cache_ttl', 300)
            self.cache.set(cache_key, data, ttl=cache_ttl)
            
            # Count data points for logging
            adx_data = data.get('Technical Analysis: ADX', {})
            print(f"  ✅ ADX {symbol}: {len(adx_data)} data points fetched")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ ADX request failed for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Unexpected error getting ADX for {symbol}: {e}")
            return None

    def get_atr(self, symbol, interval=None, time_period=None):
            """
            Get Average True Range (ATR) indicator
            Phase 5.5 - Day 22: Volatility indicator
            
            ATR measures market volatility by decomposing the entire range of an asset
            for that period. Used primarily for position sizing and stop loss placement.
            
            Args:
                symbol: Stock symbol (from config if not provided)
                interval: Time interval - typically 'daily' for ATR
                time_period: Number of periods for ATR calculation (default 14)
            
            Returns:
                API response dict or cached data
            """
            # Get ATR config - NO HARDCODED VALUES!
            atr_config = self.config.av_config['endpoints']['atr']
            default_params = atr_config['default_params']
            
            # Use provided values or fall back to config
            if interval is None:
                interval = default_params.get('interval', 'daily')
            if time_period is None:
                time_period = default_params.get('time_period', 14)
            
            # Check cache first - ATR uses longer TTL since it's daily data
            cache_key = self._make_cache_key(
            'atr',
            symbol,
            interval=interval,
            time_period=time_period
        )
            cached_data = self.cache.get(cache_key)
            
            if cached_data:
                print(f"  📊 ATR {symbol}: Cache hit (daily volatility)")
                return cached_data
            
            # Rate limit check
            self.rate_limiter.acquire()
            
            # Prepare API parameters
            params = {
                'function': atr_config['function'],
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'apikey': self.api_key,
                'datatype': atr_config.get('datatype', 'json')
            }
            
            try:
                # Make API call
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    print(f"  ❌ ATR API Error: {data['Error Message']}")
                    return None
                
                if 'Note' in data:
                    print(f"  ⚠️ Rate limit warning: {data['Note']}")
                    return None
                
                # Validate we got ATR data
                if 'Technical Analysis: ATR' not in data:
                    print(f"  ❌ No ATR data in response for {symbol}")
                    return None
                
                # Cache successful response with longer TTL for daily data
                cache_ttl = atr_config.get('cache_ttl', 300)
                self.cache.set(cache_key, data, ttl=cache_ttl)
                
                # Count data points for logging
                atr_points = len(data.get('Technical Analysis: ATR', {}))
                print(f"  📊 ATR {symbol}: {atr_points} daily volatility points fetched")
                
                return data
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ ATR request failed: {str(e)[:100]}")
                return None
            except Exception as e:
                print(f"  ❌ Unexpected error in get_atr: {str(e)[:100]}")
                return None    

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