"""Alpha Vantage client with rate limiting - Phase 2"""

import requests
import json
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.foundation.config_manager import ConfigManager
from src.data.rate_limiter import get_rate_limiter


class AlphaVantageClient:
    def __init__(self):
        self.config = ConfigManager()
        self.api_key = self.config.av_api_key
        self.base_url = self.config.av_config.get('base_url')
        self.timeout = self.config.av_config.get('timeout', 30)
        
        # Get the global rate limiter
        self.rate_limiter = get_rate_limiter()
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        print(f"AV Client initialized with rate limiting")
    
    def _make_request(self, params, description="API call"):
        """
        Make a rate-limited API request
        Phase 2: All requests go through rate limiter
        """
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
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            raise
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def get_realtime_options(self, symbol='SPY'):
        """
        Get real-time options data for a symbol
        Phase 2: Now with rate limiting
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

        print(f"Calling REALTIME_OPTIONS for {symbol}...")
        data = self._make_request(params, f"REALTIME_OPTIONS({symbol})")
        print(f"✓ Successfully retrieved options data for {symbol}")
        
        return data
    
    def get_historical_options(self, symbol='SPY', date=None):
        """
        Get historical options data for a symbol
        Phase 2.3: Second API endpoint
        """
        # First, add this endpoint to your config/apis/alpha_vantage.yaml
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'apikey': self.api_key,
            'datatype': 'json'
        }
        
        if date:
            params['date'] = date
        
        print(f"Calling HISTORICAL_OPTIONS for {symbol}...")
        data = self._make_request(params, f"HISTORICAL_OPTIONS({symbol})")
        print(f"✓ Successfully retrieved historical options for {symbol}")
        
        return data
    
    def get_rate_limit_status(self):
        """Get current rate limit statistics"""
        return self.rate_limiter.get_stats()


if __name__ == "__main__":
    # Quick test
    client = AlphaVantageClient()
    print(f"Rate limit status: {client.get_rate_limit_status()}")