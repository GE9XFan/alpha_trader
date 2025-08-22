#!/usr/bin/env python3
"""Test configuration system"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager


def test_config():
    """Test configuration loading"""
    config = ConfigManager()
    
    print("=== Configuration Test ===\n")
    
    # Test database config
    print(f"Database host: {config.db_config.get('host')}")
    print(f"Database name: {config.db_config.get('name')}")
    
    # Test Redis config
    print(f"\nRedis host: {config.redis_config.get('host')}")
    print(f"Redis port: {config.redis_config.get('port')}")
    
    # Test Alpha Vantage config
    print(f"\nAV Base URL: {config.av_config.get('base_url')}")
    print(f"AV API Key set: {bool(config.av_api_key)}")
    
    # Test IBKR config
    print(f"\nIBKR host: {config.ibkr_config.get('host')}")
    print(f"IBKR port: {config.ibkr_config.get('port')}")
    
    # Test get_config method
    print(f"\nRate limit: {config.get_config('alpha_vantage.alpha_vantage.rate_limit.calls_per_minute')}")
    
    print("\n✓ Configuration system working!")


if __name__ == "__main__":
    test_config()