#!/usr/bin/env python3
"""
Test Redis connection and basic operations
Part of Day 1-2 implementation plan
"""

import redis
import yaml
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_redis_connection():
    """Test Redis connection using config from config.yaml"""
    
    print("Testing Redis connection...")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Connect to Redis
    try:
        r = redis.Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            decode_responses=config['redis']['decode_responses']
        )
        
        # Test basic operations
        test_key = 'test:connection'
        test_value = 'AlphaTrader Redis Test'
        
        # SET operation
        r.set(test_key, test_value)
        print(f"  ✓ SET {test_key} = {test_value}")
        
        # GET operation
        retrieved = r.get(test_key)
        assert retrieved == test_value, f"Expected {test_value}, got {retrieved}"
        print(f"  ✓ GET {test_key} = {retrieved}")
        
        # TTL operation (testing 1 second TTL as per spec)
        r.setex('test:ttl', 1, 'temporary')
        ttl_response = r.ttl('test:ttl')
        ttl = int(ttl_response) if ttl_response is not None else -1  # type: ignore
        assert ttl > 0, "TTL should be positive"
        print(f"  ✓ SETEX with TTL working (TTL={ttl})")
        
        # Cleanup
        r.delete(test_key)
        print(f"  ✓ DELETE {test_key}")
        
        # Test pipeline (for batch operations as per architecture)
        pipe = r.pipeline()
        pipe.set('test:pipeline:1', 'value1')
        pipe.set('test:pipeline:2', 'value2')
        pipe.execute()
        print("  ✓ Pipeline operations working")
        
        # Cleanup pipeline test
        r.delete('test:pipeline:1', 'test:pipeline:2')
        
        print("\n✅ Redis connection successful!")
        print(f"   Connected to: {config['redis']['host']}:{config['redis']['port']}")
        return True
        
    except redis.ConnectionError as e:
        print(f"\n❌ Redis connection failed: {e}")
        print("\nPlease ensure Redis is running:")
        print("  brew services start redis")
        print("  OR")
        print("  redis-server config/redis.conf")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_redis_connection()
    sys.exit(0 if success else 1)