#!/usr/bin/env python3
import redis
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables
load_dotenv()

def test_redis_connection():
    """Test Redis connection"""
    try:
        # Don't pass password parameter if it's empty
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            decode_responses=True
        )
        
        print("Connecting to Redis...")
        
        # Test ping
        if redis_client.ping():
            print("✅ Redis connected successfully!")
        
        # Test write
        test_key = "alphatrader:test"
        test_value = {
            "timestamp": datetime.now().isoformat(),
            "message": "Redis connection test"
        }
        redis_client.set(test_key, json.dumps(test_value), ex=10)
        print("✅ Redis write successful")
        
        # Test read
        retrieved = redis_client.get(test_key)
        if retrieved:
            data = json.loads(retrieved)
            print(f"✅ Redis read successful: {data['message']}")
        
        # Test delete
        redis_client.delete(test_key)
        print("✅ Redis delete successful")
        
        # Check Redis info
        info = redis_client.info()
        print(f"   Redis version: {info['redis_version']}")
        print(f"   Memory used: {info['used_memory_human']}")
        
        print("✅ Redis test completed successfully!")
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Start Redis with: brew services start redis")
        return False
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

if __name__ == "__main__":
    test_redis_connection()