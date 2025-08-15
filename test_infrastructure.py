#!/usr/bin/env python3
"""
Test Infrastructure Components
Run from project root: python test_infrastructure.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import our modules
from src.foundation.config_manager import ConfigManager
from src.foundation.base_module import BaseModule
import psycopg2
import redis
import json


class TestModule(BaseModule):
    """Test implementation of BaseModule"""
    
    def initialize(self) -> bool:
        print(f"  Initializing {self.name}...")
        self.is_initialized = True
        return True
    
    def health_check(self) -> dict:
        return {
            'status': 'healthy',
            'test': True
        }
    
    def shutdown(self) -> bool:
        print(f"  Shutting down {self.name}...")
        self.is_initialized = False
        return True


def test_config_manager():
    """Test ConfigManager"""
    print("\n1️⃣ Testing ConfigManager")
    print("-" * 40)
    
    try:
        config = ConfigManager()
        print("✓ ConfigManager loaded")
        
        # Test getting some values
        env = config.environment
        print(f"✓ Environment: {env}")
        
        trading_mode = config.get_trading_mode()
        print(f"✓ Trading Mode: {trading_mode}")
        
        # Test getting nested config
        rate_limit = config.get('apis.rate_limits.rate_limits.alpha_vantage.calls_per_minute')
        print(f"✓ Rate Limit: {rate_limit} calls/min")
        
        return True
    except Exception as e:
        print(f"✗ ConfigManager failed: {e}")
        return False


def test_base_module():
    """Test BaseModule"""
    print("\n2️⃣ Testing BaseModule")
    print("-" * 40)
    
    try:
        config = {'test_param': 'test_value'}
        module = TestModule(config, "TestModule")
        print("✓ Module created")
        
        # Test initialization
        if module.initialize():
            print("✓ Module initialized")
        
        # Test health check
        health = module.health_check()
        print(f"✓ Health check: {health['status']}")
        
        # Test status
        status = module.get_status()
        print(f"✓ Status: initialized={status['initialized']}")
        
        # Test shutdown
        if module.shutdown():
            print("✓ Module shutdown")
        
        return True
    except Exception as e:
        print(f"✗ BaseModule failed: {e}")
        return False


def test_database():
    """Test database connection"""
    print("\n3️⃣ Testing Database Connection")
    print("-" * 40)
    
    try:
        config = ConfigManager()
        db_config = config.get_database_config()
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT COUNT(*) FROM system_config")
        count = cursor.fetchone()[0]
        print(f"✓ Database connected")
        print(f"✓ System config entries: {count}")
        
        # Check tables
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_count = cursor.fetchone()[0]
        print(f"✓ Total tables: {table_count}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Database failed: {e}")
        return False


def test_redis():
    """Test Redis connection"""
    print("\n4️⃣ Testing Redis Connection")
    print("-" * 40)
    
    try:
        config = ConfigManager()
        redis_config = config.get_redis_config()
        
        r = redis.Redis(**redis_config)
        
        # Test ping
        if r.ping():
            print("✓ Redis connected")
        
        # Test set/get
        r.set('test_key', 'test_value', ex=5)
        value = r.get('test_key')
        if value == 'test_value':
            print("✓ Redis set/get working")
        
        # Test JSON storage
        test_data = {'test': 'data', 'number': 123}
        r.set('test_json', json.dumps(test_data), ex=5)
        retrieved = json.loads(r.get('test_json'))
        if retrieved == test_data:
            print("✓ Redis JSON storage working")
        
        # Cleanup
        r.delete('test_key', 'test_json')
        
        return True
        
    except Exception as e:
        print(f"✗ Redis failed: {e}")
        return False


def main():
    """Run all infrastructure tests"""
    print("=" * 50)
    print("🔧 INFRASTRUCTURE TEST SUITE")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("ConfigManager", test_config_manager()))
    results.append(("BaseModule", test_base_module()))
    results.append(("Database", test_database()))
    results.append(("Redis", test_redis()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:15} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✅ ALL INFRASTRUCTURE TESTS PASSED!")
        print("\n🎉 Phase 0 Complete! Ready for Phase 0.5 (API Discovery)")
    else:
        print("❌ Some tests failed. Please fix before proceeding.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())