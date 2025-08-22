#!/usr/bin/env python3
"""Test database and cache connections"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.db_manager import get_db_manager
from src.data.cache_manager import get_cache_manager


def test_database():
    """Test database connection and operations"""
    print("=== Database Connection Test ===\n")
    
    db = get_db_manager()
    
    # Test query
    result = db.execute_query("SELECT version()")
    print(f"PostgreSQL version: {result[0]['version']}")
    
    # Test table check
    print(f"Table 'test' exists: {db.table_exists('test')}")
    
    print("\n✓ Database connection working!")
    return True


def test_cache():
    """Test Redis cache operations"""
    print("\n=== Cache Connection Test ===\n")
    
    cache = get_cache_manager()
    
    # Test basic operations
    test_key = "test:connection"
    test_value = {"status": "connected", "test": True}
    
    # Set
    cache.set(test_key, test_value, ttl=60)
    print(f"Set {test_key}: {test_value}")
    
    # Get
    retrieved = cache.get(test_key)
    print(f"Retrieved: {retrieved}")
    
    # Check TTL
    ttl = cache.get_ttl(test_key)
    print(f"TTL: {ttl} seconds")
    
    # Get stats
    stats = cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Total keys: {stats.get('total_keys', 0)}")
    print(f"  Memory used: {stats.get('used_memory_human', 'N/A')}")
    print(f"  Connected clients: {stats.get('connected_clients', 0)}")
    
    # Cleanup
    cache.delete(test_key)
    
    print("\n✓ Cache connection working!")
    return True


def main():
    """Run all connection tests"""
    try:
        # Test database
        db_ok = test_database()
        
        # Test cache
        cache_ok = test_cache()
        
        if db_ok and cache_ok:
            print("\n" + "="*50)
            print("✅ All connections successful!")
            print("Phase 0 foundation is ready!")
            print("="*50)
        else:
            print("\n❌ Some connections failed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())