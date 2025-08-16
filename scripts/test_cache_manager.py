#!/usr/bin/env python3
"""Test cache manager functionality"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.cache_manager import get_cache


def test_cache_operations():
    """Test basic cache operations"""
    print("=== Testing Cache Manager ===\n")
    
    # Get cache instance
    cache = get_cache()
    
    # Test 1: Basic set/get
    print("Test 1: Basic set/get")
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    assert value == "test_value", f"Expected 'test_value', got {value}"
    print("✓ Basic set/get works\n")
    
    # Test 2: JSON serialization
    print("Test 2: JSON serialization")
    test_dict = {"symbol": "SPY", "price": 650.25, "volume": 1000000}
    cache.set("test_dict", test_dict)
    retrieved = cache.get("test_dict")
    assert retrieved == test_dict, "Dictionary not properly serialized"
    print("✓ JSON serialization works\n")
    
    # Test 3: TTL
    print("Test 3: TTL (time to live)")
    cache.set("expires_soon", "temporary", ttl=2)
    assert cache.exists("expires_soon"), "Key should exist"
    print(f"TTL remaining: {cache.get_ttl('expires_soon')} seconds")
    time.sleep(3)
    assert not cache.exists("expires_soon"), "Key should have expired"
    print("✓ TTL expiration works\n")
    
    # Test 4: Delete
    print("Test 4: Delete operation")
    cache.set("to_delete", "delete me")
    assert cache.exists("to_delete")
    cache.delete("to_delete")
    assert not cache.exists("to_delete")
    print("✓ Delete works\n")
    
    # Test 5: Cache stats
    print("Test 5: Cache statistics")
    stats = cache.get_stats()
    print(f"  Keys in cache: {stats.get('keys', 0)}")
    print(f"  Memory used: {stats.get('used_memory', 'Unknown')}")
    print(f"  Commands processed: {stats.get('commands_processed', 0)}")
    print("✓ Stats retrieval works\n")
    
    # Cleanup
    cache.flush_pattern("test_*")
    
    print("✅ All cache tests passed!")
    return True


if __name__ == "__main__":
    success = test_cache_operations()
    sys.exit(0 if success else 1)