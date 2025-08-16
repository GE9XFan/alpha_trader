#!/usr/bin/env python3
"""Test rate limiter functionality"""

import sys
import time
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.rate_limiter import TokenBucketRateLimiter


def test_basic_functionality():
    """Test basic rate limiting"""
    print("=== Testing Basic Functionality ===\n")
    
    limiter = TokenBucketRateLimiter(
        max_per_minute=60,  # Lower limit for testing
        refill_rate=2,       # 2 tokens per second
        burst_capacity=5     # Small burst for testing
    )
    
    # Test burst capacity
    print("Test 1: Burst capacity (should succeed quickly)")
    for i in range(5):
        start = time.time()
        acquired = limiter.acquire()
        elapsed = time.time() - start
        print(f"  Call {i+1}: {'✓' if acquired else '✗'} ({elapsed:.3f}s)")
    
    print("\nTest 2: Rate limiting (should slow down)")
    for i in range(5):
        start = time.time()
        acquired = limiter.acquire()
        elapsed = time.time() - start
        print(f"  Call {i+1}: {'✓' if acquired else '✗'} ({elapsed:.3f}s)")
    
    # Show stats
    stats = limiter.get_stats()
    print(f"\nStats: {stats['calls_made']} successful, "
          f"{stats['calls_blocked']} blocked, "
          f"{stats['tokens_available']:.1f} tokens available")


def test_production_settings():
    """Test with production settings"""
    print("\n=== Testing Production Settings ===\n")
    
    limiter = TokenBucketRateLimiter(
        max_per_minute=600,
        refill_rate=10,
        burst_capacity=20
    )
    
    print("Simulating rapid API calls...")
    
    # Simulate 30 rapid calls
    start_time = time.time()
    
    for i in range(30):
        acquired = limiter.acquire(blocking=True, timeout=5)
        if not acquired:
            print(f"Failed to acquire token for call {i+1}")
            break
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  {i+1} calls in {elapsed:.2f}s ({rate:.1f} calls/sec)")
    
    total_time = time.time() - start_time
    print(f"\nCompleted 30 calls in {total_time:.2f} seconds")
    print(f"Average rate: {30/total_time:.1f} calls/second")
    
    stats = limiter.get_stats()
    print(f"\nFinal stats:")
    print(f"  - Calls made: {stats['calls_made']}")
    print(f"  - Tokens available: {stats['tokens_available']:.1f}")
    print(f"  - Success rate: {stats['success_rate']:.1f}%")


def test_concurrent_access():
    """Test thread safety with concurrent access"""
    print("\n=== Testing Concurrent Access ===\n")
    
    limiter = TokenBucketRateLimiter(
        max_per_minute=600,
        refill_rate=10,
        burst_capacity=20
    )
    
    results = {'success': 0, 'failed': 0}
    lock = threading.Lock()
    
    def make_calls(thread_id, num_calls):
        for i in range(num_calls):
            if limiter.acquire(blocking=False):
                with lock:
                    results['success'] += 1
            else:
                with lock:
                    results['failed'] += 1
            time.sleep(0.01)  # Small delay
    
    # Create multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=make_calls, args=(i, 10))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    print(f"Concurrent test results:")
    print(f"  - Successful calls: {results['success']}")
    print(f"  - Failed calls: {results['failed']}")
    print(f"  - Total attempts: {results['success'] + results['failed']}")


if __name__ == "__main__":
    print("=== Rate Limiter Test Suite ===\n")
    
    test_basic_functionality()
    test_production_settings()
    test_concurrent_access()
    
    print("\n✅ Rate limiter tests complete!")