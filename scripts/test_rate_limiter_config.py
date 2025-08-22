#!/usr/bin/env python3
"""
Test configuration-driven RateLimiter implementation
Phase 1: Verify zero hardcoded values in rate limiting
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.rate_limiter import RateLimiter
from src.foundation.logger import get_logger


def test_configuration_driven_rate_limiter():
    """Test that RateLimiter works with all parameters from configuration"""
    logger = get_logger('RateLimiterTest')
    
    logger.info("Testing configuration-driven RateLimiter...")
    
    # Test 1: All parameters provided (no defaults)
    logger.info("\n=== Test 1: All Parameters Required ===")
    try:
        # This should work - all parameters provided
        rate_limiter = RateLimiter(
            calls_per_minute=10,        # Low limit for testing
            burst_capacity=5,           # Small burst for testing
            refill_rate=2.0,           # 2 tokens per second
            time_window=60,            # 60 second window
            check_interval=0.1,        # Check every 0.1 seconds
            initial_tokens=5.0,        # Start with full bucket
            initial_total_calls=0,     # Start with 0 total calls
            initial_rejected_calls=0,  # Start with 0 rejected calls
            initial_window_calls=0     # Start with 0 window calls
        )
        logger.info("✓ RateLimiter created successfully with all parameters")
        
        # Verify initial state
        stats = rate_limiter.get_stats()
        logger.info(f"Initial stats: {stats}")
        
        assert stats['total_calls'] == 0, "Initial total calls should be 0"
        assert stats['rejected_calls'] == 0, "Initial rejected calls should be 0"
        assert stats['window_calls'] == 0, "Initial window calls should be 0"
        assert stats['current_tokens'] == 5.0, "Initial tokens should be 5.0"
        assert stats['calls_per_window_limit'] == 10, "Calls per window should be 10"
        assert stats['time_window'] == 60, "Time window should be 60"
        assert stats['burst_capacity'] == 5, "Burst capacity should be 5"
        assert stats['refill_rate'] == 2.0, "Refill rate should be 2.0"
        assert stats['check_interval'] == 0.1, "Check interval should be 0.1"
        
        logger.info("✓ All initial values correct from configuration")
        
    except Exception as e:
        logger.error(f"✗ Failed to create RateLimiter: {e}")
        return False
    
    # Test 2: Token acquisition
    logger.info("\n=== Test 2: Token Acquisition ===")
    try:
        # Acquire tokens rapidly
        for i in range(3):
            success = rate_limiter.acquire()
            logger.info(f"Token acquisition {i+1}: {'✓' if success else '✗'}")
            assert success, f"Token acquisition {i+1} should succeed"
        
        stats = rate_limiter.get_stats()
        logger.info(f"After 3 acquisitions: {stats}")
        
        assert stats['total_calls'] == 3, "Total calls should be 3"
        assert stats['window_calls'] == 3, "Window calls should be 3"
        assert stats['current_tokens'] == 2.0, "Should have 2 tokens left"
        
        logger.info("✓ Token acquisition working correctly")
        
    except Exception as e:
        logger.error(f"✗ Token acquisition failed: {e}")
        return False
    
    # Test 3: Token refill
    logger.info("\n=== Test 3: Token Refill ===")
    try:
        # Wait for token refill (2 tokens per second, so 1 second = 2 tokens)
        logger.info("Waiting 1 second for token refill...")
        time.sleep(1.0)
        
        # Check tokens after refill
        stats = rate_limiter.get_stats()
        logger.info(f"After 1 second refill: {stats}")
        
        # Should have refilled: 2.0 remaining + 2.0 refilled = 4.0 tokens
        expected_tokens = 4.0
        actual_tokens = stats['current_tokens']
        assert abs(actual_tokens - expected_tokens) < 0.1, f"Expected ~{expected_tokens} tokens, got {actual_tokens}"
        
        logger.info("✓ Token refill working correctly")
        
    except Exception as e:
        logger.error(f"✗ Token refill failed: {e}")
        return False
    
    # Test 4: Reconfiguration
    logger.info("\n=== Test 4: Runtime Reconfiguration ===")
    try:
        # Reconfigure rate limiter
        rate_limiter.reconfigure(
            calls_per_minute=20,
            burst_capacity=10,
            refill_rate=5.0,
            time_window=30,
            check_interval=0.05
        )
        
        stats = rate_limiter.get_stats()
        logger.info(f"After reconfiguration: {stats}")
        
        assert stats['calls_per_window_limit'] == 20, "Calls per window should be updated to 20"
        assert stats['burst_capacity'] == 10, "Burst capacity should be updated to 10"
        assert stats['refill_rate'] == 5.0, "Refill rate should be updated to 5.0"
        assert stats['time_window'] == 30, "Time window should be updated to 30"
        assert stats['check_interval'] == 0.05, "Check interval should be updated to 0.05"
        
        logger.info("✓ Runtime reconfiguration working correctly")
        
    except Exception as e:
        logger.error(f"✗ Reconfiguration failed: {e}")
        return False
    
    # Test 5: Statistics reset
    logger.info("\n=== Test 5: Statistics Reset ===")
    try:
        # Reset with custom values
        rate_limiter.reset_stats(
            reset_total_calls=100,
            reset_rejected_calls=5,
            reset_window_calls=10
        )
        
        stats = rate_limiter.get_stats()
        logger.info(f"After stats reset: {stats}")
        
        assert stats['total_calls'] == 100, "Total calls should be reset to 100"
        assert stats['rejected_calls'] == 5, "Rejected calls should be reset to 5"
        assert stats['window_calls'] == 10, "Window calls should be reset to 10"
        
        logger.info("✓ Statistics reset working correctly")
        
    except Exception as e:
        logger.error(f"✗ Statistics reset failed: {e}")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL RATE LIMITER CONFIGURATION TESTS PASSED!")
    logger.info("RateLimiter is completely configuration-driven with zero hardcoded values")
    logger.info("="*60)
    
    return True


def test_missing_parameters():
    """Test that RateLimiter fails properly when parameters are missing"""
    logger = get_logger('RateLimiterTest')
    
    logger.info("\n=== Test: Missing Parameters ===")
    
    # This should fail because we're not providing required parameters
    try:
        # Try to create with missing parameters (old style)
        rate_limiter = RateLimiter(calls_per_minute=10)  # Missing other required params
        logger.error("✗ RateLimiter should have failed with missing parameters")
        return False
    except TypeError as e:
        logger.info(f"✓ RateLimiter correctly failed with missing parameters: {e}")
        return True
    except Exception as e:
        logger.error(f"✗ Unexpected error type: {e}")
        return False


def main():
    """Run all RateLimiter configuration tests"""
    logger = get_logger('RateLimiterTest')
    
    logger.info("Starting RateLimiter configuration tests...")
    
    try:
        # Test configuration-driven functionality
        config_test = test_configuration_driven_rate_limiter()
        
        # Test missing parameter handling
        missing_test = test_missing_parameters()
        
        if config_test and missing_test:
            logger.info("\n✅ ALL TESTS PASSED!")
            logger.info("RateLimiter implementation is completely configuration-driven")
            return 0
        else:
            logger.error("\n❌ SOME TESTS FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())