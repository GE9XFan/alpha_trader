#!/usr/bin/env python3
"""Test scheduler functionality - Phase 4.2"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler
from src.connections.av_client import AlphaVantageClient
from src.data.rate_limiter import get_rate_limiter


def test_scheduler_basic():
    """Test basic scheduler functionality"""
    print("=== Testing DataScheduler ===\n")
    
    # Initialize scheduler
    scheduler = DataScheduler(test_mode=True)  # <-- Add test_mode=True
    
    # Check configuration loaded
    print("1. Configuration Check:")
    print(f"   Tier A symbols: {scheduler.tiers['tier_a']['symbols']}")
    print(f"   Tier B symbols: {len(scheduler.tiers['tier_b']['symbols'])} symbols")
    print(f"   Rate limit target: {scheduler.rate_limit_config['target_per_minute']}/min")
    print()
    
    # Start scheduler
    print("2. Starting Scheduler...")
    scheduler.start()
    print()
    
    # Check jobs created
    status = scheduler.get_status()
    print("3. Jobs Created:")
    print(f"   Total jobs: {status['total_jobs']}")
    print(f"   Market hours: {status['is_market_hours']}")
    print(f"\n   First 5 jobs:")
    for job in status['jobs'][:5]:
        print(f"   - {job['name']}: Next run at {job['next_run']}")
    print()
    
    return scheduler


def test_short_run(duration_seconds=60):
    """Run scheduler for a short duration and monitor"""
    print(f"=== Running Scheduler for {duration_seconds} seconds ===\n")
    
    # Get initial statistics
    av_client = AlphaVantageClient()
    rate_limiter = get_rate_limiter()
    
    initial_api_calls = av_client.get_rate_limit_status()['calls_made']
    initial_cache_stats = av_client.get_cache_status()
    
    print("Initial Statistics:")
    print(f"  API calls made: {initial_api_calls}")
    print(f"  Cache keys: {initial_cache_stats.get('keys', 0)}")
    print()
    
    # Create and start scheduler
    scheduler = DataScheduler(test_mode=True)  # <-- Add test_mode=True
    scheduler.start()    

    print(f"Scheduler running... (waiting {duration_seconds} seconds)")
    print("You should see scheduled jobs executing below:\n")
    print("-" * 50)
    
    # Run for specified duration
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        time.sleep(10)
        
        # Show periodic status
        elapsed = int(time.time() - start_time)
        if elapsed % 20 == 0:
            current_stats = av_client.get_rate_limit_status()
            cache_stats = av_client.get_cache_status()
            
            print(f"\n[{elapsed}s] Status Update:")
            print(f"  API calls: {current_stats['calls_made'] - initial_api_calls} new")
            print(f"  Cache keys: {cache_stats.get('av_keys', 0)} AV keys")
            print("-" * 50)
    
    # Stop scheduler
    print("\nStopping scheduler...")
    scheduler.stop()
    
    # Final statistics
    final_api_calls = av_client.get_rate_limit_status()['calls_made']
    final_cache_stats = av_client.get_cache_status()
    
    api_calls_made = final_api_calls - initial_api_calls
    
    print("\n=== Test Results ===")
    print(f"Duration: {duration_seconds} seconds")
    print(f"API calls made: {api_calls_made}")
    print(f"Cache keys: {final_cache_stats.get('av_keys', 0)}")
    print(f"Jobs created: {scheduler.jobs_created}")
    
    # Calculate expected vs actual
    # Tier A: 4 symbols @ 30s = 8 calls/min
    # Tier B: 7 symbols @ 60s = 7 calls/min
    # Total expected: ~15 calls/min (but cache will reduce this)
    expected_calls = (duration_seconds / 60) * 15
    cache_efficiency = 1 - (api_calls_made / expected_calls) if expected_calls > 0 else 0
    
    print(f"\nEfficiency:")
    print(f"  Expected calls (no cache): ~{expected_calls:.0f}")
    print(f"  Actual calls (with cache): {api_calls_made}")
    print(f"  Cache efficiency: {cache_efficiency:.1%}")
    
    return scheduler


def test_market_awareness():
    """Test market hours detection"""
    print("=== Testing Market Hours Awareness ===\n")
    
    scheduler = DataScheduler(test_mode=False)  # <-- Keep False here to test real market detection
    
    is_market = scheduler._is_market_hours()
    now = datetime.now(scheduler.timezone)
    
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Day of week: {now.strftime('%A')}")
    print(f"Market hours: {scheduler.market_hours['market_open']} - {scheduler.market_hours['market_close']}")
    print(f"Is market hours: {is_market}")
    
    if not is_market:
        print("\n⚠️  Note: Market is closed. Jobs will skip API calls.")
        print("    For testing, you may want to temporarily modify _is_market_hours()")
    
    return is_market


def main():
    """Run all scheduler tests"""
    print("=" * 60)
    print("PHASE 4.2 - SCHEDULER TESTING")
    print("=" * 60)
    print()
    
    # Test 1: Market awareness
    is_market = test_market_awareness()
    print()
    
    # Test 2: Basic functionality
    scheduler = test_scheduler_basic()
    scheduler.stop()
    print()
    
    # Test 3: Short run test (only if you want to see it in action)
    response = input("Run 60-second live test? (y/n): ")
    if response.lower() == 'y':
        test_short_run(60)
    else:
        print("Skipping live test")
    
    print("\n✅ Scheduler tests complete!")
    print("\nNext steps:")
    print("  1. Run during market hours for best results")
    print("  2. Monitor cache hit rate improvement")
    print("  3. Prepare for 24-hour test (Phase 4.3)")


if __name__ == "__main__":
    main()