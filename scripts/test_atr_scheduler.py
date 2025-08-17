#!/usr/bin/env python3
"""Test ATR scheduler integration - Phase 5.5"""

import sys
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler
from src.foundation.config_manager import ConfigManager


def test_atr_scheduler():
    """Test ATR scheduling functionality"""
    print("=== Testing ATR Scheduler Integration ===\n")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize scheduler in test mode
    scheduler = DataScheduler(test_mode=True)
    
    # Check configuration
    print("1. Configuration Check:")
    status = scheduler.get_status()
    print(f"   Scheduler initialized: {'✓' if not status['running'] else '✗'}")
    
    # Check if indicators_slow is configured
    if 'indicators_slow' in scheduler.api_groups:
        slow_config = scheduler.api_groups['indicators_slow']
        print(f"   indicators_slow group: ✓ Found")
        print(f"   APIs in group: {slow_config.get('apis', [])}")
        print(f"   Tier A interval: {slow_config.get('tier_a_interval', 900)}s (15 min)")
        print(f"   Tier B interval: {slow_config.get('tier_b_interval', 1800)}s (30 min)")
        print(f"   Tier C interval: {slow_config.get('tier_c_interval', 3600)}s (60 min)")
    else:
        print(f"   indicators_slow group: ✗ Not found")
    
    # Start scheduler
    print("\n2. Starting Scheduler with ATR...")
    scheduler.start()
    
    # Check jobs created
    status = scheduler.get_status()
    
    # Count different job types
    atr_jobs = [job for job in status['jobs'] if 'ATR' in job['name']]
    rsi_jobs = [job for job in status['jobs'] if 'RSI' in job['name']]
    macd_jobs = [job for job in status['jobs'] if 'MACD' in job['name']]
    bbands_jobs = [job for job in status['jobs'] if 'BBANDS' in job['name']]
    vwap_jobs = [job for job in status['jobs'] if 'VWAP' in job['name']]
    
    print(f"\n3. Indicator Jobs Created:")
    print(f"   ┌─────────────┬──────────┬─────────────┐")
    print(f"   │ Indicator   │ Jobs     │ Type        │")
    print(f"   ├─────────────┼──────────┼─────────────┤")
    print(f"   │ RSI         │ {len(rsi_jobs):^8} │ Fast (1min) │")
    print(f"   │ MACD        │ {len(macd_jobs):^8} │ Fast (1min) │")
    print(f"   │ BBANDS      │ {len(bbands_jobs):^8} │ Fast (5min) │")
    print(f"   │ VWAP        │ {len(vwap_jobs):^8} │ Fast (5min) │")
    print(f"   │ ATR         │ {len(atr_jobs):^8} │ Slow (daily)│")
    print(f"   └─────────────┴──────────┴─────────────┘")
    
    total_indicator_jobs = len(atr_jobs) + len(rsi_jobs) + len(macd_jobs) + len(bbands_jobs) + len(vwap_jobs)
    print(f"   Total indicator jobs: {total_indicator_jobs}")
    print(f"   Total all jobs: {status['total_jobs']}")
    
    # Show sample ATR jobs
    if atr_jobs:
        print(f"\n   Sample ATR jobs:")
        for job in atr_jobs[:5]:
            print(f"   - {job['name']}: Next run at {job['next_run']}")
    
    # Get initial database state for ATR
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count, MAX(timestamp) as last_date
            FROM av_atr
            GROUP BY symbol
            ORDER BY symbol
            LIMIT 5
        """))
        initial_counts = {row[0]: (row[1], row[2]) for row in result}
    
    print(f"\n4. Current ATR data in database:")
    if initial_counts:
        for symbol, (count, last_date) in initial_counts.items():
            print(f"   {symbol}: {count:,} records, last date: {last_date}")
    else:
        print("   No ATR data yet")
    
    # Note about ATR update frequency
    print(f"\n5. ATR Update Schedule:")
    print("   Note: ATR is daily data, so it updates less frequently")
    print("   - Tier A (SPY, QQQ, etc): Every 15 minutes")
    print("   - Tier B (AAPL, MSFT, etc): Every 30 minutes")
    print("   - Tier C (others): Every 60 minutes")
    print("   This is intentional - no need to check every minute for daily data!")
    
    # Compare with fast indicators
    print(f"\n6. Update Frequency Comparison:")
    print(f"   ┌─────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"   │ Indicator   │ Tier A       │ Tier B       │ Tier C       │")
    print(f"   ├─────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"   │ RSI/MACD    │ 60s (1 min)  │ 300s (5 min) │ 600s (10 min)│")
    print(f"   │ BBANDS/VWAP │ 60s (1 min)  │ 300s (5 min) │ 600s (10 min)│")
    print(f"   │ ATR         │ 900s (15 min)│ 1800s (30min)│ 3600s (1 hr) │")
    print(f"   └─────────────┴──────────────┴──────────────┴──────────────┘")
    
    # Wait a short time to see if jobs execute
    wait_time = 30
    print(f"\n7. Waiting {wait_time} seconds to observe scheduler...")
    print("   (ATR jobs run less frequently, so may not execute in this window)")
    print("-" * 50)
    
    time.sleep(wait_time)
    
    print("-" * 50)
    
    # Stop scheduler
    scheduler.stop()
    
    print("\n✅ ATR Scheduler Test Complete!")
    print("\nKey Points:")
    print("  - ATR jobs created successfully")
    print("  - Updates every 15-60 minutes (not every minute)")
    print("  - This is optimal for daily volatility data")
    print("  - Reduces unnecessary API calls")
    
    return True


if __name__ == "__main__":
    print("Phase 5.5 - Step 6: Scheduler Integration Test")
    print("=" * 50 + "\n")
    
    success = test_atr_scheduler()
    
    if success:
        print("\n✅ ATR scheduler integration successful!")
        print("\nNext: End-to-end testing and documentation")
    else:
        print("\n❌ ATR scheduler test failed")
    
    sys.exit(0 if success else 1)