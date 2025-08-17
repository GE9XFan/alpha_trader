#!/usr/bin/env python3
"""Test RSI scheduler integration - Phase 5.1"""

import sys
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler
from src.foundation.config_manager import ConfigManager


def test_rsi_scheduler():
    """Test RSI scheduling functionality"""
    print("=== Testing RSI Scheduler Integration ===\n")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize scheduler in test mode
    scheduler = DataScheduler(test_mode=True)
    
    # Check configuration
    print("1. Configuration Check:")
    status = scheduler.get_status()
    print(f"   Scheduler initialized: {'✓' if not status['running'] else '✗'}")
    
    # Start scheduler
    print("\n2. Starting Scheduler with RSI...")
    scheduler.start()
    
    # Check jobs created
    status = scheduler.get_status()
    rsi_jobs = [job for job in status['jobs'] if 'RSI' in job['name']]
    
    print(f"\n3. RSI Jobs Created:")
    print(f"   Total RSI jobs: {len(rsi_jobs)}")
    print(f"   Total all jobs: {status['total_jobs']}")
    
    # Show first few RSI jobs
    print(f"\n   Sample RSI jobs:")
    for job in rsi_jobs[:5]:
        print(f"   - {job['name']}: Next run at {job['next_run']}")
    
    # Get initial database state
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count, MAX(updated_at) as last_update
            FROM av_rsi
            GROUP BY symbol
            ORDER BY symbol
        """))
        initial_counts = {row[0]: (row[1], row[2]) for row in result}
    
    print(f"\n4. Initial RSI data in database:")
    for symbol, (count, last_update) in initial_counts.items():
        print(f"   {symbol}: {count} records, last update: {last_update}")
    
    # Wait for some RSI jobs to execute
    wait_time = 65  # Wait just over 60 seconds for Tier A to trigger
    print(f"\n5. Waiting {wait_time} seconds for RSI jobs to execute...")
    print("   You should see RSI fetch messages below:\n")
    print("-" * 50)
    
    time.sleep(wait_time)
    
    print("-" * 50)
    
    # Check for new data
    print("\n6. Checking for new RSI data...")
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count, MAX(updated_at) as last_update
            FROM av_rsi
            WHERE updated_at > NOW() - INTERVAL '2 minutes'
            GROUP BY symbol
            ORDER BY symbol
        """))
        
        new_data = list(result)
        
        if new_data:
            print("   ✓ New RSI data found:")
            for row in new_data:
                print(f"     {row[0]}: {row[1]} records updated at {row[2]}")
        else:
            print("   ⚠ No new RSI data found (cache may be serving requests)")
    
    # Stop scheduler
    print("\n7. Stopping scheduler...")
    scheduler.stop()
    
    # Summary
    print("\n=== RSI Scheduler Test Summary ===")
    print(f"✓ RSI jobs created: {len(rsi_jobs)}")
    print(f"✓ Expected jobs: 23 (all symbols)")
    
    if len(rsi_jobs) == 23:
        print("\n✅ RSI Scheduler Integration Successful!")
        print("\nRSI Update Schedule:")
        print("  Tier A (SPY, QQQ, IWM, IBIT): Every 60 seconds")
        print("  Tier B (MAG7 stocks): Every 300 seconds")
        print("  Tier C (Other stocks): Every 600 seconds")
        return True
    else:
        print(f"\n⚠ Warning: Expected 23 RSI jobs but found {len(rsi_jobs)}")
        return False


if __name__ == "__main__":
    success = test_rsi_scheduler()
    sys.exit(0 if success else 1)