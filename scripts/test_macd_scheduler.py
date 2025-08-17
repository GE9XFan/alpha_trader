#!/usr/bin/env python3
"""Test MACD scheduler integration - Phase 5.2"""

import sys
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler
from src.foundation.config_manager import ConfigManager


def test_macd_scheduler():
    """Test MACD scheduling functionality"""
    print("=== Testing MACD Scheduler Integration ===\n")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize scheduler in test mode
    scheduler = DataScheduler(test_mode=True)
    
    # Check configuration
    print("1. Configuration Check:")
    status = scheduler.get_status()
    print(f"   Scheduler initialized: {'✓' if not status['running'] else '✗'}")
    
    # Start scheduler
    print("\n2. Starting Scheduler with MACD...")
    scheduler.start()
    
    # Check jobs created
    status = scheduler.get_status()
    macd_jobs = [job for job in status['jobs'] if 'MACD' in job['name']]
    rsi_jobs = [job for job in status['jobs'] if 'RSI' in job['name']]
    
    print(f"\n3. Indicator Jobs Created:")
    print(f"   Total MACD jobs: {len(macd_jobs)}")
    print(f"   Total RSI jobs: {len(rsi_jobs)}")
    print(f"   Total all jobs: {status['total_jobs']}")
    
    # Show first few MACD jobs
    print(f"\n   Sample MACD jobs:")
    for job in macd_jobs[:5]:
        print(f"   - {job['name']}: Next run at {job['next_run']}")
    
    # Get initial database state
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count, MAX(updated_at) as last_update
            FROM av_macd
            GROUP BY symbol
            ORDER BY symbol
        """))
        initial_counts = {row[0]: (row[1], row[2]) for row in result}
    
    print(f"\n4. Initial MACD data in database:")
    if initial_counts:
        for symbol, (count, last_update) in initial_counts.items():
            print(f"   {symbol}: {count} records, last update: {last_update}")
    else:
        print("   No MACD data yet")
    
    # Wait for some MACD jobs to execute
    wait_time = 65  # Wait just over 60 seconds for Tier A to trigger
    print(f"\n5. Waiting {wait_time} seconds for MACD jobs to execute...")
    print("   You should see MACD fetch messages below:\n")
    print("-" * 50)
    
    time.sleep(wait_time)
    
    print("-" * 50)
    
    # Check for new data
    print("\n6. Checking for new MACD data...")
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count, MAX(updated_at) as last_update
            FROM av_macd
            WHERE updated_at > NOW() - INTERVAL '2 minutes'
            GROUP BY symbol
            ORDER BY symbol
        """))
        
        new_data = list(result)
        
        if new_data:
            print("   ✓ New MACD data found:")
            for row in new_data:
                print(f"     {row[0]}: {row[1]} records updated at {row[2]}")
        else:
            print("   ⚠ No new MACD data found (cache may be serving requests)")
    
    # Check crossover signals
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as crossovers
            FROM (
                SELECT symbol, timestamp, macd_hist,
                       LAG(macd_hist) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_hist
                FROM av_macd
            ) t
            WHERE (macd_hist > 0 AND prev_hist < 0) 
               OR (macd_hist < 0 AND prev_hist > 0)
            GROUP BY symbol
            ORDER BY crossovers DESC
            LIMIT 5
        """))
        
        print("\n7. MACD Crossover Signals by Symbol:")
        for row in result:
            print(f"   {row[0]}: {row[1]} crossovers")
    
    # Stop scheduler
    print("\n8. Stopping scheduler...")
    scheduler.stop()
    
    # Summary
    print("\n=== MACD Scheduler Test Summary ===")
    print(f"✓ MACD jobs created: {len(macd_jobs)}")
    print(f"✓ RSI jobs created: {len(rsi_jobs)}")
    print(f"✓ Expected jobs: 23 each (all symbols)")
    
    expected_indicator_jobs = 46  # 23 RSI + 23 MACD
    actual_indicator_jobs = len(rsi_jobs) + len(macd_jobs)
    
    if actual_indicator_jobs == expected_indicator_jobs:
        print(f"\n✅ MACD Scheduler Integration Successful!")
        print(f"Total indicator jobs: {actual_indicator_jobs}")
        print("\nIndicator Update Schedule:")
        print("  Tier A (SPY, QQQ, IWM, IBIT): Every 60 seconds")
        print("  Tier B (MAG7 stocks): Every 300 seconds")
        print("  Tier C (Other stocks): Every 600 seconds")
        return True
    else:
        print(f"\n⚠ Warning: Expected {expected_indicator_jobs} indicator jobs but found {actual_indicator_jobs}")
        return False


if __name__ == "__main__":
    success = test_macd_scheduler()
    sys.exit(0 if success else 1)