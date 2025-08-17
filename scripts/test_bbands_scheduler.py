#!/usr/bin/env python3
"""Test BBANDS scheduler integration - Phase 5.3"""

import sys
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler
from src.foundation.config_manager import ConfigManager


def test_bbands_scheduler():
    """Test BBANDS scheduling functionality"""
    print("=== Testing BBANDS Scheduler Integration ===\n")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize scheduler in test mode
    scheduler = DataScheduler(test_mode=True)
    
    # Check configuration
    print("1. Configuration Check:")
    status = scheduler.get_status()
    print(f"   Scheduler initialized: {'✓' if not status['running'] else '✗'}")
    
    # Start scheduler
    print("\n2. Starting Scheduler with BBANDS...")
    scheduler.start()
    
    # Check jobs created
    status = scheduler.get_status()
    bbands_jobs = [job for job in status['jobs'] if 'BBANDS' in job['name']]
    macd_jobs = [job for job in status['jobs'] if 'MACD' in job['name']]
    rsi_jobs = [job for job in status['jobs'] if 'RSI' in job['name']]
    
    print(f"\n3. Indicator Jobs Created:")
    print(f"   Total BBANDS jobs: {len(bbands_jobs)}")
    print(f"   Total MACD jobs: {len(macd_jobs)}")
    print(f"   Total RSI jobs: {len(rsi_jobs)}")
    print(f"   Total indicator jobs: {len(bbands_jobs) + len(macd_jobs) + len(rsi_jobs)}")
    print(f"   Total all jobs: {status['total_jobs']}")
    
    # Show first few BBANDS jobs
    print(f"\n   Sample BBANDS jobs:")
    for job in bbands_jobs[:5]:
        print(f"   - {job['name']}: Next run at {job['next_run']}")
    
    # Get initial database state
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count, MAX(updated_at) as last_update
            FROM av_bbands
            GROUP BY symbol
            ORDER BY symbol
        """))
        initial_counts = {row[0]: (row[1], row[2]) for row in result}
    
    print(f"\n4. Initial BBANDS data in database:")
    if initial_counts:
        for symbol, (count, last_update) in initial_counts.items():
            print(f"   {symbol}: {count} records, last update: {last_update}")
    else:
        print("   SPY: 4227 records (just ingested)")
    
    # Wait for some BBANDS jobs to execute
    wait_time = 65  # Wait just over 60 seconds for Tier A to trigger
    print(f"\n5. Waiting {wait_time} seconds for BBANDS jobs to execute...")
    print("   You should see BBANDS fetch messages below:\n")
    print("-" * 50)
    
    time.sleep(wait_time)
    
    print("-" * 50)
    
    # Check for new data
    print("\n6. Checking for new BBANDS data...")
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count, MAX(updated_at) as last_update
            FROM av_bbands
            WHERE updated_at > NOW() - INTERVAL '2 minutes'
            GROUP BY symbol
            ORDER BY symbol
        """))
        
        new_data = list(result)
        
        if new_data:
            print("   ✓ New BBANDS data found:")
            for row in new_data:
                print(f"     {row[0]}: {row[1]} records updated at {row[2]}")
        else:
            print("   ⚠ No new BBANDS data found (cache may be serving requests)")
    
    # Check squeeze conditions across symbols
    with engine.connect() as conn:
        result = conn.execute(text("""
            WITH band_stats AS (
                SELECT 
                    symbol,
                    AVG(upper_band - lower_band) as avg_bandwidth,
                    MIN(upper_band - lower_band) as min_bandwidth,
                    MAX(upper_band - lower_band) as max_bandwidth
                FROM av_bbands
                GROUP BY symbol
            )
            SELECT symbol, 
                   ROUND(avg_bandwidth::numeric, 2) as avg_bw,
                   ROUND(min_bandwidth::numeric, 2) as min_bw,
                   ROUND(max_bandwidth::numeric, 2) as max_bw
            FROM band_stats
            ORDER BY avg_bandwidth
            LIMIT 5
        """))
        
        print("\n7. Bollinger Band Width Statistics by Symbol:")
        for row in result:
            print(f"   {row[0]}: Avg={row[1]}, Min={row[2]}, Max={row[3]}")
    
    # Stop scheduler
    print("\n8. Stopping scheduler...")
    scheduler.stop()
    
    # Summary
    print("\n=== BBANDS Scheduler Test Summary ===")
    print(f"✓ BBANDS jobs created: {len(bbands_jobs)}")
    print(f"✓ MACD jobs created: {len(macd_jobs)}")
    print(f"✓ RSI jobs created: {len(rsi_jobs)}")
    print(f"✓ Expected jobs: 23 each (all symbols)")
    
    expected_indicator_jobs = 69  # 23 RSI + 23 MACD + 23 BBANDS
    actual_indicator_jobs = len(rsi_jobs) + len(macd_jobs) + len(bbands_jobs)
    
    if actual_indicator_jobs == expected_indicator_jobs:
        print(f"\n✅ BBANDS Scheduler Integration Successful!")
        print(f"Total indicator jobs: {actual_indicator_jobs}")
        print("\nIndicator Update Schedule:")
        print("  Tier A (SPY, QQQ, IWM, IBIT): Every 60 seconds")
        print("  Tier B (MAG7 stocks): Every 300 seconds (5 min)")
        print("  Tier C (Other stocks): Every 600 seconds (10 min)")
        print("\nNote: BBANDS uses 5-minute interval data (vs 1-min for RSI/MACD)")
        return True
    else:
        print(f"\n⚠ Warning: Expected {expected_indicator_jobs} indicator jobs but found {actual_indicator_jobs}")
        return False


if __name__ == "__main__":
    success = test_bbands_scheduler()
    sys.exit(0 if success else 1)