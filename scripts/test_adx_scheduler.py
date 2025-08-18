#!/usr/bin/env python3
"""Test ADX scheduler integration - Phase 5.6"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler

def test_adx_scheduler():
    print("Testing ADX Scheduler Integration...")
    
    scheduler = DataScheduler(test_mode=True)
    scheduler.start()
    
    # Check status
    status = scheduler.get_status()
    
    # Count ADX jobs
    adx_jobs = [job for job in status['jobs'] if 'ADX' in job['name']]
    
    print(f"\n✅ ADX Jobs Created: {len(adx_jobs)}")
    print("\nSample ADX jobs:")
    for job in adx_jobs[:5]:
        # FIX: Changed from job['trigger'] to job['next_run']
        print(f"  - {job['name']}: Next run at {job['next_run']}")
    
    # Show tier breakdown
    tier_a = [j for j in adx_jobs if '_A' in j['name']]
    tier_b = [j for j in adx_jobs if '_B' in j['name']]
    tier_c = [j for j in adx_jobs if '_C' in j['name']]
    
    print(f"\nADX Job Distribution:")
    print(f"  Tier A (15 min): {len(tier_a)} symbols")
    print(f"  Tier B (30 min): {len(tier_b)} symbols")
    print(f"  Tier C (60 min): {len(tier_c)} symbols")
    
    scheduler.stop()
    return len(adx_jobs) > 0

if __name__ == "__main__":
    success = test_adx_scheduler()
    if success:
        print("\n✅ ADX Scheduler Integration Successful!")
        print("Phase 5.6 - Step 6 Complete")
    sys.exit(0 if success else 1)