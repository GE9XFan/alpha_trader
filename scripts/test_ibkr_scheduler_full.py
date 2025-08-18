#!/usr/bin/env python3
"""Test full IBKR scheduler integration with monitoring"""

import sys
import time
from pathlib import Path
from datetime import datetime
import signal

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler

print("=" * 60)
print("FULL IBKR SCHEDULER TEST WITH MONITORING")
print(f"Time: {datetime.now()}")
print("=" * 60)

# Global scheduler for signal handling
scheduler = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal - shutting down...")
    if scheduler:
        scheduler.stop()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

try:
    # Create and start scheduler
    print("\n1. Creating scheduler in test mode...")
    scheduler = DataScheduler(test_mode=True)
    
    print("\n2. Starting scheduler (includes IBKR connection)...")
    scheduler.start()
    
    # Check status
    time.sleep(5)  # Give it time to connect
    
    print("\n3. Checking IBKR status...")
    print(f"   IBKR Connected: {scheduler.ibkr_connected}")
    print(f"   Active Subscriptions: {len(scheduler.ibkr_subscriptions)}")
    
    status = scheduler.get_status()
    print(f"   Scheduler Running: {status['running']}")
    print(f"   Total Jobs: {status['total_jobs']}")
    
    # Monitor for 2 minutes
    print("\n4. Monitoring for 2 minutes...")
    print("   Press Ctrl+C to stop")
    print("   You should see:")
    print("   - Real-time bars every 5 seconds")
    print("   - Quote updates")
    print("   - Connection monitor messages every 30 seconds")
    print("\n" + "-" * 40)
    
    # Monitor loop
    for i in range(24):  # 24 x 5 seconds = 2 minutes
        time.sleep(5)
        
        # Every 30 seconds, show status
        if i % 6 == 0 and i > 0:
            print(f"\n[Status Check at {datetime.now().strftime('%H:%M:%S')}]")
            print(f"  IBKR Connected: {scheduler.ibkr_connected}")
            print(f"  Active Subs: {len(scheduler.ibkr_subscriptions)}")
            print("-" * 40)
    
    print("\n5. Stopping scheduler...")
    scheduler.stop()
    
    print("\n✅ TEST COMPLETED SUCCESSFULLY")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    if scheduler:
        try:
            scheduler.stop()
        except:
            pass

print("\n" + "=" * 60)
print("FULL TEST COMPLETE")
print("=" * 60)