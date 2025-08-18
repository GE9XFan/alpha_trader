#!/usr/bin/env python3
"""Test IBKR symbol subscriptions through scheduler"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler

print("=" * 60)
print("IBKR SCHEDULER SUBSCRIPTION TEST")
print(f"Time: {datetime.now()}")
print("=" * 60)

# Create scheduler in test mode
scheduler = DataScheduler(test_mode=True)

try:
    # Connect first
    print("\n1. Connecting to IBKR...")
    if not scheduler._connect_ibkr():
        print("❌ Failed to connect")
        sys.exit(1)
    print("✓ Connected")
    
    # Now test subscriptions
    print("\n2. Subscribing to market data...")
    num_subs = scheduler._subscribe_ibkr_data()
    
    print(f"\n✅ Created {num_subs} subscriptions")
    print(f"Active subscriptions: {len(scheduler.ibkr_subscriptions)}")
    
    # Show what we're subscribed to
    print("\nActive subscriptions:")
    for req_id, info in list(scheduler.ibkr_subscriptions.items())[:10]:  # Show first 10
        print(f"  {req_id}: {info}")
    
    # Let data flow for 30 seconds
    print("\n3. Collecting data for 30 seconds...")
    print("   You should see bar and quote data in the console...")
    time.sleep(30)
    
    # Disconnect cleanly
    print("\n4. Disconnecting...")
    scheduler._disconnect_ibkr()
    print("✓ Disconnected")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to cleanup
    try:
        scheduler._disconnect_ibkr()
    except:
        pass

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)