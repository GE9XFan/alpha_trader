#!/usr/bin/env python3
"""Test IBKR connection through scheduler - BASIC TEST"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

print("=" * 60)
print("IBKR SCHEDULER BASIC CONNECTION TEST")
print(f"Time: {datetime.now()}")
print("=" * 60)

# First, just test importing
try:
    from src.data.scheduler import DataScheduler
    print("✓ Scheduler imported successfully")
except Exception as e:
    print(f"✗ Failed to import scheduler: {e}")
    sys.exit(1)

# Test creating scheduler instance
try:
    scheduler = DataScheduler(test_mode=True)
    print("✓ Scheduler instance created")
except Exception as e:
    print(f"✗ Failed to create scheduler: {e}")
    sys.exit(1)

# Test IBKR connection ONLY
try:
    print("\nTesting IBKR connection...")
    result = scheduler._connect_ibkr()
    
    if result:
        print("✅ IBKR CONNECTION SUCCESSFUL!")
        print(f"  Connected: {scheduler.ibkr_connected}")
        print(f"  Connection object exists: {scheduler.ibkr_connection is not None}")
        
        # Now disconnect cleanly
        print("\nTesting disconnection...")
        scheduler._disconnect_ibkr()
        print("✅ DISCONNECTION SUCCESSFUL!")
    else:
        print("❌ IBKR CONNECTION FAILED")
        print("Check that TWS is running and API is enabled")
        
except Exception as e:
    print(f"❌ ERROR during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)