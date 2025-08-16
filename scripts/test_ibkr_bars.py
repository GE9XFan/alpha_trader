#!/usr/bin/env python3
"""Test IBKR real-time bar data - Phase 3.2"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.ibkr_connection import IBKRConnectionManager


def test_realtime_bars():
    """Test real-time bar data from IBKR"""
    print("=== IBKR Real-Time Bars Test ===\n")
    
    # Create connection
    connection = IBKRConnectionManager()
    
    # Connect to TWS
    if not connection.connect_tws():
        print("Failed to connect to TWS")
        return False
    
    print("\n" + "="*50)
    print("Testing real-time 5-second bars for SPY")
    print("Will collect data for 30 seconds...")
    print("="*50 + "\n")
    
    # Subscribe to SPY bars
    req_id = connection.subscribe_bars('SPY', '5 secs')
    
    if req_id:
        print(f"✓ Subscription successful (ID: {req_id})")
        print("\nIncoming bar data:\n")
        
        # Collect data for 30 seconds
        time.sleep(30)
        
        print("\n" + "="*50)
        print("Test complete - stopping data feed")
        
        # Cancel the subscription
        connection.cancelRealTimeBars(req_id)
        time.sleep(1)
    
    # Disconnect
    connection.disconnect_tws()
    print("\n✅ Bar data test complete!")
    
    return True


if __name__ == "__main__":
    # Note: Market must be open for real-time bars
    from datetime import datetime
    
    now = datetime.now()
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if market is likely open (rough check)
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        print("\n⚠️  WARNING: It's the weekend - market is closed.")
        print("Real-time bars will only show cached/last values.")
        print("For best results, run during market hours (9:30 AM - 4:00 PM ET weekdays)\n")
    
    input("Press Enter to continue...")
    
    success = test_realtime_bars()
    sys.exit(0 if success else 1)