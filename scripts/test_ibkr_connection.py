#!/usr/bin/env python3
"""Test IBKR TWS connection - Phase 3.1"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.ibkr_connection import IBKRConnectionManager


def test_ibkr_connection():
    """Test basic connection to IBKR TWS"""
    print("=== IBKR Connection Test ===\n")
    
    # Create connection manager
    connection = IBKRConnectionManager()
    
    # IMPORTANT: Make sure TWS is running first!
    print("⚠️  Prerequisites:")
    print("  1. TWS must be running")
    print("  2. Go to TWS → File → Global Configuration → API → Settings")
    print("  3. Enable: 'Enable ActiveX and Socket Clients'")
    print("  4. Enable: 'Allow connections from localhost only'")
    print("  5. Socket port should be 7497 (paper) or 7496 (live)")
    print("  6. Click OK and restart TWS if needed\n")
    
    input("Press Enter when TWS is ready...")
    
    # Attempt connection
    success = connection.connect_tws()
    
    if success:
        print("\n✅ Connection test successful!")
        
        # Keep connection alive for a moment
        print("Connection stable for 3 seconds...")
        time.sleep(3)
        
        # Disconnect
        connection.disconnect_tws()
        print("\n✅ Clean disconnect successful!")
        
        return True
    else:
        print("\n❌ Connection failed!")
        print("\nTroubleshooting:")
        print("  1. Is TWS running?")
        print("  2. Is API enabled in TWS settings?")
        print("  3. Is port 7497 correct?")
        print("  4. Try restarting TWS")
        
        return False


if __name__ == "__main__":
    success = test_ibkr_connection()
    sys.exit(0 if success else 1)