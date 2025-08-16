#!/usr/bin/env python3
"""Test IBKR market data (bars + quotes) - Phase 3.3"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.ibkr_connection import IBKRConnectionManager


def test_market_data():
    """Test comprehensive market data from IBKR"""
    print("=== IBKR Market Data Test (Bars + Quotes) ===\n")
    
    # Check market hours
    now = datetime.now()
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if now.weekday() >= 5:
        print("\n⚠️  Weekend - limited data available")
    elif now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
        print("\n⚠️  Outside regular market hours")
    else:
        print("\n✓ Market is OPEN")
    
    print("\n" + "="*50)
    
    # Connect
    connection = IBKRConnectionManager()
    
    if not connection.connect_tws():
        print("Failed to connect to TWS")
        return False
    
    # Test symbols
    symbols = ['SPY', 'QQQ', 'IWM']
    
    print(f"\nTesting market data for: {', '.join(symbols)}")
    print("="*50)
    
    # Subscribe to quotes for all symbols
    quote_ids = {}
    for symbol in symbols:
        quote_id = connection.get_quotes(symbol)
        if quote_id:
            quote_ids[symbol] = quote_id
        time.sleep(0.5)  # Small delay between subscriptions
    
    print("\n" + "="*50)
    
    # Subscribe to bars for SPY only (to not overwhelm)
    bar_id = connection.subscribe_bars('SPY', '5 secs')
    
    print("\n" + "="*50)
    print("Collecting data for 20 seconds...")
    print("="*50 + "\n")
    
    # Collect data
    time.sleep(20)
    
    # Cleanup
    print("\n" + "="*50)
    print("Canceling subscriptions...")
    
    # Cancel quotes
    for symbol, req_id in quote_ids.items():
        connection.cancelMktData(req_id)
        print(f"  Canceled quotes for {symbol}")
    
    # Cancel bars
    if bar_id:
        connection.cancelRealTimeBars(bar_id)
        print(f"  Canceled bars for SPY")
    
    time.sleep(1)
    
    # Disconnect
    connection.disconnect_tws()
    print("\n✅ Market data test complete!")
    
    return True


if __name__ == "__main__":
    input("Press Enter to start market data test...")
    success = test_market_data()
    sys.exit(0 if success else 1)