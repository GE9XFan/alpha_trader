#!/usr/bin/env python3
"""Test IBKR live data ingestion during market hours - Phase 3.6"""

import sys
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.ibkr_connection import IBKRConnectionManager
from src.foundation.config_manager import ConfigManager


def check_market_status():
    """Check if market is open"""
    now = datetime.now()
    
    # Basic market hours check (9:30 AM - 4:00 PM ET, weekdays)
    if now.weekday() >= 5:
        return False, "Weekend - Market Closed"
    
    # Simple hour check (adjust for your timezone if needed)
    market_open = now.hour >= 9 and (now.hour > 9 or now.minute >= 30)
    market_close = now.hour < 16
    
    if market_open and market_close:
        return True, "Market Open"
    elif now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return False, "Pre-Market"
    else:
        return False, "After-Hours"


def test_live_data_ingestion():
    """Test live data collection and storage"""
    print("=== IBKR Live Data Ingestion Test ===\n")
    
    # Check market status
    is_open, status = check_market_status()
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market Status: {status}")
    
    if not is_open:
        print("\n⚠️  Best results during market hours (9:30 AM - 4:00 PM ET)")
    
    print("\n" + "="*50)
    
    # Connect to IBKR
    connection = IBKRConnectionManager()
    
    if not connection.connect_tws():
        print("Failed to connect to TWS")
        return False
    
    # Subscribe to data
    print("\nSubscribing to market data...")
    symbols = ['SPY', 'QQQ', 'IWM']
    
    # Subscribe to quotes
    for symbol in symbols:
        connection.get_quotes(symbol)
        time.sleep(0.5)
    
    # Subscribe to bars for SPY
    connection.subscribe_bars('SPY', '5 secs')
    
    # Collect data
    duration = 30 if is_open else 10
    print(f"\nCollecting data for {duration} seconds...")
    print("="*50 + "\n")
    
    time.sleep(duration)
    
    # Check what was stored
    print("\n" + "="*50)
    print("Checking database...")
    
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        # Check bars
        result = conn.execute(text("""
            SELECT COUNT(*), MAX(timestamp) 
            FROM ibkr_bars_5sec 
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """))
        bar_count, last_bar = result.fetchone()
        
        # Check quotes
        result = conn.execute(text("""
            SELECT COUNT(*), COUNT(DISTINCT symbol) 
            FROM ibkr_quotes 
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """))
        quote_count, symbol_count = result.fetchone()
    
    print(f"\nDatabase Summary:")
    print(f"  5-sec bars stored: {bar_count}")
    print(f"  Last bar: {last_bar}")
    print(f"  Quotes stored: {quote_count}")
    print(f"  Symbols with quotes: {symbol_count}")
    
    # Disconnect
    connection.disconnect_tws()
    
    print("\n✅ Live data test complete!")
    print("\nNext steps:")
    print("  - Run during market hours for real data")
    print("  - Phase 4 will add scheduling for automatic collection")
    
    return True


if __name__ == "__main__":
    success = test_live_data_ingestion()
    sys.exit(0 if success else 1)