#!/usr/bin/env python3
"""
Test IBKR Gateway Connection
Verifies connectivity before running full ingestion
"""

import asyncio
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ib_insync import IB, Stock


async def test_connection():
    """Test IBKR Gateway connection"""
    
    print("Testing IBKR Gateway connection...")
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ib = IB()
    
    try:
        # Try to connect
        await ib.connectAsync(
            config['ibkr']['host'],
            config['ibkr']['port'],
            clientId=config['ibkr']['client_id']
        )
        
        print(f"✓ Connected to IBKR Gateway at {config['ibkr']['host']}:{config['ibkr']['port']}")
        
        # Get account info
        account = ib.managedAccounts()
        if account:
            print(f"✓ Account: {account[0]}")
        
        # Test market data subscription
        contract = Stock('SPY', 'SMART', 'USD')
        await ib.qualifyContractsAsync(contract)
        
        # Request market data
        ticker = ib.reqMktData(contract, '', False, False)
        await asyncio.sleep(2)
        
        if ticker.last:
            print(f"✓ SPY last price: ${ticker.last}")
        else:
            print("⚠ No market data received - markets may be closed")
        
        print("✓ Connection test successful!")
        
        # Clean up
        ib.cancelMktData(contract)
        ib.disconnect()
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure IBKR Gateway or TWS is running")
        print("2. Enable API connections in Gateway/TWS settings")
        print(f"3. Check port {config['ibkr']['port']} is correct")
        print("   - TWS Live: 7496")
        print("   - TWS Paper: 7497")
        print("   - Gateway Live: 4001")
        print("   - Gateway Paper: 4002")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)