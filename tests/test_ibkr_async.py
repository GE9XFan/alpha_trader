#!/usr/bin/env python3
"""
IBKR Async Connection Test - EXACTLY like data_ingestion.py
"""

import asyncio
from ib_insync import IB, Stock
import time


async def test_connection_like_data_ingestion():
    """Test IBKR connection exactly like data_ingestion.py does it"""
    print("\n" + "="*50)
    print("IBKR ASYNC CONNECTION TEST")  
    print("(Using EXACT same method as data_ingestion.py)")
    print("="*50)
    
    # Create IB instance exactly like data_ingestion.py line 49
    ib = IB()
    
    # Connection parameters exactly like data_ingestion.py
    host = '127.0.0.1'
    port = 7497
    client_id = 999  # Using different ID to avoid conflicts
    timeout = 10
    
    print(f"\nConnection parameters:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Client ID: {client_id}")
    print(f"  Timeout: {timeout}s")
    
    # Try to connect EXACTLY like data_ingestion.py lines 166-171
    print("\nAttempting connection using connectAsync...")
    try:
        await ib.connectAsync(
            host=host,
            port=port,
            clientId=client_id,
            timeout=timeout
        )
        
        # Check if connected exactly like data_ingestion.py line 173
        if ib.isConnected():
            print("✅ Connected successfully!")
            
            # Get account info like data_ingestion.py lines 181-185
            accounts = ib.managedAccounts()
            if accounts:
                print(f"✅ Account found: {accounts[0]}")
            else:
                print("✅ Connected (no account info)")
            
            # Test qualifying a contract like data_ingestion.py line 259
            print("\nTesting contract qualification...")
            contract = Stock('SPY', 'SMART', 'USD')
            contracts = await ib.qualifyContractsAsync(contract)
            if contracts:
                print(f"✅ SPY contract qualified")
                contract = contracts[0]
                
                # Request market data like data_ingestion.py line 275
                print("\nRequesting market data...")
                ticker = ib.reqMktData(
                    contract,
                    genericTickList='233',  # RTVolume like data_ingestion
                    snapshot=False,
                    regulatorySnapshot=False
                )
                
                # Wait a bit for data
                print("Waiting for data...")
                await asyncio.sleep(2)
                
                # Check ticker data
                if ticker.bid:
                    print(f"  Bid: ${ticker.bid}")
                if ticker.ask:
                    print(f"  Ask: ${ticker.ask}")
                if ticker.last:
                    print(f"  Last: ${ticker.last}")
                
                # Cancel market data
                ib.cancelMktData(ticker)
                print("✅ Market data test complete")
            
            print("\n" + "="*50)
            print("✅ ALL TESTS PASSED!")
            print("Connection method matches data_ingestion.py exactly")
            print("="*50)
            
            return True
            
        else:
            print("❌ Connection check failed (not connected)")
            return False
            
    except asyncio.TimeoutError:
        print("❌ Connection timeout!")
        print("\nPossible issues:")
        print("1. Is IBKR Gateway/TWS running?")
        print("2. Is API access enabled in Gateway/TWS?")
        print("3. Is port 7497 correct? (7496 for live, 7497 for paper)")
        print("4. Check Gateway/TWS for any error messages")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Disconnect like data_ingestion.py line 1038
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from IBKR")


async def main():
    """Main async function"""
    print("""
This test connects to IBKR using the EXACT same method
as data_ingestion.py uses. If data_ingestion.py works,
this should work too.

Requirements:
- IBKR Gateway or TWS running
- Paper trading port 7497
- API permissions enabled
    """)
    
    input("\nPress Enter to continue...")
    
    # Run the test
    success = await test_connection_like_data_ingestion()
    return success


if __name__ == '__main__':
    # Run using asyncio.run() - standard Python async approach
    success = asyncio.run(main())
    exit(0 if success else 1)