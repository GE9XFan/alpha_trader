#!/usr/bin/env python3
"""
Simple IBKR Connection Test
Tests connection using async approach like data_ingestion.py
"""

from ib_insync import IB, Stock, MarketOrder, util
import asyncio
import time


async def test_simple_connection():
    """Test basic IBKR connection using async approach"""
    print("\n" + "="*50)
    print("SIMPLE IBKR CONNECTION TEST")
    print("="*50)
    
    ib = IB()
    
    try:
        # Try to connect using async method like data_ingestion.py
        print("\n1. Attempting connection to IBKR Gateway...")
        print("   Host: 127.0.0.1")
        print("   Port: 7497 (paper trading)")
        print("   Client ID: 999 (test)")
        
        # Use connectAsync with timeout like data_ingestion.py
        await ib.connectAsync(
            host='127.0.0.1',
            port=7497,
            clientId=999,
            timeout=10
        )
        
        if ib.isConnected():
            print("✅ Connected successfully!")
        else:
            print("❌ Connection failed!")
            return False
        
        # Get account info
        print("\n2. Getting account information...")
        accounts = ib.managedAccounts()
        if accounts:
            print(f"   Account: {accounts[0]}")
        
        account_values = ib.accountValues()
        for av in account_values[:10]:  # Show first 10 values
            if av.tag in ['NetLiquidation', 'BuyingPower', 'TotalCashValue']:
                print(f"   {av.tag}: {av.value} {av.currency}")
        
        # Get positions
        print("\n3. Getting current positions...")
        positions = ib.positions()
        
        if positions:
            for pos in positions:
                print(f"   {pos.contract.symbol}: {pos.position} @ {pos.avgCost}")
        else:
            print("   No open positions")
        
        # Test creating a contract
        print("\n4. Testing contract creation...")
        spy = Stock('SPY', 'SMART', 'USD')
        
        # Qualify the contract using async
        qualified = await ib.qualifyContractsAsync(spy)
        if qualified:
            print(f"✅ SPY contract qualified")
            contract = qualified[0]
            
            # Get market data
            print("\n5. Getting market data for SPY...")
            ticker = ib.reqMktData(contract, '', False, False)
            
            # Wait for data using asyncio
            await asyncio.sleep(2)
            
            if ticker.last:
                print(f"   Last: ${ticker.last}")
            if ticker.bid:
                print(f"   Bid: ${ticker.bid}")
            if ticker.ask:
                print(f"   Ask: ${ticker.ask}")
            if ticker.volume:
                print(f"   Volume: {ticker.volume}")
            
            # Test order creation (but don't place it)
            print("\n6. Creating test order (not placing)...")
            order = MarketOrder('BUY', 1)
            print(f"✅ Order created: BUY 1 SPY at MARKET")
            
            # Ask if user wants to place the order
            response = input("\n   Do you want to place this test order? (yes/no): ")
            
            if response.lower() == 'yes':
                print("\n7. Placing order...")
                trade = ib.placeOrder(contract, order)
                
                # Wait for fill
                for i in range(10):
                    await asyncio.sleep(1)
                    print(f"   Status: {trade.orderStatus.status}")
                    
                    if trade.isDone():
                        if trade.orderStatus.status == 'Filled':
                            print(f"✅ Order filled!")
                            if trade.fills:
                                fill = trade.fills[-1]
                                print(f"   Price: ${fill.execution.price}")
                                print(f"   Time: {fill.time}")
                            
                            # Close the position immediately
                            print("\n8. Closing position...")
                            close_order = MarketOrder('SELL', 1)
                            close_trade = ib.placeOrder(contract, close_order)
                            
                            await asyncio.sleep(3)
                            if close_trade.isDone():
                                print(f"✅ Position closed!")
                        break
            else:
                print("   Order not placed (user declined)")
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from IBKR")


def main():
    """Main entry point that properly runs the async function"""
    print("""
This test will:
1. Connect to IBKR Gateway/TWS
2. Get account information
3. Create a SPY contract
4. Get market data
5. Optionally place a test order

Requirements:
- IBKR Gateway or TWS running
- Paper trading port 7497
- API permissions enabled
    """)
    
    input("Press Enter to continue...")
    
    # Run the async function properly
    # ib_insync handles its own event loop, so we use util.run()
    util.startLoop()  # Start ib_insync's event loop
    try:
        success = util.run(test_simple_connection())
        return success
    finally:
        util.stopLoop()  # Clean up the event loop


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)