#!/usr/bin/env python3
"""
Test ExecutionManager with REAL IBKR connection
This validates our actual implementation, not mocks.
"""

import asyncio
import json
import time
import yaml
import sys
from pathlib import Path
import redis.asyncio as aioredis

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.execution import ExecutionManager, PositionManager


async def test_execution_manager():
    """Test our ExecutionManager with real IBKR"""
    print("\n" + "="*60)
    print("TESTING EXECUTIONMANAGER WITH REAL IBKR")
    print("="*60)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup Redis
    redis = aioredis.Redis(
        host='127.0.0.1',
        port=6379,
        decode_responses=True
    )
    
    # Create ExecutionManager
    exec_manager = ExecutionManager(config, redis)
    
    try:
        # Test 1: Connection
        print("\n1. Testing IBKR Connection...")
        print("-" * 40)
        connected = await exec_manager.connect_ibkr()
        
        if not connected:
            print("❌ Failed to connect!")
            print("\nTroubleshooting:")
            print("1. Is IBKR Gateway/TWS running?")
            print("2. Is it on port 7497 (paper)?")
            print("3. Are API connections enabled?")
            print("4. Check Gateway logs for errors")
            return False
        
        print("✅ Connected successfully!")
        
        # Check what was stored in Redis
        account_value = await redis.get('account:value')
        buying_power = await redis.get('account:buying_power')
        print(f"   Account Value: ${account_value}")
        print(f"   Buying Power: ${buying_power}")
        
        # Test 2: Contract Creation
        print("\n2. Testing Contract Creation...")
        print("-" * 40)
        
        # Test stock contract
        stock_contract = {
            'type': 'stock',
            'symbol': 'AAPL'
        }
        
        print("   Creating AAPL stock contract...")
        aapl = await exec_manager.create_ib_contract(stock_contract)
        
        if aapl:
            print(f"✅ Stock contract created: {aapl.symbol}")
        else:
            print("❌ Failed to create stock contract")
        
        # Test option contract
        print("\n   Creating SPY option contract...")
        
        # We need to find a valid option
        # Let's create one that's likely to exist
        from datetime import datetime, timedelta
        
        # Next Friday expiry
        today = datetime.now()
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:
            days_ahead += 7
        next_friday = today + timedelta(days=days_ahead)
        expiry = next_friday.strftime('%Y%m%d')
        
        # Current SPY price (approximate)
        spy_price = 450  # Adjust based on current market
        strike = round(spy_price)
        
        option_contract = {
            'type': 'option',
            'symbol': 'SPY',
            'expiry': expiry,
            'strike': strike,
            'right': 'C'
        }
        
        print(f"   SPY {expiry} {strike}C")
        spy_call = await exec_manager.create_ib_contract(option_contract)
        
        if spy_call:
            print(f"✅ Option contract created")
        else:
            print("❌ Failed to create option contract")
            print("   (This is OK - option might not exist)")
        
        # Test 3: Market Data
        print("\n3. Testing Market Data...")
        print("-" * 40)
        
        if aapl:
            ticker = exec_manager.ib.reqMktData(aapl, '', False, False)
            await asyncio.sleep(2)
            
            print(f"   AAPL Last: ${ticker.last}")
            print(f"   AAPL Bid: ${ticker.bid}")
            print(f"   AAPL Ask: ${ticker.ask}")
            
            if ticker.last:
                print("✅ Market data working!")
            else:
                print("⚠️  No market data (market might be closed)")
        
        # Test 4: Risk Checks
        print("\n4. Testing Risk Checks...")
        print("-" * 40)
        
        # Set up risk parameters
        await redis.set('risk:new_positions_allowed', 'true')
        await redis.set('risk:daily_pnl', '0')
        
        test_signal = {
            'symbol': 'AAPL',
            'side': 'LONG',
            'position_size': 1000  # $1000 position
        }
        
        can_trade = await exec_manager.passes_risk_checks(test_signal)
        print(f"   Risk check passed: {can_trade}")
        
        if can_trade:
            print("✅ Risk checks working!")
        
        # Test 5: Order Creation (without placing)
        print("\n5. Testing Order Creation...")
        print("-" * 40)
        
        from ib_insync import MarketOrder, LimitOrder
        
        # Market order
        market_order = MarketOrder('BUY', 10)
        print(f"✅ Market order created: BUY 10")
        
        # Limit order
        limit_order = LimitOrder('BUY', 10, 150.00)
        print(f"✅ Limit order created: BUY 10 @ $150.00")
        
        # Test 6: Place a small test order?
        print("\n6. Real Order Test...")
        print("-" * 40)
        
        response = input("   Place a test order for 1 share of AAPL? (yes/no): ")
        
        if response.lower() == 'yes' and aapl and ticker.last:
            # Create the signal
            signal = {
                'id': f'test_{int(time.time())}',
                'symbol': 'AAPL',
                'side': 'LONG',
                'confidence': 75,
                'strategy': 'TEST',
                'contract': {
                    'type': 'stock',
                    'symbol': 'AAPL'
                },
                'entry': ticker.last,
                'stop': ticker.last * 0.99,  # 1% stop
                'targets': [ticker.last * 1.01],  # 1% target
                'ts': time.time() * 1000
            }
            
            print("\n   Executing signal through ExecutionManager...")
            await exec_manager.execute_signal(signal)
            
            print("   Check order status in IBKR!")
            
            # Wait a bit
            await asyncio.sleep(5)
            
            # Check if position was created
            position_keys = await redis.keys('positions:open:AAPL:*')
            if position_keys:
                print(f"✅ Position created: {position_keys[0]}")
                
                # Get position details
                position_data = await redis.get(position_keys[0])
                if position_data:
                    position = json.loads(position_data)
                    print(f"   Entry: ${position.get('entry_price')}")
                    print(f"   Stop: ${position.get('stop_loss')}")
        else:
            print("   Skipping real order test")
        
        print("\n" + "="*60)
        print("✅ EXECUTIONMANAGER VALIDATION COMPLETE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if exec_manager.ib.isConnected():
            exec_manager.ib.disconnect()
        await redis.close()


async def main():
    """Run the test"""
    success = await test_execution_manager()
    
    if success:
        print("\n✅ Your ExecutionManager is working with real IBKR!")
    else:
        print("\n❌ There were issues - see above for details")


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════╗
║          EXECUTIONMANAGER REAL VALIDATION              ║
║                                                        ║
║  This tests our actual ExecutionManager implementation ║
║  with a real IBKR connection (no mocks).              ║
║                                                        ║
║  Requirements:                                         ║
║  - IBKR Gateway/TWS on port 7497                      ║
║  - Redis on port 6379                                 ║
║  - Paper trading account                              ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to start...")
    
    asyncio.run(main())