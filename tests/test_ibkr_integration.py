#!/usr/bin/env python3
"""
REAL IBKR Integration Test - NO MOCKS!
This actually connects to IBKR and places orders in your paper trading account.

REQUIREMENTS:
1. IBKR Gateway or TWS running on localhost:7497 (paper trading)
2. Paper trading account with some cash
3. API permissions enabled in IBKR

WARNING: This will place REAL orders in your paper trading account!
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import redis.asyncio as aioredis

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution import ExecutionManager, PositionManager


async def test_real_ibkr_connection():
    """Test REAL connection to IBKR - no mocks!"""
    print("\n" + "="*60)
    print("REAL IBKR INTEGRATION TEST - PAPER TRADING")
    print("="*60)
    
    # Load real config
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Connect to real Redis
    redis = aioredis.Redis(
        host=config['redis']['host'],
        port=config['redis']['port'],
        decode_responses=True
    )
    
    # Create REAL ExecutionManager
    exec_manager = ExecutionManager(config, redis)
    
    print("\n1. Testing IBKR Connection...")
    print("-" * 40)
    
    # Try to connect to IBKR
    connected = await exec_manager.connect_ibkr()
    
    if not connected:
        print("❌ Failed to connect to IBKR!")
        print("Make sure:")
        print("  - IBKR Gateway/TWS is running")
        print("  - Paper trading port is 7497")
        print("  - API permissions are enabled")
        return False
    
    print("✅ Connected to IBKR successfully!")
    
    # Get account info
    account_value = await redis.get('account:value')
    buying_power = await redis.get('account:buying_power')
    
    print(f"   Account Value: ${account_value}")
    print(f"   Buying Power: ${buying_power}")
    
    return exec_manager, redis


async def test_real_order_placement(exec_manager, redis):
    """Place a REAL order through IBKR!"""
    print("\n2. Testing Real Order Placement...")
    print("-" * 40)
    
    # Check market hours first
    import pytz
    import math
    from datetime import datetime, timedelta
    
    market_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(market_tz)
    weekday = now_et.weekday()
    current_time = now_et.time()
    
    market_open = False
    if weekday < 5:  # Weekday
        if datetime.strptime('09:30', '%H:%M').time() <= current_time <= datetime.strptime('16:00', '%H:%M').time():
            market_open = True
            print("✅ Market is OPEN")
        elif current_time < datetime.strptime('09:30', '%H:%M').time():
            print("⚠️  WARNING: Market is CLOSED (pre-market)")
            print("   Orders will be queued until 9:30 AM ET")
        else:
            print("⚠️  WARNING: Market is CLOSED (after-hours)")
            print("   Orders will be queued for next trading day")
    else:
        print("⚠️  WARNING: Market is CLOSED (weekend)")
        print("   Orders will be queued until Monday")
    
    # Get next Friday expiry (0DTE or nearest)
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0 and today.hour >= 16:  # If it's Friday after market close
        days_until_friday = 7
    elif days_until_friday == 0:  # Friday during market hours
        days_until_friday = 0  # True 0DTE!
    
    expiry_date = today + timedelta(days=days_until_friday if days_until_friday > 0 else 0)
    expiry_str = expiry_date.strftime('%Y%m%d')
    
    # First, get current SPY price to determine appropriate strike
    print("\n   Getting current SPY price for strike selection...")
    from ib_insync import Stock
    spy_stock = Stock('SPY', 'SMART', 'USD')
    spy_qualified = await exec_manager.ib.qualifyContractsAsync(spy_stock)
    
    spy_price = 657.0  # Default fallback
    if spy_qualified:
        spy_ticker = exec_manager.ib.reqMktData(spy_qualified[0], '', False, False)
        await asyncio.sleep(2)
        
        # Get current SPY price
        if spy_ticker.last and not math.isnan(spy_ticker.last):
            spy_price = spy_ticker.last
            print(f"   Current SPY: ${spy_price:.2f}")
        elif spy_ticker.bid and spy_ticker.bid > 0:
            spy_price = (spy_ticker.bid + spy_ticker.ask) / 2
            print(f"   Current SPY (mid): ${spy_price:.2f}")
        else:
            # Fallback to Redis if no market data
            spy_data = await redis.get('market:SPY:ticker')
            if spy_data:
                spy_json = json.loads(spy_data)
                spy_price = float(spy_json.get('last', spy_json.get('mid', 657.0)))
                print(f"   Current SPY (from Redis): ${spy_price:.2f}")
            else:
                print(f"   ⚠️  No SPY price available, using fallback: ${spy_price}")
    
    # Round strike to nearest dollar
    strike_price = round(spy_price)
    print(f"   Selected Strike: ${strike_price}")
    
    test_signal = {
        'id': f'test_{int(time.time())}',
        'symbol': 'SPY',
        'side': 'LONG',
        'confidence': 75,
        'strategy': 'TEST_0DTE' if days_until_friday <= 1 else 'TEST_OPTION',
        'contract': {
            'type': 'option',  # OPTIONS - this is what we trade!
            'symbol': 'SPY',
            'expiry': expiry_str,
            'strike': float(strike_price),  # Dynamic strike
            'right': 'C'  # Call option
        },
        'entry': None,  # Will use market price
        'stop': None,   # Will calculate
        'targets': [],  # No targets for test
        'ts': time.time() * 1000,
        'position_size': 1  # 1 contract for test
    }
    
    print(f"   Symbol: {test_signal['symbol']}")
    print(f"   Type: CALL OPTION (expiry: {expiry_str})")
    print(f"   Strike: ${test_signal['contract']['strike']}")
    print(f"   Side: {test_signal['side']}")
    
    # Create contract
    print("\n3. Creating and Qualifying Contract...")
    print("-" * 40)
    
    contract = await exec_manager.create_ib_contract(test_signal['contract'])
    
    if not contract:
        print("❌ Failed to create/qualify contract!")
        return False
    
    print("✅ Contract qualified successfully!")
    print(f"   Symbol: {contract.symbol}")
    print(f"   Exchange: {contract.exchange}")
    print(f"   Currency: {contract.currency}")
    
    # Get current market data
    print("\n4. Getting Option Market Data...")
    print("-" * 40)
    
    ticker = exec_manager.ib.reqMktData(contract, '', False, False)
    await asyncio.sleep(3)  # Wait longer for option data
    
    # Check if we have valid option market data
    has_valid_data = False
    option_price = None
    
    if ticker.last and not math.isnan(ticker.last) and ticker.last > 0:
        has_valid_data = True
        option_price = ticker.last
        print(f"✅ IBKR market data received!")
        print(f"   Last: ${ticker.last:.2f}")
        print(f"   Bid: ${ticker.bid:.2f if ticker.bid and ticker.bid > 0 else 'N/A'}")
        print(f"   Ask: ${ticker.ask:.2f if ticker.ask and ticker.ask > 0 else 'N/A'}")
    elif ticker.bid and ticker.bid > 0 and ticker.ask and ticker.ask > 0:
        has_valid_data = True
        option_price = (ticker.bid + ticker.ask) / 2
        print(f"✅ IBKR bid/ask available!")
        print(f"   Bid: ${ticker.bid:.2f}")
        print(f"   Ask: ${ticker.ask:.2f}")
        print(f"   Mid: ${option_price:.2f}")
    else:
        print("⚠️  No IBKR option data available")
        print(f"   Last: {ticker.last if ticker.last else 'N/A'}")
        print(f"   Bid: {ticker.bid if ticker.bid else 'N/A'}")
        print(f"   Ask: {ticker.ask if ticker.ask else 'N/A'}")
        
        # Try Alpha Vantage fallback - PROPERLY SEARCH ALL CONTRACTS!
        print("\n   Checking Alpha Vantage for option data...")
        av_key = f"options:SPY:calls"
        av_data = await redis.get(av_key)
        
        if av_data:
            calls = json.loads(av_data)
            print(f"   Found {len(calls)} call options in Redis!")
            
            # Find the BEST matching option (closest expiry AND strike)
            target_strike = test_signal['contract']['strike']
            target_expiry = expiry_str  # e.g., '20250919'
            
            best_option = None
            best_score = float('inf')
            
            # First, try to find options with matching or close expiry
            expiry_groups = {}
            for call in calls:
                # Alpha Vantage uses 'expiration' field
                call_expiry = call.get('expiration', '')
                if call_expiry:
                    if call_expiry not in expiry_groups:
                        expiry_groups[call_expiry] = []
                    expiry_groups[call_expiry].append(call)
            
            print(f"   Found {len(expiry_groups)} different expiry dates")
            
            # Find closest expiry date
            from datetime import datetime
            target_date = datetime.strptime(target_expiry, '%Y%m%d')
            closest_expiry = None
            min_days_diff = float('inf')
            
            for exp_str in expiry_groups.keys():
                try:
                    # Parse expiry (format: YYYY-MM-DD or YYYYMMDD)
                    if '-' in exp_str:
                        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                        exp_yyyymmdd = exp_date.strftime('%Y%m%d')
                    else:
                        exp_date = datetime.strptime(exp_str, '%Y%m%d')
                        exp_yyyymmdd = exp_str
                    
                    days_diff = abs((exp_date - target_date).days)
                    if days_diff < min_days_diff:
                        min_days_diff = days_diff
                        closest_expiry = exp_str
                except:
                    continue
            
            if closest_expiry:
                print(f"   Closest expiry: {closest_expiry} ({min_days_diff} days away)")
                
                # Now find best strike in that expiry
                expiry_options = expiry_groups[closest_expiry]
                print(f"   {len(expiry_options)} strikes available for this expiry")
                
                # Sort by strike difference
                valid_options = []
                for opt in expiry_options:
                    strike = float(opt.get('strike', 0))
                    bid = float(opt.get('bid', 0))
                    ask = float(opt.get('ask', 0))
                    
                    if strike > 0 and bid > 0 and ask > 0:
                        strike_diff = abs(strike - target_strike)
                        valid_options.append((strike_diff, opt))
                
                valid_options.sort(key=lambda x: x[0])
                
                if valid_options:
                    # Show top 3 closest strikes
                    print(f"\\n   Top strikes near ${target_strike}:")
                    for i, (diff, opt) in enumerate(valid_options[:3]):
                        strike = float(opt.get('strike'))
                        bid = float(opt.get('bid'))
                        ask = float(opt.get('ask'))
                        volume = opt.get('volume', 0)
                        oi = opt.get('open_interest', 0)
                        print(f"   {i+1}. Strike ${strike:.0f}: Bid=${bid:.2f}, Ask=${ask:.2f}, Vol={volume}, OI={oi}")
                    
                    # Use the closest one
                    best_option = valid_options[0][1]
                    
            if best_option:
                av_bid = float(best_option.get('bid', 0))
                av_ask = float(best_option.get('ask', 0))
                av_strike = float(best_option.get('strike', 0))
                
                option_price = (av_bid + av_ask) / 2
                print(f"\\n✅ Using Alpha Vantage option:")
                print(f"   Strike: ${av_strike:.0f} (target was ${target_strike:.0f})")
                print(f"   Expiry: {closest_expiry}")
                print(f"   Bid: ${av_bid:.2f}")
                print(f"   Ask: ${av_ask:.2f}")
                print(f"   Mid: ${option_price:.2f}")
                print(f"   Volume: {best_option.get('volume', 'N/A')}")
                print(f"   Open Interest: {best_option.get('open_interest', 'N/A')}")
                
                # Update the test signal to use the actual available strike
                test_signal['contract']['strike'] = av_strike
                print(f"\\n   ℹ️  Updated test strike to ${av_strike:.0f} (actual available option)")
            else:
                print("   No valid options found with bid/ask data")
        else:
            print("   No Alpha Vantage data available in Redis")
        
        if not option_price:
            print("\n⚠️  WARNING: No option pricing available!")
            print("   Order may be rejected or filled at unexpected price")
            if not market_open:
                print("   This is expected when market is closed")
    
    # Calculate order size (1 option contract = 100 shares)
    print("\n5. Placing Test Order (1 option contract)...")
    print("-" * 40)
    
    from ib_insync import MarketOrder
    order = MarketOrder('BUY', 1)  # Buy 1 option contract
    
    # Place the order
    trade = exec_manager.ib.placeOrder(contract, order)
    order_id = trade.order.orderId
    
    print(f"✅ Order placed! Order ID: {order_id}")
    print(f"   Action: BUY")
    print(f"   Quantity: 1 contract (100 shares)")
    print(f"   Type: MARKET")
    
    # Monitor order status
    print("\n6. Monitoring Order Status...")
    print("-" * 40)
    
    max_wait = 10  # Wait up to 10 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        await asyncio.sleep(0.5)
        print(f"   Status: {trade.orderStatus.status}")
        
        if trade.isDone():
            if trade.orderStatus.status == 'Filled':
                print(f"✅ Order FILLED!")
                
                # Get fill details
                if trade.fills:
                    fill = trade.fills[-1]
                    print(f"   Fill Price: ${fill.execution.price}")
                    print(f"   Fill Time: {fill.time}")
                    if fill.commissionReport:
                        print(f"   Commission: ${fill.commissionReport.commission}")
                
                # Create position record for the OPTION
                position_id = f"test_pos_{int(time.time())}"
                position = {
                    'id': position_id,
                    'symbol': 'SPY',
                    'contract_type': 'option',
                    'strike': test_signal['contract']['strike'],
                    'expiry': test_signal['contract']['expiry'],
                    'right': test_signal['contract']['right'],
                    'side': 'LONG',
                    'quantity': 1,  # 1 option contract
                    'entry_price': fill.execution.price if trade.fills else ticker.last,
                    'entry_time': datetime.now().isoformat(),
                    'order_id': order_id,
                    'status': 'OPEN'
                }
                
                await redis.setex(
                    f'positions:open:SPY:{position_id}',
                    3600,  # 1 hour TTL
                    json.dumps(position)
                )
                
                print(f"\n✅ Position created: {position_id}")
                
                # Test stop loss order
                await test_stop_loss(exec_manager, contract, position)
                
                # Close the position
                await test_close_position(exec_manager, contract, position)
                
                return True
                
            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                print(f"❌ Order {trade.orderStatus.status}")
                return False
    
    print("⏱️  Order timed out")
    # Cancel if still pending
    exec_manager.ib.cancelOrder(order)
    return False


async def test_stop_loss(exec_manager, contract, position):
    """Test placing a stop loss order"""
    print("\n7. Testing Stop Loss Order...")
    print("-" * 40)
    
    entry_price = position['entry_price']
    stop_price = round(entry_price * 0.99, 2)  # 1% stop loss
    
    from ib_insync import StopOrder
    stop_order = StopOrder('SELL', 1, stop_price)
    
    stop_trade = exec_manager.ib.placeOrder(contract, stop_order)
    stop_order_id = stop_trade.order.orderId
    
    print(f"✅ Stop loss placed! Order ID: {stop_order_id}")
    print(f"   Stop Price: ${stop_price}")
    print(f"   Action: SELL")
    
    # Wait a moment to verify it's accepted
    await asyncio.sleep(1)
    print(f"   Status: {stop_trade.orderStatus.status}")
    
    # Cancel the stop for cleanup
    exec_manager.ib.cancelOrder(stop_trade.order)
    await asyncio.sleep(1)
    print("   Stop order cancelled for cleanup")
    
    return stop_order_id


async def test_close_position(exec_manager, contract, position):
    """Test closing the OPTION position"""
    print("\n8. Closing Test Option Position...")
    print("-" * 40)
    
    from ib_insync import MarketOrder
    close_order = MarketOrder('SELL', 1)  # Sell 1 option contract to close
    
    close_trade = exec_manager.ib.placeOrder(contract, close_order)
    
    # Wait for fill
    max_wait = 10
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        await asyncio.sleep(0.5)
        
        if close_trade.isDone():
            if close_trade.orderStatus.status == 'Filled':
                print(f"✅ Position CLOSED!")
                
                if close_trade.fills:
                    fill = close_trade.fills[-1]
                    exit_price = fill.execution.price
                    entry_price = position['entry_price']
                    # Option P&L: (exit - entry) * 100 shares * 1 contract
                    pnl = (exit_price - entry_price) * 100 * 1
                    
                    print(f"   Exit Price: ${exit_price} per contract")
                    print(f"   Entry Price: ${entry_price} per contract")
                    print(f"   P&L: ${pnl:.2f} (1 contract = 100 shares)")
                    
                    if fill.commissionReport:
                        commission = fill.commissionReport.commission
                        print(f"   Commission: ${commission}")
                        pnl -= commission
                        print(f"   Net P&L: ${pnl:.2f}")
                
                return True
            break
    
    return False


async def test_option_order():
    """Test placing an option order"""
    print("\n9. Testing Option Order (Optional)...")
    print("-" * 40)
    
    # This would test actual option orders
    # Requires finding valid option contracts
    print("   [Skipped for basic test - add if needed]")


async def main():
    """Run the full integration test"""
    try:
        # Test connection
        result = await test_real_ibkr_connection()
        if not result:
            print("\n❌ Connection test failed!")
            return
        
        exec_manager, redis = result
        
        # Test order placement
        success = await test_real_order_placement(exec_manager, redis)
        
        if success:
            print("\n" + "="*60)
            print("✅ ALL INTEGRATION TESTS PASSED!")
            print("="*60)
            print("\nYour execution system is working with REAL IBKR!")
            print("Check your paper trading account to verify the trades.")
        else:
            print("\n❌ Some tests failed - check the output above")
        
        # Cleanup
        exec_manager.ib.disconnect()
        await redis.close()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    IBKR INTEGRATION TEST                      ║
║                                                               ║
║  This will place REAL orders in your paper trading account!  ║
║                                                               ║
║  Requirements:                                                ║
║  1. IBKR Gateway/TWS running on localhost:7497               ║
║  2. Paper trading account active                             ║
║  3. API permissions enabled                                  ║
║  4. Redis running on localhost:6379                          ║
║                                                               ║
║  Press Ctrl+C to cancel, or wait to continue...              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    time.sleep(3)
    
    # Run the test
    asyncio.run(main())