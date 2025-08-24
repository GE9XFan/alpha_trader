#!/usr/bin/env python3
"""Morning health checks - Operations Manual"""
import asyncio
import sys
sys.path.append('.')

from src.monitoring.health_checks import health_checker
from src.data.market_data import market_data
from src.data.alpha_vantage_client import av_client
from src.data.options_data import options_data
from src.trading.risk import risk_manager

async def morning_checks():
    """Run morning checks - Operations Manual"""
    print("🔍 Running morning checks...")
    
    # Check IBKR connection
    await market_data.connect()
    print("✅ IBKR connected (quotes & execution)")
    
    # Check Alpha Vantage
    await av_client.connect()
    print(f"✅ Alpha Vantage connected (600 calls/min tier)")
    
    # Test IBKR market data
    await market_data.subscribe_symbols(['SPY'])
    await asyncio.sleep(5)
    price = market_data.get_latest_price('SPY')
    print(f"✅ IBKR SPY price: ${price:.2f}")
    
    # Test Alpha Vantage options WITH GREEKS
    options = await av_client.get_realtime_options('SPY', require_greeks=True)
    print(f"✅ Alpha Vantage options: {len(options)} contracts with Greeks")
    
    if options:
        sample = options[0]
        print(f"✅ Sample Greeks from AV: Δ={sample.delta:.3f}, Γ={sample.gamma:.3f}")
    
    # Check rate limit
    print(f"✅ AV Rate limit: {av_client.rate_limiter.remaining}/600 calls remaining")
    
    # Check risk limits
    print(f"✅ Risk limits: {len(risk_manager.positions)} / {risk_manager.max_positions} positions")
    print(f"✅ Daily P&L: ${risk_manager.daily_pnl:.2f} / -${risk_manager.daily_loss_limit}")
    
    print("\n🎯 System ready for trading!")
    print("📊 Data sources: IBKR (quotes/execution) + Alpha Vantage (options/analytics)")

if __name__ == "__main__":
    asyncio.run(morning_checks())
