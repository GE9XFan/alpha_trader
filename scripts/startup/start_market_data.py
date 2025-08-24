#!/usr/bin/env python3
"""Start market data connections"""
import asyncio
import sys
sys.path.append('.')

from src.data.market_data import market_data
from src.data.alpha_vantage_client import av_client
from src.core.config import config

async def main():
    """Start market data feeds"""
    print("Starting market data connections...")
    
    # Connect IBKR
    await market_data.connect()
    await market_data.subscribe_symbols(config.trading.symbols)
    
    # Connect Alpha Vantage
    await av_client.connect()
    
    print("✅ Market data ready")
    print(f"  IBKR: {config.trading.mode} mode")
    print(f"  Alpha Vantage: {av_client.config.rate_limit} calls/min")
    
    # Keep running
    while True:
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
