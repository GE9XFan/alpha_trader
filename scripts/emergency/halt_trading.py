#!/usr/bin/env python3
"""Emergency halt trading"""
import asyncio
import sys
sys.path.append('.')

from src.trading.paper_trader import paper_trader
from src.trading.risk import risk_manager

async def halt_trading():
    """Emergency stop all trading"""
    print("🚨 EMERGENCY: HALTING ALL TRADING")
    
    # Stop paper trader
    await paper_trader.stop()
    
    # Log current positions
    print(f"Current positions: {len(risk_manager.positions)}")
    for symbol, pos in risk_manager.positions.items():
        print(f"  {symbol}: {pos}")
    
    print("✅ Trading halted - positions preserved")

if __name__ == "__main__":
    asyncio.run(halt_trading())
