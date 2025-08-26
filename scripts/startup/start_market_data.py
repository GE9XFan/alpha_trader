#!/usr/bin/env python3
"""
Start Market Data Services
Initializes both IBKR and Alpha Vantage connections
Production-ready startup script with health checks
"""
import asyncio
import sys
import signal
from pathlib import Path
from datetime import datetime
import time
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.market_data import market_data
from src.data.alpha_vantage_client import AlphaVantageClient
from src.core.logger import get_logger
from src.core.config import config

logger = get_logger(__name__)


class MarketDataService:
    """Production market data service manager"""
    
    def __init__(self):
        self.market = market_data
        self.av_client = AlphaVantageClient()
        self.running = False
        self.symbols = config.trading.symbols if hasattr(config.trading, 'symbols') else ['SPY', 'QQQ', 'IWM']
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def startup(self):
        """Initialize all data services"""
        print("\n" + "="*70)
        print("🚀 ALPHATRADER MARKET DATA SERVICE")
        print("="*70)
        print(f"📅 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Mode: {config.trading.mode.upper()}")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
        print("="*70 + "\n")
        
        # Initialize IBKR connection
        print("📡 Connecting to IBKR...")
        try:
            connected = await self.market.connect()
            if not connected:
                raise Exception("Failed to connect to IBKR")
            
            status = self.market.get_connection_status()
            print(f"   ✅ Connected to {status['host']}:{status['port']}")
            print(f"   🔧 Mode: {status['mode']}")
            
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            print(f"   ❌ IBKR connection failed: {e}")
            print("   ⚠️  Ensure TWS/Gateway is running on port 7497")
            return False
        
        # Initialize Alpha Vantage connection
        print("\n🌐 Connecting to Alpha Vantage...")
        try:
            await self.av_client.connect()
            print(f"   ✅ Connected (600 calls/min premium tier)")
            print(f"   📊 36 APIs available")
            print(f"   🔑 API Key: {'✅ Configured' if self.av_client.api_key else '❌ Missing'}")
            
        except Exception as e:
            logger.error(f"Alpha Vantage initialization failed: {e}")
            print(f"   ⚠️  Alpha Vantage initialization warning: {e}")
        
        # Subscribe to market data
        print(f"\n📈 Subscribing to market data for {len(self.symbols)} symbols...")
        try:
            results = await self.market.subscribe_symbols(self.symbols)
            
            for symbol, success in results.items():
                if success:
                    print(f"   ✅ {symbol}: Subscribed")
                else:
                    print(f"   ❌ {symbol}: Failed")
            
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            print(f"   ❌ Subscription failed: {e}")
            return False
        
        print("\n" + "="*70)
        print("✅ MARKET DATA SERVICE READY")
        print("="*70)
        
        self.running = True
        return True
    
    async def run_service(self):
        """Main service loop with monitoring"""
        if not await self.startup():
            logger.error("Startup failed")
            return
        
        print("\n📊 Real-time Market Data Stream:")
        print("-" * 50)
        
        last_price_log = {}
        last_av_check = time.time()
        
        while self.running:
            try:
                # Log price updates periodically
                current_time = time.time()
                
                # Check IBKR prices every 5 seconds
                for symbol in self.symbols:
                    price = self.market.get_latest_price(symbol)
                    
                    if price > 0:
                        # Only log if price changed significantly or first time
                        if symbol not in last_price_log or \
                           abs(price - last_price_log[symbol]) > 0.01:
                            
                            bar = self.market.get_latest_bar(symbol)
                            if bar:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                                      f"{symbol}: ${price:.2f} "
                                      f"(H: ${bar['high']:.2f}, L: ${bar['low']:.2f}, "
                                      f"Vol: {bar['volume']:,})")
                            
                            last_price_log[symbol] = price
                
                # Health check
                if not self.market.is_connected():
                    logger.warning("IBKR connection lost, attempting reconnect...")
                    await self.market.connect()
                
                # Sleep briefly
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Service loop error: {e}")
                await asyncio.sleep(5)
        
        # Shutdown
        await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        print("\n" + "="*70)
        print("🛑 SHUTTING DOWN MARKET DATA SERVICE")
        print("="*70)
        
        # Disconnect IBKR
        print("📡 Disconnecting from IBKR...")
        await self.market.disconnect()
        print("   ✅ IBKR disconnected")
        
        # Close Alpha Vantage session
        if hasattr(self.av_client, 'session') and self.av_client.session:
            await self.av_client.session.close()
            print("   ✅ Alpha Vantage session closed")
        
        print("\n✅ Shutdown complete")
        print("="*70)


async def main():
    """Main entry point"""
    service = MarketDataService()
    
    print("\n🔧 STARTUP CHECKS:")
    print("   1. TWS/Gateway must be running on port 7497")
    print("   2. Alpha Vantage API key must be set")
    print("   3. Database should be running")
    print("   4. Redis should be running")
    
    print("\n⚠️  Press Ctrl+C to stop the service\n")
    
    try:
        await service.run_service()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        print(f"\n❌ Service error: {e}")
    finally:
        if service.running:
            await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
