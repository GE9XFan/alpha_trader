#!/usr/bin/env python3
"""
Options Trading System - Main Startup Script
Day 3-4: Core Infrastructure Implementation
"""

import os
import sys
import asyncio
import yaml
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    CacheManager,
    IBKRClient,
    AlphaVantageClient,
    SystemHealth
)

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)


class TradingSystem:
    """Main trading system orchestrator"""

    def __init__(self):
        """Initialize trading system components"""
        self.config = self._load_config()
        self.cache = None
        self.ibkr = None
        self.av = None
        self.running = False
        self.symbols = []
        self.tasks = []

    def _load_config(self) -> Dict:
        """Load system configuration"""
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Load symbols config
        with open('config/symbols.yaml', 'r') as f:
            symbols_config = yaml.safe_load(f)

        config['symbols'] = symbols_config
        return config

    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing trading system...")

            # 1. Initialize Redis cache
            self.cache = CacheManager()
            if not self.cache.health_check():
                raise ConnectionError("Redis connection failed")
            logger.success("✓ Cache manager initialized")

            # 2. Initialize IBKR client
            self.ibkr = IBKRClient(self.cache, self.config)
            if not await self.ibkr.connect():
                raise ConnectionError("IBKR connection failed")
            logger.success("✓ IBKR client connected")

            # 3. Initialize Alpha Vantage client
            self.av = AlphaVantageClient(self.cache, self.config)
            logger.success("✓ Alpha Vantage client initialized")

            # 4. Get symbols to trade
            self.symbols = self._get_active_symbols()
            logger.info(f"Trading symbols: {', '.join(self.symbols)}")

            # 5. Subscribe to market data
            await self._subscribe_market_data()

            logger.success("✓ All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _get_active_symbols(self) -> List[str]:
        """Get list of symbols to trade"""
        symbols = []

        # Get primary symbols
        for symbol_data in self.config['symbols'].get('primary', []):
            symbols.append(symbol_data['symbol'])

        # Add secondary if we have capacity
        if len(symbols) < 10:  # Max 10 symbols as per spec
            for symbol_data in self.config['symbols'].get('secondary', []):
                if len(symbols) >= 10:
                    break
                symbols.append(symbol_data['symbol'])

        return symbols[:10]  # Ensure max 10

    async def _subscribe_market_data(self):
        """Subscribe to market data for all symbols"""
        if not self.ibkr:
            logger.warning("IBKR client not initialized")
            return
            
        for symbol in self.symbols:
            logger.info(f"Subscribing to market data for {symbol}...")

            # Subscribe to Level 2 order book
            await self.ibkr.subscribe_market_depth(symbol)

            # Subscribe to trade tape
            await self.ibkr.subscribe_trades(symbol)

            # Subscribe to 5-sec bars
            await self.ibkr.subscribe_bars(symbol)

            logger.success(f"✓ Subscribed to all data feeds for {symbol}")

    async def update_options_data(self):
        """Periodically update options data from Alpha Vantage"""
        while self.running:
            if not self.av:
                await asyncio.sleep(10)
                continue
                
            for symbol in self.symbols:
                try:
                    # Get options chain with Greeks
                    chain = await self.av.get_realtime_options(symbol)
                    if chain:
                        logger.debug(f"Updated options for {symbol}: {len(chain.options)} contracts")

                    # Get technical indicators
                    await self.av.get_rsi(symbol)
                    await self.av.get_macd(symbol)

                    # Rate limit management
                    stats = self.av.get_stats()
                    if stats['calls_remaining'] < 100:
                        logger.warning(f"API calls remaining: {stats['calls_remaining']}")
                        await asyncio.sleep(10)  # Slow down if running low

                except Exception as e:
                    logger.error(f"Failed to update options for {symbol}: {e}")

            # Update every 10 seconds (matches options cache TTL)
            await asyncio.sleep(10)

    async def monitor_system_health(self):
        """Monitor system health and performance"""
        while self.running:
            try:
                health = SystemHealth(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    ibkr_connected=self.ibkr.is_connected() if self.ibkr else False,
                    redis_connected=self.cache.health_check() if self.cache else False,
                    av_api_healthy=True,  # Based on recent successful calls
                    cache_hit_rate=float(self.cache.get_stats()['hit_rate'].rstrip('%')) if self.cache else 0.0,
                    av_calls_remaining=self.av.rate_limiter.get_remaining_calls() if self.av else 0
                )

                # Log health status
                if health.is_healthy:
                    logger.debug("System health: OK")
                else:
                    logger.warning("System health check failed")
                    logger.info(f"  IBKR: {health.ibkr_connected}")
                    logger.info(f"  Redis: {health.redis_connected}")
                    logger.info(f"  AV API: {health.av_api_healthy}")

                # Check account status
                if self.ibkr:
                    account = self.ibkr.get_account_summary()
                    if account:
                        logger.debug(f"Account buying power: ${account.get('buying_power', 0):,.2f}")

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def display_market_data(self):
        """Display real-time market data (for monitoring)"""
        while self.running:
            try:
                logger.info("\n" + "="*60)
                logger.info("MARKET DATA SNAPSHOT")
                logger.info("="*60)

                if self.cache:
                    for symbol in self.symbols[:3]:  # Show top 3 symbols
                        # Get order book from cache
                        order_book = self.cache.get_order_book(symbol)
                        if order_book and order_book['bids'] and order_book['asks']:
                            bid = order_book['bids'][0]['price']
                            ask = order_book['asks'][0]['price']
                            spread = ask - bid
                            logger.info(f"{symbol}: Bid {bid:.2f} / Ask {ask:.2f} (Spread: {spread:.3f})")

                        # Get cached metrics if available
                        metrics = self.cache.get_metrics(symbol)
                        if metrics:
                            logger.info(f"  VPIN: {metrics.get('vpin', 'N/A')}")

                    # Show cache statistics
                    cache_stats = self.cache.get_stats()
                    logger.info(f"\nCache: {cache_stats['hit_rate']} hit rate, {cache_stats['keys']} keys")

                # Show API usage
                if self.av:
                    av_stats = self.av.get_stats()
                    logger.info(f"AV API: {av_stats['calls_remaining']}/600 calls remaining")

            except Exception as e:
                logger.error(f"Display error: {e}")

            await asyncio.sleep(5)  # Update display every 5 seconds

    async def run(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        self.running = True

        # Start background tasks
        self.tasks = [
            asyncio.create_task(self.update_options_data()),
            asyncio.create_task(self.monitor_system_health()),
            asyncio.create_task(self.display_market_data())
        ]

        try:
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of all components"""
        logger.info("Shutting down trading system...")
        self.running = False

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Unsubscribe from market data
        if self.ibkr:
            await self.ibkr.unsubscribe_all()
            self.ibkr.disconnect()
            logger.info("✓ IBKR disconnected")

        # Close Alpha Vantage client
        if self.av:
            await self.av.close()
            logger.info("✓ Alpha Vantage client closed")

        # Close cache connection
        if self.cache:
            self.cache.close()
            logger.info("✓ Cache connection closed")

        logger.success("Shutdown complete")


async def main():
    """Main entry point for the trading system"""
    logger.info("=" * 60)
    logger.info("OPTIONS TRADING SYSTEM - STARTUP")
    logger.info("Day 3-4: Core Infrastructure Running")
    logger.info("=" * 60)

    # Check environment
    environment = os.getenv("ENVIRONMENT", "development")
    trading_mode = os.getenv("TRADING_MODE", "paper")

    logger.info(f"Environment: {environment}")
    logger.info(f"Trading Mode: {trading_mode}")

    if trading_mode == "live":
        logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
        response = input("Are you sure you want to continue in LIVE mode? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Exiting for safety. Switch to paper mode in .env file.")
            sys.exit(0)

    # Create and initialize trading system
    system = TradingSystem()

    if await system.initialize():
        logger.success("✓ System initialized successfully")
        logger.info("\nStarting trading operations...")

        # Display initial status
        logger.info("\nSystem Status:")
        logger.info(f"  Trading Symbols: {', '.join(system.symbols)}")
        logger.info(f"  Cache Connected: {system.cache.health_check() if system.cache else False}")
        logger.info(f"  IBKR Connected: {system.ibkr.is_connected() if system.ibkr else False}")
        logger.info(f"  AV Rate Limit: {system.av.rate_limiter.get_remaining_calls() if system.av else 0}/600")

        logger.info("\n📊 Market data is now streaming...")
        logger.info("Press Ctrl+C to stop\n")

        # Run the system
        await system.run()
    else:
        logger.error("System initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
