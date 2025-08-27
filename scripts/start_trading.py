#!/usr/bin/env python3
"""
Options Trading System - Enhanced Main Startup Script
Complete Day 1-4: Full Data Pipeline with ALL Features
"""

import os
import sys
import asyncio
import yaml
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
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


class EnhancedTradingSystem:
    """Enhanced trading system that uses ALL implemented features"""

    def __init__(self):
        """Initialize trading system components"""
        self.config = self._load_config()
        self.cache = None
        self.ibkr = None
        self.av = None
        self.running = False
        self.symbols = []
        self.tasks = []

        # Track API usage
        self.api_usage = {
            'av_apis_used': set(),
            'ibkr_features_used': set(),
            'cache_types_used': set()
        }

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
            logger.info("Initializing enhanced trading system...")

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

            # 5. Subscribe to ALL IBKR market data types
            await self._subscribe_all_market_data()

            # 6. Test account features
            await self._test_account_features()

            logger.success("✓ All components initialized with COMPLETE features")
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

    async def _subscribe_all_market_data(self):
        """Subscribe to ALL IBKR market data types"""
        if not self.ibkr:
            logger.warning("IBKR client not initialized")
            return

        for symbol in self.symbols:
            logger.info(f"Subscribing to COMPLETE market data for {symbol}...")

            # 1. Level 2 order book
            await self.ibkr.subscribe_market_depth(symbol, num_rows=10)
            self.api_usage['ibkr_features_used'].add('level2_order_book')

            # 2. Trade tape
            await self.ibkr.subscribe_trades(symbol)
            self.api_usage['ibkr_features_used'].add('trade_tape')

            # 3. 5-sec bars
            await self.ibkr.subscribe_bars(symbol, bar_size="5 secs")
            self.api_usage['ibkr_features_used'].add('realtime_bars')

            logger.success(f"✓ Subscribed to ALL data feeds for {symbol}")

    async def _test_account_features(self):
        """Test IBKR account features"""
        if not self.ibkr:
            return

        # Get account summary
        account = self.ibkr.get_account_summary()
        if account:
            logger.info(f"Account Buying Power: ${account.get('buying_power', 0):,.2f}")
            self.api_usage['ibkr_features_used'].add('account_summary')

        # Get positions
        positions = self.ibkr.get_positions()
        logger.info(f"Current Positions: {len(positions)}")
        self.api_usage['ibkr_features_used'].add('positions')

    # ==================== ALPHA VANTAGE COMPLETE USAGE ====================

    async def update_all_options_data(self):
        """Update ALL options data types from Alpha Vantage"""
        while self.running:
            if not self.av:
                logger.warning("Alpha Vantage client not initialized")
                await asyncio.sleep(10)
                continue
                
            for symbol in self.symbols:
                try:
                    # 1. REALTIME OPTIONS with Greeks
                    chain = await self.av.get_realtime_options(symbol, require_greeks=True, ibkr_client=self.ibkr)
                    if chain:
                        logger.debug(f"✓ Realtime options for {symbol}: {len(chain.options)} contracts")
                        self.api_usage['av_apis_used'].add('REALTIME_OPTIONS')
                        self.api_usage['cache_types_used'].add('options_chain')

                        # Log sample Greeks to prove they're PROVIDED
                        if chain.options:
                            sample = chain.options[0]
                            logger.debug(f"  Greeks PROVIDED: Delta={sample.delta:.4f}, Gamma={sample.gamma:.5f}")

                    # 2. HISTORICAL OPTIONS
                    hist_chain = await self.av.get_historical_options(symbol)
                    if hist_chain:
                        logger.debug(f"✓ Historical options for {symbol}: {len(hist_chain.options)} contracts")
                        self.api_usage['av_apis_used'].add('HISTORICAL_OPTIONS')

                except Exception as e:
                    logger.error(f"Failed to update options for {symbol}: {e}")

            # Rate limit check
            if self.av:
                stats = self.av.get_stats()
                if stats['calls_remaining'] < 100:
                    logger.warning(f"AV API calls remaining: {stats['calls_remaining']}/600 - slowing down")
                    await asyncio.sleep(30)
                else:
                    await asyncio.sleep(10)  # Normal pace

    async def update_all_technical_indicators(self):
        """Update ALL technical indicators from Alpha Vantage"""
        while self.running:
            if not self.av:
                logger.warning("Alpha Vantage client not initialized")
                await asyncio.sleep(10)
                continue
                
            for symbol in self.symbols[:3]:  # Limit to top 3 to conserve API
                try:
                    # 1. RSI
                    rsi = await self.av.get_rsi(symbol, interval='daily', time_period=14)
                    if rsi:
                        logger.debug(f"✓ RSI updated for {symbol}")
                        self.api_usage['av_apis_used'].add('RSI')
                        self.api_usage['cache_types_used'].add('indicator_rsi')

                    # 2. MACD
                    macd = await self.av.get_macd(symbol, interval='daily')
                    if macd:
                        logger.debug(f"✓ MACD updated for {symbol}")
                        self.api_usage['av_apis_used'].add('MACD')
                        self.api_usage['cache_types_used'].add('indicator_macd')

                    # 3. Bollinger Bands
                    bbands = await self.av.get_bbands(symbol, interval='daily')
                    if bbands:
                        logger.debug(f"✓ Bollinger Bands updated for {symbol}")
                        self.api_usage['av_apis_used'].add('BBANDS')
                        self.api_usage['cache_types_used'].add('indicator_bbands')

                    # 4. ATR
                    atr = await self.av.get_atr(symbol, interval='daily')
                    if atr:
                        logger.debug(f"✓ ATR updated for {symbol}")
                        self.api_usage['av_apis_used'].add('ATR')
                        self.api_usage['cache_types_used'].add('indicator_atr')

                    # 5. VWAP (intraday only)
                    vwap = await self.av.get_vwap(symbol, interval='15min')
                    if vwap:
                        logger.debug(f"✓ VWAP updated for {symbol}")
                        self.api_usage['av_apis_used'].add('VWAP')
                        self.api_usage['cache_types_used'].add('indicator_vwap')

                    await asyncio.sleep(2)  # Rate limiting between indicators

                except Exception as e:
                    logger.error(f"Failed to update indicators for {symbol}: {e}")

            await asyncio.sleep(60)  # Update indicators every minute

    async def update_sentiment_and_market_data(self):
        """Update sentiment and market intelligence data"""
        while self.running:
            if not self.av:
                logger.warning("Alpha Vantage client not initialized")
                await asyncio.sleep(10)
                continue
                
            try:
                # 1. NEWS SENTIMENT
                for symbol in self.symbols[:2]:  # Top 2 symbols only
                    sentiment = await self.av.get_news_sentiment(tickers=symbol, limit=50)
                    if sentiment and 'feed' in sentiment:
                        logger.debug(f"✓ News sentiment for {symbol}: {len(sentiment.get('feed', []))} articles")
                        self.api_usage['av_apis_used'].add('NEWS_SENTIMENT')
                        self.api_usage['cache_types_used'].add('sentiment')

                # 2. TOP GAINERS/LOSERS
                movers = await self.av.get_top_gainers_losers()
                if movers:
                    logger.debug(f"✓ Top movers updated")
                    self.api_usage['av_apis_used'].add('TOP_GAINERS_LOSERS')

                # 3. INSIDER TRANSACTIONS
                for symbol in self.symbols[:1]:  # Just primary symbol
                    insiders = await self.av.get_insider_transactions(symbol)
                    if insiders:
                        logger.debug(f"✓ Insider transactions for {symbol}")
                        self.api_usage['av_apis_used'].add('INSIDER_TRANSACTIONS')

            except Exception as e:
                logger.error(f"Failed to update sentiment data: {e}")

            await asyncio.sleep(300)  # Update every 5 minutes

    async def update_fundamentals(self):
        """Update fundamental data for all symbols"""
        while self.running:
            if not self.av:
                logger.warning("Alpha Vantage client not initialized")
                await asyncio.sleep(10)
                continue
                
            for symbol in self.symbols:
                try:
                    # 1. COMPANY OVERVIEW
                    overview = await self.av.get_company_overview(symbol)
                    if overview and 'Symbol' in overview:
                        logger.debug(f"✓ Company overview for {symbol}: {overview.get('Name', 'N/A')}")
                        self.api_usage['av_apis_used'].add('COMPANY_OVERVIEW')
                        self.api_usage['cache_types_used'].add('fundamentals')

                    # 2. EARNINGS
                    earnings = await self.av.get_earnings(symbol)
                    if earnings:
                        logger.debug(f"✓ Earnings data for {symbol}")
                        self.api_usage['av_apis_used'].add('EARNINGS')

                    await asyncio.sleep(5)  # Rate limiting

                except Exception as e:
                    logger.error(f"Failed to update fundamentals for {symbol}: {e}")

            await asyncio.sleep(3600)  # Update hourly

    async def update_analytics(self):
        """Update advanced analytics data"""
        while self.running:
            if not self.av:
                logger.warning("Alpha Vantage client not initialized")
                await asyncio.sleep(10)
                continue
                
            try:
                # ANALYTICS SLIDING WINDOW
                symbols_str = ','.join(self.symbols[:2])  # First 2 symbols
                analytics = await self.av.get_analytics_sliding_window(
                    SYMBOLS=symbols_str,
                    INTERVAL='DAILY',
                    RANGE='1month',
                    WINDOW_SIZE=20,
                    CALCULATIONS='MEAN,STDDEV,CUMULATIVE_RETURN',
                    OHLC='close'
                )
                if analytics:
                    logger.debug(f"✓ Analytics updated for {symbols_str}")
                    self.api_usage['av_apis_used'].add('ANALYTICS_SLIDING_WINDOW')

            except Exception as e:
                logger.error(f"Failed to update analytics: {e}")

            await asyncio.sleep(600)  # Update every 10 minutes

    # ==================== IBKR COMPLETE FEATURES ====================

    async def fetch_historical_data(self):
        """Fetch historical data from IBKR"""
        while self.running:
            if not self.ibkr or not self.ibkr.is_connected():
                logger.warning("IBKR client not connected")
                await asyncio.sleep(10)
                continue
                
            for symbol in self.symbols[:2]:  # Top 2 symbols
                try:
                    # Get historical bars
                    bars = await self.ibkr.get_historical_data(
                        symbol,
                        duration="1 D",
                        bar_size="5 mins"
                    )
                    if bars:
                        logger.debug(f"✓ Historical data for {symbol}: {len(bars)} bars")
                        self.api_usage['ibkr_features_used'].add('historical_data')

                except Exception as e:
                    logger.error(f"Failed to get historical data for {symbol}: {e}")

            await asyncio.sleep(3600)  # Update hourly

    # ==================== MONITORING & VERIFICATION ====================

    async def monitor_complete_data_flow(self):
        """Monitor and verify ALL data is flowing correctly"""
        while self.running:
            await asyncio.sleep(30)  # Let data accumulate first

            logger.info("\n" + "="*60)
            logger.info("COMPLETE DATA FLOW STATUS")
            logger.info("="*60)

            # Alpha Vantage APIs used
            logger.info(f"\nAlpha Vantage APIs Active: {len(self.api_usage['av_apis_used'])}/13")
            for api in sorted(self.api_usage['av_apis_used']):
                logger.info(f"  ✓ {api}")

            missing_av = {'REALTIME_OPTIONS', 'HISTORICAL_OPTIONS', 'RSI', 'MACD',
                         'BBANDS', 'ATR', 'VWAP', 'NEWS_SENTIMENT', 'TOP_GAINERS_LOSERS',
                         'INSIDER_TRANSACTIONS', 'COMPANY_OVERVIEW', 'EARNINGS',
                         'ANALYTICS_SLIDING_WINDOW'} - self.api_usage['av_apis_used']
            if missing_av:
                logger.warning(f"Missing AV APIs: {', '.join(missing_av)}")

            # IBKR Features used
            logger.info(f"\nIBKR Features Active: {len(self.api_usage['ibkr_features_used'])}/6")
            for feature in sorted(self.api_usage['ibkr_features_used']):
                logger.info(f"  ✓ {feature}")

            # Cache verification for first symbol
            if self.symbols and self.cache:
                symbol = self.symbols[0]
                logger.info(f"\nCache Status for {symbol}:")

                # Check each cache type
                cache_checks = {
                    'Order Book (1s TTL)': self.cache.get_order_book(symbol),
                    'Options Chain (10s TTL)': self.cache.get_options_chain(symbol),
                    'Recent Trades': self.cache.get_recent_trades(symbol, 5),
                    'RSI Indicator': self.cache.get_indicator(symbol, 'RSI_daily_14'),
                    'MACD Indicator': self.cache.get_indicator(symbol, 'MACD_daily'),
                    'BBands Indicator': self.cache.get_indicator(symbol, 'BBANDS_daily_20'),
                    'ATR Indicator': self.cache.get_indicator(symbol, 'ATR_daily_14'),
                    'News Sentiment': self.cache.get('sentiment:' + symbol),
                    'Fundamentals': self.cache.get('fundamentals:' + symbol)
                }

                for cache_type, data in cache_checks.items():
                    status = '✓' if data else '✗'
                    logger.info(f"  {status} {cache_type}")

                    # Show sample data
                    if data:
                        if 'bids' in str(data):  # Order book
                            logger.debug(f"    -> Bid levels: {len(data.get('bids', []))}")
                        elif 'options' in str(data):  # Options chain
                            logger.debug(f"    -> Options count: {len(data.get('options', []))}")

            # Performance metrics
            if self.cache:
                cache_stats = self.cache.get_stats()
                logger.info(f"\nPerformance Metrics:")
                logger.info(f"  Cache Hit Rate: {cache_stats['hit_rate']}")
                logger.info(f"  Cache Keys: {cache_stats['keys']}")
                logger.info(f"  Memory Used: {cache_stats['memory_used']}")

            if self.av:
                av_stats = self.av.get_stats()
                logger.info(f"  AV API Calls Made: {av_stats['calls_made']}")
                logger.info(f"  AV API Remaining: {av_stats['calls_remaining']}/600")
                logger.info(f"  AV Cache Hits: {av_stats['cache_hits']}")

            await asyncio.sleep(60)  # Check every minute

    async def monitor_system_health(self):
        """Monitor system health and performance"""
        while self.running:
            try:
                health = SystemHealth(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    ibkr_connected=self.ibkr.is_connected() if self.ibkr else False,
                    redis_connected=self.cache.health_check() if self.cache else False,
                    av_api_healthy=True,
                    cache_hit_rate=float(self.cache.get_stats()['hit_rate'].rstrip('%')) if self.cache else 0.0,
                    av_calls_remaining=self.av.rate_limiter.get_remaining_calls() if self.av else 0
                )

                if not health.is_healthy:
                    logger.warning("System health check failed")
                    logger.info(f"  IBKR: {health.ibkr_connected}")
                    logger.info(f"  Redis: {health.redis_connected}")
                    logger.info(f"  AV API: {health.av_api_healthy}")

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

            await asyncio.sleep(30)

    async def run(self):
        """Main loop running ALL features"""
        logger.info("Starting COMPLETE data pipeline with ALL features...")
        self.running = True

        # Start ALL background tasks
        self.tasks = [
            # Core monitoring
            asyncio.create_task(self.monitor_system_health()),
            asyncio.create_task(self.monitor_complete_data_flow()),

            # COMPLETE Alpha Vantage usage
            asyncio.create_task(self.update_all_options_data()),
            asyncio.create_task(self.update_all_technical_indicators()),
            asyncio.create_task(self.update_sentiment_and_market_data()),
            asyncio.create_task(self.update_fundamentals()),
            asyncio.create_task(self.update_analytics()),

            # COMPLETE IBKR usage
            asyncio.create_task(self.fetch_historical_data()),
        ]

        logger.info(f"Started {len(self.tasks)} background tasks")
        logger.info("All 13 Alpha Vantage APIs will be exercised")
        logger.info("All IBKR features are active")
        logger.info("\n📊 COMPLETE data pipeline is now running...")
        logger.info("Press Ctrl+C to stop\n")

        try:
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of all components"""
        logger.info("Shutting down enhanced trading system...")
        self.running = False

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Final usage report
        logger.info("\n" + "="*60)
        logger.info("FINAL USAGE REPORT")
        logger.info("="*60)
        logger.info(f"Alpha Vantage APIs used: {len(self.api_usage['av_apis_used'])}/13")
        logger.info(f"IBKR features used: {len(self.api_usage['ibkr_features_used'])}/6")
        logger.info(f"Cache types used: {len(self.api_usage['cache_types_used'])}")

        # Cleanup
        if self.ibkr:
            await self.ibkr.unsubscribe_all()
            await self.ibkr.disconnect()
            logger.info("✓ IBKR disconnected")

        if self.av:
            await self.av.close()
            logger.info("✓ Alpha Vantage client closed")

        if self.cache:
            self.cache.close()
            logger.info("✓ Cache connection closed")

        logger.success("Shutdown complete")


async def main():
    """Main entry point for enhanced trading system"""
    logger.info("=" * 60)
    logger.info("ENHANCED OPTIONS TRADING SYSTEM - COMPLETE PIPELINE")
    logger.info("Day 1-4: ALL Features Active")
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

    # Create and initialize enhanced system
    system = EnhancedTradingSystem()

    if await system.initialize():
        logger.success("✓ Enhanced system initialized successfully")

        # Display what will be tested
        logger.info("\n" + "="*60)
        logger.info("COMPLETE FEATURE SET ACTIVE:")
        logger.info("="*60)
        logger.info("Alpha Vantage APIs (13 total):")
        logger.info("  • Options: REALTIME, HISTORICAL")
        logger.info("  • Indicators: RSI, MACD, BBANDS, ATR, VWAP")
        logger.info("  • Sentiment: NEWS, TOP_MOVERS, INSIDERS")
        logger.info("  • Fundamentals: OVERVIEW, EARNINGS")
        logger.info("  • Analytics: SLIDING_WINDOW")
        logger.info("\nIBKR Features (6 total):")
        logger.info("  • Level 2 Order Book (10 levels)")
        logger.info("  • Trade Tape")
        logger.info("  • Real-time Bars (5 sec)")
        logger.info("  • Historical Data")
        logger.info("  • Account Summary")
        logger.info("  • Position Tracking")
        logger.info("="*60)

        # Run the complete system
        await system.run()
    else:
        logger.error("System initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Use the enhanced system by default
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
