#!/usr/bin/env python3
"""
Production Data Collector Service
Continuously fetches and stores all data types from Alpha Vantage and IBKR
"""
import asyncio
import sys
import signal
from pathlib import Path
from datetime import datetime, time, timedelta
import time as time_module
from typing import Dict, List, Set
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.market_data import MarketDataManager
from src.data.alpha_vantage_client import AlphaVantageClient
from src.data.database_storage import data_storage
from src.core.logger import get_logger
from src.core.config import config
from ib_insync import util

logger = get_logger(__name__)


class ProductionDataCollector:
    """
    Production data collection service that continuously fetches and stores:
    - Options data with Greeks (every 30 SECONDS during market hours + historical daily)
    - Technical indicators (every 5 minutes)
    - Market bars from IBKR (STREAMING continuous)
    - News sentiment (every 30 minutes)
    - Analytics (every 15 minutes)
    - Fundamentals (daily)
    - Economic indicators (daily)
    """
    
    def __init__(self):
        self.ibkr = MarketDataManager()
        self.av = AlphaVantageClient()
        self.storage = data_storage
        self.running = False
        
        # Configure symbols to track
        self.symbols = config.trading.symbols if hasattr(config.trading, 'symbols') else ['AAPL', 'SPY', 'QQQ', 'TSLA', 'NVDA']
        self.etf_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
        
        # Track last update times
        self.last_update = {
            'options': {},
            'indicators': {},
            'news': datetime.min,
            'analytics': datetime.min,
            'fundamentals': datetime.min,
            'economic': datetime.min
        }
        
        # Collection statistics
        self.stats = {
            'options_collected': 0,
            'indicators_collected': 0,
            'bars_collected': 0,
            'news_collected': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def is_market_hours(self) -> bool:
        """Check if market is open (simplified - weekdays 9:30-16:00 ET)"""
        now = datetime.now()
        if now.weekday() > 4:  # Weekend
            return False
        current_time = now.time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= current_time <= market_close
    
    async def startup(self) -> bool:
        """Initialize all connections"""
        print("\n" + "="*80)
        print("🚀 ALPHATRADER PRODUCTION DATA COLLECTOR")
        print("="*80)
        print(f"📅 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Tracking: {', '.join(self.symbols)}")
        print("="*80 + "\n")
        
        # Connect to IBKR
        print("📡 Connecting to IBKR...")
        try:
            connected = await self.ibkr.connect()
            if connected:
                print("   ✅ IBKR connected")
                # Subscribe to symbols
                results = await self.ibkr.subscribe_symbols(self.symbols)
                for symbol, success in results.items():
                    if success:
                        print(f"   ✅ {symbol}: Subscribed")
                    else:
                        print(f"   ⚠️  {symbol}: Subscription failed")
            else:
                print("   ⚠️  IBKR connection failed (non-critical)")
        except Exception as e:
            print(f"   ⚠️  IBKR error: {e} (continuing anyway)")
        
        # Connect to Alpha Vantage
        print("\n🌐 Connecting to Alpha Vantage...")
        try:
            await self.av.connect()
            print("   ✅ Alpha Vantage ready")
            print(f"   📊 36 APIs available")
            print(f"   🚀 Premium tier: 600 calls/minute")
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            print(f"   ❌ Alpha Vantage failed: {e}")
            return False
        
        # Test database connection
        print("\n💾 Testing Database...")
        try:
            stats = self.storage.get_storage_stats()
            print("   ✅ Database connected")
            print(f"   📊 Current records: {sum(v for k, v in stats.items() if k.endswith('_count')):,}")
        except Exception as e:
            logger.error(f"Database error: {e}")
            print(f"   ❌ Database failed: {e}")
            return False
        
        self.running = True
        return True
    
    async def collect_realtime_options(self, symbol: str):
        """Collect and store REALTIME options data with Greeks - CRITICAL FOR OPTIONS TRADING"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get realtime options with Greeks
                options = await self.av.get_realtime_options(symbol, require_greeks=True)
                if options:
                    # Store ALL options in database
                    stored = await self.storage.store_options_chain(options, data_type='realtime')
                    self.stats['options_collected'] += stored
                    logger.info(f"Stored {stored} realtime options for {symbol}")
                    return stored
                else:
                    logger.warning(f"No options data returned for {symbol}, attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.error(f"Realtime options error for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                else:
                    self.stats['errors'] += 1
        return 0
    
    async def collect_historical_options(self, symbol: str):
        """Collect and store HISTORICAL options data (daily)"""
        try:
            # Get historical options if method exists
            if hasattr(self.av, 'get_historical_options'):
                # Pass date as string format YYYY-MM-DD
                from datetime import date
                options = await self.av.get_historical_options(symbol, date=date.today().strftime('%Y-%m-%d'))
            else:
                # Fallback to realtime with historical tag
                options = await self.av.get_realtime_options(symbol, require_greeks=False)
            
            if options:
                # Store ALL options in database
                stored = await self.storage.store_options_chain(options, data_type='historical')
                self.stats['options_collected'] += stored
                logger.info(f"Stored {stored} historical options for {symbol}")
                return stored
        except Exception as e:
            logger.error(f"Historical options error for {symbol}: {e}")
            self.stats['errors'] += 1
        return 0
    
    async def collect_indicators(self, symbol: str):
        """Collect and store KEY technical indicators for options trading"""
        # Reduced set focused on options-relevant indicators
        indicators = ['RSI', 'MACD', 'BBANDS', 'ATR', 'MFI']  # Most important for options
        stored_total = 0
        
        for indicator in indicators:
            # CHECK FOR SHUTDOWN
            if not self.running:
                logger.info(f"Stopping indicator collection for {symbol} due to shutdown")
                break
                
            try:
                method_name = f'get_{indicator.lower()}'
                method = getattr(self.av, method_name)
                
                # Build parameters - FIXED for each indicator type
                params = {'interval': 'daily'}
                
                # Indicators that need series_type and time_period
                if indicator in ['RSI', 'EMA', 'SMA', 'MOM', 'BBANDS']:
                    params['series_type'] = 'close'
                    params['time_period'] = str(14 if indicator != 'BBANDS' else 20)
                # Indicators that only need time_period (NO series_type)
                elif indicator in ['WILLR', 'CCI', 'ATR', 'ADX', 'MFI']:
                    params['time_period'] = str(14 if indicator != 'CCI' else 20)  # CCI uses 20
                # MACD only needs series_type
                elif indicator == 'MACD':
                    params['series_type'] = 'close'
                # STOCH has special parameters
                elif indicator == 'STOCH':
                    params['fastkperiod'] = '5'
                    params['slowkperiod'] = '3'
                    params['slowdperiod'] = '3'
                # AROON only needs time_period
                elif indicator == 'AROON':
                    params['time_period'] = '14'
                # Volume indicators (OBV, AD) don't need extra params
                
                # Fetch indicator data
                df = await method(symbol, **params)
                if df is not None and not df.empty:
                    # STORE EVERYTHING - no head() limitation!
                    total_points = len(df)
                    stored = await self.storage.store_technical_indicator(
                        symbol, indicator, df, interval='daily'  # Store ALL data
                    )
                    stored_total += stored
                    # Log actual data processed vs new inserts
                    logger.info(f"{indicator}: Processed {total_points} points, {stored} new inserts")
                    
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Indicator {indicator} error for {symbol}: {e}")
                self.stats['errors'] += 1
        
        self.stats['indicators_collected'] += stored_total
        return stored_total
    
    async def collect_market_bars(self):
        """Collect and store IBKR market bars"""
        if not self.ibkr.is_connected():
            return 0
        
        stored_total = 0
        for symbol in self.symbols:
            try:
                # Get historical bars - store ALL bars, not limited
                bars = await self.ibkr.get_historical_bars(
                    symbol, duration='1 D', bar_size='1 min'  # More granular
                )
                if not bars.empty:
                    stored = await self.storage.store_market_bars(
                        symbol, bars, bar_size='1 min'  # Store ALL bars
                    )
                    stored_total += stored
            except Exception as e:
                logger.error(f"Market bars error for {symbol}: {e}")
                self.stats['errors'] += 1
        
        self.stats['bars_collected'] += stored_total
        return stored_total
    
    async def collect_news(self):
        """Collect and store news sentiment"""
        try:
            # Get news for all symbols
            news = await self.av.get_news_sentiment(
                tickers=','.join(self.symbols),
                limit=50
            )
            if news and 'feed' in news:
                stored = await self.storage.store_news_sentiment(news)
                self.stats['news_collected'] += stored
                return stored
        except Exception as e:
            logger.error(f"News collection error: {e}")
            self.stats['errors'] += 1
        return 0
    
    async def collect_analytics(self):
        """Collect and store analytics"""
        try:
            # Get fixed window analytics for ETFs
            analytics = await self.av.get_analytics_fixed_window(
                symbols=','.join(self.etf_symbols),
                interval='DAILY',
                range='1month'
            )
            if analytics:
                success = await self.storage.store_analytics(
                    'fixed_window',
                    self.etf_symbols,
                    analytics,
                    interval='DAILY',
                    range='1month'
                )
                return 1 if success else 0
        except Exception as e:
            logger.error(f"Analytics collection error: {e}")
            self.stats['errors'] += 1
        return 0
    
    async def collect_fundamentals(self, symbol: str):
        """Collect and store fundamental data"""
        try:
            overview = await self.av.get_overview(symbol)
            if overview:
                success = await self.storage.store_fundamentals(
                    symbol, 'overview', overview
                )
                return 1 if success else 0
        except Exception as e:
            logger.error(f"Fundamentals error for {symbol}: {e}")
            self.stats['errors'] += 1
        return 0
    
    async def collect_economic_indicators(self):
        """Collect and store economic indicators"""
        indicators = ['CPI', 'INFLATION', 'UNEMPLOYMENT']
        stored_total = 0
        
        for indicator in indicators:
            try:
                method_name = f'get_{indicator.lower()}'
                if hasattr(self.av, method_name):
                    method = getattr(self.av, method_name)
                    
                    # Fetch data
                    if indicator == 'CPI':
                        df = await method(interval='monthly')
                    else:
                        df = await method()
                    
                    if df is not None and not df.empty:
                        # Store ALL economic data, not just 50!
                        stored = await self.storage.store_economic_indicator(
                            indicator, df, interval='monthly'  # Store ALL data
                        )
                        stored_total += stored
                    
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Economic indicator {indicator} error: {e}")
                self.stats['errors'] += 1
        
        return stored_total
    
    def print_status(self):
        """Print collection status"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        
        print(f"\n📊 COLLECTION STATUS [{datetime.now().strftime('%H:%M:%S')}]")
        print("─" * 50)
        print(f"⏱️  Runtime: {hours}h {minutes}m")
        print(f"📦 Options: {self.stats['options_collected']:,} contracts")
        print(f"📈 Indicators: {self.stats['indicators_collected']:,} data points")
        print(f"📉 Market Bars: {self.stats['bars_collected']:,} bars")
        print(f"📰 News: {self.stats['news_collected']:,} articles")
        print(f"⚠️  Errors: {self.stats['errors']}")
        
        # Show database totals
        try:
            db_stats = self.storage.get_storage_stats()
            total = sum(v for k, v in db_stats.items() if k.endswith('_count'))
            print(f"💾 Total DB Records: {total:,}")
        except:
            pass
    
    async def run_collection_loop(self):
        """Main collection loop with scheduling"""
        print("\n" + "="*80)
        print("🔄 STARTING CONTINUOUS DATA COLLECTION")
        print("="*80)
        print("\nSchedule (OPTIMIZED FOR OPTIONS TRADING):")
        print("  • Options (Realtime): Every 30 SECONDS (24/7)")
        print("  • Options (Historical): Every 4 hours")
        print("  • Indicators: Every 30 minutes (reduced)")
        print("  • Market Bars: STREAMING (continuous)")
        print("  • News: Every 30 minutes")
        print("  • Analytics: Every 15 minutes")
        print("  • Fundamentals: Daily")
        print("  • Economic: Daily")
        print("\n📊 Real-time collection status will update below...")
        print("─" * 50)
        
        loop_count = 0
        
        while self.running:
            try:
                # CHECK RUNNING STATUS FIRST
                if not self.running:
                    logger.info("Collection loop stopping due to shutdown signal")
                    break
                    
                now = datetime.now()
                loop_count += 1
                
                # REALTIME OPTIONS - Every 30 SECONDS - RUN 24/7 for an options trading system!
                # Options are THE PRIORITY for this system
                for symbol in self.symbols:
                    if symbol not in self.last_update['options'] or \
                       (now - self.last_update['options'].get(symbol, datetime.min)) > timedelta(seconds=30):
                        print(f"[{now.strftime('%H:%M:%S')}] 📦 Collecting REALTIME options for {symbol}...", end='')
                        stored = await self.collect_realtime_options(symbol)
                        print(f" {stored} stored")
                        self.last_update['options'][symbol] = now
                        await asyncio.sleep(0.1)  # Small delay between symbols
                
                # INDICATORS - Every 30 minutes (reduced from 5 to prevent overwhelming)
                # Only collect for a subset of symbols at a time
                for symbol in self.symbols[:3]:  # Limit to 3 symbols per cycle
                    if symbol not in self.last_update['indicators'] or \
                       (now - self.last_update['indicators'].get(symbol, datetime.min)) > timedelta(minutes=30):
                        print(f"[{now.strftime('%H:%M:%S')}] 📈 Collecting indicators for {symbol}...", end='')
                        stored = await self.collect_indicators(symbol)
                        print(f" processed")  # More accurate than 'stored'
                        self.last_update['indicators'][symbol] = now
                        await asyncio.sleep(1)
                
                # MARKET BARS - CONTINUOUS STREAMING (every loop)
                if self.is_market_hours() and self.ibkr.is_connected():
                    # Collect bars every loop for streaming data
                    print(f"[{now.strftime('%H:%M:%S')}] 📉 Streaming market bars...", end='')
                    stored = await self.collect_market_bars()
                    print(f" {stored} bars stored")
                
                # NEWS - Every 30 minutes
                if (now - self.last_update['news']) > timedelta(minutes=30):
                    print(f"[{now.strftime('%H:%M:%S')}] 📰 Collecting news...", end='')
                    stored = await self.collect_news()
                    print(f" {stored} articles stored")
                    self.last_update['news'] = now
                
                # ANALYTICS - Every 15 minutes
                if (now - self.last_update['analytics']) > timedelta(minutes=15):
                    print(f"[{now.strftime('%H:%M:%S')}] 🔬 Collecting analytics...", end='')
                    stored = await self.collect_analytics()
                    print(f" {'✓' if stored else '✗'}")
                    self.last_update['analytics'] = now
                
                # HISTORICAL OPTIONS - Every 4 hours - essential for options analysis
                if (now - self.last_update.get('historical_options', datetime.min)) > timedelta(hours=4):
                    for symbol in self.symbols:
                        print(f"[{now.strftime('%H:%M:%S')}] 📚 Collecting HISTORICAL options for {symbol}...", end='')
                        stored = await self.collect_historical_options(symbol)
                        print(f" {stored} stored")
                        await asyncio.sleep(1)
                    self.last_update['historical_options'] = now
                
                # FUNDAMENTALS - Daily
                if (now - self.last_update['fundamentals']) > timedelta(days=1):
                    for symbol in self.symbols:
                        print(f"[{now.strftime('%H:%M:%S')}] 📋 Collecting fundamentals for {symbol}...", end='')
                        stored = await self.collect_fundamentals(symbol)
                        print(f" {'✓' if stored else '✗'}")
                        await asyncio.sleep(1)
                    self.last_update['fundamentals'] = now
                
                # ECONOMIC - Daily
                if (now - self.last_update['economic']) > timedelta(days=1):
                    print(f"[{now.strftime('%H:%M:%S')}] 💹 Collecting economic indicators...", end='')
                    stored = await self.collect_economic_indicators()
                    print(f" {stored} stored")
                    self.last_update['economic'] = now
                
                # Print status every 10 loops (10 minutes)
                if loop_count % 10 == 0:
                    self.print_status()
                
                # Sleep for 10 seconds before next loop (for 30-second options updates)
                # Break sleep into smaller chunks to check for shutdown
                for _ in range(10):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                print(f"\n❌ Error in collection loop: {e}")
                traceback.print_exc()
                self.stats['errors'] += 1
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """Graceful shutdown"""
        print("\n" + "="*80)
        print("🛑 SHUTTING DOWN DATA COLLECTOR")
        print("="*80)
        
        # Print final statistics
        self.print_status()
        
        # Disconnect services
        if self.ibkr.is_connected():
            await self.ibkr.disconnect()
            print("✅ IBKR disconnected")
        
        if hasattr(self.av, 'disconnect'):
            await self.av.disconnect()
        print("✅ Alpha Vantage disconnected")
        
        print("\n✅ Shutdown complete")
        print("="*80)


async def main():
    """Main entry point"""
    collector = ProductionDataCollector()
    
    print("\n🔧 PREREQUISITES:")
    print("   1. PostgreSQL must be running")
    print("   2. Alpha Vantage API key must be set")
    print("   3. TWS/Gateway should be running (optional)")
    print("\n⚠️  Press Ctrl+C to stop the service\n")
    
    try:
        if await collector.startup():
            await collector.run_collection_loop()
        else:
            print("\n❌ Startup failed - check prerequisites")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        print(f"\n❌ Service error: {e}")
        traceback.print_exc()
    finally:
        if collector.running:
            await collector.shutdown()


if __name__ == "__main__":
    # Use ib_insync's event loop for compatibility
    util.run(main())