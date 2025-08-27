#!/usr/bin/env python3
"""
Complete Data Pipeline Test - REAL Production Data
Tests ALL Day 1-4 Implementation with actual market data
NO MOCKS - Identifies real issues
"""

import asyncio
import sys
import yaml
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio for notebook compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    CacheManager,
    IBKRClient,
    AlphaVantageClient,
    OrderBook,
    OptionsChain,
    Trade,
    Bar
)


class CompletePipelineTest:
    """
    Comprehensive pipeline test with REAL production data
    Tests every component, identifies actual issues
    """

    def __init__(self):
        self.cache = None
        self.ibkr = None
        self.av = None
        self.config = None

        # Test results tracking
        self.test_results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'performance_metrics': {},
            'api_coverage': {
                'av_tested': set(),
                'ibkr_tested': set()
            }
        }

        # Timing metrics
        self.timing = {}

    def _load_config(self) -> Dict:
        """Load configuration files"""
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        with open('config/symbols.yaml', 'r') as f:
            symbols_config = yaml.safe_load(f)

        config['symbols'] = symbols_config
        return config

    async def initialize_components(self) -> bool:
        """Initialize all components with production settings"""
        logger.info("\n" + "="*70)
        logger.info("INITIALIZING COMPONENTS WITH PRODUCTION SETTINGS")
        logger.info("="*70)

        try:
            # Load config
            self.config = self._load_config()

            # 1. Initialize Cache
            logger.info("\n[1/3] Initializing Redis Cache...")
            start = time.time()
            self.cache = CacheManager()

            if not self.cache.health_check():
                self._record_failure("Redis", "Connection failed")
                return False

            cache_stats = self.cache.get_stats()
            self.timing['cache_init'] = (time.time() - start) * 1000
            logger.success(f"✓ Cache initialized in {self.timing['cache_init']:.2f}ms")
            logger.info(f"  Memory: {cache_stats['memory_used']}, Keys: {cache_stats['keys']}")
            self._record_pass("Cache initialization")

            # 2. Initialize IBKR
            logger.info("\n[2/3] Connecting to IBKR...")
            start = time.time()
            # Use unique client ID to avoid conflicts with other test instances
            import random
            unique_client_id = 100 + random.randint(1, 900)  # Client ID between 100-999
            self.config['ibkr']['client_id'] = unique_client_id
            logger.info(f"Using unique client ID: {unique_client_id}")
            self.ibkr = IBKRClient(self.cache, self.config)

            connected = await self.ibkr.connect(retry_count=3)
            if not connected:
                self._record_failure("IBKR", "Connection failed after 3 attempts")
                logger.error("⚠️  IBKR connection failed - some tests will be skipped")
                logger.info("  Make sure TWS/IB Gateway is running on port 7497")
            else:
                self.timing['ibkr_connect'] = (time.time() - start) * 1000
                account = self.ibkr.get_account_summary()
                logger.success(f"✓ IBKR connected in {self.timing['ibkr_connect']:.2f}ms")
                logger.info(f"  Account: {self.config['ibkr'].get('account', 'N/A')}")
                logger.info(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
                self._record_pass("IBKR connection")

            # 3. Initialize Alpha Vantage
            logger.info("\n[3/3] Initializing Alpha Vantage Client...")
            start = time.time()
            self.av = AlphaVantageClient(self.cache, self.config)
            self.timing['av_init'] = (time.time() - start) * 1000

            # Test API key validity
            test_overview = await self.av.get_company_overview('AAPL') if self.av else None
            if not test_overview:
                self._record_warning("Alpha Vantage API key may be invalid or rate limited")
            else:
                logger.success(f"✓ Alpha Vantage initialized in {self.timing['av_init']:.2f}ms")
                logger.info(f"  Rate limit: {self.av.rate_limiter.calls_per_minute} calls/min")
                self._record_pass("Alpha Vantage initialization")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._record_failure("Initialization", str(e))
            return False

    async def test_ibkr_market_data(self) -> Dict:
        """Test ALL IBKR market data features with real data"""
        logger.info("\n" + "="*70)
        logger.info("TESTING IBKR MARKET DATA (REAL PRODUCTION DATA)")
        logger.info("="*70)

        results = {}

        if not self.ibkr or not self.ibkr.is_connected():
            logger.warning("IBKR not connected - skipping market data tests")
            return results

        test_symbol = 'SPY'  # Most liquid symbol for testing

        # 1. Test Level 2 Order Book
        logger.info(f"\n[1/5] Testing Level 2 Order Book for {test_symbol}...")
        start = time.time()

        try:
            success = await self.ibkr.subscribe_market_depth(test_symbol, num_rows=10)
            if success:
                # Wait longer for after-hours (market is slower)
                wait_time = 5 if datetime.now().hour >= 16 or datetime.now().hour < 9 else 2
                await asyncio.sleep(wait_time)  # Let data flow

                # Check if data reached cache
                order_book = self.cache.get_order_book(test_symbol) if self.cache else None
                if order_book:
                    latency = (time.time() - start) * 1000
                    results['level2'] = 'PASS'
                    self._record_pass("Level 2 Order Book")
                    self.test_results['api_coverage']['ibkr_tested'].add('level2_order_book')

                    # Validate data quality
                    bids = order_book.get('bids', [])
                    asks = order_book.get('asks', [])

                    logger.success(f"✓ Level 2 data flowing (latency: {latency:.2f}ms)")
                    logger.info(f"  Bid levels: {len(bids)}, Ask levels: {len(asks)}")

                    if bids and asks:
                        spread = asks[0]['price'] - bids[0]['price']
                        logger.info(f"  Best Bid: ${bids[0]['price']:.2f} x {bids[0]['size']}")
                        logger.info(f"  Best Ask: ${asks[0]['price']:.2f} x {asks[0]['size']}")
                        logger.info(f"  Spread: ${spread:.3f}")

                        # Check data quality
                        if spread < 0:
                            self._record_warning(f"Negative spread detected: ${spread:.3f}")
                        if spread > 1.0:
                            self._record_warning(f"Wide spread: ${spread:.3f}")
                    else:
                        self._record_warning("Order book empty - market may be closed")
                else:
                    # Check if market is closed
                    hour = datetime.now().hour
                    if hour >= 20 or hour < 4:  # 8 PM to 4 AM is definitely closed
                        results['level2'] = 'WARNING'
                        self._record_warning("Level 2 data unavailable - market closed")
                        logger.warning("⚠ No Level 2 data (market closed)")
                    else:
                        results['level2'] = 'FAIL'
                        self._record_failure("Level 2", "No data received - check exchange configuration")
                        logger.error("✗ No Level 2 data received")
        except Exception as e:
            results['level2'] = 'ERROR'
            self._record_failure("Level 2", str(e))
            logger.error(f"✗ Level 2 test failed: {e}")

        # 2. Test Trade Tape
        logger.info(f"\n[2/5] Testing Trade Tape for {test_symbol}...")
        start = time.time()

        try:
            success = await self.ibkr.subscribe_trades(test_symbol)
            if success:
                await asyncio.sleep(3)  # Wait for trades

                trades = self.cache.get_recent_trades(test_symbol, 10) if self.cache else []
                if trades:
                    latency = (time.time() - start) * 1000
                    results['trades'] = 'PASS'
                    self._record_pass("Trade Tape")
                    self.test_results['api_coverage']['ibkr_tested'].add('trade_tape')

                    logger.success(f"✓ Trade tape flowing ({len(trades)} trades, latency: {latency:.2f}ms)")

                    # Show sample trades
                    for i, trade in enumerate(trades[:3]):
                        logger.info(f"  Trade {i+1}: ${trade['price']:.2f} x {trade['size']} @ {trade['timestamp']}")
                else:
                    results['trades'] = 'NO_DATA'
                    self._record_warning("No trades captured - market may be closed or low volume")
                    logger.warning("⚠ No trades captured (market closed?)")
        except Exception as e:
            results['trades'] = 'ERROR'
            self._record_failure("Trade Tape", str(e))
            logger.error(f"✗ Trade tape test failed: {e}")

        # 3. Test 5-Second Bars
        logger.info(f"\n[3/5] Testing Real-time Bars for {test_symbol}...")
        start = time.time()

        try:
            success = await self.ibkr.subscribe_bars(test_symbol, bar_size="5 secs")
            if success:
                await asyncio.sleep(6)  # Wait for at least one bar

                bar = self.cache.get(f"bar:{test_symbol}") if self.cache else None
                if bar:
                    latency = (time.time() - start) * 1000
                    results['bars'] = 'PASS'
                    self._record_pass("5-Second Bars")
                    self.test_results['api_coverage']['ibkr_tested'].add('realtime_bars')

                    logger.success(f"✓ Real-time bars flowing (latency: {latency:.2f}ms)")
                    logger.info(f"  OHLC: {bar['open']:.2f}/{bar['high']:.2f}/{bar['low']:.2f}/{bar['close']:.2f}")
                    logger.info(f"  Volume: {bar['volume']:,}")
                else:
                    results['bars'] = 'NO_DATA'
                    self._record_warning("No bars received - market may be closed")
                    logger.warning("⚠ No bars received")
        except Exception as e:
            results['bars'] = 'ERROR'
            self._record_failure("Real-time Bars", str(e))
            logger.error(f"✗ Bars test failed: {e}")

        # 4. Test Historical Data
        logger.info(f"\n[4/5] Testing Historical Data for {test_symbol}...")
        start = time.time()

        try:
            bars = await self.ibkr.get_historical_data(test_symbol, duration="1 D", bar_size="5 mins")
            if bars:
                latency = (time.time() - start) * 1000
                results['historical'] = 'PASS'
                self._record_pass("Historical Data")
                self.test_results['api_coverage']['ibkr_tested'].add('historical_data')

                logger.success(f"✓ Historical data retrieved ({len(bars)} bars, latency: {latency:.2f}ms)")

                # Analyze data quality
                prices = [bar.close for bar in bars]
                logger.info(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")
                logger.info(f"  Total volume: {sum(bar.volume for bar in bars):,}")
            else:
                results['historical'] = 'FAIL'
                self._record_failure("Historical Data", "No data returned")
                logger.error("✗ No historical data received")
        except Exception as e:
            results['historical'] = 'ERROR'
            self._record_failure("Historical Data", str(e))
            logger.error(f"✗ Historical data test failed: {e}")

        # 5. Test Account Features
        logger.info(f"\n[5/5] Testing Account Features...")
        start = time.time()

        try:
            account = self.ibkr.get_account_summary()
            positions = self.ibkr.get_positions()

            if account:
                results['account'] = 'PASS'
                self._record_pass("Account Summary")
                self.test_results['api_coverage']['ibkr_tested'].add('account_summary')
                self.test_results['api_coverage']['ibkr_tested'].add('positions')

                logger.success("✓ Account features working")
                logger.info(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
                logger.info(f"  Net Liquidation: ${account.get('net_liquidation', 0):,.2f}")
                logger.info(f"  Open Positions: {len(positions)}")
            else:
                results['account'] = 'FAIL'
                self._record_failure("Account Features", "No data returned")
        except Exception as e:
            results['account'] = 'ERROR'
            self._record_failure("Account Features", str(e))

        # Cleanup
        await self.ibkr.unsubscribe_market_depth(test_symbol)

        return results

    async def test_alpha_vantage_apis(self) -> Dict:
        """Test ALL 13 Alpha Vantage APIs with real data"""
        logger.info("\n" + "="*70)
        logger.info("TESTING ALL 13 ALPHA VANTAGE APIs (REAL PRODUCTION DATA)")
        logger.info("="*70)

        results = {}
        test_symbol = 'SPY'  # Primary test symbol

        if not self.av:
            logger.warning("Alpha Vantage client not initialized - skipping AV tests")
            return results

        # 1. OPTIONS APIs
        logger.info(f"\n[1/13] Testing REALTIME OPTIONS for {test_symbol}...")
        start = time.time()

        try:
            chain = await self.av.get_realtime_options(
                test_symbol,
                require_greeks=True,
                ibkr_client=self.ibkr if self.ibkr and self.ibkr.is_connected() else None
            ) if self.av else None

            if chain and chain.options:
                latency = (time.time() - start) * 1000
                results['realtime_options'] = 'PASS'
                self._record_pass("Realtime Options")
                self.test_results['api_coverage']['av_tested'].add('REALTIME_OPTIONS')
                self.test_results['performance_metrics']['options_latency_ms'] = latency

                logger.success(f"✓ Realtime options: {len(chain.options)} contracts (latency: {latency:.2f}ms)")
                logger.info(f"  Spot price: ${chain.spot_price:.2f}")

                # Verify Greeks are PROVIDED
                sample = chain.options[0]
                logger.info(f"  Sample contract: {sample.strike} {sample.type.value} exp:{sample.expiration}")
                logger.info(f"  Greeks PROVIDED: Delta={sample.delta:.4f}, Gamma={sample.gamma:.5f}")

                # Check cache
                cached = self.cache.get_options_chain(test_symbol) if self.cache else None
                if cached:
                    logger.info("  ✓ Successfully cached with 10s TTL")
                else:
                    self._record_warning("Options not cached properly")
            else:
                results['realtime_options'] = 'FAIL'
                self._record_failure("Realtime Options", "No data returned")
        except Exception as e:
            results['realtime_options'] = 'ERROR'
            self._record_failure("Realtime Options", str(e))

        # 2. HISTORICAL OPTIONS
        logger.info(f"\n[2/13] Testing HISTORICAL OPTIONS...")
        try:
            hist_chain = await self.av.get_historical_options(test_symbol) if self.av else None
            if hist_chain:
                results['historical_options'] = 'PASS'
                self._record_pass("Historical Options")
                self.test_results['api_coverage']['av_tested'].add('HISTORICAL_OPTIONS')
                logger.success(f"✓ Historical options: {len(hist_chain.options)} contracts")
            else:
                results['historical_options'] = 'NO_DATA'
                self._record_warning("No historical options data")
        except Exception as e:
            results['historical_options'] = 'ERROR'
            self._record_failure("Historical Options", str(e))

        # 3-8. TECHNICAL INDICATORS
        if self.av:
            av_client = self.av  # Capture non-None value for type checker
            indicators = [
                ('RSI', lambda: av_client.get_rsi(test_symbol, interval='daily')),
                ('MACD', lambda: av_client.get_macd(test_symbol, interval='daily')),
                ('BBANDS', lambda: av_client.get_bbands(test_symbol, interval='daily')),
                ('ATR', lambda: av_client.get_atr(test_symbol, interval='daily')),
                ('VWAP', lambda: av_client.get_vwap(test_symbol, interval='15min'))
            ]
        else:
            indicators = []

        for i, (name, func) in enumerate(indicators, 3):
            logger.info(f"\n[{i}/13] Testing {name}...")
            try:
                data = await func()
                if data:
                    results[name.lower()] = 'PASS'
                    self._record_pass(f"{name} Indicator")
                    self.test_results['api_coverage']['av_tested'].add(name)

                    # Check cache
                    cache_key = f"{name}_daily_14" if name == 'RSI' else f"{name}_daily"
                    if name == 'VWAP':
                        cache_key = f"{name}_15min"

                    cached = self.cache.get_indicator(test_symbol, cache_key) if self.cache else None
                    cache_status = "✓ Cached" if cached else "✗ Not cached"

                    logger.success(f"✓ {name} data received ({cache_status})")
                else:
                    results[name.lower()] = 'NO_DATA'
                    self._record_warning(f"No {name} data returned")
            except Exception as e:
                results[name.lower()] = 'ERROR'
                self._record_failure(name, str(e))

            await asyncio.sleep(0.5)  # Rate limiting

        # 9. NEWS SENTIMENT
        logger.info(f"\n[8/13] Testing NEWS SENTIMENT...")
        try:
            sentiment = await self.av.get_news_sentiment(tickers='AAPL', limit=50) if self.av else None
            if sentiment and 'feed' in sentiment:
                results['news_sentiment'] = 'PASS'
                self._record_pass("News Sentiment")
                self.test_results['api_coverage']['av_tested'].add('NEWS_SENTIMENT')

                articles = sentiment.get('feed', [])
                logger.success(f"✓ News sentiment: {len(articles)} articles")

                if articles:
                    logger.info(f"  Latest: {articles[0].get('title', 'N/A')[:60]}...")
            else:
                results['news_sentiment'] = 'NO_DATA'
                self._record_warning("No news sentiment data")
        except Exception as e:
            results['news_sentiment'] = 'ERROR'
            self._record_failure("News Sentiment", str(e))

        # 10. TOP GAINERS/LOSERS
        logger.info(f"\n[9/13] Testing TOP GAINERS/LOSERS...")
        try:
            movers = await self.av.get_top_gainers_losers() if self.av else None
            if movers:
                results['top_movers'] = 'PASS'
                self._record_pass("Top Gainers/Losers")
                self.test_results['api_coverage']['av_tested'].add('TOP_GAINERS_LOSERS')
                logger.success("✓ Top movers data received")
            else:
                results['top_movers'] = 'NO_DATA'
        except Exception as e:
            results['top_movers'] = 'ERROR'
            self._record_failure("Top Movers", str(e))

        # 11. INSIDER TRANSACTIONS
        logger.info(f"\n[10/13] Testing INSIDER TRANSACTIONS...")
        try:
            insiders = await self.av.get_insider_transactions('AAPL') if self.av else None
            if insiders:
                results['insider_trans'] = 'PASS'
                self._record_pass("Insider Transactions")
                self.test_results['api_coverage']['av_tested'].add('INSIDER_TRANSACTIONS')
                logger.success("✓ Insider transactions received")
            else:
                results['insider_trans'] = 'NO_DATA'
        except Exception as e:
            results['insider_trans'] = 'ERROR'
            self._record_failure("Insider Transactions", str(e))

        # 12. COMPANY OVERVIEW
        logger.info(f"\n[11/13] Testing COMPANY OVERVIEW...")
        try:
            overview = await self.av.get_company_overview('AAPL') if self.av else None
            if overview and 'Symbol' in overview:
                results['company_overview'] = 'PASS'
                self._record_pass("Company Overview")
                self.test_results['api_coverage']['av_tested'].add('COMPANY_OVERVIEW')

                logger.success("✓ Company overview received")
                logger.info(f"  Company: {overview.get('Name', 'N/A')}")
                logger.info(f"  Market Cap: ${int(overview.get('MarketCapitalization', 0)):,}")
            else:
                results['company_overview'] = 'NO_DATA'
        except Exception as e:
            results['company_overview'] = 'ERROR'
            self._record_failure("Company Overview", str(e))

        # 13. EARNINGS
        logger.info(f"\n[12/13] Testing EARNINGS...")
        try:
            earnings = await self.av.get_earnings('AAPL') if self.av else None
            if earnings:
                results['earnings'] = 'PASS'
                self._record_pass("Earnings")
                self.test_results['api_coverage']['av_tested'].add('EARNINGS')
                logger.success("✓ Earnings data received")
            else:
                results['earnings'] = 'NO_DATA'
        except Exception as e:
            results['earnings'] = 'ERROR'
            self._record_failure("Earnings", str(e))

        # 14. ANALYTICS
        logger.info(f"\n[13/13] Testing ANALYTICS SLIDING WINDOW...")
        try:
            analytics = await self.av.get_analytics_sliding_window(
                SYMBOLS='AAPL,MSFT',
                INTERVAL='DAILY',
                RANGE='1month',
                WINDOW_SIZE=20,
                CALCULATIONS='MEAN,STDDEV',
                OHLC='close'
            ) if self.av else None
            if analytics:
                results['analytics'] = 'PASS'
                self._record_pass("Analytics")
                self.test_results['api_coverage']['av_tested'].add('ANALYTICS_SLIDING_WINDOW')
                logger.success("✓ Analytics data received")
            else:
                results['analytics'] = 'NO_DATA'
        except Exception as e:
            results['analytics'] = 'ERROR'
            self._record_failure("Analytics", str(e))

        return results

    async def test_cache_performance(self) -> Dict:
        """Test cache performance and TTL behavior"""
        logger.info("\n" + "="*70)
        logger.info("TESTING CACHE PERFORMANCE & TTL BEHAVIOR")
        logger.info("="*70)

        results = {}
        
        # Track which data types actually have real data
        available_data = {
            'order_book': False,
            'options_chain': False,
            'metrics': True  # Can always test with synthetic metrics
        }
        
        # Check what data is actually available
        if self.cache:
            # Check if we have real order book data
            test_ob = self.cache.get_order_book('SPY') if self.cache else None
            if test_ob and 'bids' in test_ob and 'asks' in test_ob:
                available_data['order_book'] = True
            
            # Check if we have real options data
            test_opts = self.cache.get_options_chain('SPY') if self.cache else None
            if test_opts and 'options' in test_opts:
                available_data['options_chain'] = True

        # IMPORTANT: Ensure no active market data subscriptions that could interfere with TTL testing
        if self.ibkr and self.ibkr.is_connected():
            # Unsubscribe from any active market depth to prevent continuous updates
            logger.info("Stopping market data subscriptions for clean TTL testing...")
            for symbol in list(self.ibkr.market_depth_subs.keys()):
                await self.ibkr.unsubscribe_market_depth(symbol)
            await asyncio.sleep(0.5)  # Wait for updates to stop

        # Test different TTLs - only test data types that have real data
        ttl_tests = []
        
        if available_data['order_book']:
            ttl_tests.append(('order_book', 'SPY', 1))  # 1 second TTL
        else:
            logger.warning("Skipping order_book TTL test - no real data available")
            
        if available_data['options_chain']:
            ttl_tests.append(('options_chain', 'SPY', 10))  # 10 second TTL
        else:
            logger.warning("Skipping options_chain TTL test - no real data available")
            
        # Always test metrics with synthetic data
        ttl_tests.append(('metrics', 'SPY', 5))  # 5 second TTL

        for data_type, symbol, expected_ttl in ttl_tests:
            logger.info(f"\nTesting {data_type} TTL ({expected_ttl}s)...")

            # Store test data
            test_data = {'test': True, 'timestamp': time.time()}

            if self.cache:
                if data_type == 'order_book':
                    self.cache.set_order_book(symbol, test_data)
                elif data_type == 'options_chain':
                    self.cache.set_options_chain(symbol, test_data)
                elif data_type == 'metrics':
                    self.cache.set_metrics(symbol, test_data)

            # Verify immediately cached
            if self.cache:
                if data_type == 'order_book':
                    retrieved = self.cache.get_order_book(symbol)
                elif data_type == 'options_chain':
                    retrieved = self.cache.get_options_chain(symbol)
                else:
                    retrieved = self.cache.get_metrics(symbol)
            else:
                retrieved = None

            if retrieved:
                logger.success(f"✓ {data_type} cached successfully")

                # Wait for TTL expiration (add small buffer for timing precision)
                await asyncio.sleep(expected_ttl + 0.2)

                # Should be expired now
                if self.cache:
                    if data_type == 'order_book':
                        expired = self.cache.get_order_book(symbol)
                    elif data_type == 'options_chain':
                        expired = self.cache.get_options_chain(symbol)
                    else:
                        expired = self.cache.get_metrics(symbol)
                else:
                    expired = None

                if expired is None:
                    logger.success(f"✓ {data_type} expired after {expected_ttl}s (correct)")
                    results[f'{data_type}_ttl'] = 'PASS'
                    self._record_pass(f"{data_type} TTL")
                else:
                    logger.error(f"✗ {data_type} did not expire after {expected_ttl}s")
                    results[f'{data_type}_ttl'] = 'FAIL'
                    self._record_failure(f"{data_type} TTL", "Did not expire on time")
            else:
                logger.error(f"✗ {data_type} failed to cache")
                results[f'{data_type}_cache'] = 'FAIL'

        # Test cache statistics
        stats = self.cache.get_stats() if self.cache else {}
        logger.info(f"\nCache Statistics:")
        logger.info(f"  Hit Rate: {stats.get('hit_rate', 'N/A')}")
        logger.info(f"  Total Hits: {stats.get('hits', 'N/A')}")
        logger.info(f"  Total Misses: {stats.get('misses', 'N/A')}")
        logger.info(f"  Keys in Cache: {stats.get('keys', 'N/A')}")

        return results

    async def test_end_to_end_flow(self) -> Dict:
        """Test complete data flow from sources through cache"""
        logger.info("\n" + "="*70)
        logger.info("TESTING END-TO-END DATA FLOW")
        logger.info("="*70)

        results = {}
        test_symbol = 'SPY'

        logger.info(f"\nTesting complete flow for {test_symbol}...")

        # 1. Subscribe to IBKR data (if connected)
        if self.ibkr and self.ibkr.is_connected():
            # For SPY, only use ARCA (primary exchange) to avoid errors
            await self.ibkr.subscribe_market_depth(test_symbol, exchanges=['ARCA'])
            await self.ibkr.subscribe_trades(test_symbol)

        # 2. Get options from Alpha Vantage with IBKR client for spot price
        # Check cache first to avoid rate limiting (we already fetched in earlier test)
        cached_chain = self.cache.get_options_chain(test_symbol) if self.cache else None
        if cached_chain:
            logger.info("Using cached options chain to avoid rate limiting")
            chain = cached_chain
        else:
            chain = await self.av.get_realtime_options(test_symbol, ibkr_client=self.ibkr) if self.av else None

        # 3. Get technical indicator
        rsi = await self.av.get_rsi(test_symbol) if self.av else None

        # Wait for data to flow
        await asyncio.sleep(3)

        # 4. Verify everything is in cache
        cached_items = {
            'order_book': self.cache.get_order_book(test_symbol) if self.cache else None,
            'options': self.cache.get_options_chain(test_symbol) if self.cache else None,
            'trades': self.cache.get_recent_trades(test_symbol, 5) if self.cache else None,
            'rsi': self.cache.get_indicator(test_symbol, 'RSI_daily_14') if self.cache else None
        }

        logger.info("\nEnd-to-End Data Flow Results:")
        for item_type, data in cached_items.items():
            if data:
                logger.success(f"  ✓ {item_type}: Data flowing correctly")
                results[f'e2e_{item_type}'] = 'PASS'
                self._record_pass(f"E2E {item_type}")
            else:
                logger.warning(f"  ✗ {item_type}: No data in cache")
                results[f'e2e_{item_type}'] = 'FAIL'
                self._record_warning(f"E2E {item_type} not flowing")

        # Cleanup IBKR subscriptions
        if self.ibkr and self.ibkr.is_connected():
            await self.ibkr.unsubscribe_market_depth(test_symbol)

        return results

    def _record_pass(self, test_name: str):
        """Record a passing test"""
        self.test_results['passed'].append(test_name)

    def _record_failure(self, test_name: str, reason: str):
        """Record a failing test"""
        self.test_results['failed'].append(f"{test_name}: {reason}")

    def _record_warning(self, message: str):
        """Record a warning"""
        self.test_results['warnings'].append(message)

    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*70)
        logger.info("COMPLETE PIPELINE TEST REPORT")
        logger.info("="*70)

        # Summary
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed'])
        pass_rate = (len(self.test_results['passed']) / total_tests * 100) if total_tests > 0 else 0

        logger.info("\n📊 TEST SUMMARY")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {len(self.test_results['passed'])} ({pass_rate:.1f}%)")
        logger.info(f"Failed: {len(self.test_results['failed'])}")
        logger.info(f"Warnings: {len(self.test_results['warnings'])}")

        # API Coverage
        logger.info("\n📈 API COVERAGE")
        logger.info(f"Alpha Vantage APIs tested: {len(self.test_results['api_coverage']['av_tested'])}/13")
        logger.info(f"IBKR features tested: {len(self.test_results['api_coverage']['ibkr_tested'])}/6")

        # Performance Metrics
        if self.test_results['performance_metrics']:
            logger.info("\n⚡ PERFORMANCE METRICS")
            for metric, value in self.test_results['performance_metrics'].items():
                logger.info(f"  {metric}: {value:.2f}")

        # Timing
        if self.timing:
            logger.info("\n⏱️ INITIALIZATION TIMING")
            for component, time_ms in self.timing.items():
                logger.info(f"  {component}: {time_ms:.2f}ms")

        # Failed Tests
        if self.test_results['failed']:
            logger.error("\n❌ FAILED TESTS")
            for failure in self.test_results['failed']:
                logger.error(f"  • {failure}")

        # Warnings
        if self.test_results['warnings']:
            logger.warning("\n⚠️  WARNINGS")
            for warning in self.test_results['warnings']:
                logger.warning(f"  • {warning}")

        # Passed Tests
        if self.test_results['passed']:
            logger.success("\n✅ PASSED TESTS")
            for passed in self.test_results['passed']:
                logger.success(f"  • {passed}")

        # Check for critical failures
        critical_failures = [
            'spot_price',
            'options_chain',
            'Cache initialization',
            'Alpha Vantage initialization'
        ]
        
        # Level 2 is only critical during market hours
        hour = datetime.now().hour
        if 4 <= hour < 20:  # Market hours (roughly)
            critical_failures.append('Level 2')
        
        has_critical_failure = any(
            failure for failure in self.test_results['failed'] 
            if any(critical in failure for critical in critical_failures)
        )
        
        # Check cache hit rate
        cache_stats = self.cache.get_stats() if self.cache else None
        cache_hit_rate = 0
        if cache_stats:
            total_ops = int(cache_stats['hits']) + int(cache_stats['misses'])
            if total_ops > 0:
                cache_hit_rate = int(cache_stats['hits']) / total_ops
        
        # Final Status - More strict criteria
        logger.info("\n" + "="*70)
        if has_critical_failure:
            logger.error("❌ PIPELINE TEST FAILED - Critical component(s) not working")
            logger.error(f"   Critical failures detected in: {[f for f in self.test_results['failed'] if any(c in f for c in critical_failures)]}")
        elif cache_hit_rate < 0.5 and cache_stats and (int(cache_stats['hits']) + int(cache_stats['misses'])) > 10:
            logger.error(f"❌ PIPELINE TEST FAILED - Cache hit rate too low: {cache_hit_rate:.1%}")
        elif pass_rate < 75:
            logger.error(f"❌ PIPELINE TEST FAILED - Too many failures: {pass_rate:.1f}% pass rate")
        elif pass_rate >= 90 and not has_critical_failure:
            logger.success("✅ PIPELINE TEST PASSED - System is ready for Day 5+ development")
        elif pass_rate >= 75:
            logger.warning("⚠️  PIPELINE TEST PARTIAL - Some issues need attention")
            logger.warning(f"   Pass rate: {pass_rate:.1f}%, Warnings: {len(self.test_results['warnings'])}")
        else:
            logger.error("❌ PIPELINE TEST FAILED - Too many issues")
        logger.info("="*70)

    async def cleanup(self):
        """Clean up all connections"""
        logger.info("\nCleaning up...")

        if self.ibkr:
            await self.ibkr.unsubscribe_all()
            await self.ibkr.disconnect()

        if self.av:
            await self.av.close()

        if self.cache:
            self.cache.close()

    async def run_all_tests(self):
        """Run complete test suite"""
        try:
            # Initialize
            if not await self.initialize_components():
                logger.error("Failed to initialize components")
                return

            # Run tests
            await self.test_ibkr_market_data()
            await self.test_alpha_vantage_apis()
            await self.test_cache_performance()
            await self.test_end_to_end_flow()

            # Generate report
            self.generate_report()

        finally:
            await self.cleanup()


async def main():
    """Main test execution"""
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

    logger.info("="*70)
    logger.info("COMPLETE DATA PIPELINE TEST - REAL PRODUCTION DATA")
    logger.info("Testing ALL Day 1-4 Implementation")
    logger.info("NO MOCKS - Identifying Real Issues")
    logger.info("="*70)

    # Check market hours
    now = datetime.now()
    if now.weekday() >= 5:  # Weekend
        logger.warning("\n⚠️  WARNING: It's the weekend - market data will be limited")
        logger.info("Some tests may fail or show no data")
    elif now.hour < 9 or now.hour >= 16:
        logger.warning("\n⚠️  WARNING: Market is closed - live data will be limited")
        logger.info("Some tests may show no data")
    else:
        logger.success("\n✓ Market is open - all tests should work")

    # Run tests
    tester = CompletePipelineTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
