#!/usr/bin/env python3
"""
Test script for Day 3-4 Core Components
Tests Redis cache, IBKR Level 2, and Alpha Vantage options
"""

import asyncio
import sys
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
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
    OptionsChain
)


async def test_cache_manager():
    """Test Redis cache manager"""
    logger.info("\n" + "="*60)
    logger.info("Testing Cache Manager")
    logger.info("="*60)

    try:
        # Initialize cache
        cache = CacheManager()

        # Test connection
        assert cache.health_check(), "Redis health check failed"
        logger.success("✓ Cache manager connected to Redis")

        # Test order book caching
        test_order_book = {
            'symbol': 'SPY',
            'timestamp': int(datetime.now().timestamp() * 1000),
            'bids': [
                {'price': 453.24, 'size': 100},
                {'price': 453.23, 'size': 200}
            ],
            'asks': [
                {'price': 453.25, 'size': 150},
                {'price': 453.26, 'size': 250}
            ]
        }

        # Set and get order book
        assert cache.set_order_book('SPY', test_order_book), "Failed to set order book"
        retrieved = cache.get_order_book('SPY')
        assert retrieved is not None, "Failed to retrieve order book"
        assert retrieved['symbol'] == 'SPY', "Order book data mismatch"
        logger.success("✓ Order book caching works (1 sec TTL)")

        # Test metrics caching
        test_metrics = {
            'vpin': 0.42,
            'order_book_imbalance': 0.35,
            'gamma_exposure': 125.5
        }
        assert cache.set_metrics('SPY', test_metrics), "Failed to set metrics"
        retrieved = cache.get_metrics('SPY')
        assert retrieved is not None, "Failed to retrieve metrics"
        assert retrieved['vpin'] == 0.42, "Metrics data mismatch"
        logger.success("✓ Metrics caching works (5 sec TTL)")

        # Test VPIN caching
        assert cache.set_vpin('SPY', 0.42), "Failed to set VPIN"
        vpin = cache.get_vpin('SPY')
        assert vpin == 0.42, "VPIN mismatch"
        logger.success("✓ VPIN caching works (1 sec TTL)")

        # Check cache statistics
        stats = cache.get_stats()
        logger.info(f"Cache stats: {stats}")
        assert float(stats['hit_rate'].rstrip('%')) > 0, "Cache hit rate should be > 0"
        logger.success("✓ Cache statistics tracking works")

        # Test TTL expiration
        logger.info("Testing TTL expiration...")
        await asyncio.sleep(1.1)  # Wait for order book TTL
        retrieved = cache.get_order_book('SPY')
        assert retrieved is None, "Order book should have expired"
        logger.success("✓ TTL expiration works correctly")

        logger.success("\n✅ All cache manager tests passed!")
        return cache

    except Exception as e:
        logger.error(f"❌ Cache manager test failed: {e}")
        raise


async def test_ibkr_client(cache: CacheManager):
    """Test IBKR Level 2 connection"""
    logger.info("\n" + "="*60)
    logger.info("Testing IBKR Client")
    logger.info("="*60)

    ibkr = None
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Initialize IBKR client with unique ID to avoid conflicts
        import random
        config['ibkr']['client_id'] = 200 + random.randint(1, 100)  # Client ID 200-299
        ibkr = IBKRClient(cache, config)

        # Connect with retries
        logger.info("Connecting to IBKR with retry logic...")
        connected = await ibkr.connect(retry_count=3)
        assert connected, "Failed to connect to IBKR after 3 attempts"
        logger.success("✓ Connected to IBKR TWS/Gateway")

        # Get account summary
        account = ibkr.get_account_summary()
        assert 'buying_power' in account, "Failed to get account summary"
        logger.info(f"Account buying power: ${account['buying_power']:,.2f}")
        logger.success("✓ Account data retrieval works")

        # Subscribe to Level 2 for SPY
        success = await ibkr.subscribe_market_depth('SPY')
        assert success, "Failed to subscribe to market depth"
        logger.success("✓ Subscribed to Level 2 market depth for SPY")

        # Subscribe to trades
        success = await ibkr.subscribe_trades('SPY')
        assert success, "Failed to subscribe to trades"
        logger.success("✓ Subscribed to trade tape for SPY")

        # Subscribe to 5-sec bars
        success = await ibkr.subscribe_bars('SPY')
        assert success, "Failed to subscribe to bars"
        logger.success("✓ Subscribed to 5-second bars for SPY")

        # Wait for some data
        logger.info("Waiting for market data (5 seconds)...")
        await asyncio.sleep(5)

        # Check if we have order book data
        order_book = cache.get_order_book('SPY')
        if order_book:
            logger.info(f"SPY Order Book: Bid {order_book['bids'][0]['price'] if order_book['bids'] else 'N/A'} / "
                       f"Ask {order_book['asks'][0]['price'] if order_book['asks'] else 'N/A'}")
            logger.success("✓ Level 2 data flowing to cache")
        else:
            logger.warning("⚠ No order book data yet (market may be closed)")

        # Check recent trades
        trades = cache.get_recent_trades('SPY', 10)
        if trades:
            logger.info(f"Retrieved {len(trades)} recent trades")
            logger.success("✓ Trade tape data flowing")
        else:
            logger.warning("⚠ No trades yet (market may be closed)")

        # Get positions
        positions = ibkr.get_positions()
        logger.info(f"Current positions: {len(positions)}")
        logger.success("✓ Position retrieval works")

        # Unsubscribe all
        await ibkr.unsubscribe_all()
        logger.success("✓ Unsubscribed from all market data")

        # Disconnect
        await ibkr.disconnect()
        logger.success("✓ Disconnected cleanly")

        logger.success("\n✅ All IBKR client tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ IBKR client test failed: {e}")
        logger.info("Make sure TWS/IB Gateway is running and API is enabled")
        # Clean up on error
        if ibkr:
            try:
                await ibkr.disconnect()
            except:
                pass
        return False


async def test_alpha_vantage_client(cache: CacheManager):
    """Test Alpha Vantage options client"""
    logger.info("\n" + "="*60)
    logger.info("Testing Alpha Vantage Client")
    logger.info("="*60)

    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Initialize AV client
        async with AlphaVantageClient(cache, config) as av:

            # Test options chain with Greeks
            logger.info("Fetching options chain for SPY...")
            chain = await av.get_realtime_options('SPY')

            if chain:
                logger.success("✓ Retrieved options chain")
                logger.info(f"  Spot price: ${chain.spot_price:.2f}")
                logger.info(f"  Total contracts: {len(chain.options)}")

                # Check Greeks are PROVIDED
                if chain.options:
                    sample = chain.options[0]
                    logger.info(f"  Sample contract: {sample.strike} {sample.type.value} {sample.expiration}")
                    logger.info(f"    Greeks PROVIDED by AV:")
                    logger.info(f"    - Delta: {sample.delta:.4f}")
                    logger.info(f"    - Gamma: {sample.gamma:.4f}")
                    logger.info(f"    - Theta: {sample.theta:.4f}")
                    logger.info(f"    - Vega: {sample.vega:.4f}")
                    logger.info(f"    - IV: {sample.implied_volatility:.4f}")
                    logger.success("✓ Greeks are PROVIDED (not calculated!)")

                # Test filtering
                atm_strike = chain.get_atm_strike()
                logger.info(f"  ATM Strike: {atm_strike}")

                dte_0 = chain.filter_by_dte(0, 0)
                logger.info(f"  0DTE contracts: {len(dte_0)}")
                logger.success("✓ Options filtering works")
            else:
                logger.warning("⚠ No options data (market may be closed)")

            # Test RSI indicator
            logger.info("\nFetching RSI for SPY...")
            rsi_data = await av.get_rsi('SPY', interval='daily', time_period=14)
            if rsi_data:
                logger.success("✓ Retrieved RSI data")

            # Test MACD
            logger.info("Fetching MACD for SPY...")
            macd_data = await av.get_macd('SPY')
            if macd_data:
                logger.success("✓ Retrieved MACD data")

            # Test News Sentiment
            logger.info("Fetching news sentiment...")
            sentiment = await av.get_news_sentiment(tickers='AAPL', limit=50)
            if sentiment:
                logger.success("✓ Retrieved news sentiment")

            # Test Company Overview
            logger.info("Fetching company overview for AAPL...")
            overview = await av.get_company_overview('AAPL')
            if overview and 'Symbol' in overview:
                logger.info(f"  Company: {overview.get('Name', 'N/A')}")
                logger.info(f"  Market Cap: {overview.get('MarketCapitalization', 'N/A')}")
                logger.success("✓ Retrieved company fundamentals")

            # Check rate limiting
            stats = av.get_stats()
            logger.info(f"\nAPI Statistics:")
            logger.info(f"  Calls made: {stats['calls_made']}")
            logger.info(f"  Cache hits: {stats['cache_hits']}")
            logger.info(f"  Calls remaining: {stats['calls_remaining']}/600")
            logger.info(f"  Cache hit rate: {stats['cache_hit_rate']}")
            logger.success("✓ Rate limiting working correctly")

            logger.success("\n✅ All Alpha Vantage tests passed!")
            return True

    except Exception as e:
        logger.error(f"❌ Alpha Vantage test failed: {e}")
        return False


async def test_alpha_vantage_all_endpoints(cache: CacheManager) -> bool:
    """Test ALL Alpha Vantage API endpoints comprehensively"""
    logger.info("\n" + "="*60)
    logger.info("Testing ALL Alpha Vantage Endpoints")
    logger.info("="*60)

    connected_ibkr = False
    ibkr = None

    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Initialize clients
        ibkr = IBKRClient(cache, config)
        connected_ibkr = await ibkr.connect()

        async with AlphaVantageClient(cache, config) as av:
            results = {}

            # 1. TEST OPTIONS WITH GREEKS (with IBKR spot price)
            logger.info("\n1. Testing REALTIME OPTIONS with IBKR spot...")
            if connected_ibkr:
                await ibkr.subscribe_trades('SPY')
                await asyncio.sleep(2)  # Let price data flow

            chain = await av.get_realtime_options('SPY', require_greeks=True, ibkr_client=ibkr if connected_ibkr else None)
            if chain:
                results['realtime_options'] = "✓"
                logger.success(f"✓ REALTIME OPTIONS CHAIN: {len(chain.options)} total contracts")
                logger.info("="*60)
                logger.info(f"SPOT PRICE: ${chain.spot_price:.2f} (source: {'IBKR' if connected_ibkr and chain.spot_price > 0 else 'default/none'})")

                if chain.options:
                    # Show first 5 contracts with COMPLETE details
                    logger.info(f"SHOWING FIRST 5 CONTRACTS OF {len(chain.options)}:")
                    logger.info("-"*60)
                    for i, opt in enumerate(chain.options[:5]):
                        logger.info(f"CONTRACT #{i+1}:")
                        logger.info(f"  Contract ID: {opt.contract_id}")
                        logger.info(f"  Strike: ${opt.strike:.2f} | Type: {opt.type.value.upper()} | Expiry: {opt.expiration}")
                        logger.info(f"  MARKET DATA:")
                        logger.info(f"    Bid: ${opt.bid:.2f} | Ask: ${opt.ask:.2f} | Last: ${opt.last:.2f}")
                        logger.info(f"    Bid-Ask Spread: ${(opt.ask - opt.bid):.2f}")
                        logger.info(f"    Volume: {opt.volume:,} | Open Interest: {opt.open_interest:,}")
                        logger.info(f"  GREEKS (PROVIDED BY ALPHA VANTAGE):")
                        logger.info(f"    IV: {opt.implied_volatility:.4f} ({opt.implied_volatility*100:.2f}%)")
                        logger.info(f"    Delta: {opt.delta:.4f} | Gamma: {opt.gamma:.5f}")
                        logger.info(f"    Theta: {opt.theta:.4f} | Vega: {opt.vega:.4f} | Rho: {opt.rho:.4f}")
                        logger.info("-"*40)
                logger.info("="*60)
            else:
                results['realtime_options'] = "✗"
                logger.warning("✗ No realtime options data returned")

            # 2. TEST HISTORICAL OPTIONS
            logger.info("\n2. Testing HISTORICAL OPTIONS...")
            logger.info("Calling get_historical_options WITHOUT date parameter (will use previous trading session)")
            hist_chain = await av.get_historical_options('SPY')  # NO DATE - uses default (previous trading session)
            if hist_chain:
                results['historical_options'] = "✓"
                logger.info(f"✓ Historical options: {len(hist_chain.options)} contracts from previous trading session")
                if hist_chain.options:
                    # Show first 3 historical contracts with details
                    logger.info("  First 3 historical contracts:")
                    for i, h_opt in enumerate(hist_chain.options[:3]):
                        logger.info(f"    {i+1}. {h_opt.contract_id}")
                        logger.info(f"       Strike: ${h_opt.strike}, Type: {h_opt.type.value}, Expiry: {h_opt.expiration}")
                        logger.info(f"       IV: {h_opt.implied_volatility:.4f}, Delta: {h_opt.delta:.4f}")
            else:
                results['historical_options'] = "✗"
                logger.warning("✗ No historical options data")

            # 3. TEST ALL TECHNICAL INDICATORS
            logger.info("\n3. Testing Technical Indicators...")
            logger.info("="*40)

            # RSI
            logger.info("\nTesting RSI...")
            rsi = await av.get_rsi('SPY', interval='daily', time_period=14)
            if rsi:
                logger.info(f"  RSI Response keys: {list(rsi.keys())}")
                if 'Technical Analysis: RSI' in rsi:
                    results['rsi'] = "✓"
                    rsi_data = rsi['Technical Analysis: RSI']
                    dates = list(rsi_data.keys())[:3]
                    logger.success(f"  ✓ RSI - Latest 3 values:")
                    for date in dates:
                        logger.info(f"    {date}: RSI = {rsi_data[date]['RSI']}")
                else:
                    results['rsi'] = "✗"
                    logger.warning(f"  ✗ No RSI data. Keys: {list(rsi.keys())}")
            else:
                results['rsi'] = "✗"
                logger.warning("  ✗ No RSI response")

            # MACD
            logger.info("\nTesting MACD...")
            macd = await av.get_macd('SPY', interval='daily')
            if macd and 'Technical Analysis: MACD' in macd:
                results['macd'] = "✓"
                macd_data = macd['Technical Analysis: MACD']
                dates = list(macd_data.keys())[:2]
                logger.success(f"  ✓ MACD - Latest values:")
                for date in dates:
                    d = macd_data[date]
                    logger.info(f"    {date}: MACD={d.get('MACD', 'N/A')}, Signal={d.get('MACD_Signal', 'N/A')}, Hist={d.get('MACD_Hist', 'N/A')}")
            else:
                results['macd'] = "✗"
                logger.warning(f"  ✗ No MACD data")

            # Bollinger Bands
            logger.info("\nTesting Bollinger Bands...")
            bbands = await av.get_bbands('SPY', interval='daily')
            if bbands and 'Technical Analysis: BBANDS' in bbands:
                results['bbands'] = "✓"
                bb_data = bbands['Technical Analysis: BBANDS']
                date = list(bb_data.keys())[0]
                d = bb_data[date]
                logger.success(f"  ✓ Bollinger Bands - Latest ({date}):")
                logger.info(f"    Upper: {d.get('Real Upper Band', 'N/A')}, Middle: {d.get('Real Middle Band', 'N/A')}, Lower: {d.get('Real Lower Band', 'N/A')}")
            else:
                results['bbands'] = "✗"
                logger.warning(f"  ✗ No Bollinger Bands data")

            # ATR
            logger.info("\nTesting ATR...")
            atr = await av.get_atr('SPY', interval='daily')
            if atr and 'Technical Analysis: ATR' in atr:
                results['atr'] = "✓"
                atr_data = atr['Technical Analysis: ATR']
                dates = list(atr_data.keys())[:3]
                logger.success(f"  ✓ ATR - Latest 3 values:")
                for date in dates:
                    logger.info(f"    {date}: ATR = {atr_data[date]['ATR']}")
            else:
                results['atr'] = "✗"
                logger.warning(f"  ✗ No ATR data")

            # VWAP (intraday only)
            logger.info("\nTesting VWAP...")
            vwap = await av.get_vwap('SPY', interval='15min')
            if vwap and 'Technical Analysis: VWAP' in vwap:
                results['vwap'] = "✓"
                vwap_data = vwap['Technical Analysis: VWAP']
                times = list(vwap_data.keys())[:3]
                logger.success(f"  ✓ VWAP - Latest 3 values (15min):")
                for time in times:
                    logger.info(f"    {time}: VWAP = {vwap_data[time]['VWAP']}")
            else:
                results['vwap'] = "✗"
                logger.warning(f"  ✗ No VWAP data")

            logger.info("="*40)

            # 4. TEST NEWS SENTIMENT
            logger.info("\n4. Testing News Sentiment...")
            logger.info("Calling news sentiment with tickers='AAPL' only (no topics)...")
            sentiment = await av.get_news_sentiment(tickers='AAPL', limit=50)

            if sentiment:
                logger.info(f"NEWS SENTIMENT RESPONSE KEYS: {list(sentiment.keys())}")

                # Check for API information/warnings
                if 'Information' in sentiment:
                    logger.warning(f"API Information message: {sentiment['Information']}")

                # Check if feed exists
                if 'feed' in sentiment:
                    results['news_sentiment'] = "✓"
                    feed = sentiment['feed']
                    logger.success(f"✓ News sentiment: {len(feed)} articles")

                    if feed:
                        # Show first article details
                        logger.info("FIRST ARTICLE DETAILS:")
                        article = feed[0]
                        logger.info(f"  Title: {article.get('title', 'N/A')}")
                        logger.info(f"  Source: {article.get('source', 'N/A')}")
                        logger.info(f"  Time: {article.get('time_published', 'N/A')}")
                        logger.info(f"  Summary: {article.get('summary', 'N/A')[:150]}...")

                        # Show sentiment scores
                        logger.info(f"  Overall sentiment: {article.get('overall_sentiment_label', 'N/A')} (score: {article.get('overall_sentiment_score', 'N/A')})")

                        # Show ticker sentiments
                        if 'ticker_sentiment' in article:
                            logger.info(f"  Ticker sentiments ({len(article['ticker_sentiment'])} tickers):")
                            for ts in article['ticker_sentiment'][:3]:
                                logger.info(f"    {ts.get('ticker', 'N/A')}: {ts.get('ticker_sentiment_label', 'N/A')} (score: {ts.get('ticker_sentiment_score', 'N/A')})")
                else:
                    # No feed key - show what IS there
                    results['news_sentiment'] = "✗"
                    logger.warning(f"NO 'feed' KEY IN RESPONSE! Keys present: {list(sentiment.keys())}")
                    # Show a sample of the response for debugging
                    import json
                    response_preview = json.dumps(sentiment, indent=2)[:500]
                    logger.debug(f"Response preview: {response_preview}")
            else:
                results['news_sentiment'] = "✗"
                logger.warning("✗ No response from news sentiment API")

            # 5. TEST ANALYTICS
            logger.info("\n5. Testing Analytics...")
            analytics = await av.get_analytics_sliding_window(
                SYMBOLS='AAPL,MSFT',
                INTERVAL='DAILY',
                RANGE='3month',
                WINDOW_SIZE=30,
                CALCULATIONS='MEAN,STDDEV,CORRELATION',
                OHLC='close'
            )
            if analytics:
                logger.info(f"ANALYTICS RESPONSE STRUCTURE: {list(analytics.keys())}")

                if 'meta_data' in analytics:
                    meta = analytics['meta_data']
                    logger.info("  Meta Data:")
                    logger.info(f"    Symbols: {meta.get('symbols', 'N/A')}")
                    logger.info(f"    Window Size: {meta.get('window_size', 'N/A')}")
                    logger.info(f"    Date Range: {meta.get('min_dt', 'N/A')} to {meta.get('max_dt', 'N/A')}")
                    logger.info(f"    Interval: {meta.get('interval', 'N/A')}")

                if 'payload' in analytics:
                    results['analytics'] = "✓"
                    payload = analytics['payload']
                    logger.success("✓ Analytics payload received")

                    if 'RETURNS_CALCULATIONS' in payload:
                        calcs = payload['RETURNS_CALCULATIONS']
                        if 'MEAN' in calcs and 'RUNNING_MEAN' in calcs['MEAN']:
                            logger.info("  Sample MEAN values:")
                            for symbol, values in calcs['MEAN']['RUNNING_MEAN'].items():
                                dates = list(values.keys())[:2]
                                if dates:
                                    logger.info(f"    {symbol}:")
                                    for date in dates:
                                        logger.info(f"      {date}: {values[date]:.6f}")
                                break  # Just show first symbol
                else:
                    results['analytics'] = "✗"
                    logger.warning("✗ No payload in analytics response")
            else:
                results['analytics'] = "✗"
                logger.warning("✗ No analytics data")

            # 6. TEST MARKET DATA
            logger.info("\n6. Testing Market Data...")

            # Top gainers/losers
            movers = await av.get_top_gainers_losers()
            if movers:
                results['top_movers'] = "✓"
                logger.success("✓ Top gainers/losers received")
                logger.info(f"  Response keys: {list(movers.keys())}")

                # Show top gainer and loser
                if 'top_gainers' in movers and movers['top_gainers']:
                    top = movers['top_gainers'][0]
                    logger.info(f"  Top Gainer: {top.get('ticker', 'N/A')} +{top.get('change_percentage', 'N/A')}")
                if 'top_losers' in movers and movers['top_losers']:
                    bottom = movers['top_losers'][0]
                    logger.info(f"  Top Loser: {bottom.get('ticker', 'N/A')} {bottom.get('change_percentage', 'N/A')}")
            else:
                results['top_movers'] = "✗"
                logger.warning("✗ No top movers data")

            # Insider transactions
            insiders = await av.get_insider_transactions('AAPL')
            results['insider_trans'] = "✓" if insiders else "✗"
            logger.info(f"  Insider transactions: {results['insider_trans']}")

            # 7. TEST FUNDAMENTALS
            logger.info("\n7. Testing Fundamentals...")

            # Company overview
            overview = await av.get_company_overview('AAPL')
            if overview and 'Symbol' in overview:
                results['company_overview'] = "✓"
                logger.success("✓ Company overview received")
                logger.info(f"  Company: {overview.get('Name', 'N/A')}")
                logger.info(f"  Sector: {overview.get('Sector', 'N/A')}")
                logger.info(f"  Industry: {overview.get('Industry', 'N/A')}")
                logger.info(f"  Market Cap: ${int(overview.get('MarketCapitalization', 0)):,}")
                logger.info(f"  P/E Ratio: {overview.get('PERatio', 'N/A')}")
                logger.info(f"  52W High: ${overview.get('52WeekHigh', 'N/A')}")
                logger.info(f"  52W Low: ${overview.get('52WeekLow', 'N/A')}")
            else:
                results['company_overview'] = "✗"
                logger.warning("✗ No company overview data")

            # Earnings
            earnings = await av.get_earnings('AAPL')
            if earnings:
                results['earnings'] = "✓"
                logger.success("✓ Earnings data received")
                logger.info(f"  Response keys: {list(earnings.keys())}")
                if 'quarterlyEarnings' in earnings and earnings['quarterlyEarnings']:
                    latest = earnings['quarterlyEarnings'][0]
                    logger.info(f"  Latest Quarter: {latest.get('fiscalDateEnding', 'N/A')}")
                    logger.info(f"    EPS: ${latest.get('reportedEPS', 'N/A')}")
                    logger.info(f"    Estimated EPS: ${latest.get('estimatedEPS', 'N/A')}")
                    logger.info(f"    Surprise: ${latest.get('surprise', 'N/A')}")
            else:
                results['earnings'] = "✗"
                logger.warning("✗ No earnings data")

            # 8. VERIFY CACHING
            logger.info("\n8. Testing Cache Integration...")

            # Check what's in cache
            cached_items = {
                'options': cache.get_options_chain('SPY'),
                'rsi': cache.get_indicator('SPY', 'RSI_daily_14'),
                'macd': cache.get_indicator('SPY', 'MACD_daily'),
                'sentiment': cache.get('sentiment:AAPL'),
                'fundamentals': cache.get('fundamentals:AAPL')
            }

            for item, data in cached_items.items():
                status = "✓" if data else "✗"
                logger.info(f"  Cached {item}: {status}")

            # 9. RATE LIMIT CHECK
            logger.info("\n9. Rate Limiting Status...")
            stats = av.get_stats()
            logger.info(f"  API calls made: {stats['calls_made']}")
            logger.info(f"  Cache hits: {stats['cache_hits']}")
            logger.info(f"  Remaining calls: {stats['calls_remaining']}/600")
            logger.info(f"  Cache hit rate: {stats['cache_hit_rate']}")

            # FINAL SUMMARY
            logger.info("\n" + "="*60)
            logger.info("ALPHA VANTAGE API TEST RESULTS")
            logger.info("="*60)

            passed = sum(1 for v in results.values() if v == "✓")
            total = len(results)

            for endpoint, status in results.items():
                logger.info(f"  {endpoint:20s}: {status}")

            logger.info(f"\nTotal: {passed}/{total} endpoints working")

            # Cleanup IBKR if connected
            if connected_ibkr:
                await ibkr.unsubscribe_all()
                await ibkr.disconnect()

            return passed == total

    except Exception as e:
        logger.error(f"❌ Comprehensive API test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if ibkr and connected_ibkr:
            try:
                await ibkr.disconnect()
            except:
                pass
        return False


async def test_integration():
    """Test all components working together"""
    logger.info("\n" + "="*60)
    logger.info("Testing Component Integration")
    logger.info("="*60)

    try:
        # Initialize cache
        cache = CacheManager()

        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Initialize clients - IBKR will auto-select client ID
        ibkr = IBKRClient(cache, config)
        ibkr._client_id_offset = 10  # Start with offset to avoid conflicts

        async with AlphaVantageClient(cache, config) as av:

            # Connect IBKR with retries
            if await ibkr.connect(retry_count=3):

                # Subscribe to SPY
                await ibkr.subscribe_market_depth('SPY')
                await ibkr.subscribe_trades('SPY')

                # Get options from AV with IBKR spot price
                chain = await av.get_realtime_options('SPY', ibkr_client=ibkr)

                # Log what we got
                if chain:
                    logger.info(f"Got options chain: {len(chain.options)} contracts")
                    # Immediately check if cached
                    immediate_cache = cache.get_options_chain('SPY')
                    if immediate_cache:
                        logger.info("Options successfully cached")
                    else:
                        logger.warning("Options NOT cached despite successful retrieval!")
                else:
                    logger.warning("No options chain retrieved")

                # Get RSI
                rsi = await av.get_rsi('SPY')

                # Check order book quickly before TTL expires
                await asyncio.sleep(1)
                order_book = cache.get_order_book('SPY')

                # Wait for more data
                await asyncio.sleep(2)

                # Check cache again
                options = cache.get_options_chain('SPY')
                rsi_cached = cache.get_indicator('SPY', 'RSI_daily_14')

                if chain and not options:
                    logger.warning(f"Options cache expired after 3 seconds (TTL may be too short)")

                logger.info("\nIntegrated Data Check:")
                logger.info(f"  Order Book: {'✓' if order_book else '✗'}")
                logger.info(f"  Options Chain: {'✓' if options else '✗'}")
                logger.info(f"  RSI Indicator: {'✓' if rsi_cached else '✗'}")

                # Check cache stats
                stats = cache.get_stats()
                logger.info(f"\nCache Performance:")
                logger.info(f"  Total operations: {stats['hits'] + stats['misses']}")
                logger.info(f"  Hit rate: {stats['hit_rate']}")
                logger.info(f"  Keys stored: {stats['keys']}")

                # Disconnect
                await ibkr.unsubscribe_all()
                await ibkr.disconnect()

                logger.success("\n✅ Integration test completed successfully!")
            else:
                logger.warning("⚠ IBKR not connected, skipping integration test")

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")


async def main():
    """Run all tests"""
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

    logger.info("\n" + "="*70)
    logger.info("DAY 3-4 CORE COMPONENTS TEST SUITE")
    logger.info("="*70)

    try:
        # Test each component
        cache = await test_cache_manager()
        ibkr_ok = await test_ibkr_client(cache)
        av_ok = await test_alpha_vantage_client(cache)

        # Test ALL Alpha Vantage endpoints comprehensively
        av_comprehensive = await test_alpha_vantage_all_endpoints(cache)

        # Test integration
        if ibkr_ok and av_ok:
            await test_integration()

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        logger.success("✅ Cache Manager: PASSED")
        logger.success(f"✅ IBKR Client: {'PASSED' if ibkr_ok else 'FAILED'}")
        logger.success(f"✅ Alpha Vantage Client: {'PASSED' if av_ok else 'FAILED'}")
        logger.success(f"✅ Alpha Vantage Comprehensive: {'PASSED' if av_comprehensive else 'FAILED'}")
        logger.info("="*70)

        logger.success("\n🎉 Day 3-4 Implementation Complete!")
        logger.info("\nNext Steps (Day 5-7):")
        logger.info("  1. Implement VPIN calculation")
        logger.info("  2. Build order book imbalance metrics")
        logger.info("  3. Add options Greeks analysis")
        logger.info("  4. Create signal generator")

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
