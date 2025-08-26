#!/usr/bin/env python3
"""
Day 5 End-to-End Storage Test
Tests the complete data flow: API → Storage → Database
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
api_key = os.getenv('AV_API_KEY')
if api_key:
    print(f"✅ API key loaded: {api_key[:10]}...")
else:
    print("❌ WARNING: AV_API_KEY not found in environment!")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.alpha_vantage_client import AlphaVantageClient
from src.data.market_data import MarketDataManager
from src.data.database_storage import data_storage
from src.core.logger import get_logger
from ib_insync import util
from psycopg2.extras import RealDictCursor

logger = get_logger(__name__)


class StorageIntegrationTest:
    """Test the complete data pipeline"""
    
    def __init__(self):
        self.av_client = AlphaVantageClient()
        self.ibkr_client = MarketDataManager()
        self.storage = data_storage
        self.test_symbol = 'AAPL'
        self.results = {}
        
    async def run_all_tests(self):
        """Run complete storage integration tests"""
        print("\n" + "="*80)
        print("🔬 DAY 5 END-TO-END STORAGE TEST")
        print("="*80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Test Symbol: {self.test_symbol}")
        print("="*80 + "\n")
        
        # Initialize connections
        await self.av_client.connect()
        await self.ibkr_client.connect()
        
        # Test each data type
        await self.test_options_storage()
        await self.test_indicator_storage()
        await self.test_market_data_storage()
        await self.test_news_storage()
        await self.test_analytics_storage()
        await self.test_fundamentals_storage()
        await self.test_economic_storage()
        
        # Print summary
        self.print_summary()
        
        # Cleanup
        await self.av_client.disconnect()
        await self.ibkr_client.disconnect()
    
    async def test_options_storage(self):
        """Test options data storage"""
        print("\n📦 TEST 1: OPTIONS DATA STORAGE")
        print("-" * 40)
        
        try:
            # Fetch options from Alpha Vantage
            print(f"Fetching options for {self.test_symbol}...")
            options = await self.av_client.get_realtime_options(
                self.test_symbol, 
                require_greeks=True
            )
            
            if options:
                print(f"✅ Fetched {len(options)} options with Greeks")
                
                # Store in database
                print("Storing in database...")
                rows_stored = await self.storage.store_options_chain(
                    options[:100],  # Store first 100 for testing
                    data_type='realtime'
                )
                
                print(f"✅ Stored {rows_stored} options in database")
                self.results['options'] = {'fetched': len(options), 'stored': rows_stored}
                
                # Verify Greeks are stored
                sample = options[0]
                print(f"Sample Greeks: Δ={sample.delta:.3f}, Γ={sample.gamma:.3f}, "
                      f"Θ={sample.theta:.3f}, V={sample.vega:.3f}")
                
                # Show database content
                await self.verify_database_content('options_data', 
                                                  f"WHERE symbol = '{self.test_symbol}' ORDER BY timestamp DESC", limit=5)
            else:
                print("❌ No options returned")
                self.results['options'] = {'error': 'No data'}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results['options'] = {'error': str(e)}
    
    async def test_indicator_storage(self):
        """Test technical indicator storage"""
        print("\n📊 TEST 2: TECHNICAL INDICATOR STORAGE")
        print("-" * 40)
        
        # Test ALL 16 technical indicators
        indicators_to_test = [
            'RSI', 'MACD', 'STOCH', 'WILLR', 'MOM', 'BBANDS',
            'ATR', 'ADX', 'AROON', 'CCI', 'EMA', 'SMA',
            'MFI', 'OBV', 'AD', 'VWAP'
        ]
        
        print(f"Testing {len(indicators_to_test)} technical indicators...\n")
        
        successful = []
        failed = []
        
        for indicator in indicators_to_test:
            try:
                print(f"Testing {indicator}...", end=' ')
                
                # Check if method exists
                method_name = f'get_{indicator.lower()}'
                if not hasattr(self.av_client, method_name):
                    print(f"❌ Method {method_name} not found")
                    failed.append(indicator)
                    continue
                    
                method = getattr(self.av_client, method_name)
                
                # Build parameters based on indicator
                params = {'interval': 'daily'}
                
                # Indicators requiring time_period and series_type
                if indicator in ['RSI', 'EMA', 'SMA', 'MOM', 'BBANDS']:
                    # These indicators need series_type
                    params['series_type'] = 'close'
                    params['time_period'] = 14 if indicator != 'BBANDS' else 20
                elif indicator in ['WILLR', 'CCI', 'ATR', 'ADX', 'MFI']:
                    # These indicators only need time_period, no series_type
                    params['time_period'] = 14 if indicator != 'CCI' else 20  # CCI uses 20
                
                # MACD only needs series_type
                elif indicator == 'MACD':
                    params['series_type'] = 'close'
                
                # STOCH has special parameters
                elif indicator == 'STOCH':
                    params['fastkperiod'] = 5
                    params['slowkperiod'] = 3
                    params['slowdperiod'] = 3
                
                # AROON only needs time_period
                elif indicator == 'AROON':
                    params['time_period'] = 14
                
                # Volume indicators (OBV, AD, VWAP) don't need extra params
                
                df = await method(self.test_symbol, **params)
                
                if df is not None and not df.empty:
                    print(f"✅ Fetched {len(df)} points", end=' ')
                    
                    # Store in database
                    rows_stored = await self.storage.store_technical_indicator(
                        self.test_symbol,
                        indicator,
                        df.head(50),  # Store first 50 for testing
                        interval='daily'
                    )
                    
                    print(f"→ Stored {rows_stored} records")
                    successful.append(indicator)
                    self.results[f'indicator_{indicator}'] = {
                        'fetched': len(df), 
                        'stored': rows_stored
                    }
                else:
                    print(f"❌ No data returned")
                    failed.append(indicator)
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                failed.append(indicator)
                self.results[f'indicator_{indicator}'] = {'error': str(e)}
            
            await asyncio.sleep(0.2)  # Rate limiting
        
        # Summary
        print(f"\n📊 Indicator Summary:")
        print(f"  ✅ Successful: {len(successful)}/{len(indicators_to_test)} - {successful}")
        if failed:
            print(f"  ❌ Failed: {len(failed)}/{len(indicators_to_test)} - {failed}")
        
        # Show database content
        await self.verify_database_content('technical_indicators', 
                                          f"WHERE symbol = '{self.test_symbol}' GROUP BY indicator", is_grouped=True)
    
    async def test_market_data_storage(self):
        """Test IBKR market data storage"""
        print("\n📈 TEST 3: IBKR MARKET DATA STORAGE")
        print("-" * 40)
        
        try:
            # Subscribe to real-time data
            print(f"Subscribing to {self.test_symbol}...")
            results = await self.ibkr_client.subscribe_symbols([self.test_symbol])
            
            if results.get(self.test_symbol):
                print("✅ Subscription successful")
                
                # Wait for data
                print("Waiting for data (5 seconds)...")
                await asyncio.sleep(5)
                
                # Get historical bars
                print("Fetching historical bars...")
                bars = await self.ibkr_client.get_historical_bars(
                    self.test_symbol,
                    duration='1 D',
                    bar_size='5 mins'
                )
                
                if not bars.empty:
                    print(f"✅ Fetched {len(bars)} bars")
                    
                    # Store in database
                    rows_stored = await self.storage.store_market_bars(
                        self.test_symbol,
                        bars.head(100),  # Store first 100 bars
                        bar_size='5 mins'
                    )
                    
                    print(f"✅ Stored {rows_stored} market bars")
                    self.results['market_data'] = {
                        'fetched': len(bars),
                        'stored': rows_stored
                    }
                else:
                    print("❌ No bars returned")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results['market_data'] = {'error': str(e)}
    
    async def test_news_storage(self):
        """Test news sentiment storage"""
        print("\n📰 TEST 4: NEWS SENTIMENT STORAGE")
        print("-" * 40)
        
        try:
            # First check existing news in database
            print("Checking existing news in database...")
            with self.storage.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM news_sentiment WHERE title LIKE %s", (f'%{self.test_symbol}%',))
                existing = cur.fetchone()[0]
                if existing > 0:
                    print(f"  ℹ️  Found {existing} existing articles (may affect new inserts due to duplicates)")
            
            print("Fetching news sentiment...")
            news = await self.av_client.get_news_sentiment(
                tickers=self.test_symbol,
                limit=50
            )
            
            if news and 'feed' in news:
                print(f"✅ Fetched {len(news['feed'])} articles")
                
                # Store in database
                rows_stored = await self.storage.store_news_sentiment(news)
                
                if rows_stored == 0 and existing > 0:
                    print(f"⚠️  Stored {rows_stored} NEW articles (articles already exist in DB)")
                    print("   Note: News uses URL as unique ID, duplicates are skipped")
                else:
                    print(f"✅ Stored {rows_stored} news articles")
                    
                self.results['news'] = {
                    'fetched': len(news['feed']),
                    'stored': rows_stored,
                    'existing': existing
                }
                
                # Show sample sentiment
                if news['feed']:
                    sample = news['feed'][0]
                    print(f"Sample: {sample.get('title', '')[:60]}...")
                    print(f"Sentiment: {sample.get('overall_sentiment_label', 'N/A')}")
                
                # Show database content
                await self.verify_database_content('news_sentiment', 
                                                  f"WHERE ticker_relevance::text LIKE '%{self.test_symbol}%'", limit=5)
            else:
                print("❌ No news returned")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results['news'] = {'error': str(e)}
    
    async def test_analytics_storage(self):
        """Test analytics storage"""
        print("\n📈 TEST 5: ANALYTICS STORAGE")
        print("-" * 40)
        
        try:
            print("Fetching fixed window analytics...")
            analytics = await self.av_client.get_analytics_fixed_window(
                symbols='SPY,QQQ',
                interval='DAILY',
                range='1month'
            )
            
            if analytics:
                print("✅ Fetched analytics data")
                
                # Store in database
                success = await self.storage.store_analytics(
                    'fixed_window',
                    ['SPY', 'QQQ'],
                    analytics,
                    interval='DAILY',
                    range='1month'
                )
                
                if success:
                    print("✅ Stored analytics data")
                    self.results['analytics'] = {'status': 'success'}
                else:
                    print("❌ Failed to store analytics")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results['analytics'] = {'error': str(e)}
    
    async def test_fundamentals_storage(self):
        """Test fundamentals storage"""
        print("\n📋 TEST 6: FUNDAMENTALS STORAGE")
        print("-" * 40)
        
        try:
            print("Fetching company overview...")
            overview = await self.av_client.get_overview(self.test_symbol)
            
            if overview:
                print(f"✅ Fetched overview for {overview.get('Name', self.test_symbol)}")
                
                # Store in database
                success = await self.storage.store_fundamentals(
                    self.test_symbol,
                    'overview',
                    overview
                )
                
                if success:
                    print("✅ Stored fundamentals data")
                    self.results['fundamentals'] = {'status': 'success'}
                else:
                    print("❌ Failed to store fundamentals")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results['fundamentals'] = {'error': str(e)}
    
    async def test_economic_storage(self):
        """Test economic indicator storage"""
        print("\n💹 TEST 7: ECONOMIC INDICATOR STORAGE")
        print("-" * 40)
        
        try:
            print("Fetching CPI data...")
            cpi = await self.av_client.get_cpi(interval='monthly')
            
            if isinstance(cpi, pd.DataFrame) and not cpi.empty:
                print(f"✅ Fetched {len(cpi)} CPI data points")
                
                # Debug: Check the actual structure of the data
                if 'date' in cpi.columns and 'value' in cpi.columns:
                    # CPI data from Alpha Vantage has 'date' and 'value' columns
                    print(f"   Sample dates: {cpi['date'].head(3).tolist()}")
                    print(f"   Sample values: {cpi['value'].head(3).tolist()}")
                elif hasattr(cpi, 'index'):
                    # Fallback if structure is different
                    idx_sample = cpi.index[:3].tolist()
                    print(f"   Sample index: {idx_sample}")
                
                # The DataFrame already has the correct structure from Alpha Vantage
                # with 'date' and 'value' columns - no transformation needed
                
                # Store in database
                rows_stored = await self.storage.store_economic_indicator(
                    'CPI',
                    cpi.head(50),  # Store recent 50 points
                    interval='monthly'
                )
                
                print(f"✅ Stored {rows_stored} CPI values")
                self.results['economic'] = {
                    'fetched': len(cpi),
                    'stored': rows_stored
                }
                
                # Show database content
                await self.verify_database_content('economic_indicators', 
                                                  "WHERE indicator = 'CPI' ORDER BY data_date DESC", limit=5)
            else:
                print("❌ No CPI data returned")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            self.results['economic'] = {'error': str(e)}
    
    async def verify_database_content(self, table: str, where_clause: str = "", limit: int = 5, is_grouped: bool = False):
        """Verify and display database content"""
        try:
            with self.storage.get_connection() as conn:
                from psycopg2.extras import RealDictCursor
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                # Get sample records
                if is_grouped:
                    query = f"SELECT indicator, COUNT(*) as count FROM {table} {where_clause}"
                elif table == 'news_sentiment':
                    query = f"SELECT article_id, title, sentiment_label, time_published FROM {table} {where_clause}"
                    if limit:
                        query += f" LIMIT {limit}"
                elif table == 'technical_indicators':
                    query = f"SELECT indicator, symbol, timestamp, interval FROM {table} {where_clause} ORDER BY timestamp DESC"
                    if limit:
                        query += f" LIMIT {limit}"
                elif table == 'economic_indicators':
                    query = f"SELECT indicator, data_date, value, interval FROM {table} {where_clause}"
                    if limit:
                        query += f" LIMIT {limit}"
                else:
                    query = f"SELECT * FROM {table} {where_clause}"
                    if limit:
                        query += f" LIMIT {limit}"
                
                cur.execute(query)
                results = cur.fetchall()
                
                if results:
                    print(f"\n  📊 Database Content ({table}):")
                    for row in results[:5]:  # Show first 5
                        if 'count' in row:
                            print(f"    - {row.get('indicator', 'N/A')}: {row['count']} records")
                        elif table == 'news_sentiment':
                            print(f"    - {row['title'][:50]}... [{row['sentiment_label']}]")
                        elif table == 'economic_indicators':
                            print(f"    - {row['indicator']}: {row['data_date']} = {row['value']}")
                        else:
                            print(f"    - {dict(row)}")
                else:
                    print(f"  ⚠️  No records found in {table}")
                    
        except Exception as e:
            print(f"  ❌ Error checking database: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 STORAGE TEST SUMMARY")
        print("="*80)
        
        # Check overall stats
        stats = self.storage.get_storage_stats()
        
        print("\n📈 Database Record Counts:")
        for key, value in stats.items():
            if key.endswith('_count'):
                table = key.replace('_count', '')
                emoji = {
                    'options_data': '📊',
                    'technical_indicators': '📈', 
                    'market_data': '📉',
                    'news_sentiment': '📰',
                    'analytics': '🔬',
                    'fundamentals': '📋',
                    'economic_indicators': '💹'
                }.get(table, '📌')
                print(f"  {emoji} {table}: {value:,} records")
        
        print("\n✅ Test Results:")
        success_count = 0
        fail_count = 0
        
        for test_name, result in self.results.items():
            if 'error' in result:
                print(f"  ❌ {test_name}: {result['error'][:50]}")
                fail_count += 1
            elif 'status' in result:
                print(f"  ✅ {test_name}: {result['status']}")
                success_count += 1
            elif 'stored' in result:
                print(f"  ✅ {test_name}: {result['stored']}/{result.get('fetched', '?')} stored")
                success_count += 1
        
        print("\n" + "="*80)
        print(f"✅ Successful Tests: {success_count}")
        print(f"❌ Failed Tests: {fail_count}")
        
        if fail_count == 0:
            print("\n🎉 ALL STORAGE TESTS PASSED!")
            print("📊 Data is successfully flowing from APIs → Database")
        else:
            print(f"\n⚠️ {fail_count} tests failed. Review errors above.")
        
        print("="*80)


async def main():
    """Main test runner"""
    tester = StorageIntegrationTest()
    
    try:
        # Ensure IBKR is running
        print("\n⚠️  PREREQUISITES:")
        print("   1. TWS/IB Gateway must be running")
        print("   2. PostgreSQL must be running")
        print("   3. Alpha Vantage API key must be set")
        
        input("\nPress Enter to start tests...")
        
        await tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Storage test error: {e}", exc_info=True)


if __name__ == "__main__":
    # Use ib_insync's event loop
    util.run(main())