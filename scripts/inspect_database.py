#!/usr/bin/env python3
"""
Comprehensive Database Inspection Tool
Shows detailed view of what's stored for each API
"""
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from decimal import Decimal
import pandas as pd
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger import get_logger

logger = get_logger(__name__)


class DatabaseInspector:
    """Inspect and display database contents in detail"""
    
    def __init__(self):
        self.connection_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphatrader',
            'user': 'michaelmerrick'
        }
    
    def inspect_all(self):
        """Run complete database inspection"""
        print("\n" + "="*100)
        print("🔍 COMPREHENSIVE DATABASE INSPECTION")
        print("="*100)
        print(f"📅 Inspection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        # Inspect each table
        self.inspect_options_data()
        self.inspect_technical_indicators()
        self.inspect_market_data()
        self.inspect_news_sentiment()
        self.inspect_analytics()
        self.inspect_fundamentals()
        self.inspect_economic_indicators()
        
        # Summary statistics
        self.show_summary_statistics()
    
    def inspect_options_data(self):
        """Inspect options data table"""
        print("\n" + "="*80)
        print("📦 OPTIONS DATA (Alpha Vantage Realtime Options)")
        print("="*80)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM options_data")
            total = cur.fetchone()['total']
            print(f"📊 Total Records: {total:,}")
            
            # Get count by data type
            cur.execute("""
                SELECT data_type, COUNT(*) as count 
                FROM options_data 
                GROUP BY data_type
            """)
            print("\n📈 By Data Type:")
            for row in cur.fetchall():
                print(f"  - {row['data_type']}: {row['count']:,} records")
            
            # Get unique symbols
            cur.execute("SELECT DISTINCT symbol FROM options_data")
            symbols = [row['symbol'] for row in cur.fetchall()]
            print(f"\n🎯 Symbols: {', '.join(symbols)}")
            
            # Get date range
            cur.execute("""
                SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest,
                       MIN(expiry) as nearest_expiry, MAX(expiry) as farthest_expiry
                FROM options_data
            """)
            dates = cur.fetchone()
            print(f"\n📅 Data Range:")
            print(f"  - Oldest: {dates['oldest']}")
            print(f"  - Newest: {dates['newest']}")
            print(f"  - Nearest Expiry: {dates['nearest_expiry']}")
            print(f"  - Farthest Expiry: {dates['farthest_expiry']}")
            
            # Sample records with Greeks
            cur.execute("""
                SELECT symbol, strike, expiry, option_type, bid, ask, last,
                       volume, open_interest, implied_volatility,
                       delta, gamma, theta, vega, rho
                FROM options_data
                WHERE delta IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            
            print("\n🎯 Sample Options with Greeks:")
            for i, row in enumerate(cur.fetchall(), 1):
                print(f"\n  Option {i}:")
                print(f"    Symbol: {row['symbol']} | Type: {row['option_type']}")
                print(f"    Strike: ${row['strike']} | Expiry: {row['expiry']}")
                print(f"    Bid/Ask: ${row['bid']:.2f}/${row['ask']:.2f} | Last: ${row['last']:.2f}")
                print(f"    Volume: {row['volume']:,} | OI: {row['open_interest']:,}")
                print(f"    IV: {row['implied_volatility']:.4f}")
                print(f"    Greeks: Δ={row['delta']:.4f}, Γ={row['gamma']:.4f}, "
                      f"Θ={row['theta']:.4f}, V={row['vega']:.4f}, ρ={row['rho']:.4f}")
    
    def inspect_technical_indicators(self):
        """Inspect technical indicators table"""
        print("\n" + "="*80)
        print("📊 TECHNICAL INDICATORS (Alpha Vantage)")
        print("="*80)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM technical_indicators")
            total = cur.fetchone()['total']
            print(f"📊 Total Records: {total:,}")
            
            # Get count by indicator
            cur.execute("""
                SELECT indicator, COUNT(*) as count,
                       MIN(data_date) as oldest, MAX(data_date) as newest
                FROM technical_indicators 
                GROUP BY indicator
                ORDER BY indicator
            """)
            
            print("\n📈 Indicators Stored:")
            indicators_data = []
            for row in cur.fetchall():
                indicators_data.append([
                    row['indicator'],
                    f"{row['count']:,}",
                    str(row['oldest']),
                    str(row['newest'])
                ])
            
            print(tabulate(indicators_data, 
                          headers=['Indicator', 'Records', 'Oldest', 'Newest'],
                          tablefmt='grid'))
            
            # Sample values for each indicator type
            print("\n📊 Sample Values by Indicator:")
            
            # Single value indicators
            single_value = ['RSI', 'MOM', 'ATR', 'ADX', 'CCI', 'EMA', 'SMA', 'MFI', 'OBV', 'AD', 'WILLR', 'VWAP']
            for indicator in single_value:
                cur.execute("""
                    SELECT data_date, value 
                    FROM technical_indicators 
                    WHERE indicator = %s 
                    ORDER BY data_date DESC 
                    LIMIT 3
                """, (indicator,))
                results = cur.fetchall()
                if results:
                    print(f"\n  {indicator}:")
                    for row in results:
                        print(f"    {row['data_date']}: {row['value']:.4f}")
            
            # Multi-value indicators
            print("\n  MACD:")
            cur.execute("""
                SELECT data_date, value as macd, value2 as signal, value3 as histogram
                FROM technical_indicators 
                WHERE indicator = 'MACD' 
                ORDER BY data_date DESC 
                LIMIT 3
            """)
            for row in cur.fetchall():
                print(f"    {row['data_date']}: MACD={row['macd']:.4f}, "
                      f"Signal={row['signal']:.4f}, Hist={row['histogram']:.4f}")
            
            print("\n  BBANDS:")
            cur.execute("""
                SELECT data_date, value as upper, value2 as middle, value3 as lower
                FROM technical_indicators 
                WHERE indicator = 'BBANDS' 
                ORDER BY data_date DESC 
                LIMIT 3
            """)
            for row in cur.fetchall():
                print(f"    {row['data_date']}: Upper={row['upper']:.2f}, "
                      f"Middle={row['middle']:.2f}, Lower={row['lower']:.2f}")
            
            print("\n  STOCH:")
            cur.execute("""
                SELECT data_date, value as slowk, value2 as slowd
                FROM technical_indicators 
                WHERE indicator = 'STOCH' 
                ORDER BY data_date DESC 
                LIMIT 3
            """)
            for row in cur.fetchall():
                print(f"    {row['data_date']}: SlowK={row['slowk']:.2f}, SlowD={row['slowd']:.2f}")
            
            print("\n  AROON:")
            cur.execute("""
                SELECT data_date, value as aroon_up, value2 as aroon_down
                FROM technical_indicators 
                WHERE indicator = 'AROON' 
                ORDER BY data_date DESC 
                LIMIT 3
            """)
            for row in cur.fetchall():
                print(f"    {row['data_date']}: Up={row['aroon_up']:.2f}, Down={row['aroon_down']:.2f}")
    
    def inspect_market_data(self):
        """Inspect IBKR market data"""
        print("\n" + "="*80)
        print("📈 MARKET DATA (IBKR)")
        print("="*80)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM market_data")
            total = cur.fetchone()['total']
            print(f"📊 Total Records: {total:,}")
            
            # Get count by symbol and bar size
            cur.execute("""
                SELECT symbol, bar_size, COUNT(*) as count,
                       MIN(bar_time) as oldest, MAX(bar_time) as newest
                FROM market_data 
                GROUP BY symbol, bar_size
                ORDER BY symbol, bar_size
            """)
            
            print("\n📈 Market Data by Symbol and Bar Size:")
            for row in cur.fetchall():
                print(f"  - {row['symbol']} ({row['bar_size']}): {row['count']:,} bars")
                print(f"    Range: {row['oldest']} to {row['newest']}")
            
            # Sample OHLCV data
            cur.execute("""
                SELECT symbol, bar_time, open, high, low, close, volume, wap, count
                FROM market_data
                ORDER BY bar_time DESC
                LIMIT 5
            """)
            
            print("\n📊 Sample Market Bars (Most Recent):")
            bars_data = []
            for row in cur.fetchall():
                bars_data.append([
                    row['symbol'],
                    str(row['bar_time'])[:16],  # Truncate seconds
                    f"{row['open']:.2f}",
                    f"{row['high']:.2f}",
                    f"{row['low']:.2f}",
                    f"{row['close']:.2f}",
                    f"{row['volume']:,}",
                    f"{row['wap']:.2f}" if row['wap'] else "N/A"
                ])
            
            print(tabulate(bars_data,
                          headers=['Symbol', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'WAP'],
                          tablefmt='grid'))
    
    def inspect_news_sentiment(self):
        """Inspect news sentiment data"""
        print("\n" + "="*80)
        print("📰 NEWS SENTIMENT (Alpha Vantage)")
        print("="*80)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM news_sentiment")
            total = cur.fetchone()['total']
            print(f"📊 Total Articles: {total:,}")
            
            # Get sentiment distribution
            cur.execute("""
                SELECT sentiment_label, COUNT(*) as count
                FROM news_sentiment
                GROUP BY sentiment_label
                ORDER BY count DESC
            """)
            
            print("\n📊 Sentiment Distribution:")
            for row in cur.fetchall():
                bar_length = int(row['count'] / total * 50)
                bar = '█' * bar_length
                print(f"  {row['sentiment_label']:20} {bar} {row['count']:3} ({row['count']/total*100:.1f}%)")
            
            # Get date range
            cur.execute("""
                SELECT MIN(time_published) as oldest, MAX(time_published) as newest
                FROM news_sentiment
                WHERE time_published IS NOT NULL
            """)
            dates = cur.fetchone()
            print(f"\n📅 Article Date Range:")
            print(f"  - Oldest: {dates['oldest']}")
            print(f"  - Newest: {dates['newest']}")
            
            # Sample articles by sentiment
            sentiments = ['Bullish', 'Somewhat-Bullish', 'Neutral', 'Somewhat-Bearish', 'Bearish']
            
            print("\n📰 Sample Articles by Sentiment:")
            for sentiment in sentiments:
                cur.execute("""
                    SELECT title, sentiment_score, ticker_relevance
                    FROM news_sentiment
                    WHERE sentiment_label = %s
                    LIMIT 2
                """, (sentiment,))
                
                articles = cur.fetchall()
                if articles:
                    print(f"\n  {sentiment}:")
                    for article in articles:
                        print(f"    • {article['title'][:60]}...")
                        print(f"      Score: {article['sentiment_score']:.4f}")
                        
                        # Parse ticker relevance
                        if article['ticker_relevance']:
                            relevance = json.loads(article['ticker_relevance']) if isinstance(article['ticker_relevance'], str) else article['ticker_relevance']
                            if relevance:
                                tickers = list(relevance.keys())[:3]
                                print(f"      Tickers: {', '.join(tickers)}")
    
    def inspect_analytics(self):
        """Inspect analytics data"""
        print("\n" + "="*80)
        print("🔬 ANALYTICS (Alpha Vantage)")
        print("="*80)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM analytics")
            total = cur.fetchone()['total']
            print(f"📊 Total Analytics Records: {total:,}")
            
            # Get analytics by type
            cur.execute("""
                SELECT analytics_type, COUNT(*) as count,
                       array_agg(DISTINCT interval) as intervals,
                       array_agg(DISTINCT range) as ranges
                FROM analytics
                GROUP BY analytics_type
            """)
            
            print("\n📊 Analytics by Type:")
            for row in cur.fetchall():
                print(f"\n  {row['analytics_type'].replace('_', ' ').title()}:")
                print(f"    Count: {row['count']}")
                print(f"    Intervals: {', '.join(filter(None, row['intervals']))}")
                print(f"    Ranges: {', '.join(filter(None, row['ranges']))}")
            
            # Sample analytics results
            cur.execute("""
                SELECT analytics_type, symbols, interval, range, 
                       results::text as results_preview
                FROM analytics
                ORDER BY timestamp DESC
                LIMIT 3
            """)
            
            print("\n📈 Recent Analytics Runs:")
            for i, row in enumerate(cur.fetchall(), 1):
                print(f"\n  Run {i}:")
                print(f"    Type: {row['analytics_type']}")
                print(f"    Symbols: {', '.join(row['symbols'])}")
                print(f"    Interval: {row['interval']} | Range: {row['range']}")
                
                # Parse and show sample metrics
                try:
                    results = json.loads(row['results_preview'])
                    if 'ranking' in results:
                        print(f"    Top Ranked: {results['ranking'][0]['ticker'] if results['ranking'] else 'N/A'}")
                except:
                    pass
    
    def inspect_fundamentals(self):
        """Inspect fundamentals data"""
        print("\n" + "="*80)
        print("📋 FUNDAMENTALS (Alpha Vantage)")
        print("="*80)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM fundamentals")
            total = cur.fetchone()['total']
            print(f"📊 Total Fundamentals Records: {total:,}")
            
            # Get count by data type
            cur.execute("""
                SELECT data_type, symbol, COUNT(*) as count
                FROM fundamentals
                GROUP BY data_type, symbol
                ORDER BY data_type, symbol
            """)
            
            print("\n📊 Fundamentals by Type:")
            current_type = None
            for row in cur.fetchall():
                if row['data_type'] != current_type:
                    current_type = row['data_type']
                    print(f"\n  {current_type.replace('_', ' ').title()}:")
                print(f"    {row['symbol']}: {row['count']} records")
            
            # Sample overview data
            cur.execute("""
                SELECT symbol, raw_data::text
                FROM fundamentals
                WHERE data_type = 'overview'
                LIMIT 1
            """)
            
            overview = cur.fetchone()
            if overview:
                data = json.loads(overview['raw_data'])
                print(f"\n📊 Sample Company Overview ({overview['symbol']}):")
                print(f"    Name: {data.get('Name', 'N/A')}")
                print(f"    Sector: {data.get('Sector', 'N/A')}")
                print(f"    Industry: {data.get('Industry', 'N/A')}")
                print(f"    Market Cap: ${int(data.get('MarketCapitalization', 0)):,}")
                print(f"    PE Ratio: {data.get('PERatio', 'N/A')}")
                print(f"    52 Week High: ${data.get('52WeekHigh', 'N/A')}")
                print(f"    52 Week Low: ${data.get('52WeekLow', 'N/A')}")
                print(f"    Dividend Yield: {data.get('DividendYield', 'N/A')}%")
    
    def inspect_economic_indicators(self):
        """Inspect economic indicators data"""
        print("\n" + "="*80)
        print("💹 ECONOMIC INDICATORS (Alpha Vantage)")
        print("="*80)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM economic_indicators")
            total = cur.fetchone()['total']
            print(f"📊 Total Economic Data Points: {total:,}")
            
            # Get count by indicator
            cur.execute("""
                SELECT indicator, interval, COUNT(*) as count,
                       MIN(data_date) as oldest, MAX(data_date) as newest,
                       AVG(value) as avg_value
                FROM economic_indicators
                GROUP BY indicator, interval
                ORDER BY indicator, interval
            """)
            
            print("\n📊 Economic Indicators Stored:")
            for row in cur.fetchall():
                print(f"\n  {row['indicator']} ({row['interval'] or 'default'}):")
                print(f"    Records: {row['count']:,}")
                print(f"    Date Range: {row['oldest']} to {row['newest']}")
                print(f"    Average Value: {row['avg_value']:.2f}")
            
            # Recent values for each indicator
            cur.execute("""
                SELECT DISTINCT indicator FROM economic_indicators
            """)
            indicators = [row['indicator'] for row in cur.fetchall()]
            
            print("\n📈 Recent Values:")
            for indicator in indicators:
                cur.execute("""
                    SELECT data_date, value
                    FROM economic_indicators
                    WHERE indicator = %s
                    ORDER BY data_date DESC
                    LIMIT 5
                """, (indicator,))
                
                print(f"\n  {indicator}:")
                values_data = []
                for row in cur.fetchall():
                    values_data.append([str(row['data_date']), f"{row['value']:.3f}"])
                
                if values_data:
                    print(tabulate(values_data,
                                  headers=['Date', 'Value'],
                                  tablefmt='simple',
                                  colalign=('left', 'right')))
    
    def show_summary_statistics(self):
        """Show overall summary statistics"""
        print("\n" + "="*100)
        print("📊 OVERALL DATABASE SUMMARY")
        print("="*100)
        
        with psycopg2.connect(**self.connection_params) as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            tables = [
                ('options_data', '📦', 'Options'),
                ('technical_indicators', '📊', 'Technical Indicators'),
                ('market_data', '📈', 'Market Bars'),
                ('news_sentiment', '📰', 'News Articles'),
                ('analytics', '🔬', 'Analytics'),
                ('fundamentals', '📋', 'Fundamentals'),
                ('economic_indicators', '💹', 'Economic Data')
            ]
            
            print("\n📈 Record Counts by Table:")
            summary_data = []
            total_records = 0
            
            for table, emoji, name in tables:
                cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                count = cur.fetchone()['count']
                total_records += count
                summary_data.append([emoji, name, f"{count:,}"])
            
            print(tabulate(summary_data,
                          headers=['', 'Data Type', 'Records'],
                          tablefmt='grid'))
            
            print(f"\n📊 Total Records in Database: {total_records:,}")
            
            # Get database size
            cur.execute("""
                SELECT pg_database_size('alphatrader') as db_size
            """)
            db_size = cur.fetchone()['db_size']
            print(f"💾 Database Size: {db_size / 1024 / 1024:.2f} MB")
            
            # Get table sizes
            print("\n📦 Table Sizes:")
            for table, emoji, name in tables:
                cur.execute(f"""
                    SELECT pg_total_relation_size('{table}') as size
                """)
                size = cur.fetchone()['size']
                print(f"  {emoji} {name}: {size / 1024:.1f} KB")


def main():
    """Main entry point"""
    inspector = DatabaseInspector()
    
    try:
        inspector.inspect_all()
    except Exception as e:
        print(f"\n❌ Inspection failed: {e}")
        logger.error(f"Database inspection error: {e}", exc_info=True)


if __name__ == "__main__":
    main()