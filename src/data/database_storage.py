"""
Database Storage Layer - Day 5 Implementation
Stores data from Alpha Vantage and IBKR based on discovered structures
"""
import psycopg2
from psycopg2.extras import RealDictCursor, Json, execute_batch
import pandas as pd
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from contextlib import contextmanager

from src.core.logger import get_logger
from src.core.exceptions import DatabaseException
from src.data.alpha_vantage_client import OptionContract

logger = get_logger(__name__)


class DataStorage:
    """
    Storage layer for all discovered data structures
    Based on Day 5 discovery results
    """
    
    def __init__(self, connection_params: Optional[Dict] = None):
        """Initialize with connection parameters"""
        self.connection_params = connection_params or {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphatrader',
            'user': 'michaelmerrick'
        }
        
        # Track storage metrics
        self.storage_stats = {
            'options_stored': 0,
            'indicators_stored': 0,
            'market_bars_stored': 0,
            'news_stored': 0
        }
    
    @contextmanager
    def get_connection(self):
        """Database connection context manager"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        finally:
            if conn:
                conn.close()
    
    async def store_options_chain(self, options: List[OptionContract], 
                                 data_type: str = 'realtime') -> int:
        """
        Store options chain from Alpha Vantage
        
        Args:
            options: List of OptionContract objects from AV
            data_type: 'realtime' or 'historical'
            
        Returns:
            Number of options stored
        """
        if not options:
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            # Prepare batch insert data
            insert_data = []
            timestamp = datetime.now()
            
            for option in options:
                insert_data.append((
                    timestamp,
                    data_type,
                    option.symbol,
                    option.strike,
                    option.expiry,
                    option.option_type,
                    option.bid,
                    option.ask,
                    option.last,
                    option.volume,
                    option.open_interest,
                    option.implied_volatility,
                    option.delta,
                    option.gamma,
                    option.theta,
                    option.vega,
                    option.rho,
                    Json({  # Store complete data as JSONB
                        'symbol': option.symbol,
                        'strike': option.strike,
                        'expiry': option.expiry,
                        'type': option.option_type,
                        'greeks': {
                            'delta': option.delta,
                            'gamma': option.gamma,
                            'theta': option.theta,
                            'vega': option.vega,
                            'rho': option.rho
                        }
                    })
                ))
            
            # Batch insert with ON CONFLICT handling
            query = """
                INSERT INTO options_data (
                    timestamp, data_type, symbol, strike, expiry, option_type,
                    bid, ask, last, volume, open_interest, implied_volatility,
                    delta, gamma, theta, vega, rho, raw_data
                ) VALUES %s
                ON CONFLICT (symbol, strike, expiry, option_type, data_type, timestamp) 
                DO NOTHING
            """
            
            # Use execute_batch for performance
            from psycopg2.extras import execute_values
            execute_values(cur, query, insert_data)
            
            conn.commit()
            
            rows_inserted = cur.rowcount
            self.storage_stats['options_stored'] += rows_inserted
            
            logger.info(f"Stored {rows_inserted}/{len(options)} options ({data_type})")
            return rows_inserted
    
    async def store_technical_indicator(self, symbol: str, indicator: str, 
                                       df: pd.DataFrame, interval: str = 'daily') -> int:
        """
        Store technical indicator data from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            indicator: Indicator name (RSI, MACD, etc.)
            df: DataFrame with indicator data
            interval: Time interval
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            insert_data = []
            timestamp = datetime.now()
            
            for date, row in df.iterrows():  # Store ALL data points - no limitations!
                # Convert pandas/numpy datetime to Python datetime
                if hasattr(date, 'to_pydatetime'):
                    data_date = date.to_pydatetime()  # type: ignore
                else:
                    data_date = date
                
                # Handle different column structures
                if indicator == 'MACD':
                    macd_val = row.get('MACD')
                    value1 = float(macd_val) if macd_val is not None else None
                    signal_val = row.get('MACD_Signal')
                    value2 = float(signal_val) if signal_val is not None else None
                    hist_val = row.get('MACD_Hist')
                    value3 = float(hist_val) if hist_val is not None else None
                elif indicator == 'BBANDS':
                    upper_val = row.get('Real Upper Band')
                    value1 = float(upper_val) if upper_val is not None else None
                    middle_val = row.get('Real Middle Band')
                    value2 = float(middle_val) if middle_val is not None else None
                    lower_val = row.get('Real Lower Band')
                    value3 = float(lower_val) if lower_val is not None else None
                elif indicator == 'STOCH':
                    slowk_val = row.get('SlowK')
                    value1 = float(slowk_val) if slowk_val is not None else None
                    slowd_val = row.get('SlowD')
                    value2 = float(slowd_val) if slowd_val is not None else None
                    value3 = None
                elif indicator == 'AROON':
                    aroon_up_val = row.get('Aroon Up')
                    value1 = float(aroon_up_val) if aroon_up_val is not None else None
                    aroon_down_val = row.get('Aroon Down')
                    value2 = float(aroon_down_val) if aroon_down_val is not None else None
                    value3 = None
                else:
                    # Single value indicators
                    col_name = df.columns[0]
                    value1 = float(row[col_name]) if row[col_name] is not None else None
                    value2 = None
                    value3 = None
                
                insert_data.append((
                    timestamp,
                    symbol,
                    indicator,
                    interval,
                    data_date,
                    value1,
                    value2,
                    value3,
                    Json(row.to_dict())
                ))
            
            query = """
                INSERT INTO technical_indicators (
                    timestamp, symbol, indicator, interval, data_date,
                    value, value2, value3, raw_data
                ) VALUES %s
                ON CONFLICT (symbol, indicator, interval, data_date)
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    value2 = EXCLUDED.value2,
                    value3 = EXCLUDED.value3,
                    timestamp = EXCLUDED.timestamp
            """
            
            from psycopg2.extras import execute_values
            
            # Count existing records before insert
            cur.execute("""
                SELECT COUNT(*) FROM technical_indicators 
                WHERE symbol = %s AND indicator = %s AND interval = %s
            """, (symbol, indicator, interval))
            before_count = cur.fetchone()[0]
            
            execute_values(cur, query, insert_data)
            conn.commit()
            
            # Count after insert to see total
            cur.execute("""
                SELECT COUNT(*) FROM technical_indicators 
                WHERE symbol = %s AND indicator = %s AND interval = %s
            """, (symbol, indicator, interval))
            after_count = cur.fetchone()[0]
            
            new_rows = after_count - before_count
            updated_rows = len(insert_data) - new_rows
            
            self.storage_stats['indicators_stored'] += new_rows
            
            logger.info(f"Processed {len(insert_data)} {indicator} values for {symbol}: "
                       f"{new_rows} new inserts, {updated_rows} updates")
            return new_rows  # Return only new inserts for backward compatibility
    
    async def store_market_bars(self, symbol: str, bars: pd.DataFrame, 
                               bar_size: str = '5 secs') -> int:
        """
        Store market data bars from IBKR
        
        Args:
            symbol: Stock symbol
            bars: DataFrame with OHLCV data
            bar_size: Bar size (5 secs, 1 min, etc.)
            
        Returns:
            Number of bars stored
        """
        if bars.empty:
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            insert_data = []
            timestamp = datetime.now()
            
            for bar_time, row in bars.iterrows():
                insert_data.append((
                    timestamp,
                    symbol,
                    bar_time,
                    float(row.get('open', 0)) if row.get('open') is not None else None,
                    float(row.get('high', 0)) if row.get('high') is not None else None,
                    float(row.get('low', 0)) if row.get('low') is not None else None,
                    float(row.get('close', 0)) if row.get('close') is not None else None,
                    int(row.get('volume', 0)) if row.get('volume') is not None else None,
                    float(row.get('wap', 0)) if row.get('wap') is not None else None,
                    int(row.get('count', 0)) if row.get('count') is not None else None,
                    bar_size
                ))
            
            query = """
                INSERT INTO market_data (
                    timestamp, symbol, bar_time, open, high, low, close,
                    volume, wap, count, bar_size
                ) VALUES %s
                ON CONFLICT (symbol, bar_time, bar_size)
                DO UPDATE SET 
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    timestamp = EXCLUDED.timestamp
            """
            
            from psycopg2.extras import execute_values
            execute_values(cur, query, insert_data)
            
            conn.commit()
            
            rows_inserted = cur.rowcount
            self.storage_stats['market_bars_stored'] += rows_inserted
            
            logger.info(f"Stored {rows_inserted} bars for {symbol}")
            return rows_inserted
    
    async def store_news_sentiment(self, news_data: Dict) -> int:
        """
        Store news sentiment from Alpha Vantage
        
        Args:
            news_data: News sentiment response from AV
            
        Returns:
            Number of articles stored
        """
        if not news_data or 'feed' not in news_data:
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            insert_data = []
            timestamp = datetime.now()
            
            for article in news_data['feed']:
                # Extract ticker relevance scores
                ticker_relevance = {}
                for ticker_data in article.get('ticker_sentiment', []):
                    ticker_relevance[ticker_data.get('ticker')] = {
                        'relevance_score': ticker_data.get('relevance_score'),
                        'ticker_sentiment_score': ticker_data.get('ticker_sentiment_score'),
                        'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label')
                    }
                
                # Parse time published
                time_str = article.get('time_published', '')
                if time_str:
                    # Format: 20240825T120000
                    time_published = datetime.strptime(time_str[:15], '%Y%m%dT%H%M%S')
                else:
                    time_published = None
                
                insert_data.append((
                    timestamp,
                    article.get('url', '')[:100],  # Use URL as ID
                    article.get('title', ''),
                    article.get('url', ''),
                    time_published,
                    article.get('authors', []),
                    article.get('summary', ''),
                    float(article.get('overall_sentiment_score', 0)),
                    article.get('overall_sentiment_label', ''),
                    Json(ticker_relevance),
                    Json(article)
                ))
            
            query = """
                INSERT INTO news_sentiment (
                    timestamp, article_id, title, url, time_published,
                    authors, summary, sentiment_score, sentiment_label,
                    ticker_relevance, raw_data
                ) VALUES %s
                ON CONFLICT (article_id) DO NOTHING
            """
            
            from psycopg2.extras import execute_values
            execute_values(cur, query, insert_data)
            
            conn.commit()
            
            rows_inserted = cur.rowcount
            self.storage_stats['news_stored'] += rows_inserted
            
            logger.info(f"Stored {rows_inserted} news articles")
            return rows_inserted
    
    async def store_analytics(self, analytics_type: str, symbols: List[str], 
                            results: Dict, **params) -> bool:
        """
        Store analytics results from Alpha Vantage
        
        Args:
            analytics_type: 'fixed_window' or 'sliding_window'
            symbols: List of symbols analyzed
            results: Analytics results
            **params: Additional parameters (interval, range, etc.)
            
        Returns:
            Success status
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO analytics (
                    analytics_type, symbols, interval, range, window_size,
                    calculations, results, raw_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                analytics_type,
                symbols,
                params.get('interval'),
                params.get('range'),
                params.get('window_size'),
                params.get('calculations', []),
                Json(results.get('payload', {})),
                Json(results)
            ))
            
            conn.commit()
            
            logger.info(f"Stored {analytics_type} analytics for {symbols}")
            return True
    
    async def store_fundamentals(self, symbol: str, data_type: str, 
                                data: Dict) -> bool:
        """
        Store fundamental data from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            data_type: Type of fundamental data
            data: Fundamental data
            
        Returns:
            Success status
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            # Extract period ending if available
            period_ending = None
            if 'quarterlyReports' in data and data['quarterlyReports']:
                period_ending = data['quarterlyReports'][0].get('fiscalDateEnding')
            elif 'annualReports' in data and data['annualReports']:
                period_ending = data['annualReports'][0].get('fiscalDateEnding')
            
            cur.execute("""
                INSERT INTO fundamentals (
                    symbol, data_type, period_ending, raw_data
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol, data_type, period_ending)
                DO UPDATE SET 
                    raw_data = EXCLUDED.raw_data,
                    timestamp = NOW()
            """, (
                symbol,
                data_type,
                period_ending,
                Json(data)
            ))
            
            conn.commit()
            
            logger.info(f"Stored {data_type} fundamentals for {symbol}")
            return True
    
    async def store_economic_indicator(self, indicator: str, df: pd.DataFrame, 
                                      **params) -> int:
        """
        Store economic indicator data from Alpha Vantage
        
        Args:
            indicator: Indicator name
            df: DataFrame with indicator data
            **params: Additional parameters (interval, maturity)
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            insert_data = []
            
            # Handle both index-based and column-based date formats
            if 'date' in df.columns:
                # Date is in a column (typical for economic indicators from Alpha Vantage)
                for idx, row in df.iterrows():
                    date_val = row['date']
                    # Convert to Python date
                    if isinstance(date_val, str):
                        from datetime import datetime as dt
                        data_date = dt.strptime(date_val, '%Y-%m-%d').date() if '-' in date_val else dt.strptime(date_val, '%Y%m%d').date()
                    elif hasattr(date_val, 'date'):
                        data_date = date_val.date()
                    elif hasattr(date_val, 'to_pydatetime'):
                        data_date = date_val.to_pydatetime().date()
                    else:
                        data_date = date_val
                    
                    # Get value - handle if it's named 'value' or is the first numeric column
                    if 'value' in row:
                        value = float(row['value']) if row['value'] is not None else None
                    else:
                        # Find first numeric value that's not the date
                        value = None
                        for col in df.columns:
                            if col != 'date' and pd.api.types.is_numeric_dtype(df[col]):
                                value = float(row[col]) if row[col] is not None else None
                                break
                    
                    insert_data.append((
                        indicator,
                        data_date,
                        value,
                        params.get('interval'),
                        params.get('maturity'),
                        Json(row.to_dict())
                    ))
            else:
                # Date is the index (typical for technical indicators)
                for date, row in df.iterrows():
                    # Convert pandas/numpy datetime to Python date
                    if hasattr(date, 'date'):
                        data_date = date.date()  # type: ignore  # Convert to date only (no time)
                    elif hasattr(date, 'to_pydatetime'):
                        data_date = date.to_pydatetime().date()  # type: ignore
                    elif isinstance(date, str):
                        from datetime import datetime as dt
                        data_date = dt.strptime(date, '%Y-%m-%d').date() if '-' in date else dt.strptime(date, '%Y%m%d').date()
                    else:
                        data_date = date
                    
                    # Convert value to float if not None
                    if isinstance(row, pd.Series):
                        if 'value' in row:
                            value = float(row['value']) if row['value'] is not None else None
                        elif len(row) > 0:
                            # Use first numeric value
                            value = float(row.iloc[0]) if row.iloc[0] is not None else None
                        else:
                            value = None
                    else:
                        value = float(row) if row is not None else None
                    
                    insert_data.append((
                        indicator,
                        data_date,
                        value,
                        params.get('interval'),
                        params.get('maturity'),
                        Json(row.to_dict() if isinstance(row, pd.Series) else {'value': row})
                    ))
            
            query = """
                INSERT INTO economic_indicators (
                    indicator, data_date, value, interval, maturity, raw_data
                ) VALUES %s
                ON CONFLICT (indicator, data_date, interval, maturity)
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    timestamp = NOW()
            """
            
            from psycopg2.extras import execute_values
            execute_values(cur, query, insert_data)
            
            conn.commit()
            
            rows_inserted = cur.rowcount
            
            logger.info(f"Stored {rows_inserted} {indicator} values")
            return rows_inserted
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            stats = {}
            
            # Count records in each table
            tables = [
                'options_data', 'technical_indicators', 'market_data',
                'news_sentiment', 'analytics', 'fundamentals', 'economic_indicators'
            ]
            
            for table in tables:
                cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = cur.fetchone()
                stats[f'{table}_count'] = result['count']
            
            # Add runtime stats
            stats.update(self.storage_stats)
            
            return stats
    
    def clear_all_data(self) -> bool:
        """Clear all data from tables (for testing)"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            tables = [
                'options_data', 'technical_indicators', 'market_data',
                'news_sentiment', 'analytics', 'fundamentals', 'economic_indicators'
            ]
            
            for table in tables:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
            
            conn.commit()
            
            logger.warning("Cleared all data from storage tables")
            return True


# Global storage instance
data_storage = DataStorage()