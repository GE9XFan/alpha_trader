"""
Bar Aggregator - Creates 1-minute and 5-minute bars from 5-second bars
Phase 3.7: Multi-timeframe bar aggregation
"""

from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from decimal import Decimal
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.foundation.config_manager import ConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BarAggregator:
    """Aggregates 5-second bars into larger timeframes"""
    
    def __init__(self):
        """Initialize the aggregator with database connection"""
        self.config = ConfigManager()
        self.engine = create_engine(self.config.database_url)
        self.last_1min_aggregation = None
        self.last_5min_aggregation = None
        
    def aggregate_1min_bars(self, lookback_minutes=60):
        """
        Build 1-minute bars from 5-second bars
        
        Args:
            lookback_minutes: How many minutes back to check for missing bars
        
        Returns:
            Number of 1-minute bars created
        """
        logger.info("Starting 1-minute bar aggregation...")
        
        try:
            with self.engine.connect() as conn:
                # First, find the last 1-minute bar we have
                result = conn.execute(text("""
                    SELECT COALESCE(MAX(timestamp), '2024-01-01'::timestamp) as last_ts 
                    FROM ibkr_bars_1min
                """))
                row = result.fetchone()
                if row and row[0]:
                    # Ensure timezone-naive
                    last_timestamp = row[0].replace(tzinfo=None) if hasattr(row[0], 'tzinfo') else row[0]
                else:
                    last_timestamp = datetime(2024, 1, 1)
                
                # Get the current minute (truncated)
                current_minute = datetime.now().replace(second=0, microsecond=0)
                
                # Don't aggregate the current incomplete minute
                end_time = current_minute
                
                # Start from the last processed minute or lookback period
                start_time = max(
                    last_timestamp + timedelta(minutes=1),
                    current_minute - timedelta(minutes=lookback_minutes)
                )
                
                logger.info(f"  Aggregating from {start_time} to {end_time}")
                
                # Aggregate 5-second bars into 1-minute bars
                query = text("""
                    INSERT INTO ibkr_bars_1min (symbol, timestamp, open, high, low, close, volume, vwap, bar_count)
                    SELECT 
                        symbol,
                        date_trunc('minute', timestamp) as minute_timestamp,
                        (array_agg(open ORDER BY timestamp))[1] as open,
                        MAX(high) as high,
                        MIN(low) as low,
                        (array_agg(close ORDER BY timestamp DESC))[1] as close,
                        SUM(volume) as volume,
                        AVG(vwap) as vwap,
                        SUM(bar_count) as bar_count
                    FROM ibkr_bars_5sec
                    WHERE timestamp >= :start_time
                      AND timestamp < :end_time
                    GROUP BY symbol, date_trunc('minute', timestamp)
                                            HAVING COUNT(*) >= 6  -- Need at least 6 5-sec bars (30 seconds) for a valid minute
                    ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET
                        high = GREATEST(ibkr_bars_1min.high, EXCLUDED.high),
                        low = LEAST(ibkr_bars_1min.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap,
                        bar_count = EXCLUDED.bar_count
                """)
                
                result = conn.execute(query, {
                    'start_time': start_time,
                    'end_time': end_time
                })
                conn.commit()
                
                rows_affected = result.rowcount
                
                if rows_affected > 0:
                    logger.info(f"  ✓ Created/updated {rows_affected} one-minute bars")
                    
                    # Log sample of what was created
                    sample = conn.execute(text("""
                        SELECT symbol, COUNT(*) as bars, 
                               MIN(timestamp) as first_bar, 
                               MAX(timestamp) as last_bar
                        FROM ibkr_bars_1min
                        WHERE timestamp >= :start_time
                        GROUP BY symbol
                        ORDER BY symbol
                        LIMIT 5
                    """), {'start_time': start_time})
                    
                    for row in sample:
                        # Access by index: 0=symbol, 1=bars, 2=first_bar, 3=last_bar
                        logger.info(f"    {row[0]}: {row[1]} bars, "
                                  f"latest: {row[3].strftime('%H:%M') if row[3] else 'N/A'}")
                else:
                    logger.info("  No new 1-minute bars to create")
                
                self.last_1min_aggregation = datetime.now()
                return rows_affected
                
        except Exception as e:
            logger.error(f"Error aggregating 1-minute bars: {e}")
            return 0
    
    def aggregate_5min_bars(self, lookback_minutes=60):
        """
        Build 5-minute bars from 5-second bars ONLY
        Don't try to be clever with 1-minute bars - just use the source data
        """
        logger.info("Starting 5-minute bar aggregation...")
        
        try:
            with self.engine.connect() as conn:
                # Find the last 5-minute bar we have
                result = conn.execute(text("""
                    SELECT COALESCE(MAX(timestamp), '2024-01-01'::timestamp) as last_ts 
                    FROM ibkr_bars_5min
                """))
                row = result.fetchone()
                if row and row[0]:
                    last_timestamp = row[0].replace(tzinfo=None) if hasattr(row[0], 'tzinfo') else row[0]
                else:
                    last_timestamp = datetime(2024, 1, 1)
                
                # Get the current 5-minute boundary
                now = datetime.now()
                minutes = (now.minute // 5) * 5
                current_5min = now.replace(minute=minutes, second=0, microsecond=0)
                
                # Don't aggregate the current incomplete 5-minute period
                end_time = current_5min
                
                # Start from the last processed 5-min or lookback period
                start_time = max(
                    last_timestamp + timedelta(minutes=5),
                    current_5min - timedelta(minutes=lookback_minutes)
                )
                
                logger.info(f"  Aggregating from {start_time} to {end_time}")
                
                # SIMPLIFIED: Always aggregate from 1-minute bars if they exist
                # This is more reliable since 1-min bars are already validated
                query = text("""
                    INSERT INTO ibkr_bars_5min (symbol, timestamp, open, high, low, close, volume, vwap, bar_count)
                    SELECT 
                        symbol,
                        date_trunc('hour', timestamp) + 
                            INTERVAL '5 minutes' * FLOOR(EXTRACT(MINUTE FROM timestamp) / 5) as five_min_timestamp,
                        (array_agg(open ORDER BY timestamp))[1] as open,
                        MAX(high) as high,
                        MIN(low) as low,
                        (array_agg(close ORDER BY timestamp DESC))[1] as close,
                        SUM(volume) as volume,
                        CASE 
                            WHEN SUM(volume) > 0 THEN SUM(vwap * volume) / SUM(volume)
                            ELSE AVG(vwap)
                        END as vwap,
                        SUM(bar_count) as bar_count
                    FROM ibkr_bars_1min
                    WHERE timestamp >= :start_time
                    AND timestamp < :end_time
                    GROUP BY symbol, 
                            date_trunc('hour', timestamp) + 
                            INTERVAL '5 minutes' * FLOOR(EXTRACT(MINUTE FROM timestamp) / 5)
                    -- Relaxed: Accept partial 5-minute bars with at least 1 minute of data
                    HAVING COUNT(*) >= 1
                    ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET
                        high = GREATEST(ibkr_bars_5min.high, EXCLUDED.high),
                        low = LEAST(ibkr_bars_5min.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = ibkr_bars_5min.volume + EXCLUDED.volume,
                        vwap = CASE 
                            WHEN (ibkr_bars_5min.volume + EXCLUDED.volume) > 0 
                            THEN (ibkr_bars_5min.vwap * ibkr_bars_5min.volume + EXCLUDED.vwap * EXCLUDED.volume) 
                                / (ibkr_bars_5min.volume + EXCLUDED.volume)
                            ELSE EXCLUDED.vwap
                        END,
                        bar_count = ibkr_bars_5min.bar_count + EXCLUDED.bar_count
                """)
                
                result = conn.execute(query, {
                    'start_time': start_time,
                    'end_time': end_time
                })
                conn.commit()
                
                rows_affected = result.rowcount
                
                if rows_affected > 0:
                    logger.info(f"  ✓ Created/updated {rows_affected} five-minute bars")
                    # ... rest of logging
                else:
                    logger.info("  No new 5-minute bars to create")
                    # Log why no bars were created for debugging
                    check = conn.execute(text("""
                        SELECT COUNT(*) as total, 
                            COUNT(DISTINCT symbol) as symbols,
                            MIN(timestamp) as earliest,
                            MAX(timestamp) as latest
                        FROM ibkr_bars_1min
                        WHERE timestamp >= :start_time AND timestamp < :end_time
                    """), {'start_time': start_time, 'end_time': end_time})
                    
                    check_row = check.fetchone()
                    if check_row:
                        logger.info(f"    Debug: Found {check_row[0]} 1-min bars between {start_time} and {end_time}")
                        logger.info(f"    Debug: {check_row[1]} symbols, range: {check_row[2]} to {check_row[3]}")
                
                self.last_5min_aggregation = datetime.now()
                return rows_affected
                
        except Exception as e:
            logger.error(f"Error aggregating 5-minute bars: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0   
             
    def run_aggregation(self):
        """
        Run all aggregations in sequence
        
        Returns:
            Dictionary with aggregation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Bar Aggregation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        results = {
            '1min_bars': 0,
            '5min_bars': 0,
            'timestamp': datetime.now()
        }
        
        # First aggregate 1-minute bars
        results['1min_bars'] = self.aggregate_1min_bars()
        
        # Then aggregate 5-minute bars
        results['5min_bars'] = self.aggregate_5min_bars()
        
        logger.info(f"\nAggregation complete!")
        logger.info(f"  1-minute bars: {results['1min_bars']}")
        logger.info(f"  5-minute bars: {results['5min_bars']}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def get_statistics(self):
        """Get statistics about the bars in the database"""
        stats = {}
        
        try:
            with self.engine.connect() as conn:
                # Get counts for each table
                for table, interval in [
                    ('ibkr_bars_5sec', '5-second'),
                    ('ibkr_bars_1min', '1-minute'),
                    ('ibkr_bars_5min', '5-minute')
                ]:
                    result = conn.execute(text(f"""
                        SELECT 
                            COUNT(*) as total_bars,
                            COUNT(DISTINCT symbol) as symbols,
                            MIN(timestamp) as earliest,
                            MAX(timestamp) as latest
                        FROM {table}
                    """))
                    
                    row = result.fetchone()
                    if row:
                        # Access by index: 0=total_bars, 1=symbols, 2=earliest, 3=latest
                        # Handle timezone-aware dates
                        earliest = row[2].replace(tzinfo=None) if row[2] and hasattr(row[2], 'tzinfo') else row[2]
                        latest = row[3].replace(tzinfo=None) if row[3] and hasattr(row[3], 'tzinfo') else row[3]
                        
                        stats[interval] = {
                            'total_bars': row[0] or 0,
                            'symbols': row[1] or 0,
                            'earliest': earliest,
                            'latest': latest
                        }
                    else:
                        stats[interval] = {
                            'total_bars': 0,
                            'symbols': 0,
                            'earliest': None,
                            'latest': None
                        }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# For testing
if __name__ == "__main__":
    import sys
    
    aggregator = BarAggregator()
    
    # Check for backfill argument
    if len(sys.argv) > 1 and sys.argv[1] == '--backfill':
        print("Running backfill aggregation (last 6 hours)...")
        aggregator.aggregate_1min_bars(lookback_minutes=360)  # 6 hours
        aggregator.aggregate_5min_bars(lookback_minutes=360)  # 6 hours
    else:
        # Run normal aggregation
        results = aggregator.run_aggregation()
    
    # Show statistics
    stats = aggregator.get_statistics()
    print("\nDatabase Statistics:")
    for interval, data in stats.items():
        if data['total_bars'] > 0:
            print(f"\n{interval} bars:")
            print(f"  Total: {data['total_bars']:,}")
            print(f"  Symbols: {data['symbols']}")
            print(f"  Range: {data['earliest']} to {data['latest']}")