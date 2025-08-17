#!/usr/bin/env python3
"""Test complete BBANDS pipeline: API → Ingestion → Database"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.foundation.config_manager import ConfigManager


def test_bbands_pipeline():
    """Test the complete BBANDS data flow"""
    print("=== Phase 5.3: Testing BBANDS Pipeline ===\n")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    
    test_symbol = 'SPY'
    
    # Get BBANDS config values - NO HARDCODING!
    bbands_config = config.av_config['endpoints']['bbands']
    interval = bbands_config['default_params']['interval']
    time_period = bbands_config['default_params']['time_period']
    series_type = bbands_config['default_params']['series_type']
    nbdevup = bbands_config['default_params']['nbdevup']
    nbdevdn = bbands_config['default_params']['nbdevdn']
    matype = bbands_config['default_params']['matype']
    
    # 1. Fetch BBANDS data from API
    print(f"Step 1: Fetching BBANDS data for {test_symbol}...")
    bbands_data = client.get_bbands(
        test_symbol,
        interval,
        time_period,
        series_type,
        nbdevup,
        nbdevdn,
        matype
    )
    
    if not bbands_data or 'Technical Analysis: BBANDS' not in bbands_data:
        print("Failed to fetch BBANDS data")
        return False
    
    bbands_points = len(bbands_data['Technical Analysis: BBANDS'])
    print(f"  ✓ Got {bbands_points} BBANDS data points\n")
    
    # 2. Ingest into database - pass ALL parameters
    print("Step 2: Ingesting BBANDS data into database...")
    records = ingestion.ingest_bbands_data(
        bbands_data, 
        test_symbol, 
        interval,
        time_period,
        nbdevup,
        nbdevdn,
        matype,
        series_type
    )
    print(f"  ✓ Processed {records} records\n")
    
    # 3. Verify data in database
    print("Step 3: Verifying data in database...")
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        # Get summary stats
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                MIN(lower_band) as min_lower,
                MAX(upper_band) as max_upper,
                AVG(middle_band) as avg_middle,
                AVG(upper_band - lower_band) as avg_bandwidth
            FROM av_bbands
            WHERE symbol = :symbol
        """), {'symbol': test_symbol})
        
        stats = result.fetchone()
        
        print(f"  Database verification:")
        print(f"    Total records: {stats[0]}")
        print(f"    Date range: {stats[1]} to {stats[2]}")
        print(f"    Band range: {stats[3]:.2f} (lower) to {stats[4]:.2f} (upper)")
        print(f"    Average middle: {stats[5]:.2f}")
        print(f"    Average bandwidth: {stats[6]:.2f}\n")
        
        # Show recent BBANDS values
        result = conn.execute(text("""
            SELECT timestamp, upper_band, middle_band, lower_band,
                   (upper_band - lower_band) as bandwidth
            FROM av_bbands
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT 5
        """), {'symbol': test_symbol})
        
        print("  Recent BBANDS values:")
        for row in result:
            print(f"    {row[0]}: U={row[1]:.2f}, M={row[2]:.2f}, L={row[3]:.2f}, BW={row[4]:.2f}")
        
        # Check for squeeze conditions (narrow bands)
        result = conn.execute(text("""
            WITH band_stats AS (
                SELECT 
                    AVG(upper_band - lower_band) as avg_bandwidth,
                    STDDEV(upper_band - lower_band) as std_bandwidth
                FROM av_bbands
                WHERE symbol = :symbol
            )
            SELECT COUNT(*) 
            FROM av_bbands b, band_stats s
            WHERE b.symbol = :symbol
              AND (b.upper_band - b.lower_band) < (s.avg_bandwidth - s.std_bandwidth)
        """), {'symbol': test_symbol})
        
        squeeze_count = result.scalar()
        print(f"\n  Bollinger squeeze conditions detected: {squeeze_count}")
    
    print("\n✅ Phase 5.3 BBANDS Pipeline Complete!")
    return True


if __name__ == "__main__":
    success = test_bbands_pipeline()
    sys.exit(0 if success else 1)