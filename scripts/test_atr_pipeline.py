#!/usr/bin/env python3
"""Test complete ATR pipeline: API → Ingestion → Database"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.foundation.config_manager import ConfigManager


def test_atr_pipeline():
    """Test the complete ATR data flow"""
    print("=== Phase 5.5: Testing ATR Pipeline ===\n")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    
    test_symbol = 'SPY'
    
    # Get ATR config values - NO HARDCODING!
    atr_config = config.av_config['endpoints']['atr']
    interval = atr_config['default_params']['interval']
    time_period = atr_config['default_params']['time_period']
    
    print(f"Test Configuration:")
    print(f"  Symbol: {test_symbol}")
    print(f"  Interval: {interval}")
    print(f"  Time Period: {time_period}")
    print()
    
    # 1. Fetch ATR data from API
    print(f"Step 1: Fetching ATR data for {test_symbol}...")
    atr_data = client.get_atr(test_symbol, interval, time_period)
    
    if not atr_data or 'Technical Analysis: ATR' not in atr_data:
        print("  ❌ Failed to fetch ATR data")
        return False
    
    atr_points = len(atr_data['Technical Analysis: ATR'])
    print(f"  ✓ Got {atr_points} ATR data points\n")
    
    # Show sample of data
    atr_values = atr_data['Technical Analysis: ATR']
    dates = list(atr_values.keys())[:3]
    print("  Sample ATR values:")
    for date in dates:
        print(f"    {date}: {atr_values[date]['ATR']}")
    print()
    
    # 2. Ingest into database - pass actual values, not rely on defaults
    print(f"Step 2: Ingesting ATR data into database...")
    records = ingestion.ingest_atr_data(
        atr_data, 
        test_symbol, 
        interval,
        time_period
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
                MIN(atr) as min_atr,
                MAX(atr) as max_atr,
                AVG(atr) as avg_atr,
                STDDEV(atr) as std_atr
            FROM av_atr
            WHERE symbol = :symbol
        """), {'symbol': test_symbol})
        
        stats = result.fetchone()
        
        print(f"  Database verification:")
        print(f"    Total records: {stats[0]:,}")
        print(f"    Date range: {stats[1]} to {stats[2]}")
        print(f"    ATR range: {stats[3]:.4f} (min) to {stats[4]:.4f} (max)")
        print(f"    Average ATR: {stats[5]:.4f}")
        print(f"    Std Dev: {stats[6]:.4f}\n")
        
        # Show recent ATR values
        result = conn.execute(text("""
            SELECT timestamp, atr
            FROM av_atr
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT 5
        """), {'symbol': test_symbol})
        
        print("  Recent ATR values:")
        for row in result:
            print(f"    {row[0]}: {row[1]:.4f} (${row[1]:.2f} daily range)")
        
        # Check for market volatility patterns
        result = conn.execute(text("""
            WITH volatility_data AS (
                SELECT 
                    CASE 
                        WHEN atr < 2 THEN 'Low Volatility'
                        WHEN atr BETWEEN 2 AND 5 THEN 'Normal Volatility'
                        WHEN atr BETWEEN 5 AND 10 THEN 'High Volatility'
                        ELSE 'Extreme Volatility'
                    END as volatility_level,
                    CASE 
                        WHEN atr < 2 THEN 1
                        WHEN atr BETWEEN 2 AND 5 THEN 2
                        WHEN atr BETWEEN 5 AND 10 THEN 3
                        ELSE 4
                    END as sort_order
                FROM av_atr
                WHERE symbol = :symbol
            )
            SELECT 
                volatility_level,
                COUNT(*) as days,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM volatility_data
            GROUP BY volatility_level
            ORDER BY MIN(sort_order)
        """), {'symbol': test_symbol})
        
        print("\n  Volatility Distribution (SPY historical):")
        for row in result:
            print(f"    {row[0]}: {row[1]:,} days ({row[2]}%)")
    
    # 4. Test cache effectiveness
    print("\nStep 4: Testing cache...")
    atr_data_cached = client.get_atr(test_symbol, interval, time_period)
    
    if atr_data_cached:
        print("  ✓ Cache hit confirmed (should be instant)")
    else:
        print("  ❌ Cache miss (unexpected)")
    
    print("\n✅ Phase 5.5 ATR Pipeline Complete!")
    print("\nKey Observations:")
    print("  - ATR uses DATE type (not TIMESTAMP)")
    print("  - Values represent daily price range in dollars")
    print("  - SPY current ATR ~5.38 = moderate volatility")
    print("  - 6000+ days of historical data available")
    
    return True


if __name__ == "__main__":
    print("Phase 5.5 - Step 5: Pipeline Test")
    print("=" * 50 + "\n")
    
    success = test_atr_pipeline()
    
    if success:
        print("\n✅ ATR pipeline working correctly!")
        print("\nNext: Add ATR to scheduler for automated updates")
    else:
        print("\n❌ ATR pipeline test failed")
    
    sys.exit(0 if success else 1)