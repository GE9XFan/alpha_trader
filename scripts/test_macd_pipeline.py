#!/usr/bin/env python3
"""Test complete MACD pipeline: API → Ingestion → Database"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.foundation.config_manager import ConfigManager


def test_macd_pipeline():
    """Test the complete MACD data flow"""
    print("=== Phase 5.2: Testing MACD Pipeline ===\n")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    
    test_symbol = 'SPY'
    
    # Get MACD config values
    macd_config = config.av_config['endpoints']['macd']
    interval = macd_config['default_params']['interval']
    fastperiod = macd_config['default_params']['fastperiod']
    slowperiod = macd_config['default_params']['slowperiod']
    signalperiod = macd_config['default_params']['signalperiod']
    series_type = macd_config['default_params']['series_type']
    
    # 1. Fetch MACD data from API
    print(f"Step 1: Fetching MACD data for {test_symbol}...")
    macd_data = client.get_macd(test_symbol)  # Uses config defaults internally
    
    if not macd_data or 'Technical Analysis: MACD' not in macd_data:
        print("Failed to fetch MACD data")
        return False
    
    macd_points = len(macd_data['Technical Analysis: MACD'])
    print(f"  ✓ Got {macd_points} MACD data points\n")
    
    # 2. Ingest into database - pass ALL parameters
    print("Step 2: Ingesting MACD data into database...")
    records = ingestion.ingest_macd_data(
        macd_data, 
        test_symbol, 
        interval,
        fastperiod,
        slowperiod,
        signalperiod,
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
                MIN(macd) as min_macd,
                MAX(macd) as max_macd,
                AVG(macd) as avg_macd,
                MIN(macd_hist) as min_hist,
                MAX(macd_hist) as max_hist
            FROM av_macd
            WHERE symbol = :symbol
        """), {'symbol': test_symbol})
        
        stats = result.fetchone()
        
        print(f"  Database verification:")
        print(f"    Total records: {stats[0]}")
        print(f"    Date range: {stats[1]} to {stats[2]}")
        print(f"    MACD range: {stats[3]:.4f} to {stats[4]:.4f}")
        print(f"    Average MACD: {stats[5]:.4f}")
        print(f"    Histogram range: {stats[6]:.4f} to {stats[7]:.4f}\n")
        
        # Show recent MACD values
        result = conn.execute(text("""
            SELECT timestamp, macd, macd_signal, macd_hist
            FROM av_macd
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT 5
        """), {'symbol': test_symbol})
        
        print("  Recent MACD values:")
        for row in result:
            print(f"    {row[0]}: MACD={row[1]:.4f}, Signal={row[2]:.4f}, Hist={row[3]:.4f}")
        
        # Check for crossovers (histogram changes sign)
        result = conn.execute(text("""
            WITH crossovers AS (
                SELECT 
                    timestamp,
                    macd_hist,
                    LAG(macd_hist) OVER (ORDER BY timestamp) as prev_hist
                FROM av_macd
                WHERE symbol = :symbol
            )
            SELECT COUNT(*) 
            FROM crossovers
            WHERE (macd_hist > 0 AND prev_hist < 0) 
               OR (macd_hist < 0 AND prev_hist > 0)
        """), {'symbol': test_symbol})
        
        crossover_count = result.scalar()
        print(f"\n  Signal crossovers detected: {crossover_count}")
    
    print("\n✅ Phase 5.2 MACD Pipeline Complete!")
    return True


if __name__ == "__main__":
    success = test_macd_pipeline()
    sys.exit(0 if success else 1)