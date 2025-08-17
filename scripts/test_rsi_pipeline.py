#!/usr/bin/env python3
"""Test complete RSI pipeline: API → Ingestion → Database"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.foundation.config_manager import ConfigManager


def test_rsi_pipeline():
    """Test the complete RSI data flow"""
    print("=== Phase 5.1: Testing RSI Pipeline ===\n")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    
    test_symbol = 'SPY'
    
    # 1. Fetch RSI data from API
    print(f"Step 1: Fetching RSI data for {test_symbol}...")
    rsi_data = client.get_rsi(test_symbol)
    
    if not rsi_data or 'Technical Analysis: RSI' not in rsi_data:
        print("Failed to fetch RSI data")
        return False
    
    rsi_points = len(rsi_data['Technical Analysis: RSI'])
    print(f"  ✓ Got {rsi_points} RSI data points\n")
    
    # 2. Ingest into database
    print("Step 2: Ingesting RSI data into database...")
    records = ingestion.ingest_rsi_data(rsi_data, test_symbol)
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
                MIN(rsi) as min_rsi,
                MAX(rsi) as max_rsi,
                AVG(rsi) as avg_rsi
            FROM av_rsi
            WHERE symbol = :symbol
        """), {'symbol': test_symbol})
        
        stats = result.fetchone()
        
        print(f"  Database verification:")
        print(f"    Total records: {stats[0]}")
        print(f"    Date range: {stats[1]} to {stats[2]}")
        print(f"    RSI range: {stats[3]:.2f} to {stats[4]:.2f}")
        print(f"    Average RSI: {stats[5]:.2f}\n")
        
        # Show recent RSI values
        result = conn.execute(text("""
            SELECT timestamp, rsi
            FROM av_rsi
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT 5
        """), {'symbol': test_symbol})
        
        print("  Recent RSI values:")
        for row in result:
            print(f"    {row[0]}: {row[1]:.2f}")
    
    print("\n✅ Phase 5.1 RSI Pipeline Complete!")
    return True


if __name__ == "__main__":
    success = test_rsi_pipeline()
    sys.exit(0 if success else 1)