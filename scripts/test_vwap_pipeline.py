#!/usr/bin/env python3
"""Test complete VWAP pipeline: API → Ingestion → Database"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.foundation.config_manager import ConfigManager


def test_vwap_pipeline():
    """Test the complete VWAP data flow"""
    print("=== Phase 5.4: Testing VWAP Pipeline ===\n")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    
    test_symbol = 'SPY'
    
    # Get VWAP config values - NO HARDCODING!
    vwap_config = config.av_config['endpoints']['vwap']
    interval = vwap_config['default_params']['interval']
    
    # 1. Fetch VWAP data from API
    print(f"Step 1: Fetching VWAP data for {test_symbol}...")
    vwap_data = client.get_vwap(test_symbol, interval)
    
    if not vwap_data or 'Technical Analysis: VWAP' not in vwap_data:
        print("Failed to fetch VWAP data")
        return False
    
    vwap_points = len(vwap_data['Technical Analysis: VWAP'])
    print(f"  ✓ Got {vwap_points} VWAP data points\n")
    
    # 2. Ingest into database
    print("Step 2: Ingesting VWAP data into database...")
    records = ingestion.ingest_vwap_data(vwap_data, test_symbol, interval)
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
                MIN(vwap) as min_vwap,
                MAX(vwap) as max_vwap,
                AVG(vwap) as avg_vwap
            FROM av_vwap
            WHERE symbol = :symbol AND interval = :interval
        """), {'symbol': test_symbol, 'interval': interval})
        
        stats = result.fetchone()
        
        print(f"  Database verification:")
        print(f"    Total records: {stats[0]}")
        print(f"    Date range: {stats[1]} to {stats[2]}")
        print(f"    VWAP range: {stats[3]:.2f} to {stats[4]:.2f}")
        print(f"    Average VWAP: {stats[5]:.2f}\n")
        
        # Show recent VWAP values
        result = conn.execute(text("""
            SELECT timestamp, vwap
            FROM av_vwap
            WHERE symbol = :symbol AND interval = :interval
            ORDER BY timestamp DESC
            LIMIT 5
        """), {'symbol': test_symbol, 'interval': interval})
        
        print("  Recent VWAP values:")
        for row in result:
            print(f"    {row[0]}: {row[1]:.4f}")
    
    print("\n✅ Phase 5.4 VWAP Pipeline Complete!")
    return True


if __name__ == "__main__":
    success = test_vwap_pipeline()
    sys.exit(0 if success else 1)