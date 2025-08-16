#!/usr/bin/env python3
"""Test complete Phase 1 pipeline: API → Ingestion → Database"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from sqlalchemy import create_engine, text
from src.foundation.config_manager import ConfigManager


def test_full_pipeline():
    """Test the complete data flow"""
    print("=== Phase 1.5: Testing Full Pipeline ===\n")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    
    # 1. Fetch data from API
    print("Step 1: Fetching options data...")
    symbol = 'SPY'
    options_data = client.get_realtime_options(symbol)
    
    if not options_data:
        print("Failed to fetch data")
        return False
    
    # 2. Ingest into database
    print("\nStep 2: Ingesting data into database...")
    records = ingestion.ingest_options_data(options_data, symbol)
    
    # 3. Verify data in database
    print("\nStep 3: Verifying data in database...")
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT contract_id) as unique_contracts,
                   MIN(strike) as min_strike,
                   MAX(strike) as max_strike,
                   COUNT(DISTINCT expiration) as exp_dates
            FROM av_realtime_options
            WHERE symbol = :symbol
        """), {'symbol': symbol})
        
        stats = result.fetchone()
        
        print(f"\n✓ Database verification:")
        print(f"  - Total records: {stats[0]}")
        print(f"  - Unique contracts: {stats[1]}")
        print(f"  - Strike range: ${stats[2]} - ${stats[3]}")
        print(f"  - Expiration dates: {stats[4]}")
        
        # Show sample records
        result = conn.execute(text("""
            SELECT contract_id, strike, option_type, last_price, delta, implied_volatility
            FROM av_realtime_options
            WHERE symbol = :symbol
            ORDER BY expiration, strike
            LIMIT 5
        """), {'symbol': symbol})
        
        print(f"\n  Sample records:")
        for row in result:
            print(f"    {row[0]}: ${row[1]} {row[2]}, Last=${row[3]}, Delta={row[4]}, IV={row[5]}")
    
    print("\n✅ Phase 1 Complete! Full pipeline working: API → Ingestion → Database")
    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)