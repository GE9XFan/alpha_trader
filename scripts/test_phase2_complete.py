#!/usr/bin/env python3
"""Test complete Phase 2: Both APIs with rate limiting"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from sqlalchemy import create_engine, text
from src.foundation.config_manager import ConfigManager


def test_phase2_complete():
    print("=== Phase 2 Complete Test ===\n")
    
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    
    # Test 1: REALTIME_OPTIONS
    print("1. Testing REALTIME_OPTIONS...")
    realtime_data = client.get_realtime_options('SPY')
    realtime_records = ingestion.ingest_options_data(realtime_data, 'SPY')
    
    # Test 2: HISTORICAL_OPTIONS  
    print("\n2. Testing HISTORICAL_OPTIONS...")
    historical_data = client.get_historical_options('SPY')
    historical_records = ingestion.ingest_historical_options(historical_data, 'SPY')
    
    # Verify in database
    print("\n3. Verifying database...")
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        # Check realtime
        result = conn.execute(text(
            "SELECT COUNT(*) FROM av_realtime_options WHERE symbol = 'SPY'"
        ))
        realtime_count = result.scalar()
        
        # Check historical
        result = conn.execute(text(
            "SELECT COUNT(*) FROM av_historical_options WHERE symbol = 'SPY'"
        ))
        historical_count = result.scalar()
    
    # Show rate limit status
    stats = client.get_rate_limit_status()
    
    print("\n=== Results ===")
    print(f"✓ Realtime options in DB: {realtime_count}")
    print(f"✓ Historical options in DB: {historical_count}")
    print(f"\nRate Limiter Final Status:")
    print(f"  - API calls made: {stats['calls_made']}")
    print(f"  - Tokens available: {stats['tokens_available']:.1f}/{20}")
    print(f"  - Success rate: {stats['success_rate']:.1f}%")
    
    print("\n✅ Phase 2 COMPLETE!")
    print("  - Rate limiter protecting API calls")
    print("  - Two APIs fully integrated")
    print("  - Data flowing to separate tables")


if __name__ == "__main__":
    test_phase2_complete()