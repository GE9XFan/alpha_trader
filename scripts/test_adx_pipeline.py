#!/usr/bin/env python3
"""Test ADX full pipeline - Phase 5.6"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.foundation.config_manager import ConfigManager

def test_adx_pipeline():
    print("Testing ADX pipeline (client -> ingestion -> database)...")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    # Test with SPY
    symbol = 'SPY'
    
    # 1. Fetch ADX data
    print(f"\n1. Fetching ADX data for {symbol}...")
    adx_data = client.get_adx(symbol)
    
    if not adx_data:
        print("❌ Failed to fetch ADX data")
        return False
    
    # 2. Ingest the data
    print(f"2. Ingesting ADX data...")
    records = ingestion.ingest_adx_data(adx_data, symbol, interval='5min', time_period=14)
    
    if records == 0:
        print("❌ No records ingested")
        return False
    
    # 3. Verify in database
    print(f"3. Verifying database...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) as count, 
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest,
                   AVG(adx) as avg_adx,
                   MAX(adx) as max_adx
            FROM av_adx 
            WHERE symbol = :symbol
        """), {'symbol': symbol})
        
        row = result.fetchone()
        print(f"\n✅ ADX Pipeline Success!")
        print(f"   Records: {row[0]:,}")
        print(f"   Earliest: {row[1]}")
        print(f"   Latest: {row[2]}")
        print(f"   Avg ADX: {row[3]:.2f}")
        print(f"   Max ADX: {row[4]:.2f}")
        
        # Show trend strength interpretation
        avg_adx = float(row[3])
        if avg_adx < 25:
            trend = "Weak/No trend"
        elif avg_adx < 50:
            trend = "Strong trend"
        elif avg_adx < 75:
            trend = "Very strong trend"
        else:
            trend = "Extremely strong trend"
        
        print(f"   Trend Strength: {trend}")
        
        return True

if __name__ == "__main__":
    success = test_adx_pipeline()
    sys.exit(0 if success else 1)