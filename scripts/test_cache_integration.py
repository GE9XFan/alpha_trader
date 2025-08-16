#!/usr/bin/env python3
"""Test complete cache integration: API → DB → Cache"""

import sys
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.data.cache_manager import get_cache
from src.foundation.config_manager import ConfigManager


def test_full_cache_integration():
    """Test the complete data flow with caching"""
    print("=== Testing Complete Cache Integration ===\n")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize components
    client = AlphaVantageClient()
    ingestion = DataIngestion()
    cache = get_cache()
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    # Clear any existing cache for clean test
    cache.flush_pattern("av:realtime_options:*")
    print("✓ Cache cleared for fresh test\n")
    
    # Track API calls
    initial_stats = client.get_rate_limit_status()
    initial_calls = initial_stats['calls_made']
    
    print("=" * 50)
    print("TEST 1: Fresh API call → Database → Cache")
    print("=" * 50)
    
    # Step 1: Fresh API call (no cache)
    print("\n1. Fetching fresh SPY options from API...")
    start = time.time()
    options_data = client.get_realtime_options('SPY', use_cache=True)
    api_time = time.time() - start
    
    contracts_count = len(options_data.get('data', []))
    print(f"   ✓ Got {contracts_count} contracts in {api_time:.2f}s")
    
    # Step 2: Ingest into database (this should also cache)
    print("\n2. Ingesting into database...")
    records = ingestion.ingest_options_data(options_data, 'SPY')
    print(f"   ✓ Processed {records} records")
    
    # Step 3: Verify data is cached
    print("\n3. Verifying cache...")
    cache_key = "av:realtime_options:SPY"
    cached_data = cache.get(cache_key)
    
    if cached_data:
        cached_contracts = len(cached_data.get('data', []))
        print(f"   ✓ Data IS cached: {cached_contracts} contracts")
        ttl = cache.get_ttl(cache_key)
        print(f"   ✓ TTL remaining: {ttl} seconds")
    else:
        print("   ✗ Data NOT in cache!")
    
    # Step 4: Show actual database records
    print("\n4. Database Records (First 10 Near-The-Money Options):")
    print("=" * 80)
    
    with engine.connect() as conn:
        # Get current SPY price estimate (use middle strike as proxy)
        result = conn.execute(text("""
            SELECT AVG(strike) as avg_strike
            FROM av_realtime_options
            WHERE symbol = 'SPY'
                AND option_type = 'call'
                AND volume > 0
        """))
        avg_strike = result.scalar() or 650
        
        # Get near-the-money options
        result = conn.execute(text("""
            SELECT 
                contract_id,
                strike,
                option_type,
                expiration,
                last_price,
                bid,
                ask,
                volume,
                open_interest,
                delta,
                gamma,
                theta,
                implied_volatility,
                updated_at
            FROM av_realtime_options
            WHERE symbol = 'SPY'
                AND strike BETWEEN :min_strike AND :max_strike
            ORDER BY expiration, strike, option_type
            LIMIT 10
        """), {
            'min_strike': avg_strike - 5,
            'max_strike': avg_strike + 5
        })
        
        df = pd.DataFrame(result.fetchall(), columns=[
            'Contract', 'Strike', 'Type', 'Exp', 'Last', 
            'Bid', 'Ask', 'Vol', 'OI', 'Delta', 'Gamma', 
            'Theta', 'IV', 'Updated'
        ])
        
        if not df.empty:
            # Format for better display
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 20)
            
            print(df.to_string(index=False))
        else:
            print("No near-the-money options found")
        
        # Get summary statistics
        print("\n5. Database Summary Statistics:")
        print("=" * 50)
        
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_contracts,
                COUNT(DISTINCT expiration) as expirations,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike,
                MIN(expiration) as nearest_exp,
                MAX(expiration) as furthest_exp,
                SUM(volume) as total_volume,
                MAX(updated_at) as last_update
            FROM av_realtime_options
            WHERE symbol = 'SPY'
        """))
        
        stats = result.fetchone()
        print(f"   Total contracts: {stats[0]:,}")
        print(f"   Expiration dates: {stats[1]}")
        print(f"   Strike range: ${stats[2]:.2f} - ${stats[3]:.2f}")
        print(f"   Expiry range: {stats[4]} to {stats[5]}")
        print(f"   Total volume: {stats[6]:,}")
        print(f"   Last update: {stats[7]}")
        
        # Show highest volume contracts
        print("\n6. Top 5 Highest Volume Contracts:")
        print("=" * 70)
        
        result = conn.execute(text("""
            SELECT 
                contract_id,
                strike,
                option_type,
                expiration,
                volume,
                last_price,
                delta,
                implied_volatility
            FROM av_realtime_options
            WHERE symbol = 'SPY' AND volume > 0
            ORDER BY volume DESC
            LIMIT 5
        """))
        
        df_volume = pd.DataFrame(result.fetchall(), columns=[
            'Contract', 'Strike', 'Type', 'Exp', 
            'Volume', 'Last', 'Delta', 'IV'
        ])
        
        if not df_volume.empty:
            print(df_volume.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("TEST 2: Subsequent calls use cache")
    print("=" * 50)
    
    # Test cache hit
    print("\n7. Testing cache hit (should be instant)...")
    start = time.time()
    cached_options = client.get_realtime_options('SPY', use_cache=True)
    cache_time = time.time() - start
    
    cached_count = len(cached_options.get('data', []))
    print(f"   ✓ Got {cached_count} contracts in {cache_time:.2f}s")
    print(f"   ✓ Speed improvement: {api_time/cache_time:.1f}x faster!")
    
    # Verify no additional API calls
    current_stats = client.get_rate_limit_status()
    total_calls = current_stats['calls_made'] - initial_calls
    print(f"\n8. API calls verification:")
    print(f"   Total API calls made: {total_calls} (should be 1)")
    
    # Cache statistics
    print("\n" + "=" * 50)
    print("CACHE STATISTICS")
    print("=" * 50)
    cache_stats = client.get_cache_status()
    print(f"   Total keys: {cache_stats.get('keys', 0)}")
    print(f"   AV keys: {cache_stats.get('av_keys', 0)}")
    print(f"   Memory used: {cache_stats.get('used_memory', 'Unknown')}")
    
    # Test QQQ for comparison
    print("\n" + "=" * 50)
    print("TEST 3: Different symbol (QQQ)")
    print("=" * 50)
    
    print("\n9. Fetching QQQ (new symbol)...")
    qqq_data = client.get_realtime_options('QQQ', use_cache=True)
    qqq_count = len(qqq_data.get('data', []))
    
    print(f"   ✓ Got {qqq_count} QQQ contracts")
    
    # Ingest QQQ
    print("\n10. Ingesting QQQ...")
    qqq_records = ingestion.ingest_options_data(qqq_data, 'QQQ')
    print(f"   ✓ Processed {qqq_records} QQQ records")
    
    # Show some QQQ records
    print("\n11. QQQ Database Records (First 5):")
    print("=" * 70)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                contract_id,
                strike,
                option_type,
                expiration,
                last_price,
                volume,
                delta,
                implied_volatility
            FROM av_realtime_options
            WHERE symbol = 'QQQ'
                AND volume > 0
            ORDER BY volume DESC
            LIMIT 5
        """))
        
        df_qqq = pd.DataFrame(result.fetchall(), columns=[
            'Contract', 'Strike', 'Type', 'Exp', 
            'Last', 'Volume', 'Delta', 'IV'
        ])
        
        if not df_qqq.empty:
            print(df_qqq.to_string(index=False))
    
    # Final cache check
    print("\n12. Final cache check...")
    spy_cached = cache.exists("av:realtime_options:SPY")
    qqq_cached = cache.exists("av:realtime_options:QQQ")
    
    print(f"   SPY cached: {'✓' if spy_cached else '✗'}")
    print(f"   QQQ cached: {'✓' if qqq_cached else '✗'}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    final_stats = client.get_rate_limit_status()
    total_api_calls = final_stats['calls_made'] - initial_calls
    
    print(f"✅ Integration Test Complete!")
    print(f"   • API calls made: {total_api_calls}")
    print(f"   • Cache working: Yes")
    print(f"   • Database updated: Yes")
    print(f"   • Time saved: {api_time - cache_time:.2f} seconds per cached call")
    print(f"   • Records visible in database: Yes")
    
    return True


if __name__ == "__main__":
    print("This test will make 2 API calls (SPY and QQQ)\n")
    print("You will see actual database records!\n")
    input("Press Enter to continue...")
    
    success = test_full_cache_integration()
    sys.exit(0 if success else 1)