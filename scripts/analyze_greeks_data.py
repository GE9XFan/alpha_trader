#!/usr/bin/env python3
"""
Phase 6.1 - Step 1: Analyze why Greeks are NULL and what data we're actually getting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager

def analyze_greeks_data():
    """Analyze Greeks values and investigate NULL issues"""
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    print("=== Phase 6.1: Greeks Data Analysis ===\n")
    print(f"Timestamp: {datetime.now()}\n")
    
    with engine.connect() as conn:
        # 1. Check how many records have Greeks vs NULL
        print("1. Greeks Population Check:")
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(delta) as has_delta,
                COUNT(gamma) as has_gamma,
                COUNT(theta) as has_theta,
                COUNT(vega) as has_vega,
                COUNT(rho) as has_rho
            FROM av_realtime_options
            WHERE updated_at > NOW() - INTERVAL '24 hours'
        """))
        row = result.fetchone()
        total = row[0]
        print(f"   Total Records (24h): {total}")
        print(f"   Has Delta: {row[1]} ({row[1]*100/total if total > 0 else 0:.1f}%)")
        print(f"   Has Gamma: {row[2]} ({row[2]*100/total if total > 0 else 0:.1f}%)")
        print(f"   Has Theta: {row[3]} ({row[3]*100/total if total > 0 else 0:.1f}%)")
        print(f"   Has Vega:  {row[4]} ({row[4]*100/total if total > 0 else 0:.1f}%)")
        print(f"   Has Rho:   {row[5]} ({row[5]*100/total if total > 0 else 0:.1f}%)\n")
        
        # 2. Check if ANY Greeks exist
        print("2. Greeks Value Ranges (where not NULL):")
        result = conn.execute(text("""
            SELECT 
                MIN(delta) as min_delta, MAX(delta) as max_delta,
                MIN(gamma) as min_gamma, MAX(gamma) as max_gamma,
                MIN(theta) as min_theta, MAX(theta) as max_theta,
                MIN(vega) as min_vega, MAX(vega) as max_vega,
                MIN(rho) as min_rho, MAX(rho) as max_rho,
                COUNT(delta) as delta_count
            FROM av_realtime_options
            WHERE delta IS NOT NULL
            AND updated_at > NOW() - INTERVAL '24 hours'
        """))
        row = result.fetchone()
        
        if row and row[10] > 0:  # delta_count
            print(f"   Delta: {row[0]:.4f} to {row[1]:.4f}")
            print(f"   Gamma: {row[2]:.6f} to {row[3]:.6f}" if row[2] else "   Gamma: No data")
            print(f"   Theta: {row[4]:.4f} to {row[5]:.4f}" if row[4] else "   Theta: No data")
            print(f"   Vega:  {row[6]:.4f} to {row[7]:.4f}" if row[6] else "   Vega: No data")
            print(f"   Rho:   {row[8]:.4f} to {row[9]:.4f}\n" if row[8] else "   Rho: No data\n")
        else:
            print("   NO GREEKS DATA FOUND!\n")
        
        # 3. Check what other fields ARE populated
        print("3. Other Fields Population (to verify data flow):")
        result = conn.execute(text("""
            SELECT 
                COUNT(last_price) as has_price,
                COUNT(bid) as has_bid,
                COUNT(ask) as has_ask,
                COUNT(volume) as has_volume,
                COUNT(open_interest) as has_oi,
                COUNT(implied_volatility) as has_iv
            FROM av_realtime_options
            WHERE updated_at > NOW() - INTERVAL '1 hour'
        """))
        row = result.fetchone()
        print(f"   Has Price: {row[0]}")
        print(f"   Has Bid:   {row[1]}")
        print(f"   Has Ask:   {row[2]}")
        print(f"   Has Volume: {row[3]}")
        print(f"   Has OI:    {row[4]}")
        print(f"   Has IV:    {row[5]}\n")
        
        # 4. Sample recent records to see what we're getting
        print("4. Sample Recent Records (check what fields are populated):")
        result = conn.execute(text("""
            SELECT 
                symbol, strike, expiration, option_type,
                last_price, bid, ask, volume, 
                implied_volatility, delta, gamma, theta
            FROM av_realtime_options
            WHERE symbol = 'SPY'
            ORDER BY updated_at DESC
            LIMIT 3
        """))
        
        for row in result:
            print(f"\n   Symbol: {row[0]}, Strike: {row[1]}, Exp: {row[2]}, Type: {row[3]}")
            print(f"   Price: {row[4]}, Bid: {row[5]}, Ask: {row[6]}, Vol: {row[7]}")
            print(f"   IV: {row[8]}, Delta: {row[9]}, Gamma: {row[10]}, Theta: {row[11]}")

if __name__ == "__main__":
    analyze_greeks_data()