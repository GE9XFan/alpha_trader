#!/usr/bin/env python3
"""Query and display options data from both tables"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from src.foundation.config_manager import ConfigManager
import pandas as pd


def query_options_data():
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        print("=== OPTIONS DATA OVERVIEW ===\n")
        
        # 1. Summary statistics
        print("1. SUMMARY STATISTICS")
        print("-" * 50)
        
        result = conn.execute(text("""
            SELECT 
                'Realtime' as source,
                COUNT(*) as total_contracts,
                COUNT(DISTINCT symbol) as symbols,
                COUNT(DISTINCT expiration) as expirations,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike,
                MIN(expiration) as nearest_exp,
                MAX(expiration) as furthest_exp
            FROM av_realtime_options
            UNION ALL
            SELECT 
                'Historical' as source,
                COUNT(*) as total_contracts,
                COUNT(DISTINCT symbol) as symbols,
                COUNT(DISTINCT expiration) as expirations,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike,
                MIN(expiration) as nearest_exp,
                MAX(expiration) as furthest_exp
            FROM av_historical_options
        """))
        
        for row in result:
            print(f"\n{row[0]} Table:")
            print(f"  Contracts: {row[1]:,}")
            print(f"  Symbols: {row[2]}")
            print(f"  Expirations: {row[3]}")
            print(f"  Strike Range: ${row[4]} - ${row[5]}")
            print(f"  Expiry Range: {row[6]} to {row[7]}")
        
        # 2. Sample of near-the-money options
        print("\n\n2. NEAR-THE-MONEY OPTIONS (SPY ~$644)")
        print("-" * 50)
        
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
                implied_volatility
            FROM av_realtime_options
            WHERE symbol = 'SPY'
                AND strike BETWEEN 640 AND 650
                AND expiration = (SELECT MIN(expiration) FROM av_realtime_options WHERE expiration >= CURRENT_DATE)
            ORDER BY strike, option_type
            LIMIT 10
        """))
        
        df = pd.DataFrame(result.fetchall(), columns=[
            'Contract', 'Strike', 'Type', 'Exp', 'Last', 
            'Bid', 'Ask', 'Vol', 'OI', 'Delta', 'IV'
        ])
        
        if not df.empty:
            print(df.to_string(index=False))
        
        # 3. Highest volume options
        print("\n\n3. HIGHEST VOLUME OPTIONS TODAY")
        print("-" * 50)
        
        result = conn.execute(text("""
            SELECT 
                contract_id,
                strike,
                option_type,
                expiration,
                volume,
                open_interest,
                last_price,
                delta
            FROM av_realtime_options
            WHERE symbol = 'SPY'
                AND volume > 0
            ORDER BY volume DESC
            LIMIT 10
        """))
        
        df = pd.DataFrame(result.fetchall(), columns=[
            'Contract', 'Strike', 'Type', 'Exp', 
            'Volume', 'OI', 'Last', 'Delta'
        ])
        
        if not df.empty:
            print(df.to_string(index=False))
        
        # 4. Options by expiration
        print("\n\n4. CONTRACT COUNT BY EXPIRATION")
        print("-" * 50)
        
        result = conn.execute(text("""
            SELECT 
                expiration,
                COUNT(*) as contracts,
                SUM(CASE WHEN option_type = 'call' THEN 1 ELSE 0 END) as calls,
                SUM(CASE WHEN option_type = 'put' THEN 1 ELSE 0 END) as puts,
                SUM(volume) as total_volume
            FROM av_realtime_options
            WHERE symbol = 'SPY'
            GROUP BY expiration
            ORDER BY expiration
            LIMIT 10
        """))
        
        df = pd.DataFrame(result.fetchall(), columns=[
            'Expiration', 'Total', 'Calls', 'Puts', 'Volume'
        ])
        
        if not df.empty:
            print(df.to_string(index=False))
        
        # 5. High IV options
        print("\n\n5. HIGHEST IMPLIED VOLATILITY OPTIONS")
        print("-" * 50)
        
        result = conn.execute(text("""
            SELECT 
                contract_id,
                strike,
                option_type,
                expiration,
                implied_volatility,
                last_price,
                delta
            FROM av_realtime_options
            WHERE symbol = 'SPY'
                AND implied_volatility IS NOT NULL
                AND implied_volatility > 0
            ORDER BY implied_volatility DESC
            LIMIT 10
        """))
        
        df = pd.DataFrame(result.fetchall(), columns=[
            'Contract', 'Strike', 'Type', 'Exp', 'IV', 'Last', 'Delta'
        ])
        
        if not df.empty:
            print(df.to_string(index=False))
        
        # 6. Compare realtime vs historical
        print("\n\n6. REALTIME VS HISTORICAL COMPARISON")
        print("-" * 50)
        
        result = conn.execute(text("""
            WITH comparison AS (
                SELECT 
                    r.contract_id,
                    r.strike,
                    r.option_type,
                    r.last_price as realtime_price,
                    h.last_price as historical_price,
                    r.last_price - h.last_price as price_change,
                    r.volume as realtime_vol,
                    h.volume as historical_vol,
                    r.implied_volatility as realtime_iv,
                    h.implied_volatility as historical_iv
                FROM av_realtime_options r
                JOIN av_historical_options h 
                    ON r.contract_id = h.contract_id
                WHERE r.symbol = 'SPY'
                    AND r.last_price IS NOT NULL 
                    AND h.last_price IS NOT NULL
                    AND r.last_price > 10  -- Focus on contracts with meaningful prices
            )
            SELECT * FROM comparison
            ORDER BY ABS(price_change) DESC
            LIMIT 10
        """))
        
        df = pd.DataFrame(result.fetchall(), columns=[
            'Contract', 'Strike', 'Type', 'RT_Price', 'Hist_Price', 
            'Change', 'RT_Vol', 'Hist_Vol', 'RT_IV', 'Hist_IV'
        ])
        
        if not df.empty:
            print(df.to_string(index=False))


if __name__ == "__main__":
    query_options_data()