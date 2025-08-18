#!/usr/bin/env python3
"""
Test Greeks Validator - Focus on values, not staleness
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analytics.greeks_validator import GreeksValidator
from src.foundation.config_manager import ConfigManager
from sqlalchemy import create_engine, text
from datetime import datetime

def test_greeks_values():
    print("=== Testing Greeks Values (Ignoring Staleness) ===\n")
    
    validator = GreeksValidator()
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        # Get sample data
        result = conn.execute(text("""
            SELECT 
                contract_id, symbol, strike, expiration, option_type,
                delta, gamma, theta, vega, rho
            FROM av_realtime_options
            WHERE symbol = 'SPY'
            AND delta IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT 100
        """))
        
        valid_count = 0
        invalid_count = 0
        error_types = {}
        warning_types = {}
        
        for row in result:
            greeks = {
                'delta': row[5],
                'gamma': row[6],
                'theta': row[7],
                'vega': row[8],
                'rho': row[9]
            }
            
            # Validate WITHOUT timestamp to ignore staleness
            result = validator.validate_greeks_set(
                greeks, 
                row[4],  # option_type
                None     # No timestamp = no staleness check
            )
            
            if result['valid']:
                valid_count += 1
            else:
                invalid_count += 1
                for error in result['errors']:
                    error_type = error.split()[0]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for warning in result.get('warnings', []):
                warning_type = warning.split(':')[0]
                warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
        
        print(f"Results (ignoring staleness):")
        print(f"  Valid: {valid_count}")
        print(f"  Invalid: {invalid_count}")
        
        if error_types:
            print(f"\nError Types:")
            for error_type, count in error_types.items():
                print(f"    {error_type}: {count}")
        
        if warning_types:
            print(f"\nWarning Types:")
            for warning_type, count in warning_types.items():
                print(f"    {warning_type}: {count}")
        
        # Show some specific examples
        print("\n=== Sample Greeks Values ===")
        result = conn.execute(text("""
            SELECT 
                strike, expiration, option_type,
                delta, gamma, theta, vega, rho
            FROM av_realtime_options
            WHERE symbol = 'SPY'
            AND delta IS NOT NULL
            ORDER BY ABS(delta - 0.5)  -- Get near ATM
            LIMIT 5
        """))
        
        print("\nNear-ATM Options:")
        print("Strike  | Type | Delta   | Gamma    | Theta   | Vega   | Rho")
        print("-" * 65)
        for row in result:
            exp = row[1].strftime('%m/%d')
            print(f"{row[0]:7.2f} | {row[2]:4s} | {row[3]:7.4f} | {row[4]:8.5f} | {row[5]:7.3f} | {row[6]:6.3f} | {row[7]:6.3f}")

if __name__ == "__main__":
    test_greeks_values()