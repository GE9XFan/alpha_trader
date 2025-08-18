#!/usr/bin/env python3
"""
Test Greeks Validator with real data - Phase 6.1
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analytics.greeks_validator import GreeksValidator
from src.foundation.config_manager import ConfigManager
from sqlalchemy import create_engine, text
from datetime import datetime

def test_greeks_validator():
    print("=== Testing Greeks Validator ===\n")
    
    # Initialize validator
    validator = GreeksValidator()
    
    # Get some real data
    config = ConfigManager()
    engine = create_engine(config.database_url)
    
    with engine.connect() as conn:
        # First check what data we have
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as count,
                MAX(updated_at) as latest,
                MIN(updated_at) as earliest
            FROM av_realtime_options
            WHERE updated_at > NOW() - INTERVAL '24 hours'
        """))
        row = result.fetchone()
        print(f"Data available: {row[0]} records")
        print(f"Latest: {row[1]}")
        print(f"Earliest: {row[2]}\n")
        
        # Get recent options data (last 24 hours instead of 1 hour)
        result = conn.execute(text("""
            SELECT 
                contract_id, symbol, strike, expiration, option_type,
                delta, gamma, theta, vega, rho, updated_at
            FROM av_realtime_options
            WHERE symbol = 'SPY'
            AND updated_at > NOW() - INTERVAL '24 hours'
            ORDER BY updated_at DESC
            LIMIT 100
        """))
        
        options_data = []
        for row in result:
            options_data.append({
                'contract_id': row[0],
                'symbol': row[1],
                'strike': row[2],
                'expiration': row[3],
                'option_type': row[4],
                'delta': row[5],
                'gamma': row[6],
                'theta': row[7],
                'vega': row[8],
                'rho': row[9],
                'updated_at': row[10]
            })
    
    if not options_data:
        print("No data found! Checking all data...")
        
        # If still no data, check without time filter
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    contract_id, symbol, strike, expiration, option_type,
                    delta, gamma, theta, vega, rho, updated_at
                FROM av_realtime_options
                WHERE symbol = 'SPY'
                AND delta IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 100
            """))
            
            for row in result:
                options_data.append({
                    'contract_id': row[0],
                    'symbol': row[1],
                    'strike': row[2],
                    'expiration': row[3],
                    'option_type': row[4],
                    'delta': row[5],
                    'gamma': row[6],
                    'theta': row[7],
                    'vega': row[8],
                    'rho': row[9],
                    'updated_at': row[10]
                })
        
        if not options_data:
            print("Still no data found!")
            return
    
    # Test batch validation
    print(f"Validating {len(options_data)} options...\n")
    summary = validator.validate_batch(options_data)
    
    print("=== Validation Summary ===")
    print(f"Total Options: {summary['total']}")
    print(f"Valid: {summary['valid']} ({summary['valid_pct']:.1f}%)")
    print(f"Invalid: {summary['invalid']}")
    print(f"With Warnings: {summary['warnings']}\n")
    
    if summary['errors_by_type']:
        print("Error Types:")
        for error_type, count in summary['errors_by_type'].items():
            print(f"  {error_type}: {count}")
        print()
    
    if summary['sample_errors']:
        print("Sample Errors:")
        for sample in summary['sample_errors'][:3]:
            print(f"  Contract: {sample['contract']}")
            for error in sample['errors']:
                print(f"    - {error}")
    
    # Test individual validation - show first 3
    print("\n=== Individual Validation Examples ===")
    for i, option in enumerate(options_data[:3]):
        greeks = {
            'delta': option['delta'],
            'gamma': option['gamma'],
            'theta': option['theta'],
            'vega': option['vega'],
            'rho': option['rho']
        }
        
        result = validator.validate_greeks_set(
            greeks, 
            option['option_type'],
            option['updated_at']
        )
        
        print(f"\nExample {i+1}:")
        print(f"  Contract: {option['contract_id']}")
        print(f"  Strike: ${option['strike']}, Type: {option['option_type']}")
        print(f"  Delta: {option['delta']:.4f}, Gamma: {option['gamma']:.5f}")
        print(f"  Valid: {result['valid']}")
        if result['errors']:
            print(f"  Errors: {result['errors']}")
        if result['warnings']:
            print(f"  Warnings: {result['warnings']}")

if __name__ == "__main__":
    test_greeks_validator()