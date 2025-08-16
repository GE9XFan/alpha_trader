#!/usr/bin/env python3
"""Test script to verify Phase 0 setup is complete"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager
import psycopg2
from sqlalchemy import create_engine, text

def test_config_manager():
    """Test ConfigManager loads properly"""
    print("Testing ConfigManager...")
    config = ConfigManager()
    
    assert config.av_api_key is not None, "AV_API_KEY not loaded"
    assert config.database_url is not None, "DATABASE_URL not loaded"
    assert 'base_url' in config.av_config, "Alpha Vantage config not loaded"
    
    print(f"✓ Config loaded successfully")
    print(f"  - Database URL: {config.database_url[:30]}...")
    print(f"  - API Key loaded: {'Yes' if config.av_api_key else 'No'}")
    print(f"  - AV Config loaded: {bool(config.av_config)}")
    return config

def test_database_connection(database_url):
    """Test database connection and tables exist"""
    print("\nTesting Database Connection...")
    
    engine = create_engine(database_url)
    with engine.connect() as conn:
        # Test tables exist
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        tables = [row[0] for row in result]
        
        assert 'system_config' in tables, "system_config table not found"
        assert 'api_response_log' in tables, "api_response_log table not found"
        
        print(f"✓ Database connected successfully")
        print(f"  - Tables found: {', '.join(tables)}")

def main():
    print("=== Phase 0 Verification ===\n")
    
    try:
        # Test configuration
        config = test_config_manager()
        
        # Test database
        test_database_connection(config.database_url)
        
        print("\n✅ Phase 0 Complete! Ready for Phase 1.")
        print("\nNext steps:")
        print("  1. Implement REALTIME_OPTIONS client method")
        print("  2. Test API call with SPY")
        print("  3. Create schema based on response")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())