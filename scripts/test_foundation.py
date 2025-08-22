#!/usr/bin/env python3
"""Test complete Phase 0 foundation"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import get_config_manager
from src.foundation.logger import get_logger, get_logger_system
from src.database.db_manager import get_db_manager
from src.data.cache_manager import get_cache_manager


def test_foundation():
    """Test all foundation components"""
    
    # Initialize logger system first
    logger_system = get_logger_system()
    logger = get_logger('FoundationTest')
    
    logger.info("Starting Phase 0 Foundation Test")
    
    # Test configuration
    logger.info("Testing Configuration Manager...")
    config = get_config_manager()
    assert config.av_api_key, "Alpha Vantage API key not set"
    logger.info("✓ Configuration loaded")
    
    # Test database
    logger.info("Testing Database Connection...")
    db = get_db_manager()
    result = db.execute_query("SELECT 1 as test")
    assert result[0]['test'] == 1
    logger.info("✓ Database connected")
    
    # Test cache
    logger.info("Testing Cache Connection...")
    cache = get_cache_manager()
    cache.set('foundation:test', {'phase': 0}, ttl=10)
    data = cache.get('foundation:test')
    assert data['phase'] == 0
    cache.delete('foundation:test')
    logger.info("✓ Cache connected")
    
    # Log some metrics
    logger_system.log_metric('foundation.test', 1.0, {'status': 'complete'})
    
    # Test error logging
    try:
        raise ValueError("Test error handling")
    except ValueError as e:
        logger_system.log_error('FoundationTest', e, {'test': True})
        logger.info("✓ Error logging working")
    
    logger.info("="*50)
    logger.info("✅ PHASE 0 FOUNDATION COMPLETE!")
    logger.info("="*50)
    logger.info("")
    logger.info("Ready to proceed to Phase 1:")
    logger.info("- Test ALL 41 Alpha Vantage APIs")
    logger.info("- Create complete database schema")
    logger.info("- Implement batch ingestion pipeline")
    logger.info("")
    
    return True


if __name__ == "__main__":
    try:
        success = test_foundation()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Foundation test failed: {e}")
        exit(1)