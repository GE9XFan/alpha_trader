#!/usr/bin/env python3
"""Test ConfigManager functionality"""

from src.foundation.config_manager import ConfigManager

# Initialize ConfigManager
config = ConfigManager()

# Test 1: Get database config
db_config = config.get_database_config()
print(f"✓ Database: {db_config['database']} @ {db_config['host']}")

# Test 2: Get strategy parameters
min_confidence = config.get('strategies.0dte.confidence.minimum')
print(f"✓ 0DTE Min Confidence: {min_confidence}")

# Test 3: Get rate limits
rate_limit = config.get('apis.rate_limits.rate_limits.alpha_vantage.calls_per_minute')
print(f"✓ Alpha Vantage Rate Limit: {rate_limit} calls/min")

# Test 4: Get risk limits
max_delta = config.get('risk.position_limits.position_limits.max_delta')
print(f"✓ Max Delta: {max_delta}")

# Test 5: Check environment
print(f"✓ Environment: {config.environment}")
print(f"✓ Is Development: {config.is_development()}")
print(f"✓ Trading Mode: {config.get_trading_mode()}")

# Test 6: Get all Tier A symbols
tier_a = config.get('data.symbols.symbols.tier_a.symbols')
print(f"✓ Tier A Symbols: {tier_a}")

print("\n✅ All ConfigManager tests passed!")