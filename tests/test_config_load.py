#!/usr/bin/env python3
"""
Test configuration file structure and loading
Part of Day 1-2 implementation plan
"""

import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_config_structure():
    """Verify config.yaml has all required sections per spec"""
    
    print("Testing configuration structure...")
    
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML format: {e}")
        return False
    
    # Required sections from complete_tech_spec.md
    required_sections = [
        'redis',
        'ibkr', 
        'alpha_vantage',
        'trading',
        'risk',
        'strategies',
        'distribution',
        'dashboard',
        'twitter',
        'telegram',
        'openai',
        'stripe',
        'scheduled_tasks'
    ]
    
    print("\nVerifying required sections:")
    for section in required_sections:
        if section in config:
            print(f"  ✓ {section}")
        else:
            print(f"  ✗ {section} - MISSING!")
            return False
    
    # Verify trading symbols (updated list)
    expected_symbols = [
        'SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA',
        'AMD', 'GOOGL', 'META', 'AMZN', 'MSFT', 'VXX'
    ]
    
    print("\nVerifying trading symbols:")
    if 'symbols' in config['trading']:
        symbols = config['trading']['symbols']
        for symbol in expected_symbols:
            if symbol in symbols:
                print(f"  ✓ {symbol}")
            else:
                print(f"  ✗ {symbol} - MISSING!")
                return False
    else:
        print("  ✗ No symbols defined in trading config!")
        return False
    
    # Verify strategy time windows
    print("\nVerifying strategy time windows:")
    strategies = ['zero_dte', 'one_dte', 'fourteen_dte', 'moc']
    for strategy in strategies:
        if strategy in config['strategies']:
            start = config['strategies'][strategy].get('start_time')
            end = config['strategies'][strategy].get('end_time')
            print(f"  ✓ {strategy}: {start} - {end}")
        else:
            print(f"  ✗ {strategy} - MISSING!")
            return False
    
    # Verify risk parameters
    print("\nVerifying risk management parameters:")
    risk_params = {
        'kelly_fraction': 0.25,
        'max_drawdown': 0.10,
        'var_limit': 2000
    }
    
    for param, expected in risk_params.items():
        if param in config['risk']:
            value = config['risk'][param]
            print(f"  ✓ {param}: {value} (expected: {expected})")
        else:
            print(f"  ✗ {param} - MISSING!")
            return False
    
    # Verify trading limits
    print("\nVerifying trading limits:")
    trading_limits = {
        'max_positions': 5,
        'max_per_symbol': 2,
        'daily_loss_limit': 2000,
        'max_consecutive_losses': 3
    }
    
    for param, expected in trading_limits.items():
        if param in config['trading']:
            value = config['trading'][param]
            print(f"  ✓ {param}: {value} (expected: {expected})")
        else:
            print(f"  ✗ {param} - MISSING!")
            return False
    
    print("\n✅ Configuration structure valid!")
    return True

def test_redis_config():
    """Verify Redis configuration matches spec requirements"""
    
    print("\nTesting Redis configuration file...")
    
    redis_conf_path = Path(__file__).parent.parent / 'config' / 'redis.conf'
    
    if not redis_conf_path.exists():
        print(f"❌ Redis config not found: {redis_conf_path}")
        return False
    
    with open(redis_conf_path, 'r') as f:
        content = f.read()
    
    # Check required settings
    required_settings = [
        ('maxmemory 4gb', 'Memory limit'),
        ('maxmemory-policy volatile-lru', 'Eviction policy'),
        ('appendonly yes', 'AOF persistence'),
        ('appendfilename "alphatrader.aof"', 'AOF filename')
    ]
    
    print("\nVerifying Redis settings:")
    for setting, description in required_settings:
        if setting in content:
            print(f"  ✓ {description}: {setting}")
        else:
            print(f"  ✗ {description}: {setting} - MISSING!")
            return False
    
    print("\n✅ Redis configuration valid!")
    return True

if __name__ == "__main__":
    config_valid = test_config_structure()
    redis_valid = test_redis_config()
    
    if config_valid and redis_valid:
        print("\n🎉 All configuration tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some configuration tests failed!")
        sys.exit(1)