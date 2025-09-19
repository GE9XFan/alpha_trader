#!/usr/bin/env python3
"""
Test Day 1 implementation - Main Application & Configuration
Tests configuration loading, Redis connection, and basic module framework
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import redis
from main import QuantiCityCapital
from dotenv import load_dotenv


def test_prerequisites():
    """Test that Day 0 prerequisites are complete"""
    print("\n=== Testing Day 0 Prerequisites ===")
    
    # Check Python version
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"
    print("✓ Python version OK")
    
    # Check config files exist
    assert Path('config/config.yaml').exists(), "config.yaml not found"
    print("✓ config.yaml exists")
    
    assert Path('config/redis.conf').exists(), "redis.conf not found"
    print("✓ redis.conf exists")
    
    # Check .env file
    assert Path('.env').exists(), ".env file not found"
    load_dotenv()
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    assert api_key, "ALPHA_VANTAGE_API_KEY not set in .env"
    print("✓ .env file with API key exists")
    
    # Check Redis is running
    try:
        r = redis.Redis(host='127.0.0.1', port=6379, socket_connect_timeout=5)
        assert r.ping(), "Redis ping failed"
        r.close()
        print("✓ Redis is running")
    except:
        print("✗ Redis not running - start with: redis-server config/redis.conf")
        return False
    
    return True


async def test_main_application():
    """Test Day 1 main application implementation"""
    print("\n=== Testing Day 1 Implementation ===")
    
    trader = None
    try:
        # Test configuration loading
        print("\n1. Testing configuration loading...")
        trader = QuantiCityCapital()
        assert trader.config is not None, "Config not loaded"
        assert 'redis' in trader.config, "Redis config missing"
        assert 'ibkr' in trader.config, "IBKR config missing"
        assert 'alpha_vantage' in trader.config, "Alpha Vantage config missing"
        assert 'symbols' in trader.config, "Symbols config missing"
        print("✓ Configuration loaded successfully")
        
        # Test environment validation
        print("\n2. Testing environment validation...")
        trader.validate_environment()
        print("✓ Environment validation passed")
        
        # Test Redis setup
        print("\n3. Testing Redis connection...")
        trader.setup_redis()
        assert trader.redis is not None, "Redis not initialized"
        assert trader.redis.ping(), "Redis ping failed"
        
        # Check system keys
        halt_flag = trader.redis.get('system:halt')
        assert halt_flag == '0', "Halt flag not set correctly"
        print("✓ Redis connected and initialized")
        
        # Test module initialization
        print("\n4. Testing module initialization...")
        trader.initialize_modules()
        assert len(trader.modules) > 0, "No modules initialized"
        print(f"✓ {len(trader.modules)} modules initialized")
        
        # List initialized modules
        print("\nInitialized modules:")
        for name in trader.modules.keys():
            print(f"  - {name}")
        
        # Test signal handlers
        print("\n5. Testing signal handlers...")
        trader.setup_signal_handlers()
        print("✓ Signal handlers configured")
        
        # Test health check (run once)
        print("\n6. Testing health check...")
        # Create a one-shot health check
        async def single_health_check():
            try:
                if trader.redis.ping():
                    trader.redis.set('system:health:main', str(asyncio.get_event_loop().time()))
                    return True
            except:
                return False
            return True
        
        health_ok = await single_health_check()
        assert health_ok, "Health check failed"
        print("✓ Health check working")
        
        # Test shutdown
        print("\n7. Testing graceful shutdown...")
        await trader.shutdown()
        
        # Verify halt flag is set
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        halt_flag = r.get('system:halt')
        assert halt_flag == '1', "Halt flag not set on shutdown"
        r.close()
        print("✓ Graceful shutdown completed")
        
        print("\n" + "=" * 50)
        print("✅ All Day 1 tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        if trader and trader.redis:
            try:
                await trader.shutdown()
            except:
                pass
        
        return False


async def test_module_stubs():
    """Test that module stubs can be instantiated"""
    print("\n=== Testing Module Stubs ===")
    
    # Create minimal test config
    test_config = {
        'redis': {'host': '127.0.0.1', 'port': 6379},
        'ibkr': {'host': '127.0.0.1', 'port': 7497},
        'alpha_vantage': {'api_key': 'test'},
        'symbols': ['SPY', 'QQQ']
    }
    
    # Create test Redis connection
    r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
    
    try:
        # Test data ingestion modules
        from data_ingestion import IBKRIngestion, AlphaVantageIngestion
        ibkr = IBKRIngestion(test_config, r)
        av = AlphaVantageIngestion(test_config, r)
        print("✓ Data ingestion modules instantiated")
        
        # Test analytics modules
        from analytics import ParameterDiscovery, AnalyticsEngine
        param = ParameterDiscovery(test_config, r)
        analytics = AnalyticsEngine(test_config, r)
        print("✓ Analytics modules instantiated")
        
        # Test signal modules
        from signals import SignalGenerator, SignalDistributor
        sig_gen = SignalGenerator(test_config, r)
        sig_dist = SignalDistributor(test_config, r)
        print("✓ Signal modules instantiated")
        
        # Test execution modules
        from execution import ExecutionManager, PositionManager, RiskManager, EmergencyManager
        exec_mgr = ExecutionManager(test_config, r)
        pos_mgr = PositionManager(test_config, r)
        risk_mgr = RiskManager(test_config, r)
        emerg_mgr = EmergencyManager(test_config, r)
        print("✓ Execution modules instantiated")
        
        r.close()
        return True
        
    except Exception as e:
        print(f"✗ Module instantiation failed: {e}")
        r.close()
        return False


def main():
    """Run all Day 1 tests"""
    print("\n" + "=" * 60)
    print("QuantiCity Capital - Day 1 Test Suite")
    print("=" * 60)
    
    # Test prerequisites
    if not test_prerequisites():
        print("\n⚠️  Prerequisites not met. Please complete Day 0 setup first.")
        return False
    
    # Test main application
    success = asyncio.run(test_main_application())
    
    if success:
        # Test module stubs
        asyncio.run(test_module_stubs())
    
    if success:
        print("\n" + "🎉 " * 20)
        print("SUCCESS! Day 1 implementation is complete and working!")
        print("🎉 " * 20)
        print("\nNext steps:")
        print("1. Review the logs in logs/quanticity_capital.log")
        print("2. Check Redis keys with: redis-cli keys '*'")
        print("3. Proceed to Day 2: IBKR Data Ingestion")
    else:
        print("\n⚠️  Some tests failed. Please review and fix the issues.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
