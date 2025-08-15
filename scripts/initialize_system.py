#!/usr/bin/env python3
"""
System Initialization Script
Initializes all modules in correct dependency order
"""

import sys
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import signal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.foundation.config_manager import ConfigManager
from src.foundation.base_module import BaseModule, ComponentStatus, HealthStatus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemInitializer:
    """
    Manages system initialization and module lifecycle
    """
    
    # Define initialization order (dependencies first)
    INITIALIZATION_ORDER = [
        # Foundation layer (no dependencies)
        'ConfigManager',
        'DatabaseConnection',
        'RedisConnection',
        
        # Connection layer (depends on config)
        'RateLimiter',
        'AlphaVantageClient',
        'IBKRConnection',
        
        # Data layer (depends on connections)
        'CacheManager',
        'SchemaBuilder',
        'DataIngestionPipeline',
        'DataScheduler',
        
        # Analytics layer (depends on data)
        'IndicatorProcessor',
        'GreeksValidator',
        'AnalyticsEngine',
        
        # ML layer (depends on analytics)
        'FeatureBuilder',
        'ModelSuite',
        
        # Decision layer (depends on ML and analytics)
        'StrategyEngine',
        'DecisionEngine',
        
        # Risk layer (depends on decision)
        'RiskManager',
        'PositionSizer',
        
        # Execution layer (depends on risk)
        'IBKRExecutor',
        
        # Monitoring layer (depends on execution)
        'TradeMonitor',
        'DiscordPublisher',
        
        # API layer (read-only, depends on all)
        'DashboardAPI'
    ]
    
    # Module dependencies map
    DEPENDENCIES = {
        'ConfigManager': [],
        'DatabaseConnection': ['ConfigManager'],
        'RedisConnection': ['ConfigManager'],
        'RateLimiter': ['ConfigManager'],
        'AlphaVantageClient': ['ConfigManager', 'RateLimiter'],
        'IBKRConnection': ['ConfigManager'],
        'CacheManager': ['ConfigManager', 'RedisConnection'],
        'SchemaBuilder': ['ConfigManager', 'DatabaseConnection'],
        'DataIngestionPipeline': ['DatabaseConnection', 'CacheManager'],
        'DataScheduler': ['ConfigManager', 'AlphaVantageClient', 'IBKRConnection'],
        'IndicatorProcessor': ['DataIngestionPipeline'],
        'GreeksValidator': ['DataIngestionPipeline'],
        'AnalyticsEngine': ['IndicatorProcessor', 'GreeksValidator'],
        'FeatureBuilder': ['AnalyticsEngine'],
        'ModelSuite': ['ConfigManager', 'FeatureBuilder'],
        'StrategyEngine': ['ConfigManager'],
        'DecisionEngine': ['ModelSuite', 'AnalyticsEngine', 'StrategyEngine'],
        'RiskManager': ['ConfigManager', 'DecisionEngine'],
        'PositionSizer': ['ConfigManager', 'RiskManager'],
        'IBKRExecutor': ['IBKRConnection', 'RiskManager'],
        'TradeMonitor': ['IBKRExecutor'],
        'DiscordPublisher': ['ConfigManager', 'TradeMonitor'],
        'DashboardAPI': ['ConfigManager', 'TradeMonitor']
    }
    
    def __init__(self, environment: str = None):
        """
        Initialize system initializer
        
        Args:
            environment: Environment to run in (development/paper/production)
        """
        self.environment = environment
        self.modules: Dict[str, Any] = {}
        self.config: Optional[ConfigManager] = None
        self.is_running = False
        self.shutdown_requested = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize_config(self) -> bool:
        """Initialize configuration manager"""
        try:
            logger.info("Initializing ConfigManager...")
            self.config = ConfigManager(self.environment)
            self.modules['ConfigManager'] = self.config
            
            # Print configuration summary
            self.config.print_summary()
            
            # Validate critical configurations
            if not self._validate_config():
                logger.error("Configuration validation failed")
                return False
            
            logger.info("✓ ConfigManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ConfigManager: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate critical configuration settings"""
        
        # Check if we're in production without proper setup
        if self.config.is_production():
            logger.warning("=" * 60)
            logger.warning("WARNING: PRODUCTION MODE DETECTED")
            logger.warning("This will trade with REAL MONEY")
            logger.warning("=" * 60)
            
            # Additional safety checks for production
            trading_mode = self.config.get_trading_mode()
            if trading_mode == 'live':
                response = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
                if response != 'CONFIRM LIVE TRADING':
                    logger.error("Production mode not confirmed. Exiting.")
                    return False
        
        # Check API keys are set
        av_key = self.config.get('env.AV_API_KEY')
        if not av_key or av_key == 'your_alpha_vantage_key_here':
            logger.warning("Alpha Vantage API key not set")
            # Don't fail in development
            if not self.config.is_development():
                return False
        
        ibkr_user = self.config.get('env.IBKR_USERNAME')
        if not ibkr_user or ibkr_user == 'your_ibkr_username':
            logger.warning("IBKR credentials not set")
            # Don't fail in development
            if not self.config.is_development():
                return False
        
        return True
    
    async def initialize_module(self, module_name: str) -> bool:
        """
        Initialize a single module (placeholder for now)
        
        Args:
            module_name: Name of module to initialize
            
        Returns:
            True if successful
        """
        # Check dependencies are initialized
        dependencies = self.DEPENDENCIES.get(module_name, [])
        for dep in dependencies:
            if dep not in self.modules:
                logger.error(f"Cannot initialize {module_name}: dependency {dep} not initialized")
                return False
        
        logger.info(f"Initializing {module_name}...")
        
        # Module implementations will be added in subsequent phases
        # For now, we just mark them as "initialized" for testing
        
        if module_name == 'DatabaseConnection':
            # Test database connection
            try:
                import psycopg2
                db_config = self.config.get_database_config()
                conn = psycopg2.connect(**db_config)
                conn.close()
                self.modules[module_name] = {'status': 'connected'}
                logger.info(f"✓ {module_name} initialized")
                return True
            except Exception as e:
                logger.error(f"✗ {module_name} failed: {e}")
                return False
        
        elif module_name == 'RedisConnection':
            # Test Redis connection
            try:
                import redis
                redis_config = self.config.get_redis_config()
                r = redis.Redis(**redis_config)
                r.ping()
                self.modules[module_name] = {'status': 'connected'}
                logger.info(f"✓ {module_name} initialized")
                return True
            except Exception as e:
                logger.error(f"✗ {module_name} failed: {e}")
                return False
        
        else:
            # Placeholder for other modules (will be implemented in later phases)
            await asyncio.sleep(0.01)  # Simulate initialization
            self.modules[module_name] = {'status': 'placeholder'}
            logger.info(f"✓ {module_name} marked for implementation")
            return True
    
    async def initialize_all_modules(self) -> bool:
        """Initialize all modules in dependency order"""
        logger.info("=" * 60)
        logger.info("SYSTEM INITIALIZATION STARTING")
        logger.info("=" * 60)
        
        # First, initialize config
        if not await self.initialize_config():
            return False
        
        # Initialize remaining modules
        failed_modules = []
        
        for module_name in self.INITIALIZATION_ORDER[1:]:  # Skip ConfigManager
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping initialization")
                return False
            
            if not await self.initialize_module(module_name):
                failed_modules.append(module_name)
                # In development, continue even if some modules fail
                if not self.config.is_development():
                    logger.error(f"Critical module {module_name} failed. Stopping initialization.")
                    return False
        
        # Summary
        logger.info("=" * 60)
        logger.info("INITIALIZATION SUMMARY")
        logger.info("=" * 60)
        
        successful = len(self.modules) - len(failed_modules)
        total = len(self.INITIALIZATION_ORDER)
        
        logger.info(f"Modules initialized: {successful}/{total}")
        
        if failed_modules:
            logger.warning(f"Failed modules: {', '.join(failed_modules)}")
        
        if len(failed_modules) == 0:
            logger.info("✅ ALL MODULES INITIALIZED SUCCESSFULLY")
            return True
        elif self.config.is_development():
            logger.warning("⚠️ Some modules failed (development mode - continuing)")
            return True
        else:
            logger.error("❌ Critical modules failed")
            return False
    
    async def perform_health_checks(self) -> Dict[str, Any]:
        """Perform health checks on all modules"""
        logger.info("Performing system health checks...")
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'modules': {}
        }
        
        # Check each module
        for module_name, module in self.modules.items():
            if isinstance(module, dict):
                # Placeholder modules
                health_status['modules'][module_name] = {
                    'status': module.get('status', 'unknown')
                }
            elif hasattr(module, 'health_check'):
                # Real modules with health check
                try:
                    health = await module.health_check() if asyncio.iscoroutinefunction(module.health_check) else module.health_check()
                    health_status['modules'][module_name] = health
                except Exception as e:
                    health_status['modules'][module_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                health_status['modules'][module_name] = {
                    'status': 'no_health_check'
                }
        
        # Overall system health
        all_healthy = all(
            m.get('status') in ['connected', 'healthy', 'placeholder', 'no_health_check']
            for m in health_status['modules'].values()
        )
        
        health_status['system_healthy'] = all_healthy
        
        return health_status
    
    async def shutdown_all_modules(self):
        """Shutdown all modules in reverse order"""
        logger.info("=" * 60)
        logger.info("SYSTEM SHUTDOWN INITIATED")
        logger.info("=" * 60)
        
        # Shutdown in reverse order
        for module_name in reversed(self.INITIALIZATION_ORDER):
            if module_name in self.modules:
                module = self.modules[module_name]
                
                if hasattr(module, 'shutdown'):
                    try:
                        logger.info(f"Shutting down {module_name}...")
                        if asyncio.iscoroutinefunction(module.shutdown):
                            await module.shutdown()
                        else:
                            module.shutdown()
                        logger.info(f"✓ {module_name} shutdown complete")
                    except Exception as e:
                        logger.error(f"Error shutting down {module_name}: {e}")
                else:
                    logger.debug(f"Skipping {module_name} (no shutdown method)")
        
        logger.info("✓ All modules shutdown complete")
    
    async def run(self):
        """Main system run loop"""
        # Initialize all modules
        if not await self.initialize_all_modules():
            logger.error("System initialization failed")
            await self.shutdown_all_modules()
            return False
        
        # Perform initial health check
        health = await self.perform_health_checks()
        if health['system_healthy']:
            logger.info("✅ System health check passed")
        else:
            logger.warning("⚠️ System health check has warnings")
        
        # Main run loop (placeholder for now)
        self.is_running = True
        logger.info("=" * 60)
        logger.info("🚀 SYSTEM READY")
        logger.info(f"Environment: {self.config.environment}")
        logger.info(f"Trading Mode: {self.config.get_trading_mode()}")
        logger.info("=" * 60)
        
        if self.config.is_development():
            logger.info("Development mode - modules are placeholders")
            logger.info("Actual implementations will be added in Phase 0.5+")
        
        # Keep running until shutdown requested
        while self.is_running and not self.shutdown_requested:
            await asyncio.sleep(1)
            
            # In production, this would run the main trading loop
            # For now, just wait for shutdown signal
        
        # Shutdown
        logger.info("Shutdown requested, cleaning up...")
        await self.shutdown_all_modules()
        
        logger.info("=" * 60)
        logger.info("System shutdown complete")
        logger.info("=" * 60)
        
        return True


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading System Initialization')
    parser.add_argument(
        '--env',
        choices=['development', 'paper', 'production'],
        default='development',
        help='Environment to run in'
    )
    parser.add_argument(
        '--health-check-only',
        action='store_true',
        help='Only perform health check and exit'
    )
    
    args = parser.parse_args()
    
    # Create and run initializer
    initializer = SystemInitializer(environment=args.env)
    
    if args.health_check_only:
        # Just do health check
        if await initializer.initialize_config():
            health = await initializer.perform_health_checks()
            
            print("\n" + "=" * 60)
            print("HEALTH CHECK RESULTS")
            print("=" * 60)
            
            for module, status in health['modules'].items():
                status_text = status.get('status', 'unknown')
                symbol = "✓" if status_text in ['connected', 'healthy', 'placeholder'] else "✗"
                print(f"{symbol} {module:25} {status_text}")
            
            print("=" * 60)
            if health['system_healthy']:
                print("✅ SYSTEM HEALTH: GOOD")
            else:
                print("❌ SYSTEM HEALTH: ISSUES DETECTED")
    else:
        # Run full system
        success = await initializer.run()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)