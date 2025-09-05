#!/usr/bin/env python3
"""
AlphaTrader Pro - Main Application Entry Point
Day 4 Implementation with async Redis and graceful module initialization
"""

import asyncio
import signal
import sys
import os
import yaml
import redis.asyncio as aioredis
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class AlphaTrader:
    """
    Main application coordinator for the AlphaTrader system.
    Manages initialization, startup, and shutdown of all modules.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the AlphaTrader system."""
        # Set up logging first so we can log everything
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from config/config.yaml
        self.config = self._load_config(config_path)
        
        # Redis connection will be initialized in async context
        self.redis = None
        
        # Module instances (will be initialized in setup())
        self.modules = {}
        
        # Shutdown flag
        self.shutdown_requested = False
        
    def _setup_logging(self):
        """Set up logging configuration for the application."""
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging with both file and console handlers
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'logs/alphatrader.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Replace ${VAR} placeholders with environment variables
        def replace_env_vars(obj: Any) -> Any:
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                value = os.getenv(var_name)
                if value is None:
                    self.logger.warning(f"Environment variable {var_name} not set")
                    return None
                return value
            elif isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            return obj
        
        config = replace_env_vars(config)
        
        # Ensure config is a dict before validation and return
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        return config
    
    async def setup_redis(self) -> None:
        """Initialize async Redis connection - no sync fallback."""
        redis_config = self.config['redis']
        
        # Create async Redis client
        self.redis = aioredis.Redis(
            host=redis_config.get('host', '127.0.0.1'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            decode_responses=redis_config.get('decode_responses', True),
            max_connections=redis_config.get('max_connections', 50),
            socket_keepalive=redis_config.get('socket_keepalive', True)
        )
        
        # Test connection
        if not await self.redis.ping():
            raise ConnectionError("Failed to connect to Redis")
        
        # Initialize system keys
        await self.redis.set('system:halt', '0')
        await self.redis.set('system:health:main', str(datetime.now().timestamp()))
        await self.redis.set('system:startup_time', datetime.now().isoformat())
        
        self.logger.info(f"Async Redis connected at {redis_config['host']}:{redis_config['port']}")
    
    def initialize_modules(self):
        """Initialize modules with graceful fallback for missing components."""
        self.logger.info("Initializing modules...")
        
        # Ensure Redis is initialized (for type checker)
        assert self.redis is not None, "Redis connection must be initialized before modules"
        
        # REQUIRED: Data ingestion modules
        if self.config.get('modules', {}).get('data_ingestion', {}).get('enabled', True):
            try:
                from src.data_ingestion import IBKRIngestion, AlphaVantageIngestion
                self.modules['ibkr_ingestion'] = IBKRIngestion(self.config, self.redis)
                self.modules['av_ingestion'] = AlphaVantageIngestion(self.config, self.redis)
                self.logger.info("✓ Data ingestion modules initialized")
            except Exception as e:
                self.logger.error(f"FATAL: Cannot initialize data ingestion: {e}")
                raise
        
        # REQUIRED: Analytics (for Day 4)
        if self.config.get('modules', {}).get('analytics', {}).get('enabled', True):
            try:
                from src.analytics import AnalyticsEngine, ParameterDiscovery
                self.modules['analytics'] = AnalyticsEngine(self.config, self.redis)
                self.modules['param_discovery'] = ParameterDiscovery(self.config, self.redis)
                self.logger.info("✓ Analytics modules initialized")
            except ImportError as e:
                self.logger.warning(f"Analytics module missing (required for Day 4): {e}")
            except Exception as e:
                self.logger.error(f"Error initializing analytics: {e}")
        
        # Add system monitor
        try:
            from src.monitoring import SystemMonitor
            self.modules['monitor'] = SystemMonitor(self.config, self.redis)
            self.logger.info("✓ System monitor initialized")
        except ImportError:
            self.logger.warning("System monitor not available")
        
        # OPTIONAL: Future modules (Day 5+)
        optional_modules = [
            ('signals', ['SignalGenerator', 'SignalDistributor']),
            ('execution', ['ExecutionManager', 'PositionManager', 'RiskManager']),
            ('social_media', ['TwitterBot', 'TelegramBot']),
            ('dashboard', ['Dashboard'])
        ]
        
        for module_name, class_names in optional_modules:
            if not self.config.get('modules', {}).get(module_name, {}).get('enabled', False):
                continue
                
            try:
                module = __import__(f'src.{module_name}', fromlist=class_names)
                for class_name in class_names:
                    if hasattr(module, class_name):
                        cls = getattr(module, class_name)
                        instance_name = f"{module_name}_{class_name.lower()}"
                        self.modules[instance_name] = cls(self.config, self.redis)
                self.logger.info(f"✓ {module_name} module initialized")
            except ImportError:
                self.logger.info(f"⚠ {module_name} module not available (expected for Day 4)")
            except Exception as e:
                self.logger.warning(f"⚠ Error loading {module_name}: {e}")
        
        self.logger.info(f"Module initialization complete: {len(self.modules)} modules loaded")
    
    async def start(self):
        """Start all system modules asynchronously."""
        self.logger.info("Starting AlphaTrader System...")
        
        # Schedule parameter discovery after data collection if enabled
        pd_config = self.config.get('parameter_discovery', {})
        if pd_config.get('enabled') and pd_config.get('run_on_startup'):
            if 'param_discovery' in self.modules:
                initial_delay = pd_config.get('startup_delay', 60)
                self.logger.info(f"Scheduling parameter discovery to run in {initial_delay}s")
                asyncio.create_task(self._delayed_parameter_discovery(initial_delay))
        
        # Start all modules that have a start method
        tasks = []
        
        # Start data ingestion modules first
        if self.config.get('modules', {}).get('data_ingestion', {}).get('enabled', True):
            # Start IBKR ingestion
            if 'ibkr_ingestion' in self.modules:
                self.logger.info("Starting IBKR data ingestion module...")
                ibkr_module = self.modules['ibkr_ingestion']
                task = asyncio.create_task(ibkr_module.start())
                task.set_name("module_ibkr_ingestion")
                tasks.append(task)
                
                # Give IBKR time to establish connections
                await asyncio.sleep(2)
            
            # Start Alpha Vantage ingestion
            if 'av_ingestion' in self.modules:
                self.logger.info("Starting Alpha Vantage data ingestion module...")
                av_module = self.modules['av_ingestion']
                task = asyncio.create_task(av_module.start())
                task.set_name("module_av_ingestion")
                tasks.append(task)
        
        # Start analytics
        if 'analytics' in self.modules:
            self.logger.info("Starting Analytics Engine...")
            task = asyncio.create_task(self.modules['analytics'].start())
            task.set_name("module_analytics")
            tasks.append(task)
        
        # Start system monitor
        if 'monitor' in self.modules:
            self.logger.info("Starting System Monitor...")
            task = asyncio.create_task(self.modules['monitor'].start())
            task.set_name("module_monitor")
            tasks.append(task)
        
        # Start other modules
        for name, module in self.modules.items():
            if name in ['ibkr_ingestion', 'av_ingestion', 'analytics', 'monitor']:
                continue  # Already started
            if hasattr(module, 'start'):
                self.logger.info(f"Starting module: {name}")
                task = asyncio.create_task(module.start())
                task.set_name(f"module_{name}")
                tasks.append(task)
        
        # Start health monitoring
        health_task = asyncio.create_task(self.health_check())
        health_task.set_name("health_monitor")
        tasks.append(health_task)
        
        self.logger.info(f"Started {len(tasks)} async tasks")
        
        # Run all tasks concurrently until shutdown requested
        try:
            # Store tasks for cancellation if needed
            self.running_tasks = tasks
            
            # Wait for shutdown request
            while not self.shutdown_requested:
                await asyncio.sleep(1)
            
            # Shutdown was requested - cancel all tasks
            self.logger.info("Cancelling all running tasks...")
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete with a timeout
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error in async tasks: {e}")
            if not self.shutdown_requested:
                await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown all modules."""
        # Prevent multiple shutdown calls
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        
        self.logger.info("Initiating graceful shutdown...")
        
        try:
            # Set global halt flag in Redis
            if self.redis:
                await self.redis.set('system:halt', '1')
                await self.redis.set('system:shutdown_time', datetime.now().isoformat())
            
            # Stop modules in reverse order
            shutdown_order = ['monitor', 'analytics', 'av_ingestion', 'ibkr_ingestion']
            
            for module_name in shutdown_order:
                if module_name in self.modules:
                    self.logger.info(f"Stopping {module_name}...")
                    try:
                        module = self.modules[module_name]
                        if hasattr(module, 'stop'):
                            await module.stop()
                    except Exception as e:
                        self.logger.error(f"Error stopping {module_name}: {e}")
            
            # Stop any remaining modules
            for name, module in self.modules.items():
                if name not in shutdown_order and hasattr(module, 'stop'):
                    self.logger.info(f"Stopping module: {name}")
                    try:
                        await module.stop()
                    except Exception as e:
                        self.logger.error(f"Error stopping {name}: {e}")
            
            # Save final state to Redis
            if self.redis:
                await self.redis.set('system:last_shutdown', datetime.now().isoformat())
                
                # Close Redis connection
                await self.redis.aclose()
                self.logger.info("Redis connection closed")
            
            self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(sig_name):
            self.logger.info(f"Received {sig_name} signal")
            # Set a flag to trigger shutdown
            if not self.shutdown_requested:
                self.shutdown_requested = True
                # Create shutdown task
                asyncio.create_task(self.shutdown())
        
        # Register signal handlers
        signal.signal(signal.SIGINT, lambda s, f: handle_signal('SIGINT'))
        signal.signal(signal.SIGTERM, lambda s, f: handle_signal('SIGTERM'))
        
        self.logger.info("Signal handlers configured (SIGINT, SIGTERM)")
    
    async def _delayed_parameter_discovery(self, delay_seconds: int):
        """Run parameter discovery after a delay to allow data collection."""
        await asyncio.sleep(delay_seconds)
        
        try:
            if 'param_discovery' in self.modules:
                self.logger.info("Running delayed parameter discovery...")
                param_discovery = self.modules['param_discovery']
                await param_discovery.run_discovery()
                self.logger.info("Parameter discovery completed")
                
                # Schedule periodic runs if configured
                pd_config = self.config.get('parameter_discovery', {})
                interval = pd_config.get('interval_seconds', 3600)
                if interval > 0:
                    self.logger.info(f"Scheduling next parameter discovery in {interval}s")
                    asyncio.create_task(self._delayed_parameter_discovery(interval))
        except Exception as e:
            self.logger.error(f"Parameter discovery failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    async def health_check(self):
        """Continuous health monitoring of all modules."""
        self.logger.info("Health monitoring started")
        
        while True:
            try:
                # Check if halt signal is set
                if self.redis:
                    halt = await self.redis.get('system:halt')
                    if halt == '1':
                        self.logger.info("Halt signal detected in health check")
                        break
                
                # Check Redis connection
                if self.redis:
                    try:
                        if not await self.redis.ping():
                            self.logger.error("Redis connection lost!")
                    except Exception:
                        self.logger.error("Redis connection error - attempting reconnect")
                        await self.setup_redis()
                    
                    # Update main health timestamp
                    await self.redis.set('system:health:main', str(datetime.now().timestamp()))
                    
                    # Monitor IBKR connection
                    if 'ibkr_ingestion' in self.modules:
                        ibkr_status = await self.redis.get('ibkr:connected')
                        if ibkr_status != '1':
                            self.logger.warning("IBKR disconnected - data flow interrupted")
                    
                    # Monitor data freshness
                    stale_symbols = await self.redis.hgetall('monitoring:data:stale') # type: ignore
                    if stale_symbols:
                        for symbol, staleness in stale_symbols.items():
                            if float(staleness) > 30:
                                self.logger.warning(f"Data stale for {symbol}: {staleness}s")
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
            
            # Check every second
            await asyncio.sleep(self.config.get('monitoring', {}).get('health_check_interval', 1))
    
    async def validate_environment(self):
        """Validate environment before starting."""
        self.logger.info("Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            raise RuntimeError(f"Python 3.10+ required, got {sys.version}")
        
        # Check Redis is running (async)
        try:
            test_redis = aioredis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                socket_connect_timeout=5
            )
            if not await test_redis.ping():
                raise ConnectionError("Redis ping failed")
            await test_redis.aclose()
            self.logger.info("✓ Redis is running")
        except Exception as e:
            raise ConnectionError(
                f"Redis not accessible at {self.config['redis']['host']}:{self.config['redis']['port']}\n"
                f"Please start Redis with: redis-server"
            )
        
        # Check required directories exist
        log_dir = Path('logs')
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
            self.logger.info("Created logs directory")
        
        # Validate API keys are present
        if 'alpha_vantage' in self.config:
            if not self.config['alpha_vantage'].get('api_key'):
                self.logger.warning("Alpha Vantage API key not configured")
            else:
                self.logger.info("✓ Alpha Vantage API key configured")
        
        self.logger.info("Environment validation complete")


async def main():
    """Main entry point for the application."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("AlphaTrader Pro Starting...")
    logger.info("=" * 60)
    
    # Initialize the system
    trader = None
    
    try:
        # Initialize AlphaTrader
        trader = AlphaTrader()
        
        # Validate environment
        await trader.validate_environment()
        
        # Set up signal handlers
        trader.setup_signal_handlers()
        
        # Initialize Redis (async)
        await trader.setup_redis()
        
        # Initialize all modules
        trader.initialize_modules()
        
        # Start the system
        await trader.start()
        
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt...")
        if trader:
            await trader.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if trader:
            await trader.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    # Set up asyncio event loop
    asyncio.run(main())