#!/usr/bin/env python3
"""
AlphaTrader Pro - Main Application Entry Point
Redis-centric institutional options analytics and automated trading system

This module coordinates all system components and manages the application lifecycle.
All configuration is loaded from config/config.yaml
"""

import asyncio
import signal
import sys
import os
import yaml
import redis
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import all system modules
from data_ingestion import IBKRIngestion, AlphaVantageIngestion
from analytics import ParameterDiscovery, AnalyticsEngine
from signals import SignalGenerator, SignalDistributor
from execution import ExecutionManager, PositionManager, RiskManager, EmergencyManager
from social_media import TwitterBot, TelegramBot
from dashboard import Dashboard
from morning_analysis import MarketAnalysisGenerator, ScheduledTasks


class AlphaTrader:
    """
    Main application coordinator for the AlphaTrader system.
    Manages initialization, startup, and shutdown of all modules.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the AlphaTrader system.
        Production-ready initialization with all critical components.
        """
        # Set up logging first so we can log everything
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from config/config.yaml
        self.config = self._load_config(config_path)
        
        # Initialize Redis connection immediately
        self.setup_redis()
        
        # Module instances (will be initialized in setup())
        self.modules = {}
        
    def _setup_logging(self):
        """
        Set up logging configuration for the application.
        """
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
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Replace ${VAR} placeholders with environment variables
        def replace_env_vars(obj):
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(f"Environment variable {var_name} not set")
                return value
            elif isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            return obj
        
        config = replace_env_vars(config)
        
        # Validate required configuration keys
        required_keys = ['redis', 'ibkr', 'alpha_vantage', 'symbols', 'logging']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        return config
    
    def setup_redis(self):
        """
        Initialize Redis connection with configuration from config.yaml.
        """
        redis_config = self.config['redis']
        
        # Create connection pool for better performance
        pool = redis.ConnectionPool(
            host=redis_config.get('host', '127.0.0.1'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            decode_responses=redis_config.get('decode_responses', True),
            max_connections=redis_config.get('max_connections', 50),
            socket_keepalive=redis_config.get('socket_keepalive', True)
        )
        
        # Create Redis client from pool
        self.redis = redis.Redis(connection_pool=pool)
        
        # Test connection
        try:
            if not self.redis.ping():
                raise ConnectionError("Failed to connect to Redis")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise ConnectionError(f"Cannot connect to Redis at {redis_config['host']}:{redis_config['port']}")
        
        self.logger.info(f"Redis connected successfully at {redis_config['host']}:{redis_config['port']}")
        
        # Initialize system keys
        self.redis.set('system:halt', '0')
        self.redis.set('system:health:main', str(datetime.now().timestamp()))
        self.redis.set('system:startup_time', datetime.now().isoformat())
    
    def initialize_modules(self):
        """
        Initialize all system modules with configuration.
        Day 1: Only initialize core infrastructure
        """
        self.logger.info("Initializing modules...")
        
        # Day 2-3: Data ingestion modules (prepare but don't start yet)
        if self.config.get('modules', {}).get('data_ingestion', {}).get('enabled', True):
            self.modules['ibkr_ingestion'] = IBKRIngestion(self.config, self.redis)
            self.modules['av_ingestion'] = AlphaVantageIngestion(self.config, self.redis)
            self.logger.info("Data ingestion modules initialized")
        
        # Day 4-5: Analytics modules (prepare but don't start yet)
        if self.config.get('modules', {}).get('analytics', {}).get('enabled', True):
            self.modules['param_discovery'] = ParameterDiscovery(self.config, self.redis)
            self.modules['analytics'] = AnalyticsEngine(self.config, self.redis)
            self.logger.info("Analytics modules initialized")
        
        # Day 6-10: Trading modules (prepare but don't start yet)
        if self.config.get('modules', {}).get('signals', {}).get('enabled', True):
            self.modules['signal_gen'] = SignalGenerator(self.config, self.redis)
            self.modules['signal_dist'] = SignalDistributor(self.config, self.redis)
            self.logger.info("Signal modules initialized")
        
        if self.config.get('modules', {}).get('execution', {}).get('enabled', True):
            self.modules['exec_mgr'] = ExecutionManager(self.config, self.redis)
            self.modules['pos_mgr'] = PositionManager(self.config, self.redis)
            self.modules['risk_mgr'] = RiskManager(self.config, self.redis)
            self.modules['emergency_mgr'] = EmergencyManager(self.config, self.redis)
            self.logger.info("Execution modules initialized")
        
        # Day 11-15: Social/UI modules (OPTIONAL - skip for Day 1)
        # Uncomment these when implementing Phase 3
        # if self.config.get('modules', {}).get('social', {}).get('enabled', False):
        #     self.modules['twitter'] = TwitterBot(self.config, self.redis)
        #     self.modules['telegram'] = TelegramBot(self.config, self.redis)
        #     self.logger.info("Social media modules initialized")
        
        # Day 12: Dashboard (OPTIONAL - skip for Day 1)
        # if self.config.get('modules', {}).get('dashboard', {}).get('enabled', False):
        #     self.modules['dashboard'] = Dashboard(self.config, self.redis)
        #     self.logger.info("Dashboard module initialized")
        
        # Day 16-18: Analysis modules (OPTIONAL - skip for Day 1)
        # if self.config.get('modules', {}).get('analysis', {}).get('enabled', False):
        #     self.modules['market_analysis'] = MarketAnalysisGenerator(self.config, self.redis)
        #     self.modules['scheduled'] = ScheduledTasks(self.config, self.redis)
        #     self.logger.info("Analysis modules initialized")
        
        self.logger.info(f"Initialized {len(self.modules)} modules")
    
    async def start(self):
        """
        Start all system modules asynchronously.
        Day 2: Now includes IBKR data ingestion startup.
        """
        self.logger.info("Starting AlphaTrader System...")
        
        # Day 4: Run parameter discovery first (skip for Day 2)
        # if 'param_discovery' in self.modules:
        #     self.logger.info("Running parameter discovery...")
        #     discovery = self.modules['param_discovery']
        #     await discovery.discover()
        
        # Start all modules that have a start method
        tasks = []
        
        # Day 2-3: Start data ingestion modules first if enabled
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
        
        # Start other modules
        for name, module in self.modules.items():
            if name in ['ibkr_ingestion', 'av_ingestion']:  # Already started
                continue
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
        
        # Run all tasks concurrently
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in async tasks: {e}")
            await self.shutdown()
    
    async def shutdown(self):
        """
        Gracefully shutdown all modules.
        """
        self.logger.info("Initiating graceful shutdown...")
        
        try:
            # Set global halt flag in Redis
            if self.redis:
                self.redis.set('system:halt', '1')
                self.redis.set('system:shutdown_time', datetime.now().isoformat())
            
            # Day 2-3: Stop data ingestion modules cleanly
            if 'ibkr_ingestion' in self.modules:
                self.logger.info("Stopping IBKR data ingestion...")
                try:
                    await self.modules['ibkr_ingestion'].stop()
                except Exception as e:
                    self.logger.error(f"Error stopping IBKR: {e}")
            
            if 'av_ingestion' in self.modules:
                self.logger.info("Stopping Alpha Vantage data ingestion...")
                try:
                    await self.modules['av_ingestion'].stop()
                except Exception as e:
                    self.logger.error(f"Error stopping Alpha Vantage: {e}")
            
            # Day 10: Emergency shutdown (skip for Day 2)
            # if 'emergency_mgr' in self.modules:
            #     self.logger.info("Executing emergency shutdown procedures...")
            #     await self.modules['emergency_mgr'].close_all_positions()"
            
            # Stop all running modules
            for name, module in self.modules.items():
                if hasattr(module, 'stop'):
                    self.logger.info(f"Stopping module: {name}")
                    try:
                        await module.stop()
                    except Exception as e:
                        self.logger.error(f"Error stopping {name}: {e}")
            
            # Save final state to Redis
            if self.redis:
                self.redis.set('system:last_shutdown', datetime.now().isoformat())
                
                # Close Redis connection
                self.redis.close()
                self.logger.info("Redis connection closed")
            
            self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def setup_signal_handlers(self):
        """
        Set up signal handlers for graceful shutdown.
        """
        loop = asyncio.get_event_loop()
        
        def handle_signal(sig_name):
            self.logger.info(f"Received {sig_name} signal")
            # Create shutdown task
            asyncio.create_task(self.shutdown())
            # Stop the event loop
            loop.stop()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, lambda s, f: handle_signal('SIGINT'))
        signal.signal(signal.SIGTERM, lambda s, f: handle_signal('SIGTERM'))
        
        self.logger.info("Signal handlers configured (SIGINT, SIGTERM)")
    
    async def health_check(self):
        """
        Continuous health monitoring of all modules.
        """
        self.logger.info("Health monitoring started")
        
        while True:
            try:
                # Check if halt signal is set
                if self.redis and self.redis.get('system:halt') == '1':
                    self.logger.info("Halt signal detected in health check")
                    break
                
                # Check Redis connection
                if self.redis:
                    try:
                        if not self.redis.ping():
                            self.logger.error("Redis connection lost!")
                    except redis.ConnectionError:
                        self.logger.error("Redis connection error - attempting reconnect")
                        self.setup_redis()
                    
                    # Update main health timestamp
                    self.redis.set('system:health:main', str(datetime.now().timestamp()))
                    
                    # Day 2: Monitor IBKR connection
                    if 'ibkr_ingestion' in self.modules:
                        ibkr_status = self.redis.get('ibkr:connected')
                        if ibkr_status != '1':
                            self.logger.warning("IBKR disconnected - data flow interrupted")
                            # Check if we should attempt reconnection
                            last_disconnect = self.redis.get('ibkr:disconnect_time')
                            if last_disconnect:
                                disconnect_age = datetime.now().timestamp() - datetime.fromisoformat(str(last_disconnect)).timestamp()
                                if disconnect_age > 30:  # If disconnected for > 30 seconds
                                    self.logger.error("IBKR disconnected for > 30 seconds")
                        
                        # Monitor data freshness
                        stale_symbols = self.redis.hgetall('monitoring:data:stale')
                        if stale_symbols:
                            for symbol, staleness in stale_symbols.items():
                                if float(staleness) > 30:
                                    self.logger.warning(f"Data stale for {symbol}: {staleness}s")
                    
                    # Day 3: Track Alpha Vantage API usage (skip for Day 2)
                    # if 'av_ingestion' in self.modules:
                    #     api_calls = self.redis.get('monitoring:api:av:calls')
                    #     if api_calls and int(api_calls) > 590:
                    #         self.logger.warning(f"Alpha Vantage API limit approaching: {api_calls}/600")
                    
                    # Monitor module heartbeats
                    for name in self.modules.keys():
                        heartbeat_key = f'module:heartbeat:{name}'
                        last_heartbeat = self.redis.get(heartbeat_key)
                        if last_heartbeat:
                            age = datetime.now().timestamp() - float(last_heartbeat)
                            if age > 30:  # Alert if no heartbeat for 30 seconds
                                self.logger.warning(f"Module {name} heartbeat is stale: {age:.1f}s")
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
            
            # Check every second
            await asyncio.sleep(self.config.get('monitoring', {}).get('health_check_interval', 1))
    
    def validate_environment(self):
        """
        Validate environment before starting.
        """
        self.logger.info("Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            raise RuntimeError(f"Python 3.11+ required, got {sys.version}")
        
        # Check Redis is running
        try:
            test_redis = redis.Redis(
                host=self.config['redis']['host'], 
                port=self.config['redis']['port'],
                socket_connect_timeout=5
            )
            if not test_redis.ping():
                raise ConnectionError("Redis ping failed")
            test_redis.close()
            self.logger.info("✓ Redis is running")
        except (redis.ConnectionError, ConnectionError) as e:
            raise ConnectionError(
                f"Redis not accessible at {self.config['redis']['host']}:{self.config['redis']['port']}\n"
                f"Please start Redis with: redis-server config/redis.conf"
            )
        
        # Check required directories exist
        log_dir = Path('logs')
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
            self.logger.info("Created logs directory")
        
        data_dir = Path('data/redis')
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            self.logger.info("Created data/redis directory")
        
        # Day 2: Verify IBKR Gateway/TWS
        if self.config.get('modules', {}).get('data_ingestion', {}).get('enabled', True):
            try:
                from ib_insync import IB
                ib = IB()
                # Use client ID 999 for testing to avoid conflict with main connection
                ib.connect(
                    self.config['ibkr']['host'], 
                    self.config['ibkr']['port'], 
                    clientId=999,
                    timeout=5
                )
                ib.disconnect()
                self.logger.info("✓ IBKR Gateway/TWS accessible")
            except Exception as e:
                self.logger.warning(f"IBKR not accessible (will retry when module starts): {e}")
                self.logger.warning("Please ensure IBKR Gateway/TWS is running on port 7497")
        
        # Validate API keys are present
        if 'alpha_vantage' in self.config:
            if not self.config['alpha_vantage'].get('api_key'):
                raise ValueError("Alpha Vantage API key not found in config or environment")
            self.logger.info("✓ Alpha Vantage API key configured")
        
        self.logger.info("Environment validation complete")


async def main():
    """
    Main entry point for the application.
    """
    # Set up logging first
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
        trader.validate_environment()
        
        # Set up signal handlers
        trader.setup_signal_handlers()
        
        # Initialize Redis
        trader.setup_redis()
        
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