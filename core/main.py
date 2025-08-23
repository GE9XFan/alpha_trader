#!/usr/bin/env python3
"""
Main entry point for the AlphaTrader system.
Initializes and runs the entire trading platform.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bus import MessageBus
from core.persistence import EventStore
from core.plugin_manager import PluginManager
from core.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/alphatrader.log')
    ]
)

logger = logging.getLogger(__name__)


class AlphaTrader:
    """Main application class for the trading system."""
    
    def __init__(self):
        """Initialize the trading system."""
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_system_config()
        
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize components
        self.event_store = None
        self.message_bus = None
        self.plugin_manager = None
        self._running = False
        
    def _setup_database_connection(self) -> str:
        """Build database connection string from config."""
        db_config = self.config.get('database', {})
        return (
            f"postgresql://{db_config.get('user', 'alphatrader')}:"
            f"{db_config.get('password', 'alphatrader_dev')}@"
            f"{db_config.get('host', 'localhost')}:"
            f"{db_config.get('port', 5432)}/"
            f"{db_config.get('name', 'alphatrader')}"
        )
    
    async def initialize(self):
        """Initialize all system components."""
        logger.info("=" * 60)
        logger.info("AlphaTrader System Starting")
        logger.info(f"Environment: {self.config.get('system', {}).get('environment', 'development')}")
        logger.info("=" * 60)
        
        try:
            # Initialize event store
            connection_string = self._setup_database_connection()
            self.event_store = EventStore(connection_string)
            logger.info("✓ Event store initialized")
            
            # Initialize message bus
            self.message_bus = MessageBus(self.event_store)
            logger.info("✓ Message bus initialized")
            
            # Initialize plugin manager
            self.plugin_manager = PluginManager(self.message_bus, "config")
            logger.info("✓ Plugin manager initialized")
            
            # Discover and load plugins
            await self.plugin_manager.discover_and_load()
            
            # Start all plugins
            await self.plugin_manager.start_all()
            
            # Publish system started event
            self.message_bus.publish(
                event_type="system.started",
                data={
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'environment': self.config.get('system', {}).get('environment', 'development'),
                    'plugins_loaded': self.plugin_manager.list_plugins()
                }
            )
            
            logger.info("✓ System initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}", exc_info=True)
            raise
    
    async def run(self):
        """Run the trading system."""
        self._running = True
        
        logger.info("AlphaTrader system running...")
        logger.info("Press Ctrl+C to stop")
        
        # Start plugin health monitoring
        if self.plugin_manager:
            monitor_task = asyncio.create_task(
                self.plugin_manager.monitor_plugins(interval=30)
            )
        else:
            monitor_task = None
        
        try:
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(1)
                
                # Periodically log system status
                if int(datetime.now().timestamp()) % 60 == 0:
                    if self.message_bus:
                        stats = self.message_bus.get_stats()
                        logger.info(f"System stats: {stats}")
                    
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        finally:
            if monitor_task:
                monitor_task.cancel()
                await asyncio.gather(monitor_task, return_exceptions=True)
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("=" * 60)
        logger.info("AlphaTrader System Shutting Down")
        logger.info("=" * 60)
        
        self._running = False
        
        try:
            # Publish shutdown event
            if self.message_bus:
                self.message_bus.publish(
                    event_type="system.shutdown",
                    data={
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
            
            # Stop all plugins
            if self.plugin_manager:
                await self.plugin_manager.stop_all()
                logger.info("✓ All plugins stopped")
            
            # Shutdown message bus
            if self.message_bus:
                self.message_bus.shutdown()
                logger.info("✓ Message bus shutdown")
            
            # Close event store
            if self.event_store:
                self.event_store.close()
                logger.info("✓ Event store closed")
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        
        logger.info("✓ Shutdown complete")


async def main():
    """Main entry point."""
    trader = AlphaTrader()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, initiating shutdown...")
        asyncio.create_task(trader.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        await trader.initialize()
        
        # Run system
        await trader.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await trader.shutdown()


if __name__ == "__main__":
    # Run the system
    asyncio.run(main())