#!/usr/bin/env python3
"""
AlphaTrader Pro - Main Application
Redis-centric institutional options trading system
Based on complete_tech_spec.md Section 12
"""

import asyncio
import yaml
import redis
import sys
import signal
from pathlib import Path
from datetime import datetime

class AlphaTrader:
    def __init__(self, config_file='config/config.yaml'):
        """Initialize AlphaTrader system"""
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing AlphaTrader System...")
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            decode_responses=self.config['redis']['decode_responses']
        )
        
        # Test Redis connection
        try:
            self.redis.ping()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Redis connection established")
        except redis.ConnectionError:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Redis connection failed!")
            print("Please ensure Redis is running: redis-server config/redis.conf")
            sys.exit(1)
        
        # Initialize all modules (will be implemented in subsequent days)
        self.modules = {}
        self.tasks = []
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Shutdown signal received. Closing positions and shutting down...")
        asyncio.create_task(self.shutdown())
    
    async def initialize_modules(self):
        """Initialize all trading modules - to be implemented Days 3-35"""
        
        # Module initialization will be added as each is implemented:
        # Day 3-4: IBKR Ingestion
        # self.modules['ibkr_ingestion'] = IBKRIngestion()
        
        # Day 5-6: Alpha Vantage Ingestion  
        # self.modules['av_ingestion'] = AlphaVantageIngestion(self.config['alpha_vantage']['api_key'])
        
        # Day 7: Parameter Discovery
        # self.modules['parameter_discovery'] = ParameterDiscovery()
        
        # Day 8-9: Analytics Engine
        # self.modules['analytics_engine'] = AnalyticsEngine()
        
        # Day 10-11: Signal Generator
        # self.modules['signal_generator'] = SignalGenerator()
        
        # Day 15-16: Execution Manager
        # self.modules['execution_manager'] = ExecutionManager()
        
        # Day 17-18: Position Manager
        # self.modules['position_manager'] = PositionManager()
        
        # Day 12-13: Risk Manager
        # self.modules['risk_manager'] = RiskManager()
        # self.modules['emergency_manager'] = EmergencyManager()
        
        # Day 25: Signal Distributor
        # self.modules['signal_distributor'] = SignalDistributor()
        
        # Day 23-24: Discord Bot
        # self.modules['discord_bot'] = DiscordBot()
        
        # Day 22: Dashboard
        # self.modules['dashboard'] = Dashboard()
        
        # Day 29-30: Twitter Bot
        # self.modules['twitter_bot'] = TwitterBot()
        
        # Day 31-32: Telegram Bot
        # self.modules['telegram_bot'] = TelegramBot()
        
        # Day 33: Market Analysis Generator
        # self.modules['market_analysis'] = MarketAnalysisGenerator()
        
        # Day 34: Scheduled Tasks
        # self.modules['scheduled_tasks'] = ScheduledTasks()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Module initialization complete")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Active modules: {len(self.modules)}")
    
    async def run_parameter_discovery(self):
        """Run parameter discovery first (Day 7)"""
        # Will be implemented on Day 7
        # print(f"[{datetime.now().strftime('%H:%M:%S')}] Running parameter discovery...")
        # await self.modules['parameter_discovery'].run_discovery()
        pass
    
    async def start(self):
        """Start all modules - main entry point"""
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting AlphaTrader System...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trading symbols: {', '.join(self.config['trading']['symbols'])}")
        
        # Initialize modules
        await self.initialize_modules()
        
        # Run parameter discovery first (when implemented)
        await self.run_parameter_discovery()
        
        # Start all modules (will be added as each module is implemented)
        tasks = []
        
        for name, module in self.modules.items():
            if name != 'parameter_discovery':  # Already ran
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {name}...")
                # tasks.append(asyncio.create_task(module.start()))
        
        # Store tasks for shutdown
        self.tasks = tasks
        
        if tasks:
            # Run forever
            await asyncio.gather(*tasks)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No modules to run yet. System ready for module implementation.")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Next step: Implement IBKR Ingestion (Day 3-4)")
            
            # Keep alive for testing
            while True:
                await asyncio.sleep(60)
                # Heartbeat
                self.redis.setex('system:heartbeat', 65, datetime.now().isoformat())
    
    async def shutdown(self):
        """Graceful shutdown of all modules"""
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initiating graceful shutdown...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete cancellation
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Set system halt flag in Redis
        self.redis.set('global:halt', 'true')
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Shutdown complete")
        sys.exit(0)

async def main():
    """Main entry point"""
    
    # Check if config exists
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        print("❌ Configuration file not found: config/config.yaml")
        print("Please ensure config/config.yaml exists with your API keys")
        sys.exit(1)
    
    # Create and start AlphaTrader
    trader = AlphaTrader()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.shutdown()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fatal error: {e}")
        await trader.shutdown()

if __name__ == "__main__":
    # Print banner
    print("=" * 60)
    print("           AlphaTrader Pro - Institutional Options Trading")
    print("           Redis-Centric Architecture | 16 Modules")
    print("=" * 60)
    
    # Run the system
    asyncio.run(main())