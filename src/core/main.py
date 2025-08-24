"""
Main entry point for AlphaTrader system.

This module orchestrates the startup and shutdown of all system components.
"""

import sys
import signal
import asyncio
import argparse
from pathlib import Path
from src.core.config import ConfigManager, TradingConfig, TradingMode
from src.core.logging import TradingLogger, system_logger
from src.core.exceptions import ConfigurationError


class AlphaTrader:
    """
    Main application class for AlphaTrader system.
    
    Coordinates all components and manages the application lifecycle.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize AlphaTrader application.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = system_logger
        self.running = False
        self.components = {}
        
        # Initialize logging
        self.trading_logger = TradingLogger(config)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    async def initialize(self):
        """Initialize all system components."""
        self.logger.info("Initializing AlphaTrader system...")
        self.logger.info(f"Mode: {self.config.mode.value}")
        self.logger.info(f"Environment: {self.config.environment}")
        
        # Log configuration summary
        self._log_configuration()
        
        # Component initialization will be added in subsequent days
        # Day 2: IBKR connector
        # Day 3: Alpha Vantage client
        # Day 4: Data orchestrator
        # etc.
        
        self.logger.info("System initialization complete")
    
    async def start(self):
        """Start the trading system."""
        try:
            await self.initialize()
            
            self.running = True
            self.logger.info("=" * 60)
            self.logger.info("AlphaTrader system started successfully")
            self.logger.info(f"Trading symbols: {', '.join(self.config.symbols)}")
            self.logger.info(f"Max positions: {self.config.risk_limits.max_positions}")
            self.logger.info(f"Daily loss limit: ${self.config.risk_limits.daily_loss_limit:,.2f}")
            self.logger.info("=" * 60)
            
            # Main event loop
            while self.running:
                await asyncio.sleep(1)
                # Main trading loop will be implemented in later days
            
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            self.logger.critical(f"System startup failed: {e}", exc_info=True)
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        self.logger.info("Initiating system shutdown...")
        self.running = False
        
        # Shutdown components in reverse order
        # This will be expanded as components are added
        
        self.logger.info("System shutdown complete")
    
    def _handle_shutdown(self, signum, _):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    def _log_configuration(self):
        """Log configuration summary."""
        self.logger.info("Configuration Summary:")
        self.logger.info(f"  IBKR Account: {self.config.ibkr.account}")
        self.logger.info(f"  IBKR Host: {self.config.ibkr.host}:{self.config.ibkr.live_port}")
        self.logger.info(f"  Alpha Vantage Rate Limit: {self.config.alpha_vantage.rate_limit}/min")
        self.logger.info(f"  Risk Limits:")
        self.logger.info(f"    Max Positions: {self.config.risk_limits.max_positions}")
        self.logger.info(f"    Max Position Size: ${self.config.risk_limits.max_position_size:,.2f}")
        self.logger.info(f"    Daily Loss Limit: ${self.config.risk_limits.daily_loss_limit:,.2f}")
        self.logger.info(f"    VPIN Threshold: {self.config.risk_limits.vpin_threshold}")
        self.logger.info(f"  Greeks Limits:")
        self.logger.info(f"    Delta: [{self.config.greeks_limits.delta_min}, {self.config.greeks_limits.delta_max}]")
        self.logger.info(f"    Gamma: [{self.config.greeks_limits.gamma_min}, {self.config.greeks_limits.gamma_max}]")
        self.logger.info(f"    Vega: [{self.config.greeks_limits.vega_min}, {self.config.greeks_limits.vega_max}]")
        self.logger.info(f"    Theta: > {self.config.greeks_limits.theta_min}")
        
        if self.config.community.is_enabled:
            self.logger.info(f"  Community Features: ENABLED")
        else:
            self.logger.info(f"  Community Features: DISABLED")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaTrader - High-Frequency Options Trading System"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)",
        default=None
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["production", "paper", "development", "testing"],
        help="Trading mode",
        default="development"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        help="Path to .env file",
        default=None
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        
        if args.config:
            # Load from YAML file
            config_path = Path(args.config)
            config = config_manager.load_from_yaml(config_path)
        else:
            # Load from environment
            env_path = Path(args.env) if args.env else None
            config = config_manager.load_from_env(env_path)
        
        # Override mode if specified
        if args.mode:
            config.mode = TradingMode(args.mode)
        
        # Validate configuration for production
        if config.mode == TradingMode.PRODUCTION:
            issues = config.validate_for_production()
            if issues:
                print("⚠️  Production validation warnings:")
                for issue in issues:
                    print(f"  - {issue}")
                
                response = input("\nContinue anyway? (yes/no): ")
                if response.lower() != "yes":
                    print("Startup cancelled")
                    return 1
        
        # Create and start application
        app = AlphaTrader(config)
        await app.start()
        
        return 0
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        if 'app' in locals():
            await app.shutdown()
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)