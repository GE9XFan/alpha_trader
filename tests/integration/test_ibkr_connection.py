#!/usr/bin/env python3
"""
Test IBKRConnectionManager with live connection
Tests real-time bars, quotes, and MOC window
"""

import sys
import asyncio
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.foundation.config_manager import ConfigManager
from src.connections.ibkr_connection import IBKRConnectionManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IBKRConnectionTester:
    """Test IBKR connection and data feeds"""
    
    def __init__(self):
        """Initialize tester"""
        logger.info("Initializing IBKR Connection Tester...")
        
        # Initialize components
        self.config_manager = ConfigManager()
        
        # Create connection manager
        self.ibkr = IBKRConnectionManager(config_manager=self.config_manager)
        
        # Create output directory
        self.output_dir = Path("data/ibkr_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track received data
        self.bars_received = {}
        self.quotes_received = {}
        self.test_results = {}
        
        # Test configuration
        self.test_symbols = ['SPY', 'QQQ', 'AAPL']  # Test symbols
        self.test_duration = 30  # Seconds to run each test
    
    def on_bar_update(self, symbol: str, bar_size: str, bar) -> None:
        """Callback for bar updates"""
        if symbol not in self.bars_received:
            self.bars_received[symbol] = {}
        if bar_size not in self.bars_received[symbol]:
            self.bars_received[symbol][bar_size] = []
        
        bar_data = {
            'time': datetime.now().isoformat(),
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'average': bar.average,
            'barCount': bar.barCount
        }
        
        self.bars_received[symbol][bar_size].append(bar_data)
        
        logger.info(
            f"📊 Bar Update - {symbol} ({bar_size}): "
            f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
            f"C={bar.close:.2f} V={bar.volume}"
        )
    
    def on_quote_update(self, symbol: str, quote: Dict[str, Any]) -> None:
        """Callback for quote updates"""
        if symbol not in self.quotes_received:
            self.quotes_received[symbol] = []
        
        quote_data = {
            'time': datetime.now().isoformat(),
            **quote
        }
        
        self.quotes_received[symbol].append(quote_data)
        
        # Only log every 10th quote to avoid spam
        if len(self.quotes_received[symbol]) % 10 == 0:
            logger.info(
                f"💹 Quote Update - {symbol}: "
                f"Bid={quote.get('bid', 'N/A')} "
                f"Ask={quote.get('ask', 'N/A')} "
                f"Last={quote.get('last', 'N/A')}"
            )
    
    def test_connection(self) -> bool:
        """Test basic connection"""
        logger.info("\n" + "="*60)
        logger.info("🔌 TESTING CONNECTION")
        logger.info("="*60)
        
        # Check configuration
        logger.info(f"Trading Mode: {self.ibkr.trading_mode}")
        logger.info(f"Gateway: {self.ibkr.gateway_host}:{self.ibkr.gateway_port}")
        logger.info(f"Max Subscriptions: {self.ibkr.max_concurrent_subscriptions}")
        
        # Initialize connection
        initialized = self.ibkr.initialize()
        
        if initialized:
            logger.info("✅ Connection initialized successfully")
            self.test_results['connection'] = 'SUCCESS'
            
            # Get account info
            account_summary = self.ibkr.get_account_summary()
            if account_summary:
                logger.info(f"📋 Account Type: {account_summary.get('AccountType', 'N/A')}")
                logger.info(f"💰 Net Liquidation: {account_summary.get('NetLiquidation', 'N/A')}")
                logger.info(f"💵 Available Funds: {account_summary.get('AvailableFunds', 'N/A')}")
            
            # Get positions
            positions = self.ibkr.get_positions()
            logger.info(f"📈 Open Positions: {len(positions)}")
            for pos in positions:
                logger.info(f"  - {pos['symbol']}: {pos['position']} @ {pos['avg_cost']}")
            
            return True
        else:
            logger.error("❌ Connection failed")
            self.test_results['connection'] = 'FAILED'
            return False
    
    def test_realtime_bars(self) -> None:
        """Test real-time bar subscriptions"""
        logger.info("\n" + "="*60)
        logger.info("📊 TESTING REAL-TIME BARS")
        logger.info("="*60)
        
        # Add callback
        self.ibkr.add_bar_callback(self.on_bar_update)
        
        # Test different bar sizes from configuration
        test_bar_sizes = ['1 min', '5 secs']  # Critical ones
        
        for symbol in self.test_symbols[:2]:  # Test first 2 symbols
            for bar_size in test_bar_sizes:
                if bar_size in self.ibkr.bar_sizes:
                    logger.info(f"\nSubscribing to {bar_size} bars for {symbol}...")
                    
                    success = self.ibkr.subscribe_bars(symbol, bar_size)
                    if success:
                        logger.info(f"✅ Subscribed to {bar_size} bars for {symbol}")
                        self.test_results[f'bars_{symbol}_{bar_size}'] = 'SUBSCRIBED'
                    else:
                        logger.error(f"❌ Failed to subscribe to {bar_size} bars for {symbol}")
                        self.test_results[f'bars_{symbol}_{bar_size}'] = 'FAILED'
                    
                    # Small delay between subscriptions
                    time.sleep(1)
        
        # Wait for bar data
        logger.info(f"\n⏱️ Collecting bar data for {self.test_duration} seconds...")
        time.sleep(self.test_duration)
        
        # Check results
        logger.info("\n📊 Bar Data Summary:")
        for symbol, bars_by_size in self.bars_received.items():
            for bar_size, bars in bars_by_size.items():
                logger.info(f"  {symbol} ({bar_size}): {len(bars)} bars received")
                self.test_results[f'bars_{symbol}_{bar_size}_count'] = len(bars)
    
    def test_realtime_quotes(self) -> None:
        """Test real-time quote subscriptions"""
        logger.info("\n" + "="*60)
        logger.info("💹 TESTING REAL-TIME QUOTES")
        logger.info("="*60)
        
        # Add callback
        self.ibkr.add_quote_callback(self.on_quote_update)
        
        for symbol in self.test_symbols[:2]:  # Test first 2 symbols
            logger.info(f"\nSubscribing to quotes for {symbol}...")
            
            success = self.ibkr.subscribe_quotes(symbol)
            if success:
                logger.info(f"✅ Subscribed to quotes for {symbol}")
                self.test_results[f'quotes_{symbol}'] = 'SUBSCRIBED'
            else:
                logger.error(f"❌ Failed to subscribe to quotes for {symbol}")
                self.test_results[f'quotes_{symbol}'] = 'FAILED'
            
            time.sleep(1)
        
        # Wait for quote data
        logger.info(f"\n⏱️ Collecting quote data for {self.test_duration} seconds...")
        time.sleep(self.test_duration)
        
        # Check results
        logger.info("\n💹 Quote Data Summary:")
        for symbol, quotes in self.quotes_received.items():
            logger.info(f"  {symbol}: {len(quotes)} quotes received")
            self.test_results[f'quotes_{symbol}_count'] = len(quotes)
            
            if quotes:
                # Show last quote
                last_quote = quotes[-1]
                logger.info(f"    Last: Bid={last_quote.get('bid')} Ask={last_quote.get('ask')}")
    
    def test_moc_window(self) -> None:
        """Test MOC imbalance window detection"""
        logger.info("\n" + "="*60)
        logger.info("🕐 TESTING MOC WINDOW")
        logger.info("="*60)
        
        # Check current time vs MOC window
        now = datetime.now().time()
        logger.info(f"Current time: {now}")
        logger.info(f"MOC Window: {self.ibkr.moc_start} - {self.ibkr.moc_end}")
        
        in_window = self.ibkr.moc_start <= now <= self.ibkr.moc_end
        logger.info(f"In MOC Window: {'✅ Yes' if in_window else '❌ No'}")
        
        # Try to subscribe
        if self.ibkr.moc_enabled:
            success = self.ibkr.subscribe_moc_imbalance()
            if success:
                logger.info("✅ MOC imbalance subscription successful")
                self.test_results['moc_subscription'] = 'SUCCESS'
            else:
                logger.info("ℹ️ MOC subscription not active (outside window or not available)")
                self.test_results['moc_subscription'] = 'NOT_ACTIVE'
    
    def test_health_check(self) -> None:
        """Test health check functionality"""
        logger.info("\n" + "="*60)
        logger.info("🏥 TESTING HEALTH CHECK")
        logger.info("="*60)
        
        health = self.ibkr.health_check()
        
        logger.info(f"Overall Health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
        logger.info(f"Connected: {health['connected']}")
        logger.info(f"Trading Mode: {health['trading_mode']}")
        logger.info(f"Active Bar Subscriptions: {health['active_bar_subscriptions']}")
        logger.info(f"Active Quote Subscriptions: {health['active_quote_subscriptions']}")
        
        logger.info("\nHealth Checks:")
        for check, status in health['checks'].items():
            logger.info(f"  {check}: {'✅' if status else '❌'}")
        
        logger.info("\nStatistics:")
        for stat, value in health['stats'].items():
            logger.info(f"  {stat}: {value}")
        
        self.test_results['health_check'] = 'PASSED' if health['healthy'] else 'FAILED'
    
    def save_results(self) -> None:
        """Save test results and received data"""
        logger.info("\n" + "="*60)
        logger.info("💾 SAVING RESULTS")
        logger.info("="*60)
        
        # Save test results
        results_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        logger.info(f"Test results saved to: {results_file}")
        
        # Save bar data
        if self.bars_received:
            bars_file = self.output_dir / f"bars_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(bars_file, 'w') as f:
                json.dump(self.bars_received, f, indent=2, default=str)
            logger.info(f"Bar data saved to: {bars_file}")
        
        # Save quote data (last 100 only to avoid huge files)
        if self.quotes_received:
            quotes_sample = {}
            for symbol, quotes in self.quotes_received.items():
                quotes_sample[symbol] = quotes[-100:] if len(quotes) > 100 else quotes
            
            quotes_file = self.output_dir / f"quotes_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(quotes_file, 'w') as f:
                json.dump(quotes_sample, f, indent=2, default=str)
            logger.info(f"Quote data saved to: {quotes_file}")
    
    def print_summary(self) -> None:
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("📋 TEST SUMMARY")
        logger.info("="*60)
        
        # Count successes and failures
        successes = [k for k, v in self.test_results.items() if v in ['SUCCESS', 'SUBSCRIBED', 'PASSED']]
        failures = [k for k, v in self.test_results.items() if v in ['FAILED']]
        
        logger.info(f"\n✅ Successful: {len(successes)}")
        logger.info(f"❌ Failed: {len(failures)}")
        
        if successes:
            logger.info("\nSuccessful Tests:")
            for test in successes:
                logger.info(f"  ✅ {test}")
        
        if failures:
            logger.info("\nFailed Tests:")
            for test in failures:
                logger.info(f"  ❌ {test}")
        
        # Data summary
        total_bars = sum(len(bars) for bars_by_size in self.bars_received.values() for bars in bars_by_size.values())
        total_quotes = sum(len(quotes) for quotes in self.quotes_received.values())
        
        logger.info(f"\n📊 Data Received:")
        logger.info(f"  Total Bars: {total_bars}")
        logger.info(f"  Total Quotes: {total_quotes}")
    
    def run_all_tests(self) -> None:
        """Run all IBKR connection tests"""
        logger.info("="*60)
        logger.info("🚀 IBKR CONNECTION MANAGER TEST")
        logger.info("="*60)
        logger.info(f"Environment: {self.config_manager.environment}")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Test connection first
            if not self.test_connection():
                logger.error("❌ Connection failed, aborting tests")
                logger.info("\n⚠️ Make sure TWS or IB Gateway is running!")
                logger.info("For TWS: Enable API connections in Settings > API > Settings")
                logger.info("For Gateway: Default port is 4001 (paper) or 4002 (live)")
                return
            
            # Run tests
            self.test_realtime_bars()
            self.test_realtime_quotes()
            self.test_moc_window()
            self.test_health_check()
            
            # Save results
            self.save_results()
            
            # Print summary
            self.print_summary()
            
        finally:
            # Shutdown
            logger.info("\n🔌 Shutting down connection...")
            if self.ibkr.shutdown():
                logger.info("✅ Shutdown complete")
            else:
                logger.error("❌ Shutdown failed")
        
        logger.info("\n✅ All tests completed!")


def main():
    """Run the IBKR connection tests"""
    
    # Pre-flight check
    logger.info("Pre-flight checklist:")
    logger.info("1. Is TWS or IB Gateway running?")
    logger.info("2. Is API access enabled?")
    logger.info("3. Is the port correct (4001 for paper, 4002 for live)?")
    logger.info("4. Are market data subscriptions active?")
    
    response = input("\nPress Enter to continue or 'q' to quit: ")
    if response.lower() == 'q':
        return 1
    
    tester = IBKRConnectionTester()
    tester.run_all_tests()
    return 0


if __name__ == "__main__":
    sys.exit(main())