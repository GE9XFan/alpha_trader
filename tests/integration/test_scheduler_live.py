#!/usr/bin/env python3
"""
Test DataScheduler with REALTIME_OPTIONS specifically
Focus on options with Greeks through the scheduler
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Setup path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if not (project_root / 'src').exists():
    project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print(f"Using project root: {project_root}")

from src.foundation.config_manager import ConfigManager
from src.data.scheduler import DataScheduler, TaskStatus, APIType, ScheduledTask
from src.data.rate_limiter import TokenBucketRateLimiter, RequestPriority
from src.connections.av_client import AlphaVantageClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptionsSchedulerTest:
    """Test scheduler specifically for OPTIONS_WITH_GREEKS"""
    
    def __init__(self):
        """Initialize test components"""
        logger.info("="*60)
        logger.info("SCHEDULER TEST - REALTIME OPTIONS WITH GREEKS")
        logger.info("="*60)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.rate_limiter = TokenBucketRateLimiter('alpha_vantage', self.config_manager)
        self.av_client = AlphaVantageClient(self.config_manager, self.rate_limiter)
        self.scheduler = DataScheduler(self.config_manager, self.av_client, None, self.rate_limiter)
        
        # Check API key
        api_key = self.config_manager.get('env.AV_API_KEY')
        if not api_key or 'your_alpha' in api_key:
            logger.error("❌ No valid API key configured!")
            raise ValueError("Set AV_API_KEY in .env file")
        
        logger.info(f"✅ API key configured: {api_key[:10]}...")
        
        # Initialize scheduler
        self.scheduler.initialize()
        logger.info("✅ Scheduler initialized")
        
        # Output directory
        self.output_dir = Path("data/options_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_direct_options_call(self):
        """Test direct call to get_realtime_options"""
        logger.info("\n" + "-"*60)
        logger.info("TEST 1: Direct Options API Call")
        logger.info("-"*60)
        
        symbols = ['SPY', 'QQQ', 'AAPL']  # Test multiple symbols
        
        for symbol in symbols:
            logger.info(f"\nTesting {symbol}...")
            
            try:
                start = time.time()
                response = self.av_client.get_realtime_options(symbol)
                elapsed = time.time() - start
                
                logger.info(f"  Response time: {elapsed:.2f}s")
                
                if response is None:
                    logger.error(f"  ❌ No response for {symbol}")
                    continue
                
                # Analyze response
                if isinstance(response, dict):
                    # Check for errors
                    if 'Error Message' in response:
                        logger.error(f"  ❌ API Error: {response['Error Message']}")
                        logger.info("  → Options may require premium API key")
                        
                    elif 'Note' in response:
                        logger.warning(f"  ⚠️ Rate limit: {response['Note']}")
                        logger.info("  → Daily API limit reached (25 calls for free tier)")
                        
                    elif 'Information' in response:
                        logger.warning(f"  ⚠️ Info: {response['Information']}")
                        
                    else:
                        # Check for options data
                        logger.info(f"  Response keys: {list(response.keys())}")
                        
                        # Save response for analysis
                        response_file = self.output_dir / f"{symbol}_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(response_file, 'w') as f:
                            json.dump(response, f, indent=2)
                        logger.info(f"  Saved to: {response_file}")
                        
                        # Check for Greeks
                        if 'options' in response:
                            options = response['options']
                            logger.info(f"  ✅ Options data found: {len(options)} contracts")
                            
                            # Check first option for Greeks
                            if options:
                                first_key = list(options.keys())[0]
                                first_option = options[first_key]
                                
                                # Check for Greek fields
                                greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
                                found_greeks = [g for g in greeks if g in str(first_option).lower()]
                                
                                if found_greeks:
                                    logger.info(f"  ✅ Greeks found: {found_greeks}")
                                else:
                                    logger.warning(f"  ⚠️ No Greeks found in response")
                                    logger.info(f"  First option keys: {list(first_option.keys()) if isinstance(first_option, dict) else 'Not a dict'}")
                        else:
                            logger.warning(f"  ⚠️ No 'options' key in response")
                            
            except Exception as e:
                logger.error(f"  ❌ Exception: {e}")
                
            # Respect rate limits
            time.sleep(1)
    
    def test_scheduler_options_execution(self):
        """Test OPTIONS_WITH_GREEKS through scheduler"""
        logger.info("\n" + "-"*60)
        logger.info("TEST 2: Options Execution Through Scheduler")
        logger.info("-"*60)
        
        # Create high-priority options task
        task = ScheduledTask(
            symbol="SPY",
            api_type=APIType.OPTIONS_WITH_GREEKS,
            priority=1,  # Highest priority
            interval_seconds=12,  # From config
            tier="tier_a",
            next_run=datetime.now()
        )
        
        logger.info(f"Executing options task: {task.symbol}")
        
        try:
            start = time.time()
            response = self.scheduler._execute_api_call(task)
            elapsed = time.time() - start
            
            logger.info(f"  Execution time: {elapsed:.2f}s")
            
            if response:
                logger.info("  ✅ Response received through scheduler")
                
                # Save for analysis
                response_file = self.output_dir / f"scheduler_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Handle large responses
                if isinstance(response, dict):
                    # Truncate if too large
                    if 'options' in response and len(str(response)) > 10000:
                        truncated = {
                            'meta': {k: v for k, v in response.items() if k != 'options'},
                            'options_count': len(response.get('options', {})),
                            'sample_option': list(response.get('options', {}).values())[0] if response.get('options') else None
                        }
                        with open(response_file, 'w') as f:
                            json.dump(truncated, f, indent=2)
                        logger.info(f"  Saved truncated response to: {response_file}")
                    else:
                        with open(response_file, 'w') as f:
                            json.dump(response, f, indent=2)
                        logger.info(f"  Saved full response to: {response_file}")
            else:
                logger.error("  ❌ No response from scheduler")
                
        except Exception as e:
            logger.error(f"  ❌ Scheduler execution failed: {e}")
    
    def test_options_in_schedule(self):
        """Test that options tasks are properly scheduled"""
        logger.info("\n" + "-"*60)
        logger.info("TEST 3: Options Tasks in Schedule")
        logger.info("-"*60)
        
        # Get current tasks
        stats = self.scheduler.get_statistics()
        
        logger.info(f"Total tasks: {stats['current_state']['total_tasks']}")
        logger.info(f"Task types: {stats['tasks_by_type']}")
        
        # Check for options tasks
        options_count = stats['tasks_by_type'].get('options_with_greeks', 0)
        
        if options_count > 0:
            logger.info(f"✅ Found {options_count} options tasks in schedule")
            
            # Get next options tasks
            next_tasks = self.scheduler.get_next_tasks(50)
            options_tasks = [t for t in next_tasks if t['api_type'] == 'options_with_greeks']
            
            logger.info(f"\nNext options tasks to execute:")
            for task in options_tasks[:5]:
                logger.info(f"  {task['symbol']} - Priority {task['priority']} - {task['tier']}")
        else:
            logger.error("❌ No options tasks found in schedule!")
    
    def test_rate_limiting_with_options(self):
        """Test rate limiting with options calls"""
        logger.info("\n" + "-"*60)
        logger.info("TEST 4: Rate Limiting with Options")
        logger.info("-"*60)
        
        # Check current tokens
        tokens_start = self.rate_limiter.get_available_tokens()
        logger.info(f"Starting tokens: {tokens_start:.1f}")
        
        # Make multiple options calls
        symbols = ['SPY', 'QQQ']
        
        for symbol in symbols:
            task = ScheduledTask(
                symbol=symbol,
                api_type=APIType.OPTIONS_WITH_GREEKS,
                priority=1,
                interval_seconds=12,
                tier="tier_a"
            )
            
            tokens_before = self.rate_limiter.get_available_tokens()
            
            try:
                response = self.scheduler._execute_api_call(task)
                tokens_after = self.rate_limiter.get_available_tokens()
                
                logger.info(f"  {symbol}: Tokens {tokens_before:.1f} → {tokens_after:.1f}")
                
                if response and 'Error Message' not in response:
                    logger.info(f"    ✅ Call succeeded")
                elif response and 'Error Message' in response:
                    logger.warning(f"    ⚠️ API error: {response['Error Message'][:50]}...")
                else:
                    logger.error(f"    ❌ No response")
                    
            except Exception as e:
                logger.error(f"  {symbol}: Exception: {e}")
            
            time.sleep(0.5)  # Small delay between calls
        
        tokens_end = self.rate_limiter.get_available_tokens()
        logger.info(f"\nEnding tokens: {tokens_end:.1f}")
        logger.info(f"Tokens consumed: {tokens_start - tokens_end:.1f}")
    
    def generate_report(self):
        """Generate test report"""
        logger.info("\n" + "="*60)
        logger.info("OPTIONS TEST REPORT")
        logger.info("="*60)
        
        # Get scheduler stats
        stats = self.scheduler.get_statistics()
        
        logger.info(f"Scheduler Statistics:")
        logger.info(f"  Tasks executed: {stats['tasks_executed']}")
        logger.info(f"  Tasks failed: {stats['tasks_failed']}")
        logger.info(f"  Tasks rate limited: {stats['tasks_rate_limited']}")
        logger.info(f"  Options tasks: {stats['tasks_by_type'].get('options_with_greeks', 0)}")
        
        # Rate limiter stats
        rl_stats = self.rate_limiter.get_statistics()
        logger.info(f"\nRate Limiter Statistics:")
        logger.info(f"  Total requests: {rl_stats['total_requests']}")
        logger.info(f"  Current RPM: {rl_stats['current_rpm']}")
        logger.info(f"  Available tokens: {self.rate_limiter.get_available_tokens():.1f}")
        
        logger.info(f"\nOutput files saved to: {self.output_dir}")
        
        # Common issues
        logger.info("\n" + "-"*60)
        logger.info("COMMON ISSUES WITH OPTIONS API:")
        logger.info("-"*60)
        logger.info("1. 'Invalid API call' - Options may require premium subscription")
        logger.info("2. 'Thank you for using Alpha Vantage' - Daily limit reached (25 for free)")
        logger.info("3. No Greeks in response - Free tier may not include Greeks")
        logger.info("4. Empty response - Symbol may not have options available")
        logger.info("5. Rate limit - Max 5 calls/minute for free tier")
    
    def run_all_tests(self):
        """Run all options tests"""
        try:
            # Test 1: Direct API call
            self.test_direct_options_call()
            
            # Test 2: Through scheduler
            self.test_scheduler_options_execution()
            
            # Test 3: Check schedule
            self.test_options_in_schedule()
            
            # Test 4: Rate limiting
            self.test_rate_limiting_with_options()
            
            # Generate report
            self.generate_report()
            
        finally:
            # Cleanup
            logger.info("\nShutting down scheduler...")
            self.scheduler.shutdown()
            logger.info("✅ Shutdown complete")


def main():
    """Run the options test"""
    try:
        tester = OptionsSchedulerTest()
        tester.run_all_tests()
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()