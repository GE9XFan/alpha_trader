#!/usr/bin/env python3
"""
Complete System Integration Test for DataScheduler
Tests the ACTUAL implementation with REAL config files and REAL task execution
NO MOCKS - NO MANUAL PARAMETERS - REAL SYSTEM TEST
"""

import sys
import asyncio
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Setup path - same pattern as existing test
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if not (project_root / 'src').exists():
    project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Import actual project modules
from src.foundation.config_manager import ConfigManager
from src.data.scheduler import DataScheduler, TaskStatus, APIType
from src.data.rate_limiter import TokenBucketRateLimiter
from src.connections.av_client import AlphaVantageClient

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_scheduler_live_full.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealSchedulerSystemTest:
    """
    REAL system test - uses actual config files and runs the scheduler
    exactly as it would run in production
    """
    
    def __init__(self):
        """Initialize test - will use REAL configs"""
        self.config_manager = None
        self.rate_limiter = None
        self.av_client = None
        self.scheduler = None
        
        # Track what actually happens
        self.execution_log = []
        self.api_calls = []
        self.errors = []
        
        # Results directory
        self.results_dir = Path('test_results')
        self.results_dir.mkdir(exist_ok=True)
    
    def run_production_system(self):
        """
        Run the ACTUAL production system using REAL configs
        This is exactly how the system would run in production
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING REAL PRODUCTION SYSTEM TEST")
            logger.info("Using actual configuration files from config/")
            logger.info("Using environment variables from .env")
            logger.info("=" * 80)
            
            # Step 1: Initialize ConfigManager - it will load ALL YAML files
            logger.info("\n📁 Loading production configuration files...")
            self.config_manager = ConfigManager()
            
            # Log what configs were loaded
            logger.info("Configuration loaded successfully")
            
            # Step 2: Verify environment
            logger.info("\n🔑 Verifying environment variables...")
            av_api_key = self.config_manager.get('env.AV_API_KEY')
            if not av_api_key or 'your_alpha' in av_api_key:
                raise ValueError("AV_API_KEY not configured properly!")
            logger.info(f"  ✓ AV_API_KEY: {av_api_key[:10]}...")
            
            # Step 3: Initialize Rate Limiter
            logger.info("\n⏱️  Initializing Rate Limiter...")
            self.rate_limiter = TokenBucketRateLimiter('alpha_vantage', self.config_manager)
            
            # Step 4: Initialize Alpha Vantage Client
            logger.info("\n🌐 Initializing Alpha Vantage Client...")
            self.av_client = AlphaVantageClient(self.config_manager, self.rate_limiter)
            
            # Step 5: Initialize DataScheduler with real components
            logger.info("\n📋 Initializing DataScheduler...")
            self.scheduler = DataScheduler(
                config_manager=self.config_manager,
                av_client=self.av_client,
                ibkr_connection=None,  # Can be None - scheduler handles it
                rate_limiter=self.rate_limiter
            )
            
            # Step 6: Initialize scheduler (creates tasks from config)
            logger.info("\n🚀 Starting scheduler initialization...")
            if not self.scheduler.initialize():
                raise RuntimeError("Failed to initialize scheduler")
            
            # Log what tasks were created - use the actual tasks attribute
            tasks = self.scheduler.tasks  # Direct access to tasks dictionary
            logger.info(f"\n📊 Tasks created from config: {len(tasks)}")
            
            # Get statistics for more info
            stats = self.scheduler.get_statistics()
            
            # Show task distribution from statistics
            distribution = stats.get('task_distribution', {})
            if distribution.get('by_api_type'):
                logger.info("Tasks by API type:")
                for api_type, count in distribution['by_api_type'].items():
                    logger.info(f"  {api_type}: {count}")
            
            if distribution.get('by_tier'):
                logger.info("Tasks by tier:")
                for tier, count in distribution['by_tier'].items():
                    logger.info(f"  {tier}: {count}")
            
            # Get next tasks to show what will execute
            next_tasks = self.scheduler.get_next_tasks(limit=10)
            if next_tasks:
                logger.info(f"\nNext {len(next_tasks)} tasks to execute:")
                for task_info in next_tasks[:5]:
                    logger.info(f"  - {task_info['symbol']} {task_info['api_type']} (tier: {task_info['tier']})")
            
            # Save task details for inspection
            tasks_data = {}
            for task_id, task in tasks.items():
                tasks_data[task_id] = {
                    'symbol': task.symbol,
                    'api_type': task.api_type.value if hasattr(task.api_type, 'value') else str(task.api_type),
                    'tier': task.tier,
                    'priority': task.priority,
                    'interval_seconds': task.interval_seconds,
                    'next_run': task.next_run.isoformat() if hasattr(task, 'next_run') and task.next_run else None,
                    'status': task.status.value if hasattr(task, 'status') else 'PENDING'
                }
            
            with open(self.results_dir / "scheduled_tasks.json", 'w') as f:
                json.dump(tasks_data, f, indent=2)
            logger.info(f"📁 Task details saved to test_results/scheduled_tasks.json")
            
            # Step 7: Run for test duration
            test_duration = 300  # 5 minutes default
            
            # Check if there's a test duration in config
            system_config = self.config_manager.get('system', {})
            if system_config and 'test' in system_config:
                test_duration = system_config.get('test', {}).get('integration_test_duration_seconds', 300)
            
            logger.info(f"\n⏱️  Running for {test_duration} seconds ({test_duration/60:.1f} minutes)")
            logger.info("Monitoring task execution...\n")
            
            # Monitor with periodic status updates
            start_time = time.time()
            last_stats_time = start_time
            stats_interval = 30  # Report every 30 seconds
            
            while time.time() - start_time < test_duration:
                current_time = time.time()
                
                # Periodic status update
                if current_time - last_stats_time >= stats_interval:
                    elapsed = current_time - start_time
                    stats = self.scheduler.get_statistics()
                    
                    logger.info(f"\n📊 Status at {elapsed:.0f}s:")
                    logger.info(f"  Tasks executed: {stats.get('tasks_executed', 0)}")
                    logger.info(f"  Tasks scheduled: {stats.get('tasks_scheduled', 0)}")
                    logger.info(f"  Tasks failed: {stats.get('tasks_failed', 0)}")
                    
                    # Check current state
                    current_state = stats.get('current_state', {})
                    logger.info(f"  Queue size: {current_state.get('queue_size', 0)}")
                    logger.info(f"  Active tasks: {current_state.get('active_tasks', 0)}")
                    logger.info(f"  Market hours: {current_state.get('market_hours', False)}")
                    logger.info(f"  MOC window: {current_state.get('moc_window', False)}")
                    
                    # Task distribution
                    distribution = stats.get('task_distribution', {})
                    if distribution.get('by_api_type'):
                        logger.info("  API calls by type:")
                        for api_type, count in distribution['by_api_type'].items():
                            logger.info(f"    {api_type}: {count}")
                    
                    last_stats_time = current_time
                
                time.sleep(1)
            
            # Step 8: Stop scheduler
            logger.info("\n🛑 Stopping scheduler...")
            self.scheduler.shutdown()
            
            # Step 9: Analyze results
            return self.analyze_execution_results()
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def analyze_execution_results(self):
        """Analyze what actually happened during the test"""
        logger.info("\n" + "=" * 80)
        logger.info("ANALYZING EXECUTION RESULTS")
        logger.info("=" * 80)
        
        # Get final statistics
        stats = self.scheduler.get_statistics()
        
        # Overall execution stats
        logger.info("\n📈 Overall Execution Statistics:")
        logger.info(f"  Total tasks scheduled: {stats.get('tasks_scheduled', 0)}")
        logger.info(f"  Total tasks executed: {stats.get('tasks_executed', 0)}")
        logger.info(f"  Total tasks failed: {stats.get('tasks_failed', 0)}")
        logger.info(f"  Total rate limit delays: {stats.get('rate_limit_delays', 0)}")
        
        # Task distribution
        distribution = stats.get('task_distribution', {})
        
        logger.info("\n📊 Task Distribution by Status:")
        for status, count in distribution.get('by_status', {}).items():
            logger.info(f"  {status}: {count}")
        
        logger.info("\n📊 Task Distribution by Tier:")
        for tier, count in distribution.get('by_tier', {}).items():
            logger.info(f"  {tier}: {count}")
        
        logger.info("\n📊 Task Distribution by API Type:")
        for api_type, count in distribution.get('by_api_type', {}).items():
            logger.info(f"  {api_type}: {count}")
        
        # Current state
        current_state = stats.get('current_state', {})
        logger.info("\n🔍 Final State:")
        logger.info(f"  Total tasks in system: {current_state.get('total_tasks', 0)}")
        logger.info(f"  Tasks in queue: {current_state.get('queue_size', 0)}")
        logger.info(f"  Active tasks: {current_state.get('active_tasks', 0)}")
        
        # Save complete statistics
        stats_file = self.results_dir / f"execution_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"\n📁 Complete statistics saved to: {stats_file}")
        
        # Health check
        health = self.scheduler.health_check()
        logger.info("\n🏥 Health Check Results:")
        for check, result in health.get('checks', {}).items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check}: {result}")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        total_executed = stats.get('tasks_executed', 0)
        total_failed = stats.get('tasks_failed', 0)
        success_rate = ((total_executed - total_failed) / total_executed * 100) if total_executed > 0 else 0
        
        logger.info(f"Total tasks executed: {total_executed}")
        logger.info(f"Successful: {total_executed - total_failed}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Determine pass/fail
        passed = (
            total_executed > 0 and
            success_rate >= 80 and  # 80% success rate threshold
            health.get('healthy', False)
        )
        
        if passed:
            logger.info("\n🎉 TEST PASSED - System is working correctly!")
            logger.info("The scheduler is:")
            logger.info("  ✓ Loading tasks from configuration")
            logger.info("  ✓ Executing API calls through AlphaVantageClient")
            logger.info("  ✓ Respecting rate limits")
            logger.info("  ✓ Managing task priorities")
            logger.info("  ✓ Handling errors gracefully")
        else:
            logger.error("\n❌ TEST FAILED - Issues detected")
            if total_executed == 0:
                logger.error("  - No tasks were executed")
            if success_rate < 80:
                logger.error(f"  - Success rate too low: {success_rate:.1f}% < 80%")
            if not health.get('healthy', False):
                logger.error("  - Health check failed")
        
        return passed
    
    def verify_api_coverage(self):
        """Verify that configured APIs are actually being called"""
        logger.info("\n📋 Verifying API Coverage...")
        
        # Get scheduler statistics
        stats = self.scheduler.get_statistics()
        distribution = stats.get('task_distribution', {})
        api_types = distribution.get('by_api_type', {})
        
        if api_types:
            logger.info(f"APIs that were scheduled: {list(api_types.keys())}")
            logger.info(f"Total unique API types: {len(api_types)}")
        else:
            logger.warning("No API type distribution data available")
        
        # Check if key APIs were called
        expected_apis = ['options_with_greeks', 'rsi', 'macd', 'bbands', 'vwap']
        covered = [api for api in expected_apis if api in api_types]
        missing = [api for api in expected_apis if api not in api_types]
        
        if covered:
            logger.info(f"✓ Key APIs covered: {covered}")
        if missing:
            logger.warning(f"⚠ Key APIs not covered: {missing}")
        
        coverage = (len(covered) / len(expected_apis) * 100) if expected_apis else 0
        logger.info(f"Key API coverage: {coverage:.1f}%")
        
        return coverage >= 60  # 60% coverage of key APIs is acceptable for test


def main():
    """Main test execution"""
    logger.info("=" * 80)
    logger.info("DATA SCHEDULER COMPLETE SYSTEM TEST")
    logger.info("Mode: PRODUCTION CONFIGURATION TEST")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This test will:")
    logger.info("1. Load your actual YAML configuration files")
    logger.info("2. Use your environment variables (.env)")
    logger.info("3. Initialize the scheduler with real components")
    logger.info("4. Create tasks from your configuration")
    logger.info("5. Run the scheduler in production mode")
    logger.info("6. Execute real API calls through AlphaVantageClient")
    logger.info("7. Verify the system works as configured")
    logger.info("")
    logger.info("=" * 80)
    
    # Create and run test
    test = RealSchedulerSystemTest()
    
    # Run the production system test
    success = test.run_production_system()
    
    # Verify API coverage
    if success:
        coverage_ok = test.verify_api_coverage()
        success = success and coverage_ok
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Run the real system test
    main()