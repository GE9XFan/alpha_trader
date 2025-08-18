"""Data scheduler for automated API calls - Phase 4.2"""

import yaml
from pathlib import Path
from datetime import datetime, time
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from datetime import timedelta
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.connections.av_client import AlphaVantageClient
from src.data.ingestion import DataIngestion
from src.foundation.config_manager import ConfigManager
from src.data.rate_limiter import TokenBucketRateLimiter
import time
import logging
from src.data.bar_aggregator import BarAggregator


# ===== ADD THIS ENTIRE LOGGING BLOCK HERE =====
# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('logs/scheduler.log', mode='a')  # File output
    ]
)
logger = logging.getLogger(__name__)
# ===== END OF LOGGING BLOCK =====

class DataScheduler:
    """
    Manages automated data collection with market awareness and rate limit respect
    Phase 4.2 - Day 16
    """
    
    def __init__(self, test_mode=False):
        # Test mode flag
        self.test_mode = test_mode

        # Set up logging
        self._setup_logging()

        # Load configurations
        self._load_config()
        
        # Initialize ConfigManager
        self.config_manager = ConfigManager()

        # Initialize scheduler
        self._init_scheduler()

        ## IBKR connection management
        self.ibkr_connection = None
        self.ibkr_connected = False
        self.ibkr_subscriptions = {}
        self.ibkr_reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.ibkr_last_heartbeat = datetime.now()
        self.bar_aggregator = None

        
        # Track state
        self.is_running = False
        self.jobs_created = 0
        self.last_rate_check = datetime.now()
        
        print(f"DataScheduler initialized")
        print(f"  Market timezone: {self.timezone}")
        print(f"  Target rate: {self.rate_limit_config['target_per_minute']}/min")
        print(f"  Tier A symbols: {len(self.tiers['tier_a']['symbols'])}")
        print(f"  Tier B symbols: {len(self.tiers['tier_b']['symbols'])}")
        if test_mode:
            print(f"  🧪 TEST MODE ENABLED - Ignoring market hours")
    
    def _load_config(self):
        """Load scheduler configuration from YAML"""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'data' / 'schedules.yaml'
        
        with open(config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        # Extract key configurations
        self.market_hours = self.yaml_config['market_hours']
        self.timezone = pytz.timezone(self.market_hours['timezone'])
        self.rate_limit_config = self.yaml_config['rate_limit_budget']
        self.tiers = self.yaml_config['symbol_tiers']
        self.api_groups = self.yaml_config['api_groups']
        self.rules = self.yaml_config['scheduling_rules']
        self.moc_config = self.yaml_config['moc_window']
        
    def _init_scheduler(self):
        """Initialize APScheduler with proper configuration"""
        # Configure executors and job defaults
        executors = {
            'default': ThreadPoolExecutor(
                max_workers=self.yaml_config['scheduler']['max_workers']
            )
        }
        
        job_defaults = self.yaml_config['scheduler']['job_defaults']
        
        # Create scheduler
        self.scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone=self.timezone
        )

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # In test mode, also print to console
        if self.test_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
    
    def _log(self, message, level='info'):
        """Helper method for logging that also prints in test mode"""
        if level == 'info':
            logger.info(message)
            if self.test_mode:
                print(f"[INFO] {message}")
        elif level == 'error':
            logger.error(message)
            print(f"[ERROR] {message}")  # Always print errors
        elif level == 'warning':
            logger.warning(message)
            if self.test_mode:
                print(f"[WARNING] {message}")
        elif level == 'debug':
            logger.debug(message)
            if self.test_mode:
                print(f"[DEBUG] {message}")

    def start(self):
        """Start the scheduler"""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            print("✓ Scheduler started")
            
            # Create initial jobs
            self._create_jobs()
        else:
            print("Scheduler already running")
    
    def stop(self):
        """Stop the scheduler gracefully"""
        if self.is_running:
            logger.info("=" * 60)
            logger.info("SHUTTING DOWN SCHEDULER")
            
            # Disconnect IBKR first (CRITICAL - must be before scheduler shutdown)
            if self.ibkr_connected:
                logger.info("Disconnecting IBKR...")
                self._disconnect_ibkr()
            
            # Then shutdown scheduler
            logger.info("Stopping scheduler...")
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            
            logger.info("✓ Scheduler stopped cleanly")
            logger.info("=" * 60)
            
    def _create_jobs(self):
        """Create scheduled jobs based on configuration"""
        print(f"Creating scheduled jobs...")
        
        # ===== IBKR REAL-TIME DATA SETUP =====
        # This MUST come first to ensure market data is flowing
        logger.info("Checking IBKR setup requirements...")
        
        # Only connect during market hours or test mode
        should_connect_ibkr = self.test_mode or self._is_market_hours()
        
        if should_connect_ibkr:
            logger.info("Market is open (or test mode) - setting up IBKR...")
            
            # Connect to IBKR
            if self._connect_ibkr():
                # Subscribe to market data
                subscriptions = self._subscribe_ibkr_data()
                
                if subscriptions > 0:
                    # Schedule connection monitoring
                    self.scheduler.add_job(
                        func=self._monitor_ibkr_connection,
                        trigger='interval',
                        seconds=30,
                        id='ibkr_connection_monitor',
                        name='IBKR Connection Monitor',
                        replace_existing=True,
                        max_instances=1
                    )
                    self.jobs_created += 1
                    logger.info("✓ IBKR connection monitor scheduled (every 30s)")
                    
                    # ===== BAR AGGREGATION JOB =====
                    logger.info("Setting up bar aggregation...")
                    
                    # Create aggregator instance if not exists
                    if not self.bar_aggregator:
                        from src.data.bar_aggregator import BarAggregator
                        self.bar_aggregator = BarAggregator()
                    
                    # Schedule aggregation every 60 seconds
                    self.scheduler.add_job(
                        func=self.bar_aggregator.run_aggregation,
                        trigger='interval',
                        seconds=60,
                        id='bar_aggregation',
                        name='IBKR Bar Aggregation',
                        replace_existing=True,
                        max_instances=1,
                        misfire_grace_time=30
                    )
                    self.jobs_created += 1
                    logger.info("✓ Bar aggregation scheduled (every 60s)")
                else:
                    logger.error("⚠️ No IBKR subscriptions created - check symbol configuration")
            else:
                logger.error("⚠️ IBKR connection failed - will retry at market open")
        else:
            logger.info("Market closed - IBKR connection will be established at market open")
            
            # Schedule market open connection
            self.scheduler.add_job(
                func=self._handle_market_open,
                trigger='cron',
                hour=9,
                minute=25,  # 5 minutes before market open
                id='ibkr_market_open',
                name='IBKR Market Open Handler',
                replace_existing=True
            )
            self.jobs_created += 1
            logger.info("✓ IBKR market open handler scheduled for 9:25 AM ET")
        
        # ===== ALPHA VANTAGE SCHEDULING =====
        # THIS MUST BE AT THE SAME LEVEL AS IBKR, NOT INSIDE ELSE!
        
        # Schedule realtime options for critical API group
        if 'critical' in self.api_groups:
            self._schedule_realtime_options()
        
        # Schedule RSI indicators
        if 'indicators_fast' in self.api_groups:
            self._schedule_rsi_indicators()
            self._schedule_macd_indicators()
            self._schedule_bbands_indicators() 
            self._schedule_vwap_indicators()
        
        if 'indicators_slow' in self.api_groups:
            self._schedule_adx_indicators()
        
        # Handle ATR in daily_volatility group
        if 'daily_volatility' in self.api_groups:
            self._schedule_atr_indicators()
        
        # Schedule daily historical options
        if 'daily' in self.api_groups:
            self._schedule_historical_options()
        
        print(f"  Created {self.jobs_created} jobs")
        
        # List all jobs
        jobs = self.scheduler.get_jobs()
        for job in jobs[:10]:  # Show first 10 jobs
            print(f"    - {job.id}: {job.trigger}")

    def _connect_ibkr(self):
        """
        Connect to IBKR TWS with production-grade error handling
        
        Returns:
            bool: True if connection successful, False otherwise
        
        CRITICAL: This manages real money - connection must be stable
        """
        try:
            logger.info("=" * 60)
            logger.info("INITIATING IBKR TWS CONNECTION")
            logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Test Mode: {self.test_mode}")
            
            # Import here to avoid circular dependency
            from src.connections.ibkr_connection import IBKRConnectionManager
            
            # Create new connection instance
            if self.ibkr_connection:
                logger.warning("Existing IBKR connection found - cleaning up first")
                self._disconnect_ibkr()
            
            self.ibkr_connection = IBKRConnectionManager()
            
            # Attempt connection with timeout
            logger.info(f"Connecting to TWS at {self.ibkr_connection.host}:{self.ibkr_connection.port}")
            
            if self.ibkr_connection.connect_tws():
                self.ibkr_connected = True
                self.ibkr_reconnect_attempts = 0  # Reset counter on success
                self.ibkr_last_heartbeat = datetime.now()
                
                logger.info("✅ IBKR TWS CONNECTION SUCCESSFUL")
                logger.info(f"Client ID: {self.ibkr_connection.client_id}")
                logger.info("=" * 60)
                return True
            else:
                self.ibkr_connected = False
                logger.error("❌ IBKR TWS CONNECTION FAILED")
                logger.error("Possible causes:")
                logger.error("  1. TWS not running")
                logger.error("  2. API not enabled in TWS")
                logger.error("  3. Port mismatch (check 7497 for paper)")
                logger.error("  4. Another client using same ID")
                logger.error("=" * 60)
                return False
                
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR connecting to IBKR: {e}")
            logger.exception("Full traceback:")
            self.ibkr_connected = False
            return False

    def _disconnect_ibkr(self):
        """
        Safely disconnect from IBKR TWS
        
        CRITICAL: Must cancel all subscriptions before disconnecting
        """
        if not self.ibkr_connection:
            logger.info("No IBKR connection to disconnect")
            return
        
        try:
            logger.info("Disconnecting from IBKR TWS...")
            
            # First, cancel all active subscriptions
            if self.ibkr_subscriptions:
                logger.info(f"Cancelling {len(self.ibkr_subscriptions)} active subscriptions...")
                
                for req_id in list(self.ibkr_subscriptions.keys()):
                    try:
                        sub_info = self.ibkr_subscriptions[req_id]
                        
                        if 'quotes' in sub_info:
                            self.ibkr_connection.cancelMktData(req_id)
                            logger.debug(f"  Cancelled quotes for {sub_info}")
                        elif 'bars' in sub_info:
                            self.ibkr_connection.cancelRealTimeBars(req_id)
                            logger.debug(f"  Cancelled bars for {sub_info}")
                        
                        del self.ibkr_subscriptions[req_id]
                        
                    except Exception as e:
                        logger.error(f"  Error cancelling subscription {req_id}: {e}")
                
                # Give TWS time to process cancellations
                time.sleep(1)
            
            # Now disconnect
            self.ibkr_connection.disconnect_tws()
            self.ibkr_connected = False
            self.ibkr_connection = None
            
            logger.info("✅ IBKR TWS disconnected successfully")
            
        except Exception as e:
            logger.error(f"Error during IBKR disconnection: {e}")
            # Force cleanup
            self.ibkr_connected = False
            self.ibkr_connection = None

    def _handle_market_open(self):
        """
        Handle market open - establish IBKR connection
        
        CRITICAL: Must be ready before 9:30 AM
        """
        logger.info("=" * 60)
        logger.info(f"MARKET OPEN HANDLER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.ibkr_connected:
            logger.info("IBKR already connected")
        else:
            logger.info("Establishing IBKR connection for market open...")
            if self._connect_ibkr():
                subscriptions = self._subscribe_ibkr_data()
                if subscriptions > 0:
                    logger.info(f"✅ READY FOR MARKET OPEN - {subscriptions} data feeds active")
                else:
                    logger.error("⚠️ WARNING: No data feeds active!")
            else:
                logger.error("❌ CRITICAL: Failed to connect for market open!")
        
        logger.info("=" * 60)

    def _handle_market_close(self):
        """
        Handle market close - optional disconnect
        
        NOTE: Can maintain connection for after-hours data
        """
        logger.info("=" * 60)
        logger.info(f"MARKET CLOSE HANDLER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Decision: Keep connection for after-hours or disconnect?
        # For now, keeping connection alive
        if self.ibkr_connected:
            active_subs = len(self.ibkr_subscriptions)
            logger.info(f"Maintaining IBKR connection for after-hours")
            logger.info(f"Active subscriptions: {active_subs}")
        
        logger.info("=" * 60)

    def _subscribe_ibkr_data(self):
        """
        Subscribe to IBKR real-time data with tier-based differentiation
        
        PRODUCTION NOTES:
        - Maximum 50 concurrent market data lines (IBKR limit)
        - Tier A: Full data (quotes + all bar types)
        - Tier B: Medium data (quotes + 1min bars)
        - Tier C: Basic data (quotes only)
        - Total usage: ~42 of 50 lines
        
        Returns:
            int: Number of successful subscriptions
        """
        if not self.ibkr_connected or not self.ibkr_connection:
            logger.error("Cannot subscribe - IBKR not connected")
            return 0
        
        logger.info("=" * 60)
        logger.info("SETTING UP IBKR MARKET DATA SUBSCRIPTIONS")
        
        # Get configuration for IBKR subscriptions (add to your config if not exists)
        # This could be in schedules.yaml or a separate ibkr.yaml
        ibkr_config = self.yaml_config.get('ibkr_subscriptions', {
            'max_lines': 50,
            'subscription_delay': 0.2,  # seconds between subscriptions
            'symbol_delay': 0.5,  # seconds between symbols
            'tier_config': {
                'tier_a': {
                    'quotes': True,
                    'bar_sizes': ['5 secs', '1 min', '5 mins']
                },
                'tier_b': {
                    'quotes': True,
                    'bar_sizes': ['1 min']
                },
                'tier_c': {
                    'quotes': True,
                    'bar_sizes': []
                }
            }
        })
        
        # Get symbols by tier
        tier_a_symbols = self.tiers['tier_a']['symbols']
        tier_b_symbols = self.tiers['tier_b']['symbols']
        tier_c_symbols = self.tiers['tier_c']['symbols']
        
        # Calculate expected line usage
        tier_a_lines = len(tier_a_symbols) * (1 + len(ibkr_config['tier_config']['tier_a']['bar_sizes']))
        tier_b_lines = len(tier_b_symbols) * (1 + len(ibkr_config['tier_config']['tier_b']['bar_sizes']))
        tier_c_lines = len(tier_c_symbols) * (1 + len(ibkr_config['tier_config']['tier_c']['bar_sizes']))
        total_expected_lines = tier_a_lines + tier_b_lines + tier_c_lines
        
        # Log the subscription plan
        logger.info(f"\nSubscription Plan:")
        logger.info(f"  Tier A: {len(tier_a_symbols)} symbols × {1 + len(ibkr_config['tier_config']['tier_a']['bar_sizes'])} subscriptions = {tier_a_lines} lines")
        logger.info(f"  Tier B: {len(tier_b_symbols)} symbols × {1 + len(ibkr_config['tier_config']['tier_b']['bar_sizes'])} subscriptions = {tier_b_lines} lines")
        logger.info(f"  Tier C: {len(tier_c_symbols)} symbols × {1 + len(ibkr_config['tier_config']['tier_c']['bar_sizes'])} subscriptions = {tier_c_lines} lines")
        logger.info(f"  Total: {total_expected_lines}/{ibkr_config['max_lines']} lines ({total_expected_lines*100/ibkr_config['max_lines']:.1f}% utilization)")
        
        if total_expected_lines > ibkr_config['max_lines']:
            logger.warning(f"⚠️ Plan exceeds IBKR limit! {total_expected_lines} > {ibkr_config['max_lines']}")
            logger.warning("Consider reducing symbols or data types")
        
        successful_subscriptions = 0
        failed_subscriptions = []
        subscription_delay = ibkr_config['subscription_delay']
        symbol_delay = ibkr_config['symbol_delay']
        
        # Process each tier
        tiers_to_process = [
            ('tier_a', tier_a_symbols),
            ('tier_b', tier_b_symbols),
            ('tier_c', tier_c_symbols)
        ]
        
        symbol_counter = 0
        total_symbols = len(tier_a_symbols) + len(tier_b_symbols) + len(tier_c_symbols)
        
        for tier_name, tier_symbols in tiers_to_process:
            tier_config = ibkr_config['tier_config'][tier_name]
            
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing {tier_name.upper()} ({len(tier_symbols)} symbols)")
            logger.info(f"  Data: Quotes={tier_config['quotes']}, Bars={tier_config['bar_sizes']}")
            
            for symbol in tier_symbols:
                if not symbol:  # Skip empty symbols
                    continue
                    
                symbol_counter += 1
                symbol_subscriptions = 0
                
                try:
                    logger.info(f"\n[{symbol_counter}/{total_symbols}] {tier_name.upper()}: {symbol}")
                    
                    # Subscribe to quotes if configured
                    if tier_config['quotes']:
                        quote_req_id = self.ibkr_connection.get_quotes(symbol)
                        if quote_req_id:
                            self.ibkr_subscriptions[quote_req_id] = f"quotes_{symbol}"
                            logger.info(f"  ✓ Quotes subscription ID: {quote_req_id}")
                            successful_subscriptions += 1
                            symbol_subscriptions += 1
                        else:
                            logger.error(f"  ✗ Failed to subscribe to quotes")
                            failed_subscriptions.append(f"{symbol}_quotes")
                        
                        time.sleep(subscription_delay)
                    
                    # Subscribe to each configured bar size
                    for bar_size in tier_config['bar_sizes']:
                        try:
                            # Create a clean label for the bar size
                            bar_label = bar_size.replace(' ', '').replace('secs', 's').replace('mins', 'm').replace('min', 'm')
                            
                            bar_req_id = self.ibkr_connection.subscribe_bars(symbol, bar_size)
                            if bar_req_id:
                                self.ibkr_subscriptions[bar_req_id] = f"bars_{bar_label}_{symbol}"
                                logger.info(f"  ✓ {bar_size} bars subscription ID: {bar_req_id}")
                                successful_subscriptions += 1
                                symbol_subscriptions += 1
                            else:
                                logger.error(f"  ✗ Failed to subscribe to {bar_size} bars")
                                failed_subscriptions.append(f"{symbol}_{bar_size}_bars")
                            
                            time.sleep(subscription_delay)
                            
                        except Exception as e:
                            logger.error(f"  ✗ Error subscribing to {bar_size} bars: {e}")
                            failed_subscriptions.append(f"{symbol}_{bar_size}_bars")
                    
                    logger.info(f"  Summary: {symbol_subscriptions} subscriptions successful")
                    
                    # Delay between symbols
                    if symbol_counter < total_symbols:
                        time.sleep(symbol_delay)
                        
                except Exception as e:
                    logger.error(f"ERROR processing {symbol}: {e}")
                    failed_subscriptions.append(symbol)
        
        # Calculate final statistics
        quotes_count = sum(1 for v in self.ibkr_subscriptions.values() if 'quotes' in v)
        bars_5s_count = sum(1 for v in self.ibkr_subscriptions.values() if 'bars_5s' in v)
        bars_1m_count = sum(1 for v in self.ibkr_subscriptions.values() if 'bars_1m' in v)
        bars_5m_count = sum(1 for v in self.ibkr_subscriptions.values() if 'bars_5m' in v)
        
        # Summary report
        logger.info("\n" + "=" * 60)
        logger.info("SUBSCRIPTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Total Symbols Processed: {symbol_counter}")
        logger.info(f"  Successful Subscriptions: {successful_subscriptions}")
        logger.info(f"  Failed Subscriptions: {len(failed_subscriptions)}")
        logger.info(f"  Active Market Data Lines: {len(self.ibkr_subscriptions)}/{ibkr_config['max_lines']}")
        logger.info(f"  Utilization: {len(self.ibkr_subscriptions)*100/ibkr_config['max_lines']:.1f}%")
        
        logger.info(f"\nSubscription Breakdown:")
        logger.info(f"  Quotes: {quotes_count} symbols")
        logger.info(f"  5-second bars: {bars_5s_count} symbols")
        logger.info(f"  1-minute bars: {bars_1m_count} symbols")
        logger.info(f"  5-minute bars: {bars_5m_count} symbols")
        
        logger.info(f"\nTier Summary:")
        tier_a_subs = sum(1 for v in self.ibkr_subscriptions.values() if any(s in v for s in tier_a_symbols))
        tier_b_subs = sum(1 for v in self.ibkr_subscriptions.values() if any(s in v for s in tier_b_symbols))
        tier_c_subs = sum(1 for v in self.ibkr_subscriptions.values() if any(s in v for s in tier_c_symbols))
        logger.info(f"  Tier A subscriptions: {tier_a_subs}")
        logger.info(f"  Tier B subscriptions: {tier_b_subs}")
        logger.info(f"  Tier C subscriptions: {tier_c_subs}")
        
        if failed_subscriptions:
            logger.warning(f"\nFailed Subscriptions ({len(failed_subscriptions)}):")
            for failed in failed_subscriptions[:10]:  # Show first 10
                logger.warning(f"  - {failed}")
            if len(failed_subscriptions) > 10:
                logger.warning(f"  ... and {len(failed_subscriptions) - 10} more")
        
        logger.info("=" * 60)
        
        return successful_subscriptions

    def _monitor_ibkr_connection(self):
        """
        Monitor IBKR connection health and reconnect if needed
        
        CRITICAL: This prevents data loss from connection drops
        Runs every 30 seconds during market hours
        """
        try:
            current_time = datetime.now()
            
            # Skip if market is closed (unless test mode)
            if not self.test_mode and not self._is_market_hours():
                return
            
            # Check if we should have a connection
            if not self.ibkr_connection:
                logger.warning("IBKR connection object is None - attempting connection")
                if self._connect_ibkr():
                    self._subscribe_ibkr_data()
                return
            
            # Check connection health
            if not self.ibkr_connection.connected:
                logger.error(f"🚨 IBKR CONNECTION LOST at {current_time.strftime('%H:%M:%S')}")
                
                # Check reconnection attempts
                if self.ibkr_reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"MAX RECONNECTION ATTEMPTS ({self.max_reconnect_attempts}) REACHED")
                    logger.error("Manual intervention required!")
                    # Could send alert here
                    return
                
                self.ibkr_reconnect_attempts += 1
                logger.info(f"Reconnection attempt {self.ibkr_reconnect_attempts}/{self.max_reconnect_attempts}")
                
                # Wait with exponential backoff
                wait_time = min(2 ** self.ibkr_reconnect_attempts, 30)  # Max 30 seconds
                logger.info(f"Waiting {wait_time} seconds before reconnection...")
                time.sleep(wait_time)
                
                # Try to reconnect
                if self._connect_ibkr():
                    logger.info("✅ Reconnection successful!")
                    # Resubscribe to all data
                    self._subscribe_ibkr_data()
                else:
                    logger.error("❌ Reconnection failed")
            else:
                # Connection is healthy
                self.ibkr_last_heartbeat = current_time
                
                # Log status every 5 minutes
                if current_time.minute % 5 == 0 and current_time.second < 30:
                    active_subs = len(self.ibkr_subscriptions)
                    logger.info(f"[IBKR Health] Connected | {active_subs} subscriptions | Last HB: {current_time.strftime('%H:%M:%S')}")
                    
        except Exception as e:
            logger.error(f"ERROR in IBKR connection monitor: {e}")
            logger.exception("Full traceback:")

    def _schedule_realtime_options(self):
        """Schedule realtime options data collection"""
        critical_config = self.api_groups['critical']
        
        # Schedule Tier A symbols
        for symbol in self.tiers['tier_a']['symbols']:
            if symbol:  # Skip empty symbols
                interval = critical_config['tier_a_interval']
                job_id = f"realtime_options_{symbol}_tier_a"
                
                self.scheduler.add_job(
                    func=self._fetch_realtime_options,
                    trigger='interval',
                    seconds=interval,
                    args=[symbol],
                    id=job_id,
                    name=f"Realtime Options {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
                print(f"  ✓ Scheduled {symbol} realtime options every {interval}s")
        
        # Schedule Tier B symbols
        for symbol in self.tiers['tier_b']['symbols']:
            if symbol:
                interval = critical_config['tier_b_interval']
                job_id = f"realtime_options_{symbol}_tier_b"
                
                self.scheduler.add_job(
                    func=self._fetch_realtime_options,
                    trigger='interval',
                    seconds=interval,
                    args=[symbol],
                    id=job_id,
                    name=f"Realtime Options {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        # Schedule Tier C symbols
        for symbol in self.tiers['tier_c']['symbols']:
            if symbol:
                interval = critical_config.get('tier_c_interval', 180)  # Default 180s
                job_id = f"realtime_options_{symbol}_tier_c"
                
                self.scheduler.add_job(
                    func=self._fetch_realtime_options,
                    trigger='interval',
                    seconds=interval,
                    args=[symbol],
                    id=job_id,
                    name=f"Realtime Options {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
                
        if self.tiers['tier_c']['symbols']:
            print(f"  ✓ Scheduled {len(self.tiers['tier_c']['symbols'])} Tier C symbols every {interval}s")

    def _schedule_historical_options(self):
        """Schedule daily historical options collection"""
        daily_config = self.api_groups['daily']
        schedule_time = daily_config['schedule_time']  # "06:00"
        
        # Parse time
        hour, minute = map(int, schedule_time.split(':'))
        
        # Schedule for all tier A + B symbols
        all_symbols = (self.tiers['tier_a']['symbols'] + 
                      self.tiers['tier_b']['symbols'] +
                      self.tiers['tier_c']['symbols'])
        
        for symbol in all_symbols:
            if symbol:
                job_id = f"historical_options_{symbol}_daily"
                
                self.scheduler.add_job(
                    func=self._fetch_historical_options,
                    trigger='cron',
                    hour=hour,
                    minute=minute,
                    args=[symbol],
                    id=job_id,
                    name=f"Historical Options {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"  ✓ Scheduled daily historical options at {schedule_time}")
    
    def _schedule_rsi_indicators(self):
        """Schedule RSI indicator data collection"""
        if 'indicators_fast' not in self.api_groups:
            return
            
        indicators_config = self.api_groups['indicators_fast']
        
        # Only schedule if RSI is in the apis list
        if 'RSI' not in indicators_config.get('apis', []):
            print("  RSI not configured in indicators_fast group")
            return
            
        print("  Scheduling RSI indicators...")
        
        # Schedule Tier A symbols (every 60 seconds)
        for symbol in self.tiers['tier_a']['symbols']:
            if symbol:
                interval = indicators_config['tier_a_interval']
                job_id = f"rsi_{symbol}_tier_a"
                
                self.scheduler.add_job(
                    func=self._fetch_rsi,
                    trigger='interval',
                    seconds=interval,
                    args=[symbol],
                    id=job_id,
                    name=f"RSI {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier A: {len(self.tiers['tier_a']['symbols'])} symbols every {interval}s")
        
        # Schedule Tier B symbols (every 5 minutes)
        for symbol in self.tiers['tier_b']['symbols']:
            if symbol:
                interval = indicators_config['tier_b_interval']
                job_id = f"rsi_{symbol}_tier_b"
                
                self.scheduler.add_job(
                    func=self._fetch_rsi,
                    trigger='interval',
                    seconds=interval,
                    args=[symbol],
                    id=job_id,
                    name=f"RSI {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier B: {len(self.tiers['tier_b']['symbols'])} symbols every {interval}s")
        
        # Schedule Tier C symbols (every 10 minutes)
        for symbol in self.tiers['tier_c']['symbols']:
            if symbol:
                interval = indicators_config['tier_c_interval']
                job_id = f"rsi_{symbol}_tier_c"
                
                self.scheduler.add_job(
                    func=self._fetch_rsi,
                    trigger='interval',
                    seconds=interval,
                    args=[symbol],
                    id=job_id,
                    name=f"RSI {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier C: {len(self.tiers['tier_c']['symbols'])} symbols every {interval}s")

    def _schedule_macd_indicators(self):
        """Schedule MACD indicator data collection"""
        if 'indicators_fast' not in self.api_groups:
            return
            
        indicators_config = self.api_groups['indicators_fast']
        
        # Only schedule if MACD is in the apis list
        if 'MACD' not in indicators_config.get('apis', []):
            print("  MACD not configured in indicators_fast group")
            return
            
        print("  Scheduling MACD indicators...")
        
        # Get MACD config for default parameters
        macd_config = self.config_manager.av_config['endpoints']['macd']['default_params']
        interval = macd_config['interval']
        fastperiod = macd_config['fastperiod']
        slowperiod = macd_config['slowperiod']
        signalperiod = macd_config['signalperiod']
        series_type = macd_config['series_type']
        
        # Schedule Tier A symbols (every 60 seconds)
        for symbol in self.tiers['tier_a']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_a_interval']
                job_id = f"macd_{symbol}_tier_a"
                
                self.scheduler.add_job(
                    func=self._fetch_macd,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval, fastperiod, slowperiod, signalperiod, series_type],
                    id=job_id,
                    name=f"MACD {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier A: {len(self.tiers['tier_a']['symbols'])} symbols every {indicators_config['tier_a_interval']}s")
        
        # Schedule Tier B symbols (every 5 minutes)
        for symbol in self.tiers['tier_b']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_b_interval']
                job_id = f"macd_{symbol}_tier_b"
                
                self.scheduler.add_job(
                    func=self._fetch_macd,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval, fastperiod, slowperiod, signalperiod, series_type],
                    id=job_id,
                    name=f"MACD {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier B: {len(self.tiers['tier_b']['symbols'])} symbols every {indicators_config['tier_b_interval']}s")
        
        # Schedule Tier C symbols (every 10 minutes)
        for symbol in self.tiers['tier_c']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_c_interval']
                job_id = f"macd_{symbol}_tier_c"
                
                self.scheduler.add_job(
                    func=self._fetch_macd,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval, fastperiod, slowperiod, signalperiod, series_type],
                    id=job_id,
                    name=f"MACD {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier C: {len(self.tiers['tier_c']['symbols'])} symbols every {indicators_config['tier_c_interval']}s")

    def _schedule_bbands_indicators(self):
        """Schedule Bollinger Bands indicator data collection"""
        if 'indicators_fast' not in self.api_groups:
            return
            
        indicators_config = self.api_groups['indicators_fast']
        
        # Only schedule if BBANDS is in the apis list
        if 'BBANDS' not in indicators_config.get('apis', []):
            print("  BBANDS not configured in indicators_fast group")
            return
            
        print("  Scheduling BBANDS indicators...")
        
        # Get BBANDS config for default parameters - NO HARDCODING!
        bbands_config = self.config_manager.av_config['endpoints']['bbands']['default_params']
        interval = bbands_config['interval']
        time_period = bbands_config['time_period']
        series_type = bbands_config['series_type']
        nbdevup = bbands_config['nbdevup']
        nbdevdn = bbands_config['nbdevdn']
        matype = bbands_config['matype']
        
        # Schedule Tier A symbols (every 60 seconds)
        for symbol in self.tiers['tier_a']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_a_interval']
                job_id = f"bbands_{symbol}_tier_a"
                
                self.scheduler.add_job(
                    func=self._fetch_bbands,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval, time_period, series_type, nbdevup, nbdevdn, matype],
                    id=job_id,
                    name=f"BBANDS {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier A: {len(self.tiers['tier_a']['symbols'])} symbols every {indicators_config['tier_a_interval']}s")
        
        # Schedule Tier B symbols (every 5 minutes)
        for symbol in self.tiers['tier_b']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_b_interval']
                job_id = f"bbands_{symbol}_tier_b"
                
                self.scheduler.add_job(
                    func=self._fetch_bbands,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval, time_period, series_type, nbdevup, nbdevdn, matype],
                    id=job_id,
                    name=f"BBANDS {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier B: {len(self.tiers['tier_b']['symbols'])} symbols every {indicators_config['tier_b_interval']}s")
        
        # Schedule Tier C symbols (every 10 minutes)
        for symbol in self.tiers['tier_c']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_c_interval']
                job_id = f"bbands_{symbol}_tier_c"
                
                self.scheduler.add_job(
                    func=self._fetch_bbands,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval, time_period, series_type, nbdevup, nbdevdn, matype],
                    id=job_id,
                    name=f"BBANDS {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier C: {len(self.tiers['tier_c']['symbols'])} symbols every {indicators_config['tier_c_interval']}s")

    def _schedule_vwap_indicators(self):
        """Schedule Volume Weighted Average Price indicator data collection"""
        if 'indicators_fast' not in self.api_groups:
            return
        
        indicators_config = self.api_groups['indicators_fast']
        
        # Only schedule if VWAP is in the apis list (uppercase in config)
        if 'VWAP' not in indicators_config.get('apis', []):
            print("  VWAP not configured in indicators_fast group")
            return
        
        print("  Scheduling VWAP indicators...")
        
        # Get VWAP config for default parameters (lowercase in endpoints)
        vwap_config = self.config_manager.av_config['endpoints']['vwap']['default_params']
        interval = vwap_config['interval']
        
        # Schedule Tier A symbols (every 60 seconds)
        for symbol in self.tiers['tier_a']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_a_interval']
                job_id = f"vwap_{symbol}_tier_a"
                self.scheduler.add_job(
                    func=self._fetch_vwap,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval],  # Pass as two separate args
                    id=job_id,
                    name=f"VWAP {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier A: {len(self.tiers['tier_a']['symbols'])} symbols every {indicators_config['tier_a_interval']}s")
        
        # Schedule Tier B symbols (every 5 minutes)
        for symbol in self.tiers['tier_b']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_b_interval']
                job_id = f"vwap_{symbol}_tier_b"
                self.scheduler.add_job(
                    func=self._fetch_vwap,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval],  # Pass as two separate args
                    id=job_id,
                    name=f"VWAP {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier B: {len(self.tiers['tier_b']['symbols'])} symbols every {indicators_config['tier_b_interval']}s")
        
        # Schedule Tier C symbols (every 10 minutes)
        for symbol in self.tiers['tier_c']['symbols']:
            if symbol:
                tier_interval = indicators_config['tier_c_interval']
                job_id = f"vwap_{symbol}_tier_c"
                self.scheduler.add_job(
                    func=self._fetch_vwap,
                    trigger='interval',
                    seconds=tier_interval,
                    args=[symbol, interval],  # Pass as two separate args
                    id=job_id,
                    name=f"VWAP {symbol}",
                    replace_existing=True
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier C: {len(self.tiers['tier_c']['symbols'])} symbols every {indicators_config['tier_c_interval']}s")

    def _schedule_atr_indicators(self):
        """
        Schedule ATR indicator data collection
        Phase 5.5: Add to indicators_slow group
        
        ATR updates less frequently since it's daily data
        """
        if 'indicators_slow' not in self.api_groups:
            print("  indicators_slow group not configured")
            return
        
        slow_config = self.api_groups['indicators_slow']
        
        if 'ATR' not in slow_config.get('apis', []):
            print("  ATR not configured in indicators_slow group")
            return
        
        print("\n  Scheduling ATR (daily volatility) jobs:")
        
        # Schedule for each tier with SLOWER intervals
        # Tier A: Every 15 minutes (900s)
        for symbol in self.tiers['tier_a']['symbols']:
            job_name = f"ATR_{symbol}_A"
            self.scheduler.add_job(
                func=self._fetch_atr,
                trigger='interval',
                args=[symbol],
                seconds=slow_config.get('tier_a_interval', 900),
                id=job_name,
                name=job_name,
                replace_existing=True,
                max_instances=1
            )
            self.jobs_created += 1
            print(f"    ✓ {symbol}: Every {slow_config.get('tier_a_interval', 900)}s (15 min)")
        
        # Tier B: Every 30 minutes (1800s)
        for symbol in self.tiers['tier_b']['symbols']:
            job_name = f"ATR_{symbol}_B"
            self.scheduler.add_job(
                func=self._fetch_atr,
                trigger='interval',
                args=[symbol],
                seconds=slow_config.get('tier_b_interval', 1800),
                id=job_name,
                name=job_name,
                replace_existing=True,
                max_instances=1
            )
            self.jobs_created += 1
            print(f"    ✓ {symbol}: Every {slow_config.get('tier_b_interval', 1800)}s (30 min)")
        
        # Tier C: Every 60 minutes (3600s)
        for symbol in self.tiers['tier_c']['symbols']:
            job_name = f"ATR_{symbol}_C"
            self.scheduler.add_job(
                func=self._fetch_atr,
                trigger='interval',
                args=[symbol],
                seconds=slow_config.get('tier_c_interval', 3600),
                id=job_name,
                name=job_name,
                replace_existing=True,
                max_instances=1
            )
            self.jobs_created += 1
            print(f"    ✓ {symbol}: Every {slow_config.get('tier_c_interval', 3600)}s (60 min)")
        
        print(f"    Total ATR jobs created: {self.jobs_created}")    

    def _schedule_adx_indicators(self):
        """
        Schedule ADX indicator data collection - Phase 5.6
        
        ADX updates on slower intervals since trend strength
        changes more gradually than momentum indicators.
        """
        if 'indicators_slow' not in self.api_groups:
            print("  indicators_slow group not configured")
            return
        
        slow_config = self.api_groups['indicators_slow']
        
        if 'ADX' not in slow_config.get('apis', []):
            print("  ADX not configured in indicators_slow group")
            return
        
        print("\n  Scheduling ADX (trend strength) jobs:")
        
        # Tier A: Every 15 minutes
        for symbol in self.tiers['tier_a']['symbols']:
            if symbol:
                job_name = f"ADX_{symbol}_A"
                self.scheduler.add_job(
                    func=self._fetch_adx,
                    trigger='interval',
                    args=[symbol],
                    seconds=slow_config.get('tier_a_interval', 900),
                    id=job_name,
                    name=job_name,
                    replace_existing=True,
                    max_instances=1
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier A: {len(self.tiers['tier_a']['symbols'])} symbols every {slow_config.get('tier_a_interval', 900)}s")
        
        # Tier B: Every 30 minutes
        for symbol in self.tiers['tier_b']['symbols']:
            if symbol:
                job_name = f"ADX_{symbol}_B"
                self.scheduler.add_job(
                    func=self._fetch_adx,
                    trigger='interval',
                    args=[symbol],
                    seconds=slow_config.get('tier_b_interval', 1800),
                    id=job_name,
                    name=job_name,
                    replace_existing=True,
                    max_instances=1
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier B: {len(self.tiers['tier_b']['symbols'])} symbols every {slow_config.get('tier_b_interval', 1800)}s")
        
        # Tier C: Every 60 minutes
        for symbol in self.tiers['tier_c']['symbols']:
            if symbol:
                job_name = f"ADX_{symbol}_C"
                self.scheduler.add_job(
                    func=self._fetch_adx,
                    trigger='interval',
                    args=[symbol],
                    seconds=slow_config.get('tier_c_interval', 3600),
                    id=job_name,
                    name=job_name,
                    replace_existing=True,
                    max_instances=1
                )
                self.jobs_created += 1
        
        print(f"    ✓ Tier C: {len(self.tiers['tier_c']['symbols'])} symbols every {slow_config.get('tier_c_interval', 3600)}s")



    def _fetch_realtime_options(self, symbol):
        """
        Fetch realtime options data
        Called by scheduler for each symbol
        """
        try:
            # Check if market is open (basic check for now)
            if not self._is_market_hours():
                print(f"Skipping {symbol} - market closed")
                return
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Show test mode indicator
            mode_indicator = "🧪" if self.test_mode else "📊"
            print(f"[{timestamp}] {mode_indicator} Fetching realtime options for {symbol}")
            
            # Initialize clients
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            # Fetch data (cache-aware automatically)
            data = av_client.get_realtime_options(symbol)
            
            if data and 'data' in data:
                # Ingest into database (also updates cache)
                records = ingestion.ingest_options_data(data, symbol)
                print(f"  ✓ {symbol}: {records} records processed")
            else:
                print(f"  ⚠ {symbol}: No data received")
                
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
    
    def _fetch_historical_options(self, symbol):
        """
        Fetch historical options data
        Called by scheduler once per day
        """
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching historical options for {symbol}")
            
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            data = av_client.get_historical_options(symbol)
            
            if data and 'data' in data:
                records = ingestion.ingest_historical_options(data, symbol)
                print(f"  ✓ {symbol}: {records} historical records processed")
                
        except Exception as e:
            print(f"  ✗ Error fetching historical {symbol}: {e}")
    
    def _fetch_rsi(self, symbol, interval='1min', time_period=14):
        """
        Fetch RSI indicator data
        Called by scheduler for each symbol
        Phase 5.1: Technical indicator scheduling
        """
        logger.error(f"!!! _fetch_rsi START: symbol={symbol}, interval={interval}, time_period={time_period}")

        try:
            # Check if market is open
            if not self._is_market_hours():
                print(f"Skipping RSI for {symbol} - market closed")
                return
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            mode_indicator = "🧪" if self.test_mode else "📊"
            
            print(f"[{timestamp}] {mode_indicator} Fetching RSI for {symbol}")
            
            # Initialize clients
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            # Fetch RSI data (cache-aware automatically)
            data = av_client.get_rsi(symbol, interval=interval, time_period=time_period)
            
            if data and 'Technical Analysis: RSI' in data:
                # Ingest into database
                records = ingestion.ingest_rsi_data(data, symbol, interval, time_period)
                print(f"  ✓ {symbol} RSI: {records} records processed")
            else:
                print(f"  ⚠ {symbol} RSI: No data received")
                
        except Exception as e:
            print(f"  ✗ Error fetching RSI for {symbol}: {e}")

    def _fetch_macd(self, symbol, interval, fastperiod, slowperiod, signalperiod, series_type):
        """
        Fetch MACD indicator data
        Called by scheduler for each symbol
        Phase 5.2: Technical indicator scheduling
        """
        try:
            # Check if market is open
            if not self._is_market_hours():
                print(f"Skipping MACD for {symbol} - market closed")
                return
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            mode_indicator = "🧪" if self.test_mode else "📊"
            
            print(f"[{timestamp}] {mode_indicator} Fetching MACD for {symbol}")
            
            # Initialize clients
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            # Fetch MACD data (cache-aware automatically)
            data = av_client.get_macd(
                symbol, 
                interval=interval, 
                fastperiod=fastperiod,
                slowperiod=slowperiod,
                signalperiod=signalperiod,
                series_type=series_type
            )
            
            if data and 'Technical Analysis: MACD' in data:
                # Ingest into database with ALL parameters
                records = ingestion.ingest_macd_data(
                    data, 
                    symbol, 
                    interval, 
                    fastperiod, 
                    slowperiod, 
                    signalperiod, 
                    series_type
                )
                print(f"  ✓ {symbol} MACD: {records} records processed")
            else:
                print(f"  ⚠ {symbol} MACD: No data received")
                
        except Exception as e:
            print(f"  ✗ Error fetching MACD for {symbol}: {e}")

    def _fetch_bbands(self, symbol, interval, time_period, series_type,
                    nbdevup, nbdevdn, matype):
        """
        Fetch Bollinger Bands indicator data
        Called by scheduler for each symbol
        Phase 5.3: Technical indicator scheduling
        """
        try:
            # Check if market is open
            if not self._is_market_hours():
                print(f"Skipping BBANDS for {symbol} - market closed")
                return
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            mode_indicator = "🧪" if self.test_mode else "📊"
            
            print(f"[{timestamp}] {mode_indicator} Fetching BBANDS for {symbol}")
            
            # Initialize clients
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            # Fetch BBANDS data (cache-aware automatically)
            data = av_client.get_bbands(
                symbol, 
                interval,
                time_period,
                series_type,
                nbdevup,
                nbdevdn,
                matype
            )
            
            if data and 'Technical Analysis: BBANDS' in data:
                # Ingest into database with ALL parameters
                records = ingestion.ingest_bbands_data(
                    data, 
                    symbol, 
                    interval,
                    time_period,
                    nbdevup,
                    nbdevdn,
                    matype,
                    series_type
                )
                print(f"  ✓ {symbol} BBANDS: {records} records processed")
            else:
                print(f"  ⚠ {symbol} BBANDS: No data received")
                
        except Exception as e:
            print(f"  ✗ Error fetching BBANDS for {symbol}: {e}")

    def _fetch_vwap(self, symbol, interval):
        """Fetch and ingest VWAP data for a symbol"""
        try:
            # Check if market is open
            if not self._is_market_hours():
                print(f"Skipping VWAP for {symbol} - market closed")
                return
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            mode_indicator = "🧪" if self.test_mode else "📊"
            
            print(f"[{timestamp}] {mode_indicator} Fetching VWAP for {symbol}")
            
            # Initialize clients - LIKE ALL OTHER FETCH METHODS
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            # Fetch VWAP data
            vwap_data = av_client.get_vwap(symbol, interval)
            
            if vwap_data and 'Technical Analysis: VWAP' in vwap_data:
                # Ingest into database
                records = ingestion.ingest_vwap_data(
                    vwap_data,
                    symbol,
                    interval
                )
                print(f"  ✓ {symbol} VWAP: {records} records processed")
                return records
            else:
                print(f"  ⚠ {symbol} VWAP: No data received")
                return 0
                
        except Exception as e:
            print(f"  ✗ Error fetching VWAP for {symbol}: {e}")
            return 0

    def _fetch_atr(self, symbol, interval=None, time_period=None):
            """
            Fetch ATR (Average True Range) indicator data
            Called by scheduler for each symbol
            Phase 5.5: Volatility indicator scheduling
            
            ATR is different - it's daily data, so less frequent updates needed
            """
            if not self.test_mode and not self._is_market_hours():
                print(f"Skipping ATR for {symbol} - market closed")
                return
            
            # Get config defaults if not provided - NO HARDCODING!
            if interval is None:
                interval = self.config_manager.av_config['endpoints']['atr']['default_params']['interval']
            if time_period is None:
                time_period = self.config_manager.av_config['endpoints']['atr']['default_params']['time_period']
            
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            # Fetch ATR data
            data = av_client.get_atr(symbol, interval, time_period)
            
            if data and 'Technical Analysis: ATR' in data:
                # Ingest the data
                records = ingestion.ingest_atr_data(data, symbol, interval, time_period)
                print(f"  ✓ {symbol} ATR: {records} records processed (daily volatility)")
            else:
                print(f"  ⚠️ No ATR data for {symbol}")

    def _fetch_adx(self, symbol: str) -> None:
        """
        Fetch and ingest ADX data for a symbol - Phase 5.6
        
        ADX measures trend strength on a 0-100 scale.
        Runs on slower intervals since trend strength changes gradually.
        """
        try:
            # Check market hours (like other methods)
            if not self.test_mode and not self._is_market_hours():
                print(f"Skipping ADX for {symbol} - market closed")
                return
                
            # Initialize clients locally (like ALL other fetch methods)
            av_client = AlphaVantageClient()
            ingestion = DataIngestion()
            
            # Get ADX config - use config_manager, not config
            adx_config = self.config_manager.av_config['endpoints'].get('adx', {})
            interval = adx_config['default_params'].get('interval', '5min')
            time_period = adx_config['default_params'].get('time_period', 14)
            
            # Fetch data using local av_client
            adx_data = av_client.get_adx(
                symbol=symbol,
                interval=interval,
                time_period=time_period
            )
            
            if adx_data and 'Technical Analysis: ADX' in adx_data:
                # Ingest using local ingestion
                records = ingestion.ingest_adx_data(
                    adx_data, 
                    symbol,
                    interval=interval,
                    time_period=time_period
                )
                print(f"  ✅ ADX {symbol}: {records} records updated (trend strength)")
            else:
                print(f"  ⚠️ No ADX data received for {symbol}")
                
        except Exception as e:
            print(f"  ❌ Error fetching ADX for {symbol}: {e}")
            
    def _is_market_hours(self):
            """Check if current time is during market hours"""
            # Override for testing
            if self.test_mode:
                return True
                
            now = datetime.now(self.timezone)
            current_time = now.time()
            
            # Parse market hours
            market_open = datetime.strptime(self.market_hours['market_open'], '%H:%M').time()
            market_close = datetime.strptime(self.market_hours['market_close'], '%H:%M').time()
            
            # Check if weekend
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if within market hours
            return market_open <= current_time <= market_close
    
    def get_status(self):
        """Get scheduler status"""
        jobs = self.scheduler.get_jobs()
        
        status = {
            'running': self.is_running,
            'total_jobs': len(jobs),
            'is_market_hours': self._is_market_hours(),
            'jobs': []
        }
        
        for job in jobs:
            status['jobs'].append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.strftime('%Y-%m-%d %H:%M:%S') if job.next_run_time else 'Paused'
            })
        
        return status