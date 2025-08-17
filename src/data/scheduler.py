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

class DataScheduler:
    """
    Manages automated data collection with market awareness and rate limit respect
    Phase 4.2 - Day 16
    """
    
    def __init__(self, test_mode=False):
        # Load configurations
        self._load_config()
        
        # Initialize scheduler
        self._init_scheduler()

        # Initialize ConfigManager
        self.config = ConfigManager()
        
        # Test mode flag
        self.test_mode = test_mode
        
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
            self.config = yaml.safe_load(f)
        
        # Extract key configurations
        self.market_hours = self.config['market_hours']
        self.timezone = pytz.timezone(self.market_hours['timezone'])
        self.rate_limit_config = self.config['rate_limit_budget']
        self.tiers = self.config['symbol_tiers']
        self.api_groups = self.config['api_groups']
        self.rules = self.config['scheduling_rules']
        self.moc_config = self.config['moc_window']
        
    def _init_scheduler(self):
        """Initialize APScheduler with proper configuration"""
        # Configure executors and job defaults
        executors = {
            'default': ThreadPoolExecutor(
                max_workers=self.config['scheduler']['max_workers']
            )
        }
        
        job_defaults = self.config['scheduler']['job_defaults']
        
        # Create scheduler
        self.scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone=self.timezone
        )
    
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
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            print("✓ Scheduler stopped")
        
    def _create_jobs(self):
            """Create scheduled jobs based on configuration"""
            print(f"Creating scheduled jobs...")
            
            # Schedule realtime options for critical API group
            if 'critical' in self.api_groups:
                self._schedule_realtime_options()
            
            # Schedule RSI indicators (NEW - Phase 5.1)
            if 'indicators_fast' in self.api_groups:
                self._schedule_rsi_indicators()
                self._schedule_macd_indicators()
                self._schedule_bbands_indicators() 
            
            # Schedule daily historical options
            if 'daily' in self.api_groups:
                self._schedule_historical_options()
            
            print(f"  Created {self.jobs_created} jobs")
            
            # List all jobs
            jobs = self.scheduler.get_jobs()
            for job in jobs[:10]:  # Show first 10 jobs
                print(f"    - {job.id}: {job.trigger}")
                    
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
        macd_config = self.config.av_config['endpoints']['macd']['default_params']
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
        bbands_config = self.config.av_config['endpoints']['bbands']['default_params']
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
        vwap_config = self.config.av_config['endpoints']['vwap']['default_params']
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
        # Fetch data using the passed interval
        vwap_data = self.av_client.get_vwap(symbol, interval)
        
        if vwap_data and 'Technical Analysis: VWAP' in vwap_data:
            # Ingest into database
            records = self.ingestion.ingest_vwap_data(
                vwap_data,
                symbol,
                interval
            )
            print(f"[VWAP] {symbol}: {records} records ingested")
            return records
        else:
            print(f"[VWAP] {symbol}: No data received")
            return 0


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