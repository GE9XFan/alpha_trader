"""
Data Scheduler
Orchestrates all API calls based on tier priorities
Full implementation with configuration-driven logic
"""

import threading
import heapq
import queue
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from datetime import datetime, timedelta, time as datetime_time
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.foundation.base_module import BaseModule
from src.foundation.config_manager import ConfigManager
from src.data.rate_limiter import TokenBucketRateLimiter, RequestPriority

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RATE_LIMITED = "RATE_LIMITED"


class APIType(Enum):
    """API call types directly from schedules.yaml data_collection"""
    OPTIONS_WITH_GREEKS = "options_with_greeks"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bbands"
    VWAP = "vwap"
    ATR = "atr"
    ADX = "adx"
    INDICATORS_BUNDLE = "indicators_bundle"
    # Note: IBKR bars and quotes are subscriptions, not scheduled API calls


@dataclass
class ScheduledTask:
    """Represents a scheduled API call task for Alpha Vantage"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    api_type: APIType = APIType.OPTIONS_WITH_GREEKS
    priority: int = 5
    interval_seconds: int = 60
    next_run: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    tier: str = "other"
    callback: Optional[Callable] = None
    error_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Priority queue ordering: earlier time and higher priority first"""
        if self.next_run != other.next_run:
            return self.next_run < other.next_run
        return self.priority < other.priority


class DataScheduler(BaseModule):
    """
    Manages scheduling of Alpha Vantage API calls.
    Respects tier priorities and rate limits.
    Configuration-driven from YAML files.
    
    Note: IBKR uses persistent subscriptions managed by IBKRConnectionManager,
    not scheduled API calls, so this scheduler is for Alpha Vantage only.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None,
                 av_client=None, ibkr_connection=None, rate_limiter=None):
        """
        Initialize scheduler from configuration
        
        Args:
            config_manager: ConfigManager instance (creates new if None)
            av_client: AlphaVantageClient instance
            ibkr_connection: IBKRConnectionManager instance
            rate_limiter: TokenBucketRateLimiter instance
        """
        # Get configuration manager
        self.config_manager = config_manager or ConfigManager()
        
        # Load scheduler configuration
        scheduler_config = self.config_manager.get('data.schedules', {})
        
        # Initialize base module
        super().__init__(scheduler_config, "DataScheduler")
        
        # Store API client references
        self.av_client = av_client
        self.ibkr_connection = ibkr_connection
        self.rate_limiter = rate_limiter
        
        # Load all configuration sections
        self._load_configuration()
        
        # Initialize scheduler state
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_queue: List[ScheduledTask] = []  # Min heap
        self.active_tasks: Set[str] = set()
        self.task_lock = threading.Lock()
        
        # Task execution queue
        self.execution_queue = queue.PriorityQueue(maxsize=1000)
        
        # Scheduler threads
        self.scheduler_thread = None
        self.executor_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Statistics
        self.statistics = {
            'tasks_scheduled': 0,
            'tasks_executed': 0,
            'tasks_failed': 0,
            'tasks_rate_limited': 0,
            'tasks_dropped': 0,
            'moc_window_activations': 0,
            'last_execution_time': None,
            'average_execution_time': 0,
            'tasks_by_tier': {'tier_a': 0, 'tier_b': 0, 'tier_c': 0, 'other': 0},
            'tasks_by_type': {},
            'tasks_by_status': {status.value: 0 for status in TaskStatus}
        }
        
        logger.info(f"DataScheduler initialized with {len(self.tier_configs)} tier configurations")
    
    def _load_configuration(self) -> None:
        """Load all configuration from YAML files"""
        # Market hours from schedules.yaml
        schedules = self.config_manager.get('data.schedules.schedules', {})
        market_hours = schedules.get('market_hours', {})
        
        self.market_open = self._parse_time(market_hours.get('market_open', '09:30'))
        self.market_close = self._parse_time(market_hours.get('market_close', '16:00'))
        self.pre_market_start = self._parse_time(market_hours.get('pre_market_start', '04:00'))
        self.after_hours_end = self._parse_time(market_hours.get('after_hours_end', '20:00'))
        self.timezone = market_hours.get('timezone', 'America/New_York')
        
        # MOC window configuration
        moc_config = schedules.get('moc_window', {})
        self.moc_enabled = moc_config.get('enabled', True)
        self.moc_start = self._parse_time(moc_config.get('start_time', '15:40'))
        self.moc_end = self._parse_time(moc_config.get('end_time', '15:55'))
        self.moc_priority_boost = moc_config.get('priority_boost', 10)
        self.moc_update_interval = moc_config.get('update_interval', 5)
        
        # Data collection schedules by tier
        self.tier_configs = schedules.get('data_collection', {})
        
        # Symbol tiers from symbols.yaml
        symbols_config = self.config_manager.get('data.symbols.symbols', {})
        self.tier_a_symbols = set(symbols_config.get('tier_a', {}).get('symbols', []))
        self.tier_b_symbols = set(symbols_config.get('tier_b', {}).get('symbols', []))
        self.tier_c_symbols = set(symbols_config.get('tier_c', {}).get('symbols', []))
        self.tier_c_max = symbols_config.get('tier_c', {}).get('max_symbols', 20)
        
        # Tier priorities
        self.tier_priorities = {
            'tier_a': symbols_config.get('tier_a', {}).get('priority', 1),
            'tier_b': symbols_config.get('tier_b', {}).get('priority', 2),
            'tier_c': symbols_config.get('tier_c', {}).get('priority', 3),
        }
        
        # Ingestion configuration
        ingestion_config = self.config_manager.get('data.ingestion.ingestion', {})
        self.max_retries = ingestion_config.get('max_retries', 3)
        self.retry_delay = ingestion_config.get('retry_delay', 5)
        
        # Rate limiting configuration
        rate_config = self.config_manager.get('apis.rate_limits.rate_limits.alpha_vantage', {})
        self.rate_limit_warning = rate_config.get('warning_threshold', 450)
        self.rate_limit_max = rate_config.get('calls_per_minute', 600)
    
    def _parse_time(self, time_str: str) -> datetime_time:
        """Parse time string HH:MM to datetime.time"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return datetime_time(hour, minute)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid time format: {time_str}, using 00:00")
            return datetime_time(0, 0)
    
    def initialize(self) -> bool:
        """
        Initialize scheduler and load initial tasks
        Required by BaseModule
        """
        try:
            logger.info("Initializing DataScheduler...")
            
            # Create initial tasks for all configured symbols
            self._create_initial_tasks()
            
            # Start scheduler threads
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.executor_thread = threading.Thread(target=self._executor_loop, daemon=True)
            
            self.scheduler_thread.start()
            self.executor_thread.start()
            
            self.is_initialized = True
            logger.info(f"DataScheduler initialized with {len(self.tasks)} tasks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DataScheduler: {e}")
            return False
    
    def _create_initial_tasks(self) -> None:
        """Create initial tasks for all configured symbols and API types"""
        # Tier A symbols
        for symbol in self.tier_a_symbols:
            self._create_tasks_for_symbol(symbol, 'tier_a')
        
        # Tier B symbols
        for symbol in self.tier_b_symbols:
            self._create_tasks_for_symbol(symbol, 'tier_b')
        
        # Tier C symbols (if any configured)
        for symbol in self.tier_c_symbols:
            self._create_tasks_for_symbol(symbol, 'tier_c')
        
        logger.info(f"Created {len(self.tasks)} initial tasks")
    
    def _create_tasks_for_symbol(self, symbol: str, tier: str) -> None:
        """Create all API tasks for a symbol based on tier configuration"""
        tier_config = self.tier_configs.get(tier, {})
        if not tier_config:
            logger.warning(f"No configuration found for tier: {tier}")
            return
            
        base_priority = self.tier_priorities.get(tier, None)
        if base_priority is None:
            logger.warning(f"No priority configured for tier: {tier}")
            return
        
        for api_name, api_config in tier_config.items():
            if isinstance(api_config, dict):
                try:
                    api_type = APIType(api_name)
                    
                    # Everything comes from config, no defaults
                    interval = api_config.get('interval_seconds')
                    priority = api_config.get('priority')
                    
                    if interval is None:
                        logger.error(f"No interval_seconds configured for {api_name} in {tier}")
                        continue
                    
                    if priority is None:
                        logger.error(f"No priority configured for {api_name} in {tier}")
                        continue
                    
                    task = ScheduledTask(
                        symbol=symbol,
                        api_type=api_type,
                        priority=priority,
                        interval_seconds=interval,
                        tier=tier,
                        next_run=datetime.now() + timedelta(seconds=interval)
                    )
                    
                    self.add_task(task)
                    
                except ValueError:
                    logger.warning(f"Unknown API type in configuration: {api_name}")

    
    def add_task(self, task: ScheduledTask) -> bool:
        """
        Add a scheduled task
        
        Args:
            task: ScheduledTask to add
            
        Returns:
            True if task added successfully
        """
        with self.task_lock:
            # Check if task already exists
            existing_key = f"{task.symbol}_{task.api_type.value}"
            if existing_key in self.tasks:
                logger.debug(f"Task already exists: {existing_key}")
                return False
            
            # Add to tasks dictionary
            self.tasks[task.task_id] = task
            
            # Add to priority queue
            heapq.heappush(self.task_queue, task)
            
            # Update statistics
            self.statistics['tasks_scheduled'] += 1
            self.statistics['tasks_by_tier'][task.tier] += 1
            self.statistics['tasks_by_type'][task.api_type.value] = \
                self.statistics['tasks_by_type'].get(task.api_type.value, 0) + 1
            
            logger.debug(f"Added task: {task.symbol} - {task.api_type.value}")
            return True
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a scheduled task
        
        Args:
            task_id: ID of task to remove
            
        Returns:
            True if task removed successfully
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.CANCELLED
            del self.tasks[task_id]
            
            # Note: Task remains in heap but will be skipped when processed
            logger.debug(f"Removed task: {task_id}")
            return True
    
    def update_priority(self, symbol: str, priority: int) -> bool:
        """
        Update priority for all tasks of a symbol
        
        Args:
            symbol: Symbol to update
            priority: New priority level
            
        Returns:
            True if any tasks updated
        """
        with self.task_lock:
            updated = False
            for task in self.tasks.values():
                if task.symbol == symbol:
                    task.priority = priority
                    updated = True
            
            if updated:
                # Rebuild heap with new priorities
                self.task_queue = list(self.tasks.values())
                heapq.heapify(self.task_queue)
                logger.info(f"Updated priority for {symbol} to {priority}")
            
            return updated
    
    def handle_moc_window(self) -> None:
        """
        Special handling for MOC window (3:40-3:55 PM)
        Elevates priority for relevant symbols
        """
        if not self.moc_enabled or not self._is_moc_window():
            return
        
        with self.task_lock:
            logger.info("MOC window activated - elevating priorities")
            self.statistics['moc_window_activations'] += 1
            
            # Elevate Tier A symbols during MOC
            for task in self.tasks.values():
                if task.tier == 'tier_a':
                    # Boost priority
                    task.priority = max(1, task.priority - self.moc_priority_boost)
                    
                    # Reduce interval for more frequent updates
                    if task.api_type == APIType.OPTIONS_WITH_GREEKS:
                        task.interval_seconds = self.moc_update_interval
            
            # Rebuild heap with new priorities
            self.task_queue = list(self.tasks.values())
            heapq.heapify(self.task_queue)
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        current_time = now.time()
        
        # Check weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Weekend
            return False
        
        return self.market_open <= current_time <= self.market_close
    
    def _is_moc_window(self) -> bool:
        """Check if currently in MOC window"""
        if not self.moc_enabled:
            return False
        
        now = datetime.now()
        current_time = now.time()
        
        # Must be a weekday and within MOC window
        if now.weekday() >= 5:
            return False
        
        return self.moc_start <= current_time <= self.moc_end
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop - determines what tasks to run when"""
        logger.info("Scheduler loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check MOC window
                self.handle_moc_window()
                
                # Process task queue
                with self.task_lock:
                    now = datetime.now()
                    tasks_to_run = []
                    
                    # Get all tasks that are due
                    while self.task_queue and self.task_queue[0].next_run <= now:
                        task = heapq.heappop(self.task_queue)
                        
                        # Skip cancelled tasks
                        if task.status == TaskStatus.CANCELLED:
                            continue
                        
                        # Skip if task no longer exists
                        if task.task_id not in self.tasks:
                            continue
                        
                        # Check rate limiting for Alpha Vantage
                        if self.rate_limiter:
                            priority = self._get_request_priority(task)
                            # Check if tokens are available
                            available_tokens = self.rate_limiter.get_available_tokens()
                            if available_tokens < 1:
                                # Rate limited - reschedule for later
                                task.next_run = now + timedelta(seconds=5)
                                task.status = TaskStatus.RATE_LIMITED
                                heapq.heappush(self.task_queue, task)
                                self.statistics['tasks_rate_limited'] += 1
                                continue
                        
                        # Mark as ready to run
                        task.status = TaskStatus.PENDING
                        tasks_to_run.append(task)
                        self.active_tasks.add(task.task_id)
                    
                    # Add tasks to executor queue
                    for task in tasks_to_run:
                        self._queue_for_execution(task)
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(1)
        
        logger.info("Scheduler loop stopped")
    
    def _executor_loop(self) -> None:
        """Executor loop - executes API calls from the queue"""
        logger.info("Executor loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Get next task from execution queue
                task = self._get_next_execution_task()
                
                if task:
                    self._execute_task(task)
                    
            except Exception as e:
                logger.error(f"Executor loop error: {e}")
                time.sleep(1)
        
        logger.info("Executor loop stopped")
    
    def _queue_for_execution(self, task: ScheduledTask) -> None:
        """Add task to execution queue"""
        try:
            # Priority queue item: (priority, timestamp, task)
            # Lower priority number = higher priority
            self.execution_queue.put_nowait((task.priority, time.time(), task))
            task.status = TaskStatus.PENDING
        except queue.Full:
            logger.error(f"Execution queue full, dropping task: {task.symbol} - {task.api_type.value}")
            self.statistics['tasks_dropped'] = self.statistics.get('tasks_dropped', 0) + 1
    
    def _get_next_execution_task(self) -> Optional[ScheduledTask]:
        """Get next task from execution queue"""
        try:
            # Block for up to 0.1 seconds waiting for a task
            priority, timestamp, task = self.execution_queue.get(timeout=0.1)
            task.status = TaskStatus.RUNNING
            return task
        except queue.Empty:
            return None
    
    def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task"""
        start_time = time.perf_counter()
        
        try:
            logger.debug(f"Executing task: {task.symbol} - {task.api_type.value}")
            
            # Execute based on provider and API type
            if task.callback:
                # Use custom callback if provided
                result = task.callback(task)
            else:
                # Use default execution based on API type
                result = self._execute_api_call(task)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.last_run = datetime.now()
            task.error_count = 0
            
            # Schedule next run
            task.next_run = datetime.now() + timedelta(seconds=task.interval_seconds)
            
            # Update statistics
            self.statistics['tasks_executed'] += 1
            execution_time = time.perf_counter() - start_time
            self._update_execution_stats(execution_time)
            
            logger.debug(f"Task completed: {task.symbol} - {task.api_type.value} ({execution_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Task execution failed: {task.symbol} - {task.api_type.value}: {e}")
            task.status = TaskStatus.FAILED
            task.error_count += 1
            self.statistics['tasks_failed'] += 1
            
            # Exponential backoff for failed tasks
            backoff = min(300, task.interval_seconds * (2 ** task.error_count))
            task.next_run = datetime.now() + timedelta(seconds=backoff)
            
        finally:
            # Remove from active tasks and re-add to queue
            with self.task_lock:
                if task.task_id in self.active_tasks:
                    self.active_tasks.remove(task.task_id)
                
                # Re-add to queue for next run
                if task.task_id in self.tasks:
                    heapq.heappush(self.task_queue, task)
    
    def _execute_api_call(self, task: ScheduledTask) -> Any:
        """
        Execute the actual API call through Alpha Vantage client
        Note: IBKR uses persistent subscriptions, not scheduled API calls
        """
        result = None
        
        try:
            if self.av_client:
                # IMPORTANT: The av_client methods already handle rate limiting via @rate_limit decorator
                # But we need to ensure tokens are actually consumed
                
                # Map API type to Alpha Vantage client method
                if task.api_type == APIType.OPTIONS_WITH_GREEKS:
                    result = self.av_client.get_realtime_options(task.symbol)
                elif task.api_type == APIType.RSI:
                    result = self.av_client.get_rsi(task.symbol)
                elif task.api_type == APIType.MACD:
                    result = self.av_client.get_macd(task.symbol)
                elif task.api_type == APIType.BBANDS:
                    result = self.av_client.get_bbands(task.symbol)
                elif task.api_type == APIType.VWAP:
                    result = self.av_client.get_vwap(task.symbol)
                elif task.api_type == APIType.ATR:
                    result = self.av_client.get_atr(task.symbol)
                elif task.api_type == APIType.ADX:
                    result = self.av_client.get_adx(task.symbol)
                elif task.api_type == APIType.INDICATORS_BUNDLE:
                    # Bundle multiple indicators in one call if supported
                    # Otherwise make individual calls
                    results = {}
                    for indicator in ['RSI', 'MACD', 'BBANDS']:
                        method = getattr(self.av_client, f'get_{indicator.lower()}', None)
                        if method:
                            results[indicator] = method(task.symbol)
                    result = results
            else:
                logger.warning("AlphaVantageClient not configured")
                    
        except Exception as e:
            logger.error(f"API call failed for {task.symbol} - {task.api_type.value}: {e}")
            raise
        
        return result
    
    def _get_request_priority(self, task: ScheduledTask) -> RequestPriority:
        """Map task to request priority for rate limiter"""
        if self._is_moc_window() and task.tier == 'tier_a':
            return RequestPriority.MOC_WINDOW
        
        priority_map = {
            'tier_a': RequestPriority.TIER_A,
            'tier_b': RequestPriority.TIER_B,
            'tier_c': RequestPriority.TIER_C,
        }
        
        return priority_map.get(task.tier, RequestPriority.BACKGROUND)
    
    def _update_execution_stats(self, execution_time: float) -> None:
        """Update execution statistics"""
        self.statistics['last_execution_time'] = execution_time
        
        # Update rolling average execution time
        total_executed = self.statistics['tasks_executed']
        current_avg = self.statistics['average_execution_time']
        
        # Weighted moving average for execution time
        alpha = 0.1
        self.statistics['average_execution_time'] = \
            (1 - alpha) * current_avg + alpha * execution_time
    
    def get_next_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get next tasks to execute (in actual priority order)
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of upcoming tasks in execution order
        """
        with self.task_lock:
            upcoming = []
            
            # Get all tasks that are ready to run
            now = datetime.now()
            ready_tasks = []
            
            for task in self.tasks.values():
                if task.next_run <= now and task.status != TaskStatus.CANCELLED:
                    ready_tasks.append(task)
            
            # Sort by priority (lower number = higher priority = executes first)
            ready_tasks.sort(key=lambda t: (t.priority, t.next_run))
            
            # Return the first 'limit' tasks
            for task in ready_tasks[:limit]:
                upcoming.append({
                    'task_id': task.task_id,
                    'symbol': task.symbol,
                    'api_type': task.api_type.value,
                    'tier': task.tier,
                    'priority': task.priority,
                    'next_run': task.next_run.isoformat(),
                    'status': task.status.value
                })
            
            return upcoming
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'symbol': task.symbol,
                    'api_type': task.api_type.value,
                    'status': task.status.value,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'next_run': task.next_run.isoformat(),
                    'error_count': task.error_count
                }
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check scheduler health
        Required by BaseModule
        """
        with self.task_lock:
            total_tasks = len(self.tasks)
            active_count = len(self.active_tasks)
            pending_count = sum(1 for t in self.tasks.values() 
                              if t.status == TaskStatus.PENDING)
            failed_count = sum(1 for t in self.tasks.values() 
                             if t.status == TaskStatus.FAILED)
        
        health = {
            'healthy': self.is_running and self.is_initialized,
            'scheduler_running': self.is_running,
            'scheduler_thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            'executor_thread_alive': self.executor_thread.is_alive() if self.executor_thread else False,
            'total_tasks': total_tasks,
            'active_tasks': active_count,
            'pending_tasks': pending_count,
            'failed_tasks': failed_count,
            'market_hours': self._is_market_hours(),
            'moc_window': self._is_moc_window(),
            'statistics': self.statistics.copy(),
            'checks': {
                'threads_running': (
                    self.scheduler_thread and self.scheduler_thread.is_alive() and
                    self.executor_thread and self.executor_thread.is_alive()
                ),
                'tasks_configured': total_tasks > 0,
                'no_stuck_tasks': active_count < 10,
                'error_rate_ok': (
                    self.statistics['tasks_failed'] / max(1, self.statistics['tasks_executed']) < 0.1
                    if self.statistics['tasks_executed'] > 0 else True
                )
            }
        }
        
        health['healthy'] = all(health['checks'].values())
        
        return health
    
    def shutdown(self) -> bool:
        """
        Shutdown scheduler
        Required by BaseModule
        """
        logger.info("Shutting down DataScheduler...")
        
        try:
            # Signal threads to stop
            self.is_running = False
            self.stop_event.set()
            
            # Wait for threads to finish
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5)
            
            if self.executor_thread:
                self.executor_thread.join(timeout=5)
            
            # Clear queues
            with self.task_lock:
                self.task_queue.clear()
                self.active_tasks.clear()
                
                # Mark all tasks as cancelled
                for task in self.tasks.values():
                    if task.status == TaskStatus.RUNNING:
                        task.status = TaskStatus.CANCELLED
            
            # Clear execution queue
            while not self.execution_queue.empty():
                try:
                    self.execution_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.is_initialized = False
            logger.info("DataScheduler shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed scheduler statistics"""
        with self.task_lock:
            stats = self.statistics.copy()
            
            # Add current state
            stats['current_state'] = {
                'total_tasks': len(self.tasks),
                'queue_size': len(self.task_queue),
                'active_tasks': len(self.active_tasks),
                'is_running': self.is_running,
                'market_hours': self._is_market_hours(),
                'moc_window': self._is_moc_window()
            }
            
            # Task distribution
            stats['task_distribution'] = {
                'by_status': {},
                'by_tier': {},
                'by_api_type': {}
            }
            
            for task in self.tasks.values():
                # By status
                status = task.status.value
                stats['task_distribution']['by_status'][status] = \
                    stats['task_distribution']['by_status'].get(status, 0) + 1
                
                # By tier
                stats['task_distribution']['by_tier'][task.tier] = \
                    stats['task_distribution']['by_tier'].get(task.tier, 0) + 1
                
                # By API type
                api_type = task.api_type.value
                stats['task_distribution']['by_api_type'][api_type] = \
                    stats['task_distribution']['by_api_type'].get(api_type, 0) + 1
            
            return stats