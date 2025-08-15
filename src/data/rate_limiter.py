"""
Token bucket rate limiter for API calls.
Fully integrated with project configuration system.
"""

import time
import threading
import queue
import logging
import math
from typing import Optional, Dict, Any, Tuple, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as datetime_time
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.foundation.config_manager import ConfigManager
from src.foundation.base_module import BaseModule


class RequestPriority(Enum):
    """Priority levels mapped from configuration."""
    EMERGENCY = 0        # System critical
    POSITION_MONITOR = 1 # Active positions
    TIER_A = 2          # From config/data/symbols.yaml
    MOC_WINDOW = 2      # From config/data/schedules.yaml (elevated)
    TIER_B = 3          # From config/data/symbols.yaml
    TIER_C = 4          # From config/data/symbols.yaml
    BACKGROUND = 5      # Historical/research


class AlertLevel(Enum):
    """Alert levels from config/monitoring/alerts.yaml"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class RateLimitExceeded(Exception):
    """Raised when hard rate limit is exceeded."""
    pass


class TokensExhausted(Exception):
    """Raised when no tokens available and non-blocking."""
    pass


class QueueOverflow(Exception):
    """Raised when priority queue is full."""
    pass


@dataclass
class QueuedRequest:
    """Request waiting in priority queue."""
    priority: int
    timestamp: float
    tokens: int
    event: threading.Event
    timeout: float
    symbol: Optional[str] = None
    api_type: Optional[str] = None
    
    def __lt__(self, other):
        """Lower priority value = higher priority."""
        return self.priority < other.priority


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.
    Fully configuration-driven from YAML files.
    """
    
    def __init__(self, provider: str = 'alpha_vantage', 
                 config_manager: Optional[ConfigManager] = None):
        """
        Initialize rate limiter from configuration files.
        
        Args:
            provider: API provider ('alpha_vantage' or 'ibkr')
            config_manager: ConfigManager instance (creates new if None)
        """
        super().__init__()
        
        self.provider = provider
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Load all relevant configurations
        self._load_configurations()
        
        # Initialize token bucket
        self.tokens = float(self.burst_capacity)
        self.last_refill = time.perf_counter()
        self.lock = threading.RLock()
        
        # Priority queue for waiting requests
        self.wait_queue = queue.PriorityQueue(maxsize=self.queue_config.get('max_size', 100))
        self.queue_processor = None
        self.running = True
        
        # Statistics tracking
        self._initialize_statistics()
        
        # Rolling window for RPM calculation
        self.request_timestamps = deque()
        self.minute_window_size = 60.0
        
        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reset_time = None
        
        # Alert management
        self.last_alert_time = {}
        self.alert_counts = defaultdict(int)
        
        # Start queue processor thread
        self._start_queue_processor()
        
        self.logger.info(
            f"Rate limiter initialized for {provider}: "
            f"{self.tokens_per_second} tokens/sec, "
            f"capacity: {self.burst_capacity}, "
            f"target: {self.target_rpm} RPM, "
            f"hard limit: {self.hard_limit_rpm} RPM"
        )
    
    def _load_configurations(self) -> None:
        """Load all relevant configuration from YAML files."""
        
        # Load rate limits from config/apis/rate_limits.yaml
        rate_config = self.config_manager.get(
            f'apis.rate_limits.rate_limits.{self.provider}', {}
        )
        
        self.hard_limit_rpm = rate_config.get('calls_per_minute', 600)
        self.target_rpm = rate_config.get('target_calls_per_minute', 500)
        self.tokens_per_second = rate_config.get('tokens_per_second', 10.0)
        self.burst_capacity = rate_config.get('burst_size', 20)
        self.warning_threshold_rpm = rate_config.get('warning_threshold', 450)
        self.backoff_multiplier = rate_config.get('backoff_multiplier', 2)
        self.max_backoff = rate_config.get('max_backoff', 60)
        
        # Load schedules from config/data/schedules.yaml
        schedules = self.config_manager.get('data.schedules.schedules', {})
        
        # Market hours
        market_hours = schedules.get('market_hours', {})
        self.market_open = self._parse_time(market_hours.get('market_open', '09:30'))
        self.market_close = self._parse_time(market_hours.get('market_close', '16:00'))
        self.pre_market_start = self._parse_time(market_hours.get('pre_market_start', '04:00'))
        self.after_hours_end = self._parse_time(market_hours.get('after_hours_end', '20:00'))
        
        # MOC window
        moc_config = schedules.get('moc_window', {})
        self.moc_enabled = moc_config.get('enabled', True)
        self.moc_start = self._parse_time(moc_config.get('start_time', '15:40'))
        self.moc_end = self._parse_time(moc_config.get('end_time', '15:55'))
        self.moc_priority_boost = moc_config.get('priority_boost', 10)
        self.moc_update_interval = moc_config.get('update_interval', 5)
        
        # Data collection schedules by tier
        self.tier_schedules = schedules.get('data_collection', {})
        
        # Load symbols from config/data/symbols.yaml
        symbols_config = self.config_manager.get('data.symbols.symbols', {})
        self.tier_a_symbols = set(symbols_config.get('tier_a', {}).get('symbols', []))
        self.tier_b_symbols = set(symbols_config.get('tier_b', {}).get('symbols', []))
        self.tier_c_symbols = set(symbols_config.get('tier_c', {}).get('symbols', []))
        
        # Load monitoring/alerts from config/monitoring/alerts.yaml
        alerts_config = self.config_manager.get('monitoring.alerts.alerts', {})
        
        self.alert_types = alerts_config.get('types', {})
        self.rate_limit_warning_config = self.alert_types.get('api_rate_limit_warning', {})
        self.alert_rate_limiting = alerts_config.get('rate_limiting', {})
        self.max_alerts_per_minute = self.alert_rate_limiting.get('max_per_minute', 10)
        self.group_similar_alerts = self.alert_rate_limiting.get('group_similar', True)
        self.group_window_seconds = self.alert_rate_limiting.get('group_window_seconds', 60)
        
        # Load circuit breakers from config/risk/circuit_breakers.yaml
        circuit_breakers = self.config_manager.get('risk.circuit_breakers.circuit_breakers', {})
        
        # Rapid losses circuit breaker can affect API calls
        rapid_losses = circuit_breakers.get('rapid_losses', {})
        self.rapid_losses_enabled = rapid_losses.get('enabled', True)
        self.rapid_losses_count = rapid_losses.get('losses_count', 5)
        self.rapid_losses_window = rapid_losses.get('time_window_minutes', 60)
        self.pause_duration_minutes = rapid_losses.get('pause_duration_minutes', 30)
        
        # Load ingestion config from config/data/ingestion.yaml
        ingestion_config = self.config_manager.get('data.ingestion.ingestion', {})
        self.batch_size = ingestion_config.get('batch_size', 1000)
        self.max_retries = ingestion_config.get('max_retries', 3)
        self.validation_enabled = ingestion_config.get('validation', {}).get('enabled', True)
        
        # Queue configuration
        self.queue_config = {
            'max_size': 100,
            'default_timeout': 30.0,
            'starvation_threshold': 5.0
        }
        
        # Load environment-specific overrides
        environment = self.config_manager.environment
        env_overrides = self.config_manager.get(f'environments.{environment}.environment.overrides', {})
        
        # Apply any rate limit overrides from environment
        if 'apis' in env_overrides and self.provider in env_overrides['apis']:
            provider_overrides = env_overrides['apis'][self.provider]
            for key, value in provider_overrides.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _parse_time(self, time_str: str) -> datetime_time:
        """Parse time string (HH:MM) to datetime.time object."""
        try:
            parts = time_str.split(':')
            return datetime_time(int(parts[0]), int(parts[1]))
        except:
            return datetime_time(0, 0)
    
    def _initialize_statistics(self) -> None:
        """Initialize comprehensive statistics tracking."""
        self.statistics = {
            'total_requests': 0,
            'successful_acquisitions': 0,
            'failed_acquisitions': 0,
            'tokens_exhausted_count': 0,
            'total_wait_time': 0.0,
            'max_wait_time': 0.0,
            'requests_by_priority': defaultdict(int),
            'requests_by_symbol': defaultdict(int),
            'requests_by_tier': defaultdict(int),
            'requests_by_api_type': defaultdict(int),
            'last_rate_limit_hit': None,
            'token_starvation_events': 0,
            'queue_overflows': 0,
            'total_tokens_consumed': 0,
            'start_time': time.perf_counter(),
            'circuit_breaker_activations': 0,
            'moc_window_requests': 0,
            'alerts_sent': defaultdict(int),
            'queue_timeouts': 0,
            'retry_attempts': 0,
            'validation_failures': 0
        }
        
        # Per-tier statistics
        for tier in ['tier_a', 'tier_b', 'tier_c','other']:
            self.statistics[f'{tier}_requests'] = 0
            self.statistics[f'{tier}_wait_time'] = 0.0
    
    def _start_queue_processor(self) -> None:
        """Start background thread to process queued requests."""
        self.queue_processor = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name="RateLimiter-QueueProcessor"
        )
        self.queue_processor.start()
    
    def _process_queue(self) -> None:
        """Background thread to process priority queue."""
        while self.running:
            try:
                # Get request with timeout to allow checking self.running
                try:
                    request = self.wait_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check if request has timed out
                elapsed = time.perf_counter() - request.timestamp
                if elapsed > request.timeout:
                    self.statistics['queue_timeouts'] += 1
                    request.event.set()  # Signal timeout
                    continue
                
                # Try to acquire tokens
                with self.lock:
                    self._refill()
                    
                    if self.tokens >= request.tokens:
                        # Tokens available
                        self.tokens -= request.tokens
                        self.statistics['successful_acquisitions'] += 1
                        self.statistics['total_tokens_consumed'] += request.tokens
                        
                        # Record request
                        self._record_request(
                            RequestPriority(request.priority),
                            request.symbol,
                            request.api_type
                        )
                        
                        # Signal success
                        request.event.set()
                    else:
                        # Still not enough tokens, re-queue if not timed out
                        remaining_timeout = request.timeout - elapsed
                        if remaining_timeout > 0:
                            request.timeout = remaining_timeout
                            self.wait_queue.put(request)
                        else:
                            self.statistics['queue_timeouts'] += 1
                            request.event.set()
                
            except Exception as e:
                self.logger.error(f"Queue processor error: {e}")
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called with lock held."""
        now = time.perf_counter()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            # Calculate tokens to add based on configured rate
            tokens_to_add = elapsed * self.tokens_per_second
            
            # Add tokens up to burst capacity
            prev_tokens = self.tokens
            self.tokens = min(self.tokens + tokens_to_add, float(self.burst_capacity))
            
            # Update refill time
            self.last_refill = now
            
            # Check for token starvation
            if prev_tokens == 0 and self.tokens > 0:
                starvation_duration = now - getattr(self, 'last_token_time', now)
                if starvation_duration > self.queue_config['starvation_threshold']:
                    self.statistics['token_starvation_events'] += 1
                    self._send_alert(
                        AlertLevel.WARNING,
                        f"Token starvation ended after {starvation_duration:.2f} seconds"
                    )
            
            if self.tokens > 0:
                self.last_token_time = now
    
    def _get_market_period(self) -> str:
        """Determine current market period."""
        now = datetime.now()
        current_time = now.time()
        
        if current_time < self.pre_market_start:
            return "CLOSED"
        elif current_time < self.market_open:
            return "PRE_MARKET"
        elif current_time < self.market_close:
            return "MARKET_HOURS"
        elif current_time < self.after_hours_end:
            return "AFTER_HOURS"
        else:
            return "CLOSED"
    
    def _is_moc_window(self) -> bool:
        """Check if currently in MOC window."""
        if not self.moc_enabled:
            return False
        
        now = datetime.now()
        current_time = now.time()
        
        return self.moc_start <= current_time <= self.moc_end
    
    def _get_symbol_tier(self, symbol: Optional[str]) -> str:
        """Determine tier for a symbol."""
        if not symbol:
            return "other"
        
        symbol = symbol.upper()
        
        if symbol in self.tier_a_symbols:
            return "tier_a"
        elif symbol in self.tier_b_symbols:
            return "tier_b"
        elif symbol in self.tier_c_symbols:
            return "tier_c"
        else:
            return "other"
    
    def _determine_priority(self, symbol: Optional[str], api_type: Optional[str],
                           base_priority: RequestPriority) -> RequestPriority:
        """
        Determine actual priority based on symbol, API type, and market conditions.
        
        Args:
            symbol: Trading symbol
            api_type: Type of API call (options, indicators, etc.)
            base_priority: Base priority level
            
        Returns:
            Adjusted priority
        """
        # MOC window elevation
        if self._is_moc_window():
            tier = self._get_symbol_tier(symbol)
            
            # Check MOC-relevant symbols from tier schedules
            if tier in ['tier_a', 'tier_b']:
                return RequestPriority.MOC_WINDOW
        
        # Symbol-based priority from configuration
        if symbol:
            tier = self._get_symbol_tier(symbol)
            
            # Map tier to priority
            if tier == "tier_a":
                return RequestPriority.TIER_A
            elif tier == "tier_b":
                return RequestPriority.TIER_B
            elif tier == "tier_c":
                return RequestPriority.TIER_C
        
        return base_priority
    
    def _record_request(self, priority: RequestPriority, 
                       symbol: Optional[str] = None,
                       api_type: Optional[str] = None) -> None:
        """Record request for statistics. Must be called with lock held."""
        now = time.perf_counter()
        
        # Add to rolling window
        self.request_timestamps.append(now)
        
        # Clean old timestamps
        cutoff = now - self.minute_window_size
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        
        # Update statistics
        self.statistics['total_requests'] += 1
        self.statistics['requests_by_priority'][priority.name] += 1
        
        if symbol:
            self.statistics['requests_by_symbol'][symbol] += 1
            tier = self._get_symbol_tier(symbol)
            self.statistics['requests_by_tier'][tier] += 1
            self.statistics[f'{tier}_requests'] += 1
        
        if api_type:
            self.statistics['requests_by_api_type'][api_type] += 1
        
        if self._is_moc_window():
            self.statistics['moc_window_requests'] += 1
    
    def _send_alert(self, level: AlertLevel, message: str) -> None:
        """Send alert if not rate limited."""
        now = time.time()
        alert_key = f"{level.value}:{message[:50]}"
        
        # Check alert rate limiting
        if alert_key in self.last_alert_time:
            if now - self.last_alert_time[alert_key] < self.group_window_seconds:
                return  # Skip duplicate alert
        
        # Check overall alert rate
        minute_ago = now - 60
        recent_alerts = sum(
            1 for t in self.last_alert_time.values() 
            if t > minute_ago
        )
        
        if recent_alerts >= self.max_alerts_per_minute:
            return  # Rate limited
        
        # Send alert (would integrate with Discord in production)
        self.logger.log(
            getattr(logging, level.value, logging.INFO),
            f"[ALERT] {message}"
        )
        
        # Record alert
        self.last_alert_time[alert_key] = now
        self.statistics['alerts_sent'][level.value] += 1
    
    def get_current_rpm(self) -> float:
        """Calculate current requests per minute based on rolling window."""
        with self.lock:
            now = time.perf_counter()
            cutoff = now - self.minute_window_size
            
            # Clean old timestamps FIRST
            while self.request_timestamps and self.request_timestamps[0] < cutoff:
                self.request_timestamps.popleft()
            
            # Count only recent requests
            recent_count = len(self.request_timestamps)
            
            # Don't extrapolate during startup - just return actual count
            if now - self.statistics['start_time'] < self.minute_window_size:
                # During first minute, return actual requests made
                return float(recent_count)
            
            # After first minute, calculate true RPM
            return float(recent_count) 
           
    def acquire(self, tokens: int = 1, blocking: bool = True,
                timeout: Optional[float] = None,
                priority: Optional[RequestPriority] = None,
                symbol: Optional[str] = None,
                api_type: Optional[str] = None) -> bool:
        """
        Acquire tokens from bucket.
        
        Args:
            tokens: Number of tokens to acquire
            blocking: Whether to block until tokens available
            timeout: Maximum time to wait
            priority: Request priority (auto-determined if None)
            symbol: Trading symbol for the request
            api_type: Type of API call
            
        Returns:
            True if tokens acquired, False otherwise
            
        Raises:
            RateLimitExceeded: If hard limit would be exceeded
            TokensExhausted: If no tokens and non-blocking
            QueueOverflow: If queue is full
        """
        if timeout is None:
            timeout = self.queue_config['default_timeout']
        
        # Auto-determine priority if not specified
        if priority is None:
            priority = self._determine_priority(
                symbol, api_type, RequestPriority.TIER_B
            )
        else:
            # Still check for MOC elevation
            priority = self._determine_priority(symbol, api_type, priority)
        
        start_time = time.perf_counter()
        
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            if self.circuit_breaker_reset_time and time.perf_counter() < self.circuit_breaker_reset_time:
                self.statistics['failed_acquisitions'] += 1
                if not blocking:
                    raise TokensExhausted("Circuit breaker active")
                return False
            else:
                self.circuit_breaker_triggered = False
                self.circuit_breaker_reset_time = None
                self._send_alert(AlertLevel.INFO, "Circuit breaker reset")
        
        # Check current RPM
        current_rpm = self.get_current_rpm()
        
        # Hard limit check
        if current_rpm >= self.hard_limit_rpm:
            self._send_alert(
                AlertLevel.CRITICAL,
                f"Hard rate limit exceeded: {current_rpm:.1f} RPM"
            )
            self.statistics['last_rate_limit_hit'] = datetime.now()
            self.circuit_breaker_triggered = True
            self.circuit_breaker_reset_time = time.perf_counter() + 5.0
            self.statistics['circuit_breaker_activations'] += 1
            raise RateLimitExceeded(f"Rate limit: {current_rpm:.1f} >= {self.hard_limit_rpm}")
        
        # Warning threshold
        if current_rpm >= self.warning_threshold_rpm:
            self._send_alert(
                AlertLevel.WARNING,
                f"Approaching rate limit: {current_rpm:.1f} RPM"
            )
            
            # Throttle low priority
            if priority.value >= RequestPriority.TIER_C.value:
                if not blocking:
                    self.statistics['failed_acquisitions'] += 1
                    return False
                # Add delay for low priority
                time.sleep(0.5)
        
        # Target RPM soft throttling
        if current_rpm >= self.target_rpm:
            market_period = self._get_market_period()
            if market_period == "MARKET_HOURS":
                # Only throttle during market hours
                time.sleep(0.1)
        
        # Try immediate acquisition
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                # Success - immediate acquisition
                self.tokens -= tokens
                self.statistics['successful_acquisitions'] += 1
                self.statistics['total_tokens_consumed'] += tokens
                self._record_request(priority, symbol, api_type)
                
                # Update tier-specific wait time
                wait_time = time.perf_counter() - start_time
                self.statistics['total_wait_time'] += wait_time
                tier = self._get_symbol_tier(symbol)
                self.statistics[f'{tier}_wait_time'] += wait_time
                
                return True
            
            elif not blocking:
                # Non-blocking and insufficient tokens
                self.statistics['failed_acquisitions'] += 1
                self.statistics['tokens_exhausted_count'] += 1
                raise TokensExhausted(f"Need {tokens}, have {self.tokens:.2f}")
        
        # Need to queue the request
        request = QueuedRequest(
            priority=priority.value,
            timestamp=time.perf_counter(),
            tokens=tokens,
            event=threading.Event(),
            timeout=timeout,
            symbol=symbol,
            api_type=api_type
        )
        
        try:
            self.wait_queue.put_nowait(request)
        except queue.Full:
            self.statistics['queue_overflows'] += 1
            raise QueueOverflow("Priority queue full")
        
        # Wait for request to be processed
        if request.event.wait(timeout):
            # Request was processed (might be timeout)
            wait_time = time.perf_counter() - start_time
            
            if wait_time >= timeout:
                self.statistics['queue_timeouts'] += 1
                return False
            
            # Success
            self.statistics['total_wait_time'] += wait_time
            self.statistics['max_wait_time'] = max(
                self.statistics['max_wait_time'], wait_time
            )
            
            tier = self._get_symbol_tier(symbol)
            self.statistics[f'{tier}_wait_time'] += wait_time
            
            return True
        else:
            # Timeout
            self.statistics['queue_timeouts'] += 1
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Calculate estimated wait time for tokens."""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.tokens_per_second
    
    def get_available_tokens(self) -> float:
        """Get current available tokens."""
        with self.lock:
            self._refill()
            return self.tokens
    
    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        with self.lock:
            self.tokens = float(self.burst_capacity)
            self.last_refill = time.perf_counter()
            self.circuit_breaker_triggered = False
            self.circuit_breaker_reset_time = None
            self.logger.info("Rate limiter reset to full capacity")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.lock:
            stats = dict(self.statistics)
            
            # Add calculated metrics
            stats['current_rpm'] = self.get_current_rpm()
            stats['available_tokens'] = self.tokens
            stats['max_tokens'] = self.burst_capacity
            stats['queue_size'] = self.wait_queue.qsize()
            stats['uptime_seconds'] = time.perf_counter() - stats['start_time']
            stats['market_period'] = self._get_market_period()
            stats['is_moc_window'] = self._is_moc_window()
            
            # Calculate success rate
            total = stats['successful_acquisitions'] + stats['failed_acquisitions']
            if total > 0:
                stats['success_rate'] = stats['successful_acquisitions'] / total
            else:
                stats['success_rate'] = 1.0
            
            # Average wait time
            if stats['successful_acquisitions'] > 0:
                stats['avg_wait_time'] = stats['total_wait_time'] / stats['successful_acquisitions']
            else:
                stats['avg_wait_time'] = 0.0
            
            return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check rate limiter health."""
        stats = self.get_statistics()
        
        health = {
            'healthy': True,
            'current_rpm': stats['current_rpm'],
            'available_tokens': stats['available_tokens'],
            'queue_size': stats['queue_size'],
            'success_rate': stats['success_rate'],
            'circuit_breaker_active': self.circuit_breaker_triggered,
            'warnings': []
        }
        
        # Check for issues
        if stats['current_rpm'] > self.warning_threshold_rpm:
            health['warnings'].append(f"High RPM: {stats['current_rpm']:.1f}")
            health['healthy'] = False
        
        if stats['available_tokens'] < 5:
            health['warnings'].append(f"Low tokens: {stats['available_tokens']:.1f}")
        
        if stats['queue_size'] > 50:
            health['warnings'].append(f"Large queue: {stats['queue_size']}")
        
        if stats['success_rate'] < 0.95:
            health['warnings'].append(f"Low success rate: {stats['success_rate']:.2%}")
        
        if self.circuit_breaker_triggered:
            health['warnings'].append("Circuit breaker active")
            health['healthy'] = False
        
        return health
    
    def shutdown(self) -> bool:
        """Shutdown rate limiter gracefully."""
        self.logger.info("Shutting down rate limiter...")
        self.running = False
        
        if self.queue_processor:
            self.queue_processor.join(timeout=2.0)
        
        # Log final statistics
        stats = self.get_statistics()
        self.logger.info(f"Final statistics: {stats['total_requests']} total requests, "
                        f"{stats['success_rate']:.2%} success rate")
        
        return True


# Decorator for rate-limited methods
def rate_limit(priority: RequestPriority = RequestPriority.TIER_B,
               tokens: int = 1,
               timeout: float = 30.0):
    """
    Decorator for rate-limiting API methods.
    
    Args:
        priority: Request priority level
        tokens: Tokens required
        timeout: Max wait time
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Extract symbol if present
            symbol = kwargs.get('symbol') or (args[0] if args else None)
            api_type = func.__name__
            
            # Get rate limiter from self
            limiter = getattr(self, 'rate_limiter', None)
            if not limiter:
                # No rate limiter, proceed without limiting
                return func(self, *args, **kwargs)
            
            # Acquire tokens
            if not limiter.acquire(
                tokens=tokens,
                priority=priority,
                timeout=timeout,
                symbol=symbol,
                api_type=api_type
            ):
                raise TimeoutError(f"Rate limit timeout for {api_type}")
            
            # Execute function
            return func(self, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator