"""
Production-grade decorators for retry, circuit breaker, metrics, and tracing
All configuration from environment - zero hardcoding
"""
import os
import time
import functools
import random
from typing import Any, Callable, Optional
from datetime import datetime

from .logger import get_logger
from .exceptions import CircuitBreakerException, TimeoutException
from .metrics import MetricsCollector


def retry_with_backoff(
    max_attempts: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    max_delay: Optional[int] = None,
    jitter: Optional[bool] = None,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff
    All parameters from environment if not specified
    
    Args:
        max_attempts: Maximum retry attempts
        backoff_factor: Exponential backoff factor
        max_delay: Maximum delay between retries
        jitter: Add random jitter to delays
        exceptions: Exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get configuration from environment if not provided
            _max_attempts = max_attempts or int(os.environ['RETRY_MAX_ATTEMPTS'])
            _backoff_factor = backoff_factor or float(os.environ['RETRY_BACKOFF_FACTOR'])
            _max_delay = max_delay or int(os.environ['RETRY_MAX_DELAY'])
            _jitter = jitter if jitter is not None else os.environ['RETRY_JITTER'].lower() == 'true'
            
            logger = get_logger(func.__module__)
            last_exception = None
            
            for attempt in range(_max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == _max_attempts - 1:
                        logger.error(
                            f"All retry attempts exhausted for {func.__name__}",
                            attempts=_max_attempts,
                            error=str(e)
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(_backoff_factor ** attempt, _max_delay)
                    
                    # Add jitter if enabled
                    if _jitter:
                        delay *= (1 + random.random() * 0.1)
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{_max_attempts} for {func.__name__}",
                        delay=delay,
                        error=str(e)
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: Optional[int] = None,
    recovery_timeout: Optional[int] = None,
    expected_exception: Optional[type] = None
):
    """
    Circuit breaker decorator
    All parameters from environment if not specified
    
    Args:
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds before attempting recovery
        expected_exception: Exception type to catch
    """
    def decorator(func: Callable) -> Callable:
        # Get configuration from environment if not provided
        _failure_threshold = failure_threshold or int(os.environ['CB_FAILURE_THRESHOLD'])
        _recovery_timeout = recovery_timeout or int(os.environ['CB_RECOVERY_TIMEOUT'])
        _expected_exception = expected_exception or Exception
        
        # Circuit breaker state
        state = {'failures': 0, 'last_failure': None, 'state': 'closed'}
        logger = get_logger(func.__module__)
        metrics = MetricsCollector()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if circuit should reset
            if state['state'] == 'open':
                if state['last_failure'] and \
                   (time.time() - state['last_failure']) > _recovery_timeout:
                    state['state'] = 'half-open'
                    logger.info(f"Circuit breaker for {func.__name__} moved to half-open")
                else:
                    metrics.update_circuit_breaker(func.__name__, 'open')
                    raise CircuitBreakerException(
                        f"Circuit breaker for {func.__name__} is open",
                        service=func.__name__,
                        recovery_time=_recovery_timeout
                    )
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset if half-open
                if state['state'] == 'half-open':
                    state['state'] = 'closed'
                    state['failures'] = 0
                    logger.info(f"Circuit breaker for {func.__name__} closed")
                    metrics.update_circuit_breaker(func.__name__, 'closed')
                
                return result
                
            except _expected_exception as e:
                state['failures'] += 1
                state['last_failure'] = time.time()
                
                metrics.update_circuit_breaker(func.__name__, state['state'], failure=True)
                
                if state['failures'] >= _failure_threshold:
                    state['state'] = 'open'
                    logger.error(
                        f"Circuit breaker for {func.__name__} opened",
                        failures=state['failures'],
                        threshold=_failure_threshold
                    )
                    metrics.update_circuit_breaker(func.__name__, 'open')
                
                raise
        
        return wrapper
    return decorator


def trace_performance(func: Callable) -> Callable:
    """
    Trace function performance
    Logs execution time and updates metrics
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if os.environ.get('PERF_TRACE_ENABLED', 'false').lower() != 'true':
            return func(*args, **kwargs)
        
        logger = get_logger(func.__module__)
        metrics = MetricsCollector()
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Get slow threshold from environment
            slow_threshold = float(os.environ.get('PERF_SLOW_QUERY_THRESHOLD_MS', '100')) / 1000
            
            if duration > slow_threshold:
                logger.warning(
                    f"Slow function execution: {func.__name__}",
                    duration_seconds=duration,
                    threshold_seconds=slow_threshold
                )
            else:
                logger.debug(
                    f"Function executed: {func.__name__}",
                    duration_seconds=duration
                )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Function failed: {func.__name__}",
                duration_seconds=duration,
                error=str(e)
            )
            raise
    
    return wrapper


def collect_metrics(
    operation_type: str = 'general',
    operation_name: Optional[str] = None
):
    """
    Collect metrics for function execution
    
    Args:
        operation_type: Type of operation (db, cache, api, etc.)
        operation_name: Specific operation name
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            metrics = MetricsCollector()
            
            if not metrics.enabled:
                return func(*args, **kwargs)
            
            _operation_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics based on type
                if operation_type == 'db':
                    metrics.record_db_query(_operation_name, duration, 'success')
                elif operation_type == 'cache':
                    metrics.record_cache_operation(_operation_name, duration)
                elif operation_type == 'api':
                    # Extract API name from args if available
                    api_name = kwargs.get('api', 'unknown')
                    metrics.record_api_call(api_name, _operation_name, duration, 'success')
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if operation_type == 'db':
                    metrics.record_db_query(_operation_name, duration, 'error')
                elif operation_type == 'api':
                    api_name = kwargs.get('api', 'unknown')
                    metrics.record_api_call(api_name, _operation_name, duration, 'error')
                
                raise
        
        return wrapper
    return decorator


def add_correlation_id(func: Callable) -> Callable:
    """
    Add correlation ID to function context
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Check if correlation is enabled
        if os.environ.get('LOG_CORRELATION', 'false').lower() != 'true':
            return func(*args, **kwargs)
        
        # Try to get correlation ID from kwargs or generate new one
        import uuid
        correlation_id = kwargs.get('correlation_id') or str(uuid.uuid4())
        
        # Add to kwargs if not present
        if 'correlation_id' not in kwargs:
            kwargs['correlation_id'] = correlation_id
        
        # Add to logger context
        logger = get_logger(func.__module__)
        logger = logger.bind(correlation_id=correlation_id)
        
        return func(*args, **kwargs)
    
    return wrapper


def rate_limit(
    resource_name: str,
    calls_per_second: Optional[float] = None
):
    """
    Rate limiting decorator
    
    Args:
        resource_name: Name of the resource being rate limited
        calls_per_second: Maximum calls per second
    """
    def decorator(func: Callable) -> Callable:
        # Get rate limit from environment if not provided
        _calls_per_second = calls_per_second or float(
            os.environ.get(f'RATE_LIMIT_{resource_name.upper()}_CPS', '1.0')
        )
        
        # Calculate minimum interval between calls
        min_interval = 1.0 / _calls_per_second
        last_called = {'time': 0}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() != 'true':
                return func(*args, **kwargs)
            
            current_time = time.time()
            time_since_last = current_time - last_called['time']
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger = get_logger(func.__module__)
                logger.debug(
                    f"Rate limiting {func.__name__}",
                    sleep_seconds=sleep_time,
                    resource=resource_name
                )
                time.sleep(sleep_time)
            
            last_called['time'] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def timeout(seconds: Optional[int] = None):
    """
    Timeout decorator
    
    Args:
        seconds: Timeout in seconds
    """
    import signal
    
    def decorator(func: Callable) -> Callable:
        def timeout_handler(signum, frame):
            raise TimeoutException(
                f"Function {func.__name__} timed out",
                timeout_value=_seconds
            )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get timeout from environment if not provided
            _seconds = seconds or int(
                os.environ.get('DEFAULT_TIMEOUT', '30')
            )
            
            # Set signal alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(_seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel alarm
                signal.alarm(0)
            
            return result
        
        return wrapper
    return decorator