"""
Logging infrastructure for AlphaTrader system.

Production-grade structured logging with performance monitoring,
audit trails, and separate loggers for different components.
"""

import sys
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
import time

from loguru import logger

from src.core.config import TradingConfig


class PerformanceLogger:
    """Track and log performance metrics for critical operations."""
    
    def __init__(self, operation: str, component: str = "system"):
        """
        Initialize performance logger.
        
        Args:
            operation: Name of the operation being tracked
            component: System component performing the operation
        """
        self.operation = operation
        self.component = component
        self.start_time = None
        self.checkpoints = []
    
    def __enter__(self):
        """Start timing the operation."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the operation duration."""
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            
            log_data = {
                "component": self.component,
                "operation": self.operation,
                "duration_ms": round(duration_ms, 2),
                "checkpoints": self.checkpoints,
                "success": exc_type is None,
            }
            
            if exc_type:
                log_data["error"] = str(exc_val)
                logger.error(f"Operation failed: {self.operation}", **log_data)
            else:
                # Log as warning if exceeds latency targets
                if self._exceeds_target(duration_ms):
                    logger.warning(f"Operation exceeded target: {self.operation}", **log_data)
                else:
                    logger.debug(f"Operation completed: {self.operation}", **log_data)
    
    def checkpoint(self, name: str):
        """Add a checkpoint timing."""
        if self.start_time:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.checkpoints.append({
                "name": name,
                "elapsed_ms": round(elapsed_ms, 2)
            })
    
    def _exceeds_target(self, duration_ms: float) -> bool:
        """Check if operation exceeds latency target."""
        # Import here to avoid circular dependency
        from src.core.constants import LatencyTargets
        
        targets = {
            "feature_calculation": LatencyTargets.FEATURE_CALCULATION,
            "model_inference": LatencyTargets.MODEL_INFERENCE,
            "risk_validation": LatencyTargets.RISK_VALIDATION,
            "order_execution": LatencyTargets.ORDER_EXECUTION,
            "greeks_calculation": LatencyTargets.OPTIONS_GREEKS_CALC,
            "vpin_calculation": LatencyTargets.VPIN_CALCULATION,
        }
        
        target = targets.get(self.operation.lower())
        return duration_ms > target if target else False


def performance_tracked(component: str = "system"):
    """
    Decorator to track function performance.
    
    Args:
        component: System component name
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceLogger(func.__name__, component):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class TradingLogger:
    """
    Main logging configuration for AlphaTrader.
    
    Provides structured logging with separate files for:
    - System logs (general operations)
    - Trading logs (orders, fills, signals)
    - Risk logs (limit breaches, warnings)
    - Data logs (feed issues, API calls)
    - Community logs (Discord, webhooks)
    - Audit logs (compliance, changes)
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize logging system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.log_dir = config.log_dir
        self.log_level = config.log_level
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Configure loggers
        self._setup_console_logger()
        self._setup_file_loggers()
        self._setup_error_tracking()
    
    def _setup_console_logger(self):
        """Configure console output."""
        # Color-coded console output
        logger.add(
            sys.stdout,
            level=self.log_level,
            format=self._get_console_format(),
            colorize=True,
            backtrace=True,
            diagnose=False,  # Don't show variables in production
        )
    
    def _setup_file_loggers(self):
        """Configure file-based logging."""
        
        # System logs - general operations
        logger.add(
            self.log_dir / "system_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format=self._get_file_format(),
            rotation="00:00",  # Daily rotation
            retention="30 days",
            compression="gz",
            serialize=False,
            backtrace=True,
            diagnose=False,
            filter=lambda record: record["extra"].get("category", "system") == "system",
        )
        
        # Trading logs - orders, fills, signals
        logger.add(
            self.log_dir / "trading_{time:YYYY-MM-DD}.log",
            level="INFO",
            format=self._get_json_format(),
            rotation="00:00",
            retention="90 days",  # Keep trading logs longer
            compression="gz",
            serialize=True,  # JSON format for analysis
            filter=lambda record: record["extra"].get("category") == "trading",
        )
        
        # Risk logs - limits, warnings, breaches
        logger.add(
            self.log_dir / "risk_{time:YYYY-MM-DD}.log",
            level="WARNING",
            format=self._get_json_format(),
            rotation="00:00",
            retention="180 days",  # Keep risk logs longest
            compression="gz",
            serialize=True,
            filter=lambda record: record["extra"].get("category") == "risk",
        )
        
        # Data logs - feeds, APIs
        logger.add(
            self.log_dir / "data_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format=self._get_file_format(),
            rotation="00:00",
            retention="7 days",  # Short retention for high-volume
            compression="gz",
            filter=lambda record: record["extra"].get("category") == "data",
        )
        
        # Community logs - Discord, webhooks
        logger.add(
            self.log_dir / "community_{time:YYYY-MM-DD}.log",
            level="INFO",
            format=self._get_file_format(),
            rotation="00:00",
            retention="30 days",
            compression="gz",
            filter=lambda record: record["extra"].get("category") == "community",
        )
        
        # Audit logs - compliance, changes
        logger.add(
            self.log_dir / "audit_{time:YYYY-MM-DD}.log",
            level="INFO",
            format=self._get_audit_format(),
            rotation="00:00",
            retention="7 years",  # Regulatory requirement
            compression="gz",
            serialize=True,
            filter=lambda record: record["extra"].get("category") == "audit",
        )
        
        # Error logs - all errors in one place
        logger.add(
            self.log_dir / "errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            format=self._get_json_format(),
            rotation="00:00",
            retention="90 days",
            compression="gz",
            serialize=True,
            backtrace=True,
            diagnose=True,  # Include variables for errors
        )
    
    def _setup_error_tracking(self):
        """Configure error tracking and alerting."""
        
        def error_sink(message):
            """Custom sink for critical errors."""
            record = message.record
            
            # Check if it's a critical error
            if record["level"].no >= 40:  # ERROR or CRITICAL
                error_data = {
                    "timestamp": record["time"].isoformat(),
                    "level": record["level"].name,
                    "message": record["message"],
                    "module": record["module"],
                    "function": record["function"],
                    "line": record["line"],
                    "extra": record["extra"],
                }
                
                # Check for critical error codes
                error_code = record["extra"].get("error_code")
                if error_code in ["E004", "E009", "E010"]:  # Critical errors
                    # In production, this would trigger alerts
                    self._send_critical_alert(error_data)
        
        logger.add(error_sink, level="ERROR")
    
    def _send_critical_alert(self, error_data: Dict[str, Any]):
        """
        Send critical error alerts.
        
        In production, this would integrate with PagerDuty, Slack, etc.
        """
        # Placeholder for alert integration
        print(f"🚨 CRITICAL ALERT: {error_data['message']}")
    
    @staticmethod
    def _get_console_format() -> str:
        """Get console log format."""
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    @staticmethod
    def _get_file_format() -> str:
        """Get file log format."""
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    @staticmethod
    def _get_json_format() -> str:
        """Get JSON log format for structured logs."""
        return "{message}"  # Message will be serialized JSON
    
    @staticmethod
    def _get_audit_format() -> str:
        """Get audit log format."""
        return "{message}"  # Will be serialized with full context


def get_logger(category: str = "system"):
    """
    Get a logger instance for a specific category.
    
    Args:
        category: Log category (system, trading, risk, data, community, audit)
        
    Returns:
        Logger instance with category context
    """
    return logger.bind(category=category)


# Convenience logger instances
system_logger = get_logger("system")
trading_logger = get_logger("trading")
risk_logger = get_logger("risk")
data_logger = get_logger("data")
community_logger = get_logger("community")
audit_logger = get_logger("audit")


def log_trade(
    action: str,
    symbol: str,
    quantity: int,
    price: float,
    order_type: str,
    **extra
):
    """
    Log a trading action with full context.
    
    Args:
        action: Trade action (BUY, SELL, etc.)
        symbol: Trading symbol
        quantity: Number of shares/contracts
        price: Execution or order price
        order_type: Type of order
        **extra: Additional context
    """
    trading_logger.info(
        f"Trade: {action} {quantity} {symbol} @ {price}",
        action=action,
        symbol=symbol,
        quantity=quantity,
        price=price,
        order_type=order_type,
        timestamp=datetime.utcnow().isoformat(),
        **extra
    )


def log_risk_breach(
    limit_type: str,
    current_value: float,
    limit_value: float,
    action: str,
    **extra
):
    """
    Log a risk limit breach.
    
    Args:
        limit_type: Type of limit breached
        current_value: Current value that breached
        limit_value: Limit that was exceeded
        action: Action taken
        **extra: Additional context
    """
    risk_logger.critical(
        f"RISK BREACH: {limit_type} = {current_value} (limit: {limit_value})",
        limit_type=limit_type,
        current_value=current_value,
        limit_value=limit_value,
        action=action,
        timestamp=datetime.utcnow().isoformat(),
        severity="CRITICAL",
        **extra
    )


def log_api_call(
    api_name: str,
    endpoint: str,
    status_code: Optional[int] = None,
    latency_ms: Optional[float] = None,
    **extra
):
    """
    Log an API call.
    
    Args:
        api_name: Name of the API
        endpoint: API endpoint called
        status_code: HTTP status code
        latency_ms: Call latency in milliseconds
        **extra: Additional context
    """
    data_logger.debug(
        f"API Call: {api_name} - {endpoint}",
        api_name=api_name,
        endpoint=endpoint,
        status_code=status_code,
        latency_ms=latency_ms,
        timestamp=datetime.utcnow().isoformat(),
        **extra
    )