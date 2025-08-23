"""
Centralized logging system for AlphaTrader
Structured logging with correlation ID support
Zero hardcoded values - all configuration from environment
"""
import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from datetime import datetime
import json


class AlphaTraderLogger:
    """
    Institutional-grade logging with structured output
    All configuration from environment - no hardcoding
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def _initialize(cls):
        """Initialize logging system with environment configuration"""
        if cls._initialized:
            return
        
        # Get all configuration from environment - NO DEFAULTS
        log_level = os.environ['LOG_LEVEL']
        log_dir = Path(os.environ['LOG_DIR'])
        log_format = os.environ['LOG_FORMAT']
        max_bytes = int(os.environ['LOG_MAX_BYTES'])
        backup_count = int(os.environ['LOG_BACKUP_COUNT'])
        app_name = os.environ['APP_NAME']
        environment = os.environ['ENVIRONMENT']
        
        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                cls._add_app_context,
                structlog.processors.JSONRenderer() if log_format == 'json' else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure Python logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level.upper())
        )
        
        # Create file handler with rotation
        log_file = log_dir / f"{app_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=os.environ.get('LOG_ENCODING', 'utf-8'),
            delay=os.environ.get('LOG_DELAY', 'false').lower() == 'true'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Add formatter based on format type
        if log_format == 'json':
            file_handler.setFormatter(JsonFormatter())
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            file_handler.setFormatter(logging.Formatter(format_string))
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
        
        cls._initialized = True
    
    @staticmethod
    def _add_app_context(logger, log_method, event_dict):
        """Add application context to all logs"""
        # All from environment - no hardcoding
        event_dict['app'] = os.environ.get('APP_NAME')
        event_dict['environment'] = os.environ.get('ENVIRONMENT')
        
        # Add correlation ID if present in context
        if hasattr(logger, '_context') and 'correlation_id' in logger._context:
            event_dict['correlation_id'] = logger._context['correlation_id']
        
        return event_dict
    
    @classmethod
    def get_logger(cls, name: str) -> structlog.BoundLogger:
        """
        Get or create a logger instance
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured structlog logger
        """
        cls._initialize()
        
        if name not in cls._loggers:
            cls._loggers[name] = structlog.get_logger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def bind_correlation_id(cls, logger: structlog.BoundLogger, correlation_id: str) -> structlog.BoundLogger:
        """
        Bind correlation ID to logger
        
        Args:
            logger: Logger instance
            correlation_id: Correlation ID to bind
            
        Returns:
            Logger with correlation ID bound
        """
        return logger.bind(correlation_id=correlation_id)
    
    @classmethod
    def log_performance(
        cls,
        logger: structlog.BoundLogger,
        operation: str,
        duration_ms: float,
        **kwargs
    ):
        """
        Log performance metrics
        
        Args:
            logger: Logger instance
            operation: Operation name
            duration_ms: Duration in milliseconds
            **kwargs: Additional context
        """
        # Get threshold from environment
        slow_threshold = float(os.environ.get('PERF_SLOW_QUERY_THRESHOLD_MS', '100'))
        
        log_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'slow': duration_ms > slow_threshold,
            **kwargs
        }
        
        if duration_ms > slow_threshold:
            logger.warning("Slow operation detected", **log_data)
        else:
            logger.debug("Operation completed", **log_data)
    
    @classmethod
    def audit_log(
        cls,
        action: str,
        user: Optional[str] = None,
        resource: Optional[str] = None,
        result: str = 'success',
        **details
    ):
        """
        Create audit log entry
        
        Args:
            action: Action performed
            user: User who performed the action
            resource: Resource affected
            result: Result of the action
            **details: Additional details
        """
        if os.environ.get('LOG_AUDIT', 'false').lower() != 'true':
            return
        
        audit_logger = cls.get_logger('audit')
        
        audit_logger.info(
            "Audit event",
            action=action,
            user=user,
            resource=resource,
            result=result,
            timestamp=datetime.utcnow().isoformat(),
            **details
        )


class JsonFormatter(logging.Formatter):
    """JSON formatter for file output"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add environment context - from env, not hardcoded
        log_obj['app'] = os.environ.get('APP_NAME')
        log_obj['environment'] = os.environ.get('ENVIRONMENT')
        
        # Add any extra fields
        if hasattr(record, 'correlation_id'):
            log_obj['correlation_id'] = record.correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)


# Convenience function for getting logger
def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return AlphaTraderLogger.get_logger(name)