#!/usr/bin/env python3
"""
Logger - Centralized logging system
Phase 0: Foundation Setup
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import colorlog
from pythonjsonlogger import jsonlogger

from src.foundation.config_manager import get_config_manager


class TradingLogger:
    """Centralized logging system for the trading platform"""
    
    def __init__(self):
        """Initialize the logging system"""
        self.config = get_config_manager()
        
        # Load logging configuration
        self.log_config = self.config.get_config('logging.logging', {})
        if not self.log_config:
            # Fallback to basic config if YAML not loaded
            self.log_config = {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'console': {'enabled': True, 'level': 'INFO'}
            }
        
        # Create logs directory
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup root logger
        self._setup_root_logger()
        
        # Setup handlers
        self._setup_handlers()
        
        # Configure component loggers
        self._configure_component_loggers()
        
        # Log initialization
        self.logger = self.get_logger('TradingSystem')
        self.logger.info("="*50)
        self.logger.info("Trading System Logger Initialized")
        self.logger.info(f"Environment: {self.config.environment}")
        self.logger.info(f"Log Level: {self.log_config.get('level', 'INFO')}")
        self.logger.info("="*50)
    
    def _setup_root_logger(self):
        """Configure the root logger"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_config.get('level', 'INFO')))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
    
    def _setup_handlers(self):
        """Setup all logging handlers"""
        root_logger = logging.getLogger()
        
        # Console Handler
        if self.log_config.get('console', {}).get('enabled', True):
            self._add_console_handler(root_logger)
        
        # File Handler
        if self.log_config.get('file', {}).get('enabled', False):
            self._add_file_handler(root_logger)
        
        # JSON Handler
        if self.log_config.get('json', {}).get('enabled', False):
            self._add_json_handler(root_logger)
    
    def _add_console_handler(self, logger):
        """Add colored console handler"""
        console_config = self.log_config.get('console', {})
        
        if console_config.get('colored', True):
            # Colored output
            console_handler = colorlog.StreamHandler()
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s' + console_config.get(
                    'format', 
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ),
                datefmt=console_config.get('date_format', '%Y-%m-%d %H:%M:%S'),
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            # Plain output
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                console_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                datefmt=console_config.get('date_format', '%Y-%m-%d %H:%M:%S')
            )
        
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, console_config.get('level', 'INFO')))
        logger.addHandler(console_handler)
    
    def _add_file_handler(self, logger):
        """Add rotating file handler"""
        file_config = self.log_config.get('file', {})
        
        log_file = self.log_dir / file_config.get('filename', 'trading_system.log')
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=file_config.get('max_bytes', 10485760),
            backupCount=file_config.get('backup_count', 10)
        )
        
        file_formatter = logging.Formatter(
            file_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt=file_config.get('date_format', '%Y-%m-%d %H:%M:%S')
        )
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, file_config.get('level', 'DEBUG')))
        logger.addHandler(file_handler)
    
    def _add_json_handler(self, logger):
        """Add JSON file handler for structured logging"""
        json_config = self.log_config.get('json', {})
        
        log_file = self.log_dir / json_config.get('filename', 'trading_system.json')
        
        json_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=json_config.get('max_bytes', 10485760),
            backupCount=json_config.get('backup_count', 10)
        )
        
        json_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            rename_fields={'timestamp': '@timestamp', 'level': 'level', 'name': 'logger'}
        )
        
        json_handler.setFormatter(json_formatter)
        json_handler.setLevel(getattr(logging, json_config.get('level', 'INFO')))
        logger.addHandler(json_handler)
    
    def _configure_component_loggers(self):
        """Configure specific component loggers"""
        components = self.log_config.get('components', {})
        
        for component_name, level in components.items():
            component_logger = logging.getLogger(component_name)
            component_logger.setLevel(getattr(logging, level))
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name (typically __class__.__name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    def log_error(self, logger_name: str, error: Exception, context: Optional[Dict] = None):
        """
        Log an error with context
        
        Args:
            logger_name: Name of the logger
            error: Exception object
            context: Additional context dictionary
        """
        logger = self.get_logger(logger_name)
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        logger.error(f"Error occurred: {error_info}", exc_info=True)
    
    def log_trade(self, trade_info: Dict):
        """
        Log trade information
        
        Args:
            trade_info: Trade details dictionary
        """
        trade_logger = self.get_logger('TradeLog')
        trade_logger.info(f"TRADE: {trade_info}")
    
    def log_metric(self, metric_name: str, value: Any, tags: Optional[Dict] = None):
        """
        Log a metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags
        """
        metrics_logger = self.get_logger('Metrics')
        
        metric_info = {
            'metric': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'tags': tags or {}
        }
        
        metrics_logger.info(f"METRIC: {metric_info}")


# Singleton instance
_logger_system = None


def get_logger_system() -> TradingLogger:
    """Get or create singleton TradingLogger instance"""
    global _logger_system
    if _logger_system is None:
        _logger_system = TradingLogger()
    return _logger_system


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger_system = get_logger_system()
    return logger_system.get_logger(name)