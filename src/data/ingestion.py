"""
Data Ingestion Pipeline
Normalizes and stores all API data from Alpha Vantage and IBKR
Production-grade implementation for real-money trading system

Handles:
- 38 Alpha Vantage APIs (options, indicators, fundamentals, analytics, economic, sentiment)
- IBKR real-time data (bars, quotes, MOC imbalance)
- Full data normalization and validation
- Batch processing with database upserts
- Thread-safe concurrent processing
- Comprehensive error handling and recovery
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timezone, time as datetime_time
from decimal import Decimal, InvalidOperation
from collections import deque, defaultdict
from threading import Lock, Event, Thread
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Full, Empty
import logging
import json
import time
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import ThreadedConnectionPool
import redis
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from ..foundation.base_module import BaseModule

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Enumeration of supported data types"""
    OPTIONS = "options"
    GREEKS = "greeks"
    PRICE_BAR = "price_bar"
    QUOTE = "quote"
    INDICATOR = "indicator"
    FUNDAMENTAL = "fundamental"
    ANALYTICS = "analytics"
    SENTIMENT = "sentiment"
    ECONOMIC = "economic"


class IngestionStatus(Enum):
    """Status of ingestion operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


@dataclass
class IngestionTask:
    """Represents a single ingestion task"""
    task_id: str
    data_type: DataType
    source: str  # 'alpha_vantage' or 'ibkr'
    data: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    priority: int = 5  # 1-10, lower is higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: IngestionStatus = IngestionStatus.PENDING
    error_message: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker for database operations"""
    
    def __init__(self, failure_threshold: int, recovery_timeout: int, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                else:
                    raise Exception(f"Circuit breaker is open. Waiting for recovery.")
            
        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
            return result
        except self.expected_exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                elif self.state == 'half-open':
                    self.state = 'open'
            raise


class DataIngestionPipeline(BaseModule):
    """
    Production-grade data ingestion pipeline
    Handles normalization, validation, and storage of all trading data
    """
    
    def __init__(self, config: Dict[str, Any], db_connection=None, cache_manager=None):
        """
        Initialize ingestion pipeline with production configuration
        
        Args:
            config: Complete configuration from ConfigManager
            db_connection: Database connection pool
            cache_manager: Redis cache manager instance
        """
        super().__init__(config, "DataIngestionPipeline")
        
        # Load ingestion-specific configuration
        self.ingestion_config = config.get('data', {}).get('ingestion', {}).get('ingestion', {})
        self.validation_config = config.get('data', {}).get('validation', {}).get('validation', {})
        
        # Core configuration parameters
        self.batch_size = self.ingestion_config.get('batch_size', 1000)
        self.commit_interval = self.ingestion_config.get('commit_interval', 100)
        self.error_threshold = self.ingestion_config.get('error_threshold', 0.05)
        self.max_retries = self.ingestion_config.get('max_retries', 3)
        self.retry_backoff_base = 2
        self.dead_letter_enabled = self.ingestion_config.get('dead_letter_queue', True)
        
        # Validation configuration
        self.validation_enabled = self.validation_config.get('enabled', True)
        self.strict_mode = self.validation_config.get('strict_mode', False)
        self.log_validation_errors = self.validation_config.get('log_validation_errors', True)
        
        # Database configuration
        db_config = config.get('system', {}).get('database', {}).get('database', {})
        self.db_pool_size = db_config.get('pool_size', 10)
        self.db_max_overflow = db_config.get('max_overflow', 20)
        self.db_timeout = db_config.get('pool_timeout', 30)
        
        # Initialize database connection pool
        if db_connection:
            self.db_pool = db_connection
        else:
            self.db_pool = self._create_db_pool(db_config)
        
        # Cache manager
        self.cache = cache_manager
        
        # Thread-safe data structures
        self.ingestion_queue = Queue(maxsize=10000)
        self.dead_letter_queue = Queue(maxsize=1000)
        self.batch_buffer = defaultdict(list)
        self.batch_lock = Lock()
        
        # Worker pool for parallel processing
        self.worker_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="ingestion_worker")
        self.workers_active = Event()
        self.shutdown_event = Event()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'retried': 0,
            'dead_lettered': 0,
            'validation_failures': 0,
            'last_error': None,
            'processing_time_ms': deque(maxlen=1000),
            'error_rate': 0.0
        }
        self.stats_lock = Lock()
        
        # Circuit breakers for database operations
        self.db_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=psycopg2.Error
        )
        
        # Schema mappings for normalization
        self.schema_mappings = self._load_schema_mappings()
        
        # Validation rules
        self.validation_rules = self._load_validation_rules()
        
        # Start worker threads
        self._start_workers()
        
        logger.info(f"DataIngestionPipeline initialized with batch_size={self.batch_size}, "
                   f"max_retries={self.max_retries}, validation_enabled={self.validation_enabled}")
        logger.info("Ready to ingest data from 38 Alpha Vantage APIs and IBKR real-time feeds")
    
    def _create_db_pool(self, db_config: Dict[str, Any]) -> ThreadedConnectionPool:
        """Create thread-safe database connection pool"""
        try:
            # Replace environment variables in config
            import os
            connection_params = {
                'host': os.getenv('DB_HOST', db_config.get('host', 'localhost')),
                'port': db_config.get('port', 5432),
                'database': os.getenv('DB_NAME', db_config.get('name', 'trading_system')),
                'user': os.getenv('DB_USER', db_config.get('user', 'postgres')),
                'password': os.getenv('DB_PASSWORD', db_config.get('password', '')),
                'connect_timeout': db_config.get('connection_timeout', 10),
                'options': f"-c statement_timeout={db_config.get('command_timeout', 60) * 1000}"
            }
            
            pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=self.db_pool_size + self.db_max_overflow,
                **connection_params
            )
            
            logger.info(f"Database connection pool created with size {self.db_pool_size}")
            return pool
            
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    @contextmanager
    def get_db_connection(self):
        """Thread-safe database connection context manager"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _load_schema_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load schema mappings for data normalization"""
        return {
            DataType.OPTIONS: {
                'table': 'options_chain',
                'field_mappings': {
                    'contractID': 'contract_id',
                    'symbol': 'symbol',
                    'expiration': 'expiration_date',
                    'strike': 'strike_price',
                    'type': 'option_type',
                    'last': 'last_price',
                    'bid': 'bid_price',
                    'ask': 'ask_price',
                    'volume': 'volume',
                    'open_interest': 'open_interest',
                    'implied_volatility': 'implied_volatility',
                    'delta': 'delta',
                    'gamma': 'gamma',
                    'theta': 'theta',
                    'vega': 'vega',
                    'rho': 'rho'
                },
                'type_conversions': {
                    'strike_price': Decimal,
                    'last_price': Decimal,
                    'bid_price': Decimal,
                    'ask_price': Decimal,
                    'delta': Decimal,
                    'gamma': Decimal,
                    'theta': Decimal,
                    'vega': Decimal,
                    'rho': Decimal,
                    'implied_volatility': Decimal
                },
                'composite_key': ['symbol', 'expiration_date', 'strike_price', 'option_type']
            },
            DataType.PRICE_BAR: {
                'table': 'intraday_bars',
                'field_mappings': {
                    'symbol': 'symbol',
                    'timestamp': 'bar_timestamp',
                    'open': 'open_price',
                    'high': 'high_price',
                    'low': 'low_price',
                    'close': 'close_price',
                    'volume': 'volume',
                    'interval': 'interval'
                },
                'type_conversions': {
                    'open_price': Decimal,
                    'high_price': Decimal,
                    'low_price': Decimal,
                    'close_price': Decimal,
                    'volume': int,
                    'bar_timestamp': datetime
                },
                'composite_key': ['symbol', 'bar_timestamp', 'interval']
            },
            DataType.INDICATOR: {
                'table': 'technical_indicators',
                'field_mappings': {
                    'symbol': 'symbol',
                    'timestamp': 'timestamp',
                    'indicator': 'indicator_name',
                    'value': 'value',
                    'signal': 'signal_value',
                    'histogram': 'histogram_value'
                },
                'type_conversions': {
                    'value': Decimal,
                    'signal_value': Decimal,
                    'histogram_value': Decimal,
                    'timestamp': datetime
                },
                'composite_key': ['symbol', 'timestamp', 'indicator_name']
            },
            DataType.QUOTE: {
                'table': 'quotes',
                'field_mappings': {
                    'symbol': 'symbol',
                    'timestamp': 'quote_timestamp',
                    'bid': 'bid_price',
                    'ask': 'ask_price',
                    'last': 'last_price',
                    'bid_size': 'bid_size',
                    'ask_size': 'ask_size'
                },
                'type_conversions': {
                    'bid_price': Decimal,
                    'ask_price': Decimal,
                    'last_price': Decimal,
                    'bid_size': int,
                    'ask_size': int,
                    'quote_timestamp': datetime
                },
                'composite_key': ['symbol', 'quote_timestamp']
            },
            DataType.FUNDAMENTAL: {
                'table': None,  # Dynamic based on fundamental type
                'subtypes': {
                    'OVERVIEW': {
                        'table': 'company_overview',
                        'field_mappings': {
                            'Symbol': 'symbol',
                            'Name': 'company_name',
                            'Exchange': 'exchange',
                            'Sector': 'sector',
                            'Industry': 'industry',
                            'MarketCapitalization': 'market_cap',
                            'PERatio': 'pe_ratio',
                            'DividendYield': 'dividend_yield',
                            'EPS': 'eps',
                            'Beta': 'beta',
                            'HighWeek52': 'week_52_high',
                            'LowWeek52': 'week_52_low'
                        },
                        'type_conversions': {
                            'market_cap': int,
                            'pe_ratio': Decimal,
                            'dividend_yield': Decimal,
                            'eps': Decimal,
                            'beta': Decimal,
                            'week_52_high': Decimal,
                            'week_52_low': Decimal
                        },
                        'composite_key': ['symbol']
                    },
                    'EARNINGS': {
                        'table': 'earnings',
                        'field_mappings': {
                            'symbol': 'symbol',
                            'fiscalDateEnding': 'fiscal_date',
                            'reportedEPS': 'reported_eps',
                            'estimatedEPS': 'estimated_eps',
                            'surprise': 'surprise',
                            'surprisePercentage': 'surprise_percentage'
                        },
                        'type_conversions': {
                            'reported_eps': Decimal,
                            'estimated_eps': Decimal,
                            'surprise': Decimal,
                            'surprise_percentage': Decimal,
                            'fiscal_date': datetime
                        },
                        'composite_key': ['symbol', 'fiscal_date']
                    },
                    'BALANCE_SHEET': {
                        'table': 'balance_sheet',
                        'field_mappings': {
                            'symbol': 'symbol',
                            'fiscalDateEnding': 'fiscal_date',
                            'totalAssets': 'total_assets',
                            'totalLiabilities': 'total_liabilities',
                            'totalShareholderEquity': 'shareholder_equity',
                            'currentAssets': 'current_assets',
                            'currentLiabilities': 'current_liabilities',
                            'cashAndCashEquivalents': 'cash_equivalents',
                            'inventory': 'inventory',
                            'currentDebt': 'current_debt',
                            'longTermDebt': 'long_term_debt'
                        },
                        'type_conversions': {
                            'total_assets': Decimal,
                            'total_liabilities': Decimal,
                            'shareholder_equity': Decimal,
                            'current_assets': Decimal,
                            'current_liabilities': Decimal,
                            'cash_equivalents': Decimal,
                            'inventory': Decimal,
                            'current_debt': Decimal,
                            'long_term_debt': Decimal,
                            'fiscal_date': datetime
                        },
                        'composite_key': ['symbol', 'fiscal_date']
                    },
                    'INCOME_STATEMENT': {
                        'table': 'income_statement',
                        'field_mappings': {
                            'symbol': 'symbol',
                            'fiscalDateEnding': 'fiscal_date',
                            'totalRevenue': 'total_revenue',
                            'costOfRevenue': 'cost_of_revenue',
                            'grossProfit': 'gross_profit',
                            'operatingExpenses': 'operating_expenses',
                            'operatingIncome': 'operating_income',
                            'netIncome': 'net_income',
                            'ebitda': 'ebitda',
                            'ebit': 'ebit'
                        },
                        'type_conversions': {
                            'total_revenue': Decimal,
                            'cost_of_revenue': Decimal,
                            'gross_profit': Decimal,
                            'operating_expenses': Decimal,
                            'operating_income': Decimal,
                            'net_income': Decimal,
                            'ebitda': Decimal,
                            'ebit': Decimal,
                            'fiscal_date': datetime
                        },
                        'composite_key': ['symbol', 'fiscal_date']
                    },
                    'CASH_FLOW': {
                        'table': 'cash_flow',
                        'field_mappings': {
                            'symbol': 'symbol',
                            'fiscalDateEnding': 'fiscal_date',
                            'operatingCashflow': 'operating_cashflow',
                            'capitalExpenditures': 'capital_expenditures',
                            'freeCashFlow': 'free_cashflow',
                            'dividendPayout': 'dividend_payout'
                        },
                        'type_conversions': {
                            'operating_cashflow': Decimal,
                            'capital_expenditures': Decimal,
                            'free_cashflow': Decimal,
                            'dividend_payout': Decimal,
                            'fiscal_date': datetime
                        },
                        'composite_key': ['symbol', 'fiscal_date']
                    },
                    'DIVIDENDS': {
                        'table': 'dividends',
                        'field_mappings': {
                            'symbol': 'symbol',
                            'ex_dividend_date': 'ex_date',
                            'dividend_amount': 'amount',
                            'record_date': 'record_date',
                            'payment_date': 'payment_date',
                            'declaration_date': 'declaration_date'
                        },
                        'type_conversions': {
                            'amount': Decimal,
                            'ex_date': datetime,
                            'record_date': datetime,
                            'payment_date': datetime,
                            'declaration_date': datetime
                        },
                        'composite_key': ['symbol', 'ex_date']
                    }
                }
            },
            DataType.ANALYTICS: {
                'table': None,  # Dynamic based on analytics type
                'subtypes': {
                    'FIXED_WINDOW': {
                        'table': 'analytics_fixed_window',
                        'field_mappings': {
                            'symbol': 'symbol',
                            'calculation': 'calculation_type',
                            'value': 'value',
                            'start_date': 'window_start',
                            'end_date': 'window_end'
                        },
                        'type_conversions': {
                            'value': Decimal,
                            'window_start': datetime,
                            'window_end': datetime
                        },
                        'composite_key': ['symbol', 'calculation_type', 'window_start', 'window_end']
                    },
                    'SLIDING_WINDOW': {
                        'table': 'analytics_sliding_window',
                        'field_mappings': {
                            'symbol': 'symbol',
                            'calculation': 'calculation_type',
                            'value': 'value',
                            'window_size': 'window_size',
                            'timestamp': 'timestamp'
                        },
                        'type_conversions': {
                            'value': Decimal,
                            'window_size': int,
                            'timestamp': datetime
                        },
                        'composite_key': ['symbol', 'calculation_type', 'timestamp']
                    }
                }
            },
            DataType.SENTIMENT: {
                'table': 'news_sentiment',
                'field_mappings': {
                    'ticker': 'symbol',
                    'article_title': 'title',
                    'article_url': 'url',
                    'time_published': 'published_at',
                    'overall_sentiment_score': 'sentiment_score',
                    'overall_sentiment_label': 'sentiment_label',
                    'ticker_relevance_score': 'relevance_score'
                },
                'type_conversions': {
                    'sentiment_score': Decimal,
                    'relevance_score': Decimal,
                    'published_at': datetime
                },
                'composite_key': ['symbol', 'url']
            },
            DataType.ECONOMIC: {
                'table': 'economic_indicators',
                'field_mappings': {
                    'date': 'indicator_date',
                    'value': 'value',
                    'indicator': 'indicator_name',
                    'unit': 'unit'
                },
                'type_conversions': {
                    'value': Decimal,
                    'indicator_date': datetime
                },
                'composite_key': ['indicator_name', 'indicator_date']
            }
        }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration"""
        rules = self.validation_config.get('rules', {})
        
        # Greeks validation rules
        greeks_rules = rules.get('greeks', {})
        
        return {
            'greeks': {
                'delta': {
                    'min': greeks_rules.get('delta', {}).get('min', -1.0),
                    'max': greeks_rules.get('delta', {}).get('max', 1.0)
                },
                'gamma': {
                    'min': greeks_rules.get('gamma', {}).get('min', 0.0),
                    'max': greeks_rules.get('gamma', {}).get('max', float('inf'))
                },
                'theta': {
                    'calls_max': greeks_rules.get('theta', {}).get('calls_max', 0),
                    'puts_min': greeks_rules.get('theta', {}).get('puts_min', 0)
                },
                'vega': {
                    'min': greeks_rules.get('vega', {}).get('min', 0.0),
                    'max': greeks_rules.get('vega', {}).get('max', float('inf'))
                },
                'max_age_seconds': greeks_rules.get('max_age_seconds', 30)
            },
            'price': {
                'max_change_percent': rules.get('price', {}).get('max_change_percent', 20),
                'min_price': rules.get('price', {}).get('min_price', 0.01),
                'max_price': rules.get('price', {}).get('max_price', 100000)
            },
            'indicators': {
                'rsi': {'min': 0, 'max': 100},
                'macd': {'reasonable_range': [-50, 50]},
                'bollinger': {'upper_greater_than_lower': True},
                'atr': {'min': 0, 'max': float('inf')},
                'adx': {'min': 0, 'max': 100},
                'vwap': {'min': 0, 'max': float('inf')},
                'cci': {'typical_range': [-200, 200]},
                'mfi': {'min': 0, 'max': 100},
                'willr': {'min': -100, 'max': 0},
                'aroon': {'min': 0, 'max': 100},
                'stoch': {'min': 0, 'max': 100},
                'obv': {'can_be_negative': True},
                'ad': {'can_be_negative': True}
            }
        }
    
    def _start_workers(self):
        """Start background worker threads for processing"""
        self.workers_active.set()
        
        # Start batch processor thread
        batch_thread = Thread(target=self._batch_processor, name="batch_processor")
        batch_thread.daemon = True
        batch_thread.start()
        
        # Start queue processor threads
        for i in range(3):
            worker_thread = Thread(target=self._queue_processor, name=f"queue_processor_{i}")
            worker_thread.daemon = True
            worker_thread.start()
        
        # Start statistics tracker
        stats_thread = Thread(target=self._stats_tracker, name="stats_tracker")
        stats_thread.daemon = True
        stats_thread.start()
        
        logger.info("Background worker threads started")
    
    def _queue_processor(self):
        """Process tasks from the ingestion queue"""
        while self.workers_active.is_set():
            try:
                # Get task with timeout to allow checking shutdown
                task = self.ingestion_queue.get(timeout=1)
                
                if task:
                    self._process_single_task(task)
                    self.ingestion_queue.task_done()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
    
    def _process_single_task(self, task: IngestionTask):
        """Process a single ingestion task"""
        start_time = time.time()
        
        try:
            # Update task status
            task.status = IngestionStatus.PROCESSING
            
            # Add metadata to data for processing
            if task.metadata:
                task.data.update({f"_{k}": v for k, v in task.metadata.items()})
            
            # Normalize data
            normalized_data = self._normalize_data(task.data, task.data_type)
            
            # Validate if enabled
            if self.validation_enabled:
                is_valid, validation_errors = self._validate_data(normalized_data, task.data_type)
                
                if not is_valid:
                    if self.strict_mode:
                        raise ValueError(f"Validation failed: {validation_errors}")
                    elif self.log_validation_errors:
                        logger.warning(f"Validation errors for {task.task_id}: {validation_errors}")
            
            # Add metadata back for batch processing
            if task.metadata:
                normalized_data.update({f"_{k}": v for k, v in task.metadata.items()})
            
            # Add to batch buffer
            with self.batch_lock:
                self.batch_buffer[task.data_type].append(normalized_data)
                
                # Check if batch is ready
                if len(self.batch_buffer[task.data_type]) >= self.batch_size:
                    self._flush_batch(task.data_type)
            
            # Update statistics
            task.status = IngestionStatus.SUCCESS
            self._update_stats('successful', time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Failed to process task {task.task_id}: {e}")
            task.error_message = str(e)
            task.status = IngestionStatus.FAILED
            
            # Handle retry logic
            if task.retry_count < self.max_retries:
                task.retry_count += 1
                task.status = IngestionStatus.RETRY
                
                # Exponential backoff
                retry_delay = self.retry_backoff_base ** task.retry_count
                self.worker_pool.submit(self._delayed_retry, task, retry_delay)
                
                self._update_stats('retried')
            else:
                # Send to dead letter queue
                if self.dead_letter_enabled:
                    try:
                        self.dead_letter_queue.put_nowait(task)
                        self._update_stats('dead_lettered')
                    except Full:
                        logger.error(f"Dead letter queue full, dropping task {task.task_id}")
                
                self._update_stats('failed')
    
    def _normalize_data(self, data: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """Normalize data according to schema mappings"""
        if data_type not in self.schema_mappings:
            return data
        
        mapping = self.schema_mappings[data_type]
        
        # Handle subtypes for fundamentals and analytics
        if 'subtypes' in mapping:
            # Determine subtype from data
            if data_type == DataType.FUNDAMENTAL:
                subtype = data.get('_fundamental_type', 'OVERVIEW')
                mapping = mapping['subtypes'].get(subtype, {})
            elif data_type == DataType.ANALYTICS:
                subtype = data.get('_analytics_type', 'FIXED_WINDOW')
                mapping = mapping['subtypes'].get(subtype, {})
            
            if not mapping:
                logger.warning(f"No mapping found for {data_type} subtype")
                return data
        
        normalized = {}
        
        # Map fields
        for source_field, target_field in mapping.get('field_mappings', {}).items():
            if source_field in data:
                value = data[source_field]
                
                # Type conversion
                if target_field in mapping.get('type_conversions', {}):
                    converter = mapping['type_conversions'][target_field]
                    try:
                        if converter == datetime and not isinstance(value, datetime):
                            # Handle datetime conversion
                            if isinstance(value, str):
                                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            elif isinstance(value, (int, float)):
                                value = datetime.fromtimestamp(value, tz=timezone.utc)
                        elif converter == Decimal:
                            # Handle None and empty strings
                            if value is None or value == '':
                                continue
                            value = Decimal(str(value))
                        else:
                            value = converter(value)
                    except (ValueError, InvalidOperation) as e:
                        logger.warning(f"Type conversion failed for {target_field}: {e}")
                        continue
                
                normalized[target_field] = value
        
        # Add metadata
        normalized['ingested_at'] = datetime.now(timezone.utc)
        normalized['source'] = data.get('_source', 'unknown')
        
        return normalized
    
    def _validate_data(self, data: Dict[str, Any], data_type: DataType) -> Tuple[bool, List[str]]:
        """Validate data against configured rules"""
        errors = []
        
        if data_type == DataType.OPTIONS and 'delta' in data:
            # Validate Greeks
            greeks_rules = self.validation_rules.get('greeks', {})
            
            # Delta validation
            delta = float(data.get('delta', 0))
            delta_rules = greeks_rules.get('delta', {})
            if delta < delta_rules.get('min', -1.0) or delta > delta_rules.get('max', 1.0):
                errors.append(f"Delta {delta} out of range")
            
            # Gamma validation
            gamma = float(data.get('gamma', 0))
            gamma_rules = greeks_rules.get('gamma', {})
            if gamma < gamma_rules.get('min', 0):
                errors.append(f"Gamma {gamma} cannot be negative")
            
            # Theta validation for calls/puts
            theta = float(data.get('theta', 0))
            option_type = data.get('option_type', '').lower()
            theta_rules = greeks_rules.get('theta', {})
            
            if option_type == 'call' and theta > theta_rules.get('calls_max', 0):
                errors.append(f"Call theta {theta} should be <= 0")
            elif option_type == 'put' and theta < theta_rules.get('puts_min', 0):
                errors.append(f"Put theta {theta} should be >= 0")
            
            # Vega validation
            vega = float(data.get('vega', 0))
            vega_rules = greeks_rules.get('vega', {})
            if vega < vega_rules.get('min', 0):
                errors.append(f"Vega {vega} cannot be negative")
                
        elif data_type == DataType.PRICE_BAR:
            # Price validation
            price_rules = self.validation_rules.get('price', {})
            
            # Check OHLC relationships
            high = float(data.get('high_price', 0))
            low = float(data.get('low_price', 0))
            open_price = float(data.get('open_price', 0))
            close = float(data.get('close_price', 0))
            
            if high < low:
                errors.append(f"High {high} less than Low {low}")
            if close > high or close < low:
                errors.append(f"Close {close} outside High-Low range")
            if open_price > high or open_price < low:
                errors.append(f"Open {open_price} outside High-Low range")
            
            # Check price ranges
            min_price = price_rules.get('min_price', 0.01)
            max_price = price_rules.get('max_price', 100000)
            
            for price_type, price_value in [('high', high), ('low', low), ('open', open_price), ('close', close)]:
                if price_value < min_price or price_value > max_price:
                    errors.append(f"{price_type} price {price_value} outside valid range [{min_price}, {max_price}]")
                    
        elif data_type == DataType.INDICATOR:
            # Indicator validation
            indicator_rules = self.validation_rules.get('indicators', {})
            indicator_name = data.get('indicator_name', '').lower()
            value = float(data.get('value', 0))
            
            if indicator_name in indicator_rules:
                rules = indicator_rules[indicator_name]
                
                if 'min' in rules and value < rules['min']:
                    errors.append(f"{indicator_name.upper()} value {value} below minimum {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    errors.append(f"{indicator_name.upper()} value {value} above maximum {rules['max']}")
                
                # Special validations
                if indicator_name == 'bollinger' and 'upper' in data and 'lower' in data:
                    if float(data['upper']) <= float(data['lower']):
                        errors.append("Bollinger upper band must be greater than lower band")
        
        return len(errors) == 0, errors
    
    def _batch_processor(self):
        """Process batches on interval"""
        while self.workers_active.is_set():
            try:
                time.sleep(self.commit_interval)
                
                with self.batch_lock:
                    for data_type in list(self.batch_buffer.keys()):
                        if self.batch_buffer[data_type]:
                            self._flush_batch(data_type)
                            
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    def _flush_batch(self, data_type: DataType):
        """Flush a batch to the database"""
        if data_type not in self.batch_buffer or not self.batch_buffer[data_type]:
            return
        
        batch = self.batch_buffer[data_type]
        self.batch_buffer[data_type] = []
        
        try:
            mapping = self.schema_mappings.get(data_type)
            if not mapping:
                logger.error(f"No schema mapping for {data_type}")
                return
            
            # Group by table for subtypes
            if 'subtypes' in mapping:
                # Group records by their actual table
                table_batches = defaultdict(list)
                for record in batch:
                    # Determine table from record metadata
                    if data_type == DataType.FUNDAMENTAL:
                        subtype = record.pop('_fundamental_type', 'OVERVIEW')
                        subtable_mapping = mapping['subtypes'].get(subtype, {})
                        table = subtable_mapping.get('table')
                        composite_key = subtable_mapping.get('composite_key', [])
                    elif data_type == DataType.ANALYTICS:
                        subtype = record.pop('_analytics_type', 'FIXED_WINDOW')
                        subtable_mapping = mapping['subtypes'].get(subtype, {})
                        table = subtable_mapping.get('table')
                        composite_key = subtable_mapping.get('composite_key', [])
                    
                    if table:
                        table_batches[table].append((record, composite_key))
                
                # Execute batch for each table
                for table, records_and_keys in table_batches.items():
                    if records_and_keys:
                        records = [r[0] for r in records_and_keys]
                        composite_key = records_and_keys[0][1]  # All records for same table have same key
                        self._execute_batch_upsert(table, records, composite_key)
                        logger.debug(f"Flushed {len(records)} records to {table}")
            else:
                # Simple case - single table
                table = mapping['table']
                composite_key = mapping.get('composite_key', [])
                
                if table and batch:
                    self._execute_batch_upsert(table, batch, composite_key)
                    logger.debug(f"Flushed {len(batch)} records to {table}")
                
        except Exception as e:
            logger.error(f"Failed to flush batch for {data_type}: {e}")
            # Return items to buffer for retry
            self.batch_buffer[data_type].extend(batch)
    
    def _execute_batch_upsert(self, table: str, records: List[Dict], composite_key: List[str]):
        """Execute batch upsert with ON CONFLICT handling"""
        if not records:
            return
        
        # Get column names from first record
        columns = list(records[0].keys())
        
        # Build INSERT query with ON CONFLICT
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)
        
        conflict_action = "DO NOTHING"
        if composite_key:
            # Build UPDATE clause for upsert
            update_cols = [f"{col} = EXCLUDED.{col}" 
                          for col in columns if col not in composite_key]
            if update_cols:
                conflict_action = f"DO UPDATE SET {', '.join(update_cols)}"
            conflict_cols = ', '.join(composite_key)
            on_conflict = f"ON CONFLICT ({conflict_cols}) {conflict_action}"
        else:
            on_conflict = ""
        
        query = f"""
            INSERT INTO {table} ({columns_str})
            VALUES ({placeholders})
            {on_conflict}
        """
        
        # Convert records to tuples
        values = [tuple(record.get(col) for col in columns) for record in records]
        
        # Execute with circuit breaker
        def execute_batch_query():
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    execute_batch(cur, query, values, page_size=100)
        
        self.db_circuit_breaker.call(execute_batch_query)
    
    def _delayed_retry(self, task: IngestionTask, delay: float):
        """Retry a task after delay"""
        time.sleep(delay)
        try:
            self.ingestion_queue.put_nowait(task)
        except Full:
            logger.error(f"Queue full, cannot retry task {task.task_id}")
    
    def _update_stats(self, stat_type: str, processing_time: Optional[float] = None):
        """Update statistics thread-safely"""
        with self.stats_lock:
            self.stats['total_processed'] += 1
            self.stats[stat_type] += 1
            
            if processing_time:
                self.stats['processing_time_ms'].append(processing_time * 1000)
            
            # Calculate error rate
            if self.stats['total_processed'] > 0:
                self.stats['error_rate'] = self.stats['failed'] / self.stats['total_processed']
    
    def _stats_tracker(self):
        """Track and log statistics periodically"""
        while self.workers_active.is_set():
            time.sleep(60)  # Log every minute
            
            with self.stats_lock:
                if self.stats['processing_time_ms']:
                    avg_time = sum(self.stats['processing_time_ms']) / len(self.stats['processing_time_ms'])
                else:
                    avg_time = 0
                
                logger.info(f"Ingestion stats - Total: {self.stats['total_processed']}, "
                          f"Success: {self.stats['successful']}, Failed: {self.stats['failed']}, "
                          f"Error rate: {self.stats['error_rate']:.2%}, "
                          f"Avg time: {avg_time:.2f}ms")
    
    # Public interface methods
    
    def initialize(self) -> bool:
        """Initialize the pipeline and verify connections"""
        try:
            # Test database connection
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    if not result:
                        raise Exception("Database connection test failed")
            
            # Test cache connection if available
            if self.cache:
                self.cache.set("ingestion_test", "1", ttl=1)
                test_val = self.cache.get("ingestion_test")
                if test_val != "1":
                    logger.warning("Cache connection test failed")
            
            logger.info("DataIngestionPipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DataIngestionPipeline: {e}")
            return False
    
    def ingest_options_data(self, data: Dict[str, Any]) -> bool:
        """Ingest options data with Greeks"""
        task = IngestionTask(
            task_id=f"opt_{datetime.now().timestamp()}",
            data_type=DataType.OPTIONS,
            source=data.get('_source', 'alpha_vantage'),
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=3  # High priority for options
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error("Ingestion queue full for options data")
            return False
    
    def ingest_price_data(self, data: Dict[str, Any]) -> bool:
        """Ingest price bar data from IBKR"""
        task = IngestionTask(
            task_id=f"bar_{datetime.now().timestamp()}",
            data_type=DataType.PRICE_BAR,
            source='ibkr',
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=2  # Highest priority for real-time prices
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error("Ingestion queue full for price data")
            return False
    
    def ingest_quote_data(self, data: Dict[str, Any]) -> bool:
        """Ingest quote data from IBKR"""
        task = IngestionTask(
            task_id=f"quote_{datetime.now().timestamp()}",
            data_type=DataType.QUOTE,
            source='ibkr',
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=1  # Highest priority for real-time quotes
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error("Ingestion queue full for quote data")
            return False
    
    def ingest_indicator_data(self, indicator: str, data: Dict[str, Any]) -> bool:
        """Ingest technical indicator data"""
        # Add indicator name to data
        data['indicator'] = indicator
        
        task = IngestionTask(
            task_id=f"ind_{indicator}_{datetime.now().timestamp()}",
            data_type=DataType.INDICATOR,
            source='alpha_vantage',
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=5  # Medium priority
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error(f"Ingestion queue full for indicator {indicator}")
            return False
    
    def ingest_fundamental_data(self, fundamental_type: str, data: Dict[str, Any]) -> bool:
        """Ingest fundamental data (earnings, balance sheet, etc.)"""
        # Add fundamental type to metadata
        data['_fundamental_type'] = fundamental_type.upper()
        
        task = IngestionTask(
            task_id=f"fund_{fundamental_type}_{datetime.now().timestamp()}",
            data_type=DataType.FUNDAMENTAL,
            source='alpha_vantage',
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=7,  # Lower priority for fundamentals
            metadata={'fundamental_type': fundamental_type}
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error(f"Ingestion queue full for fundamental {fundamental_type}")
            return False
    
    def ingest_analytics_data(self, analytics_type: str, data: Dict[str, Any]) -> bool:
        """Ingest analytics data (fixed/sliding window)"""
        data['_analytics_type'] = analytics_type.upper()
        
        task = IngestionTask(
            task_id=f"analytics_{analytics_type}_{datetime.now().timestamp()}",
            data_type=DataType.ANALYTICS,
            source='alpha_vantage',
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=6,
            metadata={'analytics_type': analytics_type}
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error(f"Ingestion queue full for analytics {analytics_type}")
            return False
    
    def ingest_sentiment_data(self, data: Dict[str, Any]) -> bool:
        """Ingest news sentiment data"""
        task = IngestionTask(
            task_id=f"sentiment_{datetime.now().timestamp()}",
            data_type=DataType.SENTIMENT,
            source='alpha_vantage',
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=8  # Lower priority
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error("Ingestion queue full for sentiment data")
            return False
    
    def ingest_economic_data(self, indicator: str, data: Dict[str, Any]) -> bool:
        """Ingest economic indicator data"""
        data['indicator'] = indicator
        
        task = IngestionTask(
            task_id=f"econ_{indicator}_{datetime.now().timestamp()}",
            data_type=DataType.ECONOMIC,
            source='alpha_vantage',
            data=data,
            timestamp=datetime.now(timezone.utc),
            priority=9  # Lowest priority
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error(f"Ingestion queue full for economic indicator {indicator}")
            return False
    
    def ingest_moc_imbalance(self, data: Dict[str, Any]) -> bool:
        """Ingest MOC imbalance data from IBKR"""
        # MOC data is critical during 3:40-3:55 PM window
        current_time = datetime.now().time()
        is_moc_window = current_time >= datetime_time(15, 40) and \
                       current_time <= datetime_time(15, 55)
        
        task = IngestionTask(
            task_id=f"moc_{datetime.now().timestamp()}",
            data_type=DataType.QUOTE,  # Store in quotes table with special flag
            source='ibkr',
            data={**data, '_is_moc': True},
            timestamp=datetime.now(timezone.utc),
            priority=1 if is_moc_window else 4  # Highest priority during MOC window
        )
        
        try:
            self.ingestion_queue.put_nowait(task)
            return True
        except Full:
            logger.error("Ingestion queue full for MOC imbalance data")
            return False
    
    def ingest_batch(self, data_list: List[Dict[str, Any]], data_type: str) -> bool:
        """Ingest multiple records at once"""
        success_count = 0
        
        for data in data_list:
            # Route to appropriate ingestion method
            if data_type == 'options':
                success = self.ingest_options_data(data)
            elif data_type == 'price_bar':
                success = self.ingest_price_data(data)
            elif data_type == 'quote':
                success = self.ingest_quote_data(data)
            elif data_type == 'indicator':
                indicator = data.get('indicator_name', 'unknown')
                success = self.ingest_indicator_data(indicator, data)
            elif data_type.startswith('fundamental_'):
                fund_type = data_type.replace('fundamental_', '')
                success = self.ingest_fundamental_data(fund_type, data)
            elif data_type.startswith('analytics_'):
                analytics_type = data_type.replace('analytics_', '')
                success = self.ingest_analytics_data(analytics_type, data)
            elif data_type == 'sentiment':
                success = self.ingest_sentiment_data(data)
            elif data_type.startswith('economic_'):
                indicator = data_type.replace('economic_', '')
                success = self.ingest_economic_data(indicator, data)
            else:
                logger.warning(f"Unknown data type for batch ingestion: {data_type}")
                success = False
            
            if success:
                success_count += 1
        
        return success_count == len(data_list)
    
    def normalize_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Public method to normalize data - used by DataScheduler"""
        try:
            dtype = DataType[data_type.upper()]
            return self._normalize_data(data, dtype)
        except KeyError:
            logger.warning(f"Unknown data type: {data_type}")
            return data
    
    def validate_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Public validation method - implements base class interface"""
        # This method is called by DataScheduler
        # Convert schema format if needed and validate
        data_type_str = schema.get('data_type', 'unknown')
        try:
            dtype = DataType[data_type_str.upper()]
            is_valid, errors = self._validate_data(data, dtype)
            if not is_valid and self.log_validation_errors:
                logger.warning(f"Validation failed: {errors}")
            return is_valid
        except KeyError:
            logger.warning(f"Unknown data type for validation: {data_type_str}")
            return True  # Pass through if type unknown
    
    def ingest_data_from_scheduler(self, api_name: str, response: Dict[str, Any]) -> bool:
        """
        Universal ingestion method for DataScheduler
        Routes data to appropriate ingestion method based on API name
        """
        try:
            # Map API names to ingestion methods
            api_routing = {
                # Options
                'REALTIME_OPTIONS': lambda d: self.ingest_options_data(d),
                'HISTORICAL_OPTIONS': lambda d: self.ingest_options_data(d),
                
                # Indicators
                'RSI': lambda d: self.ingest_indicator_data('RSI', d),
                'MACD': lambda d: self.ingest_indicator_data('MACD', d),
                'STOCH': lambda d: self.ingest_indicator_data('STOCH', d),
                'BBANDS': lambda d: self.ingest_indicator_data('BBANDS', d),
                'ATR': lambda d: self.ingest_indicator_data('ATR', d),
                'ADX': lambda d: self.ingest_indicator_data('ADX', d),
                'VWAP': lambda d: self.ingest_indicator_data('VWAP', d),
                'EMA': lambda d: self.ingest_indicator_data('EMA', d),
                'SMA': lambda d: self.ingest_indicator_data('SMA', d),
                'AROON': lambda d: self.ingest_indicator_data('AROON', d),
                'CCI': lambda d: self.ingest_indicator_data('CCI', d),
                'MFI': lambda d: self.ingest_indicator_data('MFI', d),
                'WILLR': lambda d: self.ingest_indicator_data('WILLR', d),
                'MOM': lambda d: self.ingest_indicator_data('MOM', d),
                'AD': lambda d: self.ingest_indicator_data('AD', d),
                'OBV': lambda d: self.ingest_indicator_data('OBV', d),
                
                # Fundamentals
                'OVERVIEW': lambda d: self.ingest_fundamental_data('OVERVIEW', d),
                'EARNINGS': lambda d: self.ingest_fundamental_data('EARNINGS', d),
                'INCOME_STATEMENT': lambda d: self.ingest_fundamental_data('INCOME_STATEMENT', d),
                'BALANCE_SHEET': lambda d: self.ingest_fundamental_data('BALANCE_SHEET', d),
                'CASH_FLOW': lambda d: self.ingest_fundamental_data('CASH_FLOW', d),
                'DIVIDENDS': lambda d: self.ingest_fundamental_data('DIVIDENDS', d),
                'SPLITS': lambda d: self.ingest_fundamental_data('SPLITS', d),
                'EARNINGS_ESTIMATES': lambda d: self.ingest_fundamental_data('EARNINGS_ESTIMATES', d),
                'EARNINGS_CALENDAR': lambda d: self.ingest_fundamental_data('EARNINGS_CALENDAR', d),
                'EARNINGS_CALL_TRANSCRIPT': lambda d: self.ingest_fundamental_data('EARNINGS_CALL_TRANSCRIPT', d),
                'LISTING_STATUS': lambda d: self.ingest_fundamental_data('LISTING_STATUS', d),
                
                # Analytics
                'ANALYTICS_FIXED_WINDOW': lambda d: self.ingest_analytics_data('FIXED_WINDOW', d),
                'ANALYTICS_SLIDING_WINDOW': lambda d: self.ingest_analytics_data('SLIDING_WINDOW', d),
                
                # Economic
                'TREASURY_YIELD': lambda d: self.ingest_economic_data('TREASURY_YIELD', d),
                'FEDERAL_FUNDS_RATE': lambda d: self.ingest_economic_data('FEDERAL_FUNDS_RATE', d),
                'CPI': lambda d: self.ingest_economic_data('CPI', d),
                'INFLATION': lambda d: self.ingest_economic_data('INFLATION', d),
                'REAL_GDP': lambda d: self.ingest_economic_data('REAL_GDP', d),
                
                # Sentiment
                'NEWS_SENTIMENT': lambda d: self.ingest_sentiment_data(d),
                'TOP_GAINERS_LOSERS': lambda d: self.ingest_sentiment_data(d),
                'INSIDER_TRANSACTIONS': lambda d: self.ingest_sentiment_data(d),
            }
            
            # Get the appropriate ingestion function
            ingest_func = api_routing.get(api_name.upper())
            
            if ingest_func:
                # Add source metadata
                response['_source'] = 'alpha_vantage'
                response['_api_name'] = api_name
                return ingest_func(response)
            else:
                logger.warning(f"No ingestion handler for API: {api_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to ingest data from {api_name}: {e}")
            return False
    
    def ingest_data_from_ibkr(self, data_type: str, data: Dict[str, Any]) -> bool:
        """
        Universal ingestion method for IBKRConnectionManager
        Routes data to appropriate ingestion method based on data type
        """
        try:
            # Add source metadata
            data['_source'] = 'ibkr'
            
            # Route based on IBKR data type
            if data_type == 'bar':
                return self.ingest_price_data(data)
            elif data_type == 'quote':
                return self.ingest_quote_data(data)
            elif data_type == 'moc_imbalance':
                return self.ingest_moc_imbalance(data)
            else:
                logger.warning(f"Unknown IBKR data type: {data_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to ingest IBKR data of type {data_type}: {e}")
            return False
    
    def get_ingestion_callback(self, source: str = 'alpha_vantage'):
        """
        Returns a callback function for external components to use
        This is what DataScheduler and IBKRConnectionManager will call
        """
        if source == 'alpha_vantage':
            return self.ingest_data_from_scheduler
        elif source == 'ibkr':
            return self.ingest_data_from_ibkr
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics - thread-safe copy"""
        with self.stats_lock:
            return dict(self.stats)
    
    def reset_stats(self):
        """Reset statistics counters"""
        with self.stats_lock:
            self.stats = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'retried': 0,
                'dead_lettered': 0,
                'validation_failures': 0,
                'last_error': None,
                'processing_time_ms': deque(maxlen=1000),
                'error_rate': 0.0
            }
            logger.info("Statistics reset")
    
    def process_dead_letter_queue(self, max_items: int = 100) -> int:
        """Process items from dead letter queue manually"""
        processed = 0
        
        while processed < max_items and not self.dead_letter_queue.empty():
            try:
                task = self.dead_letter_queue.get_nowait()
                # Reset retry count and try again
                task.retry_count = 0
                task.status = IngestionStatus.PENDING
                self.ingestion_queue.put_nowait(task)
                processed += 1
            except (Empty, Full):
                break
        
        logger.info(f"Reprocessed {processed} items from dead letter queue")
        return processed
    
    @staticmethod
    def get_supported_apis() -> Dict[str, List[str]]:
        """Get list of all supported APIs for documentation/validation"""
        return {
            'alpha_vantage': [
                'REALTIME_OPTIONS', 'HISTORICAL_OPTIONS',
                'RSI', 'MACD', 'STOCH', 'BBANDS', 'ATR', 'ADX', 
                'VWAP', 'EMA', 'SMA', 'AROON', 'CCI', 'MFI', 
                'WILLR', 'MOM', 'AD', 'OBV',
                'OVERVIEW', 'EARNINGS', 'INCOME_STATEMENT', 
                'BALANCE_SHEET', 'CASH_FLOW', 'DIVIDENDS', 'SPLITS',
                'EARNINGS_ESTIMATES', 'EARNINGS_CALENDAR', 
                'EARNINGS_CALL_TRANSCRIPT', 'LISTING_STATUS',
                'ANALYTICS_FIXED_WINDOW', 'ANALYTICS_SLIDING_WINDOW',
                'TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 'CPI', 
                'INFLATION', 'REAL_GDP',
                'NEWS_SENTIMENT', 'TOP_GAINERS_LOSERS', 'INSIDER_TRANSACTIONS'
            ],
            'ibkr': [
                'bar', 'quote', 'moc_imbalance'
            ]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check pipeline health and return metrics"""
        with self.stats_lock:
            stats_copy = dict(self.stats)
        
        # Check circuit breaker states
        circuit_states = {
            'database': self.db_circuit_breaker.state
        }
        
        # Queue depths
        queue_status = {
            'ingestion_queue_size': self.ingestion_queue.qsize(),
            'dead_letter_queue_size': self.dead_letter_queue.qsize()
        }
        
        return {
            'status': 'healthy' if stats_copy['error_rate'] < self.error_threshold else 'degraded',
            'statistics': stats_copy,
            'circuit_breakers': circuit_states,
            'queues': queue_status,
            'workers_active': self.workers_active.is_set()
        }
    
    def shutdown(self) -> bool:
        """Gracefully shutdown the pipeline"""
        logger.info("Shutting down DataIngestionPipeline...")
        
        # Stop accepting new tasks
        self.workers_active.clear()
        
        # Process remaining queued items
        timeout = time.time() + 30  # 30 second grace period
        while not self.ingestion_queue.empty() and time.time() < timeout:
            time.sleep(0.1)
        
        # Flush remaining batches
        with self.batch_lock:
            for data_type in self.batch_buffer:
                if self.batch_buffer[data_type]:
                    self._flush_batch(data_type)
        
        # Shutdown worker pool
        self.worker_pool.shutdown(wait=True)
        
        # Close database pool
        if hasattr(self.db_pool, 'closeall'):
            self.db_pool.closeall()
        
        logger.info("DataIngestionPipeline shutdown complete")
        return True