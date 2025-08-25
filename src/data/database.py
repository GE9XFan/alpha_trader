"""
Database Manager - Implementation Plan Week 1 Day 5
Stores both IBKR execution data and Alpha Vantage analytics
Tech Spec Section 4 - Database Schema
"""
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import redis
import json
from typing import Any, Optional, Dict

from src.core.config import config
from src.core.logger import get_logger
from src.core.exceptions import DatabaseException


logger = get_logger(__name__)


class DatabaseManager:
    """
    Database layer - REUSED BY ALL COMPONENTS
    Stores both IBKR execution data and Alpha Vantage analytics
    """
    
    def __init__(self):
        self.db_config = config.database['postgres']
        self.redis_config = config.database['redis']
        
        # PostgreSQL connection pool
        self.pg_pool = None
        self.redis = None
        
        self._init_connections()
        self._init_tables()
    
    def _init_connections(self):
        """Initialize database connections"""
        try:
            # PostgreSQL pool
            self.pg_pool = ThreadedConnectionPool(
                1, 20,
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config.get('password', '')
            )
            
            # Redis connection
            self.redis = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config.get('db', 0),
                password=self.redis_config.get('password', ''),
                decode_responses=True
            )
            
            logger.info("Database connections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseException(f"Database initialization failed: {e}")
    
    def _init_tables(self):
        """Create tables - stores both IBKR and AV data"""
        with self.get_db() as conn:
            cur = conn.cursor()
            
            # Trades table - execution through IBKR, Greeks from AV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    mode VARCHAR(10),
                    symbol VARCHAR(10),
                    option_type VARCHAR(4),
                    strike DECIMAL(10,2),
                    expiry DATE,
                    action VARCHAR(10),
                    quantity INT,
                    fill_price DECIMAL(10,4),
                    commission DECIMAL(10,2),
                    realized_pnl DECIMAL(10,2),
                    -- Greeks at entry (from Alpha Vantage)
                    entry_delta DECIMAL(6,4),
                    entry_gamma DECIMAL(6,4),
                    entry_theta DECIMAL(8,4),
                    entry_vega DECIMAL(8,4),
                    entry_rho DECIMAL(6,4),
                    entry_iv DECIMAL(6,4),
                    -- Greeks at exit (from Alpha Vantage)
                    exit_delta DECIMAL(6,4),
                    exit_gamma DECIMAL(6,4),
                    exit_theta DECIMAL(8,4),
                    exit_vega DECIMAL(8,4),
                    exit_iv DECIMAL(6,4)
                )
            """)
            
            # Signals table - tracks all signals with AV data
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    symbol VARCHAR(10),
                    signal_type VARCHAR(20),
                    confidence DECIMAL(4,3),
                    features JSONB,
                    ibkr_features JSONB,
                    av_technical JSONB,
                    av_options JSONB,
                    av_sentiment JSONB,
                    executed BOOLEAN DEFAULT FALSE,
                    trade_id INT REFERENCES trades(id)
                )
            """)
            
            # Alpha Vantage API monitoring
            cur.execute("""
                CREATE TABLE IF NOT EXISTS av_api_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    endpoint VARCHAR(50),
                    symbol VARCHAR(10),
                    response_time_ms INT,
                    cache_hit BOOLEAN,
                    rate_limit_remaining INT,
                    response_size_bytes INT
                )
            """)
            
            # Options chain snapshots from Alpha Vantage
            cur.execute("""
                CREATE TABLE IF NOT EXISTS av_options_snapshots (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    symbol VARCHAR(10),
                    expiry DATE,
                    strike DECIMAL(10,2),
                    option_type VARCHAR(4),
                    bid DECIMAL(10,4),
                    ask DECIMAL(10,4),
                    last DECIMAL(10,4),
                    volume INT,
                    open_interest INT,
                    -- Greeks from AV (not calculated!)
                    delta DECIMAL(6,4),
                    gamma DECIMAL(6,4),
                    theta DECIMAL(8,4),
                    vega DECIMAL(8,4),
                    rho DECIMAL(6,4),
                    implied_volatility DECIMAL(6,4)
                )
            """)
            
            # Cache metrics
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cache_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    cache_type VARCHAR(20),
                    hits INT,
                    misses INT,
                    avg_response_cached_ms DECIMAL(8,2),
                    avg_response_uncached_ms DECIMAL(8,2),
                    memory_usage_mb INT
                )
            """)
            
            conn.commit()
            logger.info("Database tables initialized")
    
    @contextmanager
    def get_db(self):
        """Get database connection - reused everywhere"""
        if not self.pg_pool:
            raise DatabaseException("PostgreSQL pool not initialized")
        conn = self.pg_pool.getconn()
        try:
            yield conn
        finally:
            self.pg_pool.putconn(conn)
    
    def cache_av_response(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache Alpha Vantage responses with appropriate TTL"""
        if not self.redis:
            return  # Skip caching if Redis not available
            
        if ttl is None:
            # Use default TTLs based on data type
            if 'options' in key:
                ttl = 60  # 1 minute for options
            elif 'indicator' in key:
                ttl = 300  # 5 minutes for indicators
            else:
                ttl = 900  # 15 minutes default
        
        self.redis.setex(key, ttl, json.dumps(value))
    
    def get_av_cache(self, key: str) -> Optional[Any]:
        """Get from Alpha Vantage cache"""
        if not self.redis:
            return None
        value = self.redis.get(key)
        return json.loads(value) if value else None
    
    def log_av_api_call(self, endpoint: str, symbol: str, 
                       response_time_ms: int, cache_hit: bool,
                       rate_limit_remaining: int):
        """Log Alpha Vantage API metrics"""
        with self.get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO av_api_metrics 
                (endpoint, symbol, response_time_ms, cache_hit, rate_limit_remaining)
                VALUES (%s, %s, %s, %s, %s)
            """, (endpoint, symbol, response_time_ms, cache_hit, rate_limit_remaining))
            conn.commit()


# ONE DATABASE MANAGER FOR EVERYTHING
db = DatabaseManager()
