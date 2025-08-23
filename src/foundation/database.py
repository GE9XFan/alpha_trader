"""
Production-grade database management
Real connections, real pooling, real health checks
Zero hardcoded values - all from environment
"""
import os
import time
from contextlib import contextmanager
from typing import Optional, Generator, Dict, Any, List, Tuple
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from datetime import datetime

from .logger import get_logger
from .exceptions import DatabaseException, TimeoutException


class DatabaseManager:
    """
    Real PostgreSQL connection management
    All configuration from environment - zero hardcoding
    """
    
    def __init__(self):
        """Initialize database manager with environment configuration"""
        self.logger = get_logger(__name__)
        self.pool: Optional[pool.ThreadedConnectionPool] = None
        
        # Get ALL configuration from environment - NO DEFAULTS
        self.config = {
            'host': os.environ['DB_HOST'],
            'port': int(os.environ['DB_PORT']),
            'database': os.environ['DB_NAME'],
            'user': os.environ['DB_USER'],
            'password': os.environ['DB_PASSWORD'],
            'minconn': int(os.environ['DB_POOL_MIN_SIZE']),
            'maxconn': int(os.environ['DB_POOL_MAX_SIZE']),
            'connect_timeout': int(os.environ['DB_POOL_TIMEOUT']),
        }
        
        # Retry configuration from environment
        self.max_retries = int(os.environ['RETRY_MAX_ATTEMPTS'])
        self.backoff_factor = float(os.environ['RETRY_BACKOFF_FACTOR'])
        self.max_delay = int(os.environ['RETRY_MAX_DELAY'])
        
        # Performance tracking from environment
        self.slow_query_threshold = float(os.environ['PERF_SLOW_QUERY_THRESHOLD_MS'])
        
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Create REAL connection pool to PostgreSQL"""
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                self.logger.info(
                    "Initializing database connection pool",
                    host=self.config['host'],
                    port=self.config['port'],
                    database=self.config['database'],
                    min_size=self.config['minconn'],
                    max_size=self.config['maxconn']
                )
                
                self.pool = pool.ThreadedConnectionPool(
                    minconn=self.config['minconn'],
                    maxconn=self.config['maxconn'],
                    host=self.config['host'],
                    port=self.config['port'],
                    database=self.config['database'],
                    user=self.config['user'],
                    password=self.config['password'],
                    connect_timeout=self.config['connect_timeout']
                )
                
                # Test the connection
                with self.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    cur.close()
                
                self.logger.info("Database pool initialized successfully")
                return
                
            except Exception as e:
                last_exception = e
                attempt += 1
                
                if attempt >= self.max_retries:
                    raise DatabaseException(
                        f"Failed to initialize database pool after {self.max_retries} attempts: {e}"
                    )
                
                # Calculate backoff delay
                delay = min(self.backoff_factor ** attempt, self.max_delay)
                
                self.logger.warning(
                    f"Database connection failed, retrying in {delay} seconds",
                    attempt=attempt,
                    max_attempts=self.max_retries,
                    error=str(e)
                )
                
                time.sleep(delay)
    
    @contextmanager
    def get_connection(self, dict_cursor: bool = False) -> Generator:
        """
        Get REAL database connection from pool
        
        Args:
            dict_cursor: If True, use RealDictCursor for dict-like results
            
        Yields:
            Database connection
        """
        conn = None
        start_time = time.time()
        
        try:
            conn = self.pool.getconn()
            
            if conn is None:
                raise DatabaseException("Failed to get connection from pool")
            
            # Set cursor factory if requested
            if dict_cursor:
                conn.cursor_factory = RealDictCursor
            
            yield conn
            
            # Commit if no exception
            conn.commit()
            
            # Log performance
            duration_ms = (time.time() - start_time) * 1000
            if duration_ms > self.slow_query_threshold:
                self.logger.warning(
                    "Slow database operation",
                    duration_ms=duration_ms,
                    threshold_ms=self.slow_query_threshold
                )
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            self.logger.error("Database operation failed", error=str(e))
            raise DatabaseException(f"Database operation failed: {e}")
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise
            
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch_all: bool = True,
        dict_cursor: bool = False
    ) -> Optional[List[Any]]:
        """
        Execute a query with automatic retry and performance tracking
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_all: If True, fetch all results
            dict_cursor: If True, return results as dictionaries
            
        Returns:
            Query results or None
        """
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                start_time = time.time()
                
                with self.get_connection(dict_cursor=dict_cursor) as conn:
                    cur = conn.cursor()
                    cur.execute(query, params)
                    
                    if fetch_all and cur.description:
                        results = cur.fetchall()
                    else:
                        results = None
                    
                    cur.close()
                    
                    # Log performance
                    duration_ms = (time.time() - start_time) * 1000
                    self.logger.debug(
                        "Query executed",
                        duration_ms=duration_ms,
                        rows_affected=cur.rowcount
                    )
                    
                    return results
                    
            except Exception as e:
                last_exception = e
                attempt += 1
                
                if attempt >= self.max_retries:
                    raise DatabaseException(
                        f"Query failed after {self.max_retries} attempts: {e}"
                    )
                
                delay = min(self.backoff_factor ** attempt, self.max_delay)
                self.logger.warning(
                    f"Query failed, retrying in {delay} seconds",
                    attempt=attempt,
                    error=str(e)
                )
                time.sleep(delay)
    
    def bulk_insert(
        self,
        table: str,
        columns: List[str],
        data: List[Tuple],
        returning: Optional[str] = None
    ) -> Optional[List[Any]]:
        """
        Perform bulk insert with optimal performance
        
        Args:
            table: Table name
            columns: Column names
            data: List of tuples with data
            returning: Optional RETURNING clause
            
        Returns:
            Returned values if RETURNING specified
        """
        if not data:
            return None
        
        # Build query - no hardcoded values
        placeholders = ','.join(['%s'] * len(columns))
        columns_str = ','.join(columns)
        
        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
        
        if returning:
            query += f" RETURNING {returning}"
        
        results = []
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            for row in data:
                cur.execute(query, row)
                if returning:
                    results.append(cur.fetchone())
            
            cur.close()
        
        return results if returning else None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Test REAL database connectivity and performance
        
        Returns:
            Health check results
        """
        try:
            start_time = time.time()
            
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                result = cur.fetchone()
                cur.close()
                
                # Get pool statistics
                pool_size = self.pool.maxconn if self.pool else 0
                
                duration_ms = (time.time() - start_time) * 1000
                
                return {
                    'healthy': result[0] == 1,
                    'response_time_ms': duration_ms,
                    'pool_size': pool_size,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def close(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            self.logger.info("Database connection pool closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()