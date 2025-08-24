#!/usr/bin/env python3
"""
Database Manager Module
Handles PostgreSQL and Redis connections, providing unified data persistence.
Used by all components for storing trades, signals, and positions.
"""

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
from contextlib import contextmanager
import redis
import json
from typing import Dict, List, Optional, Any, Generator
from datetime import datetime, timedelta
import pandas as pd
import logging
from dataclasses import dataclass, asdict
import asyncio
from asyncpg import create_pool, Pool

from src.core.config import get_config, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record structure"""
    timestamp: datetime
    mode: str  # 'paper' or 'live'
    symbol: str
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float = 0.65
    pnl: float = 0.0
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Signal:
    """Signal record structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY_CALL', 'BUY_PUT', 'HOLD', 'CLOSE'
    confidence: float
    features: Dict[str, float]  # Feature vector as JSON
    executed: bool = False
    trade_id: Optional[int] = None
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        data['features'] = Json(data['features'])  # Convert to PostgreSQL JSON
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class Position:
    """Position record structure"""
    symbol: str
    option_type: str
    strike: float
    expiry: datetime
    quantity: int
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return asdict(self)


class DatabaseManager:
    """
    Database layer - REUSED BY ALL COMPONENTS
    Handles PostgreSQL for persistent storage and Redis for caching.
    Paper and live trading use same schema with mode field.
    """
    
    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize DatabaseManager
        
        Args:
            config: Trading configuration (uses global if not provided)
        """
        self.config = config or get_config()
        
        # PostgreSQL connection pool
        self.pg_pool: Optional[ThreadedConnectionPool] = None
        self.async_pg_pool: Optional[Pool] = None
        
        # Redis connection
        self.redis: Optional[redis.Redis] = None
        
        # Cache settings
        self.default_cache_ttl = 60  # seconds
        
        # Performance tracking
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize connections
        self._init_connections()
        
        # Create tables if not exist
        self._init_tables()
        
        logger.info("DatabaseManager initialized")
    
    def _init_connections(self) -> None:
        """
        Initialize database connections
        """
        # TODO: Implement connection initialization
        # 1. Create PostgreSQL connection pool
        # 2. Test PostgreSQL connection
        # 3. Create Redis connection
        # 4. Test Redis connection
        # 5. Set up connection error handlers
        # 6. Log successful connections
        pass
    
    def _init_tables(self) -> None:
        """
        Create tables if they don't exist - same schema for paper and live
        """
        # TODO: Implement table creation
        # 1. Create trades table with all fields
        # 2. Create signals table with JSONB features
        # 3. Create positions table
        # 4. Create system_status table
        # 5. Create performance_metrics table
        # 6. Create indexes for performance
        # 7. Set up constraints and foreign keys
        pass
    
    @contextmanager
    def get_db(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get database connection from pool - reused everywhere
        
        Yields:
            Database connection
        """
        # TODO: Implement connection context manager
        # 1. Get connection from pool
        # 2. Yield connection
        # 3. Handle exceptions
        # 4. Return connection to pool
        # 5. Track connection usage
        pass
    
    async def get_async_db(self) -> Pool:
        """
        Get async database pool for high-performance operations
        
        Returns:
            Async connection pool
        """
        # TODO: Implement async pool getter
        # 1. Create pool if not exists
        # 2. Test connection
        # 3. Return pool
        pass
    
    # ============= Trade Operations =============
    
    def insert_trade(self, trade: Trade) -> int:
        """
        Insert trade record
        
        Args:
            trade: Trade object
            
        Returns:
            Trade ID
        """
        # TODO: Implement trade insertion
        # 1. Get database connection
        # 2. Insert trade record
        # 3. Return generated ID
        # 4. Update cache
        # 5. Handle conflicts
        pass
    
    def update_trade_pnl(self, trade_id: int, pnl: float) -> bool:
        """
        Update trade P&L
        
        Args:
            trade_id: Trade ID
            pnl: Profit/loss amount
            
        Returns:
            True if updated successfully
        """
        # TODO: Implement P&L update
        # 1. Update trade record
        # 2. Invalidate cache
        # 3. Return success status
        pass
    
    def get_trades(self, 
                  symbol: Optional[str] = None,
                  date: Optional[datetime] = None,
                  mode: Optional[str] = None,
                  limit: int = 100) -> List[Trade]:
        """
        Get trades with filters
        
        Args:
            symbol: Filter by symbol
            date: Filter by date
            mode: Filter by mode (paper/live)
            limit: Maximum records to return
            
        Returns:
            List of Trade objects
        """
        # TODO: Implement trade retrieval
        # 1. Build query with filters
        # 2. Check cache first
        # 3. Execute query
        # 4. Convert to Trade objects
        # 5. Cache results
        # 6. Return trades
        pass
    
    def get_daily_trades_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get daily trading summary
        
        Args:
            date: Date for summary (today if None)
            
        Returns:
            Summary statistics dictionary
        """
        # TODO: Implement daily summary
        # 1. Get all trades for date
        # 2. Calculate total trades
        # 3. Calculate win rate
        # 4. Calculate total P&L
        # 5. Find best/worst trades
        # 6. Return summary dict
        pass
    
    # ============= Signal Operations =============
    
    def insert_signal(self, signal: Signal) -> int:
        """
        Insert signal record
        
        Args:
            signal: Signal object
            
        Returns:
            Signal ID
        """
        # TODO: Implement signal insertion
        # 1. Convert features to JSONB
        # 2. Insert signal record
        # 3. Return generated ID
        # 4. Update cache
        pass
    
    def mark_signal_executed(self, signal_id: int, trade_id: int) -> bool:
        """
        Mark signal as executed with trade reference
        
        Args:
            signal_id: Signal ID
            trade_id: Associated trade ID
            
        Returns:
            True if updated successfully
        """
        # TODO: Implement signal execution marking
        # 1. Update signal record
        # 2. Set executed flag and trade_id
        # 3. Invalidate cache
        # 4. Return success status
        pass
    
    def get_signals(self,
                   symbol: Optional[str] = None,
                   executed: Optional[bool] = None,
                   min_confidence: Optional[float] = None,
                   limit: int = 100) -> List[Signal]:
        """
        Get signals with filters
        
        Args:
            symbol: Filter by symbol
            executed: Filter by execution status
            min_confidence: Minimum confidence threshold
            limit: Maximum records to return
            
        Returns:
            List of Signal objects
        """
        # TODO: Implement signal retrieval
        # 1. Build query with filters
        # 2. Execute query
        # 3. Convert to Signal objects
        # 4. Return signals
        pass
    
    def get_signal_accuracy(self, lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate signal accuracy metrics
        
        Args:
            lookback_days: Days to look back
            
        Returns:
            Accuracy metrics dictionary
        """
        # TODO: Implement accuracy calculation
        # 1. Get executed signals with trades
        # 2. Calculate win rate by signal type
        # 3. Calculate average confidence
        # 4. Calculate execution rate
        # 5. Return metrics
        pass
    
    # ============= Position Operations =============
    
    def upsert_position(self, position: Position) -> bool:
        """
        Insert or update position
        
        Args:
            position: Position object
            
        Returns:
            True if successful
        """
        # TODO: Implement position upsert
        # 1. Check if position exists
        # 2. Update if exists, insert if not
        # 3. Update Greeks
        # 4. Invalidate cache
        # 5. Return success status
        pass
    
    def get_positions(self, active_only: bool = True) -> List[Position]:
        """
        Get current positions
        
        Args:
            active_only: Only return non-zero positions
            
        Returns:
            List of Position objects
        """
        # TODO: Implement position retrieval
        # 1. Build query with filters
        # 2. Check cache first
        # 3. Execute query
        # 4. Convert to Position objects
        # 5. Cache results
        # 6. Return positions
        pass
    
    def close_position(self, symbol: str) -> bool:
        """
        Mark position as closed
        
        Args:
            symbol: Position symbol
            
        Returns:
            True if closed successfully
        """
        # TODO: Implement position closing
        # 1. Set quantity to 0
        # 2. Update timestamp
        # 3. Invalidate cache
        # 4. Return success status
        pass
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary with Greeks
        
        Returns:
            Portfolio summary dictionary
        """
        # TODO: Implement portfolio summary
        # 1. Get all active positions
        # 2. Calculate total value
        # 3. Aggregate Greeks
        # 4. Calculate P&L
        # 5. Return summary
        pass
    
    # ============= Cache Operations =============
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in Redis cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        # TODO: Implement cache set
        # 1. Serialize value to JSON
        # 2. Set in Redis with TTL
        # 3. Track cache operation
        # 4. Return success status
        pass
    
    def cache_get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # TODO: Implement cache get
        # 1. Get from Redis
        # 2. Deserialize from JSON
        # 3. Track hit/miss
        # 4. Return value
        pass
    
    def cache_delete(self, pattern: str) -> int:
        """
        Delete cache entries matching pattern
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            Number of keys deleted
        """
        # TODO: Implement cache deletion
        # 1. Find matching keys
        # 2. Delete keys
        # 3. Return count
        pass
    
    # ============= Performance Operations =============
    
    def log_performance_metric(self, 
                              metric_name: str,
                              value: float,
                              tags: Optional[Dict[str, str]] = None) -> bool:
        """
        Log performance metric
        
        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
            
        Returns:
            True if logged successfully
        """
        # TODO: Implement metric logging
        # 1. Insert into performance_metrics
        # 2. Update rolling statistics
        # 3. Check for alerts
        # 4. Return success status
        pass
    
    def get_performance_metrics(self, 
                               metric_name: str,
                               start_time: datetime,
                               end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get performance metrics as DataFrame
        
        Args:
            metric_name: Metric to retrieve
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            DataFrame with metrics
        """
        # TODO: Implement metric retrieval
        # 1. Build query with time range
        # 2. Execute query
        # 3. Convert to DataFrame
        # 4. Add rolling statistics
        # 5. Return DataFrame
        pass
    
    # ============= System Operations =============
    
    def set_system_status(self, key: str, value: str) -> bool:
        """
        Set system status flag
        
        Args:
            key: Status key
            value: Status value
            
        Returns:
            True if set successfully
        """
        # TODO: Implement status setting
        # 1. Upsert into system_status
        # 2. Update cache
        # 3. Log change
        # 4. Return success status
        pass
    
    def get_system_status(self, key: str) -> Optional[str]:
        """
        Get system status flag
        
        Args:
            key: Status key
            
        Returns:
            Status value or None
        """
        # TODO: Implement status retrieval
        # 1. Check cache first
        # 2. Query database
        # 3. Cache result
        # 4. Return value
        pass
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if backup successful
        """
        # TODO: Implement database backup
        # 1. Use pg_dump for PostgreSQL
        # 2. Export Redis data
        # 3. Compress backup
        # 4. Verify backup integrity
        # 5. Return success status
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Statistics dictionary
        """
        # TODO: Implement statistics gathering
        # 1. Get table sizes
        # 2. Get cache statistics
        # 3. Get query performance
        # 4. Get connection pool stats
        # 5. Return comprehensive stats
        pass
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old data
        
        Args:
            days_to_keep: Days of data to retain
            
        Returns:
            Dictionary with deletion counts
        """
        # TODO: Implement data cleanup
        # 1. Delete old trades
        # 2. Delete old signals
        # 3. Delete old metrics
        # 4. Vacuum database
        # 5. Return deletion counts
        pass
    
    def close(self) -> None:
        """
        Close all database connections
        """
        # TODO: Implement connection cleanup
        # 1. Close PostgreSQL pool
        # 2. Close Redis connection
        # 3. Close async pool if exists
        # 4. Log closure
        pass
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()