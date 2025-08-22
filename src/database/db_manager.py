#!/usr/bin/env python3
"""
Database Manager - PostgreSQL connection and session management
Phase 0: Foundation Setup
"""

import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Generator
from datetime import datetime

from sqlalchemy import create_engine, text, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from src.foundation.config_manager import get_config_manager


class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self):
        """Initialize database manager"""
        self.config = get_config_manager()
        self.db_config = self.config.db_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create engine
        self.engine = self._create_engine()
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Test connection
        self._test_connection()
        
        self.logger.info("DatabaseManager initialized successfully")
    
    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling"""
        connection_string = self.config.get_db_connection_string()
        
        # Get pool settings from config
        pool_settings = {
            'pool_size': self.db_config.get('pool_size', 20),
            'max_overflow': self.db_config.get('max_overflow', 40),
            'pool_timeout': self.db_config.get('pool_timeout', 30),
            'pool_recycle': self.db_config.get('pool_recycle', 3600),
            'pool_pre_ping': True,  # Verify connections before using
            'echo': self.db_config.get('echo', False),
            'echo_pool': self.db_config.get('echo_pool', False)
        }
        
        engine = create_engine(
            connection_string,
            poolclass=pool.QueuePool,
            **pool_settings
        )
        
        self.logger.info(f"Created database engine with pool size {pool_settings['pool_size']}")
        return engine
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            self.logger.info("Database connection test successful")
        except SQLAlchemyError as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup
        
        Yields:
            Session object
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a raw SQL query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
                return []
        except SQLAlchemyError as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_many(self, query: str, data: List[Dict]) -> int:
        """
        Execute bulk insert/update
        
        Args:
            query: SQL query string
            data: List of parameter dictionaries
            
        Returns:
            Number of affected rows
        """
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(query), data)
                return result.rowcount
        except SQLAlchemyError as e:
            self.logger.error(f"Bulk operation failed: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table_name
            )
        """
        result = self.execute_query(query, {'table_name': table_name})
        return result[0]['exists'] if result else False
    
    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows
        """
        if not self.table_exists(table_name):
            return 0
        
        # Use parameterized query safely
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query)
        return result[0]['count'] if result else 0
    
    def close(self):
        """Close database connections"""
        self.engine.dispose()
        self.logger.info("Database connections closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Singleton instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get or create singleton DatabaseManager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager