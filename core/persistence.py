"""
Event Store for persisting all system events.
This provides complete audit trail and event sourcing capabilities.
"""

import json
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from .message import Message

logger = logging.getLogger(__name__)


class EventStore:
    """
    PostgreSQL-based event store for persisting all system events.
    
    Features:
    - Automatic table creation
    - Connection pooling for performance
    - JSON storage for flexible schemas
    - Time-based partitioning support
    - Event replay capabilities
    """
    
    def __init__(self, connection_string: str, pool_size: int = 10):
        """
        Initialize the event store.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Size of the connection pool
        """
        self.connection_string = connection_string
        
        # Create connection pool for better performance
        self.pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=pool_size,
            dsn=connection_string
        )
        
        self._init_database()
        logger.info(f"Event store initialized with pool size {pool_size}")
    
    def _init_database(self):
        """
        Create the events table and indexes if they don't exist.
        This is idempotent - safe to call multiple times.
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Create main events table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id UUID PRIMARY KEY,
                        correlation_id UUID NOT NULL,
                        event_type TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                
                # Create indexes for common queries
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_events_type 
                    ON events(event_type);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_events_created 
                    ON events(created_at DESC);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_events_correlation 
                    ON events(correlation_id);
                """)
                
                # Index for querying by symbol (common use case)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_events_symbol 
                    ON events((payload->>'symbol')) 
                    WHERE payload ? 'symbol';
                """)
                
                # Index for event type and time range queries
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_events_type_time 
                    ON events(event_type, created_at DESC);
                """)
                
                conn.commit()
                logger.info("Event store database initialized")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def save(self, message: Message) -> None:
        """
        Persist a message to the event store.
        
        This is a critical operation - if this fails, we log but don't crash.
        
        Args:
            message: The message to persist
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO events (
                        id, correlation_id, event_type, 
                        payload, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    message.id,
                    message.correlation_id,
                    message.event_type,
                    Json(message.data),
                    Json(message.metadata),
                    message.timestamp
                ))
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save message {message.id}: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def get_events(self, 
                   event_type: Optional[str] = None,
                   correlation_id: Optional[str] = None,
                   symbol: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000) -> List[Message]:
        """
        Query events from the store.
        
        Args:
            event_type: Filter by event type (supports LIKE patterns)
            correlation_id: Filter by correlation ID
            symbol: Filter by symbol in payload
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of events to return
            
        Returns:
            List of Message objects
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query dynamically
                query = "SELECT * FROM events WHERE 1=1"
                params = []
                
                if event_type:
                    query += " AND event_type LIKE %s"
                    params.append(event_type.replace('*', '%'))
                
                if correlation_id:
                    query += " AND correlation_id = %s"
                    params.append(correlation_id)
                
                if symbol:
                    query += " AND payload->>'symbol' = %s"
                    params.append(symbol)
                
                if start_time:
                    query += " AND created_at >= %s"
                    params.append(start_time)
                
                if end_time:
                    query += " AND created_at <= %s"
                    params.append(end_time)
                
                query += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                # Convert rows to Message objects
                messages = []
                for row in rows:
                    message = Message(
                        id=str(row['id']),
                        correlation_id=str(row['correlation_id']),
                        event_type=row['event_type'],
                        data=row['payload'],
                        timestamp=row['created_at'],
                        metadata=row['metadata']
                    )
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def get_latest_event(self, event_type: str, symbol: Optional[str] = None) -> Optional[Message]:
        """
        Get the most recent event of a specific type.
        
        Args:
            event_type: The event type to look for
            symbol: Optional symbol filter
            
        Returns:
            The latest Message or None if not found
        """
        events = self.get_events(
            event_type=event_type,
            symbol=symbol,
            limit=1
        )
        return events[0] if events else None
    
    def replay_events(self,
                     start_time: datetime,
                     end_time: Optional[datetime] = None,
                     event_types: Optional[List[str]] = None) -> List[Message]:
        """
        Replay events from a specific time period.
        Useful for backtesting and recovery.
        
        Args:
            start_time: Start of replay period
            end_time: End of replay period (default: now)
            event_types: Optional list of event types to replay
            
        Returns:
            List of events in chronological order
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT * FROM events 
                    WHERE created_at >= %s
                """
                params: List[Any] = [start_time]
                
                if end_time:
                    query += " AND created_at <= %s"
                    params.append(end_time)
                
                if event_types:
                    placeholders = ','.join(['%s'] * len(event_types))
                    query += f" AND event_type IN ({placeholders})"
                    params.extend(list(event_types))
                
                query += " ORDER BY created_at ASC"
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                messages = []
                for row in rows:
                    message = Message(
                        id=str(row['id']),
                        correlation_id=str(row['correlation_id']),
                        event_type=row['event_type'],
                        data=row['payload'],
                        timestamp=row['created_at'],
                        metadata=row['metadata']
                    )
                    messages.append(message)
                
                logger.info(f"Replaying {len(messages)} events from {start_time}")
                return messages
                
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the event store.
        
        Returns:
            Dictionary with store statistics
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Total event count
                cur.execute("SELECT COUNT(*) as count FROM events")
                total_events = cur.fetchone()['count']
                
                # Events by type
                cur.execute("""
                    SELECT event_type, COUNT(*) as count 
                    FROM events 
                    GROUP BY event_type 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                top_event_types = cur.fetchall()
                
                # Recent event rate
                cur.execute("""
                    SELECT COUNT(*) as count 
                    FROM events 
                    WHERE created_at > NOW() - INTERVAL '1 minute'
                """)
                events_per_minute = cur.fetchone()['count']
                
                return {
                    'total_events': total_events,
                    'events_per_minute': events_per_minute,
                    'top_event_types': top_event_types
                }
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
        finally:
            self.pool.putconn(conn)
    
    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, 'pool'):
            self.pool.closeall()
            logger.info("Event store connections closed")