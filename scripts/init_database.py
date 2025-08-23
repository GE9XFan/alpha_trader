#!/usr/bin/env python3
"""
Initialize the AlphaTrader database.
Creates all necessary tables and indexes.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.persistence import EventStore
from core.config import ConfigLoader


def init_database():
    """Initialize the database with required tables."""
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_system_config()
    
    # Build connection string from config
    db_config = config.get('database', {})
    connection_string = (
        f"postgresql://{db_config.get('user', 'alphatrader')}:"
        f"{db_config.get('password', 'alphatrader_dev')}@"
        f"{db_config.get('host', 'localhost')}:"
        f"{db_config.get('port', 5432)}/"
        f"{db_config.get('name', 'alphatrader')}"
    )
    
    print(f"Initializing database: {db_config.get('name', 'alphatrader')}")
    
    try:
        # Create event store (this will create tables)
        event_store = EventStore(connection_string)
        
        # Get stats to verify
        stats = event_store.get_stats()
        
        print("✓ Database initialized successfully")
        print(f"  Total events: {stats.get('total_events', 0)}")
        
        event_store.close()
        
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_database()