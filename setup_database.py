#!/usr/bin/env python3
"""
Database Setup Script - Fixed for Mac/Homebrew PostgreSQL
Uses system username instead of 'postgres'
Version: 1.1
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
from datetime import datetime
import getpass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseSetup:
    def __init__(self):
        """Initialize database setup with proper user detection"""
        # Get the current system username (for Homebrew PostgreSQL)
        system_user = getpass.getuser()
        
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'trading_system')
        
        # Use system username if DB_USER not set or is 'postgres'
        db_user_env = os.getenv('DB_USER', system_user)
        if db_user_env == 'postgres':
            print(f"ℹ️ 'postgres' user doesn't exist, using system user: {system_user}")
            self.db_user = system_user
        else:
            self.db_user = db_user_env
            
        self.db_password = os.getenv('DB_PASSWORD', '')
        
        print(f"📋 Database Configuration:")
        print(f"   Host: {self.db_host}:{self.db_port}")
        print(f"   Database: {self.db_name}")
        print(f"   User: {self.db_user}")
        print()
        
    def create_database(self):
        """Create the trading_system database if it doesn't exist"""
        print(f"📦 Creating database '{self.db_name}'...")
        
        try:
            # Connect to PostgreSQL server (to the default 'postgres' database)
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database='postgres'  # Connect to default database
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.db_name,)
            )
            exists = cursor.fetchone()
            
            if not exists:
                # Create database
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(self.db_name)
                    )
                )
                print(f"✓ Database '{self.db_name}' created successfully")
            else:
                print(f"ℹ️ Database '{self.db_name}' already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            print(f"\n💡 Try creating the database manually:")
            print(f"   createdb {self.db_name}")
            print(f"   OR")
            print(f"   psql -d postgres -c 'CREATE DATABASE {self.db_name};'")
            return False
    
    def create_system_tables(self):
        """Create system tables only (no data tables yet)"""
        print(f"📊 Creating system tables...")
        
        try:
            # Connect to the trading_system database
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # System configuration table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key VARCHAR(50) PRIMARY KEY,
                    value TEXT,
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_by VARCHAR(50) DEFAULT 'system'
                );
                
                -- Create index for faster lookups
                CREATE INDEX IF NOT EXISTS idx_system_config_key 
                ON system_config(key);
                
                -- Insert default configurations
                INSERT INTO system_config (key, value, description) VALUES
                    ('system_version', '0.1.0', 'System version number'),
                    ('environment', 'development', 'Current environment'),
                    ('maintenance_mode', 'false', 'System maintenance flag'),
                    ('trading_enabled', 'false', 'Global trading enable/disable'),
                    ('last_startup', NOW()::TEXT, 'Last system startup time')
                ON CONFLICT (key) DO NOTHING;
            """)
            print("✓ Created system_config table")
            
            # API call logging table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_call_log (
                    id SERIAL PRIMARY KEY,
                    api_name VARCHAR(50) NOT NULL,
                    endpoint VARCHAR(100),
                    method VARCHAR(10),
                    parameters JSONB,
                    response_status INTEGER,
                    response_time_ms INTEGER,
                    response_size_bytes INTEGER,
                    error_message TEXT,
                    rate_limit_remaining INTEGER,
                    called_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_api_call_log_api_name 
                ON api_call_log(api_name);
                CREATE INDEX IF NOT EXISTS idx_api_call_log_called_at 
                ON api_call_log(called_at DESC);
                CREATE INDEX IF NOT EXISTS idx_api_call_log_status 
                ON api_call_log(response_status);
                
                -- Partial index for errors only
                CREATE INDEX IF NOT EXISTS idx_api_call_log_errors 
                ON api_call_log(api_name, called_at) 
                WHERE error_message IS NOT NULL;
            """)
            print("✓ Created api_call_log table")
            
            # Schema versions for migration tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    api_name VARCHAR(50),
                    table_name VARCHAR(100),
                    migration_type VARCHAR(20), -- 'create', 'alter', 'index'
                    migration_sql TEXT,
                    response_sample JSONB,  -- Store sample API response
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rollback_sql TEXT,
                    notes TEXT
                );
                
                -- Track current schema version
                INSERT INTO schema_migrations (version, api_name, migration_type, notes) 
                VALUES (0, 'system', 'create', 'Initial system tables')
                ON CONFLICT (version) DO NOTHING;
            """)
            print("✓ Created schema_migrations table")
            
            # Emergency log for critical events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emergency_log (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,  -- 'circuit_breaker', 'api_failure', 'risk_violation'
                    severity VARCHAR(20) NOT NULL,     -- 'INFO', 'WARNING', 'CRITICAL', 'EMERGENCY'
                    description TEXT NOT NULL,
                    context JSONB,                     -- Additional context data
                    action_taken TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    resolved_by VARCHAR(50),
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Indexes for emergency response
                CREATE INDEX IF NOT EXISTS idx_emergency_log_severity 
                ON emergency_log(severity);
                CREATE INDEX IF NOT EXISTS idx_emergency_log_unresolved 
                ON emergency_log(logged_at DESC) 
                WHERE resolved = FALSE;
                CREATE INDEX IF NOT EXISTS idx_emergency_log_type 
                ON emergency_log(event_type, logged_at DESC);
            """)
            print("✓ Created emergency_log table")
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DECIMAL(20, 8),
                    metric_unit VARCHAR(20),
                    component VARCHAR(50),  -- 'data_ingestion', 'decision_engine', etc.
                    metadata JSONB,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for time-series queries
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_time 
                ON performance_metrics(metric_name, recorded_at DESC);
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_component 
                ON performance_metrics(component, recorded_at DESC);
            """)
            print("✓ Created performance_metrics table")
            
            # Session management table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(50) UNIQUE NOT NULL,
                    environment VARCHAR(20) NOT NULL,  -- 'development', 'paper', 'production'
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'stopped', 'crashed'
                    total_api_calls INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    total_errors INTEGER DEFAULT 0,
                    profit_loss DECIMAL(10, 2),
                    metadata JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_trading_sessions_status 
                ON trading_sessions(status, start_time DESC);
            """)
            print("✓ Created trading_sessions table")
            
            # Audit trail table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id SERIAL PRIMARY KEY,
                    action VARCHAR(100) NOT NULL,
                    entity_type VARCHAR(50),  -- 'trade', 'position', 'config', etc.
                    entity_id VARCHAR(100),
                    old_value JSONB,
                    new_value JSONB,
                    user_id VARCHAR(50),
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_audit_trail_entity 
                ON audit_trail(entity_type, entity_id);
                CREATE INDEX IF NOT EXISTS idx_audit_trail_time 
                ON audit_trail(created_at DESC);
            """)
            print("✓ Created audit_trail table")
            
            # Create a view for system health
            cursor.execute("""
                CREATE OR REPLACE VIEW system_health AS
                SELECT 
                    (SELECT value FROM system_config WHERE key = 'trading_enabled') as trading_enabled,
                    (SELECT COUNT(*) FROM emergency_log WHERE resolved = FALSE) as unresolved_emergencies,
                    (SELECT COUNT(*) FROM api_call_log 
                     WHERE called_at > NOW() - INTERVAL '1 minute') as api_calls_last_minute,
                    (SELECT AVG(response_time_ms) FROM api_call_log 
                     WHERE called_at > NOW() - INTERVAL '5 minutes') as avg_api_response_ms,
                    (SELECT COUNT(*) FROM api_call_log 
                     WHERE error_message IS NOT NULL 
                     AND called_at > NOW() - INTERVAL '1 hour') as api_errors_last_hour,
                    NOW() as checked_at;
            """)
            print("✓ Created system_health view")
            
            # Commit all changes
            conn.commit()
            cursor.close()
            conn.close()
            
            print("\n✓ All system tables created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating system tables: {e}")
            if conn:
                conn.rollback()
            return False
    
    def test_connection(self):
        """Test database connection and basic operations"""
        print(f"\n🔍 Testing database connection...")
        
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Test 1: Check version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✓ PostgreSQL version: {version.split(',')[0]}")
            
            # Test 2: Count tables
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE';
            """)
            table_count = cursor.fetchone()[0]
            print(f"✓ Number of tables created: {table_count}")
            
            # Test 3: Insert and retrieve from system_config
            cursor.execute("""
                INSERT INTO system_config (key, value, description) 
                VALUES ('test_key', 'test_value', 'Test configuration')
                ON CONFLICT (key) DO UPDATE SET value = 'test_value';
            """)
            
            cursor.execute("SELECT value FROM system_config WHERE key = 'test_key';")
            test_value = cursor.fetchone()[0]
            assert test_value == 'test_value', "Test value mismatch"
            print("✓ Write/Read operations working")
            
            # Test 4: Check system health view
            cursor.execute("SELECT * FROM system_health;")
            health = cursor.fetchone()
            print("✓ System health view accessible")
            
            # Clean up test data
            cursor.execute("DELETE FROM system_config WHERE key = 'test_key';")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("\n✅ All database tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            if conn:
                conn.rollback()
            return False
    
    def create_env_file(self):
        """Create or update .env file with database settings"""
        print(f"\n📝 Updating .env file...")
        
        env_path = Path('.env')
        
        # Read existing .env if it exists
        existing_vars = {}
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        existing_vars[key] = value
        
        # Update with database settings (using actual username)
        existing_vars.update({
            'DB_HOST': self.db_host,
            'DB_PORT': self.db_port,
            'DB_NAME': self.db_name,
            'DB_USER': self.db_user,  # This will be the system username
            'DB_PASSWORD': self.db_password or ''
        })
        
        # Write back to .env
        with open(env_path, 'w') as f:
            f.write("# Trading System Environment Variables\n")
            f.write(f"# Generated/Updated: {datetime.now().isoformat()}\n\n")
            
            f.write("# Database Configuration\n")
            f.write(f"DB_HOST={existing_vars.get('DB_HOST', 'localhost')}\n")
            f.write(f"DB_PORT={existing_vars.get('DB_PORT', '5432')}\n")
            f.write(f"DB_NAME={existing_vars.get('DB_NAME', 'trading_system')}\n")
            f.write(f"DB_USER={existing_vars.get('DB_USER', self.db_user)}\n")
            f.write(f"DB_PASSWORD={existing_vars.get('DB_PASSWORD', '')}\n\n")
            
            f.write("# Redis Configuration\n")
            f.write(f"REDIS_HOST={existing_vars.get('REDIS_HOST', 'localhost')}\n")
            f.write(f"REDIS_PORT={existing_vars.get('REDIS_PORT', '6379')}\n")
            f.write(f"REDIS_PASSWORD={existing_vars.get('REDIS_PASSWORD', '')}\n\n")
            
            f.write("# API Keys (FILL THESE IN!)\n")
            f.write(f"AV_API_KEY={existing_vars.get('AV_API_KEY', 'your_alpha_vantage_key_here')}\n")
            f.write(f"IBKR_USERNAME={existing_vars.get('IBKR_USERNAME', 'your_ibkr_username')}\n")
            f.write(f"IBKR_PASSWORD={existing_vars.get('IBKR_PASSWORD', 'your_ibkr_password')}\n")
            f.write(f"IBKR_ACCOUNT={existing_vars.get('IBKR_ACCOUNT', 'your_ibkr_account')}\n\n")
            
            f.write("# Discord Configuration\n")
            f.write(f"DISCORD_WEBHOOK_URL={existing_vars.get('DISCORD_WEBHOOK_URL', 'your_discord_webhook_url')}\n\n")
            
            f.write("# Environment\n")
            f.write(f"ENVIRONMENT={existing_vars.get('ENVIRONMENT', 'development')}\n")
        
        print(f"✓ .env file updated with database configuration (user: {self.db_user})")
        
        # Remind about .gitignore
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                if '.env' not in f.read():
                    print("⚠️ WARNING: .env not in .gitignore - adding it")
                    with open(gitignore_path, 'a') as f:
                        f.write("\n# Environment variables\n.env\n")
    
    def print_summary(self):
        """Print setup summary"""
        print("\n" + "=" * 60)
        print("📊 DATABASE SETUP COMPLETE")
        print("=" * 60)
        print(f"""
Configuration:
  Database: {self.db_name}
  Host: {self.db_host}:{self.db_port}
  User: {self.db_user}
  
System Tables Created:
  • system_config      - System configuration storage
  • api_call_log       - API call tracking
  • schema_migrations  - Schema version control
  • emergency_log      - Critical event logging
  • performance_metrics - Performance tracking
  • trading_sessions   - Session management
  • audit_trail        - Audit logging
  • system_health      - Health monitoring view

Next Steps:
  1. Review .env file and add your API keys
  2. Data tables will be created during API discovery (Phase 0.5)
  3. Each API response will generate its own schema
  4. No hardcoded schemas - everything driven by actual API responses
        """)


def main():
    """Main execution"""
    print("🚀 Starting Database Setup...")
    print("=" * 60)
    
    setup = DatabaseSetup()
    
    # Step 1: Create database
    if not setup.create_database():
        print("\n❌ Failed to create database.")
        print("You can create it manually with:")
        print(f"  createdb {setup.db_name}")
        return 1
    
    # Step 2: Create system tables
    if not setup.create_system_tables():
        print("❌ Failed to create system tables. Exiting.")
        return 1
    
    # Step 3: Test connection
    if not setup.test_connection():
        print("❌ Database tests failed. Please check configuration.")
        return 1
    
    # Step 4: Update .env file
    setup.create_env_file()
    
    # Step 5: Print summary
    setup.print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())