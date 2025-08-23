#!/usr/bin/env python3
"""
Setup PostgreSQL database and user for AlphaTrader
Tests REAL connection - no mocks
This is institutional-grade database setup
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
from dotenv import load_dotenv
import getpass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_database():
    """
    Create database and user in REAL PostgreSQL instance
    This is production-grade setup with proper error handling
    """
    
    # Load environment variables
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    else:
        print("⚠️  .env file not found. Using .env.template values.")
        print("   Please copy .env.template to .env and configure it.")
        load_dotenv(Path(__file__).parent.parent / '.env.template')
    
    # Get configuration from environment
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'alphatrader')
    db_user = os.getenv('DB_USER', 'alphatrader_user')
    db_password = os.getenv('DB_PASSWORD', 'your_secure_password_here')
    
    print("\n=== AlphaTrader Database Setup ===\n")
    print(f"Host: {db_host}:{db_port}")
    print(f"Database: {db_name}")
    print(f"User: {db_user}")
    print("\n" + "="*35 + "\n")
    
    # Get PostgreSQL superuser password
    postgres_password = getpass.getpass("Enter PostgreSQL 'postgres' superuser password: ")
    
    try:
        # Connect as superuser to create database
        print("\n🔄 Connecting to PostgreSQL as superuser...")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user='postgres',
            password=postgres_password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if user exists
        print(f"🔄 Checking if user '{db_user}' exists...")
        cur.execute(
            "SELECT 1 FROM pg_user WHERE usename = %s",
            (db_user,)
        )
        user_exists = cur.fetchone() is not None
        
        if user_exists:
            print(f"ℹ️  User '{db_user}' already exists")
            # Update password
            cur.execute(
                f"ALTER USER {db_user} WITH PASSWORD %s",
                (db_password,)
            )
            print(f"✅ Updated password for user: {db_user}")
        else:
            # Create user
            cur.execute(
                f"CREATE USER {db_user} WITH PASSWORD %s",
                (db_password,)
            )
            print(f"✅ Created user: {db_user}")
        
        # Check if database exists
        print(f"🔄 Checking if database '{db_name}' exists...")
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        db_exists = cur.fetchone() is not None
        
        if db_exists:
            print(f"ℹ️  Database '{db_name}' already exists")
            # Grant privileges anyway
            cur.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user}")
            print(f"✅ Granted privileges to {db_user}")
        else:
            # Create database
            cur.execute(f"CREATE DATABASE {db_name} OWNER {db_user}")
            print(f"✅ Created database: {db_name}")
            
            # Grant privileges
            cur.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user}")
            print(f"✅ Granted all privileges to {db_user}")
        
        # Close superuser connection
        cur.close()
        conn.close()
        
        # Test connection with new user
        print(f"\n🔄 Testing connection as '{db_user}'...")
        test_conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        
        # Create a test table to verify permissions
        test_cur = test_conn.cursor()
        test_cur.execute("""
            CREATE TABLE IF NOT EXISTS connection_test (
                id SERIAL PRIMARY KEY,
                test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        test_conn.commit()
        
        # Insert test record
        test_cur.execute("INSERT INTO connection_test DEFAULT VALUES RETURNING id")
        test_id = test_cur.fetchone()[0]
        test_conn.commit()
        
        # Clean up test
        test_cur.execute("DROP TABLE connection_test")
        test_conn.commit()
        
        test_cur.close()
        test_conn.close()
        
        print(f"✅ Successfully connected and verified permissions")
        
        print("\n" + "="*50)
        print("✨ DATABASE SETUP COMPLETE!")
        print("="*50)
        print("\nNext steps:")
        print("1. Copy .env.template to .env if not already done")
        print("2. Update the DB_PASSWORD in .env if you changed it")
        print("3. Run: pip install -r requirements.txt")
        print("4. Proceed with Day 2 implementation")
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is running")
        print("2. Check if 'postgres' password is correct")
        print("3. Verify host and port settings")
        return False
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nThis might mean:")
        print("- PostgreSQL is not installed")
        print("- PostgreSQL service is not running")
        print("- Network/firewall issues")
        return False

def check_prerequisites():
    """Check if PostgreSQL is available"""
    try:
        import psycopg2
        print("✅ psycopg2 is installed")
        return True
    except ImportError:
        print("❌ psycopg2 not installed")
        print("Run: pip install psycopg2-binary")
        return False

if __name__ == "__main__":
    print("\n🚀 AlphaTrader Database Setup Script")
    print("    Institutional-Grade PostgreSQL Configuration")
    print("    " + "="*45)
    
    if not check_prerequisites():
        sys.exit(1)
    
    success = setup_database()
    sys.exit(0 if success else 1)