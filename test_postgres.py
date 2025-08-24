#!/usr/bin/env python3
import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_postgres_connection():
    """Test PostgreSQL connection"""
    try:
        # Connection parameters
        conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'alphatrader'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
        # Attempt connection
        print("Connecting to PostgreSQL...")
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        
        # Test query
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"✅ PostgreSQL connected successfully!")
        print(f"   Version: {version[0][:40]}...")
        
        # Create a test table to verify write permissions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS connection_test (
                id SERIAL PRIMARY KEY,
                test_time TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        print("✅ Database write permissions confirmed")
        
        # Clean up
        cur.execute("DROP TABLE IF EXISTS connection_test;")
        conn.commit()
        
        cur.close()
        conn.close()
        print("✅ PostgreSQL test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

if __name__ == "__main__":
    test_postgres_connection()