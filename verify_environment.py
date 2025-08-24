#!/usr/bin/env python3
"""Complete environment verification for AlphaTrader"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_python_version():
    """Verify Python 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (need 3.11+)")
        return False

def check_env_variables():
    """Verify required environment variables"""
    required = {
        'AV_API_KEY': 'Alpha Vantage API key',
        'DB_PASSWORD': 'PostgreSQL password'
    }
    
    all_present = True
    for var, description in required.items():
        if os.getenv(var):
            print(f"✅ {description} configured")
        else:
            print(f"❌ {description} missing in .env")
            all_present = False
    
    return all_present

def main():
    print("="*50)
    print("ALPHATRADER ENVIRONMENT VERIFICATION")
    print("="*50)
    
    checks = []
    
    # Python version
    print("\n📍 Python Version:")
    checks.append(check_python_version())
    
    # Environment variables
    print("\n📍 Environment Variables:")
    checks.append(check_env_variables())
    
    # PostgreSQL
    print("\n📍 PostgreSQL:")
    from test_postgres import test_postgres_connection
    checks.append(test_postgres_connection())
    
    # Redis
    print("\n📍 Redis:")
    from test_redis import test_redis_connection
    checks.append(test_redis_connection())
    
    # Summary
    print("\n" + "="*50)
    if all(checks):
        print("✅ ENVIRONMENT READY FOR ALPHATRADER!")
        print("   You can proceed to Step 3: Implementation")
    else:
        print("❌ Some checks failed. Please fix issues above.")
    print("="*50)

if __name__ == "__main__":
    main()