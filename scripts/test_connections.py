#!/usr/bin/env python3
"""
Test Connections Script
Phase 1: Day 1-2 - Verify all components are properly configured
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import redis
import requests
import aiohttp
from ib_insync import IB, Stock
from loguru import logger
import nest_asyncio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")


class ConnectionTester:
    """Test all system connections"""

    def __init__(self):
        self.results = {}

    async def test_redis(self) -> bool:
        """Test Redis connection"""
        logger.info("Testing Redis connection...")
        try:
            # Get Redis configuration
            host = os.getenv('REDIS_HOST', 'localhost')
            port = int(os.getenv('REDIS_PORT', 6379))
            db = int(os.getenv('REDIS_DB', 0))
            password = os.getenv('REDIS_PASSWORD', None)

            # Create connection
            r = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )

            # Test operations
            test_key = 'test:connection'
            test_value = f'test_{datetime.now().timestamp()}'

            # Set value with TTL
            r.setex(test_key, 10, test_value)

            # Get value
            retrieved = r.get(test_key)

            if retrieved == test_value:
                # Check memory configuration
                info = r.info('memory')  # type: ignore
                max_memory = info.get('maxmemory_human', 'Not set')  # type: ignore
                used_memory = info.get('used_memory_human', 'Unknown')  # type: ignore

                logger.success(f"✓ Redis connected successfully")
                logger.info(f"  Memory: {used_memory} used, {max_memory} max")

                # Clean up
                r.delete(test_key)

                self.results['redis'] = {
                    'status': 'connected',
                    'host': host,
                    'port': port,
                    'memory_used': used_memory,
                    'memory_max': max_memory
                }
                return True
            else:
                raise ValueError("Redis read/write test failed")

        except Exception as e:
            logger.error(f"✗ Redis connection failed: {e}")
            self.results['redis'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def test_alpha_vantage(self) -> bool:
        """Test Alpha Vantage API connection"""
        logger.info("Testing Alpha Vantage API...")
        try:
            api_key = os.getenv('AV_API_KEY')
            if not api_key or api_key == 'your_alpha_vantage_api_key_here':
                logger.warning("⚠ Alpha Vantage API key not configured")
                logger.info("  Please update AV_API_KEY in .env file")
                self.results['alpha_vantage'] = {
                    'status': 'not_configured',
                    'error': 'API key not set'
                }
                return False

            base_url = os.getenv('AV_BASE_URL', 'https://www.alphavantage.co/query')

            # Test with a simple API call (OVERVIEW for AAPL)
            params = {
                'function': 'OVERVIEW',
                'symbol': 'AAPL',
                'apikey': api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params, timeout=10) as response:
                    data = await response.json()

                    if 'Error Message' in data:
                        raise ValueError(f"API Error: {data['Error Message']}")
                    elif 'Note' in data:
                        raise ValueError(f"API Rate limit: {data['Note']}")
                    elif 'Information' in data:
                        raise ValueError(f"API Info: {data['Information']}")
                    elif 'Symbol' in data and data['Symbol'] == 'AAPL':
                        logger.success("✓ Alpha Vantage API connected successfully")
                        logger.info(f"  Rate limit: {os.getenv('AV_RATE_LIMIT', '600')} calls/min")

                        self.results['alpha_vantage'] = {
                            'status': 'connected',
                            'rate_limit': os.getenv('AV_RATE_LIMIT', '600'),
                            'test_symbol': 'AAPL',
                            'company_name': data.get('Name', 'Unknown')
                        }
                        return True
                    else:
                        raise ValueError("Unexpected API response format")

        except Exception as e:
            logger.error(f"✗ Alpha Vantage connection failed: {e}")
            self.results['alpha_vantage'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    def test_ibkr(self) -> bool:
        """Test Interactive Brokers connection"""
        logger.info("Testing IBKR connection...")
        try:
            host = os.getenv('IBKR_HOST', '127.0.0.1')
            port = int(os.getenv('IBKR_PORT', 7497))
            client_id = int(os.getenv('IBKR_CLIENT_ID', 1))
            account = os.getenv('IBKR_ACCOUNT', '')

            # Check if TWS/Gateway is configured
            if account == 'DU1234567' or not account:
                logger.warning("⚠ IBKR account not configured")
                logger.info("  Please update IBKR_ACCOUNT in .env file")
                logger.info("  Make sure TWS/IB Gateway is running on port 7497")
                self.results['ibkr'] = {
                    'status': 'not_configured',
                    'error': 'Account not set or using default'
                }
                return False

            # Use nest_asyncio to handle event loop issues
            import nest_asyncio
            nest_asyncio.apply()

            # Create IB connection
            ib = IB()

            # Try to connect
            ib.connect(host, port, clientId=client_id, timeout=10)

            if ib.isConnected():
                # Get account info
                account_summary = ib.accountSummary()

                # Extract buying power
                buying_power = 0
                for item in account_summary:
                    if item.tag == 'BuyingPower':
                        buying_power = float(item.value)
                        break

                logger.success("✓ IBKR connected successfully")
                logger.info(f"  Account: {account}")
                logger.info(f"  Mode: {'Paper' if port == 7497 else 'Live'}")
                logger.info(f"  Buying Power: ${buying_power:,.2f}")

                self.results['ibkr'] = {
                    'status': 'connected',
                    'account': account,
                    'mode': 'paper' if port == 7497 else 'live',
                    'buying_power': buying_power,
                    'host': host,
                    'port': port
                }

                # Disconnect
                ib.disconnect()
                return True
            else:
                raise ConnectionError("Could not connect to TWS/IB Gateway")

        except Exception as e:
            logger.error(f"✗ IBKR connection failed: {e}")
            logger.info("  Make sure TWS or IB Gateway is running")
            logger.info("  Check API settings: Enable ActiveX and Socket Clients")
            logger.info(f"  Socket port should be: {os.getenv('IBKR_PORT', 7497)}")

            self.results['ibkr'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
        
    def print_summary(self):
        """Print connection test summary"""
        print("\n" + "="*60)
        print("CONNECTION TEST SUMMARY")
        print("="*60)

        all_passed = True

        for service, result in self.results.items():
            status = result['status']
            if status == 'connected':
                symbol = "✓"
                color = "\033[92m"  # Green
            elif status == 'not_configured':
                symbol = "⚠"
                color = "\033[93m"  # Yellow
                all_passed = False
            else:
                symbol = "✗"
                color = "\033[91m"  # Red
                all_passed = False

            print(f"{color}{symbol}\033[0m {service.upper():20} {status}")

            if status == 'failed':
                print(f"  Error: {result.get('error', 'Unknown error')}")
            elif status == 'connected' and service == 'redis':
                print(f"  Memory: {result['memory_used']} / {result['memory_max']}")
            elif status == 'connected' and service == 'ibkr':
                print(f"  Account: {result['account']} ({result['mode']} mode)")

        print("\n" + "="*60)

        if all_passed:
            print("\033[92m✓ All connections successful!\033[0m")
            print("\nYou're ready to proceed with Phase 1 development.")
        else:
            print("\033[93m⚠ Some connections need configuration.\033[0m")
            print("\nPlease:")
            print("1. Update the .env file with your API keys and account details")
            print("2. Start Redis server: redis-server")
            print("3. Start IB Gateway/TWS and enable API connections")
            print("4. Run this script again to verify connections")

        print("="*60)


async def main():
    """Main test function"""
    tester = ConnectionTester()

    # Run tests
    await tester.test_redis()
    await tester.test_alpha_vantage()
    tester.test_ibkr()

    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("OPTIONS TRADING SYSTEM - CONNECTION TESTER")
    print("Phase 1: Core Infrastructure")
    print("="*60 + "\n")

    # Handle the event loop properly
    import nest_asyncio
    nest_asyncio.apply()

    asyncio.run(main())
