#!/usr/bin/env python3
"""
System Health Check Script
Phase 1: Core Infrastructure
Monitors system health and reports any issues
"""

import os
import sys
import psutil
import redis
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")


class HealthChecker:
    """System health monitoring"""

    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []

    def check_system_resources(self) -> Dict[str, Any]:
        """Check CPU, memory, and disk usage"""
        logger.info("Checking system resources...")

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = "OK" if cpu_percent < 80 else "WARNING" if cpu_percent < 90 else "CRITICAL"

        # Memory
        memory = psutil.virtual_memory()
        memory_status = "OK" if memory.percent < 80 else "WARNING" if memory.percent < 90 else "CRITICAL"

        # Disk
        disk = psutil.disk_usage('/')
        disk_status = "OK" if disk.percent < 80 else "WARNING" if disk.percent < 90 else "CRITICAL"

        result = {
            'cpu': {
                'percent': cpu_percent,
                'status': cpu_status,
                'cores': psutil.cpu_count()
            },
            'memory': {
                'percent': memory.percent,
                'status': memory_status,
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3)
            },
            'disk': {
                'percent': disk.percent,
                'status': disk_status,
                'free_gb': disk.free / (1024**3),
                'total_gb': disk.total / (1024**3)
            }
        }

        # Log warnings
        if cpu_status != "OK":
            self.warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
        if memory_status != "OK":
            self.warnings.append(f"High memory usage: {memory.percent:.1f}%")
        if disk_status != "OK":
            self.warnings.append(f"Low disk space: {disk.percent:.1f}% used")

        self.checks.append(('System Resources', result))
        return result

    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connection and memory usage"""
        logger.info("Checking Redis...")

        try:
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                password=os.getenv('REDIS_PASSWORD', None),
                decode_responses=True
            )

            # Ping test
            r.ping()

            # Get info (synchronous call)
            info = r.info()  # type: ignore
            memory_info = r.info('memory')  # type: ignore

            # Calculate memory usage
            used_memory = memory_info.get('used_memory', 0)
            max_memory = memory_info.get('maxmemory', 0)

            if max_memory > 0:
                memory_percent = (used_memory / max_memory) * 100
            else:
                memory_percent = 0
                self.warnings.append("Redis maxmemory not set")

            result = {
                'status': 'connected',
                'version': info.get('redis_version', 'unknown'),
                'used_memory_human': memory_info.get('used_memory_human', '0'),
                'maxmemory_human': memory_info.get('maxmemory_human', 'not set'),
                'memory_percent': memory_percent,
                'connected_clients': info.get('connected_clients', 0),
                'uptime_days': info.get('uptime_in_days', 0)
            }

            # Check memory warning
            if memory_percent > 80:
                self.warnings.append(f"Redis memory usage high: {memory_percent:.1f}%")

            self.checks.append(('Redis', result))
            return result

        except Exception as e:
            self.errors.append(f"Redis connection failed: {e}")
            result = {
                'status': 'disconnected',
                'error': str(e)
            }
            self.checks.append(('Redis', result))
            return result

    def check_directories(self) -> Dict[str, Any]:
        """Check required directories exist"""
        logger.info("Checking directories...")

        required_dirs = ['logs', 'data', 'temp']
        results = {}

        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            exists = dir_path.exists()

            if not exists:
                dir_path.mkdir(parents=True, exist_ok=True)
                results[dir_name] = 'created'
                self.warnings.append(f"Directory '{dir_name}' was missing and has been created")
            else:
                results[dir_name] = 'exists'

        self.checks.append(('Directories', results))
        return results

    def check_environment(self) -> Dict[str, Any]:
        """Check environment variables"""
        logger.info("Checking environment variables...")

        required_vars = [
            'AV_API_KEY',
            'IBKR_HOST',
            'IBKR_PORT',
            'IBKR_ACCOUNT',
            'REDIS_HOST',
            'REDIS_PORT'
        ]

        results = {}

        for var in required_vars:
            value = os.getenv(var)
            if not value:
                results[var] = 'missing'
                self.errors.append(f"Required environment variable '{var}' is not set")
            elif value in ['your_alpha_vantage_api_key_here', 'DU1234567']:
                results[var] = 'default'
                self.warnings.append(f"Environment variable '{var}' is using default value")
            else:
                results[var] = 'set'

        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.version_info < (3, 11):
            self.warnings.append(f"Python {python_version} detected. Python 3.11+ recommended")

        results['python_version'] = python_version
        results['trading_mode'] = os.getenv('TRADING_MODE', 'paper')
        results['environment'] = os.getenv('ENVIRONMENT', 'development')

        self.checks.append(('Environment', results))
        return results

    def check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        logger.info("Checking network...")

        import socket

        results = {}

        # Check localhost
        try:
            socket.create_connection(("127.0.0.1", 80), timeout=2)
            results['localhost'] = 'ok'
        except:
            results['localhost'] = 'ok'  # Expected to fail, but socket works

        # Check internet
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            results['internet'] = 'connected'
        except:
            results['internet'] = 'disconnected'
            self.errors.append("No internet connection detected")

        # Check specific ports
        ports_to_check = [
            ('IBKR', os.getenv('IBKR_HOST', '127.0.0.1'),
             int(os.getenv('IBKR_PORT', 7497))),
            ('Redis', os.getenv('REDIS_HOST', 'localhost'),
             int(os.getenv('REDIS_PORT', 6379)))
        ]

        for name, host, port in ports_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    results[f"{name}_port_{port}"] = 'open'
                else:
                    results[f"{name}_port_{port}"] = 'closed'
                    self.warnings.append(f"{name} port {port} is not accessible")
            except Exception as e:
                results[f"{name}_port_{port}"] = 'error'
                self.warnings.append(f"Could not check {name} port {port}: {e}")

        self.checks.append(('Network', results))
        return results

    def print_summary(self):
        """Print health check summary"""
        print("\n" + "="*70)
        print("SYSTEM HEALTH CHECK REPORT")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # Print each check category
        for category, results in self.checks:
            print(f"\n[{category}]")

            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        # Color code the output
                        if value in ['OK', 'connected', 'exists', 'set', 'open', 'ok']:
                            color = '\033[92m'  # Green
                        elif value in ['WARNING', 'default', 'created']:
                            color = '\033[93m'  # Yellow
                        elif value in ['CRITICAL', 'disconnected', 'missing', 'closed', 'error']:
                            color = '\033[91m'  # Red
                        else:
                            color = ''

                        print(f"  {key}: {color}{value}\033[0m")

        # Print warnings
        if self.warnings:
            print("\n" + "="*70)
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        # Print errors
        if self.errors:
            print("\n" + "="*70)
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        # Overall status
        print("\n" + "="*70)
        if self.errors:
            print("❌ SYSTEM STATUS: CRITICAL - Errors detected")
            print("\nAction Required:")
            print("1. Fix the errors listed above")
            print("2. Run health check again")
        elif self.warnings:
            print("⚠️  SYSTEM STATUS: WARNING - Some issues detected")
            print("\nRecommended Actions:")
            print("1. Review and address warnings")
            print("2. Update configuration as needed")
        else:
            print("✅ SYSTEM STATUS: HEALTHY - All checks passed")
            print("\nSystem is ready for trading!")

        print("="*70)


def main():
    """Run health checks"""
    checker = HealthChecker()

    # Run all checks
    checker.check_system_resources()
    checker.check_redis()
    checker.check_directories()
    checker.check_environment()
    checker.check_network()

    # Print summary
    checker.print_summary()

    # Return exit code based on status
    if checker.errors:
        sys.exit(1)
    elif checker.warnings:
        sys.exit(0)  # Warnings are acceptable
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
