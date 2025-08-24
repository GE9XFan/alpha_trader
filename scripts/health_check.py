#!/usr/bin/env python3
"""
System health check script for AlphaTrader.

Validates all components are properly configured and accessible.
Returns 0 on success, non-zero on failure.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv()

from src.core.config import ConfigManager
from src.core.constants import SystemLimits, LatencyTargets
from src.core.exceptions import ConfigurationError


class HealthCheck:
    """System health check utility."""
    
    def __init__(self):
        """Initialize health check."""
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        self.start_time = time.time()
        self.config = None
    
    def run(self) -> int:
        """
        Run all health checks.
        
        Returns:
            0 if all checks pass, 1 if any fail
        """
        print("=" * 60)
        print("AlphaTrader System Health Check")
        print("=" * 60)
        print(f"Started at: {datetime.now().isoformat()}")
        print()
        
        # Run checks in order
        self._check_environment()
        self._check_configuration()
        self._check_directories()
        self._check_dependencies()
        self._check_limits()
        self._check_performance()
        
        # Print results
        self._print_results()
        
        # Return status
        return 0 if not self.checks_failed else 1
    
    def _check_environment(self):
        """Check environment setup."""
        print("Checking environment...")
        
        # Check Python version
        py_version = sys.version_info
        if py_version >= (3, 11):
            self._pass("Python version", f"{py_version.major}.{py_version.minor}.{py_version.micro}")
        else:
            self._fail("Python version", f"Need 3.11+, got {py_version.major}.{py_version.minor}")
        
        # Check environment variables
        required_env = [
            "IBKR_ACCOUNT",
            "ALPHA_VANTAGE_KEY",
        ]
        
        for var in required_env:
            if os.getenv(var):
                self._pass(f"Environment: {var}", "Set")
            else:
                self._fail(f"Environment: {var}", "Not set")
        
        # Check optional environment variables
        optional_env = [
            "DISCORD_TOKEN",
            "DISCORD_WEBHOOK_URL",
            "WHOP_API_KEY",
        ]
        
        for var in optional_env:
            if os.getenv(var):
                self._pass(f"Environment: {var}", "Set")
            else:
                self._warn(f"Environment: {var}", "Not set (optional)")
    
    def _check_configuration(self):
        """Check configuration loading."""
        print("\nChecking configuration...")
        
        try:
            config_manager = ConfigManager()
            self.config = config_manager.load_from_env()
            self._pass("Configuration loading", "Success")
            
            # Validate configuration
            issues = self.config.validate_for_production()
            if issues:
                for issue in issues:
                    self._warn("Configuration validation", issue)
            else:
                self._pass("Configuration validation", "No issues")
            
            # Check specific settings
            self._check_config_value("IBKR Account", self.config.ibkr.account, lambda x: len(x) > 0)
            self._check_config_value("Max Positions", self.config.risk_limits.max_positions, lambda x: 0 < x <= 100)
            self._check_config_value("Daily Loss Limit", self.config.risk_limits.daily_loss_limit, lambda x: x > 0)
            self._check_config_value("VPIN Threshold", self.config.risk_limits.vpin_threshold, lambda x: 0 < x <= 1)
            
            # Check Greeks limits
            self._check_config_value("Delta Range", 
                                    (self.config.greeks_limits.delta_min, self.config.greeks_limits.delta_max),
                                    lambda x: x[0] < 0 and x[1] > 0 and abs(x[0]) == abs(x[1]))
            
        except ConfigurationError as e:
            self._fail("Configuration loading", str(e))
            self.config = None
        except Exception as e:
            self._fail("Configuration loading", f"Unexpected error: {e}")
            self.config = None
    
    def _check_directories(self):
        """Check required directories."""
        print("\nChecking directories...")
        
        directories = [
            Path("src"),
            Path("src/core"),
            Path("tests"),
            Path("config"),
            Path("scripts"),
            Path("logs"),
        ]
        
        for directory in directories:
            if directory.exists() and directory.is_dir():
                self._pass(f"Directory: {directory}", "Exists")
            else:
                if directory.name == "logs":
                    # Create logs directory if missing
                    directory.mkdir(parents=True, exist_ok=True)
                    self._pass(f"Directory: {directory}", "Created")
                else:
                    self._fail(f"Directory: {directory}", "Missing")
    
    def _check_dependencies(self):
        """Check Python dependencies."""
        print("\nChecking dependencies...")
        
        critical_packages = [
            "loguru",
            "pydantic",
            "dotenv",
            "yaml",
            "pandas",
            "numpy",
        ]
        
        for package in critical_packages:
            try:
                __import__(package)
                self._pass(f"Package: {package}", "Installed")
            except ImportError:
                self._fail(f"Package: {package}", "Not installed")
    
    def _check_limits(self):
        """Check system limits configuration."""
        print("\nChecking system limits...")
        
        # Check if limits are reasonable
        checks = [
            ("Max Positions", SystemLimits.MAX_POSITIONS, lambda x: 0 < x <= 100),
            ("Max Position Size", SystemLimits.MAX_POSITION_SIZE, lambda x: x > 0),
            ("Daily Loss Limit", SystemLimits.DAILY_LOSS_LIMIT, lambda x: x > 0),
            ("VPIN Threshold", SystemLimits.VPIN_THRESHOLD, lambda x: 0 < x <= 1),
            ("Alpha Vantage Rate Limit", SystemLimits.ALPHA_VANTAGE_RATE_LIMIT, lambda x: 0 < x <= 500),
        ]
        
        for name, value, validator in checks:
            if validator(value):
                self._pass(f"Limit: {name}", f"{value}")
            else:
                self._fail(f"Limit: {name}", f"Invalid value: {value}")
    
    def _check_performance(self):
        """Check performance targets."""
        print("\nChecking performance targets...")
        
        # Check critical path latency
        components = {
            "IBKR Data": LatencyTargets.IBKR_DATA_RECEIPT,
            "Features": LatencyTargets.FEATURE_CALCULATION,
            "Model": LatencyTargets.MODEL_INFERENCE,
            "Risk": LatencyTargets.RISK_VALIDATION,
            "Execution": LatencyTargets.ORDER_EXECUTION,
        }
        
        total = sum(components.values())
        
        for name, latency in components.items():
            self._pass(f"Latency: {name}", f"{latency}ms")
        
        if total <= LatencyTargets.CRITICAL_PATH_TOTAL:
            self._pass("Critical Path Total", f"{total}ms (target: {LatencyTargets.CRITICAL_PATH_TOTAL}ms)")
        else:
            self._fail("Critical Path Total", f"{total}ms exceeds target {LatencyTargets.CRITICAL_PATH_TOTAL}ms")
    
    def _check_config_value(self, name: str, value: Any, validator):
        """Check a configuration value."""
        try:
            if validator(value):
                self._pass(f"Config: {name}", f"{value}")
            else:
                self._fail(f"Config: {name}", f"Invalid: {value}")
        except Exception as e:
            self._fail(f"Config: {name}", f"Validation error: {e}")
    
    def _pass(self, check: str, detail: str):
        """Record a passed check."""
        self.checks_passed.append((check, detail))
        print(f"  ✓ {check}: {detail}")
    
    def _fail(self, check: str, detail: str):
        """Record a failed check."""
        self.checks_failed.append((check, detail))
        print(f"  ✗ {check}: {detail}")
    
    def _warn(self, check: str, detail: str):
        """Record a warning."""
        self.warnings.append((check, detail))
        print(f"  ⚠ {check}: {detail}")
    
    def _print_results(self):
        """Print health check results."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("Health Check Results")
        print("=" * 60)
        
        print(f"\nPassed: {len(self.checks_passed)}")
        print(f"Failed: {len(self.checks_failed)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Time: {elapsed:.2f} seconds")
        
        if self.checks_failed:
            print("\n❌ HEALTH CHECK FAILED")
            print("\nFailed checks:")
            for check, detail in self.checks_failed:
                print(f"  - {check}: {detail}")
        else:
            print("\n✅ HEALTH CHECK PASSED")
        
        if self.warnings:
            print("\nWarnings:")
            for check, detail in self.warnings:
                print(f"  - {check}: {detail}")
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save health check results to file."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "passed": len(self.checks_passed),
            "failed": len(self.checks_failed),
            "warnings": len(self.warnings),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "config": self.config.to_dict() if self.config else None,
        }
        
        # Save to logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        results_file = log_dir / f"health_check_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")


def main():
    """Run health check."""
    try:
        checker = HealthCheck()
        return checker.run()
    except KeyboardInterrupt:
        print("\n\nHealth check interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Health check failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())