"""
Setup configuration for Trading System
"""

from setuptools import setup, find_packages

setup(
    name="trading-system",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "trading-system=main:main",
            "health-check=scripts.health_check:main",
            "test-api=scripts.test_api:main",
            "backup-db=scripts.backup_db:main",
        ],
    },
    author="Trading System Team",
    description="Production automated options trading system",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.11",
    ],
)
