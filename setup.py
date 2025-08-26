"""
AlphaTrader Setup
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="alphatrader",
    version="3.0.0",
    author="AlphaTrader Team",
    description="ML-driven options trading with Alpha Vantage and IBKR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[
        'ib_insync>=0.9.86',
        'aiohttp>=3.9.0',
        'xgboost>=2.0.0',
        'pandas>=2.1.0',
        'numpy>=1.25.0',
    ],
)
