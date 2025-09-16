#!/usr/bin/env python3
"""
Analytics Engine - Main Analytics Coordinator
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- market:{symbol}:ticker: Market data
- market:{symbol}:trades: Trade data
- market:{symbol}:book: Order book data
- metrics:{symbol}:*: All calculated metrics
- discovered:*: Discovered parameters
- heartbeat:analytics: Heartbeat status
"""

import numpy as np
import pandas as pd
import json
import redis.asyncio as aioredis
import time
import asyncio
import pytz
import math
from datetime import datetime, time as datetime_time
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback


class MetricsAggregator:
    """
    Aggregates metrics across symbols for portfolio-level analysis.
    """

    def __init__(self, config: Dict[str, Any], redis_conn):
        """Initialize metrics aggregator."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Load symbols from config
        syms = config.get('symbols', {})
        self.symbols = list({*syms.get('standard', []), *syms.get('level2', [])})

        # Sectors mapping (could be enhanced with more symbols)
        self.sectors = {
            'SPY': 'INDEX',
            'QQQ': 'TECH',
            'IWM': 'SMALLCAP',
            'AAPL': 'TECH',
            'MSFT': 'TECH',
            'GOOGL': 'TECH',
            'AMZN': 'CONSUMER',
            'TSLA': 'AUTO',
            'NVDA': 'TECH',
            'META': 'TECH',
            'JPM': 'FINANCE',
            'GS': 'FINANCE'
        }

    async def calculate_portfolio_metrics(self) -> dict:
        """
        Calculate portfolio-wide metrics from all symbols.
        """
        metrics = {
            'avg_vpin': 0,
            'max_vpin': 0,
            'min_vpin': 1,
            'toxic_count': 0,
            'total_gex': 0,
            'total_dex': 0,
            'most_toxic': None,
            'sector_flows': {},
            'timestamp': time.time()
        }

        vpins = []
        gex_total = 0
        dex_total = 0

        for symbol in self.symbols:
            # Get VPIN
            vpin_json = await self.redis.get(f'metrics:{symbol}:vpin')
            if vpin_json:
                vpin_data = json.loads(vpin_json)
                vpin = vpin_data.get('value', 0.5)
                vpins.append(vpin)

                if vpin > 0.7:
                    metrics['toxic_count'] += 1
                    if not metrics['most_toxic'] or vpin > metrics['max_vpin']:
                        metrics['most_toxic'] = symbol
                        metrics['max_vpin'] = vpin

                metrics['min_vpin'] = min(metrics['min_vpin'], vpin)

            # Get GEX
            gex_json = await self.redis.get(f'metrics:{symbol}:gex')
            if gex_json:
                gex_data = json.loads(gex_json)
                gex_total += gex_data.get('total_gex', 0)

            # Get DEX
            dex_json = await self.redis.get(f'metrics:{symbol}:dex')
            if dex_json:
                dex_data = json.loads(dex_json)
                dex_total += dex_data.get('total_dex', 0)

        # Calculate averages
        if vpins:
            metrics['avg_vpin'] = sum(vpins) / len(vpins)

        metrics['total_gex'] = gex_total
        metrics['total_dex'] = dex_total

        # Store portfolio metrics
        await self.redis.setex(
            'metrics:portfolio',
            60,
            json.dumps(metrics)
        )

        return metrics

    async def calculate_cross_asset_correlations(self) -> dict:
        """
        Calculate correlations between symbols.
        """
        # This would require historical price data
        # For now, return placeholder
        return {
            'status': 'placeholder',
            'message': 'Requires historical data collection'
        }

    async def calculate_sector_flows(self) -> dict:
        """
        Calculate sector-level flow metrics.
        """
        sector_metrics = {}

        for symbol in self.symbols:
            sector = self.sectors.get(symbol, 'OTHER')

            if sector not in sector_metrics:
                sector_metrics[sector] = {
                    'symbols': [],
                    'avg_vpin': [],
                    'total_volume': 0,
                    'net_flow': 0
                }

            sector_metrics[sector]['symbols'].append(symbol)

            # Get VPIN
            vpin_json = await self.redis.get(f'metrics:{symbol}:vpin')
            if vpin_json:
                vpin_data = json.loads(vpin_json)
                sector_metrics[sector]['avg_vpin'].append(vpin_data.get('value', 0.5))

            # Get volume (from ticker)
            ticker_json = await self.redis.get(f'market:{symbol}:ticker')
            if ticker_json:
                ticker = json.loads(ticker_json)
                volume = ticker.get('volume', 0)
                sector_metrics[sector]['total_volume'] += volume

        # Calculate sector averages
        for sector in sector_metrics:
            vpins = sector_metrics[sector]['avg_vpin']
            if vpins:
                sector_metrics[sector]['avg_vpin'] = sum(vpins) / len(vpins)
            else:
                sector_metrics[sector]['avg_vpin'] = 0.5

        return sector_metrics


class AnalyticsEngine:
    """
    Main analytics engine that coordinates all calculations and updates.
    Processes market data and produces trading signals.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """
        Initialize analytics engine with async Redis.
        Note: Expects async Redis connection for all operations.
        """
        self.config = config
        self.redis = redis_conn  # This is async Redis
        self.logger = logging.getLogger(__name__)

        # Load symbols from config - combine standard and level2
        syms = config.get('symbols', {})
        self.symbols = list({*syms.get('standard', []), *syms.get('level2', [])})

        # TTLs for storing metrics
        self.ttls = config.get('modules', {}).get('analytics', {}).get('store_ttls', {
            'metrics': 60,
            'alerts': 3600,
            'heartbeat': 15
        })

        # Update intervals for different calculations
        self.update_intervals = config.get('modules', {}).get('analytics', {}).get('update_intervals', {
            'vpin': 5,
            'gex_dex': 30,
            'flow_toxicity': 60,
            'order_book': 1,
            'sweep_detection': 2
        })

        # Track last update times
        self.last_updates = {}

        # Market hours
        self.market_tz = pytz.timezone('US/Eastern')

        # Initialize calculator instances (will be imported from separate modules)
        self.vpin_calculator = None
        self.gex_dex_calculator = None
        self.pattern_analyzer = None
        self.aggregator = MetricsAggregator(config, redis_conn)

        self.logger.info(f"AnalyticsEngine initialized for {len(self.symbols)} symbols")

    async def start(self):
        """Start the analytics engine."""
        self.logger.info("Starting analytics engine...")

        try:
            # Import and initialize calculators
            from vpin_calculator import VPINCalculator
            from gex_dex_calculator import GEXDEXCalculator
            from pattern_analyzer import PatternAnalyzer

            self.vpin_calculator = VPINCalculator(self.config, self.redis)
            self.gex_dex_calculator = GEXDEXCalculator(self.config, self.redis)
            self.pattern_analyzer = PatternAnalyzer(self.config, self.redis)

            # Start main calculation loop
            await self._calculation_loop()

        except Exception as e:
            self.logger.error(f"Error starting analytics engine: {e}")
            self.logger.error(traceback.format_exc())

    async def _calculation_loop(self):
        """Main loop that triggers calculations based on intervals."""
        while True:
            try:
                current_time = time.time()

                # Check each calculation type
                for calc_type, interval in self.update_intervals.items():
                    last_update = self.last_updates.get(calc_type, 0)

                    if current_time - last_update >= interval:
                        await self._run_calculation(calc_type)
                        self.last_updates[calc_type] = current_time

                # Update heartbeat
                await self._update_heartbeat()

                # Sleep briefly
                await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Error in calculation loop: {e}")
                await asyncio.sleep(1)

    async def _run_calculation(self, calc_type: str):
        """Run specific calculation type for all symbols."""
        tasks = []

        for symbol in self.symbols:
            if calc_type == 'vpin':
                if self.vpin_calculator:
                    tasks.append(self.vpin_calculator.calculate_vpin(symbol))

            elif calc_type == 'gex_dex':
                if self.gex_dex_calculator:
                    tasks.append(self.gex_dex_calculator.calculate_gex(symbol))
                    tasks.append(self.gex_dex_calculator.calculate_dex(symbol))

            elif calc_type == 'flow_toxicity':
                if self.pattern_analyzer:
                    tasks.append(self.pattern_analyzer.analyze_flow_toxicity(symbol))

            elif calc_type == 'order_book':
                if self.pattern_analyzer:
                    tasks.append(self.pattern_analyzer.calculate_order_book_imbalance(symbol))

            elif calc_type == 'sweep_detection':
                if self.pattern_analyzer:
                    tasks.append(self.pattern_analyzer.detect_sweeps(symbol))

        if tasks:
            # Run all calculations in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Calculation error: {result}")

        # After individual calculations, update portfolio metrics
        if calc_type in ['vpin', 'gex_dex']:
            await self.aggregator.calculate_portfolio_metrics()

    async def _update_heartbeat(self):
        """Update heartbeat in Redis."""
        heartbeat = {
            'ts': int(datetime.now().timestamp() * 1000),
            'symbols': len(self.symbols),
            'calculations': list(self.last_updates.keys())
        }

        await self.redis.setex(
            'heartbeat:analytics',
            self.ttls.get('heartbeat', 15),
            json.dumps(heartbeat)
        )

    def is_market_hours(self, extended: bool = False) -> bool:
        """Check if currently in market hours."""
        now = datetime.now(self.market_tz)

        # Check weekend
        if now.weekday() >= 5:
            return False

        # Check time
        current_time = now.time()
        if extended:
            return datetime_time(4, 0) <= current_time <= datetime_time(20, 0)
        else:
            return datetime_time(9, 30) <= current_time <= datetime_time(16, 0)

    async def stop(self):
        """Stop the analytics engine."""
        self.logger.info("Stopping analytics engine...")
        # Any cleanup needed