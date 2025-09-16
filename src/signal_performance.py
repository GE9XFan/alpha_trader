"""
Signal Performance Tracking Module

Tracks and analyzes the performance of trading signals and strategies.
Provides metrics calculation, win/loss tracking, and performance reporting.

Redis Keys Used:
    Read/Write:
        - performance:signal:{signal_id} (signal outcomes)
        - performance:strategy:{strategy}:wins (win counter)
        - performance:strategy:{strategy}:losses (loss counter)

Author: AlphaTrader Pro
Version: 3.0.0
"""

import json
import logging
import math
from typing import Dict, Any, List
from datetime import datetime


class PerformanceTracker:
    """
    Track signal and strategy performance metrics.

    Monitors signal outcomes, calculates win rates, and generates
    performance reports for strategy optimization.
    """

    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize performance tracker.

        Args:
            config: System configuration
            redis_conn: Redis connection for metrics storage
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        self.invalid_key = 'performance:invalid_outcomes'

    async def track_signal_performance(self, signal_id: str, outcome: Dict[str, Any]):
        """
        Track individual signal performance.

        Args:
            signal_id: Unique signal identifier
            outcome: Signal outcome data including PnL, timestamps, etc
        """
        try:
            if not self._validate_outcome(outcome):
                await self.redis.lpush(self.invalid_key, json.dumps({'signal_id': signal_id, 'outcome': outcome}))
                await self.redis.ltrim(self.invalid_key, 0, 99)
                await self.redis.expire(self.invalid_key, 86400)
                return

            # Store performance data
            perf_key = f'performance:signal:{signal_id}'
            await self.redis.hset(perf_key, mapping=outcome)
            await self.redis.expire(perf_key, 86400 * 30)  # Keep for 30 days

            # Update strategy statistics
            strategy = str(outcome.get('strategy') or 'unknown')
            pnl = float(outcome.get('pnl', 0))
            stats_key = f'performance:strategy:{strategy}:stats'
            series_key = f'performance:strategy:{strategy}:pnl_series'
            daily_key = f'performance:strategy:{strategy}:daily_pnl'

            async with self.redis.pipeline() as pipe:
                pipe.hincrbyfloat(stats_key, 'pnl_sum', pnl)
                pipe.hincrbyfloat(stats_key, 'pnl_squared_sum', pnl ** 2)
                pipe.hincrby(stats_key, 'count', 1)
                if pnl > 0:
                    pipe.hincrbyfloat(stats_key, 'win_pnl_sum', pnl)
                    pipe.hincrby(stats_key, 'win_count', 1)
                elif pnl < 0:
                    pipe.hincrbyfloat(stats_key, 'loss_pnl_sum', abs(pnl))
                    pipe.hincrby(stats_key, 'loss_count', 1)
                pipe.lpush(series_key, pnl)
                pipe.ltrim(series_key, 0, 499)
                pipe.expire(series_key, 86400 * 35)
                day_bucket = datetime.utcnow().strftime('%Y%m%d')
                pipe.hincrbyfloat(daily_key, day_bucket, pnl)
                pipe.expire(daily_key, 86400 * 35)
                await pipe.execute()

            if outcome.get('pnl', 0) > 0:
                await self.redis.incr(f'performance:strategy:{strategy}:wins')
            else:
                await self.redis.incr(f'performance:strategy:{strategy}:losses')

            self.logger.info(f"Tracked performance for signal {signal_id}")

        except Exception as e:
            self.logger.error(f"Error tracking performance: {e}")

    async def calculate_strategy_metrics(self, strategy: str) -> Dict[str, Any]:
        """
        Calculate performance metrics for a strategy.

        Args:
            strategy: Strategy name (0dte, 1dte, 14dte, moc)

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Get win/loss counts
            stats_key = f'performance:strategy:{strategy}:stats'
            stats = await self.redis.hgetall(stats_key)
            wins = int(stats.get('win_count', 0))
            losses = int(stats.get('loss_count', 0))
            total = int(stats.get('count', 0))

            if total == 0:
                return {'win_rate': 0, 'total_trades': 0}

            win_rate = wins / total if total else 0
            win_pnl_sum = float(stats.get('win_pnl_sum', 0))
            loss_pnl_sum = float(stats.get('loss_pnl_sum', 0))
            pnl_sum = float(stats.get('pnl_sum', 0))
            pnl_squared_sum = float(stats.get('pnl_squared_sum', 0))

            avg_win = win_pnl_sum / wins if wins else 0
            avg_loss = -(loss_pnl_sum / losses) if losses else 0
            profit_factor = win_pnl_sum / loss_pnl_sum if loss_pnl_sum else float('inf') if win_pnl_sum > 0 else 0

            mean_pnl = pnl_sum / total
            variance = max((pnl_squared_sum / total) - mean_pnl ** 2, 0)
            std_dev = math.sqrt(variance)
            sharpe = mean_pnl / std_dev if std_dev else 0

            pnl_series = await self.redis.lrange(f'performance:strategy:{strategy}:pnl_series', 0, -1)
            drawdown = self._calculate_max_drawdown([float(x) for x in pnl_series]) if pnl_series else 0

            return {
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses,
                'total_trades': total,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe': sharpe,
                'max_drawdown': drawdown,
                'pnl_sum': pnl_sum,
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}

    async def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary containing performance metrics for all strategies
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'strategies': {}
        }

        strategies = ['0dte', '1dte', '14dte', 'moc']
        for strategy in strategies:
            metrics = await self.calculate_strategy_metrics(strategy)
            daily_key = f'performance:strategy:{strategy}:daily_pnl'
            daily_pnl = await self.redis.hgetall(daily_key)
            daily_breakdown = {day: float(value) for day, value in daily_pnl.items()}
            report['strategies'][strategy] = {
                'metrics': metrics,
                'daily_pnl': daily_breakdown,
            }

        return report

    def _validate_outcome(self, outcome: Dict[str, Any]) -> bool:
        required_fields = ['strategy', 'pnl']
        for field in required_fields:
            if field not in outcome:
                return False
        try:
            float(outcome.get('pnl'))
        except (TypeError, ValueError):
            return False
        return True

    def _calculate_max_drawdown(self, pnl_series: List[float]) -> float:
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in reversed(pnl_series):
            cumulative += pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown