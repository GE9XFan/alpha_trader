#!/usr/bin/env python3
"""
Alpha Vantage Ingestion - Main AlphaVantage REST API Client
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- options:{symbol}:calls: Call options data
- options:{symbol}:puts: Put options data
- options:{symbol}:{contractID}:greeks: Options greeks
- sentiment:{symbol}: Sentiment data
- technicals:{symbol}:{indicator}: Technical indicators
- heartbeat:alpha_vantage: Heartbeat status
"""

import asyncio
import json
import time
import redis.asyncio as aioredis
import aiohttp
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import redis_keys as rkeys
import traceback


class RollingRateLimiter:
    """
    True rolling window rate limiter using token bucket algorithm.
    Ensures API calls don't exceed rate limits over any rolling time window.
    """

    def __init__(self, rate_per_minute: int, safety_buffer: int = 10):
        """
        Initialize rate limiter.

        Args:
            rate_per_minute: Maximum calls allowed per minute
            safety_buffer: Safety margin to stay below limit
        """
        self.capacity = max(1, rate_per_minute - safety_buffer)
        self.tokens = self.capacity
        self.updated = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)

        # Track request statistics
        self.total_requests = 0
        self.total_wait_time = 0
        self.max_wait_time = 0

    async def acquire(self, n: int = 1):
        """
        Acquire n tokens, waiting if necessary.

        Args:
            n: Number of tokens to acquire (default 1)
        """
        wait_start = asyncio.get_event_loop().time()

        async with self.lock:
            while True:
                now = asyncio.get_event_loop().time()

                # Refill tokens based on time elapsed
                elapsed = now - self.updated
                refill = elapsed * (self.capacity / 60.0)  # Tokens per second
                self.tokens = min(self.capacity, self.tokens + refill)
                self.updated = now

                # Check if we have enough tokens
                if self.tokens >= n:
                    self.tokens -= n
                    wait_time = now - wait_start
                    self.total_wait_time += wait_time
                    self.max_wait_time = max(self.max_wait_time, wait_time)
                    self.total_requests += n

                    if wait_time > 0.1:  # Log if we had to wait
                        self.logger.debug(
                            f"Rate limiter: waited {wait_time:.2f}s, "
                            f"tokens remaining: {self.tokens:.1f}/{self.capacity}"
                        )

                    return

                # Calculate how long to wait for tokens
                tokens_needed = n - self.tokens
                wait_seconds = (tokens_needed / self.capacity) * 60.0
                wait_seconds = min(wait_seconds, 1.0)  # Cap at 1 second per iteration

                await asyncio.sleep(wait_seconds)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'capacity': self.capacity,
            'current_tokens': self.tokens,
            'total_requests': self.total_requests,
            'avg_wait_time': self.total_wait_time / max(1, self.total_requests),
            'max_wait_time': self.max_wait_time
        }


class AlphaVantageIngestion:
    """
    Alpha Vantage REST API ingestion with semaphore-based rate limiting.

    Rate Limiting Approach:
    - Uses asyncio.Semaphore for concurrency control (not true rate limiting)
    - Sized to (calls_per_minute - safety_buffer) concurrent requests
    - Works well with small symbol sets and staggered update intervals
    - For larger scale: consider implementing a true rolling-window rate limiter

    Current implementation is sufficient because:
    1. Small symbol set (12 symbols)
    2. Staggered update intervals prevent bursts
    3. Retry logic with exponential backoff handles 429s
    4. Soft limit detection ("Note"/"Information") triggers backoff
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize with async Redis and unified symbol list."""
        self.config = config
        self.redis = redis_conn  # Async Redis
        self.logger = logging.getLogger(__name__)

        # Alpha Vantage configuration
        self.av_config = config.get('alpha_vantage', {})
        self.api_key = self.av_config.get('api_key')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")

        self.base_url = self.av_config.get('base_url', 'https://www.alphavantage.co/query')
        self.calls_per_minute = self.av_config.get('calls_per_minute', 600)
        self.safety_buffer = self.av_config.get('safety_buffer', 10)

        # Use rolling rate limiter for true rate control
        self.rate_limiter = RollingRateLimiter(
            rate_per_minute=self.calls_per_minute,
            safety_buffer=self.safety_buffer
        )

        # Retry configuration
        self.retry_attempts = self.av_config.get('retry_attempts', 3)
        self.retry_delay_base = self.av_config.get('retry_delay', 2)

        # Build complete symbol list
        level2_symbols = config.get('symbols', {}).get('level2', [])
        standard_symbols = config.get('symbols', {}).get('standard', [])
        self.symbols = level2_symbols + standard_symbols

        # TTLs from data_ingestion config
        self.ttls = config['modules']['data_ingestion']['store_ttls']

        # Processors are instantiated once and reused across fetch cycles
        from av_options import OptionsProcessor
        from av_sentiment import SentimentProcessor

        self.options_processor = OptionsProcessor(self.config, self.redis)
        self.sentiment_processor = SentimentProcessor(self.config, self.redis)

        # Optional downstream notification channel for new option chains
        self.options_channel = self.av_config.get('options_pubsub_channel', 'events:options:chain')
        if hasattr(self.options_processor, 'register_chain_callback'):
            self.options_processor.register_chain_callback(self._on_new_options_chain)

        # Update intervals
        self.options_interval = self.av_config.get('options_update_interval', 10)
        self.sentiment_interval = self.av_config.get('sentiment_update_interval', 300)
        self.technicals_interval = self.av_config.get('technicals_update_interval', 60)

        # Initialize staggered update times
        now = time.time()
        self.last_options_update = {}
        self.last_sentiment_update = {}
        self.last_technicals_update = {}

        for i, symbol in enumerate(self.symbols):
            if symbol in level2_symbols:
                offset = level2_symbols.index(symbol) * 0.5
            else:
                offset = 2.0 + ((i - len(level2_symbols)) * 1.0)

            self.last_options_update[symbol] = now - self.options_interval + offset
            self.last_sentiment_update[symbol] = now - self.sentiment_interval + (offset * 5)
            self.last_technicals_update[symbol] = now - self.technicals_interval + (offset * 2)

        self._last_update_map = {
            'options': self.last_options_update,
            'sentiment': self.last_sentiment_update,
            'technicals': self.last_technicals_update,
        }

        # Failure tracking for circuit breaking
        self.failure_threshold = self.av_config.get('failure_threshold', 3)
        self.failure_backoff_seconds = self.av_config.get('failure_backoff_seconds', 120)
        self.failure_backoff_max = self.av_config.get('failure_backoff_max', 900)
        self.symbol_failures: Dict[str, Dict[str, Dict[str, Any]]] = {
            symbol: {
                'options': {'failures': 0, 'suspended_until': 0.0, 'last_error': '', 'last_notice': 0.0},
                'sentiment': {'failures': 0, 'suspended_until': 0.0, 'last_error': '', 'last_notice': 0.0},
                'technicals': {'failures': 0, 'suspended_until': 0.0, 'last_error': '', 'last_notice': 0.0},
            }
            for symbol in self.symbols
        }

        self.running = False
        self.logger.info(f"AlphaVantageIngestion initialized for {len(self.symbols)} symbols")

    async def start(self):
        """Start Alpha Vantage data ingestion."""
        # Log startup with API key status (masked)
        api_key_status = "configured" if self.api_key and not self.api_key.startswith("${") else "NOT SET"
        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 8 else "INVALID"

        self.logger.info(f"Starting Alpha Vantage data ingestion...")
        self.logger.info(f"  API Key: {api_key_status} ({masked_key})")
        self.logger.info(f"  Symbols: {self.symbols}")
        self.logger.info(f"  Update intervals: options={self.options_interval}s, sentiment={self.sentiment_interval}s, technicals={self.technicals_interval}s")

        self.running = True

        try:
            # Start update loop
            update_task = asyncio.create_task(self._update_loop())

            # Start metrics loop
            metrics_task = asyncio.create_task(self._metrics_loop())

            # Wait for tasks
            await asyncio.gather(update_task, metrics_task)

        except Exception as e:
            self.logger.error(f"Error in Alpha Vantage ingestion: {e}")
            traceback.print_exc()

    async def _update_loop(self):
        """Main update loop for fetching data."""
        # Configure timeout and connection limits for robustness
        timeout = aiohttp.ClientTimeout(total=10, connect=3, sock_read=7)
        connector = aiohttp.TCPConnector(limit=200, ttl_dns_cache=300)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            while self.running:
                try:
                    current_time = time.time()
                    tasks = []

                    for symbol in self.symbols:
                        # Add jitter to prevent all symbols from aligning at minute boundaries
                        options_jitter = random.uniform(-0.25, 0.25) * self.options_interval
                        sentiment_jitter = random.uniform(-0.25, 0.25) * self.sentiment_interval
                        technicals_jitter = random.uniform(-0.25, 0.25) * self.technicals_interval

                        update_plan = [
                            ('options', self.options_interval + options_jitter, self.fetch_options_chain),
                            ('sentiment', self.sentiment_interval + sentiment_jitter, self.fetch_sentiment),
                            ('technicals', self.technicals_interval + technicals_jitter, self.fetch_technicals),
                        ]

                        for data_type, target_interval, fetcher in update_plan:
                            last_update = self._last_update_map[data_type].get(symbol, 0)
                            if current_time - last_update >= target_interval:
                                if self._is_suspended(symbol, data_type, current_time):
                                    self._mark_attempt(symbol, data_type, current_time)
                                    continue

                                tasks.append(self._run_fetch(data_type, symbol, session, fetcher))
                                self._mark_attempt(symbol, data_type, current_time)

                    if tasks:
                        self.logger.info(f"Fetching {len(tasks)} updates from Alpha Vantage...")
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Log errors
                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.debug(f"Fetch error result: {result}")

                    await asyncio.sleep(1)

                except Exception as e:
                    self.logger.error(f"Update loop error: {e}")
                    await asyncio.sleep(1)

    async def _fetch_with_retry(self, func, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Retry wrapper with exponential backoff and jitter for robust network handling."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)

            except aiohttp.ClientResponseError as e:
                last_error = e
                if e.status == 429:
                    wait_time = 60
                    self.logger.warning(f"{self._context_label(context)}Rate limited (429), waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                elif attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    # Add ±20% jitter
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.debug(f"{self._context_label(context)}HTTP {e.status}, retry in {actual_delay:.1f}s")
                    await asyncio.sleep(actual_delay)

            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.debug(f"{self._context_label(context)}Timeout, retry in {actual_delay:.1f}s")
                    await asyncio.sleep(actual_delay)

            except (aiohttp.ClientError, aiohttp.ClientConnectionError,
                    aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError) as e:
                # Handle specific network errors (DNS, connection reset, TLS issues)
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.warning(
                        f"{self._context_label(context)}Network error ({type(e).__name__}), "
                        f"retry {attempt+1}/{self.retry_attempts} in {actual_delay:.1f}s"
                    )
                    await asyncio.sleep(actual_delay)

            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay_base ** (attempt + 1)
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = max(0.1, delay + jitter)
                    self.logger.warning(
                        f"{self._context_label(context)}Unexpected error ({type(e).__name__}), "
                        f"retry in {actual_delay:.1f}s"
                    )
                    await asyncio.sleep(actual_delay)

        raise last_error or Exception(f"All {self.retry_attempts} retry attempts failed")

    def _context_label(self, context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return ''
        symbol = context.get('symbol')
        data_type = context.get('type')
        label_parts = []
        if data_type:
            label_parts.append(str(data_type))
        if symbol:
            label_parts.append(str(symbol))
        if not label_parts:
            return ''
        return f"[{'/'.join(label_parts)}] "

    def _mark_attempt(self, symbol: str, data_type: str, current_time: float) -> None:
        self._last_update_map[data_type][symbol] = current_time

    def _reset_failure(self, symbol: str, data_type: str) -> None:
        state = self.symbol_failures[symbol][data_type]
        if state['failures'] or state['suspended_until']:
            state.update({'failures': 0, 'suspended_until': 0.0, 'last_error': '', 'last_notice': 0.0})

    def _record_failure(self, symbol: str, data_type: str, error: Exception) -> None:
        state = self.symbol_failures[symbol][data_type]
        state['failures'] += 1
        state['last_error'] = str(error)
        now = time.time()
        if state['failures'] >= self.failure_threshold:
            exponent = max(0, state['failures'] - self.failure_threshold)
            backoff = min(self.failure_backoff_seconds * (2 ** exponent), self.failure_backoff_max)
            state['suspended_until'] = now + backoff
            state['last_notice'] = now
            self.logger.warning(
                "Circuit breaking %s fetch for %s after %s failures; pausing %.0fs (%s)",
                data_type,
                symbol,
                state['failures'],
                backoff,
                error,
            )
        else:
            self.logger.debug("Failure %s for %s/%s: %s", state['failures'], data_type, symbol, error)

    def _is_suspended(self, symbol: str, data_type: str, current_time: float) -> bool:
        state = self.symbol_failures[symbol][data_type]
        if current_time < state['suspended_until']:
            if current_time - state['last_notice'] > 30:
                remaining = state['suspended_until'] - current_time
                self.logger.debug(
                    "Skipping %s for %s; suspended %.0fs remaining", data_type, symbol, remaining
                )
                state['last_notice'] = current_time
            return True
        return False

    async def _run_fetch(self, data_type: str, symbol: str, session: aiohttp.ClientSession, fetcher) -> Any:
        context = {'type': data_type, 'symbol': symbol}
        try:
            result = await self._fetch_with_retry(fetcher, session, symbol, context=context)
        except Exception as exc:
            self._record_failure(symbol, data_type, exc)
            return exc
        else:
            self._reset_failure(symbol, data_type)
            return result

    async def fetch_options_chain(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch options chain with semaphore rate limiting."""
        return await self.options_processor.fetch_options_chain(
            session, symbol, self.rate_limiter, self.base_url, self.api_key, self.ttls
        )

    async def fetch_sentiment(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch sentiment data."""
        return await self.sentiment_processor.fetch_sentiment(
            session, symbol, self.rate_limiter, self.base_url, self.api_key, self.ttls
        )

    async def fetch_technicals(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch all technical indicators: RSI, MACD, BBANDS, ATR."""
        return await self.sentiment_processor.fetch_technicals(
            session, symbol, self.rate_limiter, self.base_url, self.api_key, self.ttls
        )

    async def _on_new_options_chain(self, symbol: str, payload: Dict[str, Any]) -> None:
        """Publish lightweight notifications when a fresh chain arrives."""
        try:
            message = {
                'symbol': symbol,
                'as_of': payload.get('as_of'),
                'contracts': payload.get('contract_count'),
                'source': payload.get('source', 'alpha_vantage'),
            }
            await self.redis.publish(self.options_channel, json.dumps(message))
        except Exception as exc:
            self.logger.debug("Failed to publish options chain for %s: %s", symbol, exc)

    async def _metrics_loop(self):
        """Publish metrics to Redis."""
        while self.running:
            try:
                limiter_stats = self.rate_limiter.get_stats()
                now = time.time()
                suspended = []
                total_failures = 0

                for symbol, states in self.symbol_failures.items():
                    for data_type, state in states.items():
                        total_failures += state['failures']
                        if now < state['suspended_until']:
                            suspended.append(f"{symbol}:{data_type}")

                metrics = {
                    'symbols': len(self.symbols),
                    'rate_capacity': limiter_stats.get('capacity', 0),
                    'rate_tokens': round(limiter_stats.get('current_tokens', 0.0), 2),
                    'rate_total_requests': limiter_stats.get('total_requests', 0),
                    'rate_avg_wait_ms': round(limiter_stats.get('avg_wait_time', 0.0) * 1000, 2),
                    'rate_max_wait_ms': round(limiter_stats.get('max_wait_time', 0.0) * 1000, 2),
                    'failures_total': total_failures,
                    'suspended_feeds': len(suspended),
                }

                if suspended:
                    metrics['suspended_list'] = ','.join(sorted(suspended))

                heartbeat = {
                    'ts': int(datetime.now().timestamp() * 1000),
                    'symbols': len(self.symbols)
                }

                ttl = self.ttls.get('heartbeat', 15)
                metrics_key = 'monitoring:alpha_vantage:metrics'

                async with self.redis.pipeline(transaction=False) as pipe:
                    await pipe.hset(metrics_key, mapping={k: str(v) for k, v in metrics.items()})
                    await pipe.expire(metrics_key, self.ttls.get('metrics', 60))
                    await pipe.setex(rkeys.heartbeat_key('alpha_vantage'), ttl, json.dumps(heartbeat))
                    await pipe.execute()

                await asyncio.sleep(10)

            except Exception as e:
                self.logger.error(f"Metrics error: {e}")
                await asyncio.sleep(10)

    async def stop(self):
        """Stop Alpha Vantage ingestion."""
        self.running = False
        self.logger.info("Stopping Alpha Vantage ingestion")


async def fetch_symbol_data(symbol: str, redis_conn: aioredis.Redis, av_session, av_config):
    """Fetch all data for a symbol (used for testing)."""
    # Implementation for testing
    pass