#!/usr/bin/env python3
"""
Alpha Vantage API Client for Options and Technical Indicators
Premium tier with 600 calls/minute rate limiting
"""

import asyncio
import aiohttp
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import backoff
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .models import (
    OptionContract, OptionsChain, OptionType
)
from .cache import CacheManager


class RateLimiter:
    """Rate limiter for API calls (600 per minute)"""

    def __init__(self, calls_per_minute: int = 600):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute  # Minimum seconds between calls
        self.last_call = 0
        self.call_times = []

    async def acquire(self):
        """Acquire permission to make API call"""
        now = time.time()

        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]

        # Check if we've hit the limit
        if len(self.call_times) >= self.calls_per_minute:
            # Wait until the oldest call is more than 60 seconds old
            sleep_time = 60 - (now - self.call_times[0]) + 0.1
            logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            now = time.time()

        # Ensure minimum interval between calls
        if self.last_call > 0:
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

        self.last_call = time.time()
        self.call_times.append(self.last_call)

    def get_remaining_calls(self) -> int:
        """Get number of remaining calls in current minute"""
        now = time.time()
        self.call_times = [t for t in self.call_times if now - t < 60]
        return self.calls_per_minute - len(self.call_times)


class AlphaVantageClient:
    """Alpha Vantage API client with rate limiting"""

    def __init__(self, cache_manager: CacheManager, config: Dict):
        """Initialize AV client with configuration"""
        self.cache = cache_manager
        # Process environment variables in config
        self.config = self._substitute_env_vars(config['alpha_vantage'])
        self.api_key = self.config['api_key']
        self.base_url = self.config['base_url']

        # Rate limiting - ensure integer type
        rate_limit = self.config.get('rate_limit', 600)
        if isinstance(rate_limit, str) and not rate_limit.startswith('${'):
            try:
                rate_limit = int(rate_limit)
            except ValueError:
                rate_limit = 600
        elif isinstance(rate_limit, str):
            rate_limit = 600  # Use default if env var template not processed
        self.rate_limiter = RateLimiter(rate_limit)

        # Session for connection pooling
        self.session = None

        # Statistics
        self.stats = {
            'calls_made': 0,
            'calls_failed': 0,
            'cache_hits': 0
        }

        logger.info(f"Alpha Vantage client initialized (Rate limit: {rate_limit}/min)")

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Parse ${VAR:default} format
            var_expr = config[2:-1]
            if ':' in var_expr:
                var_name, default = var_expr.split(':', 1)
                value = os.getenv(var_name, default)
            else:
                value = os.getenv(var_expr, config)

            # Try to convert to appropriate type
            if value.isdigit():
                return int(value)
            elif value.replace('.', '', 1).isdigit():
                return float(value)
            elif value.lower() in ('true', 'yes'):
                return True
            elif value.lower() in ('false', 'no'):
                return False
            return value
        return config

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting and retries"""
        # Add API key
        params['apikey'] = self.api_key

        # Rate limiting
        await self.rate_limiter.acquire()

        # Create session if needed
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 10))

            async with self.session.get(
                self.base_url,
                params=params,
                timeout=timeout
            ) as response:
                self.stats['calls_made'] += 1

                if response.status == 200:
                    data = await response.json()

                    # Check for API errors
                    if 'Error Message' in data:
                        raise ValueError(f"API Error: {data['Error Message']}")
                    elif 'Note' in data:
                        raise ValueError(f"Rate limit: {data['Note']}")
                    elif 'Information' in data:
                        logger.warning(f"API Info: {data['Information']}")

                    return data
                else:
                    raise aiohttp.ClientError(f"HTTP {response.status}: {await response.text()}")

        except Exception as e:
            self.stats['calls_failed'] += 1
            logger.error(f"API request failed: {e}")
            raise

    # OPTIONS DATA METHODS
    async def get_realtime_options(self, symbol: str, require_greeks: bool = True, ibkr_client=None) -> Optional[OptionsChain]:
        """Get real-time options chain with Greeks PROVIDED by Alpha Vantage"""
        try:
            # Check cache first
            cached = self.cache.get_options_chain(symbol)
            if cached:
                self.stats['cache_hits'] += 1
                return OptionsChain(**cached)

            # Make API request
            params = {
                'function': 'REALTIME_OPTIONS',
                'symbol': symbol
            }

            if require_greeks:
                params['require_greeks'] = 'true'

            data = await self._make_request(params)

            if 'data' not in data:
                logger.warning(f"No options data for {symbol}")
                return None

            # Get spot price from IBKR if available
            spot_price = 0
            if ibkr_client and ibkr_client.is_connected():
                try:
                    # Try async get_spot_price method first (handles subscription if needed)
                    if hasattr(ibkr_client, 'get_spot_price'):
                        price = await ibkr_client.get_spot_price(symbol)
                        if price:
                            spot_price = price
                            logger.info(f"Got spot price from IBKR: ${spot_price:.2f}")
                    # Fallback to get_last_price
                    elif hasattr(ibkr_client, 'get_last_price'):
                        price = ibkr_client.get_last_price(symbol)
                        if price:
                            spot_price = price
                            logger.info(f"Got spot price from IBKR: ${spot_price:.2f}")
                    # Try ticker_subs directly
                    elif hasattr(ibkr_client, 'ticker_subs') and symbol in ibkr_client.ticker_subs:
                        ticker = ibkr_client.ticker_subs[symbol]
                        if ticker and hasattr(ticker, 'last') and ticker.last:
                            spot_price = ticker.last
                            logger.info(f"Got spot price from IBKR ticker: ${spot_price:.2f}")
                except Exception as e:
                    logger.warning(f"Could not get spot price from IBKR: {e}")

            if spot_price == 0:
                logger.warning(f"No spot price available for {symbol}, using 0")

            # Parse options contracts
            options = []
            for contract_data in data['data']:
                contract = OptionContract(
                    symbol=symbol,
                    contract_id=contract_data['contractID'],
                    strike=float(contract_data['strike']),
                    expiration=contract_data['expiration'],
                    type=OptionType.CALL if contract_data['type'].upper() == 'CALL' else OptionType.PUT,

                    # Market data
                    bid=float(contract_data.get('bid', 0)),
                    ask=float(contract_data.get('ask', 0)),
                    last=float(contract_data.get('last', 0)),
                    volume=int(contract_data.get('volume', 0)),
                    open_interest=int(contract_data.get('open_interest', 0)),

                    # Greeks - PROVIDED, not calculated!
                    implied_volatility=float(contract_data.get('implied_volatility', 0)),
                    delta=float(contract_data.get('delta', 0)),
                    gamma=float(contract_data.get('gamma', 0)),
                    theta=float(contract_data.get('theta', 0)),
                    vega=float(contract_data.get('vega', 0)),
                    rho=float(contract_data.get('rho', 0))
                )
                options.append(contract)

            # Create chain
            chain = OptionsChain(
                symbol=symbol,
                spot_price=spot_price,
                timestamp=int(time.time() * 1000),
                options=options
            )

            # Cache the chain
            self.cache.set_options_chain(symbol, chain.dict())

            logger.info(f"Retrieved {len(options)} options for {symbol} with Greeks")
            logger.debug(f"Spot price: ${spot_price:.2f}, First option: {options[0].contract_id if options else 'None'}")
            logger.debug(f"Cached to key 'options:{symbol}' with TTL {self.cache._get_ttl('options_chain')} seconds")
            return chain

        except Exception as e:
            logger.error(f"Failed to get options for {symbol}: {e}")
            return None

    async def get_historical_options(
        self,
        symbol: str,
        date: Optional[str] = None,
        ibkr_client=None
    ) -> Optional[OptionsChain]:
        """Get historical options data for a specific date"""
        try:
            params = {
                'function': 'HISTORICAL_OPTIONS',
                'symbol': symbol
            }

            if date:
                params['date'] = date

            data = await self._make_request(params)

            # Parse similar to realtime options
            if 'data' not in data:
                logger.warning(f"No historical options data for {symbol}")
                return None

            # Get spot price (historical doesn't have live price)
            spot_price = 0

            # Parse options contracts
            options = []
            for contract_data in data['data']:
                contract = OptionContract(
                    symbol=symbol,
                    contract_id=contract_data['contractID'],
                    strike=float(contract_data['strike']),
                    expiration=contract_data['expiration'],
                    type=OptionType.CALL if contract_data['type'].upper() == 'CALL' else OptionType.PUT,

                    # Market data
                    bid=float(contract_data.get('bid', 0)),
                    ask=float(contract_data.get('ask', 0)),
                    last=float(contract_data.get('last', 0)),
                    volume=int(contract_data.get('volume', 0)),
                    open_interest=int(contract_data.get('open_interest', 0)),

                    # Greeks
                    implied_volatility=float(contract_data.get('implied_volatility', 0)),
                    delta=float(contract_data.get('delta', 0)),
                    gamma=float(contract_data.get('gamma', 0)),
                    theta=float(contract_data.get('theta', 0)),
                    vega=float(contract_data.get('vega', 0)),
                    rho=float(contract_data.get('rho', 0))
                )
                options.append(contract)

            # Create chain
            chain = OptionsChain(
                symbol=symbol,
                spot_price=spot_price,
                timestamp=int(time.time() * 1000),
                options=options
            )

            logger.info(f"Retrieved {len(options)} historical options for {symbol}")
            return chain

        except Exception as e:
            logger.error(f"Failed to get historical options for {symbol}: {e}")
            return None

    # TECHNICAL INDICATORS METHODS
    async def get_rsi(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 14
    ) -> Optional[Dict]:
        """Get RSI indicator"""
        try:
            # Check cache
            cached = self.cache.get_indicator(symbol, f"RSI_{interval}_{time_period}")
            if cached:
                self.stats['cache_hits'] += 1
                return cached

            params = {
                'function': 'RSI',
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'series_type': 'close'
            }

            data = await self._make_request(params)

            # Cache the result
            self.cache.set_indicator(symbol, f"RSI_{interval}_{time_period}", data)

            return data

        except Exception as e:
            logger.error(f"Failed to get RSI for {symbol}: {e}")
            return None

    async def get_macd(
        self,
        symbol: str,
        interval: str = "daily",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Optional[Dict]:
        """Get MACD indicator"""
        try:
            cached = self.cache.get_indicator(symbol, f"MACD_{interval}")
            if cached:
                self.stats['cache_hits'] += 1
                return cached

            params = {
                'function': 'MACD',
                'symbol': symbol,
                'interval': interval,
                'series_type': 'close',
                'fastperiod': fast_period,
                'slowperiod': slow_period,
                'signalperiod': signal_period
            }

            data = await self._make_request(params)
            self.cache.set_indicator(symbol, f"MACD_{interval}", data)

            return data

        except Exception as e:
            logger.error(f"Failed to get MACD for {symbol}: {e}")
            return None

    async def get_bbands(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        nbdev_up: int = 2,
        nbdev_dn: int = 2
    ) -> Optional[Dict]:
        """Get Bollinger Bands"""
        try:
            cached = self.cache.get_indicator(symbol, f"BBANDS_{interval}_{time_period}")
            if cached:
                self.stats['cache_hits'] += 1
                return cached

            params = {
                'function': 'BBANDS',
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'series_type': 'close',
                'nbdevup': nbdev_up,
                'nbdevdn': nbdev_dn,
                'matype': 0  # Simple moving average
            }

            data = await self._make_request(params)
            self.cache.set_indicator(symbol, f"BBANDS_{interval}_{time_period}", data)

            return data

        except Exception as e:
            logger.error(f"Failed to get Bollinger Bands for {symbol}: {e}")
            return None

    async def get_atr(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 14
    ) -> Optional[Dict]:
        """Get Average True Range"""
        try:
            cached = self.cache.get_indicator(symbol, f"ATR_{interval}_{time_period}")
            if cached:
                self.stats['cache_hits'] += 1
                return cached

            params = {
                'function': 'ATR',
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period
            }

            data = await self._make_request(params)
            self.cache.set_indicator(symbol, f"ATR_{interval}_{time_period}", data)

            return data

        except Exception as e:
            logger.error(f"Failed to get ATR for {symbol}: {e}")
            return None

    async def get_vwap(self, symbol: str, interval: str = "15min") -> Optional[Dict]:
        """Get VWAP (intraday intervals only)"""
        try:
            # VWAP only works with intraday intervals
            valid_intervals = ['1min', '5min', '15min', '30min', '60min']
            if interval not in valid_intervals:
                logger.warning(f"VWAP requires intraday interval, got {interval}")
                return None

            cached = self.cache.get_indicator(symbol, f"VWAP_{interval}")
            if cached:
                self.stats['cache_hits'] += 1
                return cached

            params = {
                'function': 'VWAP',
                'symbol': symbol,
                'interval': interval
            }

            data = await self._make_request(params)
            self.cache.set_indicator(symbol, f"VWAP_{interval}", data)

            return data

        except Exception as e:
            logger.error(f"Failed to get VWAP for {symbol}: {e}")
            return None

    # SENTIMENT & ANALYTICS METHODS
    async def get_news_sentiment(
        self,
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        sort: str = "LATEST",
        limit: int = 50
    ) -> Optional[Dict]:
        """Get news sentiment analysis"""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'sort': sort,
                'limit': limit
            }

            # Note: uses 'tickers' not 'symbol'
            if tickers:
                params['tickers'] = tickers
            if topics:
                params['topics'] = topics

            data = await self._make_request(params)

            # Cache sentiment data
            if tickers:
                self.cache.set_sentiment(tickers, data)

            return data

        except Exception as e:
            logger.error(f"Failed to get news sentiment: {e}")
            return None

    async def get_top_gainers_losers(self) -> Optional[Dict]:
        """Get top gainers and losers"""
        try:
            params = {'function': 'TOP_GAINERS_LOSERS'}
            data = await self._make_request(params)
            return data

        except Exception as e:
            logger.error(f"Failed to get top gainers/losers: {e}")
            return None

    async def get_insider_transactions(self, symbol: str) -> Optional[Dict]:
        """Get insider transactions"""
        try:
            params = {
                'function': 'INSIDER_TRANSACTIONS',
                'symbol': symbol
            }

            data = await self._make_request(params)
            return data

        except Exception as e:
            logger.error(f"Failed to get insider transactions for {symbol}: {e}")
            return None

    # FUNDAMENTALS METHODS
    async def get_company_overview(self, symbol: str) -> Optional[Dict]:
        """Get company overview and fundamentals"""
        try:
            # Check cache (longer TTL for fundamentals)
            cached = self.cache.get(f"fundamentals:{symbol}")
            if cached:
                self.stats['cache_hits'] += 1
                return cached

            params = {
                'function': 'OVERVIEW',
                'symbol': symbol
            }

            data = await self._make_request(params)

            # Cache fundamentals
            self.cache.set_fundamentals(symbol, data)

            return data

        except Exception as e:
            logger.error(f"Failed to get overview for {symbol}: {e}")
            return None

    async def get_earnings(self, symbol: str) -> Optional[Dict]:
        """Get earnings data"""
        try:
            params = {
                'function': 'EARNINGS',
                'symbol': symbol
            }

            data = await self._make_request(params)
            return data

        except Exception as e:
            logger.error(f"Failed to get earnings for {symbol}: {e}")
            return None

    # ANALYTICS METHODS
    async def get_analytics_sliding_window(
        self,
        SYMBOLS: str,
        INTERVAL: str = "DAILY",
        RANGE: str = "1month",
        WINDOW_SIZE: int = 20,
        CALCULATIONS: str = "MEAN,STDDEV,CUMULATIVE_RETURN",
        OHLC: str = "close"
    ) -> Optional[Dict]:
        """Get analytics with sliding window (ALL parameters are UPPERCASE per API docs)"""
        try:
            params = {
                'function': 'ANALYTICS_SLIDING_WINDOW',
                'SYMBOLS': SYMBOLS,           # Use the actual parameter
                'INTERVAL': INTERVAL,         # Use the actual parameter
                'RANGE': RANGE,              # Use the actual parameter
                'WINDOW_SIZE': WINDOW_SIZE,  # Required parameter!
                'OHLC': OHLC,               # Use the actual parameter
                'CALCULATIONS': CALCULATIONS  # Use the actual parameter
            }

            data = await self._make_request(params)
            return data

        except Exception as e:
            logger.error(f"Failed to get analytics for {SYMBOLS}: {e}")
            return None
    
    # Backward compatibility alias
    async def get_analytics_fixed_window(self, *args, **kwargs):
        """Deprecated: Use get_analytics_sliding_window instead"""
        logger.warning("get_analytics_fixed_window is deprecated, use get_analytics_sliding_window")
        return await self.get_analytics_sliding_window(*args, **kwargs)

    # UTILITY METHODS
    def get_stats(self) -> Dict:
        """Get client statistics"""
        return {
            'calls_made': self.stats['calls_made'],
            'calls_failed': self.stats['calls_failed'],
            'cache_hits': self.stats['cache_hits'],
            'calls_remaining': self.rate_limiter.get_remaining_calls(),
            'cache_hit_rate': (
                f"{(self.stats['cache_hits'] / max(self.stats['calls_made'], 1)) * 100:.2f}%"
            )
        }

    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Alpha Vantage client closed")
