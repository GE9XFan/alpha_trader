#!/usr/bin/env python3
"""
Options Processor - Alpha Vantage Options Chain Processing
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- options:{symbol}:calls: Call options data
- options:{symbol}:puts: Put options data
- options:{symbol}:{contractID}:greeks: Options greeks data
"""

import asyncio
import json
import redis.asyncio as aioredis
import aiohttp
from typing import Dict, Any
from datetime import datetime
import logging
from redis_keys import Keys


class OptionsProcessor:
    """
    Processes Alpha Vantage options chain data.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize options processor."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

    async def fetch_options_chain(self, session: aiohttp.ClientSession, symbol: str,
                                   rate_limiter, base_url: str, api_key: str, ttls: Dict):
        """Fetch options chain with semaphore rate limiting."""
        try:
            # Acquire rate limit token
            await rate_limiter.acquire()

            params = {
                'function': 'REALTIME_OPTIONS',
                'symbol': symbol,
                'apikey': api_key,
                'datatype': 'json',
                'require_greeks': 'true'
            }

            async with session.get(base_url, params=params) as response:
                    # CRITICAL: Raise for HTTP errors
                    response.raise_for_status()

                    data = await response.json()

                    # Check for API errors FIRST
                    if 'Error Message' in data:
                        self.logger.error(f"AV error for {symbol}: {data['Error Message']}")
                        return None

                    # Check for soft rate limits
                    if 'Note' in data:
                        self.logger.warning(f"AV soft limit: {data['Note']}")
                        await asyncio.sleep(30)
                        return None

                    if 'Information' in data:
                        self.logger.warning(f"AV info: {data['Information']}")
                        await asyncio.sleep(60)
                        return None

                    # Process options chain - Alpha Vantage uses 'data' not 'contracts'
                    if 'data' in data and isinstance(data['data'], list):
                        # Store to Redis
                        await self._store_options_data(symbol, data, ttls)
                        self.logger.info(f"✓ Fetched options for {symbol}: {len(data['data'])} contracts")
                    else:
                        self.logger.warning(f"AV options missing 'data' for {symbol}; keys={list(data.keys())[:6]}")
                        return None

                    return data

        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self.logger.warning(f"Rate limited (429) for {symbol}")
                await asyncio.sleep(60)
            else:
                self.logger.error(f"HTTP {e.status} for {symbol}: {e}")
            return None

    async def _store_options_data(self, symbol: str, data: Dict, ttls: Dict):
        """Store options chain data to Redis."""
        try:
            # Process and store options - Alpha Vantage uses 'data' array
            contracts = data.get('data', [])

            # Separate calls and puts - AV uses 'type' not 'option_type'
            calls = [c for c in contracts if c.get('type') == 'call']
            puts = [c for c in contracts if c.get('type') == 'put']

            # Store to Redis
            async with self.redis.pipeline(transaction=False) as pipe:
                if calls:
                    await pipe.setex(
                        Keys.options_calls(symbol),
                        ttls['options_chain'],
                        json.dumps(calls)
                    )

                if puts:
                    await pipe.setex(
                        Keys.options_puts(symbol),
                        ttls['options_chain'],
                        json.dumps(puts)
                    )

                # Store Greeks separately - AV has inline greeks, not nested
                for contract in contracts:
                    # Extract greeks from contract (they're inline in AV response)
                    greeks = {
                        'delta': contract.get('delta'),
                        'gamma': contract.get('gamma'),
                        'theta': contract.get('theta'),
                        'vega': contract.get('vega'),
                        'rho': contract.get('rho'),
                        'implied_volatility': contract.get('implied_volatility'),
                        'open_interest': contract.get('open_interest')
                    }

                    # AV uses 'contractID' not 'contract_id'
                    contract_id = contract.get('contractID')
                    if contract_id and any(greeks.values()):
                        # Store individual contract greeks (legacy format)
                        key = f"options:{symbol}:{contract_id}:greeks"  # TODO: Migrate to normalized greeks hash
                        await pipe.setex(
                            key,
                            ttls['greeks'],
                            json.dumps(greeks)
                        )

                await pipe.execute()

                # Log success with sample contract details
                log_msg = f"Stored options for {symbol}: calls={len(calls)}, puts={len(puts)}"

                # Sample contract details commented out for cleaner logs
                # try:
                #     # Add sample call contract with Greeks
                #     if calls:
                #         sample_call = calls[0]  # Get first call as sample
                #         log_msg += f"\n  Sample CALL: {sample_call.get('contractID', 'N/A')}"
                #         log_msg += f" Strike=${float(sample_call.get('strike', 0) or 0):.2f}"
                #         log_msg += f" Exp={sample_call.get('expiration', 'N/A')}"
                #         log_msg += f" Bid=${float(sample_call.get('bid', 0) or 0):.2f}"
                #         log_msg += f" Ask=${float(sample_call.get('ask', 0) or 0):.2f}"
                #         log_msg += f" Vol={int(float(sample_call.get('volume', 0) or 0))}"
                #         log_msg += f" OI={int(float(sample_call.get('open_interest', 0) or 0))}"
                #         log_msg += f"\n    Greeks: Δ={float(sample_call.get('delta', 0) or 0):.4f}"
                #         log_msg += f" Γ={float(sample_call.get('gamma', 0) or 0):.4f}"
                #         log_msg += f" Θ={float(sample_call.get('theta', 0) or 0):.4f}"
                #         log_msg += f" Vega={float(sample_call.get('vega', 0) or 0):.4f}"
                #         log_msg += f" IV={float(sample_call.get('implied_volatility', 0) or 0):.2%}"
                #
                #     # Add sample put contract with Greeks
                #     if puts:
                #         sample_put = puts[0]  # Get first put as sample
                #         log_msg += f"\n  Sample PUT: {sample_put.get('contractID', 'N/A')}"
                #         log_msg += f" Strike=${float(sample_put.get('strike', 0) or 0):.2f}"
                #         log_msg += f" Exp={sample_put.get('expiration', 'N/A')}"
                #         log_msg += f" Bid=${float(sample_put.get('bid', 0) or 0):.2f}"
                #         log_msg += f" Ask=${float(sample_put.get('ask', 0) or 0):.2f}"
                #         log_msg += f" Vol={int(float(sample_put.get('volume', 0) or 0))}"
                #         log_msg += f" OI={int(float(sample_put.get('open_interest', 0) or 0))}"
                #         log_msg += f"\n    Greeks: Δ={float(sample_put.get('delta', 0) or 0):.4f}"
                #         log_msg += f" Γ={float(sample_put.get('gamma', 0) or 0):.4f}"
                #         log_msg += f" Θ={float(sample_put.get('theta', 0) or 0):.4f}"
                #         log_msg += f" Vega={float(sample_put.get('vega', 0) or 0):.4f}"
                #         log_msg += f" IV={float(sample_put.get('implied_volatility', 0) or 0):.2%}"
                # except Exception as e:
                #     # If detailed logging fails, just log the basic message
                #     pass

                self.logger.info(log_msg)

        except Exception as e:
            self.logger.error(f"Error storing options for {symbol}: {e}")