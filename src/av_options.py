#!/usr/bin/env python3
"""
Options Processor - Alpha Vantage Options Chain Processing
Part of Quantisity Capital System

This module operates independently and communicates only via Redis.
Redis keys used:
- options:{symbol}:calls: Call options data
- options:{symbol}:puts: Put options data
- options:{symbol}:{contractID}:greeks: Options greeks data
"""

import asyncio
import json
import math
import redis.asyncio as aioredis
import aiohttp
from typing import Any, Dict, Callable, List
from datetime import datetime, timezone
import logging
import redis_keys as rkeys


class OptionsProcessor:
    """
    Processes Alpha Vantage options chain data.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize options processor."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        self._chain_callbacks: List[Callable[[str, Dict[str, Any]], Any]] = []

    @staticmethod
    def _to_float(value):
        """Convert Alpha Vantage numeric fields to float."""
        if value in (None, "", "null"):
            return None
        try:
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return None
            return number
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value):
        """Convert Alpha Vantage numeric fields to int."""
        if value in (None, "", "null"):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _normalize_contract(self, contract: Dict[str, Any], symbol: str, timestamp_ms: int) -> Dict[str, Any]:
        """Normalize a single contract while preserving all key data."""
        contract_id = contract.get('contractID')
        normalized = {
            'contract_id': contract_id,
            'occ_symbol': contract_id,
            'symbol': contract.get('symbol', symbol),
            'expiration': contract.get('expiration') or contract.get('date'),
            'strike': self._to_float(contract.get('strike')),
            'type': contract.get('type'),
            'last': self._to_float(contract.get('last')),
            'last_price': self._to_float(contract.get('last')),
            'mark': self._to_float(contract.get('mark')),
            'bid': self._to_float(contract.get('bid')),
            'ask': self._to_float(contract.get('ask')),
            'bid_size': self._to_int(contract.get('bid_size')),
            'ask_size': self._to_int(contract.get('ask_size')),
            'volume': self._to_int(contract.get('volume')),
            'open_interest': self._to_int(contract.get('open_interest')),
            'implied_volatility': self._to_float(contract.get('implied_volatility')),
            'delta': self._to_float(contract.get('delta')),
            'gamma': self._to_float(contract.get('gamma')),
            'theta': self._to_float(contract.get('theta')),
            'vega': self._to_float(contract.get('vega')),
            'rho': self._to_float(contract.get('rho')),
            'quote_ts': timestamp_ms,
        }

        # Carry through any remaining fields without modification for future consumers.
        extras = {}
        for key, value in contract.items():
            if key not in normalized:
                extras[key] = value
        if extras:
            normalized['extras'] = extras

        return normalized

    def register_chain_callback(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        """Allow external modules to react to new option chains."""
        self._chain_callbacks.append(callback)

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

    async def _notify_chain(self, symbol: str, payload: Dict[str, Any]) -> None:
        for callback in self._chain_callbacks:
            try:
                result = callback(symbol, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                self.logger.debug("Options chain callback error for %s: %s", symbol, exc)

    async def _store_options_data(self, symbol: str, data: Dict, ttls: Dict):
        """Store options chain data to Redis."""
        try:
            # Process and store options - Alpha Vantage uses 'data' array
            contracts = data.get('data', [])

            # Separate calls and puts - AV uses 'type' not 'option_type'
            calls = [c for c in contracts if c.get('type') == 'call']
            puts = [c for c in contracts if c.get('type') == 'put']

            skipped_contracts = 0
            missing_greeks = 0

            # Store to Redis
            async with self.redis.pipeline(transaction=False) as pipe:
                if calls:
                    await pipe.setex(
                        rkeys.options_calls_key(symbol),
                        ttls['options_chain'],
                        json.dumps(calls)
                    )

                if puts:
                    await pipe.setex(
                        rkeys.options_puts_key(symbol),
                        ttls['options_chain'],
                        json.dumps(puts)
                    )

                # Build normalized chain payload alongside raw response
                timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                by_contract = {}
                expirations = {}

                for contract in contracts:
                    contract_id = contract.get('contractID')
                    if not contract_id:
                        skipped_contracts += 1
                        continue

                    normalized = self._normalize_contract(contract, symbol, timestamp_ms)
                    by_contract[contract_id] = normalized

                    expiration = normalized.get('expiration')
                    option_type = (normalized.get('type') or '').lower()
                    if expiration:
                        summary = expirations.setdefault(expiration, {'calls': 0, 'puts': 0, 'strikes': set()})
                        if normalized.get('strike') is not None:
                            summary['strikes'].add(normalized['strike'])
                        if option_type == 'call':
                            summary['calls'] += 1
                        elif option_type == 'put':
                            summary['puts'] += 1

                    # Store individual contract greeks for backward compatibility
                    greeks = {
                        'delta': normalized.get('delta'),
                        'gamma': normalized.get('gamma'),
                        'theta': normalized.get('theta'),
                        'vega': normalized.get('vega'),
                        'rho': normalized.get('rho'),
                        'implied_volatility': normalized.get('implied_volatility'),
                        'open_interest': normalized.get('open_interest')
                    }

                    greek_values = [greeks['delta'], greeks['gamma'], greeks['theta'], greeks['vega'], greeks['rho']]
                    if not any(value is not None for value in greek_values):
                        missing_greeks += 1

                    if any(value is not None for value in greeks.values()):
                        key = f"options:{symbol}:{contract_id}:greeks"
                        await pipe.setex(
                            key,
                            ttls['greeks'],
                            json.dumps(greeks)
                        )

                expiration_summary = []
                for expiration, summary in expirations.items():
                    expiration_summary.append({
                        'expiration': expiration,
                        'calls': summary['calls'],
                        'puts': summary['puts'],
                        'strikes': sorted(summary['strikes'])
                    })
                expiration_summary.sort(key=lambda item: item['expiration'])

                chain_payload = {
                    'symbol': symbol,
                    'source': 'alpha_vantage',
                    'as_of': timestamp_ms,
                    'contract_count': len(by_contract),
                    'schema_version': 1,
                    'expiration_summary': expiration_summary,
                    'raw': data,
                    'by_contract': by_contract,
                    'metrics': {
                        'skipped_contracts': skipped_contracts,
                        'missing_greeks': missing_greeks,
                    }
                }

                await pipe.setex(
                    rkeys.options_chain_key(symbol),
                    ttls['options_chain'],
                    json.dumps(chain_payload)
                )

                await pipe.execute()

                monitoring_payload = {
                    'symbol': symbol,
                    'as_of': timestamp_ms,
                    'contracts': len(by_contract),
                    'skipped_contracts': skipped_contracts,
                    'missing_greeks': missing_greeks,
                }
                await self.redis.setex(
                    f'monitoring:options:{symbol}',
                    ttls.get('monitoring', 60),
                    json.dumps(monitoring_payload)
                )

                await self._notify_chain(symbol, chain_payload)

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
