#!/usr/bin/env python3
"""
GEX/DEX Calculator - Gamma and Delta Exposure Calculations
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- market:{symbol}:ticker: Spot price data
- options:{symbol}:*:greeks: Options Greeks data
- analytics:{symbol}:gex: Gamma exposure output
- analytics:{symbol}:dex: Delta exposure output
"""

import json
import math
import re
import redis.asyncio as aioredis
import time
from typing import Dict, Any, Tuple, Optional, Iterable
import logging
import traceback

import redis_keys as rkeys

# OCC option symbol regex pattern: UNDERLYING(1-6) + YYMMDD(6) + C/P + STRIKE(8)
_OCC_RE = re.compile(r'^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$')


class GEXDEXCalculator:
    """
    Calculates Gamma Exposure (GEX) and Delta Exposure (DEX) from options chain.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize GEX/DEX calculator."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # TTLs for storing metrics
        self.ttls = config.get('modules', {}).get('analytics', {}).get('store_ttls', {
            'metrics': 60,
            'alerts': 3600,
            'heartbeat': 15
        })

    async def _update_metric_statistics(
        self,
        symbol: str,
        metric: str,
        value: float,
        window: int = 120,
        min_samples: int = 5,
    ) -> Dict[str, float]:
        """Record metric history and compute rolling z-score statistics."""

        history_key = f'analytics:{symbol}:{metric}:history'
        stats = {
            'zscore': 0.0,
            'mean': float(value),
            'stdev': 0.0,
            'samples': 0,
            'window': window,
        }

        try:
            await self.redis.lpush(history_key, value)
            await self.redis.ltrim(history_key, 0, window - 1)
            # Retain history for an hour by default to avoid churn
            await self.redis.expire(history_key, max(window * 5, 3600))

            raw_history = await self.redis.lrange(history_key, 0, window - 1)
            values = []
            for raw in raw_history:
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode()
                    except Exception:
                        continue
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue

            if not values:
                return stats

            stats['samples'] = len(values)
            mean = sum(values) / len(values)
            stats['mean'] = mean

            if len(values) < min_samples:
                return stats

            variance = sum((v - mean) ** 2 for v in values) / len(values)
            stdev = math.sqrt(max(variance, 1e-12))
            stats['stdev'] = stdev

            if stdev > 0:
                stats['zscore'] = (float(value) - mean) / stdev

            return stats

        except Exception as exc:  # pragma: no cover - defensive telemetry
            self.logger.debug(
                f"Failed to update {metric} statistics for {symbol}: {exc}"
            )
            return stats

    def _extract_occ(self, key: str) -> Tuple[Optional[str], float]:
        """
        Extract option type and strike from Redis key using OCC format.
        Key format: options:{SYMBOL}:{OCC}:greeks
        Returns (cp, strike_float) or (None, 0.0) if not match.
        """
        try:
            parts = key.split(':', 3)  # Split into at most 4 parts
            if len(parts) >= 3:
                occ = parts[2]  # Third token is OCC
                m = _OCC_RE.match(occ)
                if not m:
                    return None, 0.0
                cp = m.group('cp')  # 'C' or 'P'
                strike = float(m.group('strike')) / 1000.0  # OCC 8 digits, 3 implied decimals
                return cp, strike
            return None, 0.0
        except Exception:
            return None, 0.0

    def _extract_option_type(self, key: str) -> str:
        """
        Return 'call' or 'put' from the OCC symbol.
        """
        cp, _ = self._extract_occ(key)
        if not cp:
            return ''
        return 'call' if cp == 'C' else 'put'

    def _extract_strike_from_key(self, key: str) -> float:
        """
        Extract strike price from Redis options key.
        """
        _, strike = self._extract_occ(key)
        return strike

    async def calculate_gex(self, symbol: str) -> dict:
        """
        Calculate Gamma Exposure (GEX) from options chain.

        GEX measures the hedging flow required by market makers.
        Positive GEX: Market makers buy dips, sell rallies (stabilizing)
        Negative GEX: Market makers sell dips, buy rallies (destabilizing)

        Returns:
            Dictionary with GEX by strike and key levels
        """
        try:
            # 1. Get current spot price
            ticker_json = await self.redis.get(rkeys.market_ticker_key(symbol))
            if not ticker_json:
                self.logger.warning(f"No ticker data for {symbol}")
                return {'error': 'No spot price'}

            ticker = json.loads(ticker_json)
            spot = ticker.get('last')
            if spot is None or spot <= 0:
                return {'error': 'Invalid spot price'}

            # 2. Load option contracts either from normalized chain or legacy greeks keys
            contracts_iter: Iterable[Dict[str, Any]] = []
            using_normalized_chain = False

            try:
                chain_json = await self.redis.get(rkeys.options_chain_key(symbol))
            except Exception:
                chain_json = None

            if chain_json:
                try:
                    chain_payload = json.loads(chain_json)
                    by_contract = chain_payload.get('by_contract') or {}
                    if isinstance(by_contract, dict) and by_contract:
                        contracts_iter = by_contract.values()
                        using_normalized_chain = True
                except Exception as exc:
                    self.logger.error(f"Failed to parse normalized chain for {symbol}: {exc}")
                    contracts_iter = []

            greek_keys = []
            if not contracts_iter:
                pattern = f'options:{symbol}:*:greeks'

                try:
                    greek_keys = await self.redis.keys(pattern)

                    if not greek_keys:
                        for alt_pattern in [f'options:{symbol}:greeks:*', f'greeks:{symbol}:*']:
                            greek_keys = await self.redis.keys(alt_pattern)
                            if greek_keys:
                                break

                    self.logger.debug(f"Found {len(greek_keys)} option contracts for {symbol}")

                except Exception as e:
                    self.logger.error(f"Failed to get option keys for {symbol}: {e}")
                    return {'error': f'Failed to get options for {symbol}'}

                if not greek_keys:
                    self.logger.warning(f"No options data for {symbol}")
                    return {'error': f'No options data for {symbol}'}

            # 3. Calculate GEX for each strike
            gex_by_strike = {}
            oi_by_strike = {}  # Track OI per strike for filtering
            total_gex = 0
            contract_multiplier = 100  # Standard equity option multiplier

            # Track statistics for debugging
            calls_processed = 0
            puts_processed = 0
            contracts_skipped = 0

            if using_normalized_chain:
                for contract in contracts_iter:
                    try:
                        gamma = contract.get('gamma') or 0
                        open_interest = contract.get('open_interest') or 0
                        strike = contract.get('strike') or 0
                        option_type = (contract.get('type') or '').lower()

                        if not isinstance(gamma, (int, float)) or not isinstance(open_interest, (int, float)):
                            contracts_skipped += 1
                            continue

                        if gamma <= 0 or open_interest <= 0:
                            contracts_skipped += 1
                            continue

                        if not isinstance(strike, (int, float)) or strike <= 0:
                            contracts_skipped += 1
                            continue

                        if option_type not in ('call', 'put'):
                            contracts_skipped += 1
                            continue

                        if gamma < 0 or gamma > 1:
                            contracts_skipped += 1
                            continue

                        if option_type == 'call':
                            contract_gex = gamma * open_interest * contract_multiplier * spot * spot
                            calls_processed += 1
                        else:
                            contract_gex = -gamma * open_interest * contract_multiplier * spot * spot
                            puts_processed += 1

                        gex_by_strike.setdefault(strike, 0)
                        gex_by_strike[strike] += contract_gex
                        total_gex += contract_gex

                        oi_by_strike.setdefault(strike, 0)
                        oi_by_strike[strike] += open_interest

                    except Exception as exc:
                        contracts_skipped += 1
                        self.logger.debug(f"Skipping normalized contract: {exc}")
            else:
                for key in greek_keys:
                    try:
                        greeks_json = await self.redis.get(key)
                        if not greeks_json:
                            continue

                        greeks = json.loads(greeks_json)
                        try:
                            gamma = float(greeks.get('gamma', 0))
                        except (ValueError, TypeError):
                            gamma = 0

                        if gamma < 0 or gamma > 1:
                            contracts_skipped += 1
                            continue

                        try:
                            open_interest = float(greeks.get('open_interest', 0))
                        except (ValueError, TypeError):
                            open_interest = 0

                        if open_interest <= 0 or gamma == 0:
                            contracts_skipped += 1
                            continue

                        strike = self._extract_strike_from_key(key)
                        option_type = self._extract_option_type(key)

                        if strike <= 0 or not option_type:
                            contracts_skipped += 1
                            continue

                        if option_type == 'call':
                            contract_gex = gamma * open_interest * contract_multiplier * spot * spot
                            calls_processed += 1
                        else:  # put
                            contract_gex = -gamma * open_interest * contract_multiplier * spot * spot
                            puts_processed += 1

                        if strike not in gex_by_strike:
                            gex_by_strike[strike] = 0
                        gex_by_strike[strike] += contract_gex
                        total_gex += contract_gex

                        if strike not in oi_by_strike:
                            oi_by_strike[strike] = 0
                        oi_by_strike[strike] += int(open_interest)

                    except Exception as e:
                        contracts_skipped += 1
                        self.logger.debug(f"Error processing option {key}: {e}")
                        continue

            # Debug logging for telemetry
            if symbol == 'SPY' or self.logger.level <= logging.DEBUG:
                self.logger.debug(f"GEX stats for {symbol}: {calls_processed} calls, {puts_processed} puts, {contracts_skipped} skipped")
                # Sample strike parsing to catch regressions
                if gex_by_strike and self.logger.isEnabledFor(logging.DEBUG):
                    sample = list(gex_by_strike.items())[:3]
                    self.logger.debug(f"[PARSE] {symbol} sample strikes: {sample}")

            if not gex_by_strike:
                return {'error': 'No valid GEX data'}

            # 4. Find key levels with OI floor to filter out ghost strikes
            MIN_OI = 5  # Minimum open interest threshold
            valid_strikes = [(s, v) for s, v in gex_by_strike.items()
                           if oi_by_strike.get(s, 0) >= MIN_OI]

            if not valid_strikes:
                self.logger.warning(f"No strikes with OI >= {MIN_OI} for {symbol}")
                return {'error': 'No valid GEX data after OI filter'}

            sorted_strikes = sorted([s for s, _ in valid_strikes])

            # Find max GEX strike (biggest hedging level) from valid strikes only
            max_gex_strike, _ = max(valid_strikes, key=lambda kv: abs(kv[1]))

            # Find zero gamma level (flip point)
            # This is where cumulative GEX crosses zero
            cumulative_gex = 0
            zero_gamma_strike = None
            for strike in sorted_strikes:
                cumulative_gex += gex_by_strike[strike]
                if zero_gamma_strike is None and cumulative_gex >= 0:
                    zero_gamma_strike = strike

            # Identify support and resistance levels
            # Support: Strikes with high positive GEX below spot
            # Resistance: Strikes with high positive GEX above spot
            supports = []
            resistances = []

            for strike in sorted_strikes:
                gex = gex_by_strike[strike]
                if gex > total_gex * 0.1:  # Significant level (>10% of total)
                    if strike < spot:
                        supports.append(strike)
                    else:
                        resistances.append(strike)

            # 5. Prepare result
            result = {
                'spot': round(spot, 2),
                'total_gex': round(total_gex, 0),
                'gex_by_strike': {str(k): round(v, 0) for k, v in gex_by_strike.items()},
                'max_gex_strike': max_gex_strike,
                'zero_gamma_strike': zero_gamma_strike,
                'supports': sorted(supports, reverse=True)[:3],  # Top 3 supports
                'resistances': sorted(resistances)[:3],  # Top 3 resistances
                'regime': 'stabilizing' if total_gex > 0 else 'destabilizing',
                'units': 'dollar',  # Explicit units for downstream consumers
                'timestamp': time.time()
            }

            stats = await self._update_metric_statistics(symbol, 'gex', float(total_gex))
            result['zscore'] = stats['zscore']
            result['zscore_mean'] = stats['mean']
            result['zscore_stddev'] = stats['stdev']
            result['zscore_samples'] = stats['samples']
            result['zscore_window'] = stats['window']

            # 6. Store in Redis with configured TTL
            ttl = self.ttls.get('analytics', self.ttls.get('metrics', 60))
            await self.redis.setex(
                rkeys.analytics_gex_key(symbol),
                ttl,
                json.dumps(result)
            )

            # Log significant levels
            self.logger.info(f"GEX for {symbol}: Total={total_gex:.0f}, Max Strike={max_gex_strike}, Regime={result['regime']}")

            return result

        except Exception as e:
            self.logger.error(f"Error calculating GEX for {symbol}: {e}")
            self.logger.error(f"Full traceback for {symbol} GEX error:\n{traceback.format_exc()}")
            return {'error': str(e)}

    async def calculate_dex(self, symbol: str) -> dict:
        """
        Calculate Delta Exposure (DEX) from options chain.

        DEX measures directional exposure from options positioning.
        Positive DEX: Net long exposure (bullish)
        Negative DEX: Net short exposure (bearish)

        Returns:
            Dictionary with DEX by strike and directional bias
        """
        try:
            # 1. Get current spot price
            ticker_json = await self.redis.get(rkeys.market_ticker_key(symbol))
            if not ticker_json:
                self.logger.warning(f"No ticker data for {symbol}")
                return {'error': 'No spot price'}

            ticker = json.loads(ticker_json)
            spot = ticker.get('last')
            if spot is None or spot <= 0:
                return {'error': 'Invalid spot price'}

            # 2. Get all option Greeks keys
            greek_keys = []
            pattern = f'options:{symbol}:*:greeks'

            try:
                greek_keys = await self.redis.keys(pattern)

                if not greek_keys:
                    for alt_pattern in [f'options:{symbol}:greeks:*', f'greeks:{symbol}:*']:
                        greek_keys = await self.redis.keys(alt_pattern)
                        if greek_keys:
                            break

                self.logger.debug(f"Found {len(greek_keys)} option contracts for {symbol} DEX")

            except Exception as e:
                self.logger.error(f"Failed to get option keys for {symbol}: {e}")
                return {'error': f'Failed to get options for {symbol}'}

            if not greek_keys:
                self.logger.warning(f"No options data for {symbol}")
                return {'error': f'No options data for {symbol}'}

            # 3. Calculate DEX for each strike
            dex_by_strike = {}
            oi_by_strike = {}
            total_call_dex = 0
            total_put_dex = 0
            contract_multiplier = 100

            # Track statistics
            calls_processed = 0
            puts_processed = 0
            contracts_skipped = 0

            for key in greek_keys:
                try:
                    # Get Greeks data
                    greeks_json = await self.redis.get(key)
                    if not greeks_json:
                        continue

                    greeks = json.loads(greeks_json)
                    # Convert string values to floats (Alpha Vantage returns strings)
                    try:
                        delta = float(greeks.get('delta', 0))
                    except (ValueError, TypeError):
                        delta = 0

                    # Guardrail: Skip invalid delta values (AV quirks)
                    if abs(delta) > 1.2:  # Delta should be between -1 and 1
                        contracts_skipped += 1
                        continue

                    try:
                        open_interest = float(greeks.get('open_interest', 0))
                    except (ValueError, TypeError):
                        open_interest = 0

                    if open_interest <= 0 or delta == 0:
                        contracts_skipped += 1
                        continue

                    # Extract strike and option type
                    strike = self._extract_strike_from_key(key)
                    option_type = self._extract_option_type(key)

                    if strike <= 0 or not option_type:
                        contracts_skipped += 1
                        continue

                    # Calculate DEX for this contract
                    # DEX = Delta * Open Interest * Contract Multiplier * Spot
                    contract_dex = delta * open_interest * contract_multiplier * spot

                    # Aggregate by strike
                    if strike not in dex_by_strike:
                        dex_by_strike[strike] = {'call': 0, 'put': 0, 'net': 0}

                    if option_type == 'call':
                        dex_by_strike[strike]['call'] += contract_dex
                        total_call_dex += contract_dex
                        calls_processed += 1
                    else:
                        dex_by_strike[strike]['put'] += contract_dex
                        total_put_dex += contract_dex
                        puts_processed += 1

                    dex_by_strike[strike]['net'] += contract_dex

                    # Track OI per strike for filtering
                    if strike not in oi_by_strike:
                        oi_by_strike[strike] = 0
                    oi_by_strike[strike] += int(open_interest)

                except Exception as e:
                    self.logger.debug(f"Error processing option {key}: {e}")
                    continue

            # Debug logging for telemetry
            if symbol == 'SPY' or self.logger.level <= logging.DEBUG:
                self.logger.debug(f"DEX stats for {symbol}: {calls_processed} calls, {puts_processed} puts, {contracts_skipped} skipped")

            if not dex_by_strike:
                return {'error': 'No valid DEX data'}

            # 4. Calculate total net DEX
            total_dex = total_call_dex + total_put_dex

            # 5. Determine directional bias
            if abs(total_dex) < abs(total_call_dex) * 0.1:
                bias = 'neutral'
            elif total_dex > 0:
                bias = 'bullish'
            else:
                bias = 'bearish'

            # 6. Find strikes with highest exposure (filter by OI floor)
            MIN_OI = 5  # Minimum open interest threshold
            valid_strikes = [(s, v) for s, v in dex_by_strike.items()
                           if oi_by_strike.get(s, 0) >= MIN_OI]

            if not valid_strikes:
                self.logger.warning(f"No strikes with OI >= {MIN_OI} for {symbol} DEX")
                return {'error': 'No valid DEX data after OI filter'}

            sorted_by_exposure = sorted(valid_strikes,
                                       key=lambda x: abs(x[1]['net']),
                                       reverse=True)

            key_strikes = [strike for strike, _ in sorted_by_exposure[:5]]

            # 7. Calculate put/call ratio
            pc_ratio = abs(total_put_dex / total_call_dex) if total_call_dex != 0 else 0

            # 8. Prepare result
            result = {
                'spot': round(spot, 2),
                'total_dex': round(total_dex, 0),
                'call_dex': round(total_call_dex, 0),
                'put_dex': round(total_put_dex, 0),
                'dex_by_strike': {
                    str(k): {
                        'net': round(v['net'], 0),
                        'call': round(v['call'], 0),
                        'put': round(v['put'], 0)
                    } for k, v in dex_by_strike.items()
                },
                'bias': bias,
                'pc_ratio': round(pc_ratio, 2),
                'key_strikes': key_strikes,
                'units': 'dollar_delta',  # Explicit units for downstream consumers
                'timestamp': time.time()
            }

            stats = await self._update_metric_statistics(symbol, 'dex', float(total_dex))
            result['zscore'] = stats['zscore']
            result['zscore_mean'] = stats['mean']
            result['zscore_stddev'] = stats['stdev']
            result['zscore_samples'] = stats['samples']
            result['zscore_window'] = stats['window']

            # 9. Store in Redis
            ttl = self.ttls.get('analytics', self.ttls.get('metrics', 60))
            await self.redis.setex(
                rkeys.analytics_dex_key(symbol),
                ttl,
                json.dumps(result)
            )

            # Log directional bias
            self.logger.info(f"DEX for {symbol}: Total={total_dex:.0f}, Bias={bias}, PC Ratio={pc_ratio:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Error calculating DEX for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
