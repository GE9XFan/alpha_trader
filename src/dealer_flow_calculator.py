#!/usr/bin/env python3
"""Dealer Flow Analytics - Vanna, Charm, Skew, and Hedging Impact."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytz
import redis.asyncio as aioredis

import redis_keys as rkeys
from option_utils import normalize_expiry

_SECONDS_IN_YEAR = 365.0 * 24 * 60 * 60
_TRADING_DAYS_PER_YEAR = 252.0
_EPS = 1e-12


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class ContractMetrics:
    symbol: str
    option_type: str
    strike: float
    expiry: str
    time_to_expiry_years: float
    days_to_expiry: float
    implied_vol: float
    delta: float
    gamma: float
    vega: float
    vanna: float
    charm_per_day: float
    open_interest: float
    volume: float


class DealerFlowCalculator:
    """Compute second-order dealer flow metrics from options chains."""

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        analytics_cfg = config.get('modules', {}).get('analytics', {})
        store_ttls = analytics_cfg.get('store_ttls', {})
        self.metric_ttl = int(store_ttls.get('analytics', 60))

        dealer_cfg = analytics_cfg.get('dealer_flow', {})
        self.contract_multiplier = int(dealer_cfg.get('contract_multiplier', 100))
        self.max_expiry_days = float(dealer_cfg.get('max_expiry_days', 30))
        self.history_window = int(dealer_cfg.get('history_window', 180))
        self.vol_beta = float(dealer_cfg.get('vol_beta', -0.30))  # Vol change per 1% price move
        self.min_open_interest = int(dealer_cfg.get('min_open_interest', 5))
        self.sigma_floor = float(dealer_cfg.get('sigma_floor', 0.02))
        self.min_samples_for_stats = int(dealer_cfg.get('min_samples', 15))

        self.eastern = pytz.timezone('US/Eastern')

    async def calculate_dealer_metrics(self, symbol: str) -> Dict[str, Any]:
        """Compute Vanna, Charm, skew, and hedging elasticity for ``symbol``."""

        try:
            chain_raw = await self.redis.get(rkeys.options_chain_key(symbol))
            ticker_raw = await self.redis.get(rkeys.market_ticker_key(symbol))
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Redis fetch failure for %s: %s", symbol, exc)
            return {'error': str(exc)}

        if not chain_raw or not ticker_raw:
            return {'error': 'missing_data'}

        try:
            chain_payload = json.loads(chain_raw)
        except Exception as exc:
            self.logger.error("Invalid chain payload for %s: %s", symbol, exc)
            return {'error': 'invalid_chain'}

        try:
            ticker = json.loads(ticker_raw)
        except Exception:
            ticker = {}

        spot = self._extract_spot_price(ticker)
        if spot <= 0:
            return {'error': 'invalid_spot'}

        contracts = self._iter_contracts(chain_payload)
        now = datetime.now(self.eastern)

        metrics: List[ContractMetrics] = []
        for contract in contracts:
            parsed = self._extract_contract_metrics(symbol, contract, spot, now)
            if parsed:
                metrics.append(parsed)

        if not metrics:
            await self._store_empty_metrics(symbol, spot)
            return {'error': 'no_valid_contracts'}

        totals = await self._aggregate_metrics(symbol, spot, metrics, now)
        return totals

    async def _store_empty_metrics(self, symbol: str, spot: float) -> None:
        timestamp = time.time()
        zero_payload = json.dumps({
            'symbol': symbol,
            'spot': spot,
            'timestamp': timestamp,
            'total_vanna_shares_per_pct_vol': 0.0,
            'total_charm_shares_per_day': 0.0,
            'samples': 0,
        })
        await self.redis.setex(rkeys.analytics_vanna_key(symbol), self.metric_ttl, zero_payload)
        await self.redis.setex(rkeys.analytics_charm_key(symbol), self.metric_ttl, zero_payload)
        await self.redis.setex(rkeys.analytics_skew_key(symbol), self.metric_ttl, zero_payload)
        await self.redis.setex(rkeys.analytics_hedging_impact_key(symbol), self.metric_ttl, zero_payload)

    def _iter_contracts(self, chain_payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        if isinstance(chain_payload, dict):
            by_contract = chain_payload.get('by_contract')
            if isinstance(by_contract, dict):
                return by_contract.values()
            contracts = chain_payload.get('contracts')
            if isinstance(contracts, list):
                return contracts
        if isinstance(chain_payload, list):
            return chain_payload
        return []

    def _extract_contract_metrics(
        self,
        symbol: str,
        contract: Dict[str, Any],
        spot: float,
        now: datetime,
    ) -> Optional[ContractMetrics]:
        option_type = str(contract.get('type') or contract.get('option_type') or '').lower()
        if option_type not in {'call', 'put'}:
            return None

        strike = self._to_float(contract.get('strike'))
        if not strike or strike <= 0:
            return None

        expiry_raw = contract.get('expiration') or contract.get('expiry')
        expiry_norm = normalize_expiry(expiry_raw)
        if not expiry_norm:
            return None

        expiry_dt = datetime.strptime(expiry_norm, '%Y%m%d')
        expiry_dt = self.eastern.localize(expiry_dt).replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_expiry = (expiry_dt - now).total_seconds()
        if time_to_expiry <= 60:
            return None

        days_to_expiry = time_to_expiry / 86400.0
        if days_to_expiry > self.max_expiry_days:
            return None

        implied_vol = self._to_float(contract.get('implied_volatility'))
        if implied_vol is None:
            return None
        if implied_vol > 3:
            implied_vol /= 100.0
        implied_vol = max(self.sigma_floor, float(implied_vol))

        T = time_to_expiry / _SECONDS_IN_YEAR
        if T <= 0:
            return None

        delta = self._sanitize_delta(contract.get('delta'), option_type, spot, strike, implied_vol, T)
        gamma = self._sanitize_gamma(contract.get('gamma'), option_type, spot, strike, implied_vol, T)
        vega = self._sanitize_vega(contract.get('vega'), option_type, spot, strike, implied_vol, T)
        vanna = self._compute_vanna(option_type, spot, strike, implied_vol, T)
        charm_per_day = self._compute_charm_per_day(option_type, spot, strike, implied_vol, T)

        open_interest = self._to_float(contract.get('open_interest'))
        if open_interest is None or open_interest < self.min_open_interest:
            return None

        volume = self._to_float(contract.get('volume')) or 0.0

        return ContractMetrics(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry_norm,
            time_to_expiry_years=T,
            days_to_expiry=days_to_expiry,
            implied_vol=implied_vol,
            delta=delta,
            gamma=gamma,
            vega=vega,
            vanna=vanna,
            charm_per_day=charm_per_day,
            open_interest=open_interest,
            volume=volume,
        )

    async def _aggregate_metrics(
        self,
        symbol: str,
        spot: float,
        metrics: List[ContractMetrics],
        now: datetime,
    ) -> Dict[str, Any]:
        total_vanna_shares_per_pct = 0.0
        total_charm_shares_per_day = 0.0
        total_gamma_shares_per_pct = 0.0
        total_vanna_vol_component = 0.0

        expiry_buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'vanna_shares_per_pct': 0.0,
            'charm_shares_per_day': 0.0,
            'gamma_shares_per_pct': 0.0,
            'open_interest': 0.0,
            'notional_traded': 0.0,
        })

        today_str = now.strftime('%Y%m%d')
        call_candidate: Optional[Tuple[float, ContractMetrics]] = None
        put_candidate: Optional[Tuple[float, ContractMetrics]] = None
        target_delta = 0.25

        for contract in metrics:
            weight = contract.open_interest * self.contract_multiplier
            vanna_shares = contract.vanna * weight
            vanna_shares_per_pct = vanna_shares * 0.01
            charm_shares_per_day = contract.charm_per_day * weight
            gamma_shares_per_pct = contract.gamma * weight * (0.01 * spot)
            vanna_vol_component_shares = vanna_shares * self.vol_beta * 0.01

            total_vanna_shares_per_pct += vanna_shares_per_pct
            total_charm_shares_per_day += charm_shares_per_day
            total_gamma_shares_per_pct += gamma_shares_per_pct
            total_vanna_vol_component += vanna_vol_component_shares

            bucket = expiry_buckets[contract.expiry]
            bucket['vanna_shares_per_pct'] += vanna_shares_per_pct
            bucket['charm_shares_per_day'] += charm_shares_per_day
            bucket['gamma_shares_per_pct'] += gamma_shares_per_pct
            bucket['open_interest'] += contract.open_interest
            bucket['notional_traded'] += contract.volume * spot

            if contract.expiry == today_str:
                delta_abs = abs(contract.delta)
                if contract.option_type == 'call':
                    diff = abs(delta_abs - target_delta)
                    if call_candidate is None or diff < call_candidate[0]:
                        call_candidate = (diff, contract)
                else:
                    diff = abs(delta_abs - target_delta)
                    if put_candidate is None or diff < put_candidate[0]:
                        put_candidate = (diff, contract)

        vanna_stats = await self._update_history(symbol, 'vanna', total_vanna_shares_per_pct * spot)
        charm_stats = await self._update_history(symbol, 'charm', total_charm_shares_per_day * spot)

        skew_payload = await self._compute_skew(symbol, call_candidate, put_candidate)

        vanna_payload = {
            'symbol': symbol,
            'spot': spot,
            'timestamp': time.time(),
            'total_vanna_shares_per_pct_vol': total_vanna_shares_per_pct,
            'total_vanna_notional_per_pct_vol': total_vanna_shares_per_pct * spot,
            'vol_beta_assumption': self.vol_beta,
            'history': vanna_stats,
            'per_expiry': self._serialize_expiry_buckets(expiry_buckets, spot, 'vanna_shares_per_pct'),
            'samples': len(metrics),
        }

        charm_payload = {
            'symbol': symbol,
            'spot': spot,
            'timestamp': time.time(),
            'total_charm_shares_per_day': total_charm_shares_per_day,
            'total_charm_notional_per_day': total_charm_shares_per_day * spot,
            'history': charm_stats,
            'per_expiry': self._serialize_expiry_buckets(expiry_buckets, spot, 'charm_shares_per_day'),
            'samples': len(metrics),
        }

        hedging_payload = {
            'symbol': symbol,
            'spot': spot,
            'timestamp': time.time(),
            'shares_per_pct_move': total_gamma_shares_per_pct + total_vanna_vol_component,
            'gamma_component_shares': total_gamma_shares_per_pct,
            'vanna_component_shares': total_vanna_vol_component,
            'notional_per_pct_move': (total_gamma_shares_per_pct + total_vanna_vol_component) * spot,
            'per_expiry': self._serialize_expiry_buckets(expiry_buckets, spot, 'gamma_shares_per_pct'),
            'charm_shares_per_day': total_charm_shares_per_day,
            'charm_notional_per_day': total_charm_shares_per_day * spot,
        }

        async with self.redis.pipeline(transaction=False) as pipe:
            await pipe.setex(rkeys.analytics_vanna_key(symbol), self.metric_ttl, json.dumps(vanna_payload))
            await pipe.setex(rkeys.analytics_charm_key(symbol), self.metric_ttl, json.dumps(charm_payload))
            await pipe.setex(rkeys.analytics_hedging_impact_key(symbol), self.metric_ttl, json.dumps(hedging_payload))
            await pipe.setex(rkeys.analytics_skew_key(symbol), self.metric_ttl, json.dumps(skew_payload))
            await pipe.execute()

        return {
            'vanna': vanna_payload,
            'charm': charm_payload,
            'skew': skew_payload,
            'hedging': hedging_payload,
        }

    def _serialize_expiry_buckets(
        self,
        buckets: Dict[str, Dict[str, float]],
        spot: float,
        field: str,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for expiry, payload in buckets.items():
            value = payload.get(field, 0.0)
            items.append({
                'expiry': expiry,
                field: value,
                f'{field}_notional': value * spot,
                'open_interest': payload.get('open_interest', 0.0),
            })
        items.sort(key=lambda item: item['expiry'])
        return items

    async def _compute_skew(
        self,
        symbol: str,
        call_candidate: Optional[Tuple[float, ContractMetrics]],
        put_candidate: Optional[Tuple[float, ContractMetrics]],
    ) -> Dict[str, Any]:
        timestamp = time.time()
        if not call_candidate or not put_candidate:
            payload = {
                'symbol': symbol,
                'timestamp': timestamp,
                'skew': None,
                'target_delta': 0.25,
                'call_iv': None,
                'put_iv': None,
                'history': {
                    'zscore': 0.0,
                    'mean': 0.0,
                    'stdev': 0.0,
                    'samples': 0,
                    'window': self.history_window,
                },
            }
            return payload

        call_contract = call_candidate[1]
        put_contract = put_candidate[1]
        call_iv = call_contract.implied_vol
        put_iv = put_contract.implied_vol

        skew_value = None
        skew_ratio = None
        if call_iv and put_iv:
            skew_value = float(put_iv - call_iv)
            if call_iv != 0:
                skew_ratio = float(put_iv / call_iv - 1.0)

        stats = await self._update_history(symbol, 'skew', skew_value if skew_value is not None else 0.0)

        payload = {
            'symbol': symbol,
            'timestamp': timestamp,
            'skew': skew_value,
            'skew_ratio': skew_ratio,
            'call_iv': call_iv,
            'put_iv': put_iv,
            'history': stats,
            'target_delta': 0.25,
            'call_strike': call_contract.strike,
            'put_strike': put_contract.strike,
        }
        return payload

    async def _update_history(self, symbol: str, metric: str, value: float) -> Dict[str, float]:
        history_key = f'analytics:{symbol}:{metric}:history'
        stats = {
            'zscore': 0.0,
            'mean': float(value),
            'stdev': 0.0,
            'samples': 0,
            'window': self.history_window,
        }
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                await pipe.lpush(history_key, value)
                await pipe.ltrim(history_key, 0, self.history_window - 1)
                await pipe.expire(history_key, max(self.history_window * 5, 3600))
                await pipe.lrange(history_key, 0, self.history_window - 1)
                _, _, _, history = await pipe.execute()
        except Exception as exc:  # pragma: no cover - defensive telemetry
            self.logger.debug("Failed to update %s history for %s: %s", metric, symbol, exc)
            return stats

        values: List[float] = []
        for raw in history:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8', errors='ignore')
                values.append(float(raw))
            except (TypeError, ValueError):
                continue

        if not values:
            return stats

        stats['samples'] = len(values)
        mean = sum(values) / len(values)
        stats['mean'] = mean

        if len(values) < self.min_samples_for_stats:
            return stats

        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stdev = math.sqrt(max(variance, _EPS))
        stats['stdev'] = stdev

        if stdev > 0:
            stats['zscore'] = (float(value) - mean) / stdev

        return stats

    def _extract_spot_price(self, ticker: Dict[str, Any]) -> float:
        price_fields = ['last', 'mid', 'close', 'price', 'mark']
        for field in price_fields:
            value = self._to_float(ticker.get(field))
            if value and value > 0:
                return float(value)
        bid = self._to_float(ticker.get('bid'))
        ask = self._to_float(ticker.get('ask'))
        if bid and ask and bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)
        return 0.0

    def _to_float(self, value: Any) -> Optional[float]:
        if value in (None, '', 'null'):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                if isinstance(value, str):
                    return float(value.replace('%', ''))
            except Exception:
                pass
        return None

    def _sanitize_delta(
        self,
        value: Any,
        option_type: str,
        spot: float,
        strike: float,
        sigma: float,
        T: float,
    ) -> float:
        delta = self._to_float(value)
        if delta is None:
            return self._bsm_delta(option_type, spot, strike, sigma, T)
        if abs(delta) > 1.5:  # Provider occasionally returns percent
            delta /= 100.0
        return float(delta)

    def _sanitize_gamma(
        self,
        value: Any,
        option_type: str,
        spot: float,
        strike: float,
        sigma: float,
        T: float,
    ) -> float:
        gamma = self._to_float(value)
        if gamma is None or gamma <= 0:
            return self._bsm_gamma(spot, strike, sigma, T)
        if gamma > 5:  # Protect against scale mismatch
            gamma /= 100.0
        return float(gamma)

    def _sanitize_vega(
        self,
        value: Any,
        option_type: str,
        spot: float,
        strike: float,
        sigma: float,
        T: float,
    ) -> float:
        vega = self._to_float(value)
        if vega is None:
            return self._bsm_vega(spot, strike, sigma, T)
        if vega > 1000:
            vega /= 100.0
        return float(vega)

    def _bsm_delta(self, option_type: str, spot: float, strike: float, sigma: float, T: float) -> float:
        if spot <= 0 or strike <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
        if option_type == 'call':
            return _norm_cdf(d1)
        return _norm_cdf(d1) - 1.0

    def _bsm_gamma(self, spot: float, strike: float, sigma: float, T: float) -> float:
        if spot <= 0 or strike <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
        return _norm_pdf(d1) / (spot * sigma * sqrt_T)

    def _bsm_vega(self, spot: float, strike: float, sigma: float, T: float) -> float:
        if spot <= 0 or strike <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
        return spot * sqrt_T * _norm_pdf(d1)

    def _compute_vanna(self, option_type: str, spot: float, strike: float, sigma: float, T: float) -> float:
        if spot <= 0 or strike <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        step = max(self.sigma_floor * 0.25, sigma * 0.1, 0.01)
        sigma_up = sigma + step
        sigma_down = max(self.sigma_floor, sigma - step)
        if sigma_up == sigma_down:
            sigma_down = max(self.sigma_floor, sigma * 0.5)
        delta_up = self._bsm_delta(option_type, spot, strike, sigma_up, T)
        delta_down = self._bsm_delta(option_type, spot, strike, sigma_down, T)
        denom = sigma_up - sigma_down
        if denom == 0:
            return 0.0
        return (delta_up - delta_down) / denom

    def _compute_charm_per_day(self, option_type: str, spot: float, strike: float, sigma: float, T: float) -> float:
        if spot <= 0 or strike <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        delta_now = self._bsm_delta(option_type, spot, strike, sigma, T)
        dt = min(max(T * 0.1, 1.0 / 3650.0), T)
        future_T = max(T - dt, 1e-6)
        delta_future = self._bsm_delta(option_type, spot, strike, sigma, future_T)
        delta_change = (delta_future - delta_now) / dt
        return delta_change / _TRADING_DAYS_PER_YEAR
