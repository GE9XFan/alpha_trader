"""
Signal Deduplication Module

Implements idempotency, deduplication, and cooldown logic for trading signals.
Ensures the same signal isn't emitted multiple times and manages contract
fingerprinting for unique identification.

Redis Keys Used:
    Read/Write:
        - signals:emitted:{signal_id} (idempotency check)
        - signals:cooldown:{contract_fp} (cooldown tracking)
        - signals:last_conf:{contract_fp} (confidence tracking)
        - signals:last_contract:{symbol}:{strategy}:{side}:{dte_band} (hysteresis)
        - signals:audit:{contract_fp} (audit trail)
        - metrics:signals:thin_update_blocked (metrics)
        - metrics:signals:duplicate_blocked (metrics)

Author: Quantisity Capital
Version: 3.0.0
"""

import time
import json
import hashlib
import pytz
from datetime import datetime
from typing import Dict, Any, Optional
import logging


def contract_fingerprint(symbol: str, strategy: str, side: str, contract: dict) -> str:
    """
    Stable identity for the specific option contract choice.
    Includes multiplier and exchange to handle edge cases (minis, different venues).
    """
    parts = (
        symbol,
        strategy,
        side,
        str(contract.get('expiry')),
        str(contract.get('right')),
        str(contract.get('strike')),
        str(contract.get('multiplier', 100)),  # Default 100 for standard equity options
        str(contract.get('exchange', 'SMART')),  # Default SMART for IB routing
    )
    return "sigfp:" + hashlib.sha1(":".join(parts).encode()).hexdigest()[:20]


def trading_day_bucket(ts: float = None) -> str:
    """
    Return YYYYMMDD in US/Eastern; aligns with the trading session day.
    Market day changes at 4PM ET, not midnight.
    """
    ET = pytz.timezone("America/New_York")
    dt = datetime.fromtimestamp(ts or time.time(), tz=ET)
    return dt.strftime("%Y%m%d")


class SignalDeduplication:
    """
    Manages signal deduplication and idempotency enforcement.

    Prevents duplicate signals from being emitted and manages cooldown
    periods between signals for the same contract.
    """

    def __init__(self, config: Dict[str, Any], redis_conn):
        """
        Initialize signal deduplication system.

        Args:
            config: System configuration
            redis_conn: Redis connection for state management
        """
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        # Configuration
        signal_config = config.get('modules', {}).get('signals', {})
        self.cooldown_s = signal_config.get('cooldown_s', 30)
        self.ttl_seconds = signal_config.get('ttl_seconds', 300)
        self.version = signal_config.get('version', 'D6.0.1')

        # Atomic Redis Lua script for idempotency + enqueue + cooldown
        self.LUA_ATOMIC_EMIT = """
        -- KEYS[1] = idempotency_key "signals:emitted:<emit_id>"
        -- KEYS[2] = cooldown_key    "signals:cooldown:<contract_fp>"
        -- KEYS[3] = queue_pending   "signals:pending:<symbol>"
        -- KEYS[4] = queue_execution "signals:execution:<symbol>"
        -- ARGV[1] = signal_json
        -- ARGV[2] = idempotency_ttl_seconds
        -- ARGV[3] = cooldown_ttl_seconds

        if redis.call('SETNX', KEYS[1], '1') == 1 then
            redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) * 1000)
            if redis.call('EXISTS', KEYS[2]) == 0 then
                redis.call('LPUSH', KEYS[3], ARGV[1])
                redis.call('LPUSH', KEYS[4], ARGV[1])
                redis.call('PEXPIRE', KEYS[2], tonumber(ARGV[3]) * 1000)
                return 1  -- Signal enqueued
            else
                return -1  -- Blocked by cooldown
            end
        else
            return 0  -- Duplicate signal
        end
        """
        self.lua_sha = None  # Will be loaded on first use

    async def check_cooldown(self, contract_fp: str) -> bool:
        """
        Check if cooldown allows new signal for this specific contract.

        Args:
            contract_fp: Contract fingerprint

        Returns:
            True if cooldown expired, False if still active
        """
        key = f"signals:cooldown:{contract_fp}"
        return not bool(await self.redis.exists(key))

    async def check_idempotency(self, signal_id: str) -> bool:
        """
        Check if signal is duplicate.

        Args:
            signal_id: Unique signal identifier

        Returns:
            True if new signal, False if duplicate
        """
        emitted_key = f'signals:emitted:{signal_id}'
        # Use SET NX (set if not exists)
        result = await self.redis.set(emitted_key, '1', nx=True, ex=self.ttl_seconds)
        return result is not None

    def generate_signal_id(self, symbol: str, side: str, contract_fp: str) -> str:
        """
        Idempotent ID for 'this contract, this side, today'.
        Avoids re-emitting micro-variants for the same contract.

        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            contract_fp: Contract fingerprint

        Returns:
            Unique signal identifier
        """
        components = f"{contract_fp}:{self.version}:{trading_day_bucket()}"
        return hashlib.sha1(components.encode()).hexdigest()[:16]

    async def check_material_change(self, contract_fp: str, confidence: int) -> bool:
        """
        Check if confidence change is material enough to warrant new signal.

        Args:
            contract_fp: Contract fingerprint
            confidence: New confidence value

        Returns:
            True if change is material, False otherwise
        """
        # Check for material change (skip minor confidence updates)
        # Use relative threshold: 3 points or 5%, whichever is greater
        last_c_key = f"signals:last_conf:{contract_fp}"
        last_conf = await self.redis.get(last_c_key)

        if last_conf is not None:
            last_conf_val = int(last_conf)
            delta = abs(confidence - last_conf_val)
            threshold = max(3, 0.05 * max(1, last_conf_val))  # 3 pts or 5%

            if delta < threshold:
                await self.redis.incr('metrics:signals:thin_update_blocked')

                # Add to audit trail
                audit_key = f"signals:audit:{contract_fp}"
                audit_entry = json.dumps({
                    "ts": time.time(),
                    "action": "blocked",
                    "reason": "thin_update",
                    "conf": confidence,
                    "last_conf": last_conf_val,
                    "delta": delta,
                    "threshold": threshold
                })
                await self.redis.lpush(audit_key, audit_entry)
                await self.redis.ltrim(audit_key, 0, 50)  # Keep last 50 entries
                await self.redis.expire(audit_key, 3600)  # 1 hour TTL
                return False

        # Update last confidence with sliding TTL
        await self.redis.setex(last_c_key, 900, int(confidence))
        return True

    async def atomic_emit(self, signal_id: str, contract_fp: str, symbol: str,
                         signal: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """
        Atomically check idempotency, cooldown, and emit signal if valid.

        Args:
            signal_id: Unique signal identifier
            contract_fp: Contract fingerprint
            symbol: Trading symbol
            signal: Signal data to emit
            ttl: Optional TTL override

        Returns:
            1: Signal successfully enqueued
            0: Duplicate signal (already emitted)
            -1: Blocked by cooldown
        """
        # Load Lua script if not already loaded
        if not self.lua_sha:
            self.lua_sha = await self.redis.script_load(self.LUA_ATOMIC_EMIT)

        # Use provided TTL or default
        if ttl is None:
            ttl = self.ttl_seconds

        # Prepare keys and arguments
        idempotency_key = f'signals:emitted:{signal_id}'
        cooldown_key = f'signals:cooldown:{contract_fp}'
        queue_key_pending = f'signals:pending:{symbol}'
        queue_key_execution = f'signals:execution:{symbol}'

        signal_json = json.dumps(signal)

        try:
            # Execute atomic operation
            result = await self.redis.evalsha(
                self.lua_sha,
                4,  # Number of keys
                idempotency_key,
                cooldown_key,
                queue_key_pending,
                queue_key_execution,
                signal_json,
                str(ttl),
                str(self.cooldown_s)
            )

            return int(result)
        except Exception as e:
            if "NOSCRIPT" in str(e):
                # Script was evicted, reload and retry
                self.lua_sha = await self.redis.script_load(self.LUA_ATOMIC_EMIT)
                result = await self.redis.evalsha(
                    self.lua_sha,
                    4,
                    idempotency_key,
                    cooldown_key,
                    queue_key_pending,
                    queue_key_execution,
                    signal_json,
                    str(ttl),
                    str(self.cooldown_s)
                )
                return int(result)
            else:
                raise

    async def add_audit_entry(self, contract_fp: str, action: str, reason: str,
                             details: Dict[str, Any]):
        """
        Add entry to signal audit trail.

        Args:
            contract_fp: Contract fingerprint
            action: Action taken (emitted, blocked, etc)
            reason: Reason for action
            details: Additional details
        """
        audit_key = f"signals:audit:{contract_fp}"
        audit_entry = json.dumps({
            "ts": time.time(),
            "action": action,
            "reason": reason,
            **details
        })

        await self.redis.lpush(audit_key, audit_entry)
        await self.redis.ltrim(audit_key, 0, 50)  # Keep last 50 entries
        await self.redis.expire(audit_key, 3600)  # 1 hour TTL

    async def get_contract_hysteresis(self, symbol: str, strategy: str, side: str,
                                     dte_band: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get last contract used for hysteresis prevention.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            side: LONG or SHORT
            dte_band: Optional DTE band identifier

        Returns:
            Last contract data or None
        """
        fp_key = f"signals:last_contract:{symbol}:{strategy}:{side}:{dte_band or 'NA'}"
        last_contract_str = await self.redis.get(fp_key)

        if last_contract_str:
            try:
                return json.loads(last_contract_str)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    async def set_contract_hysteresis(self, symbol: str, strategy: str, side: str,
                                     contract: Dict[str, Any], dte_band: Optional[str] = None):
        """
        Store contract for hysteresis prevention.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            side: LONG or SHORT
            contract: Contract data to store
            dte_band: Optional DTE band identifier
        """
        fp_key = f"signals:last_contract:{symbol}:{strategy}:{side}:{dte_band or 'NA'}"
        # Remember this contract for next time (10 minute TTL)
        await self.redis.setex(fp_key, 600, json.dumps(contract))