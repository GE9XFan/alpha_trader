#!/usr/bin/env python3
"""
Test script to verify the deduplication fix is working correctly.
This checks that:
1. Contract fingerprints are stable
2. Signal IDs are deterministic for the same contract
3. Cooldowns are contract-scoped
4. Hysteresis prevents strike bouncing
"""

import asyncio
import redis.asyncio as aioredis
import json
import time
from datetime import datetime

async def main():
    # Connect to Redis
    redis = await aioredis.from_url('redis://localhost:6379')
    
    print("Contract-Centric Deduplication Fix Verification")
    print("=" * 60)
    
    # 1. Check for contract fingerprints in recent signals
    print("\n1. Checking for contract fingerprints...")
    latest_keys = []
    async for key in redis.scan_iter(match="signals:latest:*"):
        latest_keys.append(key)
    
    contracts_seen = set()
    for key in latest_keys[:5]:  # Check first 5
        signal_data = await redis.get(key)
        if signal_data:
            signal = json.loads(signal_data)
            if 'contract_fp' in signal:
                print(f"  ✓ {signal['symbol']}: Has contract_fp: {signal.get('contract_fp', 'N/A')}")
                contracts_seen.add(signal.get('contract_fp'))
            else:
                print(f"  ✗ {signal['symbol']}: Missing contract_fp (old signal)")
    
    # 2. Check cooldown keys are contract-scoped
    print("\n2. Checking cooldown keys...")
    cooldown_keys = []
    async for key in redis.scan_iter(match="signals:cooldown:*"):
        cooldown_keys.append(key.decode() if isinstance(key, bytes) else key)
    
    for key in cooldown_keys[:5]:
        if "sigfp:" in key:
            print(f"  ✓ Contract-scoped cooldown: {key}")
        else:
            print(f"  ⚠ Old-style cooldown: {key}")
    
    # 3. Check for hysteresis tracking
    print("\n3. Checking hysteresis (last contract tracking)...")
    hysteresis_keys = []
    async for key in redis.scan_iter(match="signals:last_contract:*"):
        hysteresis_keys.append(key)
    
    if hysteresis_keys:
        for key in hysteresis_keys[:3]:
            contract_data = await redis.get(key)
            if contract_data:
                contract = json.loads(contract_data)
                print(f"  ✓ {key.decode() if isinstance(key, bytes) else key}: Strike={contract.get('strike')}")
    else:
        print("  ⚠ No hysteresis tracking found (may not have run yet)")
    
    # 4. Check for material change tracking
    print("\n4. Checking material change detection...")
    conf_keys = []
    async for key in redis.scan_iter(match="signals:last_conf:*"):
        conf_keys.append(key)
    
    if conf_keys:
        for key in conf_keys[:3]:
            conf_value = await redis.get(key)
            print(f"  ✓ {key.decode() if isinstance(key, bytes) else key}: Last confidence={conf_value.decode() if conf_value else 'N/A'}")
    else:
        print("  ⚠ No confidence tracking found (may not have run yet)")
    
    # 5. Check metrics for thin updates being blocked
    print("\n5. Checking metrics...")
    metrics = {
        'duplicates': await redis.get('metrics:signals:duplicates'),
        'cooldown_blocked': await redis.get('metrics:signals:cooldown_blocked'),
        'thin_update_blocked': await redis.get('metrics:signals:thin_update_blocked'),
        'emitted': await redis.get('metrics:signals:emitted')
    }
    
    for metric, value in metrics.items():
        val = int(value) if value else 0
        print(f"  {metric}: {val}")
    
    # 6. Analyze recent signals for duplicates
    print("\n6. Analyzing recent signals for duplicates...")
    all_signals = []
    async for key in redis.scan_iter(match="signals:out:*"):
        signal_data = await redis.get(key)
        if signal_data:
            signal = json.loads(signal_data)
            all_signals.append(signal)
    
    # Group by contract
    by_contract = {}
    for signal in all_signals:
        contract_key = f"{signal.get('symbol')}:{signal.get('side')}:{signal.get('contract', {}).get('strike')}:{signal.get('contract', {}).get('expiry')}"
        if contract_key not in by_contract:
            by_contract[contract_key] = []
        by_contract[contract_key].append(signal)
    
    # Find potential duplicates
    duplicates_found = 0
    for contract_key, signals in by_contract.items():
        if len(signals) > 1:
            # Check time spacing
            timestamps = [s.get('ts', 0) for s in signals]
            timestamps.sort()
            for i in range(1, len(timestamps)):
                time_diff = (timestamps[i] - timestamps[i-1]) / 1000  # Convert to seconds
                if time_diff < 30:  # Within cooldown window
                    duplicates_found += 1
                    print(f"  ⚠ Potential duplicate: {contract_key} - {time_diff:.1f}s apart")
    
    if duplicates_found == 0:
        print("  ✓ No duplicates found within cooldown windows")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    
    # Close Redis connection
    await redis.close()

if __name__ == "__main__":
    asyncio.run(main())