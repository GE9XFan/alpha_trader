#!/usr/bin/env python3
"""
Restore missing signals to basic and free queues by copying from premium queue.
This fixes the issue where signals were lost due to non-persistent asyncio tasks.
"""

import redis
import json
import yaml
import time

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to Redis
redis_config = config.get('redis', {})
r = redis.Redis(
    host=redis_config.get('host', 'localhost'),
    port=redis_config.get('port', 6379),
    db=redis_config.get('db', 0),
    decode_responses=True
)

print("=" * 80)
print("RESTORING MISSING SIGNALS")
print("=" * 80)

# Get current queue sizes
premium_queue = 'distribution:premium:queue'
basic_queue = 'distribution:basic:queue'
free_queue = 'distribution:free:queue'

premium_count = r.llen(premium_queue)
basic_count = r.llen(basic_queue)
free_count = r.llen(free_queue)

print(f"\nCurrent state:")
print(f"  Premium: {premium_count} signals")
print(f"  Basic: {basic_count} signals (missing {premium_count - basic_count})")
print(f"  Free: {free_count} signals (missing {premium_count - free_count})")

# Get the last N signals from premium to copy to basic
basic_missing = premium_count - basic_count
if basic_missing > 0:
    print(f"\nCopying {basic_missing} signals to basic queue...")
    # Get the oldest signals from premium (they should have been in basic by now)
    for i in range(basic_missing):
        signal_json = r.lindex(premium_queue, -(i+1))  # Get from end (oldest)
        if signal_json:
            try:
                signal = json.loads(signal_json)
                # Format as basic signal
                confidence = signal.get('confidence', 0)
                if confidence >= 80:
                    conf_band = 'HIGH'
                elif confidence >= 65:
                    conf_band = 'MEDIUM'
                else:
                    conf_band = 'LOW'
                
                basic_signal = {
                    'symbol': signal.get('symbol'),
                    'side': signal.get('side'),
                    'strategy': signal.get('strategy'),
                    'confidence_band': conf_band,
                    'ts': signal.get('ts')
                }
                
                # Add to basic queue
                r.rpush(basic_queue, json.dumps(basic_signal))
                print(f"  Added signal {i+1}/{basic_missing} to basic queue")
            except Exception as e:
                print(f"  Error processing signal {i+1}: {e}")

# Get the last N signals from premium to copy to free
free_missing = premium_count - free_count
if free_missing > 0:
    print(f"\nCopying {free_missing} signals to free queue...")
    # Get the oldest signals from premium (they should have been in free by now)
    for i in range(free_missing):
        signal_json = r.lindex(premium_queue, -(i+1))  # Get from end (oldest)
        if signal_json:
            try:
                signal = json.loads(signal_json)
                # Format as free signal
                side = signal.get('side', '')
                sentiment = 'bullish' if side == 'LONG' else 'bearish' if side == 'SHORT' else 'neutral'
                
                free_signal = {
                    'symbol': signal.get('symbol'),
                    'sentiment': sentiment,
                    'message': f"New {sentiment} signal on {signal.get('symbol')}. Upgrade for full details!",
                    'ts': signal.get('ts')
                }
                
                # Add to free queue
                r.rpush(free_queue, json.dumps(free_signal))
                print(f"  Added signal {i+1}/{free_missing} to free queue")
            except Exception as e:
                print(f"  Error processing signal {i+1}: {e}")

print("\n" + "=" * 80)
print("FINAL STATE")
print("=" * 80)

# Check final queue sizes
premium_count = r.llen(premium_queue)
basic_count = r.llen(basic_queue)
free_count = r.llen(free_queue)

print(f"\nFinal queue sizes:")
print(f"  Premium: {premium_count} signals")
print(f"  Basic: {basic_count} signals")
print(f"  Free: {free_count} signals")

if premium_count == basic_count == free_count:
    print("\n✓ SUCCESS: All queues now have the same number of signals!")
else:
    print("\n⚠ WARNING: Queue counts still don't match. Check for errors above.")

print("\nNOTE: The fix in signals.py will prevent this issue for all future signals.")
print("Future signals will be stored in Redis sorted sets, surviving process restarts.")