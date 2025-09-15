#!/usr/bin/env python3
"""
Fix distribution queue issues by:
1. Processing any scheduled signals that are ready
2. Ensuring all tiers have the same signal count
"""

import redis
import json
import yaml
import time
from datetime import datetime

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
print("FIXING DISTRIBUTION QUEUES")
print("=" * 80)

# Check current queue sizes
queues = {
    'Premium': 'distribution:premium:queue',
    'Basic': 'distribution:basic:queue',
    'Free': 'distribution:free:queue'
}

print("\nCurrent queue sizes:")
for tier, queue_key in queues.items():
    queue_len = r.llen(queue_key)
    print(f"  {tier}: {queue_len} signals")

# Check for scheduled signals
print("\nChecking scheduled signals...")
current_time = time.time()

# Process basic tier scheduled signals
basic_scheduled_key = 'distribution:scheduled:basic'
basic_ready = r.zrangebyscore(
    basic_scheduled_key,
    min=0,
    max=current_time,
    withscores=True
)

if basic_ready:
    print(f"\nFound {len(basic_ready)} basic signals ready to publish")
    for signal_json, score in basic_ready:
        r.lpush('distribution:basic:queue', signal_json)
        r.zrem(basic_scheduled_key, signal_json)
        print(f"  Published basic signal scheduled for {datetime.fromtimestamp(score)}")

# Check all scheduled basic signals (including future)
all_basic_scheduled = r.zcard(basic_scheduled_key)
if all_basic_scheduled > 0:
    print(f"  {all_basic_scheduled} basic signals still scheduled for future")

# Process free tier scheduled signals
free_scheduled_key = 'distribution:scheduled:free'
free_ready = r.zrangebyscore(
    free_scheduled_key,
    min=0,
    max=current_time,
    withscores=True
)

if free_ready:
    print(f"\nFound {len(free_ready)} free signals ready to publish")
    for signal_json, score in free_ready:
        r.lpush('distribution:free:queue', signal_json)
        r.zrem(free_scheduled_key, signal_json)
        print(f"  Published free signal scheduled for {datetime.fromtimestamp(score)}")

# Check all scheduled free signals (including future)
all_free_scheduled = r.zcard(free_scheduled_key)
if all_free_scheduled > 0:
    print(f"  {all_free_scheduled} free signals still scheduled for future")

print("\n" + "=" * 80)
print("AFTER PROCESSING")
print("=" * 80)

# Check queue sizes after processing
print("\nUpdated queue sizes:")
for tier, queue_key in queues.items():
    queue_len = r.llen(queue_key)
    print(f"  {tier}: {queue_len} signals")

# Get premium count as reference
premium_count = r.llen('distribution:premium:queue')
basic_count = r.llen('distribution:basic:queue')
free_count = r.llen('distribution:free:queue')

# Calculate missing signals
basic_missing = premium_count - basic_count - r.zcard(basic_scheduled_key)
free_missing = premium_count - free_count - r.zcard(free_scheduled_key)

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if basic_missing > 0:
    print(f"WARNING: {basic_missing} signals missing from basic tier (lost during restarts)")
if free_missing > 0:
    print(f"WARNING: {free_missing} signals missing from free tier (lost during restarts)")

if basic_missing == 0 and free_missing == 0:
    print("✓ All signals accounted for (including scheduled)")
else:
    print("\nTo fully fix, you would need to:")
    print("1. Copy missing signals from premium queue to basic/free")
    print("2. Or wait for new signals with the fixed code")
    print("\nThe fix in signals.py will prevent this issue for future signals.")