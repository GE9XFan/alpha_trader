#!/usr/bin/env python3
"""
Check distribution queue status and diagnose why signals aren't moving.
"""

import redis
import json
import yaml
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
print("DISTRIBUTION QUEUE ANALYSIS")
print("=" * 80)

# Check distribution queues
queues = {
    'Premium': 'distribution:premium:queue',
    'Basic': 'distribution:basic:queue',
    'Free': 'distribution:free:queue'
}

for tier, queue_key in queues.items():
    queue_len = r.llen(queue_key)
    print(f"\n{tier} Queue: {queue_len} signals")
    
    if queue_len > 0:
        # Get first and last signals
        first_signal = r.lindex(queue_key, 0)
        last_signal = r.lindex(queue_key, -1)
        
        try:
            first_data = json.loads(first_signal)
            last_data = json.loads(last_signal)
            
            print(f"  First signal: {first_data.get('symbol', 'N/A')} at {first_data.get('ts', 'N/A')}")
            print(f"  Last signal: {last_data.get('symbol', 'N/A')} at {last_data.get('ts', 'N/A')}")
            
            # Check timestamp to see how old signals are
            if 'ts' in first_data:
                ts = datetime.fromtimestamp(first_data['ts'])
                age = (datetime.now() - ts).total_seconds()
                print(f"  Oldest signal age: {age:.0f} seconds")
                
        except Exception as e:
            print(f"  Error parsing signals: {e}")

print("\n" + "=" * 80)
print("CHECKING PENDING QUEUES (SOURCE)")
print("=" * 80)

# Check pending queues for each symbol
symbols = config.get('symbols', {})
all_symbols = list(set(symbols.get('level2', []) + symbols.get('standard', [])))

total_pending = 0
for symbol in all_symbols:
    pending_key = f'signals:pending:{symbol}'
    pending_count = r.llen(pending_key)
    if pending_count > 0:
        print(f"{symbol}: {pending_count} pending signals")
        total_pending += pending_count

print(f"\nTotal pending signals: {total_pending}")

print("\n" + "=" * 80)
print("CHECKING SIGNAL GENERATION")
print("=" * 80)

# Check if signals are being generated
signal_count_key = 'metrics:signals:generated'
signal_count = r.get(signal_count_key)
print(f"Total signals generated (counter): {signal_count}")

# Check recent signals
recent_signals_key = 'signals:recent'
recent_count = r.llen(recent_signals_key)
print(f"Recent signals list: {recent_count} signals")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# The issue is clear: signals are being added to distribution queues
# but nothing is consuming them (no BRPOP operations found)

print("""
ISSUE IDENTIFIED: 
1. SignalDistributor is pushing signals to distribution queues
2. No consumer is reading from these queues (no BRPOP operations found)
3. Signals accumulate in queues without being processed

SOLUTION NEEDED:
- Implement consumers for distribution queues
- Social media module has TODO comments but no implementation
- Execution module doesn't consume from distribution queues
- Need to implement queue consumers that:
  a) Read from distribution:premium:queue
  b) Read from distribution:basic:queue  
  c) Read from distribution:free:queue
  d) Send signals to appropriate destinations (Discord, trading, etc.)
""")