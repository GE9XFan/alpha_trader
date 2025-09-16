#!/usr/bin/env python3
"""
Fix Redis position data - cleans up stale position references
"""
import redis
import json

def fix_positions():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    print("🔧 Fixing Redis position data...")

    # 1. Clean up empty positions:by_symbol sets
    symbol_keys = r.keys('positions:by_symbol:*')
    for key in symbol_keys:
        members = r.scard(key)
        if members == 0:
            r.delete(key)
            print(f"  ✓ Deleted empty set: {key}")
        else:
            # Check if members actually have positions
            symbol = key.split(':')[-1]
            position_ids = r.smembers(key)
            valid_ids = []
            for pid in position_ids:
                if r.exists(f'positions:open:{symbol}:{pid}'):
                    valid_ids.append(pid)
                else:
                    r.srem(key, pid)
                    print(f"  ✓ Removed stale ID {pid} from {symbol}")

            # If no valid IDs remain, delete the set
            if not valid_ids:
                r.delete(key)
                print(f"  ✓ Deleted empty set: {key}")

    # 2. Count actual open positions and set positions:count
    position_keys = r.keys('positions:open:*')
    actual_count = len(position_keys)
    r.set('positions:count', actual_count)
    print(f"  ✓ Set positions:count to {actual_count}")

    # 3. Ensure positions:pnl:realized:total exists (initialize to 0 if missing)
    if not r.exists('positions:pnl:realized:total'):
        r.set('positions:pnl:realized:total', 0)
        print("  ✓ Initialized positions:pnl:realized:total to 0")

    # 4. Check if daily P&L needs to be preserved (don't reset if midday)
    from datetime import datetime
    now = datetime.now()
    last_reset = r.get('risk:daily:reset_time')

    if last_reset:
        last_reset_dt = datetime.fromtimestamp(float(last_reset))
        if last_reset_dt.date() != now.date():
            print("  ⚠️ Daily P&L data is from a previous day - will be reset at market open")
    else:
        print("  ⚠️ No daily reset timestamp found - will be set at next market open")

    print("\n✅ Redis position data fixed!")

    # Show current state
    print("\n📊 Current State:")
    print(f"  Open Positions: {actual_count}")
    print(f"  Daily P&L: ${float(r.get('risk:daily_pnl') or 0):.2f}")
    print(f"  Total Realized P&L: ${float(r.get('positions:pnl:realized:total') or 0):.2f}")

    symbol_keys = r.keys('positions:by_symbol:*')
    if symbol_keys:
        print(f"  Active Symbols: {len(symbol_keys)}")
        for key in symbol_keys:
            symbol = key.split(':')[-1]
            count = r.scard(key)
            if count > 0:
                print(f"    - {symbol}: {count} position(s)")

if __name__ == "__main__":
    fix_positions()