#!/usr/bin/env python3
"""
Test script to verify signal generation fixes
"""

import asyncio
import json
import redis.asyncio as aioredis
import time

async def test_fixes():
    """Test the signal generation fixes."""
    
    # Connect to Redis
    redis = aioredis.Redis(
        host='127.0.0.1',
        port=6379,
        decode_responses=True
    )
    
    print("Testing Signal Generation Fixes")
    print("=" * 50)
    
    try:
        # Test 1: Check market:last format
        print("\n1. Checking market:last format...")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            last_data = await redis.get(f'market:{symbol}:last')
            if last_data:
                try:
                    # Should be JSON now
                    if last_data.startswith('{'):
                        parsed = json.loads(last_data)
                        print(f"  ✓ {symbol}: price={parsed.get('price', 0)}, ts={parsed.get('ts', 0)}")
                    else:
                        print(f"  ⚠ {symbol}: Still stored as float: {last_data}")
                except json.JSONDecodeError:
                    print(f"  ✗ {symbol}: Failed to parse JSON")
            else:
                print(f"  - {symbol}: No data")
        
        # Test 2: Check OBI format
        print("\n2. Checking OBI format...")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            obi_data = await redis.get(f'metrics:{symbol}:obi')
            if obi_data:
                try:
                    if obi_data.startswith('{'):
                        parsed = json.loads(obi_data)
                        print(f"  ✓ {symbol}: JSON format with level1_imbalance={parsed.get('level1_imbalance', 0)}")
                    else:
                        obi_val = float(obi_data)
                        print(f"  ✓ {symbol}: Float format: {obi_val}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"  ✗ {symbol}: Parse error: {e}")
            else:
                print(f"  - {symbol}: No data")
        
        # Test 3: Check hidden orders format
        print("\n3. Checking hidden orders format...")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            hidden_data = await redis.get(f'metrics:{symbol}:hidden')
            if hidden_data:
                try:
                    if hidden_data.startswith('{'):
                        parsed = json.loads(hidden_data)
                        print(f"  ✓ {symbol}: JSON format with score={parsed.get('score', 0)}")
                    else:
                        hidden_val = float(hidden_data)
                        print(f"  ✓ {symbol}: Float format: {hidden_val}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"  ✗ {symbol}: Parse error: {e}")
            else:
                print(f"  - {symbol}: No data")
        
        # Test 4: Check if signals are being generated
        print("\n4. Checking signal generation...")
        heartbeat = await redis.get('health:signals:heartbeat')
        if heartbeat:
            print(f"  ✓ Signal heartbeat: {heartbeat}")
        else:
            print(f"  ✗ No signal heartbeat")
        
        considered = await redis.get('metrics:signals:considered') or 0
        emitted = await redis.get('metrics:signals:emitted') or 0
        skipped_stale = await redis.get('metrics:signals:skipped_stale') or 0
        
        print(f"  Considered: {considered}")
        print(f"  Emitted: {emitted}")
        print(f"  Skipped (stale): {skipped_stale}")
        
        # Test 5: Check for any latest signals
        print("\n5. Latest signals...")
        for symbol in ['SPY', 'QQQ', 'IWM']:
            signal_json = await redis.get(f'signals:latest:{symbol}')
            if signal_json:
                signal = json.loads(signal_json)
                print(f"  ✓ {symbol}: {signal.get('side')} signal with confidence {signal.get('confidence')}%")
            else:
                print(f"  - {symbol}: No signal")
        
        print("\n" + "=" * 50)
        print("Test complete!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await redis.aclose()

if __name__ == '__main__':
    asyncio.run(test_fixes())