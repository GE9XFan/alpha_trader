#!/usr/bin/env python3
"""
Day 6 Verification Script - Test Signal Generation System
Quick verification that signals are being generated properly
"""

import asyncio
import json
import redis.asyncio as aioredis
import time
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def verify_signals():
    """Verify signal generation is working."""
    
    # Connect to Redis
    redis = aioredis.Redis(
        host='127.0.0.1',
        port=6379,
        decode_responses=True
    )
    
    print("Day 6 Signal Generation Verification")
    print("=" * 50)
    
    try:
        # Check if system is running
        heartbeat = await redis.get('health:signals:heartbeat')
        if heartbeat:
            print(f"✓ Signal generator heartbeat: {heartbeat}")
        else:
            print("✗ Signal generator not running")
        
        # Check signal metrics
        considered = await redis.get('metrics:signals:considered') or 0
        emitted = await redis.get('metrics:signals:emitted') or 0
        skipped_stale = await redis.get('metrics:signals:skipped_stale') or 0
        cooldown_blocked = await redis.get('metrics:signals:cooldown_blocked') or 0
        duplicates = await redis.get('metrics:signals:duplicates') or 0
        
        print(f"\nSignal Metrics:")
        print(f"  Considered: {considered}")
        print(f"  Emitted: {emitted}")
        print(f"  Skipped (stale): {skipped_stale}")
        print(f"  Cooldown blocked: {cooldown_blocked}")
        print(f"  Duplicates: {duplicates}")
        
        # Check for latest signals
        print(f"\nLatest Signals:")
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA']
        
        for symbol in symbols:
            signal_json = await redis.get(f'signals:latest:{symbol}')
            if signal_json:
                signal = json.loads(signal_json)
                print(f"\n  {symbol}:")
                print(f"    Strategy: {signal.get('strategy')}")
                print(f"    Side: {signal.get('side')}")
                print(f"    Confidence: {signal.get('confidence')}%")
                print(f"    Entry: ${signal.get('entry')}")
                print(f"    Stop: ${signal.get('stop')}")
                print(f"    Targets: {signal.get('targets')}")
                print(f"    Reasons: {', '.join(signal.get('reasons', []))}")
                
                # Check age
                ts = signal.get('ts', 0)
                age_s = (time.time() * 1000 - ts) / 1000
                print(f"    Age: {age_s:.1f}s")
        
        # Check distribution queues
        print(f"\nDistribution Queues:")
        premium_len = await redis.llen('distribution:premium:queue')
        basic_len = await redis.llen('distribution:basic:queue')
        free_len = await redis.llen('distribution:free:queue')
        
        print(f"  Premium queue: {premium_len} signals")
        print(f"  Basic queue: {basic_len} signals")
        print(f"  Free queue: {free_len} signals")
        
        # Check feature data freshness
        print(f"\nFeature Data Status:")
        for symbol in ['SPY', 'QQQ']:
            # Check VPIN
            vpin = await redis.get(f'metrics:{symbol}:vpin')
            obi = await redis.get(f'metrics:{symbol}:obi')
            gex = await redis.get(f'metrics:{symbol}:gex')
            
            print(f"\n  {symbol}:")
            if vpin:
                print(f"    VPIN: {float(vpin):.3f}")
            if obi:
                # OBI might be JSON or float
                try:
                    if obi.startswith('{'):
                        obi_data = json.loads(obi)
                        obi_val = obi_data.get('level1_imbalance', 0)
                        # Normalize to 0-1 range
                        obi_normalized = (obi_val + 1.0) / 2.0
                        print(f"    OBI: {obi_normalized:.3f} (raw: {obi_val:.3f})")
                    else:
                        print(f"    OBI: {float(obi):.3f}")
                except (json.JSONDecodeError, ValueError, AttributeError):
                    print(f"    OBI: {float(obi):.3f}")
            if gex:
                print(f"    GEX: ${float(gex)/1e9:.2f}B")
            
            # Check sweep detection
            sweep = await redis.get(f'options:{symbol}:sweep')
            if sweep:
                print(f"    Sweep: {'YES' if float(sweep) > 0 else 'NO'}")
            
            # Check unusual options
            unusual = await redis.get(f'options:{symbol}:unusual_activity')
            if unusual:
                print(f"    Unusual Options: {float(unusual):.2f}")
        
        # Check MOC imbalance data
        print(f"\nMOC Imbalance Data:")
        for symbol in ['SPY', 'QQQ']:
            imb_json = await redis.get(f'imbalance:{symbol}:raw')
            if imb_json:
                imb = json.loads(imb_json)
                print(f"  {symbol}: {imb.get('side')} ${imb.get('total', 0)/1e9:.2f}B (ratio: {imb.get('ratio', 0):.2f})")
        
        print("\n" + "=" * 50)
        print("Signal verification complete!")
        
        # Final status
        if int(emitted) > 0:
            print("✓ Signals are being generated successfully")
        elif int(considered) > 0:
            print("⚠ Signals are being considered but not emitted (check thresholds)")
        else:
            print("✗ No signals being generated (check data feeds)")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await redis.aclose()

if __name__ == '__main__':
    asyncio.run(verify_signals())