#!/usr/bin/env python3
"""
Quick test to verify signal generation is working after the fix.
"""
import redis
import json
import time
from datetime import datetime

def main():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    print("Signal Generation Test")
    print("=" * 50)
    
    # Check heartbeat
    heartbeat = r.get('health:signals:heartbeat')
    if heartbeat:
        print(f"✓ Signal generator heartbeat: {heartbeat}")
    else:
        print("✗ No heartbeat found - signal generator may not be running")
    
    # Check metrics
    metrics = {
        'considered': r.get('metrics:signals:considered'),
        'emitted': r.get('metrics:signals:emitted'),
        'skipped_stale': r.get('metrics:signals:skipped_stale'),
        'cooldown_blocked': r.get('metrics:signals:cooldown_blocked')
    }
    
    print("\nSignal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value or 0}")
    
    # Check for fresh signals
    print("\nChecking for recent signals...")
    symbols = ['SPY', 'QQQ', 'IWM']
    found_signals = False
    
    for symbol in symbols:
        latest = r.get(f'signals:latest:{symbol}')
        if latest:
            signal = json.loads(latest)
            ts = datetime.fromtimestamp(signal['ts']/1000)
            age_min = (time.time() - signal['ts']/1000) / 60
            
            if age_min < 5:  # Signal from last 5 minutes
                print(f"\n✓ Fresh signal for {symbol}:")
                print(f"  Strategy: {signal['strategy']}")
                print(f"  Side: {signal['side']}")
                print(f"  Confidence: {signal['confidence']}%")
                print(f"  Age: {age_min:.1f} minutes")
                if 'contract' in signal:
                    c = signal['contract']
                    print(f"  Contract: {c.get('strike', '')} {c.get('right', '')} {c.get('expiry', '')}")
                found_signals = True
    
    if not found_signals:
        print("\n⚠️  No fresh signals found (last 5 minutes)")
        print("    Checking data freshness...")
        
        # Check if data is fresh
        for symbol in symbols:
            ticker = r.get(f'market:{symbol}:ticker')
            if ticker:
                data = json.loads(ticker)
                age = (time.time() * 1000 - data['timestamp']) / 1000
                print(f"    {symbol} ticker age: {age:.1f}s")
    
    # Check queue lengths
    print("\nSignal Queues:")
    for tier in ['premium', 'basic', 'free']:
        length = r.llen(f'distribution:{tier}:queue')
        print(f"  {tier}: {length} signals")
    
    print("\n" + "=" * 50)
    print("Test complete. Restart main.py to apply the fix.")

if __name__ == "__main__":
    main()