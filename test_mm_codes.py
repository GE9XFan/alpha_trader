#!/usr/bin/env python3
"""Test script to see actual market maker codes from IBKR Level 2 data"""

from ib_insync import IB, Stock
import time

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=999)  # Use different clientId to avoid conflicts

# Try multiple exchanges and symbols
test_cases = [
    ('SPY', 'ARCA'),
    ('SPY', 'ISLAND'),
    ('QQQ', 'ISLAND'),
    ('QQQ', 'ARCA'),
]

for symbol, exchange in test_cases:
    print(f"\n{'='*50}")
    print(f"Testing {symbol} on {exchange}")
    print(f"{'='*50}")
    
    try:
        contract = Stock(symbol, exchange, 'USD')
        ticker = ib.reqMktDepth(contract, numRows=10, isSmartDepth=False)
        
        # Give it time to populate
        time.sleep(3)
        
        # Check if we got any data
        if ticker.domBids or ticker.domAsks:
            print(f"\n{'Side':<4} {'Pos':<4} {'Price':<10} {'Size':<8} {'MarketMaker'}")
            print("-" * 50)
            
            for lvl in ticker.domBids:
                mm = getattr(lvl, 'marketMaker', 'N/A')
                print(f"BID  {lvl.position:<4} {lvl.price:<10.2f} {lvl.size:<8} {mm}")
            
            if ticker.domBids and ticker.domAsks:
                print()
            
            for lvl in ticker.domAsks:
                mm = getattr(lvl, 'marketMaker', 'N/A')
                print(f"ASK  {lvl.position:<4} {lvl.price:<10.2f} {lvl.size:<8} {mm}")
            
            # Show unique market makers
            all_mm = set()
            for lvl in ticker.domBids + ticker.domAsks:
                if hasattr(lvl, 'marketMaker'):
                    all_mm.add(lvl.marketMaker)
            
            if all_mm:
                print(f"\nUnique Market Makers found: {sorted(all_mm)}")
        else:
            print("No depth data received (market may be closed or no L2 access)")
            
        # Cancel the subscription
        ib.cancelMktDepth(ticker)
        
    except Exception as e:
        print(f"Error: {e}")

# Check what's in our data_ingestion stored books
print("\n" + "="*50)
print("Checking Redis for stored market maker codes...")
print("="*50)

import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

for symbol in ['SPY', 'QQQ', 'IWM']:
    for exchange in ['ARCA', 'ISLAND', 'NSDQ']:
        key = f"market:{symbol}:{exchange}:book"
        data = r.get(key)
        if data:
            book = json.loads(data)
            mm_codes = set()
            for side in ['bids', 'asks']:
                for level in book.get(side, []):
                    mm = level.get('mm')
                    if mm and mm != 'UNKNOWN':
                        mm_codes.add(mm)
            
            if mm_codes:
                print(f"{symbol}:{exchange} - Market Makers: {sorted(mm_codes)}")

# Also check aggregated books
for symbol in ['SPY', 'QQQ', 'IWM']:
    key = f"market:{symbol}:book"
    data = r.get(key)
    if data:
        book = json.loads(data)
        mm_codes = set()
        for side in ['bids', 'asks']:
            for level in book.get(side, []):
                mm = level.get('mm')
                if mm and mm != 'UNKNOWN':
                    mm_codes.add(mm)
        
        if mm_codes:
            print(f"{symbol} (aggregated) - Market Makers: {sorted(mm_codes)}")

ib.disconnect()
print("\nDisconnected.")