#!/usr/bin/env python3
"""
Monitor Market Data in Redis
Real-time display of market data flowing through Redis
"""

import redis
import json
import time
import sys
import argparse
from datetime import datetime


def monitor_symbol(symbol: str, redis_host='localhost', redis_port=6379):
    """Monitor market data for a specific symbol"""
    
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    print(f"Monitoring {symbol} - Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            # Clear screen
            print("\033[H\033[J", end="")
            
            # Header
            print(f"AlphaTrader Pro - Market Data Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print(f"Symbol: {symbol}")
            print("=" * 60)
            
            # Last price
            last = r.get(f'market:{symbol}:last')
            if last:
                print(f"Last Price: ${float(last):.2f}")
            
            # NBBO
            nbbo = r.get(f'market:{symbol}:nbbo')
            if nbbo:
                nbbo_data = json.loads(nbbo)
                bid = nbbo_data.get('bid', {})
                ask = nbbo_data.get('ask', {})
                print(f"\nNBBO:")
                print(f"  Bid: ${bid.get('price', 0):.2f} x {bid.get('size', 0)} [{bid.get('exchange', 'N/A')}]")
                print(f"  Ask: ${ask.get('price', 0):.2f} x {ask.get('size', 0)} [{ask.get('exchange', 'N/A')}]")
                print(f"  Spread: ${nbbo_data.get('spread', 0):.2f} ({nbbo_data.get('spread_bps', 0):.1f} bps)")
            
            # LULD Bands
            luld = r.get(f'market:{symbol}:luld:bands')
            if luld:
                luld_data = json.loads(luld)
                print(f"\nLULD Bands:")
                print(f"  Upper: ${luld_data.get('upper', 0):.2f}")
                print(f"  Lower: ${luld_data.get('lower', 0):.2f}")
                print(f"  Width: {luld_data.get('band_width_pct', 0):.1f}%")
            
            # Auction Imbalance
            auction = r.get(f'market:{symbol}:auction:closing')
            if auction:
                auction_data = json.loads(auction)
                if auction_data.get('imbalance_qty'):
                    print(f"\nClosing Auction:")
                    print(f"  Imbalance: {auction_data['imbalance_qty']:,} shares ({auction_data.get('imbalance_side', 'N/A')})") 
                    print(f"  Indicative: ${auction_data.get('indicative_price', 0):.2f}")
            
            # Microstructure Metrics
            micro = r.get(f'market:{symbol}:micro:metrics')
            if micro:
                micro_data = json.loads(micro)
                print(f"\nMicrostructure:")
                print(f"  Spread: ${micro_data.get('spread', {}).get('current', 0):.4f}")
                print(f"  Depth: {micro_data.get('depth', {}).get('current', 0):.0f}")
                print(f"  Toxicity: {micro_data.get('toxicity', {}).get('vpin_adjusted', 0):.3f}")
            
            # Latency
            latency = r.get(f'market:{symbol}:latency:total')
            if latency:
                print(f"\nLatency: {float(latency):.1f} ms")
            
            # Halt Status
            halted = r.get(f'market:{symbol}:halted')
            if halted == 'true':
                halt_state = r.get(f'market:{symbol}:halt:state')
                if halt_state:
                    halt_data = json.loads(halt_state)
                    print(f"\n⚠️  HALTED: {halt_data.get('halt_description', 'Unknown')}")
            
            # Hidden Orders
            hidden = r.get(f'market:{symbol}:hidden:detection')
            if hidden:
                hidden_data = json.loads(hidden)
                if hidden_data.get('iceberg_candidates'):
                    print(f"\nHidden Orders Detected: {hidden_data['iceberg_candidates']} locations")
            
            # MM Hedging
            hedging_keys = r.keys(f'market:{symbol}:mm:hedging:*')
            if hedging_keys:
                print(f"\nOptions MM Hedging Detected:")
                for key in hedging_keys[:3]:  # Show top 3
                    mm_name = key.split(':')[-1]
                    print(f"  - {mm_name}")
            
            time.sleep(0.5)  # Update every 500ms
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def main():
    parser = argparse.ArgumentParser(description='Monitor market data in Redis')
    parser.add_argument('symbol', nargs='?', default='SPY', help='Symbol to monitor (default: SPY)')
    parser.add_argument('--host', default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    monitor_symbol(args.symbol.upper(), args.host, args.port)


if __name__ == "__main__":
    main()