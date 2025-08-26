#!/usr/bin/env python3
"""Check Alpha Vantage rate limit"""
import sys
sys.path.append('.')

from src.data.alpha_vantage_client import av_client

def check_av_rate_limit():
    """Check AV rate limit status"""
    print(f"Alpha Vantage API Status:")
    print(f"  Calls remaining: {av_client.rate_limiter.remaining}/600")
    print(f"  Reset in: {av_client.rate_limiter.reset_time} seconds")
    print(f"  Cache stats:")
    hit_rate = av_client.cache_hits / max(av_client.total_calls, 1)
    print(f"    Hit rate: {hit_rate:.1%}")
    print(f"    Cached items: {len(av_client.cache)}")

if __name__ == "__main__":
    check_av_rate_limit()
