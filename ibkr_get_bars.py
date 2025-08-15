#!/usr/bin/env python3
"""
Get 1-minute bars from IBKR and save to JSON
"""

from ib_insync import *
import json
from datetime import datetime

# Connect to IBKR (7497 for paper trading, 7496 for live)
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define contract (AAPL stock)
contract = Stock('AAPL', 'SMART', 'USD')

# Get 1-minute bars for last 2 days
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',  # Now
    durationStr='2 D',  # 2 days
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=False,  # Include pre/post market
    formatDate=1
)

# Convert to JSON-serializable format
bars_data = []
for bar in bars:
    bars_data.append({
        'timestamp': str(bar.date),
        'open': float(bar.open),
        'high': float(bar.high),
        'low': float(bar.low),
        'close': float(bar.close),
        'volume': int(bar.volume),
        'average': float(bar.average),
        'barCount': int(bar.barCount)
    })

# Save to JSON file
response = {
    'symbol': 'AAPL',
    'bar_size': '1 min',
    'duration': '2 D',
    'timestamp': datetime.now().isoformat(),
    'total_bars': len(bars_data),
    'bars': bars_data
}

with open('ibkr_1min_bars.json', 'w') as f:
    json.dump(response, f, indent=2)

print(f"✅ Saved {len(bars_data)} bars to ibkr_1min_bars.json")

# Disconnect
ib.disconnect()