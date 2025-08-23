#!/usr/bin/env python3
"""
Simple API tester - test multiple APIs at once
"""
import json
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Load .env
load_dotenv()

# API Reference
API_REFERENCE = {
    "OPTIONS": {
        "REALTIME_OPTIONS": {"example": {"symbol": "AAPL", "require_greeks": "true"}},
        "HISTORICAL_OPTIONS": {"example": {"symbol": "AAPL", "date": "2025-08-20"}}
    },
    "TECHNICAL_INDICATORS": {
        "RSI": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 14, "series_type": "close"}},
        "MACD": {"example": {"symbol": "AAPL", "interval": "daily", "series_type": "close", "fastperiod": 12, "slowperiod": 26, "signalperiod": 9}},
        "STOCH": {"example": {"symbol": "AAPL", "interval": "daily", "fastkperiod": 5, "slowkperiod": 3, "slowdperiod": 3, "slowkmatype": 0, "slowdmatype": 0}},
        "WILLR": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}},
        "MOM": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 10, "series_type": "close"}},
        "BBANDS": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 20, "series_type": "close", "nbdevup": 2, "nbdevdn": 2, "matype": 0}},
        "ATR": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}},
        "ADX": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}},
        "AROON": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}},
        "CCI": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 20}},
        "EMA": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 20, "series_type": "close"}},
        "SMA": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 20, "series_type": "close"}},
        "MFI": {"example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}},
        "OBV": {"example": {"symbol": "AAPL", "interval": "daily"}},
        "AD": {"example": {"symbol": "AAPL", "interval": "daily"}},
        "VWAP": {"example": {"symbol": "AAPL", "interval": "15min"}}
    },
    "ANALYTICS": {
        "ANALYTICS_FIXED_WINDOW": {"example": {"SYMBOLS": "AAPL,QQQ", "INTERVAL": "DAILY", "OHLC": "close", "RANGE": "1month", "CALCULATIONS": "MIN,MAX,MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,MAX_DRAWDOWN,HISTOGRAM,AUTOCORRELATION,COVARIANCE,CORRELATION"}},
        "ANALYTICS_SLIDING_WINDOW": {"example": {"SYMBOLS": "AAPL,QQQ", "INTERVAL": "DAILY", "OHLC": "close", "RANGE": "6month", "WINDOW_SIZE": 90, "CALCULATIONS": "MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,COVARIANCE,CORRELATION"}}
    },
    "SENTIMENT": {
        "NEWS_SENTIMENT": {"example": {"tickers": "AAPL", "sort": "LATEST", "limit": 50}},
        "TOP_GAINERS_LOSERS": {"example": {}},
        "INSIDER_TRANSACTIONS": {"example": {"symbol": "AAPL"}}
    },
    "FUNDAMENTALS": {
        "OVERVIEW": {"example": {"symbol": "AAPL"}},
        "EARNINGS": {"example": {"symbol": "AAPL"}},
        "INCOME_STATEMENT": {"example": {"symbol": "AAPL"}},
        "BALANCE_SHEET": {"example": {"symbol": "AAPL"}},
        "CASH_FLOW": {"example": {"symbol": "AAPL"}},
        "DIVIDENDS": {"example": {"symbol": "AAPL"}},
        "SPLITS": {"example": {"symbol": "AAPL"}},
        "EARNINGS_CALENDAR": {"example": {}}
    },
    "ECONOMIC": {
        "TREASURY_YIELD": {"example": {"interval": "monthly", "maturity": "10year"}},
        "FEDERAL_FUNDS_RATE": {"example": {"interval": "monthly"}},
        "CPI": {"example": {"interval": "monthly"}},
        "INFLATION": {"example": {}},
        "REAL_GDP": {"example": {"interval": "quarterly"}}
    }
}

def test_api(api_name, category, params, base_url):
    """Test a single API and save response"""
    print(f"\n📍 Testing {api_name}...")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        
        # Save response
        save_dir = Path(f"data/api_responses/{category.lower()}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save based on content
        if 'json' in response.headers.get('Content-Type', '') or response.text.startswith('{'):
            file_path = save_dir / f"{api_name.lower()}_response.json"
            with open(file_path, 'w') as f:
                json.dump(response.json(), f, indent=2)
        else:
            file_path = save_dir / f"{api_name.lower()}_response.csv"
            with open(file_path, 'w') as f:
                f.write(response.text)
        
        print(f"   ✅ Saved to {file_path}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

# Get API key
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
print(f"✅ Using API key from environment")

base_url = "https://www.alphavantage.co/query"

while True:
    print("\n" + "="*50)
    start = input("Start test? (y/n): ").lower()
    if start != 'y':
        break
    
    # Show categories
    print("\nCategories:")
    categories = list(API_REFERENCE.keys())
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    
    # Pick category
    cat_num = int(input("Select category number: ")) - 1
    category = categories[cat_num]
    
    # Show APIs
    print(f"\nAPIs in {category}:")
    apis = list(API_REFERENCE[category].keys())
    for i, api in enumerate(apis, 1):
        print(f"  {i}. {api}")
    
    print("\nOptions:")
    print("  a. Test ALL APIs in this category")
    print("  m. Select multiple APIs")
    print("  s. Select single API")
    
    choice = input("\nYour choice (a/m/s): ").lower()
    
    selected_apis = []
    
    if choice == 'a':
        # Test all
        selected_apis = apis
        print(f"\n🚀 Will test ALL {len(apis)} APIs in {category}")
        
    elif choice == 'm':
        # Multiple selection
        print("\nEnter API numbers separated by commas (e.g., 1,3,5):")
        numbers = input("Numbers: ").split(',')
        for num in numbers:
            try:
                idx = int(num.strip()) - 1
                if 0 <= idx < len(apis):
                    selected_apis.append(apis[idx])
            except:
                pass
        print(f"\n🚀 Will test {len(selected_apis)} APIs: {', '.join(selected_apis)}")
        
    else:
        # Single selection
        api_num = int(input("Select API number: ")) - 1
        selected_apis = [apis[api_num]]
    
    # Test selected APIs
    success_count = 0
    fail_count = 0
    
    for api_name in selected_apis:
        # Get params and add function/apikey
        params = API_REFERENCE[category][api_name]['example'].copy()
        params['function'] = api_name
        params['apikey'] = api_key
        
        # Convert numbers to strings
        for key, val in params.items():
            if isinstance(val, (int, float)) and key != 'WINDOW_SIZE':
                params[key] = str(val)
        
        # Test this API
        if test_api(api_name, category, params, base_url):
            success_count += 1
        else:
            fail_count += 1
        
        # Small delay between calls to avoid rate limit
        if len(selected_apis) > 1:
            time.sleep(0.5)
    
    # Summary
    print(f"\n" + "="*50)
    print(f"SUMMARY: ✅ {success_count} succeeded, ❌ {fail_count} failed")