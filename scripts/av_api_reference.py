#!/usr/bin/env python3
"""
Alpha Vantage API Reference - Quick reference for all 38 APIs
"""

# All 38 Alpha Vantage APIs with their parameters
API_REFERENCE = {
    "OPTIONS": {
        "REALTIME_OPTIONS": {
            "required": ["symbol"],
            "optional": ["require_greeks"],
            "example": {"symbol": "AAPL", "require_greeks": "true"}
        },
        "HISTORICAL_OPTIONS": {
            "required": ["symbol"],
            "optional": ["date"],
            "example": {"symbol": "AAPL", "date": "2025-08-20"}
        }
    },
    
    "TECHNICAL_INDICATORS": {
        "RSI": {
            "required": ["symbol", "interval", "time_period", "series_type"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 14, "series_type": "close"}
        },
        "MACD": {
            "required": ["symbol", "interval", "series_type"],
            "optional": ["fastperiod", "slowperiod", "signalperiod", "datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "series_type": "close", 
                       "fastperiod": 12, "slowperiod": 26, "signalperiod": 9}
        },
        "STOCH": {
            "required": ["symbol", "interval"],
            "optional": ["fastkperiod", "slowkperiod", "slowdperiod", "slowkmatype", "slowdmatype", "datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "fastkperiod": 5, 
                       "slowkperiod": 3, "slowdperiod": 3, "slowkmatype": 0, "slowdmatype": 0}
        },
        "WILLR": {
            "required": ["symbol", "interval", "time_period"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}
        },
        "MOM": {
            "required": ["symbol", "interval", "time_period", "series_type"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 10, "series_type": "close"}
        },
        "BBANDS": {
            "required": ["symbol", "interval", "time_period", "series_type"],
            "optional": ["nbdevup", "nbdevdn", "matype", "datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 20,"series_type": "close", "nbdevup": 2, "nbdevdn": 2, "matype": 0}
        },
        "ATR": {
            "required": ["symbol", "interval", "time_period"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}
        },
        "ADX": {
            "required": ["symbol", "interval", "time_period"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}
        },
        "AROON": {
            "required": ["symbol", "interval", "time_period"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}
        },
        "CCI": {
            "required": ["symbol", "interval", "time_period"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 20}
        },
        "EMA": {
            "required": ["symbol", "interval", "time_period", "series_type"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 20, "series_type": "close"}
        },
        "SMA": {
            "required": ["symbol", "interval", "time_period", "series_type"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 20, "series_type": "close"}
        },
        "MFI": {
            "required": ["symbol", "interval", "time_period"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily", "time_period": 14}
        },
        "OBV": {
            "required": ["symbol", "interval"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily"}
        },
        "AD": {
            "required": ["symbol", "interval"],
            "optional": ["datatype", "outputsize"],
            "example": {"symbol": "AAPL", "interval": "daily"}
        },
        "VWAP": {
            "required": ["symbol", "interval"],
            "optional": ["datatype"],
            "note": "Intraday intervals only (1min, 5min, 15min, 30min, 60min)",
            "example": {"symbol": "AAPL", "interval": "15min"}
        }
    },
    
    "ANALYTICS": {
        "ANALYTICS_FIXED_WINDOW": {
            "required": ["SYMBOLS", "INTERVAL"],
            "optional": ["RANGE", "OHLC", "CALCULATIONS"],
            "note": "Parameters are UPPERCASE",
            "example": {"SYMBOLS": "AAPL,QQQ", "INTERVAL": "DAILY", "OHLC": "close","RANGE": "1month", "CALCULATIONS": "MIN,MAX,MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,MAX_DRAWDOWN,HISTOGRAM,AUTOCORRELATION,COVARIANCE,CORRELATION"}
        },
        "ANALYTICS_SLIDING_WINDOW": {
            "required": ["SYMBOLS", "INTERVAL", "RANGE", "WINDOW_SIZE"],
            "optional": ["OHLC", "CALCULATIONS"],
            "note": "Parameters are UPPERCASE",
            "example": {"SYMBOLS": "AAPL,QQQ", "INTERVAL": "DAILY", "OHLC": "close","RANGE": "6month", "WINDOW_SIZE": 30, "CALCULATIONS": "MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,COVARIANCE,CORRELATION"}
        }
    },
    
    "SENTIMENT": {
        "NEWS_SENTIMENT": {
            "required": [],
            "optional": ["tickers", "topics", "sort", "limit"],
            "note": "Uses 'tickers' not 'symbol'",
            "example": {"tickers": "AAPL", "sort": "LATEST", "limit": 50}
        },
        "TOP_GAINERS_LOSERS": {
            "required": [],
            "optional": [],
            "example": {}
        },
        "INSIDER_TRANSACTIONS": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        }
    },
    
    "FUNDAMENTALS": {
        "OVERVIEW": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        },
        "EARNINGS": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        },
        "INCOME_STATEMENT": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        },
        "BALANCE_SHEET": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        },
        "CASH_FLOW": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        },
        "DIVIDENDS": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        },
        "SPLITS": {
            "required": ["symbol"],
            "optional": [],
            "example": {"symbol": "AAPL"}
        },
        "EARNINGS_CALENDAR": {
            "required": [],
            "optional": ["horizon"],
            "note": "⚠️ Returns CSV format - will be saved as raw CSV text",
            "example": {}
        },
        "IPO_CALENDAR": {
            "required": [],
            "optional": [],
            "note": "🚫 DESCOPED - Returns CSV by default",
            "descoped": True,
            "example": {}
        },
        "LISTING_STATUS": {
            "required": [],
            "optional": ["state", "date"],
            "note": "🚫 DESCOPED - Returns CSV by default",
            "descoped": True,
            "example": {}
        }
    },
    
    "ECONOMIC": {
        "TREASURY_YIELD": {
            "required": ["interval", "maturity"],
            "optional": ["datatype"],
            "example": {"interval": "monthly", "maturity": "10year"}
        },
        "FEDERAL_FUNDS_RATE": {
            "required": ["interval"],
            "optional": ["datatype"],
            "example": {"interval": "monthly"}
        },
        "CPI": {
            "required": [],
            "optional": ["interval", "datatype"],
            "example": {"interval": "monthly"}
        },
        "INFLATION": {
            "required": [],
            "optional": ["datatype"],
            "example": {}
        },
        "REAL_GDP": {
            "required": [],
            "optional": ["interval", "datatype"],
            "example": {"interval": "quarterly"}
        }
    }
}

def print_api_reference():
    """Print all APIs with their parameters"""
    print("\n" + "="*70)
    print("ALPHA VANTAGE API REFERENCE - 38 APIs")
    print("="*70)
    
    for category, apis in API_REFERENCE.items():
        print(f"\n📁 {category}")
        print("-" * 40)
        
        for api_name, details in apis.items():
            print(f"\n  🔹 {api_name}")
            print(f"     Required: {details.get('required', [])}")
            print(f"     Optional: {details.get('optional', [])}")
            if 'note' in details:
                print(f"     ⚠️  Note: {details['note']}")
            print(f"     Example: {details['example']}")

def get_test_commands():
    """Generate test commands for all APIs"""
    print("\n" + "="*70)
    print("TEST COMMANDS FOR ALL APIs (Excluding Descoped)")
    print("="*70)
    
    commands = []
    descoped = []
    for category, apis in API_REFERENCE.items():
        for api_name, details in apis.items():
            if details.get('descoped', False):
                descoped.append(api_name)
                continue
            example = details['example']
            params_json = json.dumps(example)
            cmd = f"python scripts/test_single_av_api.py --function {api_name} --params '{params_json}'"
            commands.append((api_name, cmd))
    
    for api_name, cmd in commands:
        print(f"\n# {api_name}")
        print(cmd)
    
    if descoped:
        print("\n" + "="*70)
        print("DESCOPED APIs (Not Testing):")
        print(", ".join(descoped))

if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--commands":
        get_test_commands()
    else:
        print_api_reference()
        print("\n💡 Tip: Run with --commands to get copy-paste commands for all APIs")