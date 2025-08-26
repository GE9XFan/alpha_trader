"""
System constants - Tech Spec v3.0
"""

# Alpha Vantage API endpoints (38 total)
AV_ENDPOINTS = {
    # OPTIONS (2)
    'REALTIME_OPTIONS': 'REALTIME_OPTIONS',
    'HISTORICAL_OPTIONS': 'HISTORICAL_OPTIONS',
    
    # TECHNICAL INDICATORS (16)
    'RSI': 'RSI',
    'MACD': 'MACD',
    'STOCH': 'STOCH',
    'WILLR': 'WILLR',
    'MOM': 'MOM',
    'BBANDS': 'BBANDS',
    'ATR': 'ATR',
    'ADX': 'ADX',
    'AROON': 'AROON',
    'CCI': 'CCI',
    'EMA': 'EMA',
    'SMA': 'SMA',
    'MFI': 'MFI',
    'OBV': 'OBV',
    'AD': 'AD',
    'VWAP': 'VWAP',
    
    # ANALYTICS (2)
    'ANALYTICS_FIXED_WINDOW': 'ANALYTICS_FIXED_WINDOW',
    'ANALYTICS_SLIDING_WINDOW': 'ANALYTICS_SLIDING_WINDOW',
    
    # SENTIMENT (3)
    'NEWS_SENTIMENT': 'NEWS_SENTIMENT',
    'TOP_GAINERS_LOSERS': 'TOP_GAINERS_LOSERS',
    'INSIDER_TRANSACTIONS': 'INSIDER_TRANSACTIONS',
    
    # FUNDAMENTALS (7)
    'OVERVIEW': 'OVERVIEW',
    'EARNINGS': 'EARNINGS',
    'INCOME_STATEMENT': 'INCOME_STATEMENT',
    'BALANCE_SHEET': 'BALANCE_SHEET',
    'CASH_FLOW': 'CASH_FLOW',
    'DIVIDENDS': 'DIVIDENDS',
    'SPLITS': 'SPLITS',
    
    # ECONOMIC (5)
    'TREASURY_YIELD': 'TREASURY_YIELD',
    'FEDERAL_FUNDS_RATE': 'FEDERAL_FUNDS_RATE',
    'CPI': 'CPI',
    'INFLATION': 'INFLATION',
    'REAL_GDP': 'REAL_GDP'
}

# Feature names - Tech Spec Section 3.2
FEATURE_NAMES = [
    # Price action (from IBKR bars)
    'returns_5m', 'returns_30m', 'returns_1h',
    'volume_ratio', 'high_low_ratio',
    
    # Technical indicators (from Alpha Vantage)
    'rsi', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_position',
    'atr', 'adx', 'obv_slope', 'vwap_distance',
    'ema_20', 'sma_50', 'momentum', 'cci',
    
    # Options metrics (from Alpha Vantage)
    'iv_rank', 'iv_percentile', 'put_call_ratio',
    'atm_delta', 'atm_gamma', 'atm_theta', 'atm_vega',
    'gamma_exposure', 'max_pain_distance',
    'call_volume', 'put_volume', 'oi_ratio',
    
    # Sentiment (from Alpha Vantage)
    'news_sentiment_score', 'news_volume',
    'insider_sentiment', 'social_sentiment',
    
    # Market structure (from Alpha Vantage)
    'spy_correlation', 'qqq_correlation',
    'vix_level', 'term_structure', 'market_regime'
]

# Trading signals
SIGNALS = ['BUY_CALL', 'BUY_PUT', 'HOLD', 'CLOSE']

# Market hours (ET)
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)
PRE_MARKET_OPEN = (4, 0)
AFTER_MARKET_CLOSE = (20, 0)

# Performance targets - Tech Spec Section 1.5
PERFORMANCE_TARGETS = {
    'critical_path_latency_ms': 150,
    'greeks_retrieval_cached_ms': 5,
    'greeks_fetch_api_ms': 300,
    'av_api_efficiency_calls_per_trade': 5,
    'ml_inference_ms': 15,
    'position_limit': 20,
    'daily_trades': 50,
    'discord_latency_seconds': 5
}
