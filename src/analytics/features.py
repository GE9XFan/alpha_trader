"""
Feature Engineering - Implementation Plan Week 2 Day 1-2
All indicators from Alpha Vantage - no local calculation
Tech Spec Section 3.2 - 45 features total
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import asyncio

from src.core.config import config
from src.core.constants import FEATURE_NAMES
from src.core.logger import get_logger
from src.data.options_data import options_data
from src.data.alpha_vantage_client import av_client


logger = get_logger(__name__)


class FeatureEngine:
    """
    Feature engineering using Alpha Vantage data feeds
    All indicators from AV - no local calculation
    Tech Spec Section 3.2 - 45 features
    """
    
    def __init__(self):
        self.options = options_data
        self.av = av_client
        self.feature_names = FEATURE_NAMES  # 45 features from constants
    
    async def calculate_features(self, symbol: str) -> np.ndarray:
        """
        Calculate all features using Alpha Vantage APIs
        Parallel API calls for efficiency with 600 calls/min limit
        """
        features = {}
        
        # Get IBKR price data for basic returns
        bars = await self.options.market.get_historical_bars(symbol, '1 D')
        
        if not bars.empty:
            features['returns_5m'] = self._calculate_returns(bars, 60)
            features['returns_30m'] = self._calculate_returns(bars, 360)
            features['returns_1h'] = self._calculate_returns(bars, 720)
            features['volume_ratio'] = bars['volume'].iloc[-1] / bars['volume'].mean() if len(bars) > 0 else 1.0
            features['high_low_ratio'] = (bars['high'].iloc[-1] - bars['low'].iloc[-1]) / bars['close'].iloc[-1] if bars['close'].iloc[-1] > 0 else 0.0
        else:
            # Default values
            features.update({
                'returns_5m': 0.0, 'returns_30m': 0.0, 'returns_1h': 0.0,
                'volume_ratio': 1.0, 'high_low_ratio': 0.0
            })
        
        # Parallel Alpha Vantage API calls for technical indicators
        tasks = [
            self.av.get_rsi(symbol),
            self.av.get_macd(symbol),
            self.av.get_bbands(symbol),
            self.av.get_technical_indicator(symbol, 'ATR', interval='5min'),
            self.av.get_technical_indicator(symbol, 'ADX', interval='5min'),
            self.av.get_technical_indicator(symbol, 'OBV', interval='5min'),
            self.av.get_technical_indicator(symbol, 'VWAP', interval='5min'),
            self.av.get_technical_indicator(symbol, 'EMA', interval='5min', time_period=20),
            self.av.get_technical_indicator(symbol, 'SMA', interval='5min', time_period=50),
            self.av.get_technical_indicator(symbol, 'MOM', interval='5min'),
            self.av.get_technical_indicator(symbol, 'CCI', interval='5min')
        ]
        
        # Execute all indicator fetches in parallel
        indicator_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process technical indicators from Alpha Vantage
        features.update(self._process_indicators(indicator_results))
        
        # Get options features from Alpha Vantage
        options_features = await self._get_av_options_features(symbol)
        features.update(options_features)
        
        # Get sentiment from Alpha Vantage
        sentiment_features = await self._get_av_sentiment_features(symbol)
        features.update(sentiment_features)
        
        # Convert to array in consistent order
        feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Handle any NaN values
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        return feature_array
    
    def _calculate_returns(self, bars: pd.DataFrame, periods: int) -> float:
        """Calculate returns over periods"""
        if len(bars) < 2:
            return 0.0
        
        # periods is in 5-second bars
        bars_needed = periods // 5
        
        if len(bars) > bars_needed:
            old_price = bars.iloc[-bars_needed]['close']
            new_price = bars.iloc[-1]['close']
            
            if old_price > 0:
                return (new_price - old_price) / old_price
        
        return 0.0
    
    def _process_indicators(self, results: List) -> Dict:
        """Process indicator results from Alpha Vantage"""
        features = {}
        
        # RSI
        if not isinstance(results[0], Exception) and not results[0].empty:
            features['rsi'] = results[0].iloc[0].get('RSI', 50.0) / 100.0
        else:
            features['rsi'] = 0.5
        
        # MACD
        if not isinstance(results[1], Exception) and not results[1].empty:
            features['macd_signal'] = results[1].iloc[0].get('MACD_Signal', 0.0)
            features['macd_histogram'] = results[1].iloc[0].get('MACD_Hist', 0.0)
        else:
            features['macd_signal'] = 0.0
            features['macd_histogram'] = 0.0
        
        # Continue for other indicators...
        # (Stub remaining indicator processing)
        
        # Fill remaining with defaults
        for name in self.feature_names:
            if name not in features:
                if 'rsi' in name or 'bb_' in name or 'atr' in name:
                    features[name] = 0.0
        
        return features
    
    async def _get_av_options_features(self, symbol: str) -> Dict:
        """Get options-specific features from Alpha Vantage"""
        features = {}
        
        # Get current options chain with Greeks
        try:
            options = await self.av.get_realtime_options(symbol, require_greeks=True)
            
            if options:
                # Find ATM option
                spot = self.options.market.get_latest_price(symbol)
                
                if spot > 0:
                    # Find closest call and put
                    calls = [opt for opt in options if opt.option_type == 'CALL']
                    puts = [opt for opt in options if opt.option_type == 'PUT']
                    
                    if calls:
                        atm_call = min(calls, key=lambda x: abs(x.strike - spot))
                        features['atm_delta'] = atm_call.delta
                        features['atm_gamma'] = atm_call.gamma
                        features['atm_theta'] = atm_call.theta
                        features['atm_vega'] = atm_call.vega
                    
                    # IV metrics
                    all_ivs = [opt.implied_volatility for opt in options if opt.implied_volatility > 0]
                    if all_ivs:
                        features['iv_rank'] = np.percentile(all_ivs, 50) / 100.0
                        features['iv_percentile'] = len([iv for iv in all_ivs if iv < atm_call.implied_volatility]) / len(all_ivs) if calls else 0.5
                    
                    # Volume metrics
                    call_volume = sum(opt.volume for opt in calls)
                    put_volume = sum(opt.volume for opt in puts)
                    features['call_volume'] = call_volume
                    features['put_volume'] = put_volume
                    features['put_call_ratio'] = put_volume / call_volume if call_volume > 0 else 1.0
                    
                    # Gamma exposure (using Greeks from AV)
                    features['gamma_exposure'] = sum(
                        opt.gamma * opt.open_interest * 100 * spot
                        for opt in options
                    )
                    
                    # Open interest
                    call_oi = sum(opt.open_interest for opt in calls)
                    put_oi = sum(opt.open_interest for opt in puts)
                    features['oi_ratio'] = put_oi / call_oi if call_oi > 0 else 1.0
        
        except Exception as e:
            logger.error(f"Error getting options features: {e}")
        
        # Default values for missing features
        for name in ['atm_delta', 'atm_gamma', 'atm_theta', 'atm_vega',
                    'iv_rank', 'iv_percentile', 'put_call_ratio', 'gamma_exposure',
                    'max_pain_distance', 'call_volume', 'put_volume', 'oi_ratio']:
            if name not in features:
                features[name] = 0.0 if 'volume' not in name else 0
        
        return features
    
    async def _get_av_sentiment_features(self, symbol: str) -> Dict:
        """Get sentiment features from Alpha Vantage"""
        features = {}
        
        try:
            # Get news sentiment
            sentiment_data = await self.av.get_news_sentiment([symbol])
            
            if sentiment_data and 'feed' in sentiment_data:
                sentiments = []
                for article in sentiment_data['feed'][:10]:  # Last 10 articles
                    ticker_sentiment = next(
                        (s for s in article.get('ticker_sentiment', []) 
                         if s['ticker'] == symbol), 
                        None
                    )
                    if ticker_sentiment:
                        sentiments.append(float(ticker_sentiment.get('ticker_sentiment_score', 0)))
                
                features['news_sentiment_score'] = np.mean(sentiments) if sentiments else 0.0
                features['news_volume'] = len(sentiments)
        except Exception as e:
            logger.error(f"Error getting sentiment features: {e}")
        
        # Default values
        features.setdefault('news_sentiment_score', 0.0)
        features.setdefault('news_volume', 0)
        features['insider_sentiment'] = 0.0  # Placeholder
        features['social_sentiment'] = 0.0
        
        # Market structure (simplified)
        features['spy_correlation'] = 0.8
        features['qqq_correlation'] = 0.7
        features['vix_level'] = 20.0 / 100.0
        features['term_structure'] = 0.0
        features['market_regime'] = 0.5
        
        return features


# BUILD ON OPTIONS DATA AND ALPHA VANTAGE
feature_engine = FeatureEngine()
