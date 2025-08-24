"""
Signal Generator - Implementation Plan Week 3 Day 1-2
Generates trading signals using Alpha Vantage data
"""
from datetime import datetime, time
from typing import Optional, Dict, List
import asyncio

from src.core.config import config
from src.core.logger import get_logger
from src.analytics.ml_model import ml_model
from src.analytics.features import feature_engine
from src.data.market_data import market_data
from src.data.options_data import options_data


logger = get_logger(__name__)


class SignalGenerator:
    """
    Generates trading signals using Alpha Vantage data
    Reused by paper and live trading
    Implementation Plan Week 3 Day 1-2
    """
    
    def __init__(self):
        self.ml = ml_model
        self.features = feature_engine
        self.market = market_data  # IBKR for spot prices
        self.options = options_data  # Alpha Vantage for options
        
        self.signals_today = []
        self.last_signal_time = {}
        self.min_time_between_signals = 300  # 5 minutes
    
    async def generate_signals(self, symbols: List[str]) -> List[Dict]:
        """
        Generate signals for symbols using Alpha Vantage data
        Called by both paper and live traders
        """
        signals = []
        
        for symbol in symbols:
            try:
                # Check if enough time has passed
                if symbol in self.last_signal_time:
                    time_since = (datetime.now() - self.last_signal_time[symbol]).seconds
                    if time_since < self.min_time_between_signals:
                        continue
                
                # Calculate features using Alpha Vantage APIs
                features = await self.features.calculate_features(symbol)
                
                # Get ML prediction
                signal_type, confidence = self.ml.predict(features)
                
                if signal_type != 'HOLD':
                    # Find best option to trade using Alpha Vantage data
                    option = await self._select_option_from_av(symbol, signal_type)
                    
                    if option:
                        signal = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'signal_type': signal_type,
                            'confidence': confidence,
                            'option': option,
                            'features': features.tolist(),
                            'av_greeks': option['greeks'],  # Greeks from Alpha Vantage
                        }
                        
                        signals.append(signal)
                        self.signals_today.append(signal)
                        self.last_signal_time[symbol] = datetime.now()
                        
                        logger.info(f"Signal generated: {symbol} {signal_type} (conf: {confidence:.2f})")
                        
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    async def _select_option_from_av(self, symbol: str, signal_type: str) -> Optional[Dict]:
        """Select best option contract using Alpha Vantage data"""
        # Get ATM options with Greeks from Alpha Vantage
        atm_options = self.options.find_atm_options(symbol, dte_min=0, dte_max=7)
        
        if not atm_options:
            logger.warning(f"No ATM options found for {symbol}")
            return None
        
        # Filter by option type
        if 'CALL' in signal_type:
            candidates = [opt for opt in atm_options if opt['type'] == 'CALL']
        else:
            candidates = [opt for opt in atm_options if opt['type'] == 'PUT']
        
        if not candidates:
            return None
        
        # Select based on Greeks from Alpha Vantage
        best_option = max(
            candidates,
            key=lambda x: abs(x['greeks']['delta']) / abs(x['greeks']['theta']) 
            if x['greeks']['theta'] != 0 else 0
        )
        
        return {
            'strike': best_option['strike'],
            'expiry': best_option['expiry'],
            'type': best_option['type'],
            'contract': best_option['contract'],
            'greeks': best_option['greeks'],  # Include AV Greeks
            'iv': best_option.get('iv', 0.20)
        }


# SIGNAL GENERATOR USING ALPHA VANTAGE DATA
signal_generator = SignalGenerator()
