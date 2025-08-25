"""
Options Data Manager - Implementation Plan Week 1 Day 5
Greeks are PROVIDED by Alpha Vantage - NO local calculation needed!
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
import pandas as pd

from src.core.config import config
from src.core.logger import get_logger
from src.data.alpha_vantage_client import av_client, OptionContract
from src.data.market_data import market_data


logger = get_logger(__name__)


class OptionsDataManager:
    """
    Options data manager using Alpha Vantage
    Greeks are PROVIDED by Alpha Vantage - NO local calculation needed!
    Built on top of MarketDataManager for spot prices
    Implementation Plan Week 1 Day 5
    """
    
    def __init__(self):
        self.market = market_data  # For spot prices
        self.av = av_client  # For options and Greeks
        self.chains = {}
        self.latest_greeks = {}  # Cache latest Greeks from AV
        
    async def fetch_option_chain(self, symbol: str) -> List[OptionContract]:
        """
        Fetch option chain from Alpha Vantage
        Greeks are INCLUDED - no calculation needed!
        """
        logger.info(f"Fetching option chain for {symbol} from Alpha Vantage")
        
        # Get real-time options with Greeks from Alpha Vantage
        options = await self.av.get_realtime_options(symbol, require_greeks=True)
        
        # Cache the chain
        self.chains[symbol] = options
        
        # Cache Greeks for quick access
        for option in options:
            key = f"{symbol}_{option.strike}_{option.expiry}_{option.option_type}"
            self.latest_greeks[key] = {
                'delta': option.delta,
                'gamma': option.gamma,
                'theta': option.theta,
                'vega': option.vega,
                'rho': option.rho
            }
            
        logger.info(f"Fetched {len(options)} options with Greeks for {symbol}")
        return options
    
    def get_option_greeks(self, symbol: str, strike: float, 
                         expiry: str, option_type: str) -> Dict[str, float]:
        """
        Get Greeks for specific option - FROM ALPHA VANTAGE CACHE
        No calculation - just retrieval!
        """
        key = f"{symbol}_{strike}_{expiry}_{option_type}"
        
        if key in self.latest_greeks:
            return self.latest_greeks[key]
        else:
            # If not in cache, fetch fresh data
            logger.warning(f"Greeks not in cache for {key}, fetching...")
            asyncio.create_task(self.fetch_option_chain(symbol))
            # Return zeros temporarily
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def find_atm_options(self, symbol: str, dte_min: int = 0, 
                        dte_max: int = 7) -> List[Dict]:
        """Find ATM options for trading - using Alpha Vantage data"""
        spot = self.market.get_latest_price(symbol)
        chain = self.chains.get(symbol, [])
        
        if not spot:
            logger.warning(f"No spot price for {symbol}")
            return []
        
        atm_options = []
        for option in chain:
            try:
                expiry_date = datetime.strptime(option.expiry, '%Y-%m-%d')
                dte = (expiry_date - datetime.now()).days
                
                if dte_min <= dte <= dte_max:
                    # Find options near the money
                    if abs(option.strike - spot) / spot < 0.02:  # Within 2% of spot
                        atm_options.append({
                            'contract': option,
                            'strike': option.strike,
                            'expiry': option.expiry,
                            'dte': dte,
                            'type': option.option_type,
                            # Greeks from Alpha Vantage!
                            'greeks': {
                                'delta': option.delta,
                                'gamma': option.gamma,
                                'theta': option.theta,
                                'vega': option.vega,
                                'rho': option.rho
                            },
                            'iv': option.implied_volatility
                        })
            except Exception as e:
                logger.error(f"Error processing option: {e}")
        
        return sorted(atm_options, key=lambda x: abs(x['strike'] - spot))
    
    async def get_historical_options_ml_data(self, symbol: str, 
                                            days_back: int = 30) -> pd.DataFrame:
        """
        Get historical options data for ML training
        Alpha Vantage provides up to 20 YEARS of historical options with Greeks!
        """
        logger.info(f"Fetching {days_back} days of historical options for {symbol}")
        
        historical_data = []
        
        for day in range(days_back):
            date = (datetime.now() - timedelta(days=day)).strftime('%Y-%m-%d')
            
            try:
                options = await self.av.get_historical_options(symbol, date)
                
                for option in options:
                    historical_data.append({
                        'date': date,
                        'symbol': symbol,
                        'strike': option.strike,
                        'expiry': option.expiry,
                        'type': option.option_type,
                        'price': option.last,
                        'iv': option.implied_volatility,
                        # Historical Greeks from Alpha Vantage!
                        'delta': option.delta,
                        'gamma': option.gamma,
                        'theta': option.theta,
                        'vega': option.vega,
                        'volume': option.volume,
                        'oi': option.open_interest
                    })
            except Exception as e:
                logger.error(f"Error fetching historical data for {date}: {e}")
        
        return pd.DataFrame(historical_data)


# BUILD ON TOP OF MARKET DATA AND ALPHA VANTAGE
options_data = OptionsDataManager()
