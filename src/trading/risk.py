"""
Risk Manager - Implementation Plan Week 3 Day 3-4
Risk management using Alpha Vantage Greeks
"""
from typing import Dict, List, Tuple
import numpy as np
import asyncio

from src.core.config import config
from src.core.logger import get_logger
from src.core.exceptions import RiskLimitException, PositionLimitException, GreeksLimitException
from src.data.options_data import options_data
from src.data.database import db


logger = get_logger(__name__)


class RiskManager:
    """
    Risk management using Alpha Vantage Greeks
    Same rules for paper and live
    Implementation Plan Week 3 Day 3-4
    """
    
    def __init__(self):
        self.trading_config = config.trading
        self.risk_config = config.risk
        self.options = options_data  # Gets Greeks from Alpha Vantage
        self.db = db
        
        # Risk limits
        self.max_positions = self.trading_config.max_positions
        self.max_position_size = self.trading_config.max_position_size
        self.daily_loss_limit = self.trading_config.daily_loss_limit
        
        # Greeks limits (using AV data)
        self.greeks_limits = self.trading_config.greeks_limits
        
        # Current state
        self.positions = {}
        self.daily_pnl = 0.0
        self.portfolio_greeks = {
            'delta': 0.0, 'gamma': 0.0, 
            'vega': 0.0, 'theta': 0.0
        }
    
    async def can_trade(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if trade is allowed using Alpha Vantage Greeks
        Used by paper and live equally
        """
        # Check position count
        if len(self.positions) >= self.max_positions:
            return False, "Max positions reached"
        
        # Check daily loss
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, "Daily loss limit reached"
        
        # Calculate position size
        position_size = await self._calculate_position_size(signal)
        if position_size > self.max_position_size:
            return False, f"Position size ${position_size:.2f} exceeds limit"
        
        # Check Greeks impact using Alpha Vantage data
        projected_greeks = signal.get('av_greeks', {})  # Greeks from signal (from AV)
        
        for greek, (min_val, max_val) in self.greeks_limits.items():
            current = self.portfolio_greeks.get(greek, 0)
            # Multiply by 5 contracts (standard size)
            new_value = current + (projected_greeks.get(greek, 0) * 5)
            
            if new_value < min_val or new_value > max_val:
                return False, f"Would breach {greek} limit: {new_value:.3f}"
        
        return True, "OK"
    
    async def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size in dollars"""
        try:
            # Get current option price from Alpha Vantage
            options = await options_data.av.get_realtime_options(signal['symbol'])
            
            # Find the specific option
            option = next(
                (opt for opt in options 
                 if opt.strike == signal['option']['strike'] 
                 and opt.option_type == signal['option']['type']),
                None
            )
            
            if option:
                # Use mid price
                option_price = (option.bid + option.ask) / 2
            else:
                option_price = 2.0  # Default estimate
            
            contracts = 5  # Standard position size
            return contracts * 100 * option_price
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 10000  # Default max
    
    def update_position(self, symbol: str, position: Dict):
        """Update position tracking with Alpha Vantage Greeks"""
        self.positions[symbol] = position
        asyncio.create_task(self._update_portfolio_greeks_from_av())
    
    async def _update_portfolio_greeks_from_av(self):
        """Update portfolio Greeks using fresh Alpha Vantage data"""
        total_greeks = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        for symbol, position in self.positions.items():
            try:
                # Get current Greeks from Alpha Vantage
                greeks = self.options.get_option_greeks(
                    symbol,
                    position['strike'],
                    position['expiry'],
                    position['option_type']
                )
                
                for key in total_greeks:
                    total_greeks[key] += greeks.get(key, 0) * position['quantity']
                    
            except Exception as e:
                logger.error(f"Error updating Greeks for {symbol}: {e}")
        
        self.portfolio_greeks = total_greeks
        
        # Log to database
        try:
            with self.db.get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO portfolio_greeks_history 
                    (timestamp, delta, gamma, theta, vega)
                    VALUES (NOW(), %s, %s, %s, %s)
                """, (total_greeks['delta'], total_greeks['gamma'], 
                      total_greeks['theta'], total_greeks['vega']))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging Greeks to database: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        logger.info("Daily risk stats reset")


# RISK MANAGER USING ALPHA VANTAGE GREEKS
risk_manager = RiskManager()
