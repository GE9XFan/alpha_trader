"""
Paper Trader - Implementation Plan Week 5-6
Paper trading using IBKR execution and Alpha Vantage data
"""
import asyncio
from datetime import datetime, time
from typing import Dict, List

from src.core.config import config
from src.core.logger import get_logger
from src.trading.signals import signal_generator
from src.trading.risk import risk_manager
from src.data.market_data import market_data
from src.data.options_data import options_data
from src.data.alpha_vantage_client import av_client
from src.data.database import db


logger = get_logger(__name__)


class PaperTrader:
    """
    Paper trading using IBKR execution and Alpha Vantage data
    REUSES ALL COMPONENTS FROM PHASE 1
    Implementation Plan Week 5-6
    """
    
    def __init__(self):
        # Reuse everything!
        self.signals = signal_generator
        self.risk = risk_manager
        self.market = market_data  # IBKR for execution
        self.options = options_data  # Alpha Vantage for options
        self.av = av_client  # Alpha Vantage for all analytics
        self.db = db
        
        # Paper trading specific
        self.starting_capital = 100000
        self.cash = self.starting_capital
        self.positions = {}
        self.trades = []
        self.running = False
    
    async def run(self):
        """Main paper trading loop"""
        logger.info("Starting paper trading...")
        logger.info("Data sources: IBKR (quotes/execution), Alpha Vantage (options/analytics)")
        
        self.running = True
        
        while self.running:
            try:
                # Market hours check
                if not self._is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Generate signals using Alpha Vantage data
                signals = await self.signals.generate_signals(config.trading.symbols)
                
                for signal in signals:
                    # Use existing risk manager (with AV Greeks)
                    can_trade, reason = await self.risk.can_trade(signal)
                    
                    if can_trade:
                        await self.execute_paper_trade(signal)
                    else:
                        logger.info(f"Signal rejected: {reason}")
                
                # Update positions with fresh AV Greeks
                await self.update_positions()
                
                # Monitor Alpha Vantage rate limit
                logger.info(f"AV API calls remaining: {self.av.rate_limiter.remaining}/600")
                
                # Wait for next cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Paper trading error: {e}")
                await asyncio.sleep(60)
    
    async def execute_paper_trade(self, signal: Dict):
        """Execute paper trade with IBKR paper account"""
        try:
            # Get option price from Alpha Vantage
            options = await self.av.get_realtime_options(signal['symbol'])
            option = next(
                (opt for opt in options 
                 if opt.strike == signal['option']['strike']),
                None
            )
            
            if option:
                fill_price = (option.bid + option.ask) / 2
            else:
                fill_price = 2.0  # Default
            
            trade = {
                'timestamp': datetime.now(),
                'mode': 'paper',
                'symbol': signal['symbol'],
                'option_type': signal['option']['type'],
                'strike': signal['option']['strike'],
                'expiry': signal['option']['expiry'],
                'action': 'BUY',
                'quantity': 5,  # 5 contracts
                'price': fill_price,
                'commission': 0.65 * 5,
                'pnl': 0,  # Updated later
                # Store Greeks from Alpha Vantage at entry
                'entry_delta': signal['av_greeks']['delta'],
                'entry_gamma': signal['av_greeks']['gamma'],
                'entry_theta': signal['av_greeks']['theta'],
                'entry_vega': signal['av_greeks']['vega'],
                'entry_iv': option.implied_volatility if option else 0.2
            }
            
            # Store in database
            with self.db.get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO trades 
                    (timestamp, mode, symbol, option_type, strike, expiry, 
                     action, quantity, fill_price, commission, realized_pnl,
                     entry_delta, entry_gamma, entry_theta, entry_vega, entry_iv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    trade['timestamp'], trade['mode'], trade['symbol'],
                    trade['option_type'], trade['strike'], trade['expiry'],
                    trade['action'], trade['quantity'], trade['price'],
                    trade['commission'], trade['pnl'], trade['entry_delta'],
                    trade['entry_gamma'], trade['entry_theta'], trade['entry_vega'],
                    trade['entry_iv']
                ))
                trade_id = cur.fetchone()[0]
                conn.commit()
            
            # Update position tracking
            self.positions[signal['symbol']] = trade
            self.risk.update_position(signal['symbol'], trade)
            
            logger.info(f"Paper trade executed: {trade['symbol']} {trade['option_type']} "
                       f"${trade['strike']} x{trade['quantity']}")
            logger.info(f"  Greeks (from AV): Δ={trade['entry_delta']:.3f}, "
                       f"Γ={trade['entry_gamma']:.3f}, Θ={trade['entry_theta']:.3f}")
                       
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
    
    async def update_positions(self):
        """Update positions with current prices and Greeks"""
        # TODO: Implement position updates
        pass
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_time = now.time()
        return market_time >= time(9, 30) and market_time <= time(16, 0)
    
    async def stop(self):
        """Stop paper trading"""
        self.running = False
        logger.info("Paper trading stopped")


# PAPER TRADER REUSES EVERYTHING
paper_trader = PaperTrader()
