"""
Live Trader - Implementation Plan Week 9-10
Live trading - IBKR execution with Alpha Vantage analytics
"""
from ib_insync import Option, MarketOrder

from src.core.logger import get_logger
from src.trading.paper_trader import paper_trader


logger = get_logger(__name__)


class LiveTrader:
    """
    Live trading - IBKR execution with Alpha Vantage analytics
    REUSES PAPER TRADER LOGIC
    Implementation Plan Week 9-10
    """
    
    def __init__(self):
        # Reuse all paper trader components!
        self.paper = paper_trader
        
        # Just change execution mode
        self.market = paper_trader.market  # IBKR
        self.av = paper_trader.av  # Alpha Vantage
        self.positions = {}
    
    async def run(self):
        """Live trading - same as paper but real orders through IBKR"""
        logger.warning("Starting LIVE trading with real money!")
        logger.info("Execution: IBKR | Analytics: Alpha Vantage")
        
        # TODO: Implement live trading logic
        # Similar to paper but with real IBKR orders
        pass
    
    async def execute_live_trade(self, signal):
        """Execute real trade through IBKR with Alpha Vantage analytics"""
        # Create option contract for IBKR
        contract = Option(
            signal['symbol'],
            signal['option']['expiry'].replace('-', ''),
            signal['option']['strike'],
            signal['option']['type'][0],  # 'C' or 'P'
            'SMART'
        )
        
        # Create order
        order = MarketOrder('BUY', 5)  # 5 contracts
        
        # Place order through IBKR
        trade = await self.market.execute_order(contract, order)
        
        logger.info(f"LIVE trade executed: {signal['symbol']}")
        
        # Store with AV Greeks
        # TODO: Complete implementation


# LIVE TRADER REUSES PAPER TRADER WITH DUAL DATA SOURCES
live_trader = LiveTrader()
