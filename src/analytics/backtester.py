"""
Backtester - Using 20 years of Alpha Vantage historical data
"""
from src.core.logger import get_logger

logger = get_logger(__name__)

class Backtester:
    """Backtesting using AV historical data"""
    
    def __init__(self):
        pass
    
    async def run_backtest(self, strategy, symbols, start_date, end_date):
        """Run backtest using Alpha Vantage 20-year historical data"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        # TODO: Implement backtesting logic
        pass

backtester = Backtester()
