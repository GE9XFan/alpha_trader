"""
Order Executor - IBKR execution
"""
from src.core.logger import get_logger

logger = get_logger(__name__)

class OrderExecutor:
    """Handle order execution through IBKR"""
    
    def __init__(self):
        pass
    
    async def execute_order(self, contract, order_type, quantity):
        """Execute order through IBKR"""
        logger.info(f"Executing order: {contract}")
        # TODO: Implement order execution
        pass

executor = OrderExecutor()
