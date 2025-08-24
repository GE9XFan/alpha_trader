"""
Health Checks - Operations Manual
System health monitoring
"""
import asyncio
from datetime import datetime

from src.core.logger import get_logger
from src.data.market_data import market_data
from src.data.alpha_vantage_client import av_client
from src.trading.risk import risk_manager

logger = get_logger(__name__)

class HealthChecker:
    """System health checks"""
    
    async def check_ibkr_connection(self) -> bool:
        """Check IBKR connection"""
        return market_data.connected
    
    async def check_av_api_health(self) -> bool:
        """Check Alpha Vantage API health"""
        try:
            # Test with a simple call
            await av_client.get_realtime_options('SPY')
            return True
        except:
            return False
    
    async def check_av_rate_limit(self) -> bool:
        """Check AV rate limit status"""
        return av_client.rate_limiter.remaining > 100
    
    async def run_all_checks(self):
        """Run all health checks"""
        checks = {
            'ibkr_connection': await self.check_ibkr_connection(),
            'av_api_health': await self.check_av_api_health(),
            'av_rate_limit': await self.check_av_rate_limit(),
        }
        
        for name, status in checks.items():
            logger.info(f"Health check {name}: {'✅' if status else '❌'}")
        
        return all(checks.values())

health_checker = HealthChecker()
