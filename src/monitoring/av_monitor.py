"""
Alpha Vantage Monitor - AV specific monitoring
"""
from src.core.logger import get_logger
from src.data.alpha_vantage_client import av_client

logger = get_logger(__name__)

class AVMonitor:
    """Monitor Alpha Vantage API usage"""
    
    def get_status(self):
        """Get AV API status"""
        return {
            'rate_limit_remaining': av_client.rate_limiter.remaining,
            'total_calls': av_client.total_calls,
            'cache_hits': av_client.cache_hits,
            'hit_rate': av_client.cache_hits / max(av_client.total_calls, 1),
            'avg_response_time': av_client.avg_response_time
        }
    
    def log_status(self):
        """Log current status"""
        status = self.get_status()
        logger.info(f"AV API Status: {status}")

av_monitor = AVMonitor()
