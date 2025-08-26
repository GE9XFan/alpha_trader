"""
Metrics Collection - Tech Spec Section 8.1
Prometheus metrics for monitoring
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from src.core.config import config
from src.core.logger import get_logger

logger = get_logger(__name__)

# Alpha Vantage metrics
av_api_calls = Counter('av_api_calls_total', 'Total AV API calls', ['endpoint'])
av_cache_hits = Counter('av_cache_hits_total', 'AV cache hits', ['data_type'])
av_response_time = Histogram('av_response_seconds', 'AV response time', ['endpoint'])
av_rate_limit_remaining = Gauge('av_rate_limit_remaining', 'AV rate limit remaining')

# Portfolio Greeks from Alpha Vantage
portfolio_greeks = Gauge('portfolio_greeks', 'Portfolio Greeks', ['greek'])

# Trading metrics
signals_generated = Counter('signals_generated_total', 'Total signals generated', ['symbol', 'signal_type'])
trades_executed = Counter('trades_executed_total', 'Total trades executed', ['mode'])
daily_pnl = Gauge('daily_pnl', 'Daily P&L')

def start_metrics_server():
    """Start Prometheus metrics server"""
    port = config.monitoring['metrics'].get('port', 9090)
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")

start_metrics_server()
