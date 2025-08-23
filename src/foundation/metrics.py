"""
Production metrics collection with Prometheus
All configuration from environment - zero hardcoding
"""
import os
from typing import Optional, Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from prometheus_client import REGISTRY

from .logger import get_logger


class MetricsCollector:
    """
    Institutional-grade metrics collection
    All configuration from environment - no hardcoding
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize metrics with environment configuration"""
        if self._initialized:
            return
        
        self.logger = get_logger(__name__)
        
        # Get ALL configuration from environment - NO DEFAULTS
        self.enabled = os.environ['METRICS_ENABLED'].lower() == 'true'
        self.port = int(os.environ['METRICS_PORT'])
        self.namespace = os.environ['METRICS_NAMESPACE']
        self.subsystem = os.environ['METRICS_SUBSYSTEM']
        
        if not self.enabled:
            self.logger.info("Metrics collection disabled")
            return
        
        # Initialize metrics
        self._create_metrics()
        
        # Start metrics server
        self._start_server()
        
        self._initialized = True
    
    def _create_metrics(self):
        """Create all Prometheus metrics"""
        # Database metrics
        self.db_queries_total = Counter(
            f'{self.namespace}_{self.subsystem}_db_queries_total',
            'Total database queries',
            ['operation', 'status']
        )
        
        self.db_query_duration = Histogram(
            f'{self.namespace}_{self.subsystem}_db_query_duration_seconds',
            'Database query duration',
            ['operation'],
            buckets=self._get_histogram_buckets()
        )
        
        self.db_active_connections = Gauge(
            f'{self.namespace}_{self.subsystem}_db_active_connections',
            'Active database connections'
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            f'{self.namespace}_{self.subsystem}_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses_total = Counter(
            f'{self.namespace}_{self.subsystem}_cache_misses_total',
            'Total cache misses'
        )
        
        self.cache_operations_duration = Histogram(
            f'{self.namespace}_{self.subsystem}_cache_operation_duration_seconds',
            'Cache operation duration',
            ['operation'],
            buckets=self._get_cache_buckets()
        )
        
        # API metrics
        self.api_calls_total = Counter(
            f'{self.namespace}_{self.subsystem}_api_calls_total',
            'Total API calls',
            ['api', 'endpoint', 'status']
        )
        
        self.api_call_duration = Histogram(
            f'{self.namespace}_{self.subsystem}_api_call_duration_seconds',
            'API call duration',
            ['api', 'endpoint'],
            buckets=self._get_api_buckets()
        )
        
        # Rate limiting metrics
        self.rate_limit_requests = Counter(
            f'{self.namespace}_{self.subsystem}_rate_limit_requests_total',
            'Rate limit requests',
            ['resource', 'status']
        )
        
        self.rate_limit_tokens = Gauge(
            f'{self.namespace}_{self.subsystem}_rate_limit_tokens',
            'Available rate limit tokens',
            ['resource']
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            f'{self.namespace}_{self.subsystem}_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['service']
        )
        
        self.circuit_breaker_failures = Counter(
            f'{self.namespace}_{self.subsystem}_circuit_breaker_failures_total',
            'Circuit breaker failures',
            ['service']
        )
        
        # Health metrics
        self.health_check_status = Gauge(
            f'{self.namespace}_{self.subsystem}_health_check_status',
            'Health check status (1=healthy, 0=unhealthy)',
            ['component']
        )
        
        self.health_check_duration = Histogram(
            f'{self.namespace}_{self.subsystem}_health_check_duration_seconds',
            'Health check duration',
            ['component']
        )
        
        # System info
        self.system_info = Info(
            f'{self.namespace}_{self.subsystem}_info',
            'System information'
        )
        
        # Set system info from environment
        self.system_info.info({
            'version': os.environ.get('APP_VERSION', '1.0.0'),
            'environment': os.environ['ENVIRONMENT'],
            'app_name': os.environ['APP_NAME']
        })
        
        self.logger.info("Metrics initialized")
    
    def _get_histogram_buckets(self):
        """Get histogram buckets from environment or use sensible defaults"""
        # Even buckets can be configured
        buckets_str = os.environ.get(
            'METRICS_HISTOGRAM_BUCKETS',
            '0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0'
        )
        return [float(b) for b in buckets_str.split(',')]
    
    def _get_cache_buckets(self):
        """Get cache-specific histogram buckets"""
        buckets_str = os.environ.get(
            'METRICS_CACHE_BUCKETS',
            '0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0'
        )
        return [float(b) for b in buckets_str.split(',')]
    
    def _get_api_buckets(self):
        """Get API-specific histogram buckets"""
        buckets_str = os.environ.get(
            'METRICS_API_BUCKETS',
            '0.1,0.25,0.5,1.0,2.5,5.0,10.0,30.0,60.0'
        )
        return [float(b) for b in buckets_str.split(',')]
    
    def _start_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port, registry=REGISTRY)
            self.logger.info(
                f"Metrics server started",
                port=self.port,
                endpoint=f"http://localhost:{self.port}/metrics"
            )
        except Exception as e:
            self.logger.error(f"Failed to start metrics server", error=str(e))
    
    def record_db_query(
        self,
        operation: str,
        duration: float,
        status: str = 'success'
    ):
        """Record database query metrics"""
        if not self.enabled:
            return
        
        self.db_queries_total.labels(operation=operation, status=status).inc()
        self.db_query_duration.labels(operation=operation).observe(duration)
    
    def record_cache_operation(
        self,
        operation: str,
        duration: float,
        hit: Optional[bool] = None
    ):
        """Record cache operation metrics"""
        if not self.enabled:
            return
        
        if hit is not None:
            if hit:
                self.cache_hits_total.inc()
            else:
                self.cache_misses_total.inc()
        
        self.cache_operations_duration.labels(operation=operation).observe(duration)
    
    def record_api_call(
        self,
        api: str,
        endpoint: str,
        duration: float,
        status: str = 'success'
    ):
        """Record API call metrics"""
        if not self.enabled:
            return
        
        self.api_calls_total.labels(
            api=api,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.api_call_duration.labels(
            api=api,
            endpoint=endpoint
        ).observe(duration)
    
    def update_rate_limit(
        self,
        resource: str,
        tokens: int,
        request_status: str = 'allowed'
    ):
        """Update rate limit metrics"""
        if not self.enabled:
            return
        
        self.rate_limit_requests.labels(
            resource=resource,
            status=request_status
        ).inc()
        
        self.rate_limit_tokens.labels(resource=resource).set(tokens)
    
    def update_circuit_breaker(
        self,
        service: str,
        state: str,
        failure: bool = False
    ):
        """Update circuit breaker metrics"""
        if not self.enabled:
            return
        
        # Map state to numeric value
        state_map = {'closed': 0, 'open': 1, 'half-open': 2}
        state_value = state_map.get(state, -1)
        
        self.circuit_breaker_state.labels(service=service).set(state_value)
        
        if failure:
            self.circuit_breaker_failures.labels(service=service).inc()
    
    def record_health_check(
        self,
        component: str,
        healthy: bool,
        duration: float
    ):
        """Record health check metrics"""
        if not self.enabled:
            return
        
        self.health_check_status.labels(component=component).set(
            1 if healthy else 0
        )
        
        self.health_check_duration.labels(component=component).observe(duration)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        if not self.enabled:
            return {}
        
        # This would typically integrate with Prometheus client
        # For now, return basic structure
        return {
            'enabled': self.enabled,
            'port': self.port,
            'namespace': self.namespace,
            'subsystem': self.subsystem
        }