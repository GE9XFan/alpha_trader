"""
Health check system for monitoring
All configuration from environment - zero hardcoding
"""
import os
import time
import json
from typing import Dict, Any
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from .logger import get_logger
from .metrics import MetricsCollector


class HealthChecker:
    """
    Health check system with HTTP endpoint
    All configuration from environment
    """
    
    def __init__(self):
        """Initialize health checker with environment configuration"""
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Get ALL configuration from environment
        self.enabled = os.environ['HEALTH_ENABLED'].lower() == 'true'
        self.port = int(os.environ['HEALTH_PORT'])
        self.path = os.environ['HEALTH_PATH']
        self.timeout = int(os.environ['HEALTH_TIMEOUT'])
        self.check_db = os.environ['HEALTH_CHECK_DB'].lower() == 'true'
        self.check_redis = os.environ['HEALTH_CHECK_REDIS'].lower() == 'true'
        
        self.components = {}
        self.server_thread = None
        
        if self.enabled:
            self._start_server()
    
    def register_component(self, name: str, check_func):
        """Register a component health check function"""
        self.components[name] = check_func
        self.logger.info(f"Registered health check for {name}")
    
    def check_component(self, name: str) -> Dict[str, Any]:
        """Check health of a specific component"""
        if name not in self.components:
            return {
                'name': name,
                'healthy': False,
                'error': 'Component not registered',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        start_time = time.time()
        
        try:
            result = self.components[name]()
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics.record_health_check(name, result.get('healthy', False), duration)
            
            return {
                'name': name,
                'healthy': result.get('healthy', False),
                'response_time_ms': duration * 1000,
                'details': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_health_check(name, False, duration)
            
            return {
                'name': name,
                'healthy': False,
                'error': str(e),
                'response_time_ms': duration * 1000,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def check_all(self) -> Dict[str, Any]:
        """Check health of all registered components"""
        results = {}
        overall_healthy = True
        
        for name in self.components:
            result = self.check_component(name)
            results[name] = result
            if not result['healthy']:
                overall_healthy = False
        
        return {
            'overall': 'healthy' if overall_healthy else 'unhealthy',
            'components': results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _start_server(self):
        """Start HTTP health check server"""
        class HealthHandler(BaseHTTPRequestHandler):
            def __init__(handler_self, *args, **kwargs):
                handler_self.health_checker = self
                super().__init__(*args, **kwargs)
            
            def do_GET(handler_self):
                if handler_self.path == self.path:
                    # Overall health check
                    result = self.check_all()
                    status_code = 200 if result['overall'] == 'healthy' else 503
                    
                    handler_self.send_response(status_code)
                    handler_self.send_header('Content-Type', 'application/json')
                    handler_self.end_headers()
                    handler_self.wfile.write(json.dumps(result).encode())
                    
                elif handler_self.path == f"{self.path}/ready":
                    # Readiness probe
                    result = self.check_all()
                    status_code = 200 if result['overall'] == 'healthy' else 503
                    
                    handler_self.send_response(status_code)
                    handler_self.send_header('Content-Type', 'application/json')
                    handler_self.end_headers()
                    handler_self.wfile.write(json.dumps({'ready': result['overall'] == 'healthy'}).encode())
                    
                elif handler_self.path == f"{self.path}/live":
                    # Liveness probe - basic check
                    handler_self.send_response(200)
                    handler_self.send_header('Content-Type', 'application/json')
                    handler_self.end_headers()
                    handler_self.wfile.write(json.dumps({'live': True}).encode())
                    
                else:
                    handler_self.send_response(404)
                    handler_self.end_headers()
            
            def log_message(handler_self, format, *args):
                # Suppress default logging
                pass
        
        def run_server():
            server = HTTPServer(('', self.port), HealthHandler)
            self.logger.info(f"Health check server started on port {self.port}")
            server.serve_forever()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()