#!/usr/bin/env python3
"""
Timestamp Tracker Module
Tracks exchange, gateway, and local timestamps for latency analysis
Detects stale quotes and network issues
"""

import time
import json
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

import redis
import orjson

logger = logging.getLogger(__name__)


class LatencyStats:
    """Maintains running statistics for latency measurements"""
    
    def __init__(self, window_size: int = 1000):
        self.window = deque(maxlen=window_size)
        self.percentiles_cache = {}
        self.cache_time = 0
        self.cache_ttl = 1.0  # 1 second cache
        
    def add(self, latency_ns: int):
        """Add latency measurement in nanoseconds"""
        self.window.append(latency_ns)
        
    def get_stats(self) -> Dict:
        """Get comprehensive latency statistics"""
        
        if not self.window:
            return {
                'mean': 0, 'median': 0, 'std': 0,
                'p50': 0, 'p95': 0, 'p99': 0,
                'min': 0, 'max': 0, 'count': 0
            }
            
        # Check cache
        now = time.time()
        if now - self.cache_time < self.cache_ttl:
            return self.percentiles_cache
            
        # Calculate fresh stats
        data = list(self.window)
        data_ms = [x / 1e6 for x in data]  # Convert to milliseconds
        
        stats = {
            'mean': np.mean(data_ms),
            'median': np.median(data_ms),
            'std': np.std(data_ms),
            'p50': np.percentile(data_ms, 50),
            'p95': np.percentile(data_ms, 95),
            'p99': np.percentile(data_ms, 99),
            'min': np.min(data_ms),
            'max': np.max(data_ms),
            'count': len(data)
        }
        
        # Cache results
        self.percentiles_cache = stats
        self.cache_time = now
        
        return stats


class TimestampTracker:
    """Tracks timestamps for latency analysis and stale detection"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.symbols = config['trading']['symbols']
        
        # Latency tracking per symbol
        self.latency_stats = {
            symbol: {
                'exchange_to_gateway': LatencyStats(),
                'gateway_to_local': LatencyStats(),
                'total': LatencyStats(),
                'processing': LatencyStats()
            } for symbol in self.symbols
        }
        
        # Timestamp sequences for analysis
        self.timestamp_sequences = defaultdict(lambda: {
            'exchange': deque(maxlen=1000),
            'gateway': deque(maxlen=1000),
            'local': deque(maxlen=1000)
        })
        
        # Clock drift detection
        self.clock_drift = {
            'exchange': deque(maxlen=100),
            'gateway': deque(maxlen=100)
        }
        
        # Stale quote detection
        self.stale_thresholds = {
            'warning': 100_000_000,  # 100ms in nanoseconds
            'critical': 500_000_000,  # 500ms
            'stale': 1_000_000_000   # 1 second
        }
        
        # Latency spike detection
        self.latency_spikes = defaultdict(list)
        self.spike_threshold = 3  # Standard deviations
        
        # Network quality metrics
        self.network_quality = {
            'packet_loss': 0.0,
            'jitter': 0.0,
            'quality_score': 100.0
        }
        
        # Time synchronization tracking
        self.time_sync = {
            'last_ntp_sync': None,
            'drift_ms': 0.0,
            'synchronized': True
        }
        
    def track_message(self, symbol: str, msg_type: str, 
                     exchange_ts: Optional[int] = None,
                     gateway_ts: Optional[int] = None,
                     local_ts: Optional[int] = None):
        """Track timestamps for a message"""
        
        try:
            # Use current time if local timestamp not provided
            if local_ts is None:
                local_ts = time.time_ns()
                
            # Store in sequences
            if exchange_ts:
                self.timestamp_sequences[symbol]['exchange'].append(exchange_ts)
            if gateway_ts:
                self.timestamp_sequences[symbol]['gateway'].append(gateway_ts)
            self.timestamp_sequences[symbol]['local'].append(local_ts)
            
            # Calculate latencies
            latencies = self._calculate_latencies(exchange_ts, gateway_ts, local_ts)
            
            # Update statistics
            if latencies['exchange_to_gateway']:
                self.latency_stats[symbol]['exchange_to_gateway'].add(latencies['exchange_to_gateway'])
                
            if latencies['gateway_to_local']:
                self.latency_stats[symbol]['gateway_to_local'].add(latencies['gateway_to_local'])
                
            if latencies['total']:
                self.latency_stats[symbol]['total'].add(latencies['total'])
                
                # Check for stale quotes
                self._check_stale_quote(symbol, latencies['total'])
                
                # Check for latency spikes
                self._check_latency_spike(symbol, latencies['total'])
                
            # Track clock drift if we have exchange timestamps
            if exchange_ts:
                self._track_clock_drift(exchange_ts, local_ts)
                
            # Update network quality
            self._update_network_quality(latencies)
            
            # Store metrics in Redis
            if time.time() % 1 < 0.01:  # Every second
                self._store_latency_metrics(symbol)
                
        except Exception as e:
            logger.error(f"Error tracking timestamp for {symbol}: {e}")
            
    def _calculate_latencies(self, exchange_ts: Optional[int],
                            gateway_ts: Optional[int],
                            local_ts: int) -> Dict:
        """Calculate various latency measurements"""
        
        latencies = {
            'exchange_to_gateway': None,
            'gateway_to_local': None,
            'total': None,
            'processing': None
        }
        
        if exchange_ts and gateway_ts:
            latencies['exchange_to_gateway'] = gateway_ts - exchange_ts
            
        if gateway_ts and local_ts:
            latencies['gateway_to_local'] = local_ts - gateway_ts
            
        if exchange_ts and local_ts:
            latencies['total'] = local_ts - exchange_ts
            
        return latencies
        
    def _check_stale_quote(self, symbol: str, total_latency: int):
        """Check if quote is stale based on latency"""
        
        if total_latency > self.stale_thresholds['stale']:
            # Quote is stale
            self._trigger_stale_alert(symbol, 'stale', total_latency)
            
        elif total_latency > self.stale_thresholds['critical']:
            # Quote is critically delayed
            self._trigger_stale_alert(symbol, 'critical', total_latency)
            
        elif total_latency > self.stale_thresholds['warning']:
            # Quote is delayed
            self._trigger_stale_alert(symbol, 'warning', total_latency)
            
    def _trigger_stale_alert(self, symbol: str, severity: str, latency_ns: int):
        """Trigger stale quote alert"""
        
        alert = {
            'symbol': symbol,
            'alert_type': 'STALE_QUOTE',
            'severity': severity,
            'latency_ms': latency_ns / 1e6,
            'timestamp': time.time_ns()
        }
        
        # Store alert
        self.redis.setex(
            f'alerts:{symbol}:stale',
            5,
            orjson.dumps(alert).decode('utf-8')
        )
        
        if severity == 'stale':
            logger.warning(f"STALE QUOTE: {symbol} - Latency: {latency_ns/1e6:.1f}ms")
            
    def _check_latency_spike(self, symbol: str, latency_ns: int):
        """Detect latency spikes"""
        
        stats = self.latency_stats[symbol]['total'].get_stats()
        
        if stats['count'] < 100:
            return  # Need enough data
            
        latency_ms = latency_ns / 1e6
        
        # Check if this is a spike (> 3 std deviations)
        if latency_ms > stats['mean'] + (self.spike_threshold * stats['std']):
            spike = {
                'symbol': symbol,
                'latency_ms': latency_ms,
                'mean_ms': stats['mean'],
                'std_ms': stats['std'],
                'deviation': (latency_ms - stats['mean']) / stats['std'],
                'timestamp': time.time()
            }
            
            self.latency_spikes[symbol].append(spike)
            
            # Keep only recent spikes
            if len(self.latency_spikes[symbol]) > 100:
                self.latency_spikes[symbol] = self.latency_spikes[symbol][-100:]
                
            logger.info(f"Latency spike on {symbol}: {latency_ms:.1f}ms ({spike['deviation']:.1f} std)")
            
    def _track_clock_drift(self, exchange_ts: int, local_ts: int):
        """Track clock drift between exchange and local"""
        
        # Simple drift estimation
        # In production, would use more sophisticated NTP-like algorithms
        
        drift = local_ts - exchange_ts
        self.clock_drift['exchange'].append(drift)
        
        # Calculate average drift
        if len(self.clock_drift['exchange']) >= 10:
            avg_drift = np.mean(list(self.clock_drift['exchange']))
            self.time_sync['drift_ms'] = avg_drift / 1e6
            
            # Check if drift is excessive
            if abs(self.time_sync['drift_ms']) > 1000:  # > 1 second
                self.time_sync['synchronized'] = False
                logger.warning(f"Clock drift detected: {self.time_sync['drift_ms']:.1f}ms")
                
    def _update_network_quality(self, latencies: Dict):
        """Update network quality metrics"""
        
        # Simple quality scoring
        # In production, would use more sophisticated algorithms
        
        if latencies['total']:
            latency_ms = latencies['total'] / 1e6
            
            # Quality score based on latency
            if latency_ms < 10:
                quality = 100
            elif latency_ms < 50:
                quality = 90
            elif latency_ms < 100:
                quality = 70
            elif latency_ms < 500:
                quality = 50
            else:
                quality = 20
                
            # Exponential moving average
            alpha = 0.1
            self.network_quality['quality_score'] = (
                alpha * quality + (1 - alpha) * self.network_quality['quality_score']
            )
            
    def _store_latency_metrics(self, symbol: str):
        """Store latency metrics in Redis"""
        
        try:
            # Get all stats
            metrics = {
                'exchange_to_gateway': self.latency_stats[symbol]['exchange_to_gateway'].get_stats(),
                'gateway_to_local': self.latency_stats[symbol]['gateway_to_local'].get_stats(),
                'total': self.latency_stats[symbol]['total'].get_stats(),
                'network_quality': self.network_quality,
                'time_sync': self.time_sync,
                'spike_count': len(self.latency_spikes[symbol]),
                'timestamp': time.time_ns()
            }
            
            # Store in Redis
            pipe = self.redis.pipeline()
            
            # Full metrics
            pipe.setex(
                f'market:{symbol}:latency:metrics',
                5,
                orjson.dumps(metrics).decode('utf-8')
            )
            
            # Quick access values
            pipe.setex(f'market:{symbol}:latency:total', 5, metrics['total']['mean'])
            pipe.setex(f'market:{symbol}:latency:p99', 5, metrics['total']['p99'])
            
            # Network quality
            pipe.setex(
                'system:network:quality',
                5,
                orjson.dumps(self.network_quality).decode('utf-8')
            )
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Error storing latency metrics for {symbol}: {e}")
            
    def get_latency_stats(self, symbol: str) -> Dict:
        """Get current latency statistics for a symbol"""
        
        return {
            'exchange_to_gateway': self.latency_stats[symbol]['exchange_to_gateway'].get_stats(),
            'gateway_to_local': self.latency_stats[symbol]['gateway_to_local'].get_stats(),
            'total': self.latency_stats[symbol]['total'].get_stats()
        }
        
    def analyze_timestamp_sequence(self, symbol: str) -> Dict:
        """Analyze timestamp sequences for patterns"""
        
        sequences = self.timestamp_sequences[symbol]
        
        analysis = {
            'gaps_detected': 0,
            'out_of_order': 0,
            'duplicates': 0
        }
        
        # Check for gaps in sequence
        if len(sequences['exchange']) >= 2:
            diffs = np.diff(list(sequences['exchange']))
            
            # Gaps (> 1 second)
            analysis['gaps_detected'] = np.sum(diffs > 1e9)
            
            # Out of order (negative diffs)
            analysis['out_of_order'] = np.sum(diffs < 0)
            
            # Duplicates (zero diffs)
            analysis['duplicates'] = np.sum(diffs == 0)
            
        return analysis
        
    def get_network_health(self) -> Dict:
        """Get overall network health status"""
        
        return {
            'quality_score': self.network_quality['quality_score'],
            'synchronized': self.time_sync['synchronized'],
            'drift_ms': self.time_sync['drift_ms'],
            'status': self._get_health_status()
        }
        
    def _get_health_status(self) -> str:
        """Determine overall health status"""
        
        if not self.time_sync['synchronized']:
            return 'DEGRADED'
            
        if self.network_quality['quality_score'] >= 90:
            return 'EXCELLENT'
        elif self.network_quality['quality_score'] >= 70:
            return 'GOOD'
        elif self.network_quality['quality_score'] >= 50:
            return 'FAIR'
        else:
            return 'POOR'