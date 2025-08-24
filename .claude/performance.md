# Performance Optimization Guide

## Critical Path Latency Budget (50ms Total)

| Component | Target | Current | Optimization Priority |
|-----------|--------|---------|----------------------|
| IBKR Data Receipt | 5ms | - | LOW (network bound) |
| Greeks Calculation | 5ms | - | CRITICAL |
| Feature Calculation | 15ms | - | HIGH |
| Model Inference | 10ms | - | HIGH |
| Risk Check | 5ms | - | MEDIUM |
| Order Execution | 15ms | - | LOW (API bound) |

## Options-Specific Optimizations

### Greeks Calculation (<5ms for 100 contracts)

```python
import numpy as np
from scipy.stats import norm
from numba import jit, vectorize
import functools

# Vectorized Greeks calculation
@vectorize(['float64(float64, float64, float64, float64, float64)'], target='parallel')
def calculate_delta_vectorized(S, K, T, r, sigma):
    """
    Vectorized delta calculation using NumPy
    Performance: O(n) but parallelized
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# JIT-compiled for hot paths
@jit(nopython=True, cache=True, parallel=True)
def calculate_portfolio_greeks(positions, spot_prices, rates, times, vols):
    """
    Calculate portfolio-wide Greeks with Numba JIT
    Target: <5ms for 100 positions
    """
    n = len(positions)
    portfolio_delta = 0.0
    portfolio_gamma = 0.0
    portfolio_vega = 0.0
    portfolio_theta = 0.0
    
    for i in range(n):
        # Inline calculations to avoid function call overhead
        # ... Greek calculations here ...
        pass
    
    return portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta

# Cache frequently accessed Greeks
@functools.lru_cache(maxsize=10000)
def get_cached_greeks(symbol, strike, expiry, spot, vol):
    """Cache Greeks for frequently accessed options"""
    return calculate_all_greeks(symbol, strike, expiry, spot, vol)
```

### Option Chain Processing

```python
# Efficient chain processing with pandas
def process_option_chain_optimized(chain_df):
    """
    Process entire option chain efficiently
    Target: <10ms for 500 contracts
    """
    # Use numpy arrays instead of iterating
    chain_array = chain_df[['strike', 'bid', 'ask', 'volume']].values
    
    # Vectorized operations
    mid_prices = (chain_array[:, 1] + chain_array[:, 2]) / 2
    
    # Calculate all Greeks at once
    strikes = chain_array[:, 0]
    spot = chain_df['underlying_price'].iloc[0]
    
    # Pre-allocate result arrays
    n = len(chain_df)
    deltas = np.empty(n)
    gammas = np.empty(n)
    
    # Bulk calculate using vectorization
    deltas[:] = calculate_delta_vectorized(spot, strikes, ...)
    gammas[:] = calculate_gamma_vectorized(spot, strikes, ...)
    
    # Update DataFrame in place
    chain_df['delta'] = deltas
    chain_df['gamma'] = gammas
    
    return chain_df
```

### VPIN Calculation Optimization

```python
from collections import deque
import numpy as np

class VPINCalculator:
    """Optimized VPIN calculator with O(1) updates"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.buy_volumes = deque(maxlen=window_size)
        self.sell_volumes = deque(maxlen=window_size)
        self.running_buy_sum = 0
        self.running_sell_sum = 0
        
    def update(self, buy_volume, sell_volume):
        """O(1) VPIN update"""
        # Remove oldest if at capacity
        if len(self.buy_volumes) == self.window_size:
            self.running_buy_sum -= self.buy_volumes[0]
            self.running_sell_sum -= self.sell_volumes[0]
        
        # Add new volumes
        self.buy_volumes.append(buy_volume)
        self.sell_volumes.append(sell_volume)
        self.running_buy_sum += buy_volume
        self.running_sell_sum += sell_volume
        
        # Calculate VPIN
        total = self.running_buy_sum + self.running_sell_sum
        if total > 0:
            imbalance = abs(self.running_buy_sum - self.running_sell_sum)
            return imbalance / total
        return 0.0
```

## Feature Engineering Optimization

### Technical Indicators (<15ms for 147 features)

```python
import talib
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class FeatureEngineOptimized:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.feature_cache = {}
        
    def calculate_features_parallel(self, data):
        """
        Calculate 147 features in parallel
        Target: <15ms total
        """
        # Split features into groups for parallel processing
        price_features = ['RSI', 'MACD', 'BBANDS', 'SMA', 'EMA']
        volume_features = ['OBV', 'AD', 'MFI', 'VWAP']
        volatility_features = ['ATR', 'BBWIDTH', 'KC']
        momentum_features = ['MOM', 'ROC', 'WILLR', 'STOCH']
        
        # Execute in parallel
        futures = []
        futures.append(self.executor.submit(self._calc_price_features, data))
        futures.append(self.executor.submit(self._calc_volume_features, data))
        futures.append(self.executor.submit(self._calc_volatility_features, data))
        futures.append(self.executor.submit(self._calc_momentum_features, data))
        
        # Combine results
        results = {}
        for future in futures:
            results.update(future.result())
        
        return results
    
    def _calc_price_features(self, data):
        """Calculate price-based features"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        features = {}
        features['RSI'] = talib.RSI(close, timeperiod=14)[-1]
        features['MACD'], features['MACD_signal'], _ = talib.MACD(close)
        features['BB_upper'], features['BB_middle'], features['BB_lower'] = talib.BBANDS(close)
        
        return features
```

## Model Inference Optimization

### XGBoost Optimization (<10ms)

```python
import xgboost as xgb
import onnxruntime as ort

class ModelServerOptimized:
    def __init__(self, model_path):
        # Load model
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(model_path)
        
        # Convert to ONNX for faster inference
        self.onnx_session = self._convert_to_onnx(model_path)
        
        # Pre-allocate arrays
        self.feature_buffer = np.zeros((1, 147), dtype=np.float32)
        
    def predict_fast(self, features):
        """
        Ultra-fast prediction
        Target: <10ms
        """
        # Use pre-allocated buffer
        self.feature_buffer[0, :] = features
        
        # ONNX inference (faster than XGBoost)
        input_name = self.onnx_session.get_inputs()[0].name
        pred = self.onnx_session.run(None, {input_name: self.feature_buffer})[0]
        
        return pred[0]
    
    def _convert_to_onnx(self, model_path):
        """Convert XGBoost to ONNX for faster inference"""
        # ... ONNX conversion code ...
        pass
```

## Database Optimization

### PostgreSQL Performance

```sql
-- Optimized options_data table
CREATE TABLE options_data (
    id BIGSERIAL,
    symbol VARCHAR(10) NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    expiry DATE NOT NULL,
    option_type CHAR(4) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DECIMAL(10,2),
    ask DECIMAL(10,2),
    delta REAL,
    gamma REAL,
    vega REAL,
    theta REAL,
    PRIMARY KEY (symbol, strike, expiry, option_type, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for each day
CREATE TABLE options_data_2024_01_19 PARTITION OF options_data
FOR VALUES FROM ('2024-01-19') TO ('2024-01-20');

-- Optimized indexes
CREATE INDEX idx_options_symbol_expiry ON options_data(symbol, expiry)
WHERE expiry >= CURRENT_DATE;  -- Partial index for active options

CREATE INDEX idx_options_0dte ON options_data(expiry)
WHERE expiry = CURRENT_DATE;  -- Special index for 0DTE

-- Use BRIN index for time-series data
CREATE INDEX idx_options_timestamp_brin ON options_data
USING BRIN (timestamp) WITH (pages_per_range = 128);
```

### Redis Caching Strategy

```python
import redis
import pickle
import lz4.frame

class GreeksCacheOptimized:
    def __init__(self):
        # Use Redis pipeline for batch operations
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False  # Binary mode for speed
        )
        self.pipeline = self.redis.pipeline()
        
    def cache_greeks_batch(self, greeks_data):
        """
        Cache multiple Greeks calculations at once
        Uses LZ4 compression for speed
        """
        pipe = self.redis.pipeline()
        
        for key, greeks in greeks_data.items():
            # Compress with LZ4 (faster than gzip)
            compressed = lz4.frame.compress(pickle.dumps(greeks))
            pipe.setex(key, 5, compressed)  # 5 second TTL
        
        pipe.execute()
    
    def get_greeks_batch(self, keys):
        """Retrieve multiple Greeks in one call"""
        pipe = self.redis.pipeline()
        for key in keys:
            pipe.get(key)
        
        results = pipe.execute()
        
        # Decompress and deserialize
        greeks = []
        for data in results:
            if data:
                greeks.append(pickle.loads(lz4.frame.decompress(data)))
            else:
                greeks.append(None)
        
        return greeks
```

## Memory Optimization

### Memory-Mapped Files for Shared Data

```python
import mmap
import struct

class SharedMemoryGreeks:
    """Share Greeks between processes using memory-mapped files"""
    
    def __init__(self, max_positions=20):
        self.max_positions = max_positions
        # Each position: symbol(10) + 5 Greeks(4*5) = 30 bytes
        self.size = max_positions * 30
        
        # Create memory-mapped file
        self.f = open('/dev/shm/greeks.dat', 'r+b')
        self.mm = mmap.mmap(self.f.fileno(), self.size)
        
    def write_greeks(self, position_id, symbol, delta, gamma, vega, theta, rho):
        """Write Greeks to shared memory"""
        offset = position_id * 30
        
        # Pack data
        data = struct.pack(
            '10s5f',
            symbol.encode()[:10],
            delta, gamma, vega, theta, rho
        )
        
        self.mm[offset:offset+30] = data
        
    def read_greeks(self, position_id):
        """Read Greeks from shared memory"""
        offset = position_id * 30
        data = self.mm[offset:offset+30]
        
        unpacked = struct.unpack('10s5f', data)
        return {
            'symbol': unpacked[0].decode().strip(),
            'delta': unpacked[1],
            'gamma': unpacked[2],
            'vega': unpacked[3],
            'theta': unpacked[4],
            'rho': unpacked[5]
        }
```

## Network Optimization

### IBKR Connection

```python
class IBKRConnectionOptimized:
    def __init__(self):
        # Use multiple connections for parallel requests
        self.connections = [
            self._create_connection() for _ in range(3)
        ]
        self.current_conn = 0
        
        # Pre-compile regex patterns
        self.patterns = {
            'price': re.compile(r'PRICE:(\d+\.\d+)'),
            'size': re.compile(r'SIZE:(\d+)')
        }
        
    def request_market_data_batch(self, symbols):
        """Request data for multiple symbols in parallel"""
        # Round-robin across connections
        tasks = []
        for symbol in symbols:
            conn = self.connections[self.current_conn]
            self.current_conn = (self.current_conn + 1) % len(self.connections)
            
            tasks.append(conn.reqMktData(symbol))
        
        return tasks
```

## Profiling & Monitoring

### Performance Monitoring

```python
import time
import psutil
from prometheus_client import Histogram, Counter, Gauge

# Metrics
latency_histogram = Histogram('trading_latency_ms', 'Component latency',
                             ['component'])
trade_counter = Counter('trades_total', 'Total trades executed')
greeks_gauge = Gauge('portfolio_greeks', 'Current portfolio Greeks',
                    ['greek'])

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        
    def measure(self, component):
        """Context manager for measuring component latency"""
        class Timer:
            def __enter__(timer_self):
                timer_self.start = time.perf_counter()
                return timer_self
                
            def __exit__(timer_self, *args):
                elapsed = (time.perf_counter() - timer_self.start) * 1000
                self.timings[component] = elapsed
                latency_histogram.labels(component=component).observe(elapsed)
                
                # Alert if exceeding target
                if component == 'critical_path' and elapsed > 50:
                    logger.critical(f"Critical path latency {elapsed}ms > 50ms!")
        
        return Timer()
    
    def get_resource_usage(self):
        """Monitor system resources"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'open_files': len(psutil.Process().open_files()),
            'threads': psutil.Process().num_threads()
        }
```

## Optimization Checklist

### Pre-Deployment
- [ ] Profile all hot paths with cProfile
- [ ] Verify Greeks calculation <5ms
- [ ] Verify feature calculation <15ms
- [ ] Load test with 1000+ options contracts
- [ ] Memory usage <2GB under load
- [ ] Database query plans optimized
- [ ] Redis hit rate >80%

### Daily Monitoring
- [ ] Check p50, p95, p99 latencies
- [ ] Review slow query log
- [ ] Monitor cache hit rates
- [ ] Check memory fragmentation
- [ ] Review thread pool utilization
- [ ] Verify no memory leaks