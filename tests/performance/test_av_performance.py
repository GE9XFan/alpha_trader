"""
Performance tests for Alpha Vantage
Tech Spec Section 10.3
"""
import pytest
import asyncio
import time

@pytest.mark.asyncio
async def test_av_performance():
    """Test Alpha Vantage API performance"""
    from src.data.alpha_vantage_client import av_client
    
    await av_client.connect()
    
    start = time.time()
    
    # Parallel API calls
    tasks = [
        av_client.get_realtime_options('SPY'),
        av_client.get_technical_indicator('SPY', 'RSI'),
        av_client.get_news_sentiment(['SPY'])
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in under 1 second
    assert all(r is not None for r in results)

@pytest.mark.asyncio
async def test_cache_performance():
    """Test caching reduces API calls"""
    from src.data.cache_manager import cache_manager
    
    # First call - not cached
    start = time.time()
    async def fetch_func():
        await asyncio.sleep(0.5)  # Simulate API call
        return {'data': 'test'}
    
    result1 = await cache_manager.get_with_cache('test_key', fetch_func)
    uncached_time = time.time() - start
    
    # Second call - should be cached
    start = time.time()
    result2 = await cache_manager.get_with_cache('test_key', fetch_func)
    cached_time = time.time() - start
    
    assert cached_time < uncached_time / 10  # 10x faster from cache
    assert result1 == result2
