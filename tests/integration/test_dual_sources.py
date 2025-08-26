"""
Integration tests for dual data sources
Tech Spec Section 10.2
"""
import pytest
import asyncio

@pytest.mark.asyncio
async def test_dual_data_sources():
    """Test IBKR and Alpha Vantage integration"""
    from src.data.market_data import market_data
    from src.data.alpha_vantage_client import av_client
    
    # Initialize components
    await market_data.connect()  # IBKR for quotes
    await av_client.connect()    # Alpha Vantage for options
    
    # Subscribe to market data
    await market_data.subscribe_symbols(['SPY'])
    
    # Wait for data
    await asyncio.sleep(10)
    
    # Verify IBKR data flowing
    assert market_data.get_latest_price('SPY') > 0
    
    # Test Alpha Vantage options data
    options = await av_client.get_realtime_options('SPY', require_greeks=True)
    assert len(options) > 0
    
    # Verify Greeks are provided, not calculated
    first_option = options[0]
    assert hasattr(first_option, 'delta')
    assert first_option.delta is not None

@pytest.mark.asyncio
async def test_signal_generation_pipeline():
    """Test complete signal generation pipeline"""
    from src.data.market_data import market_data
    from src.data.alpha_vantage_client import av_client
    from src.trading.signals import signal_generator
    
    # Setup
    await market_data.connect()  # IBKR
    await av_client.connect()    # Alpha Vantage
    
    # Generate signal
    signals = await signal_generator.generate_signals(['SPY'])
    
    # Verify signal has all required data
    if signals:
        assert signals[0]['av_greeks'] is not None
        assert signals[0]['confidence'] > 0
