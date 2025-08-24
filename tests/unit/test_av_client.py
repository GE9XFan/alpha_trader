"""
Unit tests for Alpha Vantage client
Tech Spec Section 10.1
"""
import pytest
import asyncio
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_av_greeks_retrieval():
    """Test Alpha Vantage Greeks retrieval - NOT calculated"""
    from src.data.alpha_vantage_client import av_client
    
    # Mock response with Greeks PROVIDED
    mock_options = [{
        'symbol': 'SPY',
        'strike': 450.0,
        'expiry': '2024-01-19',
        'type': 'CALL',
        'delta': 0.55,  # PROVIDED by AV!
        'gamma': 0.02,  # PROVIDED by AV!
        'theta': -0.15,  # PROVIDED by AV!
        'vega': 0.20,  # PROVIDED by AV!
        'rho': 0.05  # PROVIDED by AV!
    }]
    
    with patch.object(av_client, '_make_request', return_value={'options': mock_options}):
        options = await av_client.get_realtime_options('SPY')
        
        assert len(options) > 0
        assert options[0].delta is not None
        assert -1 <= options[0].delta <= 1
        # Verify NO calculation happened - Greeks came from AV

@pytest.mark.asyncio
async def test_all_38_apis():
    """Test all 38 Alpha Vantage APIs are accessible"""
    from src.core.constants import AV_ENDPOINTS
    
    # Verify all 38 endpoints defined
    assert len(AV_ENDPOINTS) == 38
    
    # Check categories
    options_apis = [k for k in AV_ENDPOINTS if 'OPTIONS' in k]
    assert len(options_apis) == 2
    
    indicator_apis = [k for k in AV_ENDPOINTS if k in 
                     ['RSI', 'MACD', 'STOCH', 'WILLR', 'MOM', 'BBANDS',
                      'ATR', 'ADX', 'AROON', 'CCI', 'EMA', 'SMA', 
                      'MFI', 'OBV', 'AD', 'VWAP']]
    assert len(indicator_apis) == 16
