#!/usr/bin/env python3
"""
Analytics Module - Institutional-Grade Market Analytics
Integrates microstructure, technical indicators, and options analytics
"""

from typing import Dict, Any, Optional
from loguru import logger

# Import analytics components
from .microstructure import (
    TradeSide,
    VolumeBar,
    MarketMakerProfile,
    VPINCalculator,
    HiddenOrderDetector,
    SweepDetector,
    initialize_microstructure_analytics
)

from .indicators import (
    OrderBookImbalance,
    TechnicalIndicators,
    OrderBookMetrics,
    BookPressure,
    initialize_indicators
)

from .options import (
    GammaExposureCalculator,
    GammaExposureMetrics,
    OptionsFlowMetrics,
    GammaProfile,
    initialize_options_analytics
)

__all__ = [
    # Microstructure
    'TradeSide',
    'VolumeBar',
    'MarketMakerProfile',
    'VPINCalculator',
    'HiddenOrderDetector',
    'SweepDetector',
    'initialize_microstructure_analytics',
    
    # Indicators
    'OrderBookImbalance',
    'TechnicalIndicators',
    'OrderBookMetrics',
    'BookPressure',
    'initialize_indicators',
    
    # Options
    'GammaExposureCalculator',
    'GammaExposureMetrics',
    'OptionsFlowMetrics',
    'GammaProfile',
    'initialize_options_analytics',
    
    # Main initialization
    'initialize_analytics'
]


async def initialize_analytics(cache_manager, config: Dict, av_client=None) -> Dict[str, Any]:
    """
    Initialize all analytics components with proper dependency injection
    
    Args:
        cache_manager: CacheManager instance for data storage
        config: Configuration dictionary from config.yaml
        av_client: Optional Alpha Vantage client for historical data
        
    Returns:
        Dictionary containing all initialized analytics components
    """
    try:
        logger.info("Initializing analytics module with configuration-driven architecture")
        
        # Initialize microstructure analytics
        logger.info("Initializing microstructure analytics...")
        microstructure = await initialize_microstructure_analytics(cache_manager, config)
        
        # Initialize technical indicators
        logger.info("Initializing technical indicators...")
        indicators = await initialize_indicators(cache_manager, config)
        
        # Initialize options analytics with AV client
        logger.info("Initializing options analytics with historical data support...")
        options = await initialize_options_analytics(cache_manager, config, av_client)
        
        # Combine all components
        analytics = {
            'microstructure': microstructure,
            'indicators': indicators,
            'options': options,
            'status': 'initialized',
            'config': {
                'vpin_enabled': config.get('analytics', {}).get('vpin', {}).get('enable_discovery', True),
                'vamp_enabled': config.get('analytics', {}).get('obi', {}).get('enable_vamp', True),
                'mm_discovery_enabled': config.get('analytics', {}).get('market_makers', {}).get('enable_discovery', True),
                'historical_iv_enabled': av_client is not None
            }
        }
        
        logger.info("Analytics module initialized successfully")
        logger.info(f"Configuration: VPIN Discovery={analytics['config']['vpin_enabled']}, "
                   f"VAMP={analytics['config']['vamp_enabled']}, "
                   f"MM Discovery={analytics['config']['mm_discovery_enabled']}, "
                   f"Historical IV={analytics['config']['historical_iv_enabled']}")
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to initialize analytics: {e}")
        raise


def get_analytics_metrics(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get performance metrics from all analytics components
    
    Args:
        analytics: Dictionary containing initialized analytics components
        
    Returns:
        Combined metrics from all components
    """
    metrics = {}
    
    try:
        # Get microstructure metrics
        if 'microstructure' in analytics:
            if 'vpin_calculator' in analytics['microstructure']:
                metrics['vpin'] = analytics['microstructure']['vpin_calculator'].get_metrics()
            if 'hidden_order_detector' in analytics['microstructure']:
                metrics['hidden_orders'] = analytics['microstructure']['hidden_order_detector'].get_metrics()
            if 'sweep_detector' in analytics['microstructure']:
                metrics['sweep_orders'] = analytics['microstructure']['sweep_detector'].get_metrics()
        
        # Get indicator metrics
        if 'indicators' in analytics:
            if 'obi_calculator' in analytics['indicators']:
                metrics['order_book_imbalance'] = analytics['indicators']['obi_calculator'].get_metrics()
        
        # Get options metrics
        if 'options' in analytics:
            if 'gex_calculator' in analytics['options']:
                metrics['gamma_exposure'] = analytics['options']['gex_calculator'].get_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {e}")
        return {}


def validate_analytics_config(config: Dict) -> bool:
    """
    Validate that all required analytics configuration is present
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_sections = [
        'analytics.vpin',
        'analytics.obi',
        'analytics.gex',
        'analytics.market_patterns',
        'analytics.cache_limits',
        'analytics.volatility'
    ]
    
    for section in required_sections:
        keys = section.split('.')
        current = config
        
        for key in keys:
            if key not in current:
                logger.warning(f"Missing configuration section: {section}")
                return False
            current = current[key]
    
    logger.info("Analytics configuration validated successfully")
    return True