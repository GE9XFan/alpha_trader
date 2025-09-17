"""
Redis Key Schema - Standardized keys for AlphaTrader Pro
This module defines the canonical Redis key patterns used across all modules.
"""

from typing import Optional

# Market Data Keys
MARKET_KEYS = {
    'book': 'market:{symbol}:book',          # L2 order book
    'ticker': 'market:{symbol}:ticker',      # Latest ticker/quote
    'trades': 'market:{symbol}:trades',      # Recent trades list
    'bars': 'market:{symbol}:bars',          # OHLCV bars
    'dom': 'market:{symbol}:dom',            # Depth of market snapshot
    'last': 'market:{symbol}:last',          # Cached last trade price
}

# Options Data Keys  
OPTIONS_KEYS = {
    'chain': 'options:{symbol}:chain',       # Full normalized option chain
    'greeks': 'options:{symbol}:greeks',     # Greeks hash (type:expiry:strike -> JSON)
    'gex': 'options:{symbol}:gex',          # Gamma exposure
    'dex': 'options:{symbol}:dex',          # Delta exposure  
    'pin': 'options:{symbol}:pin',          # Gamma pin levels
}

# Analytics Keys
ANALYTICS_KEYS = {
    'vpin': 'analytics:{symbol}:vpin',            # VPIN score
    'obi': 'analytics:{symbol}:obi',              # Order book imbalance
    'regime': 'analytics:{symbol}:regime',        # Market regime
    'hidden': 'analytics:{symbol}:hidden',        # Hidden order detection
    'trend': 'analytics:{symbol}:trend',          # Trend metrics
    'gex': 'analytics:{symbol}:gex',              # Gamma exposure snapshot
    'dex': 'analytics:{symbol}:dex',              # Delta exposure snapshot
    'toxicity': 'analytics:{symbol}:toxicity',    # Flow toxicity summary
    'vanna': 'analytics:{symbol}:vanna',          # Vanna exposure snapshot
    'charm': 'analytics:{symbol}:charm',          # Charm exposure snapshot
    'skew': 'analytics:{symbol}:skew',            # Intraday skew monitoring
    'hedging_impact': 'analytics:{symbol}:hedging_impact',  # Dealer hedging elasticity
    'flow_clusters': 'analytics:{symbol}:flow_clusters',    # Flow clustering output
}

VOLATILITY_KEYS = {
    'vix1d': 'analytics:volatility:vix1d',
}

# Portfolio/Sector Analytics Keys
PORTFOLIO_ANALYTICS_KEYS = {
    'summary': 'analytics:portfolio:summary',             # Portfolio level aggregates
    'correlation': 'analytics:portfolio:correlation',     # Cross-asset correlation view
    'sector': 'analytics:sector:{sector}',                # Sector level aggregates
}

# Signal Keys
SIGNAL_KEYS = {
    'pending': 'signals:pending',                      # Pending signal queue
    'execution': 'signals:execution',                  # Dedicated execution queue
    'out': 'signals:out:{symbol}:{timestamp}',        # Published signal
    'latest': 'signals:latest:{symbol}',              # Latest signal per symbol
    'fingerprint': 'signals:fingerprint:{symbol}',    # Dedup fingerprint
    'cooldown': 'signals:cooldown:{symbol}',          # Cooldown tracker
}

# Distribution Queue Keys
DISTRIBUTION_KEYS = {
    'premium': 'distribution:premium:queue',
    'basic': 'distribution:basic:queue', 
    'free': 'distribution:free:queue',
}

# Risk Management Keys
RISK_KEYS = {
    'daily_pnl': 'risk:daily_pnl',
    'daily_trades': 'risk:daily_trades',
    'consecutive_losses': 'risk:consecutive_losses',
    'drawdown_current': 'risk:drawdown:current',
    'drawdown_max': 'risk:drawdown:max',
    'halt_status': 'risk:halt:status',
    'emergency_active': 'risk:emergency:active',
    'emergency_timestamp': 'risk:emergency:timestamp',
    'var_current': 'risk:var:current',
    'correlation_matrix': 'risk:correlation:matrix',
}

# Position Management Keys
POSITION_KEYS = {
    'open': 'positions:open:{position_id}',
    'by_symbol': 'positions:by_symbol:{symbol}',
    'by_strategy': 'positions:by_strategy:{strategy}',
    'count': 'positions:count',
    'exposure_total': 'positions:exposure:total',
    'exposure_by_symbol': 'positions:exposure:{symbol}',
}

# Order Management Keys
ORDER_KEYS = {
    'pending': 'orders:pending:{order_id}',
    'active': 'orders:active:{order_id}',
    'filled': 'orders:filled:{order_id}',
    'rejected': 'orders:rejected:{order_id}',
    'by_symbol': 'orders:by_symbol:{symbol}',
}

# System & Monitoring Keys
SYSTEM_KEYS = {
    'halt': 'system:halt',                          # System-wide halt flag
    'errors_count': 'system:errors:count',
    'errors_recent': 'system:errors:recent',
    'heartbeat': 'module:heartbeat:{module}',       # Module heartbeats
    'metrics': 'monitoring:{module}:metrics',       # Module metrics
    'status': 'monitoring:{module}:status',         # Module status
}

# Circuit Breaker Keys
BREAKER_KEYS = {
    'status': 'circuit_breakers:status',
    'daily_loss': 'circuit_breakers:daily_loss:triggered',
    'drawdown': 'circuit_breakers:drawdown:triggered',
    'consecutive': 'circuit_breakers:consecutive:triggered',
    'volatility': 'circuit_breakers:volatility:triggered',
    'errors': 'circuit_breakers:errors:triggered',
    'position': 'circuit_breakers:position:triggered',
}

# Account Keys
ACCOUNT_KEYS = {
    'value': 'account:value',
    'cash': 'account:cash',
    'margin_used': 'account:margin:used',
    'margin_available': 'account:margin:available',
    'buying_power': 'account:buying_power',
}

# TTL Configuration (seconds)
TTL_CONFIG = {
    # Market data - balanced for analytics needs
    'market_book': 60,         # Keep order books for 1 minute
    'market_ticker': 60,       # Keep tickers for 1 minute
    'market_trades': 3600,     # Keep trades for 1 hour (VPIN needs 1000+)
    'market_bars': 3600,       # Keep bars for 1 hour (analytics needs 100+)
    
    # Options data - moderate TTLs
    'options_chain': 60,       # Update chain every minute
    'options_greeks': 60,      # Greeks for 1 minute
    'options_exposures': 60,   # GEX/DEX for 1 minute
    
    # Analytics - moderate TTLs
    'analytics': 60,               # Symbol-level analytics data
    'analytics_portfolio': 60,     # Portfolio summary cadence
    'analytics_sector': 120,       # Sector aggregates decay slower
    'analytics_correlation': 900,  # Correlation matrices update less often
    
    # Signals - longer TTLs for audit
    'signal_out': 3600,        # Keep published signals 1 hour
    'signal_latest': 300,      # Latest signal for 5 mins
    'signal_cooldown': 60,     # Cooldown window
    
    # Risk & positions - persistent
    'risk_metrics': 86400,     # Daily metrics
    'positions': None,         # No TTL - manual cleanup
    'orders': 3600,           # Keep order history 1 hour
    
    # System
    'heartbeat': 15,          # Heartbeat timeout
    'metrics': 60,            # Metrics refresh
}

def get_market_key(symbol: str, data_type: str) -> str:
    """Get market data Redis key for symbol."""
    if data_type not in MARKET_KEYS:
        raise ValueError(f"Unknown market data type: {data_type}")
    return MARKET_KEYS[data_type].format(symbol=symbol)

def get_options_key(symbol: str, data_type: str) -> str:
    """Get options data Redis key for symbol."""
    if data_type not in OPTIONS_KEYS:
        raise ValueError(f"Unknown options data type: {data_type}")
    return OPTIONS_KEYS[data_type].format(symbol=symbol)

def get_signal_key(key_type: str, **kwargs) -> str:
    """Get signal Redis key with parameters."""
    if key_type not in SIGNAL_KEYS:
        raise ValueError(f"Unknown signal key type: {key_type}")
    return SIGNAL_KEYS[key_type].format(**kwargs)

def get_position_key(key_type: str, **kwargs) -> str:
    """Get position Redis key with parameters."""
    if key_type not in POSITION_KEYS:
        raise ValueError(f"Unknown position key type: {key_type}")
    return POSITION_KEYS[key_type].format(**kwargs)

def get_order_key(key_type: str, **kwargs) -> str:
    """Get order Redis key with parameters."""
    if key_type not in ORDER_KEYS:
        raise ValueError(f"Unknown order key type: {key_type}")
    return ORDER_KEYS[key_type].format(**kwargs)

def get_system_key(key_type: str, **kwargs) -> str:
    """Get system/monitoring Redis key."""
    if key_type not in SYSTEM_KEYS:
        raise ValueError(f"Unknown system key type: {key_type}")
    return SYSTEM_KEYS[key_type].format(**kwargs)

def get_portfolio_key(key_type: str, **kwargs) -> str:
    """Get portfolio/sector analytics Redis key."""
    if key_type not in PORTFOLIO_ANALYTICS_KEYS:
        raise ValueError(f"Unknown portfolio analytics key type: {key_type}")
    return PORTFOLIO_ANALYTICS_KEYS[key_type].format(**kwargs)

def get_ttl(data_type: str) -> int:
    """Get TTL for data type in seconds."""
    return TTL_CONFIG.get(data_type, 60)  # Default 60s


# Convenience helpers -----------------------------------------------------------------

def market_book_key(symbol: str, exchange: Optional[str] = None) -> str:
    """Return the order-book key for the symbol, optionally scoped to an exchange."""
    if exchange:
        return f"market:{symbol}:{exchange}:book"
    return get_market_key(symbol, "book")


def market_ticker_key(symbol: str) -> str:
    """Return the consolidated ticker key for a symbol."""
    return get_market_key(symbol, "ticker")


def market_last_key(symbol: str) -> str:
    """Return the cached last trade price key."""
    return get_market_key(symbol, "last")


def market_trades_key(symbol: str) -> str:
    """Return the rolling trades list key."""
    return get_market_key(symbol, "trades")


def market_bars_key(symbol: str, timeframe: Optional[str] = None) -> str:
    """Return the OHLCV bars key, optionally namespaced by timeframe."""
    base = get_market_key(symbol, "bars")
    if timeframe:
        return f"{base}:{timeframe}"
    return base


def market_dom_key(symbol: str) -> str:
    """Return the depth of market snapshot key."""
    return get_market_key(symbol, "dom")


def options_chain_key(symbol: str) -> str:
    """Return the normalized option chain key."""
    return get_options_key(symbol, "chain")


def options_calls_key(symbol: str) -> str:
    """Return the cached calls slice key."""
    return f"options:{symbol}:calls"


def options_puts_key(symbol: str) -> str:
    """Return the cached puts slice key."""
    return f"options:{symbol}:puts"


def options_greeks_key(symbol: str) -> str:
    """Return the option Greeks hash key."""
    return get_options_key(symbol, "greeks")


def analytics_vpin_key(symbol: str) -> str:
    """Return the VPIN analytics key."""
    return ANALYTICS_KEYS["vpin"].format(symbol=symbol)


def analytics_gex_key(symbol: str) -> str:
    """Return the gamma exposure analytics key."""
    return ANALYTICS_KEYS["gex"].format(symbol=symbol)


def analytics_dex_key(symbol: str) -> str:
    """Return the delta exposure analytics key."""
    return ANALYTICS_KEYS["dex"].format(symbol=symbol)


def analytics_toxicity_key(symbol: str) -> str:
    """Return the flow toxicity analytics key."""
    return ANALYTICS_KEYS["toxicity"].format(symbol=symbol)


def analytics_obi_key(symbol: str) -> str:
    """Return the order-book imbalance analytics key."""
    return ANALYTICS_KEYS["obi"].format(symbol=symbol)


def analytics_vanna_key(symbol: str) -> str:
    """Return the dealer Vanna exposure key."""
    return ANALYTICS_KEYS["vanna"].format(symbol=symbol)


def analytics_charm_key(symbol: str) -> str:
    """Return the dealer Charm exposure key."""
    return ANALYTICS_KEYS["charm"].format(symbol=symbol)


def analytics_skew_key(symbol: str) -> str:
    """Return the intraday skew analytics key."""
    return ANALYTICS_KEYS["skew"].format(symbol=symbol)


def analytics_hedging_impact_key(symbol: str) -> str:
    """Return the dynamic hedging impact analytics key."""
    return ANALYTICS_KEYS["hedging_impact"].format(symbol=symbol)


def analytics_flow_clusters_key(symbol: str) -> str:
    """Return the flow clustering analytics key."""
    return ANALYTICS_KEYS["flow_clusters"].format(symbol=symbol)


def analytics_vix1d_key() -> str:
    """Return the global VIX1D analytics key."""
    return VOLATILITY_KEYS["vix1d"]


def analytics_metric_key(symbol: str, metric: str) -> str:
    """Return an arbitrary analytics metric key."""
    return f"analytics:{symbol}:{metric}"


def analytics_portfolio_summary_key() -> str:
    """Return the portfolio summary key."""
    return PORTFOLIO_ANALYTICS_KEYS["summary"]


def analytics_portfolio_correlation_key() -> str:
    """Return the portfolio correlation key."""
    return PORTFOLIO_ANALYTICS_KEYS["correlation"]


def analytics_sector_key(sector: str) -> str:
    """Return the sector analytics key for the supplied sector."""
    return PORTFOLIO_ANALYTICS_KEYS["sector"].format(sector=sector)


def heartbeat_key(module: str) -> str:
    """Return the heartbeat key for a module."""
    return get_system_key("heartbeat", module=module)
