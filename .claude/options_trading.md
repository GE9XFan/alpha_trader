# Options Trading Domain Knowledge

## Options Fundamentals

### Option Pricing
- **Black-Scholes Model**: For European options
- **Binomial Model**: For American options with early exercise
- **Monte Carlo**: For exotic options and path-dependent payoffs

### The Greeks

#### Delta (Δ)
- Rate of change of option price with respect to underlying price
- Call delta: 0 to 1, Put delta: -1 to 0
- Portfolio delta should be kept between -0.3 and 0.3

```python
def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """
    S: Spot price
    K: Strike price
    T: Time to expiry (years)
    r: Risk-free rate
    sigma: Implied volatility
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
```

#### Gamma (Γ)
- Rate of change of delta with respect to underlying price
- Highest for ATM options near expiry
- Portfolio gamma limit: ±0.75

#### Vega (ν)
- Sensitivity to implied volatility changes
- Highest for ATM options with more time to expiry
- Portfolio vega limit: ±1000

#### Theta (Θ)
- Time decay of option value
- Accelerates as expiry approaches
- Portfolio theta minimum: -500

#### Rho (ρ)
- Sensitivity to interest rate changes
- Less critical for short-dated options

## Critical Concepts

### VPIN (Volume-synchronized Probability of Informed Trading)
- Measures flow toxicity in options markets
- Threshold: 0.7 (above = toxic flow, close all positions)
- Calculation based on volume buckets and order imbalance

```python
def calculate_vpin(volume_buckets, n_buckets=50):
    """
    Calculate VPIN for options flow toxicity
    Critical threshold: 0.7
    """
    buy_volume = volume_buckets['buy_volume']
    sell_volume = volume_buckets['sell_volume']
    
    order_imbalance = abs(buy_volume - sell_volume)
    total_volume = buy_volume + sell_volume
    
    vpin = order_imbalance.rolling(n_buckets).sum() / total_volume.rolling(n_buckets).sum()
    return vpin
```

### 0DTE (Zero Days to Expiration)
- Options expiring same day
- **MUST close all 0DTE positions by 3:59 PM**
- High gamma risk - can move dramatically
- Pin risk at strikes near spot price

### Pin Risk
- Risk that underlying closes exactly at strike
- Uncertainty about exercise/assignment
- Manage by closing positions before expiry

### Early Exercise (American Options)
- Monitor for dividend dates
- Deep ITM puts may be exercised early
- Calculate early exercise boundary

## Options Strategies

### Single Leg
- Long/Short Calls and Puts
- Simple directional or volatility plays

### Spreads
- **Vertical**: Same expiry, different strikes
- **Calendar**: Same strike, different expiries
- **Diagonal**: Different strikes and expiries

### Complex Strategies
- **Straddle/Strangle**: Volatility plays
- **Butterfly/Condor**: Range-bound strategies
- **Iron Condor**: Premium collection with defined risk

## Market Microstructure

### Options Market Making
- Maintain bid-ask spreads
- Hedge delta continuously
- Manage inventory risk

### Order Types
- **Single Leg**: Standard limit/market orders
- **Complex**: Multi-leg with price contingencies
- **Smart Routing**: Best execution across exchanges

### Liquidity Considerations
- Prefer high open interest strikes
- Monitor bid-ask spreads
- Consider market impact for large orders

## Risk Management

### Position Sizing
```python
def calculate_position_size(account_value, kelly_fraction=0.25):
    """
    Kelly Criterion with safety factor
    Never exceed $50,000 per position
    """
    base_size = account_value * kelly_fraction * win_probability
    return min(base_size, 50000)
```

### Portfolio Greeks Management
```python
# Real-time portfolio Greeks aggregation
portfolio_greeks = {
    'delta': sum(pos.delta * pos.quantity for pos in positions),
    'gamma': sum(pos.gamma * pos.quantity for pos in positions),
    'vega': sum(pos.vega * pos.quantity for pos in positions),
    'theta': sum(pos.theta * pos.quantity for pos in positions)
}

# Check limits
assert -0.3 <= portfolio_greeks['delta'] <= 0.3
assert -0.75 <= portfolio_greeks['gamma'] <= 0.75
assert -1000 <= portfolio_greeks['vega'] <= 1000
assert portfolio_greeks['theta'] >= -500
```

### Volatility Analysis
- **IV Rank**: Current IV vs 52-week range
- **IV Percentile**: Percentage of days below current IV
- **Volatility Smile**: IV across strikes
- **Term Structure**: IV across expiries

## MOC (Market-on-Close) Trading

### MOC Window (3:40 - 4:00 PM)
- Monitor order imbalances
- Calculate imbalance signals
- Submit MOC orders by 3:55 PM
- Special handling for options expiry days

## Options Data Requirements

### From IBKR
- Real-time option chains
- Greeks (if available, else calculate)
- Open interest and volume
- Bid-ask spreads

### From Alpha Vantage
- REALTIME_OPTIONS (30-second updates)
- HISTORICAL_OPTIONS (backtesting)
- Implied volatility data
- Options flow analysis

## Performance Optimization

### Greeks Calculation
- Vectorize across entire chain
- Cache static inputs (rates, dividends)
- Use analytical formulas where possible
- Target: <5ms for 100 contracts

### Chain Processing
```python
# Efficient chain processing
def process_option_chain(chain_df):
    """Process entire chain with vectorization"""
    # Use numpy arrays
    spots = chain_df['underlying_price'].values
    strikes = chain_df['strike'].values
    
    # Vectorized Greeks
    chain_df['delta'] = np.vectorize(calculate_delta)(
        spots, strikes, expiries, rates, ivs
    )
    
    return chain_df
```

## Daily Workflow

### Pre-Market
1. Review overnight implied volatility changes
2. Check for earnings/events on positions
3. Identify 0DTE positions
4. Plan adjustment trades

### Market Hours
1. Monitor real-time Greeks
2. Track VPIN levels
3. Execute signals with proper sizing
4. Manage 0DTE positions

### Close
1. **3:59 PM**: Close all 0DTE positions
2. Review day's Greeks attribution
3. Calculate volatility P&L
4. Plan next day's trades

## Common Pitfalls to Avoid

1. **Ignoring pin risk** - Always close near-the-money 0DTE
2. **Excessive gamma** - Can cause large swings
3. **Volatility collapse** - After events/earnings
4. **Wide bid-ask** - Increases execution costs
5. **Early assignment** - Monitor ITM American options
6. **VPIN spikes** - Indicates toxic flow, exit immediately