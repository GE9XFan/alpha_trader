# Alternative Toxicity Detection Approach

## Since IBKR doesn't expose real market maker identities, use these signals instead:

### 1. VPIN Itself (Primary Toxicity Measure)
- VPIN > 0.7 = Toxic flow likely
- VPIN < 0.3 = Informed/institutional flow
- This is what VPIN was designed for!

### 2. Venue-Based Toxicity Scoring
```python
venue_toxicity_scores = {
    # Retail-heavy venues (more toxic)
    "EDGX": 0.8,  # Retail order flow
    "BATS": 0.7,  # Mixed retail/HFT
    
    # Institutional venues (less toxic)
    "IEX": 0.2,   # Speed bump protects from HFT
    "ARCA": 0.4,  # NYSE institutional
    "NSDQ": 0.5,  # NASDAQ main
    
    # Dark/Hidden (unknown toxicity)
    "DARK": 0.5,
    "IBKRATS": 0.6,  # IB internal
}
```

### 3. Trade Pattern Analysis
```python
def calculate_trade_toxicity(trades):
    # Odd lots (100-900 shares) = likely retail
    odd_lot_ratio = sum(1 for t in trades if t.size % 100 != 0) / len(trades)
    
    # Rapid same-price trades = sweep/aggressive
    sweep_ratio = detect_sweeps(trades) / len(trades)
    
    # Large blocks at round prices = institutional
    block_ratio = sum(1 for t in trades if t.size >= 10000) / len(trades)
    
    toxicity = (odd_lot_ratio * 0.7 + 
                sweep_ratio * 0.9 - 
                block_ratio * 0.5)
    return max(0, min(1, toxicity))
```

### 4. Order Book Imbalance Patterns
```python
def detect_toxic_book_patterns(book_history):
    # Rapid bid pulling before sells = toxic
    # Persistent ask stacking = accumulation
    # Balanced book = market making
    
    imbalances = []
    for book in book_history:
        bid_volume = sum(b['size'] for b in book['bids'][:5])
        ask_volume = sum(a['size'] for a in book['asks'][:5])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        imbalances.append(imbalance)
    
    # High variance in imbalance = toxic/manipulative
    return np.std(imbalances)
```

### 5. Time-of-Day Patterns
```python
toxic_times = {
    "09:30-09:45": 0.8,  # Opening volatility
    "10:00-10:30": 0.5,  # Post-open equilibrium  
    "14:30-15:00": 0.4,  # Institutional positioning
    "15:45-16:00": 0.7,  # Close/MOC imbalances
}
```

## Implementation Priority

1. **Use VPIN as primary signal** - It already measures toxicity!
2. **Add venue-based scoring** from tick data exchange tags
3. **Track sweep patterns** from rapid multi-level takes
4. **Monitor odd-lot ratios** as retail flow proxy
5. **Calculate book imbalance variance** for manipulation detection

## Why This Works Better

- **VPIN** directly measures information asymmetry
- **Venue patterns** are consistent and observable
- **Trade sizes** reveal participant types
- **Sweep detection** catches aggressive toxic flow
- **No dependency** on unavailable MM identities

## Config Update Suggestion

```yaml
parameter_discovery:
  toxicity_detection:
    method: "pattern_based"  # not "market_maker_based"
    
    venue_scores:
      EDGX: 0.8
      BATS: 0.7
      NSDQ: 0.5
      ARCA: 0.4
      IEX: 0.2
      
    trade_patterns:
      odd_lot_weight: 0.7
      sweep_weight: 0.9
      block_weight: -0.5
      
    vpin_thresholds:
      toxic: 0.7
      neutral: 0.5
      informed: 0.3
```

This approach actually measures toxic BEHAVIOR rather than trying to identify specific firms that we can't see anyway.