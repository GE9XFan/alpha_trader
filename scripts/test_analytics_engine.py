#!/usr/bin/env python3
"""
Test Analytics Engine - Phase 6.2
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analytics.analytics_engine import AnalyticsEngine
import json
from datetime import datetime

def test_analytics_engine():
    print("=== Testing Analytics Engine ===\n")
    
    engine = AnalyticsEngine()
    symbol = 'SPY'
    
    # Test 1: Put/Call Ratio
    print("1. Put/Call Ratios:")
    for ratio_type in ['volume', 'open_interest', 'premium']:
        result = engine.calculate_put_call_ratio(symbol, ratio_type)
        print(f"   {ratio_type.title()}: {result['ratio']:.3f} (Put: {result['put_value']:,.0f}, Call: {result['call_value']:,.0f})")
    
    # Test 2: Gamma Exposure
    print("\n2. Gamma Exposure:")
    gex = engine.calculate_gamma_exposure(symbol)
    print(f"   Total GEX: ${gex['total_gex']:,.0f}")
    print(f"   Max Pain Strike: ${gex['max_pain_strike']:.2f}" if gex['max_pain_strike'] else "   Max Pain: N/A")
    print(f"   High GEX Signal: {gex['is_high_gex']}")
    if gex['gex_by_type']:
        for opt_type, data in gex['gex_by_type'].items():
            print(f"   {opt_type.title()}: ${data['gex']:,.0f}")
    
    # Test 3: IV Metrics
    print("\n3. Implied Volatility Metrics:")
    iv = engine.calculate_iv_metrics(symbol)
    if 'error' not in iv:
        print(f"   Current IV: {iv['current_iv']:.2%}")
        print(f"   IV Percentile: {iv['iv_percentile']:.1f}%")
        print(f"   Skew: {iv['skew']:.3f}")
        print(f"   Term Structure: {iv['term_structure']:.3f}")
        print(f"   Signals: High IV: {iv['is_high_iv']}, Low IV: {iv['is_low_iv']}")
    else:
        print(f"   Error: {iv['error']}")
    
    # Test 4: Aggregate Indicators
    print("\n4. Technical Indicators:")
    indicators = engine.aggregate_indicators(symbol)
    print(f"   Composite Score: {indicators['composite_score']:.1f}/100")
    print(f"   Signal: {indicators['signal']}")
    print("   Individual Scores:")
    for ind, score in indicators['scores'].items():
        print(f"     {ind}: {score:.1f}")
    
    # Test 5: Unusual Activity
    print("\n5. Unusual Options Activity:")
    unusual = engine.calculate_unusual_activity(symbol)
    print(f"   Unusual Contracts: {unusual['unusual_count']}")
    if unusual['unusual_options']:
        print("   Top 3 Unusual:")
        for i, opt in enumerate(unusual['unusual_options'][:3], 1):
            print(f"     {i}. Strike ${opt['strike']:.2f} {opt['option_type']} - Volume Ratio: {opt['volume_ratio']:.1f}x")
    
    # Test 6: Complete Summary
    print("\n6. Complete Analytics Summary:")
    summary = engine.generate_analytics_summary(symbol)
    print(f"   Symbol: {summary['symbol']}")
    print(f"   Signals Generated: {summary['signal_count']}")
    if summary['signals']:
        print("   Active Signals:")
        for signal in summary['signals']:
            print(f"     - {signal}")
    
    # Save full summary to file for review
    output_file = Path('data/analytics_summary.json')
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert datetime objects to strings for JSON
    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=serialize)
    
    print(f"\n   Full summary saved to: {output_file}")

if __name__ == "__main__":
    test_analytics_engine()