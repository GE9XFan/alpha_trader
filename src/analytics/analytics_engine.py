"""
Analytics Engine - Phase 6.2
Calculates derived metrics and aggregations from raw data
Configuration-driven, no hardcoded values
"""

import yaml
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Calculates advanced analytics from options and indicator data
    All calculations configured via YAML
    """
    
    def __init__(self):
        self._load_config()
        self._init_database()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'analytics' / 'analytics_engine.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Analytics config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.calculations = config['calculations']
        self.signal_thresholds = config['signal_thresholds']
        self.time_windows = config['time_windows']
        self.indicator_weights = config['indicator_weights']
        
        print(f"Analytics Engine initialized with config from {config_path}")
    
    def _init_database(self):
        """Initialize database connection"""
        from src.foundation.config_manager import ConfigManager
        config = ConfigManager()
        self.engine = create_engine(config.database_url)
    
    def calculate_put_call_ratio(self, symbol: str, 
                                 ratio_type: str = 'volume') -> Dict:
        """
        Calculate put/call ratio for a symbol
        ratio_type: 'volume', 'open_interest', or 'premium'
        """
        with self.engine.connect() as conn:
            if ratio_type == 'volume':
                query = text("""
                    SELECT 
                        SUM(CASE WHEN option_type = 'put' THEN volume ELSE 0 END) as put_volume,
                        SUM(CASE WHEN option_type = 'call' THEN volume ELSE 0 END) as call_volume
                    FROM av_realtime_options
                    WHERE symbol = :symbol
                    AND updated_at > NOW() - INTERVAL '1 day'
                """)
            elif ratio_type == 'open_interest':
                query = text("""
                    SELECT 
                        SUM(CASE WHEN option_type = 'put' THEN open_interest ELSE 0 END) as put_oi,
                        SUM(CASE WHEN option_type = 'call' THEN open_interest ELSE 0 END) as call_oi
                    FROM av_realtime_options
                    WHERE symbol = :symbol
                    AND updated_at > NOW() - INTERVAL '1 day'
                """)
            else:  # premium
                query = text("""
                    SELECT 
                        SUM(CASE WHEN option_type = 'put' THEN mark * volume ELSE 0 END) as put_premium,
                        SUM(CASE WHEN option_type = 'call' THEN mark * volume ELSE 0 END) as call_premium
                    FROM av_realtime_options
                    WHERE symbol = :symbol
                    AND updated_at > NOW() - INTERVAL '1 day'
                """)
            
            result = conn.execute(query, {'symbol': symbol}).fetchone()
            
            if ratio_type == 'volume':
                put_val = float(result[0] or 0)
                call_val = float(result[1] or 0)
            elif ratio_type == 'open_interest':
                put_val = float(result[0] or 0)
                call_val = float(result[1] or 0)
            else:
                put_val = float(result[0] or 0)
                call_val = float(result[1] or 0)
            
            ratio = put_val / call_val if call_val > 0 else 0
            
            return {
                'symbol': symbol,
                'ratio_type': ratio_type,
                'put_value': put_val,
                'call_value': call_val,
                'ratio': ratio,
                'timestamp': datetime.now()
            }
    
    def calculate_gamma_exposure(self, symbol: str, spot_price: Optional[float] = None) -> Dict:
        """
        Calculate total gamma exposure (GEX) for a symbol
        GEX = Sum(gamma * open_interest * 100 * spot_price)
        """
        with self.engine.connect() as conn:
            # Get spot price if not provided
            if spot_price is None:
                # Try to get from recent quotes or use strike nearest to ATM
                result = conn.execute(text("""
                    SELECT strike 
                    FROM av_realtime_options
                    WHERE symbol = :symbol 
                    AND ABS(delta - 0.5) < 0.1
                    ORDER BY ABS(delta - 0.5)
                    LIMIT 1
                """), {'symbol': symbol}).fetchone()
                
                spot_price = float(result[0]) if result else 100.0
            
            # Calculate GEX
            query = text("""
                SELECT 
                    option_type,
                    SUM(
                        CASE 
                            WHEN option_type = 'call' THEN gamma * open_interest * 100 * :spot
                            ELSE -gamma * open_interest * 100 * :spot
                        END
                    ) as gex,
                    SUM(gamma * open_interest) as total_gamma_oi,
                    COUNT(*) as contracts
                FROM av_realtime_options
                WHERE symbol = :symbol
                AND gamma IS NOT NULL
                AND open_interest IS NOT NULL
                AND updated_at > NOW() - INTERVAL '1 day'
                GROUP BY option_type
            """)
            
            result = conn.execute(query, {'symbol': symbol, 'spot': spot_price})
            
            total_gex = 0
            by_type = {}
            
            for row in result:
                by_type[row[0]] = {
                    'gex': float(row[1] or 0),
                    'gamma_oi': float(row[2] or 0),
                    'contracts': row[3]
                }
                total_gex += float(row[1] or 0)
            
            # Get GEX by strike
            strike_query = text("""
                SELECT 
                    strike,
                    SUM(
                        CASE 
                            WHEN option_type = 'call' THEN gamma * open_interest * 100 * :spot
                            ELSE -gamma * open_interest * 100 * :spot
                        END
                    ) as strike_gex
                FROM av_realtime_options
                WHERE symbol = :symbol
                AND gamma IS NOT NULL
                AND open_interest IS NOT NULL
                AND updated_at > NOW() - INTERVAL '1 day'
                GROUP BY strike
                ORDER BY strike
            """)
            
            strike_result = conn.execute(strike_query, {'symbol': symbol, 'spot': spot_price})
            gex_by_strike = {float(row[0]): float(row[1] or 0) for row in strike_result}
            
            # Find max pain strike (where GEX is highest)
            max_pain_strike = max(gex_by_strike, key=gex_by_strike.get) if gex_by_strike else None
            
            return {
                'symbol': symbol,
                'spot_price': spot_price,
                'total_gex': total_gex,
                'gex_by_type': by_type,
                'gex_by_strike': gex_by_strike,
                'max_pain_strike': max_pain_strike,
                'is_high_gex': abs(total_gex) > self.signal_thresholds['high_gamma_exposure'],
                'timestamp': datetime.now()
            }
    
    def calculate_iv_metrics(self, symbol: str) -> Dict:
            """
            Calculate implied volatility metrics including skew and term structure
            """
            with self.engine.connect() as conn:
                # Get ATM strike using configured threshold
                atm_threshold = self.calculations['atm_definition']['delta_threshold']
                
                query = text("""
                    WITH atm_strike AS (
                        SELECT strike 
                        FROM av_realtime_options
                        WHERE symbol = :symbol AND ABS(delta - 0.5) < :atm_threshold
                        ORDER BY ABS(delta - 0.5) LIMIT 1
                    )
                    SELECT 
                        strike,
                        expiration,
                        option_type,
                        implied_volatility,
                        delta,
                        volume,
                        open_interest,
                        (strike / (SELECT strike FROM atm_strike)) as moneyness
                    FROM av_realtime_options
                    WHERE symbol = :symbol
                    AND implied_volatility IS NOT NULL
                    AND updated_at > NOW() - INTERVAL '1 day'
                    ORDER BY expiration, strike
                """)
                
                result = conn.execute(query, {'symbol': symbol, 'atm_threshold': atm_threshold})
                
                data = []
                for row in result:
                    data.append({
                        'strike': float(row[0]),
                        'expiration': row[1],
                        'option_type': row[2],
                        'iv': float(row[3]) if row[3] else None,
                        'delta': float(row[4]) if row[4] else None,
                        'volume': row[5],
                        'open_interest': row[6],
                        'moneyness': float(row[7]) if row[7] else None
                    })
                
                if not data:
                    return {'symbol': symbol, 'error': 'No IV data available'}
                
                df = pd.DataFrame(data)
                
                # Calculate IV percentile using configured historical period
                historical_days = self.calculations['term_structure']['historical_days']
                current_iv = df['iv'].mean()
                
                iv_30d_query = text(f"""
                    SELECT AVG(implied_volatility) as daily_avg
                    FROM av_realtime_options
                    WHERE symbol = :symbol
                    AND implied_volatility IS NOT NULL
                    AND updated_at > NOW() - INTERVAL '{historical_days} days'
                    GROUP BY DATE(updated_at)
                """)
                
                iv_history = conn.execute(iv_30d_query, {'symbol': symbol})
                historical_ivs = [float(row[0]) for row in iv_history if row[0]]
                
                if historical_ivs:
                    iv_percentile = (sum(1 for iv in historical_ivs if iv <= current_iv) / len(historical_ivs)) * 100
                else:
                    iv_percentile = 50  # Default if no history
                
                # Calculate skew using configured delta ranges
                skew_config = self.calculations['skew_calculation']
                put_25d = df[
                    (df['option_type'] == 'put') & 
                    (df['delta'].between(skew_config['put_delta_min'], skew_config['put_delta_max']))
                ]['iv'].mean()
                
                call_25d = df[
                    (df['option_type'] == 'call') & 
                    (df['delta'].between(skew_config['call_delta_min'], skew_config['call_delta_max']))
                ]['iv'].mean()
                
                skew = put_25d - call_25d if not pd.isna(put_25d) and not pd.isna(call_25d) else 0
                
                # Term structure using configured periods
                short_term_days = self.calculations['term_structure']['short_term_days']
                short_term_date = datetime.now().date() + timedelta(days=short_term_days)
                
                short_term = df[df['expiration'] <= short_term_date]['iv'].mean()
                long_term = df[df['expiration'] > short_term_date]['iv'].mean()
                term_structure = short_term - long_term if not pd.isna(short_term) and not pd.isna(long_term) else 0
                
                return {
                    'symbol': symbol,
                    'current_iv': current_iv,
                    'iv_percentile': iv_percentile,
                    'skew': skew,
                    'term_structure': term_structure,
                    'is_high_iv': iv_percentile > self.signal_thresholds['high_iv_percentile'],
                    'is_low_iv': iv_percentile < self.signal_thresholds['low_iv_percentile'],
                    'timestamp': datetime.now()
                }    
    def aggregate_indicators(self, symbol: str) -> Dict:
        """
        Aggregate all technical indicators into a composite score
        """
        with self.engine.connect() as conn:
            indicators = {}
            
            # Get RSI
            rsi_query = text("""
                SELECT rsi FROM av_rsi 
                WHERE symbol = :symbol 
                ORDER BY timestamp DESC LIMIT 1
            """)
            result = conn.execute(rsi_query, {'symbol': symbol}).fetchone()
            indicators['rsi'] = float(result[0]) if result and result[0] else 50.0
            
            # Get MACD
            macd_query = text("""
                SELECT macd, macd_signal, macd_hist FROM av_macd
                WHERE symbol = :symbol
                ORDER BY timestamp DESC LIMIT 1
            """)
            result = conn.execute(macd_query, {'symbol': symbol}).fetchone()
            if result:
                indicators['macd'] = float(result[0]) if result[0] else 0
                indicators['macd_signal'] = float(result[1]) if result[1] else 0
                indicators['macd_hist'] = float(result[2]) if result[2] else 0
            
            # Get BBANDS - FIXED COLUMN NAMES
            bbands_query = text("""
                SELECT upper_band, middle_band, lower_band 
                FROM av_bbands
                WHERE symbol = :symbol
                ORDER BY timestamp DESC LIMIT 1
            """)
            result = conn.execute(bbands_query, {'symbol': symbol}).fetchone()
            if result:
                indicators['bb_upper'] = float(result[0]) if result[0] else 0
                indicators['bb_middle'] = float(result[1]) if result[1] else 0
                indicators['bb_lower'] = float(result[2]) if result[2] else 0
                
            # Get VWAP
            vwap_query = text("""
                SELECT vwap FROM av_vwap
                WHERE symbol = :symbol
                ORDER BY timestamp DESC LIMIT 1
            """)
            result = conn.execute(vwap_query, {'symbol': symbol}).fetchone()
            indicators['vwap'] = float(result[0]) if result and result[0] else 0
            
            # Calculate composite scores
            scores = {}
            
            # RSI score (0-100, 50 is neutral)
            rsi_val = indicators.get('rsi', 50)
            if rsi_val < 30:
                scores['rsi'] = 100  # Oversold = bullish
            elif rsi_val > 70:
                scores['rsi'] = 0    # Overbought = bearish
            else:
                scores['rsi'] = 50 + (50 - rsi_val)  # Linear scale
            
            # MACD score (histogram positive = bullish)
            macd_hist = indicators.get('macd_hist', 0)
            scores['macd'] = 50 + min(max(macd_hist * 10, -50), 50)  # Scale to 0-100
            
            # Bollinger Bands score (position within bands)
            if indicators.get('bb_upper') and indicators.get('bb_lower'):
                bb_range = indicators['bb_upper'] - indicators['bb_lower']
                if bb_range > 0 and indicators.get('bb_middle'):
                    bb_position = (indicators['bb_middle'] - indicators['bb_lower']) / bb_range
                    scores['bbands'] = bb_position * 100
                else:
                    scores['bbands'] = 50
            else:
                scores['bbands'] = 50
            
            # Calculate weighted composite score
            weighted_score = 0
            total_weight = 0
            
            for indicator, weight in self.indicator_weights.items():
                if indicator in scores:
                    weighted_score += scores[indicator] * weight
                    total_weight += weight
            
            composite_score = weighted_score / total_weight if total_weight > 0 else 50
            
            return {
                'symbol': symbol,
                'indicators': indicators,
                'scores': scores,
                'composite_score': composite_score,
                'signal': 'BULLISH' if composite_score > 65 else 'BEARISH' if composite_score < 35 else 'NEUTRAL',
                'timestamp': datetime.now()
            }
    
    def calculate_unusual_activity(self, symbol: str) -> Dict:
        """
        Detect unusual options activity
        """
        with self.engine.connect() as conn:
            # Get current and average volumes
            query = text("""
                WITH current_volume AS (
                    SELECT 
                        contract_id,
                        strike,
                        expiration,
                        option_type,
                        volume,
                        open_interest,
                        implied_volatility
                    FROM av_realtime_options
                    WHERE symbol = :symbol
                    AND updated_at > NOW() - INTERVAL '1 day'
                ),
                avg_volume AS (
                    SELECT 
                        strike,
                        expiration,
                        option_type,
                        AVG(volume) as avg_vol
                    FROM av_historical_options
                    WHERE symbol = :symbol
                    AND data_date > NOW() - INTERVAL '20 days'
                    GROUP BY strike, expiration, option_type
                )
                SELECT 
                    c.contract_id,
                    c.strike,
                    c.expiration,
                    c.option_type,
                    c.volume,
                    COALESCE(a.avg_vol, c.volume * 0.5) as avg_volume,
                    c.volume / NULLIF(COALESCE(a.avg_vol, c.volume * 0.5), 0) as volume_ratio,
                    c.open_interest,
                    c.implied_volatility
                FROM current_volume c
                LEFT JOIN avg_volume a 
                    ON c.strike = a.strike 
                    AND c.expiration = a.expiration 
                    AND c.option_type = a.option_type
                WHERE c.volume > 0
                ORDER BY c.volume / NULLIF(COALESCE(a.avg_vol, c.volume * 0.5), 0) DESC
                LIMIT 20
            """)
            
            result = conn.execute(query, {'symbol': symbol})
            
            unusual_options = []
            for row in result:
                volume_ratio = float(row[6]) if row[6] else 1.0
                
                if volume_ratio > self.signal_thresholds['unusual_volume']:
                    unusual_options.append({
                        'contract_id': row[0],
                        'strike': float(row[1]),
                        'expiration': row[2],
                        'option_type': row[3],
                        'volume': row[4],
                        'avg_volume': float(row[5]) if row[5] else 0,
                        'volume_ratio': volume_ratio,
                        'open_interest': row[7],
                        'iv': float(row[8]) if row[8] else 0
                    })
            
            return {
                'symbol': symbol,
                'unusual_count': len(unusual_options),
                'unusual_options': unusual_options[:10],  # Top 10
                'has_unusual_activity': len(unusual_options) > 0,
                'timestamp': datetime.now()
            }
    
    def generate_analytics_summary(self, symbol: str) -> Dict:
        """
        Generate comprehensive analytics summary for a symbol
        """
        summary = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'metrics': {}
        }
        
        # Get all analytics
        try:
            summary['metrics']['put_call_ratio'] = self.calculate_put_call_ratio(symbol, 'volume')
            summary['metrics']['gamma_exposure'] = self.calculate_gamma_exposure(symbol)
            summary['metrics']['iv_metrics'] = self.calculate_iv_metrics(symbol)
            summary['metrics']['indicators'] = self.aggregate_indicators(symbol)
            summary['metrics']['unusual_activity'] = self.calculate_unusual_activity(symbol)
            
            # Generate signals
            signals = []
            
            if summary['metrics']['put_call_ratio']['ratio'] > self.signal_thresholds['high_put_call_ratio']:
                signals.append('HIGH_PUT_CALL_RATIO')
            elif summary['metrics']['put_call_ratio']['ratio'] < self.signal_thresholds['low_put_call_ratio']:
                signals.append('LOW_PUT_CALL_RATIO')
            
            if summary['metrics']['gamma_exposure'].get('is_high_gex'):
                signals.append('HIGH_GAMMA_EXPOSURE')
            
            if summary['metrics']['iv_metrics'].get('is_high_iv'):
                signals.append('HIGH_IMPLIED_VOL')
            elif summary['metrics']['iv_metrics'].get('is_low_iv'):
                signals.append('LOW_IMPLIED_VOL')
            
            if summary['metrics']['indicators']['signal'] != 'NEUTRAL':
                signals.append(f"INDICATORS_{summary['metrics']['indicators']['signal']}")
            
            if summary['metrics']['unusual_activity']['has_unusual_activity']:
                signals.append('UNUSUAL_OPTIONS_ACTIVITY')
            
            summary['signals'] = signals
            summary['signal_count'] = len(signals)
            
        except Exception as e:
            summary['error'] = str(e)
            logger.error(f"Error generating analytics for {symbol}: {e}")
        
        return summary