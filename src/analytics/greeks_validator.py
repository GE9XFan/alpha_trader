"""
Greeks Validator - Phase 6.1
Validates option Greeks for data quality and theoretical bounds
Configuration-driven, no hardcoded values
"""

import yaml
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class GreeksValidator:
    """
    Validates Greeks values from Alpha Vantage options data
    All validation rules loaded from configuration
    """
    
    def __init__(self):
        # Load validation rules from config
        self._load_config()
    
    def _load_config(self):
        """Load validation rules from configuration file"""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'analytics' / 'greeks_validation.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Greeks validation config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.validation_rules = config['validation_rules']
        self.freshness = config['freshness']
        self.consistency_checks = config['consistency_checks']
        
        print(f"Greeks Validator initialized with config from {config_path}")
    
    def validate_single_greek(self, greek_name: str, value: Optional[float], 
                            option_type: str = 'call') -> Tuple[bool, Optional[str]]:
        """
        Validate a single Greek value against configured rules
        Returns: (is_valid, error_message)
        """
        # Check for None
        if value is None:
            return False, f"{greek_name} is NULL"
        
        # Convert to float if Decimal
        if isinstance(value, Decimal):
            value = float(value)
        
        # Get rules for this Greek
        if greek_name not in self.validation_rules:
            return True, None  # No rules defined, assume valid
        
        rules = self.validation_rules[greek_name]
        
        # Special handling for delta (option type dependent)
        if greek_name == 'delta':
            if option_type in rules:
                limits = rules[option_type]
                if value < limits['min'] or value > limits['max']:
                    return False, f"Delta {value:.4f} outside {option_type} bounds [{limits['min']}, {limits['max']}]"
        else:
            # Check min/max for other Greeks
            if 'min' in rules and value < rules['min']:
                return False, f"{greek_name} {value:.4f} below minimum {rules['min']}"
            if 'max' in rules and value > rules['max']:
                return False, f"{greek_name} {value:.4f} above maximum {rules['max']}"
        
        return True, None
    
    def validate_greeks_set(self, greeks: Dict, option_type: str = 'call',
                          timestamp: Optional[datetime] = None) -> Dict:
        """
        Validate a complete set of Greeks
        Returns dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'values': {}
        }
        
        # Check freshness if timestamp provided
        if timestamp:
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            if age_minutes > self.freshness['critical_staleness_minutes']:
                results['valid'] = False
                results['errors'].append(f"Data critically stale: {age_minutes:.1f} minutes old")
            elif age_minutes > self.freshness['staleness_minutes']:
                results['warnings'].append(f"Data is {age_minutes:.1f} minutes old")
        
        # Validate each Greek
        for greek_name in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            if greek_name in greeks:
                value = greeks[greek_name]
                results['values'][greek_name] = value
                
                is_valid, error_msg = self.validate_single_greek(
                    greek_name, value, option_type
                )
                
                if not is_valid:
                    results['valid'] = False
                    results['errors'].append(error_msg)
                
                # Check for warnings based on config
                self._check_warnings(greek_name, value, results)
        
        # Cross-validation checks
        self._check_consistency(results, option_type)
        
        return results
    
    def _check_warnings(self, greek_name: str, value: Optional[float], results: Dict):
        """Check if Greek value triggers any warnings"""
        if value is None or greek_name not in self.validation_rules:
            return
        
        rules = self.validation_rules[greek_name]
        value = float(value)
        
        if 'warning_high' in rules and value > rules['warning_high']:
            results['warnings'].append(f"High {greek_name}: {value:.5f}")
        
        if 'warning_low' in rules and value < rules['warning_low']:
            results['warnings'].append(f"Low {greek_name}: {value:.5f}")
        
        if 'warning_extreme' in rules and value < rules['warning_extreme']:
            results['warnings'].append(f"Extreme {greek_name}: {value:.5f}")
    
    def _check_consistency(self, results: Dict, option_type: str):
        """Check consistency between Greeks"""
        values = results['values']
        
        if 'delta' in values and 'gamma' in values:
            delta = abs(float(values['delta']))
            gamma = float(values['gamma'])
            
            # Near-zero delta should have low gamma
            if delta < self.consistency_checks['low_delta_threshold']:
                if gamma > self.consistency_checks['low_delta_max_gamma']:
                    results['warnings'].append(
                        f"Inconsistent: Low delta {delta:.3f} with high gamma {gamma:.4f}"
                    )
            
            # Near 1 delta should have near-zero gamma
            if delta > self.consistency_checks['high_delta_threshold']:
                if gamma > self.consistency_checks['high_delta_max_gamma']:
                    results['warnings'].append(
                        f"Inconsistent: High delta {delta:.3f} with gamma {gamma:.4f}"
                    )
    
    def validate_batch(self, options_data: List[Dict]) -> Dict:
        """
        Validate a batch of options
        Returns summary statistics
        """
        summary = {
            'total': len(options_data),
            'valid': 0,
            'invalid': 0,
            'warnings': 0,
            'errors_by_type': {},
            'sample_errors': []
        }
        
        for option in options_data:
            greeks = {
                'delta': option.get('delta'),
                'gamma': option.get('gamma'),
                'theta': option.get('theta'),
                'vega': option.get('vega'),
                'rho': option.get('rho')
            }
            
            result = self.validate_greeks_set(
                greeks, 
                option.get('option_type', 'call'),
                option.get('updated_at')
            )
            
            if result['valid']:
                summary['valid'] += 1
            else:
                summary['invalid'] += 1
                
                # Track error types
                for error in result['errors']:
                    error_type = error.split(':')[0]
                    summary['errors_by_type'][error_type] = \
                        summary['errors_by_type'].get(error_type, 0) + 1
                
                # Keep sample of errors
                if len(summary['sample_errors']) < 5:
                    summary['sample_errors'].append({
                        'contract': option.get('contract_id'),
                        'errors': result['errors']
                    })
            
            if result['warnings']:
                summary['warnings'] += 1
        
        summary['valid_pct'] = (summary['valid'] / summary['total'] * 100) if summary['total'] > 0 else 0
        
        return summary