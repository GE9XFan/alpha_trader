"""
Base Strategy Abstract Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.get('strategy', {}).get('name', 'Unknown Strategy')
        self.enabled = config.get('strategy', {}).get('enabled', True)
        self.min_confidence = config.get('confidence', {}).get('minimum', 0.7)
        
        # Strategy metrics
        self.total_evaluations = 0
        self.signals_generated = 0
        self.trades_executed = 0
        self.win_count = 0
        self.loss_count = 0
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trading opportunity
        
        Args:
            context: Market context
            
        Returns:
            Evaluation results with confidence
        """
        pass
    
    @abstractmethod
    def generate_signal(self, evaluation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal from evaluation
        
        Args:
            evaluation: Evaluation results
            
        Returns:
            Trading signal or None
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], capital: float) -> int:
        """
        Calculate position size
        
        Args:
            signal: Trading signal
            capital: Available capital
            
        Returns:
            Number of contracts
        """
        pass
    
    def validate_rules(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate strategy rules
        
        Args:
            context: Market context
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check if strategy is enabled
        if not self.enabled:
            violations.append("Strategy is disabled")
            return False, violations
        
        # Check trading hours if specified
        if 'timing' in self.config:
            current_time = datetime.now().strftime('%H:%M')
            entry_window = self.config['timing'].get('entry_window', {})
            
            if entry_window:
                start = entry_window.get('start', '00:00')
                end = entry_window.get('end', '23:59')
                
                if not (start <= current_time <= end):
                    violations.append(f"Outside trading window ({start}-{end})")
        
        # Subclasses can add more validation
        return len(violations) == 0, violations
    
    def should_exit(self, position: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if position should be exited
        
        Args:
            position: Current position details
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Check stop loss
        if 'exit_rules' in self.config:
            exit_rules = self.config['exit_rules']
            
            # Stop loss check
            if 'stop_loss' in exit_rules:
                stop_loss = exit_rules['stop_loss']
                if position.get('pnl_pct', 0) <= -stop_loss:
                    return True, f"Stop loss hit ({stop_loss * 100}%)"
            
            # Take profit check
            if 'take_profit' in exit_rules:
                take_profit = exit_rules['take_profit']
                if position.get('pnl_pct', 0) >= take_profit:
                    return True, f"Take profit hit ({take_profit * 100}%)"
            
            # Time stop check
            if 'time_stop' in exit_rules:
                time_stop = exit_rules['time_stop']
                current_time = datetime.now().strftime('%H:%M')
                if current_time >= time_stop:
                    return True, f"Time stop reached ({time_stop})"
        
        return False, ""
    
    def update_metrics(self, trade_result: Dict[str, Any]) -> None:
        """Update strategy performance metrics"""
        self.trades_executed += 1
        
        if trade_result.get('pnl', 0) > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
    
    def get_performance(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / max(total_trades, 1)
        
        return {
            'name': self.name,
            'enabled': self.enabled,
            'total_evaluations': self.total_evaluations,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'wins': self.win_count,
            'losses': self.loss_count,
            'win_rate': win_rate,
            'signal_rate': self.signals_generated / max(self.total_evaluations, 1)
        }