"""
Base Strategy Abstract Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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
        self.name = config.get('name', 'Unknown')
        self.enabled = config.get('enabled', True)
        
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
    
    def validate_rules(self, context: Dict[str, Any]) -> bool:
        """Validate strategy rules"""
        # Common validation logic
        pass
