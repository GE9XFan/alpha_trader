"""
Exception hierarchy for AlphaTrader system.

Production-grade exception handling with detailed error information
and proper inheritance structure for granular error catching.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone


class AlphaTraderException(Exception):
    """
    Base exception for all AlphaTrader errors.
    
    Provides structured error information with timestamps and context.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize AlphaTrader exception.
        
        Args:
            message: Human-readable error message
            error_code: System error code (e.g., 'E001')
            details: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(AlphaTraderException):
    """
    Raised when configuration is invalid or missing.
    
    Examples:
    - Missing required environment variables
    - Invalid parameter ranges
    - Conflicting settings
    """
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error description
            field: Configuration field that caused the error
        """
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        kwargs["details"] = details
        kwargs["error_code"] = kwargs.get("error_code", "E001")
        super().__init__(message, **kwargs)


class DataSourceError(AlphaTraderException):
    """
    Raised when data source operations fail.
    
    Examples:
    - IBKR connection lost
    - Alpha Vantage API errors
    - Rate limit exceeded
    """
    
    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize data source error.
        
        Args:
            message: Error description
            source: Data source name (e.g., 'IBKR', 'AlphaVantage')
            operation: Operation that failed (e.g., 'subscribe', 'fetch')
        """
        details = kwargs.get("details", {})
        if source:
            details["source"] = source
        if operation:
            details["operation"] = operation
        kwargs["details"] = details
        kwargs["error_code"] = kwargs.get("error_code", "E002")
        super().__init__(message, **kwargs)


class ConnectionError(AlphaTraderException):
    """
    Raised when network/connection issues occur.
    
    Examples:
    - Cannot connect to IBKR TWS
    - Database connection failed
    - Network timeout
    """
    
    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        service: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize connection error.
        
        Args:
            message: Error description
            host: Host that failed to connect
            port: Port number
            service: Service name
        """
        details = kwargs.get("details", {})
        if host:
            details["host"] = host
        if port:
            details["port"] = port
        if service:
            details["service"] = service
        kwargs["details"] = details
        kwargs["error_code"] = kwargs.get("error_code", "E005")
        super().__init__(message, **kwargs)


class RiskLimitExceeded(AlphaTraderException):
    """
    Raised when risk limits are breached.
    
    CRITICAL: This exception triggers immediate position closure.
    
    Examples:
    - Portfolio Greeks exceed limits
    - Daily loss limit reached
    - VPIN threshold breached
    - Position size limit exceeded
    """
    
    def __init__(
        self,
        message: str,
        limit_type: str,
        current_value: float,
        limit_value: float,
        action_required: str = "HALT_TRADING",
        **kwargs
    ):
        """
        Initialize risk limit error.
        
        Args:
            message: Error description
            limit_type: Type of limit breached (e.g., 'DAILY_LOSS', 'DELTA')
            current_value: Current value that breached limit
            limit_value: The limit that was exceeded
            action_required: Required action (e.g., 'HALT_TRADING', 'CLOSE_POSITIONS')
        """
        details = kwargs.get("details", {})
        details.update({
            "limit_type": limit_type,
            "current_value": current_value,
            "limit_value": limit_value,
            "action_required": action_required,
            "severity": "CRITICAL",
        })
        kwargs["details"] = details
        kwargs["error_code"] = kwargs.get("error_code", "E004")
        super().__init__(message, **kwargs)


class ModelError(AlphaTraderException):
    """
    Raised when model operations fail.
    
    Examples:
    - Model loading failed
    - Inference error
    - Low confidence predictions
    - Model version mismatch
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        confidence: Optional[float] = None,
        fallback_available: bool = True,
        **kwargs
    ):
        """
        Initialize model error.
        
        Args:
            message: Error description
            model_name: Name of the model that failed
            confidence: Model confidence if applicable
            fallback_available: Whether fallback model is available
        """
        details = kwargs.get("details", {})
        if model_name:
            details["model_name"] = model_name
        if confidence is not None:
            details["confidence"] = confidence
        details["fallback_available"] = fallback_available
        kwargs["details"] = details
        kwargs["error_code"] = kwargs.get("error_code", "E003")
        super().__init__(message, **kwargs)


class OrderExecutionError(AlphaTraderException):
    """
    Raised when order execution fails.
    
    Examples:
    - Order rejected by broker
    - Insufficient funds
    - Symbol halted
    - Invalid order parameters
    """
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize order execution error.
        
        Args:
            message: Error description
            symbol: Trading symbol
            order_type: Type of order (e.g., 'MARKET', 'LIMIT')
            reason: Rejection reason from broker
        """
        details = kwargs.get("details", {})
        if symbol:
            details["symbol"] = symbol
        if order_type:
            details["order_type"] = order_type
        if reason:
            details["reason"] = reason
        kwargs["details"] = details
        kwargs["error_code"] = kwargs.get("error_code", "E007")
        super().__init__(message, **kwargs)


class VPINThresholdExceeded(RiskLimitExceeded):
    """
    Specialized exception for VPIN toxicity threshold breach.
    
    CRITICAL: Triggers immediate closure of all positions.
    """
    
    def __init__(self, current_vpin: float, threshold: float = 0.7, **kwargs):
        """
        Initialize VPIN threshold error.
        
        Args:
            current_vpin: Current VPIN value
            threshold: VPIN threshold that was exceeded
        """
        message = f"VPIN toxicity threshold exceeded: {current_vpin:.3f} > {threshold}"
        super().__init__(
            message=message,
            limit_type="VPIN_THRESHOLD",
            current_value=current_vpin,
            limit_value=threshold,
            action_required="CLOSE_ALL_POSITIONS",
            error_code="E009",
            **kwargs
        )


class ZeroDTEClosureError(AlphaTraderException):
    """
    Raised when 0DTE options fail to close at end of day.
    
    CRITICAL: Requires immediate manual intervention.
    """
    
    def __init__(
        self,
        message: str,
        positions: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize 0DTE closure error.
        
        Args:
            message: Error description
            positions: List of positions that failed to close
        """
        details = kwargs.get("details", {})
        if positions:
            details["positions"] = positions
        details["severity"] = "CRITICAL"
        details["action_required"] = "MANUAL_INTERVENTION"
        kwargs["details"] = details
        kwargs["error_code"] = "E010"
        super().__init__(message, **kwargs)


class RateLimitError(DataSourceError):
    """
    Raised when API rate limits are exceeded.
    
    Includes retry timing and fallback options.
    """
    
    def __init__(
        self,
        message: str,
        api_name: str,
        retry_after: Optional[int] = None,
        calls_made: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error description
            api_name: Name of the API that hit rate limit
            retry_after: Seconds to wait before retry
            calls_made: Number of calls made
            limit: Rate limit threshold
        """
        details = kwargs.get("details", {})
        details.update({
            "api_name": api_name,
            "retry_after": retry_after,
            "calls_made": calls_made,
            "limit": limit,
        })
        kwargs["details"] = details
        kwargs["source"] = api_name
        super().__init__(message, **kwargs)


class ValidationError(AlphaTraderException):
    """
    Raised when data validation fails.
    
    Examples:
    - Invalid price data
    - Malformed API response
    - Data integrity issues
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error description
            field: Field that failed validation
            expected: Expected value/type
            actual: Actual value received
        """
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if expected is not None:
            details["expected"] = str(expected)
        if actual is not None:
            details["actual"] = str(actual)
        kwargs["details"] = details
        super().__init__(message, **kwargs)