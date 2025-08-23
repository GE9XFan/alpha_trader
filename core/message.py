"""
Message and Event classes for the AlphaTrader system.
These are immutable data structures that flow through the message bus.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid
import json


@dataclass(frozen=True)
class Message:
    """
    Immutable message that flows through the system.
    
    Attributes:
        id: Unique identifier for this message
        correlation_id: ID to trace related messages through the system
        event_type: Hierarchical event type (domain.entity.action)
        data: The actual payload of the message
        timestamp: When the message was created
        metadata: Additional context about the message
    """
    id: str
    correlation_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, 
               event_type: str, 
               data: Dict[str, Any], 
               correlation_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> 'Message':
        """
        Factory method to create a new message with generated IDs.
        
        Args:
            event_type: The type of event (e.g., 'ibkr.bar.5s')
            data: The message payload
            correlation_id: Optional ID to link related messages
            metadata: Optional metadata about the message
            
        Returns:
            A new Message instance
        """
        return cls(
            id=str(uuid.uuid4()),
            correlation_id=correlation_id or str(uuid.uuid4()),
            event_type=event_type,
            data=data,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'id': self.id,
            'correlation_id': self.correlation_id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            id=data['id'],
            correlation_id=data['correlation_id'],
            event_type=data['event_type'],
            data=data['data'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Message(event_type='{self.event_type}', "
                f"id='{self.id[:8]}...', "
                f"timestamp='{self.timestamp.isoformat()}')")


class EventType:
    """
    Constants for event type patterns.
    Event types follow the pattern: domain.entity.action
    """
    
    # Data source events
    DATA_ALPHAAVANTAGE = "alphavantage.data.*"
    DATA_IBKR = "ibkr.data.*"
    BAR_5S = "ibkr.bar.5s"
    BAR_1M = "aggregator.bar.1m"
    BAR_5M = "aggregator.bar.5m"
    
    # Feature events
    FEATURES_CALCULATED = "featureengine.features.calculated"
    
    # ML events
    PREDICTION_GENERATED = "modelserver.prediction.generated"
    
    # Signal events
    SIGNAL_ENTRY = "*.signal.entry"
    SIGNAL_EXIT = "*.signal.exit"
    SIGNAL_APPROVED = "riskmanager.signal.approved"
    SIGNAL_REJECTED = "riskmanager.signal.rejected"
    
    # Order events
    ORDER_PLACED = "executor.order.placed"
    ORDER_FILLED = "executor.order.filled"
    ORDER_CANCELLED = "executor.order.cancelled"
    ORDER_REJECTED = "executor.order.rejected"
    
    # Risk events
    RISK_LIMIT_BREACH = "riskmanager.limit.breach"
    RISK_CIRCUIT_BREAK = "riskmanager.circuit.break"
    
    # System events
    PLUGIN_STARTED = "system.plugin.started"
    PLUGIN_STOPPED = "system.plugin.stopped"
    PLUGIN_ERROR = "system.plugin.error"
    HEALTH_CHECK = "system.health.check"