"""
Central Message Bus for the AlphaTrader system.
This is the spine of the entire system - all communication flows through here.
"""

import asyncio
import logging
import re
import traceback
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
import threading

from .message import Message
from .persistence import EventStore

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Central nervous system of the trading platform.
    All components communicate through this bus via pub/sub pattern.
    
    Critical features:
    - Pattern-based subscriptions with wildcards
    - Automatic persistence before distribution
    - Error isolation (one failure doesn't crash the bus)
    - Support for both sync and async handlers
    - Thread-safe operations
    """
    
    def __init__(self, event_store: Optional[EventStore] = None):
        """
        Initialize the message bus.
        
        Args:
            event_store: Optional persistence layer for events
        """
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_store = event_store
        self._running = False
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._message_count = 0
        self._error_count = 0
        
        logger.info("Message bus initialized")
    
    def publish(self, 
                event_type: str, 
                data: Dict, 
                correlation_id: Optional[str] = None,
                metadata: Optional[Dict] = None) -> Message:
        """
        Publish an event to all matching subscribers.
        
        This is the PRIMARY method for all system communication.
        Events are persisted first, then distributed to subscribers.
        
        Args:
            event_type: Type of event (e.g., 'ibkr.bar.5s')
            data: Event payload
            correlation_id: Optional ID to link related events
            metadata: Optional metadata
            
        Returns:
            The created Message object
        """
        # Create message
        message = Message.create(
            event_type=event_type,
            data=data,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        # Persist first (critical for event sourcing)
        if self.event_store:
            try:
                self.event_store.save(message)
            except Exception as e:
                logger.error(f"Failed to persist message: {e}")
                # Continue - we don't want persistence failure to stop the system
        
        # Distribute to subscribers
        self._distribute(message)
        
        self._message_count += 1
        
        logger.debug(f"Published: {message}")
        return message
    
    def subscribe(self, pattern: str, handler: Callable) -> None:
        """
        Subscribe to events matching a pattern.
        
        Patterns support wildcards:
        - '*' matches any single segment
        - 'ibkr.*' matches 'ibkr.bar', 'ibkr.order', etc.
        - '*.signal.*' matches 'strategy.signal.entry', 'ml.signal.exit', etc.
        
        Args:
            pattern: Event pattern to match
            handler: Function to call when matching event occurs
        """
        with self._lock:
            self.subscribers[pattern].append(handler)
            logger.info(f"Subscribed {handler.__name__} to pattern '{pattern}'")
    
    def unsubscribe(self, pattern: str, handler: Callable) -> None:
        """
        Remove a subscription.
        
        Args:
            pattern: The pattern that was subscribed to
            handler: The handler to remove
        """
        with self._lock:
            if pattern in self.subscribers and handler in self.subscribers[pattern]:
                self.subscribers[pattern].remove(handler)
                logger.info(f"Unsubscribed {handler.__name__} from '{pattern}'")
    
    def _distribute(self, message: Message) -> None:
        """
        Distribute message to all matching subscribers.
        
        This method ensures that:
        1. All matching subscribers receive the message
        2. One subscriber failure doesn't affect others
        3. Both sync and async handlers are supported
        
        Args:
            message: The message to distribute
        """
        matched_handlers = []
        
        # Find all matching handlers
        with self._lock:
            for pattern, handlers in self.subscribers.items():
                if self._matches_pattern(pattern, message.event_type):
                    matched_handlers.extend(handlers)
        
        # Call each handler (outside the lock to prevent deadlocks)
        for handler in matched_handlers:
            self._call_handler(handler, message)
    
    def _call_handler(self, handler: Callable, message: Message) -> None:
        """
        Call a handler with proper error isolation.
        
        Args:
            handler: The handler function to call
            message: The message to pass to the handler
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                # Handle async functions
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                loop.create_task(self._call_async_handler(handler, message))
            else:
                # Handle sync functions in thread pool to prevent blocking
                self._executor.submit(self._call_sync_handler, handler, message)
                
        except Exception as e:
            self._error_count += 1
            logger.error(
                f"Failed to submit handler {handler.__name__}: {e}\n"
                f"Message: {message}\n"
                f"Traceback: {traceback.format_exc()}"
            )
    
    async def _call_async_handler(self, handler: Callable, message: Message) -> None:
        """
        Call an async handler with error handling.
        
        Args:
            handler: Async handler function
            message: Message to process
        """
        try:
            await handler(message)
        except Exception as e:
            self._error_count += 1
            logger.error(
                f"Async handler {handler.__name__} failed: {e}\n"
                f"Message: {message}\n"
                f"Traceback: {traceback.format_exc()}"
            )
    
    def _call_sync_handler(self, handler: Callable, message: Message) -> None:
        """
        Call a sync handler with error handling.
        
        Args:
            handler: Sync handler function
            message: Message to process
        """
        try:
            handler(message)
        except Exception as e:
            self._error_count += 1
            logger.error(
                f"Sync handler {handler.__name__} failed: {e}\n"
                f"Message: {message}\n"
                f"Traceback: {traceback.format_exc()}"
            )
    
    def _matches_pattern(self, pattern: str, event_type: str) -> bool:
        """
        Check if an event type matches a subscription pattern.
        
        Patterns use '.' as separator and '*' as wildcard.
        Examples:
        - 'ibkr.bar.5s' matches 'ibkr.bar.5s' exactly
        - 'ibkr.*' matches 'ibkr.bar', 'ibkr.order', etc.
        - '*.bar.*' matches 'ibkr.bar.5s', 'aggregator.bar.1m', etc.
        
        Args:
            pattern: Subscription pattern
            event_type: Actual event type
            
        Returns:
            True if the event type matches the pattern
        """
        # Convert pattern to regex
        # Escape dots, then replace * with .*
        regex_pattern = pattern.replace('.', r'\.')
        regex_pattern = regex_pattern.replace('*', '[^.]+')
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, event_type))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the message bus.
        
        Returns:
            Dictionary with bus statistics
        """
        with self._lock:
            total_handlers = sum(len(handlers) for handlers in self.subscribers.values())
            
            return {
                'message_count': self._message_count,
                'error_count': self._error_count,
                'subscription_patterns': len(self.subscribers),
                'total_handlers': total_handlers,
                'patterns': list(self.subscribers.keys())
            }
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown the message bus.
        """
        logger.info("Shutting down message bus...")
        self._running = False
        self._executor.shutdown(wait=True)
        
        if self.event_store:
            self.event_store.close()
        
        logger.info(f"Message bus shutdown complete. "
                   f"Processed {self._message_count} messages with {self._error_count} errors.")


class AsyncMessageBus(MessageBus):
    """
    Async version of the message bus for high-performance scenarios.
    """
    
    def __init__(self, event_store: Optional[EventStore] = None):
        super().__init__(event_store)
        self._queue = asyncio.Queue()
        self._task = None
    
    async def start(self):
        """Start the async message processor."""
        self._running = True
        self._task = asyncio.create_task(self._process_messages())
        logger.info("Async message bus started")
    
    async def stop(self):
        """Stop the async message processor."""
        self._running = False
        if self._task:
            await self._task
        logger.info("Async message bus stopped")
    
    async def _process_messages(self):
        """Process messages from the queue."""
        while self._running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=1.0
                )
                self._distribute(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")