"""
Base Plugin class that all system components inherit from.
This ensures consistent interface and behavior across all plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from enum import Enum
import logging
import asyncio
from datetime import datetime, timezone

from .bus import MessageBus
from .message import Message


class PluginState(Enum):
    """Plugin lifecycle states."""
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class Plugin(ABC):
    """
    Base class for all AlphaTrader plugins.
    
    Every component in the system is a plugin that:
    - Subscribes to events from the message bus
    - Publishes events to the message bus
    - Has a defined lifecycle (start, stop, health_check)
    - Is configured via YAML
    - Never communicates directly with other plugins
    """
    
    def __init__(self, name: str, bus: MessageBus, config: Dict[str, Any]):
        """
        Initialize the plugin.
        
        Args:
            name: Name of the plugin instance
            bus: The message bus for all communication
            config: Plugin-specific configuration from YAML
        """
        self.name = name
        self.bus = bus
        self.config = config
        self.logger = logging.getLogger(self.name)
        self.state = PluginState.INITIALIZED
        self._start_time = None
        self._message_count = 0
        self._error_count = 0
        
        # Set up structured logging
        self.logger.info(f"Plugin {self.name} initialized", extra={
            'plugin': self.name,
            'config': config
        })
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the plugin and subscribe to necessary events.
        
        This method should:
        1. Initialize any resources (connections, files, etc.)
        2. Subscribe to relevant events on the message bus
        3. Start any background tasks
        4. Set state to RUNNING
        
        Must be implemented by all plugins.
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the plugin and clean up resources.
        
        This method should:
        1. Stop any background tasks
        2. Close connections
        3. Clean up resources
        4. Set state to STOPPED
        
        Must be implemented by all plugins.
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Return the health status of the plugin.
        
        This method should return a dictionary with at minimum:
        - 'healthy': bool indicating if the plugin is functioning
        - 'state': current PluginState
        - Additional plugin-specific metrics
        
        Returns:
            Dictionary with health information
        """
        pass
    
    def publish(self, event_type: str, data: Dict[str, Any], 
                correlation_id: Optional[str] = None) -> Message:
        """
        Publish an event to the message bus with plugin prefix.
        
        The event type will be prefixed with the plugin name for traceability.
        For example, if plugin "RiskManager" publishes "signal.approved",
        the actual event type will be "riskmanager.signal.approved".
        
        Args:
            event_type: The event type (will be prefixed with plugin name)
            data: The event data
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            The created Message
        """
        # Add plugin name to event type for traceability
        full_event_type = f"{self.name.lower()}.{event_type}"
        
        # Add plugin metadata
        metadata = {
            'plugin': self.name,
            'plugin_version': getattr(self, 'version', '1.0.0')
        }
        
        message = self.bus.publish(
            event_type=full_event_type,
            data=data,
            correlation_id=correlation_id,
            metadata=metadata
        )
        
        self._message_count += 1
        return message
    
    def subscribe(self, pattern: str, handler: Callable) -> None:
        """
        Subscribe to events matching a pattern.
        
        This is a convenience wrapper around bus.subscribe that
        automatically adds error handling and logging.
        
        Args:
            pattern: Event pattern to subscribe to
            handler: Function to handle matching events
        """
        # Wrap handler with error handling
        def wrapped_handler(message: Message):
            try:
                self.logger.debug(f"Handling {message.event_type}")
                handler(message)
            except Exception as e:
                self._error_count += 1
                self.logger.error(
                    f"Error handling {message.event_type}: {e}",
                    exc_info=True,
                    extra={
                        'plugin': self.name,
                        'event_type': message.event_type,
                        'message_id': message.id
                    }
                )
        
        self.bus.subscribe(pattern, wrapped_handler)
        self.logger.info(f"Subscribed to pattern: {pattern}")
    
    async def start_plugin(self) -> None:
        """
        Wrapper for start() that handles state transitions.
        Called by the plugin manager.
        """
        # Only start if not already running
        if self.state == PluginState.RUNNING:
            self.logger.warning(f"Plugin {self.name} already running")
            return
            
        try:
            self.state = PluginState.STARTING
            self.logger.info(f"Starting plugin {self.name}")
            
            self._start_time = datetime.now(timezone.utc)
            await self.start()
            
            self.state = PluginState.RUNNING
            self.logger.info(f"Plugin {self.name} started successfully")
            
            # Publish startup event
            self.bus.publish(
                event_type="system.plugin.started",
                data={
                    'plugin': self.name,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            self.state = PluginState.ERROR
            self.logger.error(f"Failed to start plugin {self.name}: {e}", exc_info=True)
            
            # Publish error event
            self.bus.publish(
                event_type="system.plugin.error",
                data={
                    'plugin': self.name,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            raise
    
    async def stop_plugin(self) -> None:
        """
        Wrapper for stop() that handles state transitions.
        Called by the plugin manager.
        """
        # Only stop if actually running
        if self.state != PluginState.RUNNING:
            self.logger.warning(f"Cannot stop plugin {self.name} - not running (state: {self.state})")
            return
            
        try:
            self.state = PluginState.STOPPING
            self.logger.info(f"Stopping plugin {self.name}")
            
            await self.stop()
            
            self.state = PluginState.STOPPED
            self.logger.info(f"Plugin {self.name} stopped successfully")
            
            # Publish shutdown event
            self.bus.publish(
                event_type="system.plugin.stopped",
                data={
                    'plugin': self.name,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'messages_processed': self._message_count,
                    'errors': self._error_count
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error stopping plugin {self.name}: {e}", exc_info=True)
            # Don't re-raise - we want to stop other plugins even if one fails
    
    def get_base_health(self) -> Dict[str, Any]:
        """
        Get base health information common to all plugins.
        
        Returns:
            Dictionary with base health metrics
        """
        uptime = None
        if self._start_time:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        
        return {
            'healthy': self.state == PluginState.RUNNING,
            'state': self.state.value,
            'name': self.name,
            'uptime_seconds': uptime,
            'messages_published': self._message_count,
            'errors': self._error_count,
            'config': {
                'enabled': self.config.get('enabled', True)
            }
        }


class AsyncPlugin(Plugin):
    """
    Base class for plugins that need async operations.
    Provides additional async utilities.
    """
    
    def __init__(self, bus: MessageBus, config: Dict[str, Any]):
        super().__init__(bus, config)
        self._tasks = []
    
    def create_task(self, coro) -> asyncio.Task:
        """
        Create and track an async task.
        Tasks are automatically cancelled on plugin stop.
        
        Args:
            coro: Coroutine to run as task
            
        Returns:
            The created Task
        """
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task
    
    async def stop(self) -> None:
        """
        Stop all tracked tasks.
        Override this in subclasses but call super().stop()
        """
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()