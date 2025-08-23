"""
Plugin Manager for discovering, loading, and managing all system plugins.
This orchestrates the entire plugin ecosystem.
"""

import importlib
import inspect
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Type
import yaml
import logging
from datetime import datetime

from .plugin import Plugin, PluginState
from .bus import MessageBus

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages the lifecycle of all plugins in the system.
    
    Responsibilities:
    - Auto-discover plugins from directories
    - Load plugin configurations
    - Start/stop plugins in correct order
    - Monitor plugin health
    - Handle plugin failures gracefully
    """
    
    def __init__(self, bus: MessageBus, config_dir: str = "config"):
        """
        Initialize the plugin manager.
        
        Args:
            bus: The message bus for all communication
            config_dir: Directory containing plugin configurations
        """
        self.bus = bus
        self.config_dir = Path(config_dir)
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_configs: Dict[str, Dict] = {}
        self._running = False
        
        # Load system configuration
        self._load_system_config()
        
        logger.info(f"Plugin manager initialized with config dir: {config_dir}")
    
    def _load_system_config(self) -> None:
        """Load the main system configuration."""
        system_config_path = self.config_dir / "system.yaml"
        
        if system_config_path.exists():
            with open(system_config_path, 'r') as f:
                self.system_config = yaml.safe_load(f)
                logger.info("System configuration loaded")
        else:
            self.system_config = {
                'plugin_manager': {
                    'auto_discover': True,
                    'plugin_dirs': ['plugins']
                }
            }
            logger.warning("No system.yaml found, using defaults")
    
    def load_plugin_configs(self) -> None:
        """
        Load all plugin configuration files from config/plugins directory.
        """
        plugins_config_dir = self.config_dir / "plugins"
        
        if not plugins_config_dir.exists():
            logger.warning(f"Plugin config directory {plugins_config_dir} does not exist")
            return
        
        for config_file in plugins_config_dir.glob("*.yaml"):
            plugin_name = config_file.stem
            
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self.plugin_configs[plugin_name] = config
                    logger.info(f"Loaded config for plugin: {plugin_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
    
    async def discover_and_load(self) -> None:
        """
        Auto-discover and load all plugins from configured directories.
        
        This method:
        1. Scans plugin directories for Python files
        2. Imports modules and finds Plugin subclasses
        3. Instantiates plugins with their configurations
        4. Starts plugins that are enabled
        """
        # Load plugin configurations first
        self.load_plugin_configs()
        
        # Get plugin directories from config
        plugin_dirs = self.system_config.get('plugin_manager', {}).get(
            'plugin_dirs', ['plugins']
        )
        
        for plugin_dir in plugin_dirs:
            await self._discover_plugins_in_directory(plugin_dir)
        
        logger.info(f"Discovered and loaded {len(self.plugins)} plugins")
    
    async def _discover_plugins_in_directory(self, directory: str) -> None:
        """
        Discover plugins in a specific directory.
        
        Args:
            directory: Directory path to scan for plugins
        """
        plugins_path = Path(directory)
        
        if not plugins_path.exists():
            logger.warning(f"Plugin directory {directory} does not exist")
            return
        
        # Find all Python files recursively
        for py_file in plugins_path.rglob("*.py"):
            # Skip __init__ and test files
            if py_file.name.startswith("_") or "test" in py_file.name:
                continue
            
            # Convert path to module name
            # e.g., plugins/datasources/alpha_vantage.py -> plugins.datasources.alpha_vantage
            relative_path = py_file.relative_to(Path.cwd())
            module_path = str(relative_path).replace("/", ".").replace(".py", "")
            
            try:
                await self._load_plugin_from_module(module_path)
            except Exception as e:
                logger.error(f"Failed to load plugin from {module_path}: {e}")
    
    async def _load_plugin_from_module(self, module_path: str) -> None:
        """
        Load plugins from a Python module.
        
        Args:
            module_path: Dot-separated module path
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find all Plugin subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin and
                    not inspect.isabstract(obj)):
                    
                    # Get configuration for this plugin
                    plugin_name = name.lower()
                    config = self.plugin_configs.get(plugin_name, {})
                    
                    # Check if plugin is enabled
                    if not config.get('enabled', True):
                        logger.info(f"Plugin {name} is disabled in configuration")
                        continue
                    
                    # Instantiate the plugin
                    try:
                        plugin_instance = obj(name, self.bus, config)
                        self.plugins[name] = plugin_instance
                        
                        # Start the plugin if auto-start is enabled
                        if config.get('auto_start', True):
                            await plugin_instance.start_plugin()
                        
                        logger.info(f"✓ Loaded plugin: {name} from {module_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to instantiate plugin {name}: {e}")
                        
        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
    
    async def start_all(self) -> None:
        """
        Start all loaded plugins in dependency order.
        
        Some plugins may depend on others being started first.
        This method ensures proper startup sequence.
        """
        self._running = True
        
        # Define startup order (critical plugins first)
        startup_order = [
            # Data sources first
            "AlphaVantagePlugin",
            "IBKRPlugin",
            # Processing plugins
            "BarAggregatorPlugin",
            "DataValidatorPlugin",
            # Feature and ML
            "FeatureEnginePlugin",
            "ModelServerPlugin",
            # Analytics
            "VPINPlugin",
            "GEXPlugin",
            # Risk and execution
            "RiskManagerPlugin",
            "ExecutorPlugin",
            # Strategies last
            "SimpleMomentumStrategy",
            "MLStrategyPlugin",
            "DTE0StrategyPlugin"
        ]
        
        # Start plugins in order
        started = []
        for plugin_name in startup_order:
            if plugin_name in self.plugins:
                plugin = self.plugins[plugin_name]
                if plugin.state != PluginState.RUNNING:
                    try:
                        await plugin.start_plugin()
                        started.append(plugin_name)
                    except Exception as e:
                        logger.error(f"Failed to start {plugin_name}: {e}")
        
        # Start any remaining plugins not in the order list
        for name, plugin in self.plugins.items():
            if name not in startup_order and plugin.state != PluginState.RUNNING:
                try:
                    await plugin.start_plugin()
                    started.append(name)
                except Exception as e:
                    logger.error(f"Failed to start {name}: {e}")
        
        logger.info(f"Started {len(started)} plugins: {started}")
    
    async def stop_all(self) -> None:
        """
        Stop all plugins in reverse order of startup.
        
        This ensures dependencies are respected during shutdown.
        """
        self._running = False
        
        # Stop in reverse order
        plugins_to_stop = list(reversed(list(self.plugins.items())))
        
        for name, plugin in plugins_to_stop:
            if plugin.state == PluginState.RUNNING:
                try:
                    await plugin.stop_plugin()
                    logger.info(f"Stopped plugin: {name}")
                except Exception as e:
                    logger.error(f"Error stopping plugin {name}: {e}")
        
        logger.info("All plugins stopped")
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a specific plugin by name.
        
        Args:
            name: Plugin class name
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(name)
    
    def get_health_status(self) -> Dict[str, Dict]:
        """
        Get health status of all plugins.
        
        Returns:
            Dictionary mapping plugin names to their health status
        """
        health_status = {}
        
        for name, plugin in self.plugins.items():
            try:
                health_status[name] = plugin.health_check()
            except Exception as e:
                health_status[name] = {
                    'healthy': False,
                    'error': str(e),
                    'state': plugin.state.value
                }
        
        return health_status
    
    async def monitor_plugins(self, interval: int = 30) -> None:
        """
        Monitor plugin health and restart failed plugins.
        
        Args:
            interval: Check interval in seconds
        """
        while self._running:
            await asyncio.sleep(interval)
            
            for name, plugin in self.plugins.items():
                if plugin.state == PluginState.ERROR:
                    logger.warning(f"Plugin {name} in error state, attempting restart")
                    
                    try:
                        await plugin.start_plugin()
                        logger.info(f"Successfully restarted plugin {name}")
                    except Exception as e:
                        logger.error(f"Failed to restart plugin {name}: {e}")
            
            # Publish health status
            health_status = self.get_health_status()
            self.bus.publish(
                event_type="system.health.status",
                data={
                    'timestamp': datetime.utcnow().isoformat(),
                    'plugins': health_status,
                    'total_plugins': len(self.plugins),
                    'healthy_plugins': sum(
                        1 for h in health_status.values() 
                        if h.get('healthy', False)
                    )
                }
            )
    
    def list_plugins(self) -> List[str]:
        """
        Get list of all loaded plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())