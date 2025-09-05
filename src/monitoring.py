#!/usr/bin/env python3
"""System monitoring with proper async Redis and state tracking."""
import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
import redis.asyncio as aioredis


class SystemMonitor:
    """Monitor subsystem health with state transition tracking."""
    
    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        self.config = config
        self.redis = redis_conn  # Async Redis
        self.logger = logging.getLogger(__name__)
        
        # Track state transitions
        self.last_states = {}
        
        # Thresholds from config
        ttls = config['modules']['data_ingestion']['store_ttls']
        self.heartbeat_ttl = ttls.get('heartbeat', 15)
        self.fresh_threshold = self.heartbeat_ttl
        self.late_threshold = self.heartbeat_ttl * 2
        
    async def start(self):
        """Start monitoring loop."""
        self.logger.info("Starting System Monitor")
        
        while True:
            try:
                await self._check_subsystems()
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _check_subsystems(self):
        """Check all subsystem health."""
        status = {}
        current_time = time.time()
        
        for subsystem in ['ibkr', 'av', 'analytics']:
            hb_data = await self.redis.get(f'hb:{subsystem}')
            
            if hb_data:
                try:
                    hb = json.loads(hb_data)
                    age = current_time - (hb['ts'] / 1000)
                    
                    if age < self.fresh_threshold:
                        status[subsystem] = 'up'
                    elif age < self.late_threshold:
                        status[subsystem] = 'late'
                    else:
                        status[subsystem] = 'down'
                except:
                    status[subsystem] = 'error'
            else:
                status[subsystem] = 'down'
        
        # Store status
        # Store system status to Redis
        await self.redis.hset('status:system', mapping=status)
        
        # Log transitions only (not same-state)
        for subsystem, state in status.items():
            old_state = self.last_states.get(subsystem)
            if old_state != state:
                if old_state is not None:  # Skip first check
                    self.logger.info(f"Status: {subsystem} {old_state} → {state}")
                self.last_states[subsystem] = state
    
    async def stop(self):
        """Stop monitoring."""
        self.logger.info("Stopping System Monitor")