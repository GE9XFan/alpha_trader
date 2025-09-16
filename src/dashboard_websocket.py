#!/usr/bin/env python3
"""
Dashboard WebSocket Module

WebSocket connection management and real-time data streaming for the dashboard.
This module extends the WebSocket functionality in dashboard_server.py.

Redis Keys Used:
    Read:
        - All real-time data keys
    Write:
        - dashboard:ws:connections (active WebSocket connections)
        - dashboard:ws:metrics (WebSocket metrics)

Author: AlphaTrader Pro
Version: 3.0.0
"""

import asyncio
import json
import time
import redis
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket
import logging


class WebSocketManager:
    """
    Manage WebSocket connections and broadcasting.
    """

    def __init__(self, redis_conn: redis.Redis):
        """
        Initialize WebSocket manager.
        """
        self.redis = redis_conn
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket):
        """
        Accept new WebSocket connection.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection.
        """
        self.active_connections.remove(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients.
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)