from fastapi import WebSocket
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    WebSocket connection manager for handling multiple client connections.
    """
    
    def __init__(self):
        """Initialize an empty list of active connections."""
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """
        Accept a new WebSocket connection and add it to the active connections.
        
        Args:
            websocket: The WebSocket connection to accept
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection from the active connections.
        
        Args:
            websocket: The WebSocket connection to remove
        """
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            message: The message to send
            websocket: The WebSocket connection to send the message to
        """
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """
        Broadcast a message to all active WebSocket connections.
        
        Args:
            message: The message to broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up any disconnected connections
        for connection in disconnected:
            try:
                self.active_connections.remove(connection)
            except ValueError:
                pass 