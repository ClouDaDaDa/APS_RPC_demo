"""
WebSocket server for scheduling service
Handles client connections and message processing
"""

import asyncio
import json
import logging
import websockets
from typing import Set, Dict, Any
from datetime import datetime

from .websocket_config import WebSocketConfig, WebSocketMessage, DEFAULT_CONFIG
from .message_handler import WebSocketMessageHandler

# Configure logging
import os

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'websocket_scheduling_server.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WebSocketSchedulingServer:
    """WebSocket server for scheduling algorithms"""
    
    def __init__(self, config: WebSocketConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.host = self.config.host
        self.port = self.config.port
        self.max_connections = self.config.max_connections
        self.heartbeat_interval = self.config.heartbeat_interval
        self.connection_timeout = self.config.connection_timeout
        
        # Server state
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.message_handler = WebSocketMessageHandler()
        self.server = None
        self.heartbeat_task = None
        self.is_running = False
        
        logger.info(f"Initialized WebSocket server: {self.host}:{self.port}")
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection"""
        if len(self.clients) >= self.max_connections:
            logger.warning(f"Maximum connections reached ({self.max_connections})")
            await websocket.close(1013, "Maximum connections reached")
            return
        
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_info} (Total: {len(self.clients)})")
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client disconnected: {client_info} (Total: {len(self.clients)})")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle individual client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse incoming message
                    message_data = json.loads(message)
                    ws_message = WebSocketMessage.from_dict(message_data)
                    # print(message)
                    # print(message_data)
                    
                    logger.debug(f"Received message: {ws_message.message_type} from {websocket.remote_address}")
                    
                    # Handle message and get response
                    response = await self.message_handler.handle_message(ws_message)
                    
                    # Send response back to client
                    response_json = response.to_json()
                    await websocket.send(response_json)
                    
                    logger.debug(f"Sent response: {response.message_type} to {websocket.remote_address}")
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON format: {str(e)}"
                    logger.warning(f"JSON decode error from {websocket.remote_address}: {error_msg}")
                    
                    error_response = WebSocketMessage(
                        message_type="error_response",
                        message_id="invalid_json",
                        timestamp=datetime.now().isoformat(),
                        data={},
                        error=error_msg
                    )
                    await websocket.send(error_response.to_json())
                
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    logger.error(f"Message processing error from {websocket.remote_address}: {error_msg}")
                    
                    error_response = WebSocketMessage(
                        message_type="error_response",
                        message_id="processing_error",
                        timestamp=datetime.now().isoformat(),
                        data={},
                        error=error_msg
                    )
                    await websocket.send(error_response.to_json())
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client connection closed: {websocket.remote_address}")
        
        except Exception as e:
            logger.error(f"Client handling error: {e}", exc_info=True)
        
        finally:
            await self.unregister_client(websocket)
    
    async def broadcast_heartbeat(self):
        """Broadcast heartbeat to all connected clients"""
        while self.is_running:
            try:
                if self.clients:
                    heartbeat_msg = self.message_handler.create_heartbeat_message()
                    heartbeat_json = heartbeat_msg.to_json()
                    
                    # Create tasks for all clients
                    tasks = [
                        asyncio.create_task(client.send(heartbeat_json))
                        for client in self.clients.copy()
                    ]
                    
                    if tasks:
                        # Wait for all sends to complete or timeout
                        done, pending = await asyncio.wait(
                            tasks, 
                            timeout=5.0,
                            return_when=asyncio.ALL_COMPLETED
                        )
                        
                        # Cancel any pending tasks
                        for task in pending:
                            task.cancel()
                        
                        logger.debug(f"Sent heartbeat to {len(done)} clients")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info("Starting WebSocket Scheduling Server...")
        logger.info(f"Server will listen on ws://{self.host}:{self.port}")
        logger.info(f"Max connections: {self.max_connections}")
        logger.info(f"Heartbeat interval: {self.heartbeat_interval}s")
        
        try:
            # Start server
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                max_size=self.config.max_message_size,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_running = True
            
            # Start heartbeat task
            self.heartbeat_task = asyncio.create_task(self.broadcast_heartbeat())
            
            logger.info("WebSocket server started successfully")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"Server start error: {e}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        logger.info("Stopping WebSocket server...")
        
        self.is_running = False
        
        # Cancel heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        if self.clients:
            close_tasks = [
                asyncio.create_task(client.close(1000, "Server shutdown"))
                for client in self.clients.copy()
            ]
            if close_tasks:
                await asyncio.wait(close_tasks, timeout=5.0)
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")

async def main():
    """Main function to run the WebSocket server"""
    server = WebSocketSchedulingServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await server.stop_server()

if __name__ == '__main__':
    asyncio.run(main()) 