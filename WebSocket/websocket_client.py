"""
WebSocket client for scheduling service
Provides interface for interacting with the WebSocket server
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .websocket_config import (
    WebSocketMessage, MessageType, 
    create_request_message, DEFAULT_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketSchedulingClient:
    """WebSocket client for scheduling algorithms"""
    
    def __init__(self, host='localhost', port=8765, config=None):
        self.host = host
        self.port = port
        self.config = config or DEFAULT_CONFIG
        self.websocket = None
        self.is_connected = False
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.heartbeat_task = None
        self.receive_task = None
        
        # Register default message handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_message_handler(MessageType.HEARTBEAT.value, self._handle_heartbeat)
        self.register_message_handler(MessageType.ERROR_RESPONSE.value, self._handle_error)
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")
    
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            uri = f"ws://{self.host}:{self.port}"
            self.websocket = await websockets.connect(uri)
            self.is_connected = True
            
            # Start receive task
            self.receive_task = asyncio.create_task(self._receive_messages())
            
            logger.info(f"Connected to WebSocket server at {uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        try:
            self.is_connected = False
            
            # Cancel tasks
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            
            if self.receive_task:
                self.receive_task.cancel()
            
            # Close websocket
            if self.websocket:
                await self.websocket.close()
            
            logger.info("Disconnected from WebSocket server")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    async def _receive_messages(self):
        """Receive and handle incoming messages"""
        try:
            async for message in self.websocket:
                try:
                    # Parse message
                    message_data = json.loads(message)
                    ws_message = WebSocketMessage.from_dict(message_data)
                    
                    logger.debug(f"Received message: {ws_message.message_type}")
                    
                    # Handle message
                    await self._handle_message(ws_message)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Receive task error: {e}")
        finally:
            self.is_connected = False
    
    async def _handle_message(self, message: WebSocketMessage):
        """Handle incoming message"""
        message_type = message.message_type
        message_id = message.message_id
        
        # Check if this is a response to a pending request
        if message_id in self.pending_requests:
            future = self.pending_requests.pop(message_id)
            if not future.done():
                future.set_result(message)
            return
        
        # Handle by registered handler
        if message_type in self.message_handlers:
            try:
                await self.message_handlers[message_type](message)
            except Exception as e:
                logger.error(f"Error in message handler for {message_type}: {e}")
        else:
            logger.debug(f"No handler registered for message type: {message_type}")
    
    async def _handle_heartbeat(self, message: WebSocketMessage):
        """Handle heartbeat message"""
        logger.debug("Received heartbeat")
    
    async def _handle_error(self, message: WebSocketMessage):
        """Handle error message"""
        logger.error(f"Server error: {message.error}")
    
    async def send_message(self, message: WebSocketMessage) -> WebSocketMessage:
        """Send message and wait for response"""
        if not self.is_connected:
            raise ConnectionError("Not connected to server")
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[message.message_id] = future
        
        try:
            # Send message
            message_json = message.to_json()
            await self.websocket.send(message_json)
            
            logger.debug(f"Sent message: {message.message_type}")
            
            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=300.0)  # 5 minutes timeout
            return response
            
        except asyncio.TimeoutError:
            # Remove from pending requests
            self.pending_requests.pop(message.message_id, None)
            raise TimeoutError(f"Timeout waiting for response to {message.message_type}")
        
        except Exception as e:
            # Remove from pending requests
            self.pending_requests.pop(message.message_id, None)
            raise
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get server status"""
        try:
            message = create_request_message(
                MessageType.SERVER_STATUS_REQUEST,
                {},
                f"status_{int(time.time())}"
            )
            
            response = await self.send_message(message)
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting server status: {e}")
            return {"error": str(e)}
    
    async def get_supported_algorithms(self) -> Dict[str, Any]:
        """Get list of supported algorithms"""
        try:
            message = create_request_message(
                MessageType.ALGORITHMS_REQUEST,
                {},
                f"algorithms_{int(time.time())}"
            )
            
            response = await self.send_message(message)
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting supported algorithms: {e}")
            return {"error": str(e)}
    
    async def get_algorithm_info(self, algorithm_name: str) -> Dict[str, Any]:
        """Get information about a specific algorithm"""
        try:
            message = create_request_message(
                MessageType.ALGORITHM_INFO_REQUEST,
                {"algorithm_name": algorithm_name},
                f"info_{algorithm_name}_{int(time.time())}"
            )
            
            response = await self.send_message(message)
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting algorithm info: {e}")
            return {"error": str(e)}
    
    async def est_eet_weighted_scheduling(self, input_data: Dict[str, Any], 
                                        alpha: float = 0.7, 
                                        ) -> Dict[str, Any]:
        """Run EST-EET weighted scheduling algorithm"""
        try:
            request_data = {
                "algorithm": "est_eet_weighted",
                "input_data": input_data,
                "alpha": alpha
            }
            
            message = create_request_message(
                MessageType.SCHEDULING_REQUEST,
                request_data,
                f"scheduling_{int(time.time())}"
            )
            
            logger.info(f"Running EST-EET weighted scheduling with alpha={alpha}...")
            start_time = time.time()
            
            response = await self.send_message(message)
            
            execution_time = time.time() - start_time
            logger.info(f"EST-EET weighted scheduling completed in {execution_time:.2f} seconds")
            
            return response.data
            
        except Exception as e:
            logger.error(f"Error in EST-EET weighted scheduling: {e}")
            return {"error": str(e)}
    
    async def est_spt_weighted_scheduling(self, input_data: Dict[str, Any], 
                                        alpha: float = 0.7, 
                                        ) -> Dict[str, Any]:
        """Run EST-SPT weighted scheduling algorithm"""
        try:
            request_data = {
                "algorithm": "est_spt_weighted",
                "input_data": input_data,
                "alpha": alpha
            }
            message = create_request_message(
                MessageType.SCHEDULING_REQUEST,
                request_data,
                f"scheduling_{int(time.time())}"
            )
            logger.info(f"Running EST-SPT weighted scheduling with alpha={alpha}...")
            start_time = time.time()
            response = await self.send_message(message)
            execution_time = time.time() - start_time
            logger.info(f"EST-SPT weighted scheduling completed in {execution_time:.2f} seconds")
            return response.data
        except Exception as e:
            logger.error(f"Error in EST-SPT weighted scheduling: {e}")
            return {"error": str(e)}

    async def ga_scheduling(self, input_data: Dict[str, Any], 
                          alpha: float = 0.7, 
                          population_size: int = 100, 
                          generations: int = 30, 
                          crossover_rate: float = 0.8, 
                          mutation_rate: float = 0.1
                          ) -> Dict[str, Any]:
        """Run Genetic Algorithm (GA) scheduling algorithm"""
        try:
            request_data = {
                "algorithm": "ga",
                "input_data": input_data,
                "alpha": alpha,
                "population_size": population_size,
                "generations": generations,
                "crossover_rate": crossover_rate,
                "mutation_rate": mutation_rate
            }
            message = create_request_message(
                MessageType.SCHEDULING_REQUEST,
                request_data,
                f"scheduling_{int(time.time())}"
            )
            logger.info(f"Running GA scheduling with alpha={alpha}, population_size={population_size}, generations={generations}, crossover_rate={crossover_rate}, mutation_rate={mutation_rate}...")
            start_time = time.time()
            response = await self.send_message(message)
            execution_time = time.time() - start_time
            logger.info(f"GA scheduling completed in {execution_time:.2f} seconds")
            return response.data
        except Exception as e:
            logger.error(f"Error in GA scheduling: {e}")
            return {"error": str(e)}
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

async def demo_scheduling_client():
    """Demo function showing how to use the WebSocket scheduling client"""

    import json
    import os
    input_case_name = 'input_test_1.json'
    input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'Data', 'InputData', input_case_name)
    with open(input_path, 'r') as f:
        example_input = json.load(f)
    
    # Use async context manager for automatic connection/disconnection
    async with WebSocketSchedulingClient() as client:
        # Get server status
        status = await client.get_server_status()
        print(f"Server Status: {status}")
        
        # Get supported algorithms
        algorithms = await client.get_supported_algorithms()
        print(f"Supported Algorithms: {algorithms}")
        
        # Run EST-EET weighted scheduling with different alpha values
        print("\n=== Running EST-EET Weighted Scheduling ===")
        result = await client.est_eet_weighted_scheduling(
            example_input,
        )
        print(f"EST-EET Weighted Scheduling Result: {result}")

        # Run EST-SPT weighted scheduling with different alpha values
        print("\n=== Running EST-SPT Weighted Scheduling ===")
        result = await client.est_spt_weighted_scheduling(
            example_input,
        )
        print(f"EST-SPT Weighted Scheduling Result: {result}")

        # Run GA scheduling
        print("\n=== Running GA Scheduling ===")
        result = await client.ga_scheduling(
            example_input,
        )
        print(f"GA Scheduling Result: {result}")


if __name__ == '__main__':
    asyncio.run(demo_scheduling_client()) 
