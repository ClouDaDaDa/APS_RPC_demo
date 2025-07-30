"""
WebSocket message handler for scheduling service
Handles message routing and response generation
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .websocket_config import (
    WebSocketMessage, MessageType, 
    create_response_message, create_error_response
)
from .scheduling_processor import scheduling_processor

logger = logging.getLogger(__name__)

class WebSocketMessageHandler:
    """Handles WebSocket messages for scheduling service"""
    
    def __init__(self):
        self.request_count = 0
        self.start_time = datetime.now()
    
    def _log_request(self, message_type: str, message_id: str, response_status: str):
        """Log request details"""
        self.request_count += 1
        logger.info(f"Request #{self.request_count}: {message_type} (ID: {message_id}) -> {response_status}")
    
    async def handle_message(self, message: WebSocketMessage) -> WebSocketMessage:
        """Handle incoming WebSocket message and return response"""
        try:
            message_type = message.message_type
            message_id = message.message_id
            data = message.data
            
            logger.debug(f"Handling message: {message_type} (ID: {message_id})")
            
            # Route message based on type
            if message_type == MessageType.SCHEDULING_REQUEST.value:
                return await self._handle_scheduling_request(message)
            
            elif message_type == MessageType.SERVER_STATUS_REQUEST.value:
                return await self._handle_server_status_request(message)
            
            elif message_type == MessageType.ALGORITHMS_REQUEST.value:
                return await self._handle_algorithms_request(message)
            
            elif message_type == MessageType.ALGORITHM_INFO_REQUEST.value:
                return await self._handle_algorithm_info_request(message)
            
            else:
                error_msg = f"Unknown message type: {message_type}"
                logger.warning(error_msg)
                return create_error_response(error_msg, message_id)
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return create_error_response(error_msg, message.message_id)
    
    async def _handle_scheduling_request(self, message: WebSocketMessage) -> WebSocketMessage:
        """Handle scheduling request"""
        try:
            # Run scheduling algorithm in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                scheduling_processor.process_scheduling_request, 
                message.data
            )
            
            self._log_request("scheduling_request", message.message_id, result.get("status", "unknown"))
            
            return create_response_message(
                MessageType.SCHEDULING_RESPONSE,
                result,
                message.message_id
            )
            
        except Exception as e:
            error_msg = f"Scheduling request error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_request("scheduling_request", message.message_id, "error")
            return create_error_response(error_msg, message.message_id)
    
    async def _handle_server_status_request(self, message: WebSocketMessage) -> WebSocketMessage:
        """Handle server status request"""
        try:
            uptime = datetime.now() - self.start_time
            status_data = {
                "status": "running",
                "uptime_seconds": uptime.total_seconds(),
                "request_count": self.request_count,
                "start_time": self.start_time.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_request("server_status_request", message.message_id, "success")
            
            return create_response_message(
                MessageType.SERVER_STATUS_RESPONSE,
                status_data,
                message.message_id
            )
            
        except Exception as e:
            error_msg = f"Server status request error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_request("server_status_request", message.message_id, "error")
            return create_error_response(error_msg, message.message_id)
    
    async def _handle_algorithms_request(self, message: WebSocketMessage) -> WebSocketMessage:
        """Handle algorithms request"""
        try:
            algorithms_data = scheduling_processor.get_supported_algorithms()
            
            self._log_request("algorithms_request", message.message_id, "success")
            
            return create_response_message(
                MessageType.ALGORITHMS_RESPONSE,
                algorithms_data,
                message.message_id
            )
            
        except Exception as e:
            error_msg = f"Algorithms request error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_request("algorithms_request", message.message_id, "error")
            return create_error_response(error_msg, message.message_id)
    
    async def _handle_algorithm_info_request(self, message: WebSocketMessage) -> WebSocketMessage:
        """Handle algorithm info request"""
        try:
            algorithm_name = message.data.get("algorithm_name")
            if not algorithm_name:
                return create_error_response("Missing algorithm_name parameter", message.message_id)
            
            info_data = scheduling_processor.get_algorithm_info(algorithm_name)
            
            self._log_request("algorithm_info_request", message.message_id, info_data.get("status", "unknown"))
            
            return create_response_message(
                MessageType.ALGORITHM_INFO_RESPONSE,
                info_data,
                message.message_id
            )
            
        except Exception as e:
            error_msg = f"Algorithm info request error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_request("algorithm_info_request", message.message_id, "error")
            return create_error_response(error_msg, message.message_id)
    
    def create_heartbeat_message(self) -> WebSocketMessage:
        """Create heartbeat message"""
        return WebSocketMessage(
            message_type=MessageType.HEARTBEAT.value,
            message_id="heartbeat",
            timestamp=datetime.now().isoformat(),
            data={"timestamp": datetime.now().isoformat()}
        ) 