"""
WebSocket configuration and message format definitions
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from enum import Enum

class MessageType(Enum):
    """Message types for WebSocket communication"""
    # Client to Server
    SCHEDULING_REQUEST = "scheduling_request"
    SERVER_STATUS_REQUEST = "server_status_request"
    ALGORITHMS_REQUEST = "algorithms_request"
    ALGORITHM_INFO_REQUEST = "algorithm_info_request"
    
    # Server to Client
    SCHEDULING_RESPONSE = "scheduling_response"
    SERVER_STATUS_RESPONSE = "server_status_response"
    ALGORITHMS_RESPONSE = "algorithms_response"
    ALGORITHM_INFO_RESPONSE = "algorithm_info_response"
    ERROR_RESPONSE = "error_response"
    HEARTBEAT = "heartbeat"

@dataclass
class WebSocketMessage:
    """Standard WebSocket message format"""
    message_type: str
    message_id: str
    timestamp: str
    data: Dict[str, Any]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

@dataclass
class WebSocketConfig:
    """WebSocket server configuration"""
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 100
    heartbeat_interval: float = 30.0  # seconds
    connection_timeout: float = 300.0  # seconds
    max_message_size: int = 1024 * 1024  # 1MB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

# Default configuration
DEFAULT_CONFIG = WebSocketConfig()

# Message templates
def create_request_message(message_type: MessageType, data: Dict[str, Any], message_id: str = None) -> WebSocketMessage:
    """Create a request message"""
    import uuid
    from datetime import datetime
    
    if message_id is None:
        message_id = str(uuid.uuid4())
    
    return WebSocketMessage(
        message_type=message_type.value,
        message_id=message_id,
        timestamp=datetime.now().isoformat(),
        data=data
    )

def create_response_message(message_type: MessageType, data: Dict[str, Any], 
                          request_message_id: str, error: str = None) -> WebSocketMessage:
    """Create a response message"""
    from datetime import datetime
    
    return WebSocketMessage(
        message_type=message_type.value,
        message_id=request_message_id,  # Use same ID as request for correlation
        timestamp=datetime.now().isoformat(),
        data=data,
        error=error
    )

def create_error_response(error: str, request_message_id: str) -> WebSocketMessage:
    """Create an error response message"""
    from datetime import datetime
    
    return WebSocketMessage(
        message_type=MessageType.ERROR_RESPONSE.value,
        message_id=request_message_id,
        timestamp=datetime.now().isoformat(),
        data={},
        error=error
    ) 