# Enhanced RPC Framework
import json
import socket
import inspect
import logging
import time
from threading import Thread, Lock
from typing import Any, Dict, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """Connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

@dataclass
class RPCConfig:
    """RPC configuration class"""
    buffer_size: int = 4096
    timeout: float = 60.0
    max_connections: int = 10
    enable_compression: bool = False
    enable_encryption: bool = False

class RPCError(Exception):
    """Custom RPC exception"""
    pass

class ConnectionError(RPCError):
    """Connection related errors"""
    pass

class MethodNotFoundError(RPCError):
    """Method not found error"""
    pass

class RPCServer:
    """
    Enhanced RPC Server with improved error handling, logging, and configuration
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080, config: Optional[RPCConfig] = None) -> None:
        self.host = host
        self.port = port
        self.address = (host, port)
        self._methods: Dict[str, Callable] = {}
        self._config = config or RPCConfig()
        self._lock = Lock()
        self._active_connections = 0
        self._running = False
        
        logger.info(f"RPC Server initialized on {self.address}")
    
    def register_method(self, function: Callable) -> None:
        """
        Register a function as an RPC method
        
        Args:
            function: The function to register
            
        Raises:
            ValueError: If function is not callable
        """
        if not callable(function):
            raise ValueError(f"Function must be callable, got {type(function)}")
        
        with self._lock:
            self._methods[function.__name__] = function
            logger.info(f"Registered method: {function.__name__}")
    
    def register_instance(self, instance: Any) -> None:
        """
        Register all public methods of an instance
        
        Args:
            instance: The instance whose methods should be registered
            
        Raises:
            ValueError: If instance is None or not an object
        """
        if instance is None:
            raise ValueError("Instance cannot be None")
        
        try:
            with self._lock:
                for function_name, function in inspect.getmembers(instance, predicate=inspect.ismethod):
                    if not function_name.startswith('_'):
                        self._methods[function_name] = function
                        logger.info(f"Registered instance method: {function_name}")
        except Exception as e:
            raise ValueError(f"Failed to register instance: {e}")
    
    def _validate_request(self, data: Any) -> tuple:
        """
        Validate incoming request data
        
        Args:
            data: The request data to validate
            
        Returns:
            tuple: (function_name, args, kwargs)
            
        Raises:
            RPCError: If data is invalid
        """
        if not isinstance(data, (list, tuple)) or len(data) != 3:
            raise RPCError("Invalid request format: expected (function_name, args, kwargs)")
        
        function_name, args, kwargs = data
        
        if not isinstance(function_name, str):
            raise RPCError("Function name must be a string")
        
        if not isinstance(args, (list, tuple)):
            raise RPCError("Args must be a list or tuple")
        
        if not isinstance(kwargs, dict):
            raise RPCError("Kwargs must be a dictionary")
        
        return function_name, args, kwargs
    
    def _handle_client(self, client: socket.socket, address: tuple) -> None:
        """
        Handle client connection and requests
        
        Args:
            client: Client socket
            address: Client address
        """
        logger.info(f"Managing requests from {address}")
        
        try:
            with self._lock:
                self._active_connections += 1
            
            while self._running:
                try:
                    # Set timeout for receiving data
                    client.settimeout(self._config.timeout)
                    
                    # Receive data
                    data = client.recv(self._config.buffer_size)
                    if not data:
                        logger.info(f"Client {address} disconnected (no data)")
                        break
                    
                    # Parse request
                    try:
                        request_data = json.loads(data.decode('utf-8'))
                        function_name, args, kwargs = self._validate_request(request_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from {address}: {e}")
                        continue
                    except RPCError as e:
                        logger.error(f"Invalid request from {address}: {e}")
                        continue
                    
                    logger.info(f"Request from {address}: {function_name}({args})")
                    
                    # Execute method
                    try:
                        if function_name not in self._methods:
                            raise MethodNotFoundError(f"Method '{function_name}' not found")
                        
                        start_time = time.time()
                        response = self._methods[function_name](*args, **kwargs)
                        execution_time = time.time() - start_time
                        
                        logger.info(f"Method {function_name} executed in {execution_time:.4f}s")
                        
                        # Send response
                        response_data = {
                            "success": True,
                            "result": response,
                            "execution_time": execution_time,
                            "timestamp": time.time()
                        }
                        
                    except MethodNotFoundError as e:
                        response_data = {
                            "success": False,
                            "error": str(e),
                            "error_type": "MethodNotFoundError"
                        }
                        logger.warning(f"Method not found: {e}")
                        
                    except Exception as e:
                        response_data = {
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        logger.error(f"Method execution error: {e}")
                    
                    # Send response
                    try:
                        response_json = json.dumps(response_data, ensure_ascii=False)
                        client.sendall(response_json.encode('utf-8'))
                    except Exception as e:
                        logger.error(f"Failed to send response to {address}: {e}")
                        break
                        
                except socket.timeout:
                    logger.debug(f"Timeout waiting for data from {address}")
                    continue
                except Exception as e:
                    logger.error(f"Error handling request from {address}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Unexpected error with client {address}: {e}")
        finally:
            try:
                client.close()
            except:
                pass
            
            with self._lock:
                self._active_connections -= 1
            
            logger.info(f"Completed requests from {address}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get server status information
        
        Returns:
            Dict containing server status
        """
        return {
            "running": self._running,
            "address": self.address,
            "active_connections": self._active_connections,
            "registered_methods": list(self._methods.keys()),
            "config": {
                "buffer_size": self._config.buffer_size,
                "timeout": self._config.timeout,
                "max_connections": self._config.max_connections
            }
        }
    
    def run(self) -> None:
        """Start the RPC server"""
        self._running = True
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Set socket options
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(self.address)
                sock.listen(self._config.max_connections)
                sock.settimeout(1.0)  # Allow for graceful shutdown
                
                logger.info(f"Server {self.address} running")
                logger.info(f"Configuration: {self._config}")
                
                while self._running:
                    try:
                        client, address = sock.accept()
                        
                        # Check connection limit
                        if self._active_connections >= self._config.max_connections:
                            logger.warning(f"Connection limit reached, rejecting {address}")
                            client.close()
                            continue
                        
                        # Start client handler thread
                        thread = Thread(
                            target=self._handle_client,
                            args=(client, address),
                            daemon=True
                        )
                        thread.start()
                        
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self._running:
                            logger.error(f"Error accepting connection: {e}")
                        
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self._running = False
            logger.info("Server stopped")
    
    def stop(self) -> None:
        """Stop the server gracefully"""
        self._running = False
        logger.info("Stopping server...")


class RPCClient:
    """
    Enhanced RPC Client with improved error handling and connection management
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8080, config: Optional[RPCConfig] = None) -> None:
        self.host = host
        self.port = port
        self.address = (host, port)
        self._config = config or RPCConfig()
        self._socket: Optional[socket.socket] = None
        self._status = ConnectionStatus.DISCONNECTED
        self._lock = Lock()
        
        logger.info(f"RPC Client initialized for {self.address}")
    
    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self._status
    
    @property
    def connected(self) -> bool:
        """Check if client is connected"""
        return self._status == ConnectionStatus.CONNECTED and self._socket is not None
    
    def connect(self, timeout: Optional[float] = None) -> None:
        """
        Connect to the RPC server
        
        Args:
            timeout: Connection timeout in seconds
            
        Raises:
            ConnectionError: If connection fails
        """
        if self.connected:
            logger.warning("Already connected")
            return
        
        self._status = ConnectionStatus.CONNECTING
        
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(timeout or self._config.timeout)
            self._socket.connect(self.address)
            self._status = ConnectionStatus.CONNECTED
            
            logger.info(f"Connected to server {self.address}")
            
        except socket.timeout:
            self._status = ConnectionStatus.ERROR
            raise ConnectionError(f"Connection timeout to {self.address}")
        except ConnectionRefusedError:
            self._status = ConnectionStatus.ERROR
            raise ConnectionError(f"Connection refused by {self.address}")
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            raise ConnectionError(f"Failed to connect to {self.address}: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from the server"""
        if not self.connected:
            return
        
        try:
            if self._socket:
                self._socket.close()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self._socket = None
            self._status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from server")
    
    def _send_request(self, method_name: str, args: tuple, kwargs: dict) -> Any:
        """
        Send request to server and receive response
        
        Args:
            method_name: Name of the method to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Response from server
            
        Raises:
            ConnectionError: If not connected
            RPCError: If request fails
        """
        if not self.connected:
            raise ConnectionError("Not connected to server")
        
        try:
            # Prepare request
            request_data = (method_name, args, kwargs)
            request_json = json.dumps(request_data, ensure_ascii=False)
            
            # Send request
            self._socket.sendall(request_json.encode('utf-8'))
            
            # Receive response
            response_data = b""
            while True:
                chunk = self._socket.recv(self._config.buffer_size)
                if not chunk:
                    break
                response_data += chunk
                
                # Try to parse JSON (in case response is complete)
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    break
                except json.JSONDecodeError:
                    continue
            
            # Parse response
            if not response_data:
                raise RPCError("No response received from server")
            
            response = json.loads(response_data.decode('utf-8'))
            
            # Check for errors
            if not response.get("success", True):
                error_msg = response.get("error", "Unknown error")
                error_type = response.get("error_type", "RPCError")
                
                if error_type == "MethodNotFoundError":
                    raise MethodNotFoundError(error_msg)
                else:
                    raise RPCError(error_msg)
            
            return response.get("result")
            
        except json.JSONDecodeError as e:
            raise RPCError(f"Invalid JSON response: {e}")
        except Exception as e:
            if isinstance(e, (RPCError, MethodNotFoundError)):
                raise
            raise RPCError(f"Request failed: {e}")
    
    def __getattr__(self, method_name: str) -> Callable:
        """
        Dynamic method creation for RPC calls
        
        Args:
            method_name: Name of the method to call
            
        Returns:
            Callable function that will make the RPC call
        """
        def execute(*args, **kwargs):
            return self._send_request(method_name, args, kwargs)
        
        return execute
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Backward compatibility aliases
def registerMethod(self, function):
    """Alias for register_method for backward compatibility"""
    return self.register_method(function)

def registerInstance(self, instance):
    """Alias for register_instance for backward compatibility"""
    return self.register_instance(instance)

# Add backward compatibility methods to classes
RPCServer.registerMethod = registerMethod
RPCServer.registerInstance = registerInstance 