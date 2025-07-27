import json
import time
import logging
from datetime import datetime
from RPC.rpc_class import RPCServer, RPCConfig
from RPC.scheduling_algorithms import scheduling_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduling_rpc_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchedulingRPCServer:
    """RPC server for scheduling algorithms"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        # Create configuration for scheduling RPC server
        config = RPCConfig(
            buffer_size=8192,  # Larger buffer for scheduling data
            timeout=300.0,     # Longer timeout for complex scheduling algorithms
            max_connections=20  # More connections for multiple clients
        )
        self.server = RPCServer(host, port, config)
        self.request_count = 0
        self.start_time = datetime.now()
        
        # Register scheduling methods
        self._register_scheduling_methods()
        
        # Register server management methods
        self._register_server_methods()
    
    def _register_scheduling_methods(self):
        """Register all scheduling algorithm methods"""
        # EST-EET weighted algorithm
        self.server.register_method(self.est_eet_weighted_scheduling_wrapper)
        
        logger.info("Scheduling algorithm methods registered")
    
    def _register_server_methods(self):
        """Register server management methods"""
        self.server.register_method(self.get_server_status)
        self.server.register_method(self.get_supported_algorithms)
        self.server.register_method(self.get_algorithm_info)
    
    def _log_request(self, method_name, data, response):
        """Log request details"""
        self.request_count += 1
        logger.info(f"Request #{self.request_count}: {method_name}")
        logger.info(f"Input data size: {len(json.dumps(data, ensure_ascii=False))} bytes")
        logger.info(f"Response status: {response.get('status', 'unknown')}")
        if response.get('status') == 'success':
            logger.info(f"Algorithm: {response.get('algorithm')}, Makespan: {response.get('makespan')}")
    
    def est_eet_weighted_scheduling_wrapper(self, data):
        """EST-EET weighted scheduling wrapper"""
        try:
            start_time = time.time()
            result = scheduling_processor.est_eet_weighted_scheduling(data)
            execution_time = time.time() - start_time
            
            self._log_request("est_eet_weighted_scheduling", data, result)
            return result
        except Exception as e:
            error_response = {"error": str(e), "algorithm": "est_eet_weighted_scheduling"}
            logger.error(f"EST-EET weighted scheduling error: {e}")
            return error_response
    
    def get_server_status(self):
        """Get server status"""
        uptime = datetime.now() - self.start_time
        return {
            "status": "running",
            "uptime_seconds": uptime.total_seconds(),
            "request_count": self.request_count,
            "start_time": self.start_time.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_supported_algorithms(self):
        """Get list of supported algorithms"""
        return scheduling_processor.get_supported_algorithms()
    
    def get_algorithm_info(self, algorithm_name):
        """Get information about a specific algorithm"""
        return scheduling_processor.get_algorithm_info(algorithm_name)
    
    def run(self):
        """Run the RPC server"""
        logger.info("Starting Scheduling RPC Server...")
        logger.info(f"Server will listen on {self.server.host}:{self.server.port}")
        logger.info("Supported algorithms:")
        for alg_name, alg_desc in scheduling_processor.supported_algorithms.items():
            logger.info(f"  - {alg_name}: {alg_desc}")
        
        try:
            self.server.run()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            logger.info("Server shutdown complete")

if __name__ == '__main__':
    server = SchedulingRPCServer()
    server.run() 