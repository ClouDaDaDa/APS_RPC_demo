import json
import time
import logging
from datetime import datetime
from RPC.rpc_class import RPCServer, RPCConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rpc_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedRPCServer:
    """Enhanced RPC server supporting algorithm processing and detailed logging"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        # Create configuration for enhanced RPC server
        config = RPCConfig(
            buffer_size=8192,  # Larger buffer for algorithm data
            timeout=60.0,      # Longer timeout for complex algorithms
            max_connections=20  # More connections for multiple clients
        )
        self.server = RPCServer(host, port, config)
        self.request_count = 0
        self.start_time = datetime.now()
        
        # Register all algorithm methods
        self._register_algorithms()
        
        # Register server management methods
        self._register_server_methods()
    
    def _register_algorithms(self):
        """Register all algorithm methods"""
        # Sorting algorithms
        self.server.register_method(self.bubble_sort_wrapper)
        self.server.register_method(self.quick_sort_wrapper)
        
        # Search algorithms
        self.server.register_method(self.binary_search_wrapper)
        
        # Mathematical algorithms
        self.server.register_method(self.fibonacci_wrapper)
        self.server.register_method(self.matrix_operations_wrapper)
        
        # Statistical analysis
        self.server.register_method(self.statistical_analysis_wrapper)
        
        # Graph algorithms
        self.server.register_method(self.graph_algorithms_wrapper)
        
        logger.info("All algorithm methods registered")
    
    def _register_server_methods(self):
        """Register server management methods"""
        self.server.register_method(self.get_server_status)
        self.server.register_method(self.get_available_algorithms)
        self.server.register_method(self.get_algorithm_info)
    
    def _log_request(self, method_name, data, response):
        """Log request details"""
        self.request_count += 1
        logger.info(f"Request #{self.request_count}: {method_name}")
        logger.info(f"Input data: {json.dumps(data, ensure_ascii=False, indent=2)}")
        logger.info(f"Response: {json.dumps(response, ensure_ascii=False, indent=2)}")
    
    def get_server_status(self):
        """Get server status"""
        uptime = datetime.now() - self.start_time
        status = {
            "server_status": "running",
            "uptime_seconds": uptime.total_seconds(),
            "uptime_formatted": str(uptime).split('.')[0],
            "total_requests": self.request_count,
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Server status query: {status}")
        return status
    
    def get_available_algorithms(self):
        """Get available algorithms list"""
        algorithms = {
            "sorting_algorithms": [
                "bubble_sort",
                "quick_sort"
            ],
            "search_algorithms": [
                "binary_search"
            ],
            "mathematical_algorithms": [
                "fibonacci_sequence",
                "matrix_operations"
            ],
            "statistical_algorithms": [
                "statistical_analysis"
            ],
            "graph_algorithms": [
                "graph_algorithms"
            ],
            "server_methods": [
                "get_server_status",
                "get_available_algorithms",
                "get_algorithm_info"
            ],
            "timestamp": datetime.now().isoformat()
        }
        return algorithms
    
    def get_algorithm_info(self, algorithm_name):
        """Get detailed information about specific algorithm"""
        algorithm_info = {
            "bubble_sort": {
                "description": "Bubble sort algorithm",
                "input_format": "List[Union[int, float]]",
                "output_format": "Dict with sorted data and performance metrics",
                "time_complexity": "O(n²)",
                "space_complexity": "O(1)",
                "example_input": [64, 34, 25, 12, 22, 11, 90]
            },
            "quick_sort": {
                "description": "Quick sort algorithm",
                "input_format": "List[Union[int, float]]",
                "output_format": "Dict with sorted data and performance metrics",
                "time_complexity": "O(n log n) average, O(n²) worst",
                "space_complexity": "O(log n)",
                "example_input": [64, 34, 25, 12, 22, 11, 90]
            },
            "binary_search": {
                "description": "Binary search algorithm",
                "input_format": "Dict with 'sorted_array' and 'target'",
                "output_format": "Dict with search result and steps",
                "time_complexity": "O(log n)",
                "space_complexity": "O(1)",
                "example_input": {"sorted_array": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "target": 7}
            },
            "fibonacci_sequence": {
                "description": "Fibonacci sequence calculation",
                "input_format": "Dict with 'n' and optional 'method'",
                "output_format": "Dict with sequence and sum",
                "time_complexity": "O(n)",
                "space_complexity": "O(n)",
                "example_input": {"n": 10, "method": "iterative"}
            },
            "matrix_operations": {
                "description": "Matrix operations (multiplication, transpose, determinant)",
                "input_format": "Dict with 'operation', 'matrix_a', and optional 'matrix_b'",
                "output_format": "Dict with operation result",
                "time_complexity": "O(n³) for multiply, O(n²) for transpose",
                "space_complexity": "O(n²)",
                "example_input": {
                    "operation": "multiply",
                    "matrix_a": [[1, 2], [3, 4]],
                    "matrix_b": [[5, 6], [7, 8]]
                }
            },
            "statistical_analysis": {
                "description": "Statistical analysis (mean, median, variance, etc.)",
                "input_format": "List[Union[int, float]]",
                "output_format": "Dict with statistical measures",
                "time_complexity": "O(n)",
                "space_complexity": "O(1)",
                "example_input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            "graph_algorithms": {
                "description": "Graph algorithms (DFS, BFS)",
                "input_format": "Dict with 'algorithm', 'graph', and 'start_node'",
                "output_format": "Dict with traversal result",
                "time_complexity": "O(V + E)",
                "space_complexity": "O(V)",
                "example_input": {
                    "algorithm": "dfs",
                    "graph": {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": [], "F": []},
                    "start_node": "A"
                }
            }
        }
        
        if algorithm_name in algorithm_info:
            result = algorithm_info[algorithm_name]
            result["algorithm_name"] = algorithm_name
            result["timestamp"] = datetime.now().isoformat()
            return result
        else:
            return {
                "error": f"Algorithm '{algorithm_name}' not found",
                "available_algorithms": list(algorithm_info.keys()),
                "timestamp": datetime.now().isoformat()
            }
    
    def run(self):
        """Start the server"""
        logger.info("Starting enhanced RPC server...")
        logger.info(f"Server address: {self.server.address}")
        logger.info("Available algorithms:")
        algorithms = self.get_available_algorithms()
        for category, algos in algorithms.items():
            if category != "timestamp":
                logger.info(f"  {category}: {', '.join(algos)}")
        
        try:
            self.server.run()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")

if __name__ == "__main__":
    server = AdvancedRPCServer()
    server.run() 