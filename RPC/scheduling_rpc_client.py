import json
import time
import logging
from typing import Dict, Any, Optional
from RPC.rpc_class import RPCClient, RPCConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchedulingRPCClient:
    """RPC client for scheduling algorithms"""
    
    def __init__(self, host='localhost', port=8080):
        # Create configuration for scheduling RPC client
        config = RPCConfig(
            buffer_size=8192,  # Larger buffer for scheduling data
            timeout=300.0,     # Longer timeout for complex scheduling algorithms
        )
        self.client = RPCClient(host, port, config)
        self.host = host
        self.port = port
    
    def connect(self):
        """Connect to the RPC server"""
        try:
            self.client.connect()
            logger.info(f"Connected to scheduling RPC server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the RPC server"""
        try:
            self.client.disconnect()
            logger.info("Disconnected from scheduling RPC server")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def get_server_status(self):
        """Get server status"""
        try:
            return self.client.get_server_status()
        except Exception as e:
            logger.error(f"Error getting server status: {e}")
            return {"error": str(e)}
    
    def get_supported_algorithms(self):
        """Get list of supported algorithms"""
        try:
            return self.client.get_supported_algorithms()
        except Exception as e:
            logger.error(f"Error getting supported algorithms: {e}")
            return {"error": str(e)}
    
    def get_algorithm_info(self, algorithm_name):
        """Get information about a specific algorithm"""
        try:
            return self.client.get_algorithm_info(algorithm_name)
        except Exception as e:
            logger.error(f"Error getting algorithm info: {e}")
            return {"error": str(e)}
    
    def est_eet_weighted_scheduling(self, input_data: Dict[str, Any], alpha: float = 0.7, input_filename: str = "input.json"):
        """Run EST-EET weighted scheduling algorithm"""
        try:
            data = {
                "input_data": input_data,
                "input_filename": input_filename,
                "alpha": alpha
            }
            
            logger.info(f"Running EST-EET weighted scheduling algorithm with alpha={alpha}...")
            start_time = time.time()
            result = self.client.est_eet_weighted_scheduling_wrapper(data)
            execution_time = time.time() - start_time
            
            logger.info(f"EST-EET weighted scheduling completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in EST-EET weighted scheduling: {e}")
            return {"error": str(e)}
    
    def load_input_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load input data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded input data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading input file {file_path}: {e}")
            return {}
    
    def save_result_to_file(self, result: Dict[str, Any], file_path: str):
        """Save result to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving result to {file_path}: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

def demo_scheduling_client():
    """Demo function showing how to use the scheduling RPC client"""
    
    # Example input data (you can load from file)
    example_input = {
        "scheduling_instance": {
            "instance_id": "DEMO_W2_O2_P2",
            "workshop": {
                "workshop_id": "WSHOP1",
                "workstations": [
                    {
                        "workstation_id": "WS1",
                        "capacity": {"buffer_capacity": 1000},
                        "machines": [
                            {
                                "machine_id": "M1",
                                "capacity": {"max_parallel_jobs": 1},
                                "availability": {
                                    "status": "available",
                                    "unavailable_periods": []
                                }
                            },
                            {
                                "machine_id": "M2",
                                "capacity": {"max_parallel_jobs": 1},
                                "availability": {
                                    "status": "available",
                                    "unavailable_periods": []
                                }
                            }
                        ]
                    }
                ]
            },
            "work_order": {
                "work_order_id": "WO1",
                "orders": [
                    {
                        "order_id": "O1",
                        "order_priority": 3,
                        "release_time": "2024-01-01T08:00:00",
                        "due_date": "2024-01-01T18:00:00",
                        "products": [
                            {
                                "product_id": "P1",
                                "quantity": 2,
                                "operations": [
                                    {
                                        "operation_id": "OP1",
                                        "operation_sequence": 1,
                                        "process_workstation": "WS1",
                                        "eligible_machines": [
                                            {
                                                "machine_id": "M1",
                                                "standard_duration": 2.0
                                            },
                                            {
                                                "machine_id": "M2",
                                                "standard_duration": 1.5
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
    }
    
    # Use context manager for automatic connection/disconnection
    with SchedulingRPCClient() as client:
        # Get server status
        status = client.get_server_status()
        print(f"Server Status: {status}")
        
        # Get supported algorithms
        algorithms = client.get_supported_algorithms()
        print(f"Supported Algorithms: {algorithms}")
        
        # Run EST-EET weighted scheduling with different alpha values
        print("\n=== Running EST-EET Weighted Scheduling ===")
        for alpha in [0.5, 0.7, 0.9]:
            result = client.est_eet_weighted_scheduling(example_input, alpha=alpha, input_filename="demo_input.json")
            print(f"Alpha={alpha}: {result}")

if __name__ == '__main__':
    demo_scheduling_client() 