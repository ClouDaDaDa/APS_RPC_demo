import json
import time
import sys
from datetime import datetime
from rpc_class import RPCClient, RPCConfig, ConnectionError, RPCError, MethodNotFoundError

class AdvancedRPCClient:
    """Enhanced RPC client supporting multiple algorithm calls"""
    
    def __init__(self, host='localhost', port=8080):
        # Create configuration for enhanced RPC client
        config = RPCConfig(
            buffer_size=8192,  # Larger buffer for algorithm data
            timeout=60.0,      # Longer timeout for complex algorithms
        )
        self.client = RPCClient(host, port, config)
        self.connected = False
        self.request_history = []
    
    def connect(self):
        """Connect to server"""
        try:
            self.client.connect()
            self.connected = True
            print(f"Successfully connected to server {self.client.address}")
            
            # Get server status
            status = self.get_server_status()
            print(f"Server status: {status['server_status']}")
            print(f"Uptime: {status['uptime_formatted']}")
            print(f"Total requests: {status['total_requests']}")
            
        except ConnectionError as e:
            print(f"Connection failed: {e}")
            self.connected = False
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.connected = False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            print("Disconnected")
    
    def _log_request(self, method_name, data, response, execution_time):
        """Log request history"""
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "input_data": data,
            "response": response,
            "execution_time": execution_time
        }
        self.request_history.append(request_info)
    
    def _make_request(self, method_name, data):
        """Send request and log"""
        if not self.connected:
            print("Not connected to server")
            return None
        
        start_time = time.time()
        try:
            # Use dynamic method call
            method = getattr(self.client, method_name)
            response = method(data)
            execution_time = time.time() - start_time
            
            self._log_request(method_name, data, response, execution_time)
            return response
            
        except MethodNotFoundError as e:
            print(f"Method not found: {e}")
            return None
        except ConnectionError as e:
            print(f"Connection error: {e}")
            self.connected = False
            return None
        except RPCError as e:
            print(f"RPC error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def get_server_status(self):
        """Get server status"""
        return self.client.get_server_status()
    
    def get_available_algorithms(self):
        """Get available algorithms list"""
        return self.client.get_available_algorithms()
    
    def get_algorithm_info(self, algorithm_name):
        """Get algorithm information"""
        return self.client.get_algorithm_info(algorithm_name)
    
    def bubble_sort(self, data):
        """Bubble sort"""
        print(f"Executing bubble sort, data length: {len(data)}")
        return self.client.bubble_sort_wrapper(data)
    
    def quick_sort(self, data):
        """Quick sort"""
        print(f"Executing quick sort, data length: {len(data)}")
        return self.client.quick_sort_wrapper(data)
    
    def binary_search(self, sorted_array, target):
        """Binary search"""
        print(f"Executing binary search, target value: {target}")
        data = {"sorted_array": sorted_array, "target": target}
        return self.client.binary_search_wrapper(data)
    
    def fibonacci_sequence(self, n, method="iterative"):
        """Fibonacci sequence"""
        print(f"Calculating Fibonacci sequence, n={n}, method={method}")
        data = {"n": n, "method": method}
        return self.client.fibonacci_wrapper(data)
    
    def matrix_operations(self, operation, matrix_a, matrix_b=None):
        """Matrix operations"""
        print(f"Executing matrix operation: {operation}")
        data = {"operation": operation, "matrix_a": matrix_a}
        if matrix_b:
            data["matrix_b"] = matrix_b
        return self.client.matrix_operations_wrapper(data)
    
    def statistical_analysis(self, data):
        """Statistical analysis"""
        print(f"Executing statistical analysis, data length: {len(data)}")
        return self.client.statistical_analysis_wrapper(data)
    
    def graph_algorithms(self, algorithm, graph, start_node):
        """Graph algorithms"""
        print(f"Executing graph algorithm: {algorithm}, start node: {start_node}")
        data = {"algorithm": algorithm, "graph": graph, "start_node": start_node}
        return self.client.graph_algorithms_wrapper(data)
    
    def print_response(self, response, title="Response Result"):
        """Format and print response"""
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        
        if isinstance(response, dict):
            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                # Format output
                for key, value in response.items():
                    if key == "timestamp":
                        print(f"{key}: {value}")
                    elif key == "execution_time":
                        print(f"{key}: {value:.6f} seconds")
                    elif key == "algorithm":
                        print(f"{key}: {value}")
                    elif isinstance(value, list) and len(value) > 10:
                        print(f"{key}: [{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] (total {len(value)} items)")
                    else:
                        print(f"{key}: {value}")
        else:
            print(response)
        print(f"{'='*50}\n")
    
    def run_demo(self):
        """Run demonstration program"""
        print("Starting RPC Algorithm Demo")
        print("="*60)
        
        # 1. Get available algorithms
        print("Getting available algorithms list...")
        algorithms = self.get_available_algorithms()
        self.print_response(algorithms, "Available Algorithms List")
        
        # 2. Sorting algorithms demo
        print("Sorting Algorithms Demo")
        test_data = [64, 34, 25, 12, 22, 11, 90, 88, 76, 54, 32, 21, 19, 8, 5, 3, 1]
        
        # Bubble sort
        bubble_result = self.bubble_sort(test_data)
        self.print_response(bubble_result, "Bubble Sort Result")
        
        # Quick sort
        quick_result = self.quick_sort(test_data)
        self.print_response(quick_result, "Quick Sort Result")
        
        # 3. Search algorithms demo
        print("Search Algorithms Demo")
        sorted_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        search_target = 7
        
        search_result = self.binary_search(sorted_data, search_target)
        self.print_response(search_result, "Binary Search Result")
        
        # 4. Mathematical algorithms demo
        print("Mathematical Algorithms Demo")
        
        # Fibonacci sequence
        fib_result = self.fibonacci_sequence(15, "iterative")
        self.print_response(fib_result, "Fibonacci Sequence Result")
        
        # Matrix operations
        matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrix_b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        
        matrix_multiply = self.matrix_operations("multiply", matrix_a, matrix_b)
        self.print_response(matrix_multiply, "Matrix Multiplication Result")
        
        matrix_transpose = self.matrix_operations("transpose", matrix_a)
        self.print_response(matrix_transpose, "Matrix Transpose Result")
        
        # 5. Statistical analysis demo
        print("Statistical Analysis Demo")
        stats_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        stats_result = self.statistical_analysis(stats_data)
        self.print_response(stats_result, "Statistical Analysis Result")
        
        # 6. Graph algorithms demo
        print("Graph Algorithms Demo")
        graph = {
            "A": ["B", "C"],
            "B": ["D", "E"],
            "C": ["F", "G"],
            "D": [],
            "E": [],
            "F": [],
            "G": []
        }
        
        dfs_result = self.graph_algorithms("dfs", graph, "A")
        self.print_response(dfs_result, "Depth-First Search Result")
        
        bfs_result = self.graph_algorithms("bfs", graph, "A")
        self.print_response(bfs_result, "Breadth-First Search Result")
        
        # 7. Get algorithm information
        print("Getting Algorithm Details")
        bubble_info = self.get_algorithm_info("bubble_sort")
        self.print_response(bubble_info, "Bubble Sort Algorithm Information")
        
        # 8. Final status
        print("Final Server Status")
        final_status = self.get_server_status()
        self.print_response(final_status, "Final Server Status")
        
        # 9. Request history
        print(f"Request History (Total {len(self.request_history)} requests)")
        for i, request in enumerate(self.request_history, 1):
            print(f"  {i}. {request['method']} - {request['execution_time']:.6f}s")
    
    def interactive_mode(self):
        """Interactive mode"""
        print("Entering Interactive Mode")
        print("Type 'help' for help, type 'quit' to exit")
        
        while True:
            try:
                command = input("\nEnter command: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'help':
                    self.show_help()
                elif command.lower() == 'status':
                    status = self.get_server_status()
                    self.print_response(status, "Server Status")
                elif command.lower() == 'algorithms':
                    algorithms = self.get_available_algorithms()
                    self.print_response(algorithms, "Available Algorithms")
                elif command.lower() == 'history':
                    self.show_history()
                elif command.startswith('sort'):
                    self.handle_sort_command(command)
                elif command.startswith('search'):
                    self.handle_search_command(command)
                elif command.startswith('fibonacci'):
                    self.handle_fibonacci_command(command)
                elif command.startswith('matrix'):
                    self.handle_matrix_command(command)
                elif command.startswith('stats'):
                    self.handle_stats_command(command)
                elif command.startswith('graph'):
                    self.handle_graph_command(command)
                elif command.startswith('info'):
                    self.handle_info_command(command)
                else:
                    print("Unknown command, type 'help' for help")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
  help                    - Show this help information
  quit                    - Exit program
  status                  - Get server status
  algorithms              - Get available algorithms list
  history                 - Show request history
  
Algorithm Commands:
  sort bubble [data]      - Bubble sort (e.g., sort bubble [64,34,25,12,22])
  sort quick [data]       - Quick sort
  search [target] [sorted_array] - Binary search (e.g., search 7 [1,2,3,4,5,6,7,8,9,10])
  fibonacci [n] [method]  - Fibonacci sequence (e.g., fibonacci 10 iterative)
  matrix multiply [matrixA] [matrixB] - Matrix multiplication
  matrix transpose [matrix] - Matrix transpose
  stats [data]            - Statistical analysis (e.g., stats [1,2,3,4,5,6,7,8,9,10])
  graph dfs [start_node]  - Depth-first search
  graph bfs [start_node]  - Breadth-first search
  
Information Commands:
  info [algorithm_name]   - Get algorithm details (e.g., info bubble_sort)
        """
        print(help_text)
    
    def show_history(self):
        """Show request history"""
        if not self.request_history:
            print("No request history")
            return
        
        print(f"Request History (Total {len(self.request_history)} requests):")
        for i, request in enumerate(self.request_history, 1):
            print(f"  {i}. {request['method']} - {request['execution_time']:.6f}s - {request['timestamp']}")
    
    def handle_sort_command(self, command):
        """Handle sort command"""
        parts = command.split()
        if len(parts) < 3:
            print("Usage: sort [bubble|quick] [data]")
            return
        
        try:
            sort_type = parts[1]
            data_str = ' '.join(parts[2:])
            data = eval(data_str)  # Simple parsing, use safer method in production
            
            if sort_type == 'bubble':
                result = self.bubble_sort(data)
                self.print_response(result, "Bubble Sort Result")
            elif sort_type == 'quick':
                result = self.quick_sort(data)
                self.print_response(result, "Quick Sort Result")
            else:
                print("Unsupported sort type")
        except Exception as e:
            print(f"Command parsing error: {e}")
    
    def handle_search_command(self, command):
        """Handle search command"""
        parts = command.split()
        if len(parts) < 3:
            print("Usage: search [target] [sorted_array]")
            return
        
        try:
            target = int(parts[1])
            data_str = ' '.join(parts[2:])
            sorted_array = eval(data_str)
            
            result = self.binary_search(sorted_array, target)
            self.print_response(result, "Binary Search Result")
        except Exception as e:
            print(f"Command parsing error: {e}")
    
    def handle_fibonacci_command(self, command):
        """Handle fibonacci command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: fibonacci [n] [method]")
            return
        
        try:
            n = int(parts[1])
            method = parts[2] if len(parts) > 2 else "iterative"
            
            result = self.fibonacci_sequence(n, method)
            self.print_response(result, "Fibonacci Sequence Result")
        except Exception as e:
            print(f"Command parsing error: {e}")
    
    def handle_matrix_command(self, command):
        """Handle matrix command"""
        parts = command.split()
        if len(parts) < 3:
            print("Usage: matrix [multiply|transpose] [matrix_data]")
            return
        
        try:
            operation = parts[1]
            data_str = ' '.join(parts[2:])
            matrix_data = eval(data_str)
            
            if operation == "multiply" and len(parts) > 3:
                matrix_b_str = ' '.join(parts[3:])
                matrix_b = eval(matrix_b_str)
                result = self.matrix_operations(operation, matrix_data, matrix_b)
            else:
                result = self.matrix_operations(operation, matrix_data)
            
            self.print_response(result, f"Matrix {operation} Result")
        except Exception as e:
            print(f"Command parsing error: {e}")
    
    def handle_stats_command(self, command):
        """Handle stats command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: stats [data]")
            return
        
        try:
            data_str = ' '.join(parts[1:])
            data = eval(data_str)
            
            result = self.statistical_analysis(data)
            self.print_response(result, "Statistical Analysis Result")
        except Exception as e:
            print(f"Command parsing error: {e}")
    
    def handle_graph_command(self, command):
        """Handle graph command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: graph [dfs|bfs] [start_node]")
            return
        
        try:
            algorithm = parts[1]
            start_node = parts[2] if len(parts) > 2 else "A"
            
            # Use default graph
            graph = {
                "A": ["B", "C"],
                "B": ["D", "E"],
                "C": ["F", "G"],
                "D": [],
                "E": [],
                "F": [],
                "G": []
            }
            
            result = self.graph_algorithms(algorithm, graph, start_node)
            self.print_response(result, f"Graph Algorithm {algorithm} Result")
        except Exception as e:
            print(f"Command parsing error: {e}")
    
    def handle_info_command(self, command):
        """Handle info command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: info [algorithm_name]")
            return
        
        try:
            algorithm_name = parts[1]
            result = self.get_algorithm_info(algorithm_name)
            self.print_response(result, f"Algorithm {algorithm_name} Information")
        except Exception as e:
            print(f"Command parsing error: {e}")

def main():
    """Main function"""
    print("Enhanced RPC Client")
    print("="*50)
    
    # Create client
    client = AdvancedRPCClient()
    
    # Connect to server
    client.connect()
    
    if not client.connected:
        print("Cannot connect to server, exiting")
        return
    
    try:
        # Choose run mode
        print("\nChoose run mode:")
        print("1. Demo mode (run all algorithm demos)")
        print("2. Interactive mode (manual command input)")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "1":
            client.run_demo()
        elif choice == "2":
            client.interactive_mode()
        else:
            print("Invalid choice, running demo mode")
            client.run_demo()
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main() 