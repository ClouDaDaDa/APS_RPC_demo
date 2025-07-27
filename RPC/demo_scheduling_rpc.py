#!/usr/bin/env python3
"""
Demo script for Scheduling RPC Service
"""

import json
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from RPC.scheduling_rpc_client import SchedulingRPCClient

def demo_with_real_data():
    """Demo using real input data"""
    print("=== Scheduling RPC Demo with Real Data ===")
    
    # Use existing input file
    input_file = "Data/InputData/input_data_example_W3_O3_P10.json"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return False
    
    try:
        # Load input data
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        print(f"Loaded input data from: {input_file}")
        print(f"Input data size: {len(json.dumps(input_data))} bytes")
        
        # Connect to RPC server
        with SchedulingRPCClient() as client:
            # Get server status
            status = client.get_server_status()
            print(f"\nServer Status: {status}")
            
            # Get supported algorithms
            algorithms = client.get_supported_algorithms()
            print(f"Supported Algorithms: {algorithms}")
            
            # Test different alpha values
            alpha_values = [0.5, 0.7, 0.9]
            results = {}
            
            for alpha in alpha_values:
                print(f"\n--- Testing EST-EET Weighted with alpha={alpha} ---")
                
                try:
                    result = client.est_eet_weighted_scheduling(input_data, alpha=alpha, input_filename=os.path.basename(input_file))
                    
                    results[alpha] = result
                    
                    if result.get('status') == 'success':
                        print(f"EST-EET Weighted (alpha={alpha}) completed successfully")
                        print(f"  Makespan: {result.get('makespan')}")
                        print(f"  Execution Time: {result.get('execution_time', 0):.2f} seconds")
                        print(f"  Output File: {result.get('output_file')}")
                    else:
                        print(f"EST-EET Weighted (alpha={alpha}) failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"EST-EET Weighted (alpha={alpha}) error: {e}")
                    results[alpha] = {"error": str(e)}
            
            # Summary
            print("\n=== Demo Summary ===")
            successful = sum(1 for r in results.values() if r.get('status') == 'success')
            total = len(results)
            print(f"Successful runs: {successful}/{total}")
            
            if successful > 0:
                print("RPC service is working correctly!")
                return True
            else:
                print("All runs failed")
                return False
                
    except Exception as e:
        print(f"Demo error: {e}")
        return False

def demo_server_info():
    """Demo server information"""
    print("\n=== Server Information Demo ===")
    
    try:
        with SchedulingRPCClient() as client:
            # Get server status
            status = client.get_server_status()
            print(f"Server Status: {status}")
            
            # Get supported algorithms
            algorithms = client.get_supported_algorithms()
            print(f"Supported Algorithms: {algorithms}")
            
            # Get algorithm info
            info = client.get_algorithm_info('est_eet_weighted')
            print(f"Algorithm Info: {info}")
                
    except Exception as e:
        print(f"Server info demo error: {e}")

def main():
    """Main demo function"""
    print("Scheduling RPC Service Demo")
    print("=" * 50)
    
    # Demo server information
    demo_server_info()
    
    # Demo with real data
    success = demo_with_real_data()
    
    if success:
        print("\nDemo completed successfully!")
        print("The RPC service is working correctly.")
    else:
        print("\nDemo failed.")
        print("Please check the server status and try again.")

if __name__ == '__main__':
    main() 