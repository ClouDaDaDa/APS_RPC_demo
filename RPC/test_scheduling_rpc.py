#!/usr/bin/env python3
"""
Test script for Scheduling RPC functionality
"""

import json
import time
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from RPC.scheduling_rpc_client import SchedulingRPCClient

def create_test_input():
    """Create test input data"""
    return {
        "scheduling_instance": {
            "instance_id": "TEST_W2_O2_P2",
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
                    },
                    {
                        "workstation_id": "WS2",
                        "capacity": {"buffer_capacity": 800},
                        "machines": [
                            {
                                "machine_id": "M3",
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
                                    },
                                    {
                                        "operation_id": "OP2",
                                        "operation_sequence": 2,
                                        "process_workstation": "WS2",
                                        "eligible_machines": [
                                            {
                                                "machine_id": "M3",
                                                "standard_duration": 1.0
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "order_id": "O2",
                        "order_priority": 2,
                        "release_time": "2024-01-01T09:00:00",
                        "due_date": "2024-01-01T17:00:00",
                        "products": [
                            {
                                "product_id": "P2",
                                "quantity": 1,
                                "operations": [
                                    {
                                        "operation_id": "OP1",
                                        "operation_sequence": 1,
                                        "process_workstation": "WS1",
                                        "eligible_machines": [
                                            {
                                                "machine_id": "M1",
                                                "standard_duration": 1.0
                                            },
                                            {
                                                "machine_id": "M2",
                                                "standard_duration": 1.2
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

def test_server_connection(client):
    """Test server connection and basic functionality"""
    print("=== Testing Server Connection ===")
    
    # Test server status
    status = client.get_server_status()
    print(f"Server Status: {status}")
    
    # Test supported algorithms
    algorithms = client.get_supported_algorithms()
    print(f"Supported Algorithms: {algorithms}")
    
    # Test algorithm info
    info = client.get_algorithm_info('est_eet_weighted')
    print(f"Algorithm Info for est_eet_weighted: {info}")
    
    print()

def test_scheduling_algorithm(client, test_input):
    """Test EST-EET weighted scheduling algorithm"""
    print("=== Testing EST-EET Weighted Scheduling Algorithm ===")
    
    alpha_values = [0.5, 0.7, 0.9]
    results = {}
    
    for alpha in alpha_values:
        print(f"\n--- Testing EST-EET Weighted with alpha={alpha} ---")
        
        try:
            result = client.est_eet_weighted_scheduling(test_input, alpha=alpha, input_filename="test_input.json")
            
            results[alpha] = result
            
            if result.get('status') == 'success':
                print(f"EST-EET Weighted (alpha={alpha}) completed successfully")
                print(f"  Makespan: {result.get('makespan')}")
                print(f"  Execution Time: {result.get('execution_time', 0):.2f} seconds")
            else:
                print(f"EST-EET Weighted (alpha={alpha}) failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"EST-EET Weighted (alpha={alpha}) error: {e}")
            results[alpha] = {"error": str(e)}
    
    return results

def test_with_real_input_file(client):
    """Test with real input file from Data directory"""
    print("\n=== Testing with Real Input File ===")
    
    # Try to find a real input file
    input_files = [
        "Data/InputData/input_data_example_W3_O3_P10.json",
        "Data/InputData/input_data_example_W3_O3_P46.json"
    ]
    
    input_file = None
    for file_path in input_files:
        if os.path.exists(file_path):
            input_file = file_path
            break
    
    if input_file is None:
        print("No real input files found, skipping real file test")
        return
    
    print(f"Using input file: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            real_input = json.load(f)
        
        # Test with EST-EET weighted algorithm
        result = client.est_eet_weighted_scheduling(real_input, alpha=0.7, input_filename=os.path.basename(input_file))
        
        if result.get('status') == 'success':
            print(f"Real file test completed successfully")
            print(f"  Makespan: {result.get('makespan')}")
            print(f"  Execution Time: {result.get('execution_time', 0):.2f} seconds")
            print(f"  Output File: {result.get('output_file')}")
        else:
            print(f"Real file test failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Real file test error: {e}")

def main():
    """Main test function"""
    print("Scheduling RPC Test Suite")
    print("=" * 50)
    
    # Create test input
    test_input = create_test_input()
    
    try:
        with SchedulingRPCClient() as client:
            # Test server connection
            test_server_connection(client)
            
            # Test scheduling algorithm
            results = test_scheduling_algorithm(client, test_input)
            
            # Test with real input file
            test_with_real_input_file(client)
            
            # Summary
            print("\n=== Test Summary ===")
            successful = sum(1 for r in results.values() if r.get('status') == 'success')
            total = len(results)
            print(f"Successful runs: {successful}/{total}")
            
            if successful == total:
                print("All tests passed!")
            else:
                print("Some tests failed!")
                
    except Exception as e:
        print(f"Test suite error: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 