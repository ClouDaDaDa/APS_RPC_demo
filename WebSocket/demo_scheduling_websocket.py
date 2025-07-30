#!/usr/bin/env python3
"""
Demo script for WebSocket Scheduling Service
Demonstrates complete frontend-backend interaction functionality
"""

import json
import os
import sys
import asyncio
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from WebSocket.websocket_client import WebSocketSchedulingClient
from WebSocket.websocket_config import create_request_message, MessageType

async def demo_with_real_data():
    """Demo using real input data"""

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
        
        # Connect to WebSocket server
        async with WebSocketSchedulingClient() as client:
            # Get server status
            status = await client.get_server_status()
            print(f"\nServer Status: {status}")
            
            # Get supported algorithms
            algorithms = await client.get_supported_algorithms()
            print(f"Supported Algorithms: {algorithms}")
            
            results = {}
            # 1. Test EST-EET Weighted
            alpha_values = [0.5, 0.7, 0.9]
            for alpha in alpha_values:
                print(f"\n--- Testing EST-EET Weighted with alpha={alpha} ---")
                try:
                    result = await client.est_eet_weighted_scheduling(
                        input_data,
                        alpha=alpha,
                        input_filename=os.path.basename(input_file)
                    )
                    results[f"est_eet_weighted_alpha_{alpha}"] = result
                    if result.get('status') == 'success':
                        print(f"EST-EET Weighted (alpha={alpha}) completed successfully")
                        print(f"  Makespan: {result.get('makespan')}")
                        print(f"  Execution Time: {result.get('execution_time', 0):.2f} seconds")
                        print(f"  Output File: {result.get('output_file')}")
                    else:
                        print(f"EST-EET Weighted (alpha={alpha}) failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"EST-EET Weighted (alpha={alpha}) error: {e}")
                    results[f"est_eet_weighted_alpha_{alpha}"] = {"error": str(e)}

            # 2. Test EST-SPT Weighted
            print(f"\n--- Testing EST-SPT Weighted (alpha=0.7) ---")
            try:
                spt_request = {
                    "algorithm": "est_spt_weighted",
                    "input_data": input_data,
                    "alpha": 0.7,
                    "input_filename": os.path.basename(input_file)
                }
                spt_message = create_request_message(
                    MessageType.SCHEDULING_REQUEST,
                    spt_request,
                    f"scheduling_spt_{int(time.time())}"
                )
                spt_response = await client.send_message(spt_message)
                result = spt_response.data
                results["est_spt_weighted"] = result
                if result.get('status') == 'success':
                    print(f"EST-SPT Weighted completed successfully")
                    print(f"  Makespan: {result.get('makespan')}")
                    print(f"  Execution Time: {result.get('execution_time', 0):.2f} seconds")
                    print(f"  Output File: {result.get('output_file')}")
                else:
                    print(f"EST-SPT Weighted failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"EST-SPT Weighted error: {e}")
                results["est_spt_weighted"] = {"error": str(e)}

            # 3. Test GA
            print(f"\n--- Testing Genetic Algorithm (GA) ---")
            try:
                ga_request = {
                    "algorithm": "ga",
                    "input_data": input_data,
                    "alpha": 0.7,
                    "pop": 30,
                    "gen": 10,
                    "input_filename": os.path.basename(input_file)
                }
                ga_message = create_request_message(
                    MessageType.SCHEDULING_REQUEST,
                    ga_request,
                    f"scheduling_ga_{int(time.time())}"
                )
                ga_response = await client.send_message(ga_message)
                result = ga_response.data
                results["ga"] = result
                if result.get('status') == 'success':
                    print(f"GA completed successfully")
                    print(f"  Makespan: {result.get('makespan')}")
                    print(f"  Execution Time: {result.get('execution_time', 0):.2f} seconds")
                    print(f"  Output File: {result.get('output_file')}")
                else:
                    print(f"GA failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"GA error: {e}")
                results["ga"] = {"error": str(e)}

            # Summary
            successful = sum(1 for r in results.values() if r.get('status') == 'success')
            total = len(results)
            print(f"Successful runs: {successful}/{total}")
            if successful > 0:
                print("WebSocket service is working correctly!")
                return True
            else:
                print("All runs failed")
                return False
    except Exception as e:
        print(f"Demo error: {e}")
        return False

async def demo_server_info():
    """Demo server information"""
    print("\n=== Server Information Demo ===")
    
    try:
        async with WebSocketSchedulingClient() as client:
            # Get server status
            status = await client.get_server_status()
            print(f"Server Status: {status}")
            
            # Get supported algorithms
            algorithms = await client.get_supported_algorithms()
            print(f"Supported Algorithms: {algorithms}")
            
            # Get algorithm info
            info = await client.get_algorithm_info('est_eet_weighted')
            print(f"Algorithm Info: {info}")
                
    except Exception as e:
        print(f"Server info demo error: {e}")


async def main():
    """Main demo function"""
    
    # Demo server information
    await demo_server_info()
    
    # Demo with real data
    success = await demo_with_real_data()
    
    if success:
        print("\nDemo completed successfully!")
    else:
        print("\nDemo failed.")

if __name__ == '__main__':
    asyncio.run(main()) 