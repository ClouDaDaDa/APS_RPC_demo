#!/usr/bin/env python3
"""
Start script for Scheduling RPC Client
"""

import sys
import os
import argparse
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from RPC.scheduling_rpc_client import SchedulingRPCClient

def main():
    parser = argparse.ArgumentParser(description='Scheduling RPC Client')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--input-file', help='Input JSON file path')
    parser.add_argument('--alpha', type=float, default=0.7, 
                       help='Alpha parameter for weighted algorithm (default: 0.7)')
    parser.add_argument('--output-file', help='Output JSON file path')
    parser.add_argument('--demo', action='store_true', help='Run demo with example data')
    
    args = parser.parse_args()
    
    try:
        with SchedulingRPCClient(host=args.host, port=args.port) as client:
            if args.demo:
                # Run demo
                print("Running demo with example data...")
                from RPC.scheduling_rpc_client import demo_scheduling_client
                demo_scheduling_client()
                return
            
            # Load input data
            if args.input_file:
                if not os.path.exists(args.input_file):
                    print(f"Error: Input file '{args.input_file}' not found")
                    sys.exit(1)
                
                with open(args.input_file, 'r') as f:
                    input_data = json.load(f)
                input_filename = os.path.basename(args.input_file)
            else:
                print("Error: Input file is required (use --input-file)")
                sys.exit(1)
            
            # Run scheduling algorithm
            print(f"Running EST-EET weighted scheduling algorithm with alpha={args.alpha}...")
            result = client.est_eet_weighted_scheduling(input_data, alpha=args.alpha, input_filename=input_filename)
            
            # Print result
            print(f"\nAlgorithm: est_eet_weighted")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                print(f"Makespan: {result.get('makespan')}")
                print(f"Execution Time: {result.get('execution_time', 0):.2f} seconds")
                print(f"Output File: {result.get('output_file', 'N/A')}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            # Save result to file if specified
            if args.output_file:
                client.save_result_to_file(result, args.output_file)
                print(f"Result saved to: {args.output_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 