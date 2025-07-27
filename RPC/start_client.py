#!/usr/bin/env python3
"""
RPC Client Startup Script
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rpc_client import AdvancedRPCClient

def main():
    print("Starting Enhanced RPC Algorithm Client")
    print("="*50)
    print("Client will connect to localhost:8080")
    print("="*50)
    
    try:
        client = AdvancedRPCClient()
        client.connect()
        
        if not client.connected:
            print("Cannot connect to server, exiting")
            return
        
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
        if 'client' in locals():
            client.disconnect()

if __name__ == "__main__":
    main() 