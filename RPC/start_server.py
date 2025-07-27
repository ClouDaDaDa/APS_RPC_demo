#!/usr/bin/env python3
"""
RPC Server Startup Script
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rpc_server import AdvancedRPCServer

def main():
    print("Starting Enhanced RPC Algorithm Server")
    print("="*50)
    print("Server will listen on all network interfaces on port 8080")
    print("Press Ctrl+C to stop server")
    print("="*50)
    
    try:
        server = AdvancedRPCServer()
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main() 