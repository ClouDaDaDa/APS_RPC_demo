#!/usr/bin/env python3
"""
Start script for Scheduling RPC Server
"""

import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from RPC.scheduling_rpc_server import SchedulingRPCServer

def main():
    parser = argparse.ArgumentParser(description='Start Scheduling RPC Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set log level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print(f"Starting Scheduling RPC Server on {args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        server = SchedulingRPCServer(host=args.host, port=args.port)
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 