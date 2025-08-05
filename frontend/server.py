#!/usr/bin/env python3
"""
Simple HTTP server to serve the Gantt Chart frontend application.
Run this script and open http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

def main():
    # Change to the frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    PORT = 8000
    
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers to allow local file access
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def do_GET(self):
            # Serve index.html for root path
            if self.path == '/':
                self.path = '/index.html'
            return super().do_GET()
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"Gantt Chart Server starting...")
            print(f"Open your browser and navigate to: http://localhost:{PORT}")
            print(f"Press Ctrl+C to stop the server")
            print("-" * 50)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()