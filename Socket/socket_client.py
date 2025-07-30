#!/usr/bin/env python3
"""
Simple Socket client test script
Used to test Socket.py server
"""

import json
import socket
import time

def send_request(host='127.0.0.1', port=23333, algorithm='est_eet', data=None):
    """
    Send request to Socket server
    """
    if data is None:
        data = {
            'workstations': 3,
            'machines': 5,
            'jobs': 10
        }
    # Construct request data
    request = {
        'key1': algorithm,
        'key2': data
    }
    # Pack as protocol format
    message = '<' + json.dumps(request, separators=(',', ':')) + '>'
    try:
        # Create Socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print(f"Connected to server {host}:{port}")
            # Send request
            print(f"Send request: {algorithm}")
            s.sendall(message.encode('utf-8'))
            # Receive response
            response = ''
            while True:
                piece = s.recv(8192)
                if not piece:
                    break
                piece = piece.decode('utf-8')
                if piece[0] == '<':
                    response = piece
                else:
                    response += piece
                if response[-1] == '>':
                    break
            # Parse response
            response_data = json.loads(response[1:-1])
            print(f"Received response:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
            return response_data
    except ConnectionRefusedError:
        print(f"Connection refused, please make sure the server is running")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_different_algorithms():
    """
    Test different scheduling algorithms
    """
    print("Socket.py server test")
    print("=" * 50)
    # Test EST-EET algorithm
    print("\n1. Test EST-EET algorithm")
    print("-" * 30)
    send_request(algorithm='est_eet', data={
        'workstations': 4,
        'machines': 6,
        'jobs': 15
    })
    time.sleep(1)
    # Test SPT algorithm
    print("\n2. Test SPT algorithm")
    print("-" * 30)
    send_request(algorithm='spt', data={
        'jobs': 12,
        'priority': 'high'
    })
    time.sleep(1)
    # Test random algorithm
    print("\n3. Test random algorithm")
    print("-" * 30)
    send_request(algorithm='random', data={
        'tasks': 8
    })
    time.sleep(1)
    # Test unknown algorithm
    print("\n4. Test unknown algorithm")
    print("-" * 30)
    send_request(algorithm='unknown', data={
        'test': 'data'
    })

def test_concurrent_requests():
    """
    Test concurrent requests
    """
    print("\n5. Test concurrent requests")
    print("-" * 30)
    import threading
    def single_request(thread_id):
        result = send_request(algorithm='est_eet', data={
            'workstations': 2,
            'machines': 3,
            'jobs': 5 + thread_id
        })
        print(f"Thread {thread_id} finished")
    # Create 3 concurrent threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=single_request, args=(i,))
        threads.append(t)
        t.start()
    # Wait for all threads to finish
    for t in threads:
        t.join()
    print("All concurrent requests finished")

if __name__ == '__main__':
    # Test different algorithms
    test_different_algorithms()
    # Test concurrent requests
    test_concurrent_requests()
    print("\nTest completed!") 