# WebSocket Scheduling Service

## Structure
```
WebSocket/
├── __init__.py                 # Module initialization
├── websocket_config.py         # Configuration and message format
├── scheduling_processor.py     # Scheduling algorithm processor
├── message_handler.py          # Message handler
├── websocket_server.py         # WebSocket server
├── websocket_client.py         # WebSocket client
└── demo_scheduling_websocket.py # Demo script
```

## Message format

### Request message
```json
{
    "message_type": "scheduling_request",
    "message_id": "unique_id",
    "timestamp": "2024-01-01T08:00:00",
    "data": {
        "algorithm": "est_eet_weighted",
        "input_data": {...},
        "alpha": 0.7,
        "input_filename": "input.json"
    }
}
```

### Response message
```json
{
    "message_type": "scheduling_response",
    "message_id": "unique_id",
    "timestamp": "2024-01-01T08:00:00",
    "data": {
        "algorithm": "est_eet_weighted",
        "status": "success",
        "makespan": 15.5,
        "execution_time": 2.3,
        "output_file": "path/to/output.json",
        "output_data": {...}
    }
}
```

## Usage
### 1. Start server
```bash
python -m WebSocket.websocket_server
```

### 2. Client usage
```bash
python WebSocket.websocket_client
```

### 3. Demo
```bash
python WebSocket.demo_scheduling_websocket
```

## Configuration parameters

### Server configuration
- `host`: Server listening address (default: "0.0.0.0")
- `port`: Server port (default: 8765)
- `max_connections`: Maximum number of connections (default: 100)
- `heartbeat_interval`: Heartbeat interval (default: 30 seconds)
- `connection_timeout`: Connection timeout (default: 300 seconds)
- `max_message_size`: Maximum message size (default: 1MB)

### Client configuration
- `host`: Server address (default: "localhost")
- `port`: Server port (default: 8765)
