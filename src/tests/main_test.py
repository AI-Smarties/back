from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_server_sends_ready_signal():
    with client.websocket_connect("/ws/") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "control"
        assert data["cmd"] == "ready"

def test_server_handles_invalid_json():
    with client.websocket_connect("/ws/") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_text("not a json")
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Invalid JSON"

def test_server_handles_unknown_command():
    with client.websocket_connect("/ws/") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"type": "control", "cmd": "unknown"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Unknown command"

def test_server_handles_unknown_message_type():
    with client.websocket_connect("/ws/") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"type": "unknown", "cmd": "start"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Unknown message type"

def test_server_handles_missing_type():
    with client.websocket_connect("/ws/") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"cmd": "start"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Missing type in message"

def test_server_handles_missing_command():
    with client.websocket_connect("/ws/") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"type": "control"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Missing command in control message"

def test_server_handles_audio_before_start():
    with client.websocket_connect("/ws/") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_bytes(b"audio chunk")
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Gemini Live not started"
