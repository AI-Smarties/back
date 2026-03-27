import time
from unittest.mock import AsyncMock, patch
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

class GeminiInstanceBuilder:
    def __init__(self):
        self._ins = AsyncMock()
        self._ins.running = False
        async def start():
            self._ins.running = True
        self._ins.start = AsyncMock(side_effect=start)
        async def stop():
            self._ins.running = False
            return ""
        self._ins.stop = AsyncMock(side_effect=stop)


    def with_transcript(self, transcript):
        async def stop():
            self._ins.running = False
            return transcript
        self._ins.stop = AsyncMock(side_effect=stop)
        return self


    def build(self):
        return self._ins


def test_two_users_have_isolated_gemini_instances():
    with patch("main.GeminiLiveSession") as mock_gemini:
        instance1 = GeminiInstanceBuilder().build()

        instance2 = GeminiInstanceBuilder().build()

        mock_gemini.side_effect=[instance1, instance2]

        with client.websocket_connect("/ws/") as ws1:
            with client.websocket_connect("/ws/") as ws2:
                assert mock_gemini.call_count == 0
                ws1.send_json({"type": "control", "cmd": "start"})
                time.sleep(0.1)
                assert instance1.start.call_count == 1
                assert instance2.start.call_count == 0
                assert mock_gemini.call_count == 1

                ws2.send_json({"type": "control", "cmd": "start"})
                time.sleep(0.1)
                assert instance1.start.call_count == 1
                assert instance2.start.call_count == 1
                assert mock_gemini.call_count == 2

                # Each connection should create its own instance
def test_disconnect_cleans_up_only_own_session():
    with patch("main.GeminiLiveSession") as mock_gemini:

        instance1 = GeminiInstanceBuilder().build()
        instance2 = GeminiInstanceBuilder().build()


        mock_gemini.side_effect = [instance1, instance2]

        with client.websocket_connect("/ws/") as ws1:
            ws1.send_json({"type": "control", "cmd": "start"})

            with client.websocket_connect("/ws/") as ws2:
                assert instance2.call_count == 0
                ws2.send_json({"type": "control", "cmd": "start"})
                time.sleep(0.1)
                ws2.send_json({"type": "control", "cmd": "stop"})
                time.sleep(0.1)
                assert instance2.stop.call_count == 1
                assert instance1.stop.call_count == 0

            # ws1 still alive — only instance2 should have been stopped
            assert instance1.stop.call_count == 0
            ws1.send_json({"type": "control", "cmd": "stop"})
            time.sleep(0.1)
            assert instance1.stop.call_count == 1

def test_start_multiple_times_keeps_first_instance_of_gemini_live():
    with patch("main.GeminiLiveSession") as mock_gemini:
        ins1 = GeminiInstanceBuilder().build()
        ins2 = GeminiInstanceBuilder().build()

        mock_gemini.side_effect = [ins1, ins2]

        with client.websocket_connect("/ws/") as ws1:
            ws1.send_json({"type": "control", "cmd": "start"})

            with client.websocket_connect("/ws/") as ws2:
                ws2.send_json({"type": "control", "cmd": "start"})
                time.sleep(0.1)
                assert mock_gemini.call_count == 2
                ws2.send_json({"type": "control", "cmd": "start"})
                time.sleep(0.1)
                assert ins2.start.call_count == 1
                assert mock_gemini.call_count == 2
                ws2.send_json({"type": "control", "cmd": "stop"})
                time.sleep(0.1)

                assert ins2.stop.call_count == 1
                ws2.send_json({"type": "control", "cmd": "start"})
                time.sleep(0.1)
                assert ins2.start.call_count == 2


            # ws1 still alive — only instance2 should have been stopped
            assert ins1.stop.call_count == 0
            ws1.send_json({"type": "control", "cmd": "stop"})
            time.sleep(0.1)
            assert mock_gemini.call_count == 2

            assert ins1.stop.call_count == 1

def test_selected_category_is_used_on_stop():
    with patch("main.GeminiLiveSession") as mock_gemini:
        ins1 = GeminiInstanceBuilder().with_transcript("some transcript").build()

        mock_gemini.return_value = ins1

        with patch(
            "main.extract_and_save_information_to_database",
            new_callable=AsyncMock
        ) as mock_extract:
            with client.websocket_connect("/ws/") as ws:
                ws.send_json({"type": "selected_category", "category_id": 42})
                ws.send_json({"type": "control", "cmd": "start"})
                time.sleep(0.1)
                ws.send_json({"type": "control", "cmd": "stop"})

            mock_extract.assert_called_with("some transcript", cat_id=42)

def test_selected_category_is_isolated_by_sessions():
    with patch("main.GeminiLiveSession") as mock_gemini:
        instance1 = GeminiInstanceBuilder().with_transcript("some transcript1").build()
        instance2 = GeminiInstanceBuilder().with_transcript("some transcript2").build()
        instance3 = GeminiInstanceBuilder().with_transcript("some transcript3").build()
        mock_gemini.side_effect = [instance1, instance2, instance3]

        with patch(
            "main.extract_and_save_information_to_database",
            new_callable=AsyncMock
        ) as mock_extract:
            with client.websocket_connect("/ws/") as ws1:
                ws1.send_json({"type": "selected_category", "category_id": 42})
                ws1.send_json({"type": "control", "cmd": "start"})
                with client.websocket_connect("/ws/") as ws2:
                    # category is set to None
                    ws2.send_json({"type": "control", "cmd": "start"})
                    ws1.send_json({"type": "selected_category", "category_id": 43})
                    ws2.send_json({"type": "control", "cmd": "stop"})
                    time.sleep(0.1)
                    mock_extract.assert_called_once_with("some transcript2", cat_id=None)

                    # category change isnt affecting to other sessions

                    ws2.send_json({"type": "control", "cmd": "start"})
                    ws2.send_json({"type": "control", "cmd": "stop"})
                    time.sleep(0.1)
                    mock_extract.assert_called_with("some transcript2", cat_id=None)

                    ws1.send_json({"type": "control", "cmd": "stop"})
                    time.sleep(0.1)
                    mock_extract.assert_called_with("some transcript1", cat_id=43)
