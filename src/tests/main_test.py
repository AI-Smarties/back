from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def fake_verify_token(_):
    return {"uid": "test-user"}


@pytest.fixture(autouse=True)
def mock_verify_token():
    with patch("main.verify_token", fake_verify_token):
        yield


def test_server_sends_ready_signal():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "control"
        assert data["cmd"] == "ready"


def test_server_handles_invalid_json():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_text("not a json")
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Invalid JSON"


def test_server_handles_unknown_command():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"type": "control", "cmd": "unknown"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Unknown command"


def test_server_handles_unknown_message_type():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"type": "unknown", "cmd": "start"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Unknown message type"


def test_server_handles_missing_type():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"cmd": "start"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Missing type in message"


def test_server_handles_missing_command():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({"type": "control"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Missing command in control message"


def test_server_handles_audio_before_start():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_bytes(b"audio chunk")
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "ASR not started"

def test_calendar_context_null_event():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({
            "type": "calendar_context",
            "data": {
                "title": "General Conversation",
                "description": None,
                "start": None,
                "end": None
            }
        })

        data = websocket.receive_json()
        assert data["type"] == "control"
        assert data["cmd"] == "calendar_context_received"


def test_calendar_context_real_event():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({
            "type": "calendar_context",
            "data": {
                "title": "Team sync",
                "description": "Weekly check-in",
                "start": "2026-03-26T10:00:00.000+0200",
                "end": "2026-03-26T10:45:00.000+0200"
            }
        })

        data = websocket.receive_json()
        assert data["type"] == "control"
        assert data["cmd"] == "calendar_context_received"


def test_calendar_context_missing_fields():
    with client.websocket_connect("/ws/?token=test-token") as websocket:
        websocket.receive_json()  # ready signal
        websocket.send_json({
            "type": "calendar_context",
            "data": {
                "title": None,
                "description": None,
                "start": None
                #missing end
            }
        })

        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Invalid calendar context format"


class ASRInstanceBuilder:  # pylint: disable=protected-access
    def __init__(self):
        self._ins = MagicMock()
        # Mirror key `StreamingASR` state used by production code.
        self._ins.transcript = ""
        self._ins._stopped = False

        async def start():
            self._ins._stopped = False

        self._ins.start = AsyncMock(side_effect=start)

        def stop():
            self._ins._stopped = True
            return self._ins.transcript.strip()

        self._ins.stop = MagicMock(side_effect=stop)
        def push_audio(_: bytes):
            if self._ins._stopped:
                raise RuntimeError("Cannot push audio after ASR is stopped")

        self._ins.push_audio = MagicMock(side_effect=push_audio)

    def with_transcript(self, transcript: str):
        self._ins.transcript = transcript
        return self

    def build(self):
        return self._ins


def test_two_users_have_isolated_asr_instances():
    with patch("main.StreamingASR") as mock_asr:
        asr1 = ASRInstanceBuilder().build()
        asr2 = ASRInstanceBuilder().build()

        mock_asr.side_effect = [asr1, asr2]

        with client.websocket_connect("/ws/?token=test-token") as ws1:
            ws1.receive_json()
            with client.websocket_connect("/ws/?token=test-token") as ws2:
                ws2.receive_json()

                assert mock_asr.call_count == 0

                ws1.send_json({"type": "control", "cmd": "start"})
                assert ws1.receive_json() == {"type": "control", "cmd": "asr_started"}
                assert asr1.start.call_count == 1
                assert asr2.start.call_count == 0
                assert mock_asr.call_count == 1

                ws2.send_json({"type": "control", "cmd": "start"})
                assert ws2.receive_json() == {"type": "control", "cmd": "asr_started"}
                assert asr1.start.call_count == 1
                assert asr2.start.call_count == 1
                assert mock_asr.call_count == 2

                # Each connection should create its own ASR instance


def test_disconnect_cleans_up_only_own_session():
    with patch("main.StreamingASR") as mock_asr:
        asr1 = ASRInstanceBuilder().build()
        asr2 = ASRInstanceBuilder().build()

        mock_asr.side_effect = [asr1, asr2]

        with client.websocket_connect("/ws/?token=test-token") as ws1:
            ws1.receive_json()
            ws1.send_json({"type": "control", "cmd": "start"})
            assert ws1.receive_json() == {"type": "control", "cmd": "asr_started"}

            with client.websocket_connect("/ws/?token=test-token") as ws2:
                ws2.receive_json()
                assert asr2.start.call_count == 0

                ws2.send_json({"type": "control", "cmd": "start"})
                assert ws2.receive_json() == {"type": "control", "cmd": "asr_started"}

                ws2.send_json({"type": "control", "cmd": "stop"})
                assert ws2.receive_json() == {"type": "control", "cmd": "asr_stopped"}

                assert asr2.stop.call_count == 1
                assert asr1.stop.call_count == 0

            # ws1 still alive — only instance2 should have been stopped
            assert asr1.stop.call_count == 0

            ws1.send_json({"type": "control", "cmd": "stop"})
            assert ws1.receive_json() == {"type": "control", "cmd": "asr_stopped"}

            assert asr1.stop.call_count == 1


def test_start_multiple_times_replaces_previous_asr_instance():
    with patch("main.StreamingASR") as mock_asr:
        asr1 = ASRInstanceBuilder().build()
        asr2 = ASRInstanceBuilder().build()
        asr3 = ASRInstanceBuilder().build()
        asr4 = ASRInstanceBuilder().build()

        mock_asr.side_effect = [asr1, asr2, asr3, asr4]

        with client.websocket_connect("/ws/?token=test-token") as ws1:
            ws1.receive_json()
            ws1.send_json({"type": "control", "cmd": "start"})
            assert ws1.receive_json() == {"type": "control", "cmd": "asr_started"}

            with client.websocket_connect("/ws/?token=test-token") as ws2:
                ws2.receive_json()
                ws2.send_json({"type": "control", "cmd": "start"})
                assert ws2.receive_json() == {"type": "control", "cmd": "asr_started"}

                ws2.send_json({"type": "control", "cmd": "start"})
                assert ws2.receive_json() == {"type": "control", "cmd": "asr_started"}

                assert mock_asr.call_count == 3
                assert asr2.start.call_count == 1
                assert asr2.stop.call_count == 1
                assert asr3.start.call_count == 1

                ws2.send_json({"type": "control", "cmd": "stop"})
                assert ws2.receive_json() == {"type": "control", "cmd": "asr_stopped"}

                assert asr3.stop.call_count == 1
                ws2.send_json({"type": "control", "cmd": "start"})
                assert ws2.receive_json() == {"type": "control", "cmd": "asr_started"}

                assert mock_asr.call_count == 4
                assert asr4.start.call_count == 1

            # ws1 still alive — only instance2 should have been stopped
            assert asr1.stop.call_count == 0

            ws1.send_json({"type": "control", "cmd": "stop"})
            assert ws1.receive_json() == {"type": "control", "cmd": "asr_stopped"}

            assert mock_asr.call_count == 4
            assert asr1.stop.call_count == 1


def test_selected_category_is_used_on_stop():
    with patch("main.StreamingASR") as mock_asr:
        asr1 = ASRInstanceBuilder().with_transcript("some transcript").build()
        mock_asr.return_value = asr1

        with patch(
            "main.extract_and_save_information_to_database",
            new_callable=AsyncMock
        ) as mock_extract:

            with client.websocket_connect("/ws/?token=test-token") as ws:
                ws.receive_json()
                ws.send_json({"type": "selected_category", "category_id": 42})
                assert ws.receive_json() == {
                    "type": "control",
                    "cmd": "selected_category_received",
                }
                ws.send_json({"type": "control", "cmd": "start"})
                assert ws.receive_json() == {"type": "control", "cmd": "asr_started"}
                ws.send_json({"type": "control", "cmd": "stop"})
                assert ws.receive_json() == {"type": "control", "cmd": "asr_stopped"}

            mock_extract.assert_called_with("some transcript", user_id="test-user", cat_id=42)


def test_selected_category_is_isolated_by_sessions():
    with patch("main.StreamingASR") as mock_asr:
        asr1 = ASRInstanceBuilder().with_transcript("some transcript1").build()
        asr2 = ASRInstanceBuilder().with_transcript("some transcript2").build()
        asr3 = ASRInstanceBuilder().with_transcript("some transcript3").build()

        mock_asr.side_effect = [asr1, asr2, asr3]

        with patch(
            "main.extract_and_save_information_to_database",
            new_callable=AsyncMock
        ) as mock_extract:

            with client.websocket_connect("/ws/?token=test-token") as ws1:
                ws1.receive_json()
                ws1.send_json({"type": "selected_category", "category_id": 42})
                assert ws1.receive_json() == {
                    "type": "control",
                    "cmd": "selected_category_received",
                }
                ws1.send_json({"type": "control", "cmd": "start"})
                assert ws1.receive_json() == {"type": "control", "cmd": "asr_started"}

                with client.websocket_connect("/ws/?token=test-token") as ws2:
                    ws2.receive_json()

                    ws2.send_json({"type": "control", "cmd": "start"})
                    assert ws2.receive_json() == {"type": "control", "cmd": "asr_started"}
                    ws1.send_json({"type": "selected_category", "category_id": 43})
                    assert ws1.receive_json() == {
                        "type": "control",
                        "cmd": "selected_category_received",
                    }

                    ws2.send_json({"type": "control", "cmd": "stop"})
                    assert ws2.receive_json() == {"type": "control", "cmd": "asr_stopped"}

                    assert mock_extract.call_args_list[-1] == (
                        ("some transcript2",),
                        {"user_id": "test-user", "cat_id": None},
                    )

                    ws2.send_json({"type": "control", "cmd": "start"})
                    assert ws2.receive_json() == {"type": "control", "cmd": "asr_started"}
                    ws2.send_json({"type": "control", "cmd": "stop"})
                    assert ws2.receive_json() == {"type": "control", "cmd": "asr_stopped"}

                    assert mock_extract.call_args_list[-1] == (
                        ("some transcript3",),
                        {"user_id": "test-user", "cat_id": None},
                    )

                ws1.send_json({"type": "control", "cmd": "stop"})
                assert ws1.receive_json() == {"type": "control", "cmd": "asr_stopped"}

                assert mock_extract.call_args_list[-1] == (
                    ("some transcript1",),
                    {"user_id": "test-user", "cat_id": 43},
                )
