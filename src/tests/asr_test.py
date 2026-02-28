from types import SimpleNamespace
from asr import StreamingASR

# pylint: disable=too-few-public-methods
class StubWebSocket:
    def __init__(self):
        self.sent = []

    def send_json(self, data):
        self.sent.append(data)

# pylint: disable=too-few-public-methods
class StubClient:
    def __init__(self, responses):
        self._responses = responses

    def streaming_recognize(self, requests):  # pylint: disable=unused-argument
        return iter(self._responses)

def resp(transcript: str, is_final: bool, stability: float = 0.9):
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[SimpleNamespace(transcript=transcript)],
                is_final=is_final,
                stability=stability,
            )
        ]
    )

def test_worker_emits_correct_json_and_applies_punctuation():
    ws = StubWebSocket()
    client = StubClient([
        resp("hello", False),
        resp("hello", True),
    ])
    asr = StreamingASR(ws, testing=True, client=client)
    asr._worker()  # pylint: disable=protected-access
    assert ws.sent[0] == {"type": "transcript", "data": {"status": "partial", "text": " hello"}}
    assert ws.sent[1] == {"type": "transcript", "data": {"status": "final", "text": "Hello."}}

def test_worker_accumulates_final_buffer_and_preserves_punctuation():
    ws = StubWebSocket()
    client = StubClient([
        resp("hello", True),
        resp("world!", True),
    ])
    asr = StreamingASR(ws, testing=True, client=client)
    asr._worker()  # pylint: disable=protected-access
    assert ws.sent[0]["data"]["text"] == "Hello."
    assert ws.sent[1]["data"]["text"] == "Hello. World!"

def test_push_audio_after_stop_raises_error():
    ws = StubWebSocket()
    asr = StreamingASR(ws, testing=True, client=StubClient([]))
    asr.stop()
    try:
        asr.push_audio(b"123")
        assert False, "Expected RuntimeError"
    except RuntimeError:
        assert True
