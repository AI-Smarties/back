from unittest.mock import Mock, AsyncMock
import pytest
from gemini_live import GeminiLiveSession, MODEL


# pylint: disable=protected-access


class TestGeminiLiveSession:
    """Test cases for GeminiLiveSession class"""

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock websocket"""
        ws = Mock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.fixture
    def session(self, mock_websocket):
        """Create a GeminiLiveSession instance"""
        return GeminiLiveSession(mock_websocket)

    def test_session_initialization(self, session, mock_websocket):
        """Test that session initializes correctly"""
        assert session.ws == mock_websocket
        assert session.tokens_used == 0
        assert session._task is None
        assert session._queue.maxsize == 10

    def test_push_data_adds_to_queue(self, session):
        """Test that audio chunks are added to the queue"""
        audio_chunk = b"\x00\x01" * 8  # valid int16 PCM bytes
        session.push_data(audio_chunk)
        assert session._queue.qsize() == 1

    def test_push_data_ignores_when_queue_full(self, session):
        """Test that audio is silently dropped when queue is full"""
        pcm_chunk = b"\x00\x01" * 4  # valid int16 PCM bytes
        # Fill the queue
        for _ in range(10):
            session.push_data(pcm_chunk)
        # Try to add one more
        session.push_data(pcm_chunk)
        assert session._queue.qsize() == 10

    @pytest.mark.asyncio
    async def test_stop_adds_none_to_queue(self, session):
        """Test that stop adds None to queue to signal termination"""
        session.stop()
        item = await session._queue.get()
        assert item is None


class TestConfig:  # pylint: disable=too-few-public-methods
    """Test cases for Gemini Live configuration"""

    def test_model_constant(self):
        """Test that MODEL constant is set correctly"""
        assert MODEL == "gemini-live-2.5-flash-native-audio"
