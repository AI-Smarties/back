import pytest
from unittest.mock import Mock, AsyncMock
from gemini_live import GeminiLiveSession, MODEL


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


class TestConfig:
    """Test cases for Gemini Live configuration"""

    def test_model_constant(self):
        """Test that MODEL constant is set correctly"""
        assert MODEL == "gemini-2.5-flash-native-audio-preview-12-2025"
