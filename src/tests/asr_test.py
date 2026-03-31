import asyncio
from unittest.mock import Mock

import pytest

from asr import StreamingASR
from gemini_live import GeminiLiveSession


# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture
def immediate_loop():
    """Loop-like object that executes thread-safe callbacks immediately."""
    loop = Mock(spec=asyncio.AbstractEventLoop)

    def _call_soon_threadsafe(callback, *args):
        callback(*args)

    loop.call_soon_threadsafe.side_effect = _call_soon_threadsafe
    return loop


@pytest.mark.asyncio
async def test_streaming_asr_stop_pushes_none_to_gemini_live_queue(immediate_loop):
    ws = Mock()
    session = GeminiLiveSession(ws, text=True)
    session._loop = immediate_loop

    asr = StreamingASR(session)
    asr.stop()

    item = await session._queue.get()
    assert item is None


def test_streaming_asr_dispatch_forwards_text_to_gemini_live():
    gemini_live = Mock()
    gemini_live.push_data = Mock()

    asr = StreamingASR(gemini_live)
    asr._dispatch("hello world")

    gemini_live.push_data.assert_called_once_with("hello world")


def test_streaming_asr_start_starts_gemini_live_and_worker_thread():
    gemini_live = Mock()
    gemini_live.start = Mock()

    asr = StreamingASR(gemini_live)
    asr._prepare_streaming_metadata = Mock()
    asr._thread.start = Mock()

    asr.start()

    asr._prepare_streaming_metadata.assert_called_once()
    gemini_live.start.assert_called_once()
    asr._thread.start.assert_called_once()
