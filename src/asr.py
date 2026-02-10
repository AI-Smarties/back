import threading
import queue
import asyncio
from google.cloud import speech


class StreamingASR:
    def __init__(self, ws, testing=False, client=None):
        self.ws = ws
        self.testing = testing
        self.audio_q = queue.Queue()
        self.final_buffer = ""
        self.stopped = False
        if self.testing:
            self.client = client
        else:
            self.client = speech.SpeechClient()
            self.worker = threading.Thread(target=self._worker, daemon=True)
            self.worker.start()
            self.loop = asyncio.get_running_loop()

    def stop(self):
        self.audio_q.put(None)
        self.stopped = True

    def push_audio(self, chunk: bytes):
        if self.stopped:
            raise RuntimeError("Cannot push audio after ASR is stopped")
        self.audio_q.put(chunk)

    def _dispatch(self, data):
        if self.testing:
            self.ws.send_json(data)
            return
        asyncio.run_coroutine_threadsafe(self.ws.send_json(data), self.loop)

    def _worker(self):
        def request_gen():
            while True:
                chunk = self.audio_q.get()
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="fi-FI",
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )

        # pylint: disable=unexpected-keyword-arg
        responses = self.client.streaming_recognize(config=config, requests=request_gen())
        for response in responses:
            for result in response.results:
                if not result.alternatives:
                    continue
                text = result.alternatives[0].transcript.strip()
                if not result.is_final:
                    payload = {
                        "status": "partial",
                        "text": text
                    }
                else:
                    if text:
                        text = text[0].upper() + text[1:]
                        if text[-1] not in ".!?":
                            text += "."
                    self.final_buffer += text + " "
                    payload = {
                        "status": "final",
                        "text": self.final_buffer.strip()
                    }
                data = {"type": "transcript", "data": payload}
                self._dispatch(data)
