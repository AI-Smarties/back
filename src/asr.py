import json
import threading
import queue
import asyncio
from google.cloud import speech


class StreamingASR:
    def __init__(self, ws):
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.audio_q = queue.Queue()
        self.final_buffer = ""
        self.client = speech.SpeechClient()

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True
        )
        self.worker.start()

    def stop(self):
        self.audio_q.put(None)

    def push_audio(self, chunk: bytes):
        self.audio_q.put(chunk)

    def _worker(self):
        def request_gen():
            while True:
                chunk = self.audio_q.get()
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(
                    audio_content=chunk
                )

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
        responses = self.client.streaming_recognize(
            config=config,
            requests=request_gen(),
        )

        for response in responses:
            for result in response.results:
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript.strip()

                if not result.is_final:
                    payload = {
                        "type": "partial",
                        "text": transcript
                    }
                else:
                    text = transcript.strip()
                    if text:
                        text = text[0].upper() + text[1:]
                        if text[-1] not in ".!?":
                            text += "."

                    self.final_buffer += text + " "
                    payload = {
                        "type": "final",
                        "text": self.final_buffer.strip()
                    }


                asyncio.run_coroutine_threadsafe(
                    self.ws.send_text(json.dumps(payload)),
                    self.loop
                )
