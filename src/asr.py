import threading
import queue
import asyncio
from google import auth
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


class StreamingASR:  # pylint: disable=too-many-instance-attributes
    def __init__(self, ws, testing=False, client=None):
        self.ws = ws
        self.testing = testing
        self.audio_q = queue.Queue()
        self.final_buffer = ""
        self.stopped = False
        if self.testing:
            self.client = client
        else:
            self.client = SpeechClient()
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
            _, project = auth.default()
            if not project:
                raise RuntimeError(
                    "Could not determine GCP project id from Application Default Credentials."
                )
            recognizer = self.client.recognizer_path(project, "global", "_")

            config = cloud_speech.StreamingRecognitionConfig(
                config=cloud_speech.RecognitionConfig(
                    explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                        encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=16000,
                        audio_channel_count=1,
                    ),
                    language_codes=["fi-FI"],
                    features=cloud_speech.RecognitionFeatures(
                        enable_automatic_punctuation=True,
                    ),
                ),
                streaming_features=cloud_speech.StreamingRecognitionFeatures(
                    interim_results=True,
                ),
            )

            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=recognizer,
                streaming_config=config,
            )

            while True:
                chunk = self.audio_q.get()
                if chunk is None:
                    break
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

        responses = self.client.streaming_recognize(requests=request_gen())
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
