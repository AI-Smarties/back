import threading
import queue

from google import auth
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from gemini_live import amplify_chunk


CONFIG = cloud_speech.StreamingRecognitionConfig(
    config=cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            audio_channel_count=1,
        ),
        language_codes=["fi-FI"],
        model='chirp_3',
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
        ),
    ),
    streaming_features=cloud_speech.StreamingRecognitionFeatures(
        interim_results=False,
        enable_voice_activity_events=True,
    ),
)


class StreamingASR:
    def __init__(self, gemini_live):
        self.current_q = queue.Queue()
        self.transcript = ""
        self._stopped = False
        self._recognizer = None
        self._client = None
        self._gemini_live = gemini_live
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        print("[ASR] Starting ASR session...")
        self._prepare_streaming_metadata()
        self._gemini_live.start()
        self._thread.start()

    def _prepare_streaming_metadata(self):
        _, project = auth.default()
        if not project:
            raise RuntimeError(
                "Could not determine GCP project id from Application Default Credentials."
            )
        self._client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint="eu-speech.googleapis.com",
            )
        )
        self._recognizer = self._client.recognizer_path(project, "eu", "_")

    def stop(self):
        print("[ASR] Stopping ASR session...")
        self._stopped = True
        self.current_q.put(None)
        if self._thread.is_alive():
            self._thread.join()
        self._gemini_live.stop()
        return self.transcript.strip()

    def push_audio(self, chunk: bytes):
        if self._stopped:
            raise RuntimeError("Cannot push audio after ASR is stopped")
        self.current_q.put(chunk)

    def _dispatch(self, text):
        self._gemini_live.push_data(text)

    def _worker(self):  # pylint: disable=too-many-branches
        while not self._stopped:
            # Each stream iteration gets its own queue reference. On restart we
            # swap _current_q so push_audio feeds the new stream, then send None
            # to the old queue to unblock the zombie generator in the gRPC thread.
            my_q = self.current_q
            try:
                def request_gen():
                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=self._recognizer,
                        streaming_config=CONFIG,
                    )

                    while True:
                        chunk = my_q.get()
                        if chunk is None:
                            print('[ASR] Stream closed')
                            break
                        chunk = amplify_chunk(chunk, gain=35.0)
                        yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

                responses = self._client.streaming_recognize(requests=request_gen())
                for response in responses:
                    for result in response.results:
                        if not result.alternatives:
                            continue
                        text = result.alternatives[0].transcript.strip()
                        self.transcript += text + " "
                        self._dispatch(text)

            except Exception as e:  # pylint: disable=broad-exception-caught
                if self._stopped:
                    break
                print(f"[ASR] Stream error: '{e}', restarting with a new queue...")

            # Rotate queue: give push_audio a fresh queue for the next stream and
            # send None to the old one so the zombie generator unblocks and exits.
            old_q = my_q
            self.current_q = queue.Queue()
            old_q.put(None)
