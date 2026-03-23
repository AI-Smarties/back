import threading
import queue
import asyncio
from google import auth
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions


class _RestartStream(Exception):
    """Raised to break out of the response loop and restart the stream."""


# pylint: disable=too-many-instance-attributes
class StreamingASR:
    def __init__(self, ws, testing=False, client=None):
        self.ws = ws
        self.testing = testing
        self._current_q = queue.Queue()
        self.transcript = ""
        self.stopped = False
        if self.testing:
            self.client = client
        else:
            self.client = SpeechClient(client_options=ClientOptions(
                api_endpoint="eu-speech.googleapis.com",
            ))
            self.worker = threading.Thread(target=self._worker, daemon=True)
            self.loop = asyncio.get_running_loop()

    def start(self):
        self.worker.start()

    def stop(self):
        self._current_q.put(None)
        self.stopped = True
        return self.transcript.strip()

    def push_audio(self, chunk: bytes):
        if self.stopped:
            raise RuntimeError("Cannot push audio after ASR is stopped")
        self._current_q.put(chunk)

    def _dispatch(self, data):
        if self.testing:
            self.ws.send_json(data)
            return
        asyncio.run_coroutine_threadsafe(self.ws.send_json(data), self.loop)

    def _worker(self):  # pylint: disable=too-many-branches
        while not self.stopped:
            # Each stream iteration gets its own queue reference. On restart we
            # swap _current_q so push_audio feeds the new stream, then send None
            # to the old queue to unblock the zombie generator in the gRPC thread.
            my_q = self._current_q
            try:
                def request_gen():
                    _, project = auth.default()
                    if not project:
                        raise RuntimeError(
                            "Could not determine GCP project id from Application Default "
                            "Credentials."
                        )
                    recognizer = self.client.recognizer_path(project, "eu", "_")

                    config = cloud_speech.StreamingRecognitionConfig(
                        config=cloud_speech.RecognitionConfig(
                            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                                sample_rate_hertz=16000,
                                audio_channel_count=1,
                            ),
                            language_codes=["fi-FI"],
                            model='chirp_3'
                        ),
                        streaming_features=cloud_speech.StreamingRecognitionFeatures(
                            interim_results=True,
                            enable_voice_activity_events=True,
                        ),
                    )

                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=recognizer,
                        streaming_config=config,
                    )

                    while True:
                        chunk = my_q.get()
                        if chunk is None:
                            print('[ASR] stream closed')
                            break
                        yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

                responses = self.client.streaming_recognize(requests=request_gen())
                for response in responses:
                    with open('./scripts/response.txt', 'a') as file:  # pylint: disable=unspecified-encoding
                        file.write(str(response) + "\n")

                    # Restart the stream when GCP signals end of voice activity.
                    # chirp_3 stops responding after an utterance ends so we must
                    # open a fresh stream to pick up the next one.
                    speech_activity_end = (
                        cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
                    )
                    if response.speech_event_type == speech_activity_end:
                        print("[ASR] speech activity ended, restarting stream...")
                        raise _RestartStream()

                    for result in response.results:
                        if not result.alternatives:
                            continue
                        text = result.alternatives[0].transcript.strip()
                        if not result.is_final:
                            payload = {"status": "partial", "text": text}
                        else:
                            if text:
                                text = text[0].upper() + text[1:]
                                if text[-1] not in ".!?":
                                    text += "."
                            self.transcript += text + " "
                            payload = {"status": "final", "text": self.transcript.strip()}
                        self._dispatch({"type": "transcript", "data": payload})

            except _RestartStream:
                if self.stopped:
                    break
            except Exception as e:  # pylint: disable=broad-exception-caught
                if self.stopped:
                    break
                print(f"[ASR] stream error, restarting... {e}")

            # Rotate queue: give push_audio a fresh queue for the next stream and
            # send None to the old one so the zombie generator unblocks and exits.
            old_q = my_q
            self._current_q = queue.Queue()
            old_q.put(None)
