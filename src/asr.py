import threading
import queue
from google import auth
from google.cloud.speech_v2.types import cloud_speech
from gemini_live import amplify_chunk


class _RestartStream(Exception):
    """Raised to break out of the response loop and restart the stream."""


class StreamingASR:
    def __init__(self, client, gemini_live):
        self.current_q = queue.Queue()
        self.transcript = ""
        self.stopped = False
        self.client = client
        self.gemini_live = gemini_live
        self.worker = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.gemini_live.start()
        self.worker.start()

    def stop(self):
        self.current_q.put(None)
        self.stopped = True
        self.gemini_live.stop()
        return self.transcript.strip()

    def push_audio(self, chunk: bytes):
        if self.stopped:
            raise RuntimeError("Cannot push audio after ASR is stopped")
        self.current_q.put(chunk)

    def _dispatch(self, text):
        self.gemini_live.push_data(text)

    def _worker(self):  # pylint: disable=too-many-branches
        while not self.stopped:
            # Each stream iteration gets its own queue reference. On restart we
            # swap _current_q so push_audio feeds the new stream, then send None
            # to the old queue to unblock the zombie generator in the gRPC thread.
            my_q = self.current_q
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
                        chunk = amplify_chunk(chunk, gain=35.0)
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
                            print(f"[ASR] partial transcript: {text}")
                        else:
                            if text:
                                text = text[0].upper() + text[1:]
                                if text[-1] not in ".!?":
                                    text += "."
                            self.transcript += text + " "
                            self._dispatch(text)

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
            self.current_q = queue.Queue()
            old_q.put(None)
