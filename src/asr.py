import threading
import queue
from google import auth
from google.cloud.speech_v2.types import cloud_speech
from gemini_live import amplify_chunk


class StreamingASR:  # pylint: disable=too-many-instance-attributes
    def __init__(self, client, gemini_live):
        self.current_q = queue.Queue()
        self.transcript = ""
        self.stopped = False
        self.client = client
        self.gemini_live = gemini_live
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self._recognizer = None
        self._streaming_config = None

    def start(self):
        print("[ASR] Starting ASR session...")
        self._prepare_streaming_metadata()
        self.gemini_live.start()
        self.worker.start()

    def _prepare_streaming_metadata(self):
        _, project = auth.default()
        if not project:
            raise RuntimeError(
                "Could not determine GCP project id from Application Default Credentials."
            )

        self._recognizer = self.client.recognizer_path(project, "eu", "_")
        self._streaming_config = cloud_speech.StreamingRecognitionConfig(
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

    def stop(self):
        print("[ASR] Stopping ASR session...")
        self.stopped = True
        self.current_q.put(None)
        if self.worker.is_alive():
            self.worker.join()
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
                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=self._recognizer,
                        streaming_config=self._streaming_config,
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
                    for result in response.results:
                        if not result.alternatives:
                            continue
                        text = result.alternatives[0].transcript.strip()
                        self.transcript += text + " "
                        self._dispatch(text)

            except Exception as e:  # pylint: disable=broad-exception-caught
                if self.stopped:
                    break
                print(f"[ASR] stream error: '{e}', restarting with a new queue...")

            # Rotate queue: give push_audio a fresh queue for the next stream and
            # send None to the old one so the zombie generator unblocks and exits.
            old_q = my_q
            self.current_q = queue.Queue()
            old_q.put(None)
