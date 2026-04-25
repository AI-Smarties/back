import asyncio

from google import auth
from google.cloud.speech_v2 import SpeechAsyncClient
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
        language_codes=["auto"],
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
        self.current_q = asyncio.Queue(maxsize=200)
        self.transcript = ""
        self._stopped = False
        self._recognizer = None
        self._client = None
        self._gemini_live = gemini_live
        self._task: asyncio.Task | None = None

    async def start(self):
        print("[ASR] Starting ASR session...")
        await self._prepare_streaming_metadata()
        self._gemini_live.start()
        self._task = asyncio.create_task(self._worker())

    async def _prepare_streaming_metadata(self):
        _, project = await asyncio.to_thread(auth.default)
        if not project:
            raise RuntimeError(
                "Could not determine GCP project id from Application Default Credentials."
            )
        self._client = SpeechAsyncClient(
            client_options=ClientOptions(
                api_endpoint="eu-speech.googleapis.com",
            )
        )
        self._recognizer = self._client.recognizer_path(project, "eu", "_")

    def stop(self) -> str:
        print("[ASR] Stopping ASR session...")
        self._stopped = True
        try:
            self.current_q.put_nowait(None)
        except asyncio.QueueFull:
            self.current_q.get_nowait()
            self.current_q.put_nowait(None)
        if self._task:
            self._task.cancel()
        self._gemini_live.stop()
        return self.transcript.strip()

    def push_audio(self, chunk: bytes):
        if self._stopped:
            raise RuntimeError("Cannot push audio after ASR is stopped")
        try:
            self.current_q.put_nowait(chunk)
        except asyncio.QueueFull:
            self.current_q.get_nowait()
            self.current_q.put_nowait(chunk)

    def _dispatch(self, text):
        self._gemini_live.push_data(text)

    async def _worker(self):  # pylint: disable=too-many-branches
        while not self._stopped:
            # Each stream iteration gets its own queue reference. On restart we
            # swap _current_q so push_audio feeds the new stream, then send None
            # to the old queue to unblock the zombie generator in the gRPC thread.
            my_q = self.current_q
            try:
                async def request_gen():
                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=self._recognizer,
                        streaming_config=CONFIG,
                    )

                    while True:
                        chunk = await my_q.get()
                        if chunk is None:
                            print('[ASR] Stream closed')
                            break
                        chunk = amplify_chunk(chunk, gain=30.0)
                        yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

                responses = await self._client.streaming_recognize(requests=request_gen())
                async for response in responses:
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
            self.current_q = asyncio.Queue(maxsize=200)
            old_q.put_nowait(None)
