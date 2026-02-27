import asyncio
from google import genai

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
SYSTEM_INSTRUCTION = """You are a Finnish keyword spotter. Listen to the audio.
When you hear the word "cappi" or any inflected form (capin, capille, capilla, cappiin, capissa, cappia), call the keyword_detected function immediately.
Do not speak. Do not generate audio. Only call the function when you hear "cappi"."""


CONFIG = genai.types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription=genai.types.AudioTranscriptionConfig(),
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[genai.types.Tool(function_declarations=[
        genai.types.FunctionDeclaration(
            name="keyword_detected",
            description="Call this function when you hear the word 'cappi' or"
                        "any of its Finnish inflected forms in the audio.",
        )
    ])],
)

class GeminiLiveSession:
    def __init__(self, ws):
        self.ws = ws
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._task: asyncio.Task | None = None
        self.tokens_used = 0

    async def start(self):
        self._task = asyncio.create_task(self._run())

    def push_audio(self, chunk: bytes):
        try:
            self._audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    async def stop(self):
        try:
            self._audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
                pass
        print(f"session total tokens: {self.tokens_used}")

    async def _run(self):
        # wait for first audio chunk before opening the connection
        first_chunk = await self._audio_queue.get()
        if first_chunk is None:
            return  # stopped before any audio arrived
        client = genai.Client()
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                print("Gemini Live connected")
                await session.send_realtime_input(
                    audio={"data": first_chunk, "mime_type": "audio/pcm;rate=16000"}
                )
                send_task = asyncio.create_task(self._send(session))
                recv_task = asyncio.create_task(self._receive(session))
                await send_task
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:  # pylint: disable=broad-except
            print(f"Gemini Live error: {e}")
        finally:
            await client.aio.aclose()

    async def _send(self, session):
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            await session.send_realtime_input(
                audio={"data": chunk, "mime_type": "audio/pcm;rate=16000"}
            )

    async def _receive(self, session):
        input_buf: list[str] = []
        try:
            while True:  # session.receive() only covers one turn
                async for response in session.receive():
                    if response.usage_metadata:
                        self.tokens_used += response.usage_metadata.total_token_count or 0
                        print(f"tokens: {self.tokens_used}")
                    # function call = keyword detected
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            print(f"tool call: {fc.name}")
                            if fc.name == "keyword_detected":
                                print('DETECTED')
                                await self.ws.send_json({"type": "ai", "data": "true"})
                            await session.send_tool_response(function_responses=[
                                genai.types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response={"result": "ok"}
                                )
                            ])
                    server_content = response.server_content
                    if not server_content:
                        continue
                    # accumulate input transcription chunks
                    if server_content.input_transcription and \
                        server_content.input_transcription.text:
                        input_buf.append(server_content.input_transcription.text)
                    # on turn end: flush buffered transcript as one message
                    if server_content.turn_complete and input_buf:
                        text = "".join(input_buf)
                        print(f"sending user transcript: {text!r}")
                        await self.ws.send_json({"type": "user", "data": text})
                        input_buf.clear()
        except Exception as e:  # pylint: disable=broad-except
            print(f"_receive error: {e}")
