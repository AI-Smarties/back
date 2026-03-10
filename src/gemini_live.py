import asyncio
from google import genai, auth
from gemini_tools import fetch_information


MODEL = "gemini-live-2.5-flash-native-audio"

SYSTEM_INSTRUCTION = """
You are a silent conversation observer. You NEVER produce audio.

You have in your possession the vector database that contains ONLY useful information that is necessary to remember. Things that when forgotten can cause problems.
Your job is simple: when you hear something that you might have additional information on in vector database make the fetch_information call.

Skip small talk (greetings, coffee, parking, lunch, scheduling chatter).
Do NOT fetch speculatively — only when a concrete fact, decision, deadline, 
constraint, or named person/project is explicitly mentioned.

"""


CONFIG = genai.types.LiveConnectConfig(
    input_audio_transcription=genai.types.AudioTranscriptionConfig(),
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[
        genai.types.Tool(function_declarations=[
            genai.types.FunctionDeclaration(
                name="fetch_information",
                description=(
                    "Flag a moment where past context might be relevant. "
                    "Call this when speakers discuss a topic that might have "
                    "related stored facts."
                ),
                parameters={
                    "type":         "object",
                    "properties":   {
                        "query": {
                            "type":         "string",
                            "description":  "The text query that is used to query vector database"
                        },
                        "thinking_context":{
                            "type":         "string",
                            "description":  "Thought process of the gemini live why it called this tool"
                        }
                    },
                    "required":     ["query", "thinking_context"]
                }
            ),
           
        ]),
    ],
)

class GeminiLiveSession:
    def __init__(self, ws):
        self.ws = ws
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._task: asyncio.Task | None = None
        self.tokens_used = 0
        self.transcript: str = ""

    async def start(self):
        self._task = asyncio.create_task(self._run())

    def push_audio(self, chunk: bytes):
        try:
            self._audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            print('Que is full of audio, dropping oldest audio packet')
            pass  
   
    async def stop(self) -> str:
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

        return self.transcript

    async def _run(self):
        # wait for first audio chunk before opening the connection
        first_chunk = await self._audio_queue.get()
        if first_chunk is None:
            return  # stopped before any audio arrived
        _, project = auth.default()
        client = genai.Client(vertexai=True, project=project, location="europe-north1")
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


    async def _fetch_in_background(self, thinking_context, query, transcript):
        tool_response = await fetch_information(thinking_context, query, transcript)
        print(tool_response)
        if tool_response["status"] == "found":
            await self.ws.send_json({"type": "ai", "data": tool_response["information"]})


    async def _receive(self, session):
        try:
            while True:  # session.receive() only covers one turn
                async for response in session.receive():
                    ## accumulate the used tokens so we can monitor token usage
                    if response.usage_metadata:
                        self.tokens_used += response.usage_metadata.total_token_count or 0
                   
                    server_content = response.server_content
                    # accumulate the transcriptions before toolcall so we have the latest updated transcription to function call
                    if server_content:
                        if server_content.input_transcription and server_content.input_transcription.text:
                            self.transcript += server_content.input_transcription.text + " "
                    # catch gemini lives toolcall
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            print(f"tool call: {fc.name}")
                            # provide full accumulated context to 
                            if fc.name == "fetch_information":
                                # response immediately so the gemini live can continue processing audio
                                await session.send_tool_response(function_responses=[
                                    genai.types.FunctionResponse(id=fc.id, name=fc.name, response={"response": "ok"})
                                ])
                                query = fc.args['query']
                                thinking_context = fc.args['thinking_context']

                                # make new background task so gemini live is not interrupted
                                asyncio.create_task(self._fetch_in_background(thinking_context, query, self.transcript))

                               
                                
  


        except Exception as e:  # pylint: disable=broad-except
            print(f"_receive error: {e}")
