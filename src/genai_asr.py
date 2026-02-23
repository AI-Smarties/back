import threading
import queue
import asyncio
import io
import struct
from google import genai
from google.genai import types

# pylint: disable=broad-exception-caught
# pylint: disable=duplicate-code
# pylint: disable=too-many-instance-attributes
class StreamingGenAIASR:
    """
    Streaming ASR using Google Vertex AI Gemini with audio input.
    Similar architecture to StreamingASR but uses GenAI multimodal streaming.
    """
    def __init__(self, ws, testing=False, client=None):
        self.ws = ws
        self.testing = testing
        self.audio_q = queue.Queue()
        self.final_buffer = ""
        self.stopped = False

        if self.testing:
            self.client = client
        else:
            # Initialize GenAI client with Vertex AI
            self.client = genai.Client(vertexai=True)
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
        """
        Worker thread that processes audio chunks and streams them to Gemini.
        Uses Gemini's multimodal streaming capabilities for real-time transcription.
        """
        try:
            # Configure the model for audio transcription
            model_id = "gemini-2.0-flash-exp"  # Use latest Gemini model with audio support

            # Create streaming config
            config = types.GenerateContentConfig(
                temperature=0.0,  # Deterministic output for transcription
                system_instruction="Muuta kuulemasi suomenkielinen audio tekstiksi. "
                                "Älä sisällytä vastaukseesi mitään ylimääräistä."
            )

            # Collect audio chunks into a buffer for batch processing
            # Gemini streaming works best with larger audio chunks
            audio_buffer = io.BytesIO()
            chunk_size_threshold = 16000 * 2 * 2  # ~2 seconds of 16kHz 16-bit audio

            while True:
                chunk = self.audio_q.get()
                if chunk is None:
                    # Process remaining buffer if any
                    if audio_buffer.tell() > 0:
                        self._process_audio_chunk(audio_buffer.getvalue(), config, model_id)
                    break

                audio_buffer.write(chunk)

                # Process when buffer reaches threshold>
                if audio_buffer.tell() >= chunk_size_threshold:
                    audio_data = audio_buffer.getvalue()
                    self._process_audio_chunk(audio_data, config, model_id)
                    audio_buffer = io.BytesIO()  # Reset buffer

        except Exception as e:
            error_data = {"type": "error", "message": f"GenAI ASR error: {str(e)}"}
            self._dispatch(error_data)
            print(f"GenAI ASR worker error: {e}")

    def _process_audio_chunk(self, audio_data: bytes, config, model_id: str):
        """
        Process a chunk of audio data through Gemini.
        Sends audio as inline data with proper MIME type.
        """
        try:
            # Create audio part with proper format
            # PCM 16kHz, 16-bit, mono matches the Speech API format
            audio_part = types.Part.from_bytes(
                data=audio_data,
                mime_type="audio/pcm"  # Or "audio/wav" if you add WAV header
            )

            # Create prompt for transcription
            prompt_part = types.Part.from_text(text="Transcribe this audio in Finnish:")

            # Stream the request
            response_stream = self.client.models.generate_content_stream(
                model=model_id,
                contents=[prompt_part, audio_part],
                config=config
            )

            # Process streaming response
            partial_text = ""
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    partial_text += chunk.text

                    # Send partial results
                    payload = {
                        "status": "partial",
                        "text": partial_text.strip()
                    }
                    data = {"type": "transcript", "data": payload}
                    self._dispatch(data)

            # Send final result
            if partial_text.strip():
                text = partial_text.strip()
                # Capitalize first letter and add period if needed
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

        except Exception as e:
            error_data = {"type": "error", "message": f"Audio processing error: {str(e)}"}
            self._dispatch(error_data)
            print(f"Error processing audio chunk: {e}")


    def _create_wav_header(self, data_size: int, sample_rate: int = 16000,
                          channels: int = 1, bits_per_sample: int = 16) -> bytes:
        """
        Create WAV header for raw PCM data.
        Useful if GenAI requires WAV format instead of raw PCM.
        """

        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8

        header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            data_size + 36,  # File size - 8
            b'WAVE',
            b'fmt ',
            16,  # fmt chunk size
            1,   # PCM format
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        return header
