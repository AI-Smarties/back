import argparse
import asyncio
import json
import time
import os

import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # LINEAR16
CHUNK_DURATION = 0.1  # seconds
CHUNK_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_DURATION)  # 3200

async def stream(url: str, path: str):
    audio = open(path, "rb").read()  # pylint: disable=consider-using-with
    print(f"Connecting to {url} ...")

    async with websockets.connect(url) as ws:
        await ws.recv()  # ready
        await ws.send(json.dumps({"type": "control", "cmd": "start"}))  # start asr

        # Wait until the backend confirms ASR is started before streaming audio.
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            print(f"\r[WS MSG] {msg}", flush=True)
            if msg.get("type") == "control" and msg.get("cmd") == "asr_started":
                break

        async def recv_loop(): # receive events from ws
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    print(msg, flush=True)
            except websockets.exceptions.ConnectionClosed:
                pass

        asyncio.create_task(recv_loop())  # add listener for events
        ## mock audiofile streaming as it would stream through ws
        total_chunks = (len(audio) + CHUNK_BYTES - 1) // CHUNK_BYTES
        start = time.perf_counter()
        for idx, i in enumerate(range(0, len(audio), CHUNK_BYTES), 1):
            await ws.send(audio[i : i + CHUNK_BYTES])
            progress = idx / total_chunks * 100
            elapsed = time.perf_counter() - start

            print(f"\rStreaming... {progress:.0f}% {elapsed:.2f}s", end="", flush=True)
            await asyncio.sleep(CHUNK_DURATION)

        await ws.send(json.dumps({"type": "control", "cmd": "stop"})) ## stop asr
        await asyncio.sleep(5.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    token = os.environ.get("TOKEN")
    parser.add_argument("audio", help="Raw LINEAR16 PCM file (16kHz mono)")
    # parser.add_argument("--url", default=f"wss://<staging-address>/ws/?token={token}")
    parser.add_argument("--url", default=f"ws://localhost:8000/ws/?token={token}")
    args = parser.parse_args()
    asyncio.run(stream(args.url, args.audio))


# set firebase token as environment variable before running:
# export TOKEN=token_value_here
# run in root
# source ./venv/bin/activate
# python3 ./scripts/audio_streaming_helper.py ./scripts/output.raw
