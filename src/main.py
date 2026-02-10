import json
from fastapi import FastAPI, WebSocket
from asr import StreamingASR


# pylint: disable=invalid-name, global-statement
app = FastAPI()
asr = None


@app.websocket("/ws/")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "control", "cmd": "ready"})
    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            break
        if msg["type"] == "websocket.receive":
            if "bytes" in msg:  # audio tulee binäärinä
                asr.push_audio(msg["bytes"])
            elif "text" in msg:  # kaikki muu kuin audio tulee tekstinä
                await handle_text(msg["text"], ws)


async def handle_text(text: str, ws: WebSocket):
    global asr
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        await ws.send_json({"type": "error", "message": "Invalid JSON"})
        print(f"Invalid JSON: {text}")
        return
    if "type" not in payload:
        await ws.send_json({"type": "error", "message": "Missing type in message"})
        print(f"Missing type in message: {payload}")
        return
    if payload["type"] == "control":
        if "cmd" not in payload:
            await ws.send_json({"type": "error", "message": "Missing command in control message"})
            print(f"Missing command in control message: {payload}")
            return
        if payload["cmd"] == "start":
            if asr:
                asr.stop()
            asr = StreamingASR(ws)
        elif payload["cmd"] == "stop":
            if asr:
                asr.stop()
            asr = None
        else:
            await ws.send_json({"type": "error", "message": "Unknown command"})
            print(f"Unknown command: {payload['cmd']}")
            return
    else:
        await ws.send_json({"type": "error", "message": "Unknown message type"})
        print(f"Unknown message type: {payload['type']}")
        return
