import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from asr import StreamingASR


app = FastAPI()


@app.websocket("/ws/")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "control", "cmd": "ready"})

    asr = None

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] == "websocket.receive":
                if "bytes" in msg:  # audio tulee binäärinä
                    asr.push_audio(msg["bytes"])

                elif "text" in msg:  # kaikki muu kuin audio tulee tekstinä
                    try:
                        payload = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "message": "Invalid JSON"})
                        print("Invalid JSON", msg["text"])
                        continue
                    if payload["type"] == "control":
                        if payload["cmd"] == "start":
                            asr = StreamingASR(ws)
                        elif payload["cmd"] == "stop":
                            asr.stop()
                            asr = None
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        if asr:
            asr.stop()
