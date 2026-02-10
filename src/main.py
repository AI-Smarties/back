import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from asr import StreamingASR


app = FastAPI()


@app.websocket("/ws/")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "control", "cmd": "ready"})

    asr = None

    # pylint: disable=too-many-nested-blocks
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
                        print(f"Invalid JSON: {msg['text']}")
                        continue
                    if "type" not in payload:
                        await ws.send_json({"type": "error", "message": "Missing type in message"})
                        print(f"Missing type in message: {payload}")
                        continue
                    if payload["type"] == "control":
                        if "cmd" not in payload:
                            await ws.send_json(
                                {"type": "error", "message": "Missing command in control message"}
                            )
                            print(f"Missing command in control message: {payload}")
                            continue
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
                            continue
                    else:
                        await ws.send_json({"type": "error", "message": "Unknown message type"})
                        print(f"Unknown message type: {payload['type']}")
                        continue

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    finally:
        if asr:
            asr.stop()
