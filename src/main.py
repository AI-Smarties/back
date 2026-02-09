import json
from fastapi import FastAPI, WebSocket
from asr import StreamingASR


app = FastAPI()


@app.websocket("/ws/")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({"type": "control", "cmd": "ready"}))  # valmis vastaanottamaan

    asr = None

    while True:
        msg = await ws.receive()

        if msg["type"] == "websocket.disconnect":
            break

        elif msg["type"] == "websocket.receive":
            if "bytes" in msg:  # audio tulee binäärinä
                if asr is None:
                    asr = StreamingASR(ws)  # alustetaan ASR
                asr.push_audio(msg["bytes"])
                continue

            if "text" in msg:  # kaikki muu kuin audio tulee tekstinä
                payload = json.loads(msg["text"])
                if payload["type"] == "control":  # ohjaussignaali
                    if payload["cmd"] == "stop":  # lopetetaan
                        if asr:
                            asr.stop()
                        continue
