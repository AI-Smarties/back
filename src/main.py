import json
from fastapi import FastAPI, WebSocket
from asr import StreamingASR


app = FastAPI()


# endpoint audiolle
@app.websocket("/ws/audio/")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({"type": "control", "cmd": "ready"})) # valmis vastaanottamaan

    asr = None

    while True:
        msg = await ws.receive() # vastaanottaa audiota tai ohjaussignaalin

        if msg["type"] == "websocket.disconnect":
            break

        elif msg["type"] == "websocket.receive":

            payload = json.loads(msg["text"])

            if "bytes" in payload and payload["bytes"]: # audio tulee binäärinä
                if asr is None:
                    asr = StreamingASR(ws) # alustetaan ASR
                asr.push_audio(payload["bytes"]) # pushataan audio
                continue

            if payload["type"] == "control": # ohjaussignaali
                if payload["cmd"] == "stop": # lopetetaan
                    if asr:
                        asr.stop()
                    continue
