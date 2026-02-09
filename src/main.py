import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from asr import StreamingASR


app = FastAPI()


# endpoint audiolle
@app.websocket("/ws/audio/")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({"type": "ready"})) # lähetetään vahvistus valmiudesta

    asr = None

    try:
        while True:
            msg = await ws.receive() # vastaanottaa audiota tai ohjaussignaalin

            if "bytes" in msg and msg["bytes"]: # audio tulee binäärinä
                if asr is None:
                    asr = StreamingASR(ws) # alustetaan ASR
                asr.push_audio(msg["bytes"]) # pushataan audio
                continue

            if "text" in msg and msg["text"]: # ohjaussignaali
                data = json.loads(msg["text"])
                if data.get("action") == "stop": # lopetetaan
                    if asr:
                        asr.stop()
                    continue

    except WebSocketDisconnect:
        pass

    finally:
        if asr:
            asr.stop()
