import json
from fastapi import FastAPI, WebSocket
from asr import StreamingASR
import db_utils


app = FastAPI()
asr = None  # pylint: disable=invalid-name


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
                if not asr:
                    await ws.send_json({"type": "error", "message": "ASR not started"})
                    print("Received audio chunk but ASR not started")
                    continue
                asr.push_audio(msg["bytes"])
            elif "text" in msg:  # kaikki muu kuin audio tulee tekstinä
                await handle_text(msg["text"], ws)


async def handle_text(text: str, ws: WebSocket):
    global asr  # pylint: disable=global-statement
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


@app.get("/get/vectors")
async def get_vectors(vector_id: int = None, conversation_id: int = None):
    if vector_id:
        return await db_utils.get_vector_by_id(vector_id)
    if conversation_id:
        return await db_utils.get_vectors_by_conversation_id(conversation_id)
    return await db_utils.get_vectors()


@app.get("/get/conversations")
async def get_conversations(conversation_id: int = None):
    if conversation_id:
        return await db_utils.get_conversation_by_id(conversation_id)
    return await db_utils.get_conversations()


@app.get("/get/categories")
async def get_categories(category_id: int = None, name: str = None):
    if category_id:
        return await db_utils.get_category_by_id(category_id)
    if name:
        return await db_utils.get_category_by_name(name)
    return await db_utils.get_categories()
