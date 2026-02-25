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
async def get_vectors(vec_id: int = None, conv_id: int = None):
    if vec_id is not None:
        vec = await db_utils.get_vector_by_id(vec_id)
        if vec is None:
            return []
        return [{"id": vec.id, "text": vec.text, "conversation_id": vec.conversation_id}]
    if conv_id is not None:
        vecs = await db_utils.get_vectors_by_conversation_id(conv_id)
    else:
        vecs = await db_utils.get_vectors()
    return [{
        "id": vec.id,
        "text": vec.text,
        "conversation_id": vec.conversation_id,
        } for vec in vecs]


@app.get("/get/conversations")
async def get_conversations(conv_id: int = None, cat_id: int = None, cat_name: str = None):
    if conv_id is not None:
        conv = await db_utils.get_conversation_by_id(conv_id)
        if conv is None:
            return []
        return [{
            "id": conv.id,
            "name": conv.name,
            "summary": conv.summary,
            "category_id": conv.category_id,
            "timestamp": conv.timestamp.isoformat(),
        }]
    if cat_id is not None:
        convs = await db_utils.get_conversations_by_category_id(cat_id)
    elif cat_name is not None:
        convs = await db_utils.get_conversations_by_category_name(cat_name)
    else:
        convs = await db_utils.get_conversations()
    return [{
        "id": conv.id,
        "name": conv.name,
        "summary": conv.summary,
        "category_id": conv.category_id,
        "timestamp": conv.timestamp.isoformat(),
    } for conv in convs]


@app.get("/get/categories")
async def get_categories(cat_id: int = None, cat_name: str = None):
    if cat_id is not None:
        cat = await db_utils.get_category_by_id(cat_id)
    elif cat_name is not None:
        cat = await db_utils.get_category_by_name(cat_name)
    else:
        cats = await db_utils.get_categories()
        return [{"id": cat.id, "name": cat.name} for cat in cats]
    if cat is None:
        return []
    return [{"id": cat.id, "name": cat.name}]
