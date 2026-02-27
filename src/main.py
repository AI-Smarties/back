import json
from fastapi import FastAPI, WebSocket, HTTPException
from sqlalchemy.exc import IntegrityError
from gemini_live import GeminiLiveSession
import db_utils
import db


# pylint: disable=invalid-name global-statement
app = FastAPI()
geminiLive = None


@app.websocket("/ws/")
async def audio_ws(ws: WebSocket):
    global geminiLive
    await ws.accept()
    await ws.send_json({"type": "control", "cmd": "ready"})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] == "websocket.receive":
                if "bytes" in msg:  # audio tulee binäärinä
                    if not geminiLive:
                        await ws.send_json({"type": "error", "message": "ASR not started"})
                        print("Received audio chunk but ASR not started")
                        continue
                    geminiLive.push_audio(msg["bytes"])
                elif "text" in msg:  # kaikki muu kuin audio tulee tekstinä
                    await handle_text(msg["text"], ws)
    finally:
        if geminiLive:
            await geminiLive.stop()
            geminiLive = None


async def handle_text(text: str, ws: WebSocket):
    global geminiLive
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
            print('start asr')
            if geminiLive:
                await geminiLive.stop()
            geminiLive = GeminiLiveSession(ws)
            await geminiLive.start()
        elif payload["cmd"] == "stop":
            print('stop asr')
            if geminiLive:
                await geminiLive.stop()
            geminiLive = None
        else:
            await ws.send_json({"type": "error", "message": "Unknown command"})
            print(f"Unknown command: {payload['cmd']}")
            return
    else:
        await ws.send_json({"type": "error", "message": "Unknown message type"})
        print(f"Unknown message type: {payload['type']}")
        return


@app.get("/get/vectors")
def get_vectors(vec_id: int = None, conv_id: int = None):
    if vec_id is not None:
        vec = db_utils.get_vector_by_id(vec_id)
        if vec is None:
            return []
        return [{"id": vec.id, "text": vec.text, "conversation_id": vec.conversation_id}]
    if conv_id is not None:
        vecs = db_utils.get_vectors_by_conversation_id(conv_id)
    else:
        vecs = db_utils.get_vectors()
    return [{
        "id": vec.id,
        "text": vec.text,
        "conversation_id": vec.conversation_id,
        } for vec in vecs]


@app.get("/get/conversations")
def get_conversations(conv_id: int = None, cat_id: int = None, cat_name: str = None):
    if conv_id is not None:
        conv = db_utils.get_conversation_by_id(conv_id)
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
        convs = db_utils.get_conversations_by_category_id(cat_id)
    elif cat_name is not None:
        cat_name = cat_name.strip()
        convs = db_utils.get_conversations_by_category_name(cat_name)
    else:
        convs = db_utils.get_conversations()
    return [{
        "id": conv.id,
        "name": conv.name,
        "summary": conv.summary,
        "category_id": conv.category_id,
        "timestamp": conv.timestamp.isoformat(),
    } for conv in convs]


@app.get("/get/categories")
def get_categories(cat_id: int = None, cat_name: str = None):
    if cat_id is not None:
        cat = db_utils.get_category_by_id(cat_id)
    elif cat_name is not None:
        cat_name = cat_name.strip()
        cat = db_utils.get_category_by_name(cat_name)
    else:
        cats = db_utils.get_categories()
        return [{"id": cat.id, "name": cat.name} for cat in cats]
    if cat is None:
        return []
    return [{"id": cat.id, "name": cat.name}]


@app.post("/create/category")
def create_category(name: str):
    name = name.strip()
    try:
        cat = db_utils.create_category(name)
    except IntegrityError as e:
        raise HTTPException(409, "Category already exists") from e
    return {"id": cat.id, "name": cat.name}


@app.post("/create/tables")
def create_tables():
    db.create_tables()
    return {"message": "Tables created"}


@app.post("/drop/tables")
def drop_tables():
    db.drop_tables()
    return {"message": "Tables dropped"}
