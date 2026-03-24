from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, WebSocket, HTTPException
from sqlalchemy.exc import IntegrityError

from gemini_live import GeminiLiveSession
from memory_extractor import extract_and_save_information_to_database
import db_utils
import db

from fastapi import Depends
from auth import get_current_user, verify_token

GEMINI_LIVE = None
LATEST_CALENDAR_CONTEXT = None
SELECTED_CATEGORY_ID = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create database tables automatically when the app starts."""
    print("Creating database tables on startup (if missing)")
    db.create_tables()
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/")
async def audio_ws(ws: WebSocket):
    token = ws.query_params.get("token")

    if not token:
        await ws.close(code=1008)
        return

    try:
        decoded = verify_token(f"Bearer {token}")
        user_id = decoded["uid"]
    except Exception:
        await ws.close(code=1008)
        return

    await ws.accept()
    await ws.send_json({"type": "control", "cmd": "ready"})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] == "websocket.receive":
                if "bytes" in msg:
                    if not GEMINI_LIVE:
                        await ws.send_json(
                            {"type": "error", "message": "Gemini Live not started"}
                        )
                        print("Received audio chunk but Gemini Live not started")
                        continue
                    GEMINI_LIVE.push_audio(msg["bytes"])
                elif "text" in msg:
                    await handle_text(msg["text"], ws, user_id)
    finally:
        await stop_gemini_live()


async def handle_text(  # pylint: disable=too-many-return-statements
    text: str,
    ws: WebSocket,
    user_id: str,
):
    global LATEST_CALENDAR_CONTEXT, SELECTED_CATEGORY_ID  # pylint: disable=global-statement

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        await ws.send_json({"type": "error", "message": "Invalid JSON"})
        print(f"Invalid JSON: {text}")
        return

    payload_type = payload.get("type")
    if payload_type is None:
        await ws.send_json({"type": "error", "message": "Missing type in message"})
        print(f"Missing type in message: {payload}")
        return

    if payload_type == "control":
        cmd = payload.get("cmd")
        if cmd is None:
            await ws.send_json(
                {"type": "error", "message": "Missing command in control message"}
            )
            print(f"Missing command in control message: {payload}")
            return

        if cmd == "start":
            await start_gemini_live(ws)
            return
        if cmd == "stop":
            await stop_gemini_live()
            return

        await ws.send_json({"type": "error", "message": "Unknown command"})
        print(f"Unknown command: {cmd}")
        return

    if payload_type == "calendar_context":
        LATEST_CALENDAR_CONTEXT = payload.get("data")
        print(f"Received calendar context: {LATEST_CALENDAR_CONTEXT}")
        await ws.send_json(
            {"type": "control", "cmd": "calendar_context_received"}
        )
        return

    if payload_type == "selected_category":
        SELECTED_CATEGORY_ID = payload.get("category_id")
        print(f"Received selected category id: {SELECTED_CATEGORY_ID}")
        await ws.send_json(
            {"type": "control", "cmd": "selected_category_received"}
        )
        return

    await ws.send_json({"type": "error", "message": "Unknown message type"})
    print(f"Unknown message type: {payload_type}")


async def start_gemini_live(ws: WebSocket):
    global GEMINI_LIVE  # pylint: disable=global-statement
    print("Starting Gemini Live")
    if GEMINI_LIVE:
        await GEMINI_LIVE.stop()
    GEMINI_LIVE = GeminiLiveSession(ws)
    await GEMINI_LIVE.start()


async def stop_gemini_live():
    global GEMINI_LIVE  # pylint: disable=global-statement
    print("Stopping Gemini Live")
    if GEMINI_LIVE:
        transcript = await GEMINI_LIVE.stop()
        print(transcript)

        transcript = transcript.strip()
        if transcript:
            asyncio.create_task(
                extract_and_save_information_to_database(
                    transcript,
                    cat_id=SELECTED_CATEGORY_ID,
                )
            )

    GEMINI_LIVE = None


@app.get("/get/vectors")
def get_vectors(vec_id: int = None, conv_id: int = None, user=Depends(get_current_user)):
    if vec_id is not None:
        vec = db_utils.get_vector_by_id(vec_id)
        if vec is None:
            return []
        return [{
            "id": vec.id,
            "text": vec.text,
            "conversation_id": vec.conversation_id,
        }]

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
def get_conversations(conv_id: int = None, cat_id: int = None,  user=Depends(get_current_user)):
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
def get_categories(cat_id: int = None, name: str = None, user=Depends(get_current_user)):
    if cat_id is not None:
        cat = db_utils.get_category_by_id(cat_id)
    elif name is not None:
        name = name.strip()
        cat = db_utils.get_category_by_name(name)
    else:
        cats = db_utils.get_categories()
        return [{"id": cat.id, "name": cat.name} for cat in cats]

    if cat is None:
        return []

    return [{"id": cat.id, "name": cat.name}]


@app.post("/create/vector")
def create_vector(text: str, conv_id: int, user=Depends(get_current_user)):
    text = text.strip()
    try:
        vec = db_utils.create_vector(text=text, conv_id=conv_id)
    except IntegrityError as e:
        raise HTTPException(409, "Foreign key constraint failed") from e
    return {"id": vec.id, "text": vec.text, "conversation_id": vec.conversation_id}


@app.post("/create/conversation")
def create_conversation(
    name: str,
    summary: str = None,
    cat_id: int = None,
    timestamp: str = None,
    user=Depends(get_current_user)
):
    name = name.strip()
    summary = summary.strip() if summary else None
    timestamp = datetime.fromisoformat(
        timestamp.strip()) if timestamp else None
    try:
        conv = db_utils.create_conversation(
            name=name,
            summary=summary,
            cat_id=cat_id,
            timestamp=timestamp,
        )
    except IntegrityError as e:
        raise HTTPException(409, "Foreign key constraint failed") from e
    return {
        "id": conv.id,
        "name": conv.name,
        "summary": conv.summary,
        "category_id": conv.category_id,
        "timestamp": conv.timestamp.isoformat(),
    }


@app.post("/create/category")
def create_category(name: str, user=Depends(get_current_user)):
    name = name.strip()
    try:
        cat = db_utils.create_category(name=name)
    except IntegrityError as e:
        raise HTTPException(409, "Category already exists") from e
    return {"id": cat.id, "name": cat.name}


@app.post("/update/conversation/category")
def update_conversation_category(conv_id: int, cat_id: int, user=Depends(get_current_user)):
    try:
        conv = db_utils.update_conversation_category(
            conv_id=conv_id, cat_id=cat_id)
    except IntegrityError as e:
        raise HTTPException(409, "Foreign key constraint failed") from e
    except (LookupError, ValueError) as e:
        raise HTTPException(
            404, f"Conversation or category not found: {e}") from e

    return {
        "id": conv.id,
        "name": conv.name,
        "summary": conv.summary,
        "category_id": conv.category_id,
        "timestamp": conv.timestamp.isoformat(),
    }


@app.post("/create/tables")
def create_tables():
    db.create_tables()
    return {"message": "Tables created"}


@app.post("/drop/tables")
def drop_tables():
    db.drop_tables()
    return {"message": "Tables dropped"}
