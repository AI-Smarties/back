from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, WebSocket, HTTPException, Depends
from sqlalchemy.exc import IntegrityError, NoResultFound

from asr import StreamingASR
from memory_extractor import extract_and_save_information_to_database
import db_utils
import db
from gemini_live import GeminiLiveSession
from auth import get_current_user, verify_token


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create database tables automatically when the app starts."""
    print("[FastAPI] Creating database tables on startup (if missing)")
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
    except Exception: # pylint: disable=broad-exception-caught
        await ws.close(code=1008)
        return

    ws.state.USER_ID = decoded["uid"]
    ws.state.ASR = None
    ws.state.LATEST_CALENDAR_CONTENT = None
    ws.state.SELECTED_CATEGORY_ID = None

    await ws.accept()
    await ws.send_json({"type": "control", "cmd": "ready"})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] == "websocket.receive":
                if "bytes" in msg:
                    if not ws.state.ASR:
                        await ws.send_json(
                            {"type": "error", "message": "ASR not started"}
                        )
                        print("[FastAPI] Received audio chunk but ASR not started")
                        continue
                    ws.state.ASR.push_audio(msg["bytes"])
                if "text" in msg:
                    await handle_text(msg["text"], ws)
    finally:
        # The client may already be disconnected; don't attempt to send confirmation
        await stop_asr(ws, notify=False)


async def handle_text(text: str, ws: WebSocket):  # pylint: disable=too-many-return-statements
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        await ws.send_json({"type": "error", "message": "Invalid JSON"})
        print(f"[FastAPI] Invalid JSON: {text}")
        return

    payload_type = payload.get("type")
    if payload_type is None:
        await ws.send_json({"type": "error", "message": "Missing type in message"})
        print(f"[FastAPI] Missing type in message: {payload}")
        return

    if payload_type == "control":
        cmd = payload.get("cmd")
        if cmd is None:
            await ws.send_json(
                {"type": "error", "message": "Missing command in control message"}
            )
            print(f"[FastAPI] Missing command in control message: {payload}")
            return

        if cmd == "start":
            await start_asr(ws)
            return
        if cmd == "stop":
            await stop_asr(ws)
            return

        await ws.send_json({"type": "error", "message": "Unknown command"})
        print(f"[FastAPI] Unknown command: {cmd}")
        return

    if payload_type == "calendar_context":
        ws.state.LATEST_CALENDAR_CONTENT = payload.get("data")
        print(f"[FastAPI] Received calendar context: {ws.state.LATEST_CALENDAR_CONTENT}")
        await ws.send_json(
            {"type": "control", "cmd": "calendar_context_received"}
        )
        return

    if payload_type == "selected_category":
        ws.state.SELECTED_CATEGORY_ID = payload.get("category_id")
        print(f"[FastAPI] Received selected category id: {ws.state.SELECTED_CATEGORY_ID}")
        await ws.send_json(
            {"type": "control", "cmd": "selected_category_received"}
        )
        return

    await ws.send_json({"type": "error", "message": "Unknown message type"})
    print(f"[FastAPI] Unknown message type: {payload_type}")


async def start_asr(ws: WebSocket, notify: bool = True):
    if ws.state.ASR:
        ws.state.ASR.stop()
    gemini_live = GeminiLiveSession(ws, text=True)
    ws.state.ASR = StreamingASR(gemini_live)
    ws.state.ASR.start()
    if not notify:
        return
    await ws.send_json({"type": "control", "cmd": "asr_started"})


async def stop_asr(ws: WebSocket, notify: bool = True):
    if ws.state.ASR:
        transcript = ws.state.ASR.stop()
        if transcript:
            asyncio.create_task(
                extract_and_save_information_to_database(
                    transcript,
                    user_id=ws.state.USER_ID,
                    cat_id=ws.state.SELECTED_CATEGORY_ID,
                )
            )
    ws.state.ASR = None
    if not notify:
        return
    await ws.send_json({"type": "control", "cmd": "asr_stopped"})


@app.get("/get/vectors")
def get_vectors(vec_id: int = None, conv_id: int = None, user=Depends(get_current_user)):
    if vec_id is not None:
        try:
            vec = db_utils.get_vector_by_id(vec_id, user["user_id"])
        except NoResultFound as e:
            raise HTTPException(404, "Not found") from e
        return [{
            "id": vec.id,
            "text": vec.text,
            "conversation_id": vec.conversation_id,
        }]

    if conv_id is not None:
        vecs = db_utils.get_vectors_by_conversation_id(
            conv_id, user["user_id"])
    else:
        vecs = db_utils.get_vectors(user["user_id"])

    return [{
        "id": vec.id,
        "text": vec.text,
        "conversation_id": vec.conversation_id,
    } for vec in vecs]


@app.get("/get/conversations")
def get_conversations(conv_id: int = None, cat_id: int = None, user=Depends(get_current_user)):
    if conv_id is not None:
        try:
            conv = db_utils.get_conversation_by_id(conv_id, user["user_id"])
        except NoResultFound as e:
            raise HTTPException(404, "Not found") from e
        return [{
            "id": conv.id,
            "name": conv.name,
            "summary": conv.summary,
            "category_id": conv.category_id,
            "timestamp": conv.timestamp.isoformat(),
        }]

    if cat_id is not None:
        convs = db_utils.get_conversations_by_category_id(
            cat_id, user["user_id"])
    else:
        convs = db_utils.get_conversations(user["user_id"])

    return [{
        "id": conv.id,
        "name": conv.name,
        "summary": conv.summary,
        "category_id": conv.category_id,
        "timestamp": conv.timestamp.isoformat(),
    } for conv in convs]


@app.get("/get/categories")
def get_categories(cat_id: int = None, name: str = None, user=Depends(get_current_user)):
    try:
        if cat_id is not None:
            cat = db_utils.get_category_by_id(cat_id, user["user_id"])
            return [{"id": cat.id, "name": cat.name}]

        if name is not None:
            name = name.strip()
            cat = db_utils.get_category_by_name(name, user["user_id"])
            return [{"id": cat.id, "name": cat.name}]

        cats = db_utils.get_categories(user["user_id"])
        return [{"id": cat.id, "name": cat.name} for cat in cats]

    except NoResultFound as e:
        raise HTTPException(404, "Not found") from e


@app.post("/create/vector")
def create_vector(text: str, conv_id: int, user=Depends(get_current_user)):
    text = text.strip()
    try:
        db_utils.get_conversation_by_id(conv_id, user["user_id"])
        vec = db_utils.create_vector(text=text, conv_id=conv_id)
    except NoResultFound as e:
        raise HTTPException(404, "Conversation not found") from e
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
            user_id=user["user_id"],
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
        cat = db_utils.create_category(name=name, user_id=user["user_id"],)
    except IntegrityError as e:
        raise HTTPException(409, "Category already exists") from e
    return {"id": cat.id, "name": cat.name}


@app.post("/update/conversation/category")
def update_conversation_category(conv_id: int, cat_id: int, user=Depends(get_current_user)):
    try:
        conv = db_utils.update_conversation_category(
            conv_id=conv_id, cat_id=cat_id, user_id=user["user_id"],)
    except IntegrityError as e:
        raise HTTPException(409, "Foreign key constraint failed") from e
    except (LookupError, ValueError, NoResultFound) as e:
        raise HTTPException(404, f"Conversation or category not found: {e}") from e

    return {
        "id": conv.id,
        "name": conv.name,
        "summary": conv.summary,
        "category_id": conv.category_id,
        "timestamp": conv.timestamp.isoformat(),
    }


@app.post("/create/tables")
def create_tables(_=Depends(get_current_user)):
    db.create_tables()
    return {"message": "Tables created"}


@app.post("/drop/tables")
def drop_tables(_=Depends(get_current_user)):
    db.drop_tables()
    return {"message": "Tables dropped"}


@app.get("/users/me")
def get_me(user=Depends(get_current_user)):
    return {
        "user_id": user["user_id"],
        "email": user.get("email"),
    }
