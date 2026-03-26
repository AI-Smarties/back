import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from google import auth, genai

from db_utils import (
    create_conversation,
    create_vector,
    get_conversation_by_id,
    get_vectors_by_conversation_id,
    update_conversation_summary,
)
from summary_service import generate_summary

CLIENT = None


def get_client():
    """Create Gemini client lazily so tests can import without ADC."""
    global CLIENT  # pylint: disable=global-statement
    if CLIENT is None:
        _, project = auth.default()
        CLIENT = genai.Client(
            vertexai=True,
            project=project,
            location="global",
        )
    return CLIENT

SYSTEM_PROMPT = """
You extract facts from meeting transcripts that would cause real problems if forgotten.

SAVE: deadlines, decisions, scope changes, budget figures, named responsibilities, technical blockers
SKIP: how a decision was reached, confirmations of things already stated, small talk, food, office logistics, parking

Rules:
- Only save facts explicitly stated in the transcript. Never infer, assume, or fill in details not directly mentioned — especially numbers, names, and dates.
- Save only if forgetting in 3 months would cause a real mistake.
- Save outcomes, not process steps. Good: "Client Elisa approved a backend hire — team lacked capacity to meet current backend requirements." Bad: "Hiring process initiated."
- One atomic fact per vector. Do not combine multiple facts into one entry.
- Always store in English, regardless of the language of the transcript.
- Each fact must be self-contained: include who decided it and in what context.
  Bad: "Priority 1: meeting summary feature."
  Good: "Client Elisa set meeting summary as Priority 1 — save conversation summaries to the database and auto-share to Google Drive."
- Phrase for retrieval: include the actor, subject, and key detail so a semantic search for any of them finds the fact.
- Avoid generic statements. If a fact would be true of any similar project, add specifics that make it unique to this one.
- Do not save the same fact twice with different wording within this transcript.
- If there is nothing worth saving, return an empty vectors array. Never store meta-comments about the transcript itself (e.g. "no facts found", "transcript lacks content").

For "name": create a short English title capturing the key topic (e.g. "Elisa client meeting - March 2026").
"""


def _default_conversation_name(transcript: str) -> str:
    transcript = transcript.strip()
    if not transcript:
        return "Untitled conversation"

    first_line = transcript.splitlines()[0].strip()
    if not first_line:
        return "Untitled conversation"

    return first_line[:80]


async def memory_extractor_worker(transcript):
    """
    AI model extracts useful information from transcript.

    Args:
        transcript: str

    Returns:
        {
            "name": str,
            "vectors": [{"data": str, "reason": str}]
        }
    """
    client = get_client()
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=transcript,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "vectors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "data": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["data", "reason"],
                        },
                    },
                },
                "required": ["vectors", "name"],
            },
        ),
    )
    return response.parsed


async def extract_and_save_information_to_database(
    transcript,
    conversation_id=None,
    name=None,
    cat_id=None,
):
    """
    Extract information from transcript with AI model and store it to database.

    Also generates and stores a session summary from the final validated transcript.

    Args:
        transcript: str
        conversation_id: int | None
        name: str | None
        cat_id: int | None
    """
    transcript = transcript.strip()
    if not transcript:
        print("[Memory Extractor] Transcript empty, skipping extraction and summary generation")
        return

    print("[Memory Extractor] extracting information from transcript")

    extracted_name = name
    extracted_vectors = []

    try:
        information_vectors = await memory_extractor_worker(transcript)
        if information_vectors:
            extracted_name = extracted_name or information_vectors.get("name")
            extracted_vectors = information_vectors.get("vectors", [])
            print(f"[Memory Extractor] {json.dumps(information_vectors, indent=2)}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Memory Extractor] memory_extractor_worker failed: {e}")

    if not extracted_vectors:
        print("[Memory Extractor] No vectors extracted, skipping store")
        return

    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: store_data(
                transcript=transcript,
                vectors=extracted_vectors,
                conversation_id=conversation_id,
                name=extracted_name or _default_conversation_name(transcript),
                cat_id=cat_id,
            ),
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Memory Extractor] store_data failed: {e}")


def store_data(transcript, vectors, name=None, conversation_id=None, cat_id=None):
    """
    Persist conversation, vectors, and summary.

    This runs in an executor thread so it can safely perform blocking DB and
    summary-generation work without blocking the event loop.
    """
    conv_id = conversation_id

    if conv_id is None:
        conv_id = create_conversation(
            name=name or _default_conversation_name(transcript),
            summary=None,
            cat_id=cat_id,
            timestamp=datetime.now(ZoneInfo("Europe/Helsinki")),
        ).id

    for vector in vectors:
        create_vector(vector["data"], conv_id)

    try:
        summary = generate_summary(transcript)
        if summary:
            update_conversation_summary(conv_id, summary)
            print(f"[Memory Extractor] summary saved for conversation {conv_id}")
        else:
            print(f"[Memory Extractor] no summary generated for conversation {conv_id}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Memory Extractor] summary generation failed for conversation {conv_id}: {e}")

    conv = get_conversation_by_id(conv_id)
    saved_vectors = get_vectors_by_conversation_id(conv_id)

    print("[Memory Extractor]")
    print(f"conversation: {conv.id} {conv.name}")
    print(f"summary: {conv.summary}")
    print(f"category_id: {conv.category_id}")
    for vector in saved_vectors:
        print(f"  vector {vector.id}: {vector.text}")
