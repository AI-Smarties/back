import asyncio
import json
from datetime import datetime

from zoneinfo import ZoneInfo
from google import auth, genai

from db_utils import (
    create_conversation,
    create_vectors_batch,
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
Extract every factual claim from this meeting transcript worth storing in a searchable database.

ALWAYS SAVE:
- Every number mentioned: budgets, costs, overruns, percentages, counts, targets — include units and context
- Every date or deadline mentioned
- Every named person and what they decided, recommended, or are responsible for
- Every risk or concern raised
- Every decision made or option presented for approval
- Every change in scope, budget, or schedule
- Any action item with a clear owner or timeline

SKIP:
- Small talk, greetings, weather, parking spaces, coffee, office logistics completely unrelated to projects
- Phrases that merely confirm what was already said ("yes", "exactly", "good point")
- The fact that a report will be sent — only save the content of the report

Rules:
- Save EVERY number, date, and named person — do not summarize away quantitative data.
- One atomic fact per vector. Do not combine multiple separate facts into one entry.
- Always store BOTH "data" and "reason" fields in the same language as the transcript. Never use English if the transcript is in Finnish.
- Each fact must be self-contained: include subject and key detail so a semantic search finds it.
- Only save facts explicitly stated — never infer or assume details not directly mentioned.
- Do not save the same fact twice with different wording.
- If there is nothing worth saving, return an empty vectors array.

For "name": create a short title capturing the key topic (e.g. "Q2 projektipalaveri - huhtikuu 2026").
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
        model="gemini-2.5-flash-lite",
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
    user_id,
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

    print("[Memory Extractor] Extracting information from transcript")

    extracted_name = name
    extracted_vectors = []

    try:
        information_vectors = await memory_extractor_worker(transcript)
        if information_vectors:
            extracted_name = extracted_name or information_vectors.get("name")
            extracted_vectors = information_vectors.get("vectors", [])
            print(f"[Memory Extractor] {json.dumps(information_vectors, indent=2)}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Memory Extractor] Memory_extractor_worker failed: {e}")

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
                user_id=user_id,
            ),
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Memory Extractor] Store_data failed: {e}")


def store_data(transcript, vectors, user_id, name=None, conversation_id=None, cat_id=None): # pylint: disable=too-many-arguments,too-many-positional-arguments
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
            user_id=user_id,
        ).id

    get_conversation_by_id(conv_id, user_id)
    create_vectors_batch([v["data"] for v in vectors], conv_id)

    try:
        summary = generate_summary(transcript)
        if summary:
            update_conversation_summary(conv_id, summary, user_id)
            print(f"[Memory Extractor] Summary saved for conversation {conv_id}")
        else:
            print(f"[Memory Extractor] No summary generated for conversation {conv_id}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Memory Extractor] Summary generation failed for conversation {conv_id}: {e}")

    conv = get_conversation_by_id(conv_id, user_id)
    saved_vectors = get_vectors_by_conversation_id(conv_id, user_id)

    print("[Memory Extractor]:")
    print(f"conversation: {conv.id} {conv.name}")
    print(f"summary: {conv.summary}")
    print(f"category_id: {conv.category_id}")
    for vector in saved_vectors:
        print(f"  vector {vector.id}: {vector.text}")
