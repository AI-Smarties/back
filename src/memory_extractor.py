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

_, project = auth.default()
client = genai.Client(vertexai=True, project=project, location="europe-north1")

system_prompt = """
You extract facts from meeting transcripts that would cause real problems if forgotten.

SAVE: deadlines, decisions, scope changes, budget figures, named responsibilities, technical blockers
SKIP: how a decision was reached, confirmations of things already stated, small talk, food, office logistics, parking

For "name": create a short descriptive title that captures the key topic of the conversation.
A fact is worth saving only if forgetting it in 3 months would cause a mistake.
Do not save process steps (e.g. "escalation initiated"), save the outcome (e.g. "Lisa approved backend hire").
Do not save the same fact twice with different wording.
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
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=transcript,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
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
):
    """
    Extract information from transcript with AI model and store it to database.

    Also generates and stores a session summary from the final validated transcript.

    Args:
        transcript: str
        conversation_id: int | None
        name: str | None
    """
    transcript = transcript.strip()
    if not transcript:
        print("Transcript empty, skipping extraction and summary generation")
        return

    print("extracting information from transcript")

    extracted_name = name
    extracted_vectors = []

    try:
        information_vectors = await memory_extractor_worker(transcript)
        if information_vectors:
            extracted_name = extracted_name or information_vectors.get("name")
            extracted_vectors = information_vectors.get("vectors", [])
            print(json.dumps(information_vectors, indent=2))
    except Exception as e:
        print(f"memory_extractor_worker failed: {e}")

    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: store_data(
                transcript=transcript,
                vectors=extracted_vectors,
                conversation_id=conversation_id,
                name=extracted_name or _default_conversation_name(transcript),
            ),
        )
    except Exception as e:
        print(f"store_data failed: {e}")


def store_data(transcript, vectors, name=None, conversation_id=None):
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
            timestamp=datetime.now(ZoneInfo("Europe/Helsinki")),
        ).id

    for vector in vectors:
        create_vector(vector["data"], conv_id)

    try:
        summary = generate_summary(transcript)
        if summary:
            update_conversation_summary(conv_id, summary)
            print(f"summary saved for conversation {conv_id}")
        else:
            print(f"no summary generated for conversation {conv_id}")
    except Exception as e:
        print(f"summary generation failed for conversation {conv_id}: {e}")

    conv = get_conversation_by_id(conv_id)
    saved_vectors = get_vectors_by_conversation_id(conv_id)

    print(f"conversation: {conv.id} {conv.name}")
    print(f"summary: {conv.summary}")
    for v in saved_vectors:
        print(f"  vector {v.id}: {v.text}")