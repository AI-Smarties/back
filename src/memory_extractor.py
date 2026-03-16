import asyncio
from google import genai, auth
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from db_utils import (
    create_conversation,
    create_vector,
    get_conversation_by_id,
    get_vectors_by_conversation_id
)

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


async def memory_extractor_worker(transcript):
    """
    Ai model extracts usefull information from transcript

    Args
        transcript: str     # transcript of conversation
    Returns
        name: str           # ai models suggestion for conversation name 
        vectors: [{
            data: str       # data to be stored in vector database
            reason: str     # ai reasoning why it needst to be stored
        }]
    """
    response =  await client.aio.models.generate_content(
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
                                "data":   {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["data", "reason"]
                        },
                    },
                },
                "required": ["vectors", "name"]
            },
        )
    )
    return response.parsed


async def extract_and_save_information_to_database(transcript, conversation_id = None, name = None):
    """
        Extract information from transcript with Ai model and store them to vector database

        Args:
            transcript:          str
            conversation_id:    int | None     // OPTIONAL: id of existing conversation where data is going to be linked. If not provided new conversation is created
            name:               str | None     // OPTIONAL: name for new conversation. Will not be used when conversation_id is provided
    """
    print("extracting information from transcript")
    try:
        information_vectors = await memory_extractor_worker(transcript)
    except Exception as e:
        print(f"memory_extractor_worker failed: {e}")
        return

    if not information_vectors or not information_vectors.get("vectors"):
        print("Nothing worth saving")
        return

    print(json.dumps(information_vectors, indent=2))
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: store_data(information_vectors["vectors"], conversation_id=conversation_id, name=name or information_vectors["name"])
        )
    except Exception as e:
        print(f"store_data failed: {e}")
        

def store_data(vectors, name=None, conversation_id=None):
    conv_id = conversation_id
    if conv_id is None:
        conv_id = create_conversation(
            name=name,
            timestamp=datetime.now(ZoneInfo("Europe/Helsinki")),
        ).id
    for vector in vectors:
        create_vector(vector["data"], conv_id)
    conv = get_conversation_by_id(conv_id)
    saved_vectors = get_vectors_by_conversation_id(conv_id)

    print(f"conversation: {conv.id} {conv.name}")
    for v in saved_vectors:
        print(f"  vector {v.id}: {v.text}")