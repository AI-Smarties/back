"""
Tool functions for Gemini Live API.
These functions can be called by Gemini as tool calls.
"""


import asyncio
import json
from typing import Literal, Sequence, TypedDict

from google import auth, genai  # pylint: disable=no-name-in-module

from db_utils import search_vectors
from models import Vector


# pylint: disable=duplicate-code


CLIENT = None


def get_client():
    """Create Gemini client lazily so tests can import without ADC."""
    global CLIENT  # pylint: disable=global-statement
    if CLIENT is None:
        _, project = auth.default()
        CLIENT = genai.Client(
            vertexai=True, project=project, location="global")
    return CLIENT


SYSTEM_PROMPT = """
Your job: decide if vector_database_responses contain information that would genuinely help the user RIGHT NOW based on the current conversation.

Step 1 Validate thought_context against transcript:
Check if the thought_context is grounded in what has actually been said in the transcript.
If the thought_context is speculative or goes beyond what the transcript says, return "not_relevant".

Step 2 Check if vectors from database answer it or help the user in this moment:
Only return status: "found" if the vector_database_responses explicitly answer the thought_context.
Return information only based on the vector_database_responses.
Do not generate information not stated in the vectors though you can combine multiple vectors to get a more complete picture. For example, if one vector says "Client Elisa approved the budget increase" and another vector says "Budget increase was for $10k", you can combine these to say "Client Elisa approved a budget increase of $10k".
If the vectors contain useful information, return status: "found" and 1 concise sentence constructed from the vectors that helps the user in some way. The answer must be in the same language as the transcript. It will be shown on smart glasses, so keep it short.
Do not return information already present in the transcript.
Do not return information that can be found from previous_queries_and_answers as they have already been sent to the user in this session.
If the vectors don't directly answer the thought_context, return status: "not_relevant" but include your thinking on why it's not relevant.

Be strict. Only return status: "found" if the information would genuinely help the user right now.

Return status: "not_relevant" if:
- The thought_context is not grounded in the transcript
- The vectors don't answer the question
- The information is already in the transcript
- The information was already sent in this session
- Based on the trancript, the user likely doesn't care about this information right now or already knows it.

Given:
- transcript: The conversation so far
- thought_context: Why Gemini Live made this query
- vector_database_responses: Historical data, your ONLY source for "found" answers
- previous_queries_and_answers: Already sent this session, do not repeat
"""


class SendToUserResponse(TypedDict):
    status: Literal["found"]
    information: str
    score: float
    thinking: str


class DontSendToUserResponse(TypedDict):
    status: Literal["not_relevant"]
    thinking: str


class Error(TypedDict):
    status: Literal["error"]
    error_message: str


EvaluateResponse = SendToUserResponse | DontSendToUserResponse | Error


async def evaluate_db_data(
    transcript: str,
    vector_database_response: Sequence[Vector],
    thinking_context: str,
    query_history: list[dict],
) -> EvaluateResponse:
    formatted_vectors = "\n".join(
        f"- {vector.conversation.timestamp} {vector.text}"
        for vector in vector_database_response
    )
    print(f"[Gemini Tools] Formatted vectors: {formatted_vectors}")

    contents = (
        f"full_conversation_transcript: {transcript}\n"
        f"thought_context: {thinking_context}\n"
        f"vector_database_responses:\n{formatted_vectors}\n"
        f"previous_queries_and_answers: {json.dumps(query_history)}"
    )

    client = get_client()
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema={
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["found"],
                            },
                            "information": {"type": "string"},
                            "score": {"type": "number"},
                            "thinking": {"type": "string"},
                        },
                        "required": [
                            "status",
                            "information",
                            "score",
                            "thinking",
                        ],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["not_relevant"],
                            },
                            "thinking": {"type": "string"},
                        },
                        "required": ["status", "thinking"],
                    },
                ]
            },
        ),
    )

    print(f"[Gemini Tools] evaluate_db_data tokens: {response.usage_metadata.total_token_count}")
    data = response.parsed
    status = data.get("status")

    if status == "found":
        return {
            "status": "found",
            "information": data.get("information", ""),
            "score": data.get("score", 0.0),
            "thinking": data.get("thinking", ""),
        }

    if status == "not_relevant":
        return {
            "status": "not_relevant",
            "thinking": data.get("thinking", ""),
        }

    return {
        "status": "error",
        "error_message": data.get("error_message", "unknown"),
    }


async def fetch_information(
    thinking_context: str,
    query: str,
    transcript: str,
    query_history: list[dict] | None = None,
    user_id: str | None = None,
) -> EvaluateResponse:
    """
    Fetch useful information based on a text query from vector database.
    """
    if not query:
        return {"status": "not_relevant", "thinking": ""}

    try:
        print(f"[Gemini Tools] Query: {query}\nthinking: {thinking_context}")
        print(f"[Gemini Tools] {json.dumps(query_history, indent=2)}")

        results = await asyncio.to_thread(
            search_vectors,
            query,
            user_id=user_id,
            limit=5,
            max_distance=0.5,
        )

        if not results:
            print("[Gemini Tools] No vector data found")
            return {"status": "not_relevant", "thinking": ""}

        return await evaluate_db_data(
            transcript,
            results,
            thinking_context,
            query_history or [],
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"[Gemini Tools] Failed to fetch information: {error}")
        return {
            "status": "error",
            "error_message": f"Failed to fetch information: {error}",
        }
