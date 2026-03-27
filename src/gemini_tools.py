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
Answer the question in thought_context using only the vector_database_responses as source.
If the results don't answer the question directly, return "not_relevant".
You can use the transcript to understand context but not as an information source.

Dont send information to user if transcript is providing the information already.
You can also combine the information from database vector responses to have more updated information

Given:
- transcript: The conversation transcript
- thought_context: Why the query was made by Gemini Live
- vector_database_responses: Data from vector database, use only these as source of information
- previous_queries_and_answers: Earlier tool calls this session with their answers. Do not repeat information already sent.

Decide:
- "found": vector_database_responses answers the question or fulfills the reason in thought_context
- "not_relevant": results exist but are not actually relevant to the current moment
- "error": something is wrong with the inputs

you return always your thought context
Be strict. Only return "found" if the information would genuinely help the user right now.
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
    print(formatted_vectors)

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

    print(
        f"evaluate_db_data tokens: {response.usage_metadata.total_token_count}")
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
        print(f"query: {query}\nthinking: {thinking_context}")
        print(json.dumps(query_history, indent=2))

        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_vectors(
                query, user_id=user_id, limit=5, max_distance=0.5),
        )

        if not results:
            print("no vector data")
            return {"status": "not_relevant", "thinking": ""}

        return await evaluate_db_data(
            transcript,
            results,
            thinking_context,
            query_history or [],
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        return {
            "status": "error",
            "error_message": f"Failed to fetch information: {error}",
        }
