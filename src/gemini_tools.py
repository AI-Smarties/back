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
SEARCH_CLIENT = None


def get_client():
    """Create Gemini client lazily so tests can import without ADC."""
    global CLIENT  # pylint: disable=global-statement
    if CLIENT is None:
        _, project = auth.default()
        CLIENT = genai.Client(
            vertexai=True, project=project, location="global")
    return CLIENT


def get_search_client():
    """Separate client for Google Search grounding (requires us-central1)."""
    global SEARCH_CLIENT  # pylint: disable=global-statement
    if SEARCH_CLIENT is None:
        _, project = auth.default()
        SEARCH_CLIENT = genai.Client(
            vertexai=True, project=project, location="us-central1")
    return SEARCH_CLIENT


SYSTEM_PROMPT = """
Your job: decide if vector_database_responses contain information that would genuinely help the user RIGHT NOW based on the current conversation.

Step 1 Validate thought_context against transcript:
Check if the thought_context is grounded in what has actually been said in the transcript.
If the thought_context is speculative or goes beyond what the transcript says, return status: "not_relevant".

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
- Based on the transcript, the user likely doesn't care about this information right now or already knows it.

Given:
- transcript: The conversation so far
- thought_context: Why Gemini Live made this query
- vector_database_responses: Historical data, your ONLY source from which to draw answers
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


WEB_EVAL_PROMPT = """
Your job: decide if web_search_result contains information that the user GENUINELY NEEDS RIGHT NOW based on the current conversation.

Step 1 – Validate the need:
Read the transcript and thought_context carefully.
Is the user clearly stuck, confused, or explicitly asking a factual question they cannot answer themselves?
If not, return status: "not_relevant".

Step 2 – Evaluate the web result:
Only return status: "found" if the web_search_result DIRECTLY answers the user's need in a way they would benefit from.
Do NOT include information that is already obvious from the transcript.
Do NOT repeat anything already in previous_queries_and_answers.
If status is "found", return exactly 1 concise sentence in the SAME LANGUAGE as the transcript, short enough for smart glasses.

Return status: "not_relevant" if:
- The user is just making conversation or musing aloud — they are not genuinely stuck
- The web result does not directly answer what the user is uncertain about
- The information is trivially available or already in the transcript
- The answer was already sent this session

Be strict. Err on the side of not sending.
"""


async def evaluate_web_data(
    transcript: str,
    web_search_result: str,
    thinking_context: str,
    query_history: list[dict],
) -> EvaluateResponse:
    """Evaluate if a web search result should be forwarded to the user."""
    contents = (
        f"full_conversation_transcript: {transcript}\n"
        f"thought_context: {thinking_context}\n"
        f"web_search_result: {web_search_result}\n"
        f"previous_queries_and_answers: {json.dumps(query_history)}"
    )

    client = get_client()
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=WEB_EVAL_PROMPT,
            response_mime_type="application/json",
            response_schema={
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["found"]},
                            "information": {"type": "string"},
                            "score": {"type": "number"},
                            "thinking": {"type": "string"},
                        },
                        "required": ["status", "information", "score", "thinking"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["not_relevant"]},
                            "thinking": {"type": "string"},
                        },
                        "required": ["status", "thinking"],
                    },
                ]
            },
        ),
    )

    print(f"[Gemini Tools] evaluate_web_data tokens: {response.usage_metadata.total_token_count}")
    data = response.parsed
    status = data.get("status")

    if status == "found":
        return {
            "status": "found",
            "information": data.get("information", ""),
            "score": data.get("score", 0.0),
            "thinking": data.get("thinking", ""),
        }

    return {
        "status": "not_relevant",
        "thinking": data.get("thinking", ""),
    }


async def fetch_general_knowledge(
    thinking_context: str,
    query: str,
    transcript: str,
    query_history: list[dict] | None = None,
) -> EvaluateResponse:
    """Search the web for general knowledge and return only if genuinely needed."""
    if not query:
        return {"status": "not_relevant", "thinking": "empty query"}

    try:
        print(f"[Gemini Tools] Web search query: {query}\nthinking: {thinking_context}")

        search_client = get_search_client()
        search_response = await search_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Answer this question accurately and concisely: {query}",
            config=genai.types.GenerateContentConfig(
                tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
                temperature=0.1,
            ),
        )

        web_text = search_response.text or ""
        if not web_text.strip():
            print("[Gemini Tools] Web search returned empty result")
            return {"status": "not_relevant", "thinking": "no web results"}

        print(f"[Gemini Tools] Web search result (truncated): {web_text[:200]}")

        return await evaluate_web_data(
            transcript,
            web_text,
            thinking_context,
            query_history or [],
        )

    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"[Gemini Tools] Web search failed: {error}")
        return {
            "status": "error",
            "error_message": f"Web search failed: {error}",
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
