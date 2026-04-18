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

Step 1 – Validate thought_context against transcript:
Check if the thought_context is grounded in what has actually been said in the transcript.
If the thought_context is speculative or goes beyond what the transcript says, return status: "not_relevant".

Step 2 – Evaluate EACH vector individually, then decide:

For each vector ask: "Does this vector contain a fact that the user has NOT stated in the transcript?"
- If the user already stated the same fact (even partially), that vector is REDUNDANT — skip it.
- Partial overlap rule: if the user stated the main fact and the vector only adds ONE extra detail to that same fact, treat the entire vector as REDUNDANT — even if the extra detail seems useful. Examples of REDUNDANT vectors:
  * User says "flat face" → vector says "flat face and round eyes" → REDUNDANT
  * User says "long fur" → vector says "long fur that needs daily brushing" → REDUNDANT
  * User says "X has Y" → vector says "X has Y and also Z" → REDUNDANT (Z is not worth sending)
  The rule is strict: if the user stated the core subject+property, any vector that only extends that same subject+property is REDUNDANT.
- Stating vs asking: if the user is STATING a fact (not asking a question or expressing uncertainty), do NOT return expansions of that fact. "The cat has long fur" is a statement — do not respond with fur-care tips.
- If the vector contains a clearly DIFFERENT fact that stands on its own (e.g. user mentions player count, vector mentions match duration — these are independent facts), that is genuinely NEW.

After evaluating all vectors:
- If ANY vector is genuinely new and useful → return status: "found" with 1 concise sentence built from the new vectors only. Ignore redundant vectors entirely.
- If ALL vectors are redundant or unrelated → return status: "not_relevant".

Query history rule — IMPORTANT:
If previous_queries_and_answers already contains an answer about the same entity or topic (e.g. already answered about Toyota Yaris, already answered about pizza toppings), do NOT return additional facts about that same entity/topic even if they are technically new. The user has already been informed about that topic in this session. Return status: "not_relevant".

Additional rules:
- You may combine multiple genuinely new vectors into one sentence.
- The answer must be in the same language as the transcript.
- Keep it short — it is shown on smart glasses.

Return status: "not_relevant" if:
- The thought_context is not grounded in the transcript
- All vectors are redundant (already stated by the user, exactly or with only minor additions)
- The same topic was already answered in previous_queries_and_answers
- The user clearly does not need this information right now

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


async def _generate_with_retry(client, model: str, contents: str, config, label: str):
    """Call generate_content with exponential backoff on 429."""
    for attempt in range(3):
        try:
            return await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"[Gemini Tools] {label} rate limited, retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            raise


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

    config = genai.types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
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
    )

    response = await _generate_with_retry(
        get_client(), "gemini-2.5-flash-lite", contents, config, "evaluate_db_data"
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

    config = genai.types.GenerateContentConfig(
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
    )

    response = await _generate_with_retry(
        get_client(), "gemini-2.5-flash-lite", contents, config, "evaluate_web_data"
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
