"""
Tool functions for Gemini Live API.
These functions can be called by Gemini as tool calls.
"""
from typing import Literal, TypedDict, Sequence
from models import Vector
from db_utils import search_vectors
from google import genai

client = genai.Client()

system_prompt = """
You are evaluating whether vector database results are relevant to an ongoing conversation. You can use transcript and thought_context as help but
only use the vector_database_responses as source of truth. Dont sent information to user if transcript is providing the information already

Given:
- transcript: The conversation transcript
- thought_context: Why the query was made by Gemini Live
- vector_database_responses: Data from vector database, use only these as source of information

Decide:
- "found": results are relevant and useful to surface to the user → write a concise 1 sentence summary suitable for smart glasses display
- "not_relevant": results exist but are not actually relevant to the current moment
- "error": something is wrong with the inputs

you return allways your thought context
Be strict. Only return "found" if the information would genuinely help the user right now.
"""

class SendToUserResponse(TypedDict):
    status: Literal["found"]    # result is relevant, send to user
    information: str
    score: float
    thinking: str

class DontSendToUserResponse(TypedDict):
    status: Literal["not_relevant"]    # query worked but nothing matched
    thinking: str

class Error(TypedDict):
    status: Literal['error']    # fetch failed (network, db, etc.)
    error_message: str

EvaluateResponse = SendToUserResponse | DontSendToUserResponse | Error

async def evaluate_db_data(transcript: str, vector_database_response: Sequence[Vector], thinking_context: str) -> EvaluateResponse:
    formatted_vectors = "\n".join(
        f"- {vector.text}"
        for vector in vector_database_response
    )
    print(formatted_vectors)
    contents = (
        f"full_conversation_transcript: {transcript}\n"
        f"thought_context: {thinking_context}\n"
        f"vector_database_responses:\n{formatted_vectors}"
    )
    ## might throw 503 error, googles services are overloaded
    ## todo create retry
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema={
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "status":      {"type": "string", "enum": ["found"]},
                            "information": {"type": "string"},
                            "score":       {"type": "number"},
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
            }
        )
    )

    data = response.parsed
    status = data.get("status")
    if status == "found":
        return {"status": "found", "information": data.get("information", ""), "score": data.get("score", 0.0), "thinking": data.get("thinking", "")}
    elif status == "not_relevant":
        return {"status": "not_relevant", "thinking": data.get("thinking", "")}
    else:
        return {"status": "error", "error_message": data.get("error_message", "unknown")}


async def fetch_information(thinking_context: str, query: str, transcript: str) -> EvaluateResponse:
    """
    Fetch useful information based on a text query from vector database.

    Args:
        thinking_context: Thinking context of gemini live why it invoked this function
        query: The text query to search for information
        transcript: Transcript of the whole conversation

    Returns:
        EvaluateResponse — one of:
            {
                "status": "found",     // found relevant information from database that should be sent to client
                "information": str,     // Ai summary of the information
                "score": float,         // 0-1 score of relevance
                "thinking": str         // Ai thinking process
            }

            {
                "status": "not_relevant"  // relevant data not found
                "thinking": str           // Ai thinking process
            }

            {
                "status": "error",
                "error_message": str
            }
    """
    if not query:
        return {"status": "not_relevant", "thinking": ""}
    try:
        # fetch from database the closest things that are stored in database
        print(f"query: {query} \n thinking: {thinking_context}")
        results = search_vectors(query, limit=5, max_distance=0.3)
        if not results:
            print("no vector data")
            return {"status": "not_relevant", "thinking": ""}
        return await evaluate_db_data(transcript, results, thinking_context)
    except Exception as e:
        return {"status": "error", "error_message": f"Failed to fetch information: {e}"}
