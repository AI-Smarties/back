from google import auth, genai

MODEL = "gemini-2.5-flash"
_, PROJECT = auth.default()

SYSTEM_PROMPT = """
You are summarizing a final validated transcript from a spoken conversation.

Your task:
- Write a concise factual summary in plain text.
- Focus on the most important discussion points, decisions, plans, and follow-up items.
- Do not invent information.
- Do not add interpretation that is not supported by the transcript.
- Keep the summary compact and useful for later review.
- Prefer 3-6 sentences.
- If the transcript is too short or contains no meaningful content, return an empty string.
"""


def generate_summary(transcript: str) -> str:
    """
    Generate a concise summary from the final transcript text using Gemini.

    Returns:
        str: summary text, or empty string if transcript is empty or not useful.

    Raises:
        Exception: propagated to caller so the caller can handle generation errors.
    """
    transcript = transcript.strip()
    if not transcript:
        return ""

    if len(transcript) < 20:
        return ""

    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location="europe-north1",
    )

    prompt = f"{SYSTEM_PROMPT}\n\nTranscript:\n{transcript}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    summary = getattr(response, "text", "") or ""
    return summary.strip()
