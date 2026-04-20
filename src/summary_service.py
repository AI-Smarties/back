from google import auth, genai


# pylint: disable=duplicate-code


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


def generate_summary(transcript: str) -> str | None:
    transcript = transcript.strip()
    if not transcript:
        return None

    client = get_client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=transcript,
        config=genai.types.GenerateContentConfig(
            system_instruction=(
                "Summarize this meeting/session briefly and clearly. "
                "Focus on the key decision, topic, or outcome. "
                "Keep the summary in the same language as the transcript. "
            ),
        ),
    )

    text = getattr(response, "text", None)
    if not text:
        return None

    text = text.strip()
    return text or None
