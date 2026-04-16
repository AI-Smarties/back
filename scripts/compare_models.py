"""
Compare memory extraction results between two Gemini models.

Steps:
  1. Transcribe output.mp3 via Gemini Flash (multimodal audio support)
  2. Run memory_extractor_worker with both models on the same transcript
  3. Print side-by-side comparison — no database writes

Usage:
    python scripts/compare_models.py
"""

import asyncio
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# pylint: disable=wrong-import-position
import memory_extractor
from google import auth, genai

AUDIO_MP3 = Path(__file__).parent.parent / "output.mp3"

MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
]


# ---------------------------------------------------------------------------
# Step 1: Transcribe using Gemini Flash (supports audio natively)
# ---------------------------------------------------------------------------

async def transcribe_audio(mp3_path: Path) -> str:
    """Transcribe an MP3 file using Gemini Flash multimodal."""
    _, project = auth.default()
    client = genai.Client(vertexai=True, project=project, location="europe-north1")

    audio_bytes = mp3_path.read_bytes()

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part(
                        inline_data=genai.types.Blob(
                            mime_type="audio/mp3",
                            data=audio_bytes,
                        )
                    ),
                    genai.types.Part(text="Transcribe this audio verbatim. Output only the spoken text, nothing else."),
                ],
            )
        ],
    )
    await client.aio.aclose()
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Step 2: Extract with each model
# ---------------------------------------------------------------------------

async def run_extraction(transcript: str, model: str) -> dict:
    _, project = auth.default()
    client = genai.Client(vertexai=True, project=project, location="global")
    response = await client.aio.models.generate_content(
        model=model,
        contents=transcript,
        config=genai.types.GenerateContentConfig(
            system_instruction=memory_extractor.SYSTEM_PROMPT,
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
    await client.aio.aclose()
    return response.parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 70)
    print("STEP 1: Transcribing output.mp3 (Finnish) via Gemini Flash")
    print("=" * 70)
    transcript = await transcribe_audio(AUDIO_MP3)
    if not transcript:
        print("ERROR: Transcription returned empty result. Aborting.")
        return

    print(f"\nTRANSCRIPT ({len(transcript)} chars):")
    print("-" * 70)
    print(transcript)
    print("=" * 70)

    print("\nSTEP 2: Memory extraction with each model")
    results = {}
    for model in MODELS:
        print(f"\n  Running: {model} ...")
        try:
            result = await run_extraction(transcript, model)
            results[model] = result
            n = len(result.get("vectors", []))
            print(f"  -> {n} vectors extracted")
        except Exception as e:  # pylint: disable=broad-exception-caught
            results[model] = {"error": str(e)}
            print(f"  -> ERROR: {e}")

    print("\n" + "=" * 70)
    print("EXTRACTION RESULTS")
    print("=" * 70)

    for model, result in results.items():
        print(f"\n{'─' * 70}")
        print(f"MODEL: {model}")
        print(f"{'─' * 70}")
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        print(f"Name:    {result.get('name')}")
        vectors = result.get("vectors", [])
        print(f"Vectors: {len(vectors)}")
        for i, v in enumerate(vectors, 1):
            print(f"  [{i:2d}] {v['data']}")
            print(f"        reason: {v['reason']}")

    # Summary diff
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    valid = {m: r for m, r in results.items() if "error" not in r}
    for model, result in valid.items():
        print(f"  {model}: {len(result.get('vectors', []))} vectors")

    if len(valid) == 2:
        model_a, model_b = MODELS
        texts_a = {v["data"] for v in valid[model_a].get("vectors", [])}
        texts_b = {v["data"] for v in valid[model_b].get("vectors", [])}
        common = texts_a & texts_b
        only_a = texts_a - texts_b
        only_b = texts_b - texts_a
        print(f"\nShared vectors:       {len(common)}")
        print(f"Only in {model_a}: {len(only_a)}")
        for t in sorted(only_a):
            print(f"  - {t}")
        print(f"Only in {model_b}: {len(only_b)}")
        for t in sorted(only_b):
            print(f"  - {t}")


if __name__ == "__main__":
    asyncio.run(main())
