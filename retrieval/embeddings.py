import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

import hashlib


# 1. Load configuration.
project_folder = Path(__file__).resolve().parent.parent

load_dotenv(project_folder / ".env")

model_name = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-2",
)

embedding_dimension = int(
    os.getenv("EMBEDDING_DIM", "768")
)


# 2. Define a reusable embedding function.
def embed_text(text, kind="document"):
    if not text.strip():
        raise ValueError("Cannot embed empty text.")

    if kind not in ["document", "query"]:
        raise ValueError("kind must be 'document' or 'query'.")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key.startswith("your_"):
        raise ValueError("Set your actual GEMINI_API_KEY in .env.")

    # Format documents and queries for their different roles.
    if kind == "document":
        prepared_text = (
            "title: NASA Economic Impact Study 2023 | text: "
            + text
        )
    else:
        prepared_text = "task: search result | query: " + text


        # Reuse an embedding if this exact input was processed before.
    cache_folder = (
        project_folder / "artifacts" / "embedding_cache"
    )
    cache_folder.mkdir(parents=True, exist_ok=True)

    cache_input = json.dumps(
        {
            "model": model_name,
            "dimensions": embedding_dimension,
            "text": prepared_text,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    cache_key = hashlib.sha256(
        cache_input.encode("utf-8")
    ).hexdigest()

    cache_path = cache_folder / f"{cache_key}.json"

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as file:
            cached_vector = json.load(file)

        if (
            len(cached_vector) != embedding_dimension
            or not all(math.isfinite(v) for v in cached_vector)
        ):
            raise ValueError(
                f"Invalid cached embedding: {cache_path}"
            )

        return cached_vector


    # Send one text and receive one embedding.
    with genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000),
    ) as client:
        response = client.models.embed_content(
            model=model_name,
            contents=prepared_text,
            config=types.EmbedContentConfig(
                output_dimensionality=embedding_dimension,
            ),
        )

    if not response.embeddings or len(response.embeddings) != 1:
        raise ValueError("Expected exactly one embedding.")

    vector = response.embeddings[0].values

    if vector is None or len(vector) != embedding_dimension:
        raise ValueError("Unexpected embedding dimensions.")

    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains invalid numbers.")

    # Normalize the vector to length 1.
    magnitude = math.sqrt(
        sum(value * value for value in vector)
    )

    if magnitude == 0:
        raise ValueError("Received a zero-length vector.")

    normalized_vector = [
        value / magnitude
        for value in vector
    ]

    # Save only a successfully generated and normalized vector.
    # Write to a temporary file first, then replace the final file.
    temporary_path = cache_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(normalized_vector, file)

    temporary_path.replace(cache_path)


    return normalized_vector


# 3. Run this test only when this file is executed directly.
if __name__ == "__main__":
    chunks_path = (
        project_folder
        / "artifacts"
        / "chunks_paragraph.json"
    )

    with chunks_path.open(encoding="utf-8") as file:
        chunks = json.load(file)

    # Select the first text chunk from page 3.
    selected_chunk = next(
        chunk
        for chunk in chunks
        if chunk["metadata"]["page"] == 3
        and chunk["metadata"]["content_type"] == "text"
        and chunk["metadata"]["chunk_index"] == 0
    )

    text = selected_chunk["text"]

    print("Model:", model_name)
    print("Source page:", selected_chunk["metadata"]["page"])
    print("Text characters:", len(text))

    vector = embed_text(text, kind="document")

    print("Embedding dimensions:", len(vector))
    print("First 5 values:", vector[:5])

    normalized_magnitude = math.sqrt(
        sum(value * value for value in vector)
    )

    print(
        "Normalized vector magnitude:",
        round(normalized_magnitude, 6),
    )

    print("Embedding test: PASS")