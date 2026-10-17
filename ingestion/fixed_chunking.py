import hashlib
import json
from collections import Counter
from pathlib import Path


# 1. Locate the input and output files.
project_folder = Path(__file__).resolve().parent.parent
artifacts_folder = project_folder / "artifacts"

input_path = artifacts_folder / "units.json"
output_path = artifacts_folder / "chunks_fixed.json"


# 2. Configure chunking.
chunk_size = 1000
chunk_overlap = 150

if not 0 <= chunk_overlap < chunk_size:
    raise ValueError(
        "Overlap must be nonnegative and smaller than chunk size."
    )


# 3. Load our combined content.
with input_path.open(encoding="utf-8") as file:
    units = json.load(file)

chunks = []


# 4. Split each unit separately.
for unit in units:
    text = unit["text"]

    if not text.strip():
        raise ValueError("An empty content unit was found.")

    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk_text = text[start:end]

        # Copy all original metadata.
        metadata = unit["metadata"].copy()

        # Build a repeatable ID from the source and chunk content.
        identity = (
            f"fixed|{metadata['unit_id']}|"
            f"{chunk_index}|{chunk_text}"
        )

        chunk_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        metadata.update({
            "strategy": "fixed",
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "character_count": len(chunk_text),
            "start_character": start,
            "end_character": end,
            "configured_chunk_size": chunk_size,
            "configured_overlap": chunk_overlap,
        })

        chunks.append({
            "text": chunk_text,
            "metadata": metadata,
        })

        # Stop once we have included the end of this unit.
        if end == len(text):
            break

        # Move forward while retaining the overlap.
        start = end - chunk_overlap
        chunk_index += 1


# 5. Check the output.
chunk_ids = [
    chunk["metadata"]["chunk_id"]
    for chunk in chunks
]

if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError("Duplicate chunk IDs found.")

if any(len(chunk["text"]) > chunk_size for chunk in chunks):
    raise ValueError("A chunk exceeds the configured size.")


# 6. Save the chunks.
with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        chunks,
        file,
        indent=2,
        ensure_ascii=False,
    )


# 7. Print a summary.
counts = Counter(
    chunk["metadata"]["content_type"]
    for chunk in chunks
)

print("Strategy: fixed")
print("Input units:", len(units))
print("Total chunks:", len(chunks))
print("Chunks by content type:", dict(counts))
print(
    "Largest chunk:",
    max(len(chunk["text"]) for chunk in chunks),
)
print("Output file:", output_path)

print("\n--- First text chunk from page 3 ---")

for chunk in chunks:
    metadata = chunk["metadata"]

    if metadata["page"] == 3 and metadata["content_type"] == "text":
        print(json.dumps(chunk, indent=2, ensure_ascii=False))
        break