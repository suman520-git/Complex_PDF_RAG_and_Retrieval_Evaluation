import hashlib
import json
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Locate the input and output files.
project_folder = Path(__file__).resolve().parent.parent
artifacts_folder = project_folder / "artifacts"

input_path = artifacts_folder / "units.json"
output_path = artifacts_folder / "chunks_paragraph.json"


# 2. Use the same size settings as the fixed-size strategy.
chunk_size = 1000
chunk_overlap = 150

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=[
        "\n\n",  # Prefer paragraph boundaries.
        "\n",    # Otherwise try line boundaries.
        ". ",    # Then try periods followed by spaces.
        " ",     # Then word boundaries.
        "",      # Finally split at character boundaries.
    ],
    is_separator_regex=False,
)


# 3. Load the same content units.
with input_path.open(encoding="utf-8") as file:
    units = json.load(file)

chunks = []


# 4. Split each unit independently.
for unit in units:
    text = unit["text"]

    if not text.strip():
        raise ValueError("An empty content unit was found.")

    pieces = splitter.split_text(text)

    for chunk_index, chunk_text in enumerate(pieces):
        metadata = unit["metadata"].copy()

        identity = (
            f"paragraph|{metadata['unit_id']}|"
            f"{chunk_index}|{chunk_text}"
        )

        chunk_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        metadata.update({
            "strategy": "paragraph",
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "character_count": len(chunk_text),
            "configured_chunk_size": chunk_size,
            "configured_overlap": chunk_overlap,
        })

        chunks.append({
            "text": chunk_text,
            "metadata": metadata,
        })


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

print("Strategy: paragraph")
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