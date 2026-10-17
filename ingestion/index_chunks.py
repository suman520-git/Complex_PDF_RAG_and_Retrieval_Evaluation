import argparse
import hashlib
import json
import uuid
from pathlib import Path

from qdrant_client import models

from retrieval.embeddings import (
    embed_text,
    embedding_dimension,
    model_name,
)
from retrieval.vector_store import (
    collection_names,
    ensure_collection,
    get_client,
)


# 1. Choose which chunking strategy to index.
parser = argparse.ArgumentParser()

parser.add_argument(
    "--strategy",
    choices=["fixed", "paragraph"],
    required=True,
)

args = parser.parse_args()
strategy = args.strategy


# 2. Locate the input and progress files.
project_folder = Path(__file__).resolve().parent.parent
artifacts_folder = project_folder / "artifacts"

chunks_path = artifacts_folder / f"chunks_{strategy}.json"
state_path = artifacts_folder / f"index_state_{strategy}.json"

with chunks_path.open(encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("The chunk file is empty.")

chunk_ids = [
    chunk["metadata"]["chunk_id"]
    for chunk in chunks
]

if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError("Duplicate chunk IDs found.")

if any(
    chunk["metadata"]["strategy"] != strategy
    for chunk in chunks
):
    raise ValueError("Chunk strategy does not match the selected strategy.")

collection_name = collection_names[strategy]


# 3. Identify this exact indexing input.
# This prevents mixing changed inputs into a partially filled collection.
embedding_code_path = (
    project_folder / "retrieval" / "embeddings.py"
)

settings = {
    "collection": collection_name,
    "strategy": strategy,
    "model": model_name,
    "dimensions": embedding_dimension,
    "chunks_sha256": hashlib.sha256(
        chunks_path.read_bytes()
    ).hexdigest(),
    "embedding_code_sha256": hashlib.sha256(
        embedding_code_path.read_bytes()
    ).hexdigest(),
}

fingerprint = hashlib.sha256(
    json.dumps(settings, sort_keys=True).encode("utf-8")
).hexdigest()


# 4. Connect to Qdrant.
client = get_client()

try:
    ensure_collection(client, collection_name)

    existing_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    if state_path.exists():
        with state_path.open(encoding="utf-8") as file:
            previous_state = json.load(file)

        if previous_state["fingerprint"] != fingerprint:
            raise ValueError(
                "Indexing inputs or embedding code changed. "
                "Stop and review before rebuilding this collection."
            )

    elif existing_count > 0:
        raise ValueError(
            "The collection contains points but has no local index state. "
            "Stop and inspect it before adding more data."
        )

    # Mark the run incomplete until every point is stored and counted.
    state = {
        **settings,
        "fingerprint": fingerprint,
        "expected_points": len(chunks),
        "complete": False,
    }

    with state_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    print("\nIndexing strategy:", strategy)
    print("Target collection:", collection_name)
    print("Chunks to index:", len(chunks))

    # 5. Embed and store each chunk.
    for position, chunk in enumerate(chunks, start=1):
        vector = embed_text(
            chunk["text"],
            kind="document",
        )

        # Convert the chunk ID into a repeatable UUID for Qdrant.
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["metadata"]["chunk_id"],
            )
        )

        metadata = chunk["metadata"].copy()

        metadata.update({
            "embedding_model": model_name,
            "embedding_dimension": embedding_dimension,
        })

        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "text": chunk["text"],
                "metadata": metadata,
            },
        )

        client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True,
        )

        if (
            position == 1
            or position % 20 == 0
            or position == len(chunks)
        ):
            print(f"Stored {position}/{len(chunks)} chunks")

    # 6. Verify the final point count.
    final_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    if final_count != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} points, found {final_count}."
        )

    state["complete"] = True
    state["stored_points"] = final_count

    with state_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    # 7. Read one stored point for inspection.
    sample_points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    sample = sample_points[0]

    print("\n--- Stored sample ---")
    print("Point ID:", sample.id)
    print("Vector dimensions:", len(sample.vector))
    print(
        "Metadata:",
        json.dumps(sample.payload["metadata"], indent=2),
    )
    print("Text preview:", sample.payload["text"][:200])

    print("\nExpected points:", len(chunks))
    print("Stored points:", final_count)
    print("Indexing validation: PASS")

finally:
    client.close()