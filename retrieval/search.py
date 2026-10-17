import argparse
import hashlib
import json
from pathlib import Path

from retrieval.embeddings import (
    embed_text,
    embedding_dimension,
    model_name,
)
from retrieval.vector_store import (
    collection_names,
    get_client,
)


project_folder = Path(__file__).resolve().parent.parent
artifacts_folder = project_folder / "artifacts"


def search(query, top_k=5, strategy="paragraph"):
    # 1. Validate the input.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a nonempty string.")

    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer between 1 and 20.")

    if strategy not in collection_names:
        raise ValueError("Strategy must be 'fixed' or 'paragraph'.")

    query = query.strip()
    collection_name = collection_names[strategy]

    # 2. Check the indexing state.
    state_path = (
        artifacts_folder / f"index_state_{strategy}.json"
    )

    if not state_path.exists():
        raise ValueError("Index state is missing. Run indexing first.")

    with state_path.open(encoding="utf-8") as file:
        state = json.load(file)

    if not state.get("complete"):
        raise ValueError("Indexing is incomplete. Finish it before searching.")

    # Reconstruct the settings used to identify the index.
    chunks_path = (
        artifacts_folder / f"chunks_{strategy}.json"
    )

    embedding_code_path = (
        project_folder / "retrieval" / "embeddings.py"
    )

    current_settings = {
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

    current_fingerprint = hashlib.sha256(
        json.dumps(
            current_settings,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    if current_fingerprint != state["fingerprint"]:
        raise ValueError(
            "The chunks, embedding settings, or embedding code "
            "changed after indexing. Review the index before searching."
        )

    client = get_client()

    try:
        # 3. Check the selected collection.
        if not client.collection_exists(
            collection_name=collection_name
        ):
            raise ValueError("The Qdrant collection is missing.")

        stored_count = client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        if stored_count != state["expected_points"]:
            raise ValueError(
                "The collection point count does not match the index state."
            )

        # 4. Embed the question using the query-specific format.
        query_vector = embed_text(query, kind="query")

        # 5. Retrieve the nearest vectors.
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        # 6. Convert Qdrant results into ordinary dictionaries.
        results = []

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload
            metadata = payload["metadata"]

            if (
                metadata["embedding_model"] != model_name
                or metadata["embedding_dimension"]
                != embedding_dimension
            ):
                raise ValueError(
                    "A retrieved point has incompatible embedding settings."
                )

            results.append({
                "rank": rank,
                "score": float(point.score),
                "text": payload["text"],
                "page": metadata["page"],
                "source_page": metadata["source_page"],
                "section": metadata["section"],
                "content_type": metadata["content_type"],
                "chunk_id": metadata["chunk_id"],
                "metadata": metadata,
            })

        return results

    finally:
        client.close()


# Run a command-line search when this file is executed.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", required=True)

    parser.add_argument(
        "--strategy",
        choices=["fixed", "paragraph"],
        default="paragraph",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    results = search(
        args.query,
        top_k=args.top_k,
        strategy=args.strategy,
    )

    print("Question:", args.query)
    print("Strategy:", args.strategy)
    print("Results returned:", len(results))

    for result in results:
        print(
            f"\nRank {result['rank']} | "
            f"Score {result['score']:.4f} | "
            f"Page {result['page']} | "
            f"Type {result['content_type']}"
        )

        print("Section:", result["section"])
        print("Content:")
        print(result["text"])