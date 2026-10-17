import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# 1. Load project configuration.
project_folder = Path(__file__).resolve().parent.parent

load_dotenv(project_folder / ".env")

qdrant_url = os.getenv(
    "QDRANT_URL",
    "http://localhost:6335",
)

embedding_dimension = int(
    os.getenv("EMBEDDING_DIM", "768")
)

collection_names = {
    "fixed": "complex_pdf_fixed",
    "paragraph": "complex_pdf_paragraph",
}


# 2. Provide a reusable connection function.
def get_client():
    return QdrantClient(
        url=qdrant_url,
        timeout=30,
    )


# 3. Create a collection or check its existing configuration.
def ensure_collection(client, collection_name):
    exists = client.collection_exists(
        collection_name=collection_name
    )

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )

        print("Created collection:", collection_name)

    else:
        print("Collection already exists:", collection_name)

    # Read the actual collection configuration.
    collection_info = client.get_collection(
        collection_name=collection_name
    )

    vector_config = collection_info.config.params.vectors

    # This project uses one unnamed dense vector per record.
    if not isinstance(vector_config, models.VectorParams):
        raise ValueError(
            f"{collection_name} has an unexpected vector configuration."
        )

    if vector_config.size != embedding_dimension:
        raise ValueError(
            f"{collection_name} has {vector_config.size} dimensions, "
            f"but our configuration requires {embedding_dimension}."
        )

    if vector_config.distance != models.Distance.COSINE:
        raise ValueError(
            f"{collection_name} does not use cosine similarity."
        )

    # Get an exact count of stored records.
    stored_points = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    print("Vector dimensions:", vector_config.size)
    print("Distance metric:", vector_config.distance.value)
    print("Stored points:", stored_points)
    print("Collection validation: PASS")


# 4. Run setup when this file is executed directly.
if __name__ == "__main__":
    client = get_client()

    try:
        # This request verifies that Qdrant is reachable.
        client.get_collections()

        print("Connected to Qdrant:", qdrant_url)

        for strategy, collection_name in collection_names.items():
            print(f"\n--- {strategy.upper()} ---")

            ensure_collection(
                client,
                collection_name,
            )

    finally:
        client.close()