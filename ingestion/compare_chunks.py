import json
from pathlib import Path


# 1. Locate our processing results.
project_folder = Path(__file__).resolve().parent.parent
artifacts_folder = project_folder / "artifacts"

with (artifacts_folder / "units.json").open(
    encoding="utf-8"
) as file:
    units = json.load(file)

# Allow us to find an original unit using its ID.
units_by_id = {
    unit["metadata"]["unit_id"]: unit
    for unit in units
}


# 2. Inspect both strategies.
for strategy in ["fixed", "paragraph"]:
    input_path = artifacts_folder / f"chunks_{strategy}.json"

    with input_path.open(encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError(f"No chunks found for {strategy}.")

    seen_chunk_ids = set()
    covered_unit_ids = set()
    lengths = []

    for chunk in chunks:
        text = chunk["text"]
        metadata = chunk["metadata"]

        unit_id = metadata["unit_id"]
        chunk_id = metadata["chunk_id"]

        # Verify that the original unit exists.
        if unit_id not in units_by_id:
            raise ValueError(f"Unknown source unit: {unit_id}")

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        if not text.strip():
            raise ValueError(f"Empty chunk: {chunk_id}")

        if len(text) > 1000:
            raise ValueError(f"Oversized chunk: {chunk_id}")

        if metadata["character_count"] != len(text):
            raise ValueError(
                f"Incorrect character count: {chunk_id}"
            )

        if metadata["strategy"] != strategy:
            raise ValueError("Incorrect strategy metadata.")

        # Check that every original metadata field was preserved.
        original_metadata = units_by_id[unit_id]["metadata"]

        for field, original_value in original_metadata.items():
            if metadata.get(field) != original_value:
                raise ValueError(
                    f"Metadata changed for {unit_id}: {field}"
                )

        seen_chunk_ids.add(chunk_id)
        covered_unit_ids.add(unit_id)
        lengths.append(len(text))

    # Verify that no complete unit was dropped.
    missing_units = set(units_by_id) - covered_unit_ids

    if missing_units:
        raise ValueError(f"Missing units: {missing_units}")

    # 3. Print the comparison information.
    print(f"\n--- {strategy.upper()} ---")
    print("Total chunks:", len(chunks))
    print("Source units covered:", len(covered_unit_ids))
    print("Unique chunk IDs:", len(seen_chunk_ids))
    print("Smallest chunk:", min(lengths))
    print("Largest chunk:", max(lengths))
    print(
        "Average chunk size:",
        round(sum(lengths) / len(lengths), 1),
    )
    print("Validation: PASS")

    # Inspect how the first text chunk on page 3 ends.
    for chunk in chunks:
        metadata = chunk["metadata"]

        if (
            metadata["page"] == 3
            and metadata["content_type"] == "text"
            and metadata["chunk_index"] == 0
        ):
            print("\nLast 120 characters of page 3's first chunk:")
            print(chunk["text"][-120:])
            break