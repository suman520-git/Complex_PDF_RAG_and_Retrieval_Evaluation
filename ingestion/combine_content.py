import json
from collections import Counter
from pathlib import Path


# 1. Locate the processing results.
project_folder = Path(__file__).resolve().parent.parent
output_folder = project_folder / "artifacts"


# 2. Load the three extraction outputs.
with (output_folder / "pages.json").open(
    encoding="utf-8"
) as file:
    pages = json.load(file)

with (output_folder / "tables.json").open(
    encoding="utf-8"
) as file:
    tables = json.load(file)

with (output_folder / "visuals.json").open(
    encoding="utf-8"
) as file:
    visuals = json.load(file)

units = []


# 3. Add each nonempty text page.
for record in pages:
    text = record["text"].strip()

    if not text:
        continue

    metadata = record["metadata"].copy()

    page_number = metadata["page"]
    metadata["unit_id"] = f"text_p{page_number}"

    units.append({
        "text": text,
        "metadata": metadata,
    })


# 4. Convert each table row into labeled text.
for table in tables:
    for row_number, row in enumerate(table["rows"], start=1):
        parts = []

        for column_name, value in row.items():
            parts.append(f"{column_name}: {value}")

        row_text = "; ".join(parts)

        metadata = table["metadata"].copy()

        table_id = metadata["table_id"]
        page_number = metadata["page"]

        metadata["row_number"] = row_number
        metadata["unit_id"] = (
            f"{table_id}_p{page_number}_r{row_number}"
        )

        # Include the table title to give the row context.
        text = metadata["section"] + "\n" + row_text

        units.append({
            "text": text,
            "metadata": metadata,
        })


# 5. Add each visual-page description.
for record in visuals:
    text = record["text"].strip()

    if not text:
        raise ValueError("An empty visual description was found.")

    metadata = record["metadata"].copy()

    page_number = metadata["page"]
    metadata["unit_id"] = f"visual_p{page_number}"

    units.append({
        "text": text,
        "metadata": metadata,
    })


# 6. Validate the common structure.
unit_ids = set()

for unit in units:
    metadata = unit["metadata"]

    for field in [
        "document",
        "page",
        "section",
        "content_type",
        "unit_id",
    ]:
        if field not in metadata:
            raise ValueError(f"Missing metadata field: {field}")

    if not 1 <= metadata["page"] <= 34:
        raise ValueError("Invalid page number.")

    if metadata["unit_id"] in unit_ids:
        raise ValueError("Duplicate unit ID.")

    unit_ids.add(metadata["unit_id"])


# 7. Save the combined content.
output_path = output_folder / "units.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        units,
        file,
        indent=2,
        ensure_ascii=False,
    )


# 8. Print a summary.
counts = Counter(
    unit["metadata"]["content_type"]
    for unit in units
)

print("Text units:", counts["text"])
print("Table-row units:", counts["table"])
print("Visual units:", counts["visual"])
print("Total units:", len(units))
print("Unique unit IDs:", len(unit_ids))
print("Output file:", output_path)

print("\n--- First table-row unit ---")

for unit in units:
    if unit["metadata"]["content_type"] == "table":
        print(json.dumps(unit, indent=2, ensure_ascii=False))
        break