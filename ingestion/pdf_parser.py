import json
from pathlib import Path

import pymupdf


# 1. Locate input and output files.
project_folder = Path(__file__).resolve().parent.parent

pdf_path = project_folder / "data" / "document.pdf"

output_folder = project_folder / "artifacts"
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "pages.json"

if not pdf_path.is_file():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")


# 2. Section ranges for this specific 34-page PDF.
# Each tuple contains:
# (first page, last page, section name)
section_ranges = [
    (1, 1, "Cover"),
    (2, 2, "Acknowledgments"),
    (3, 5, "Executive Summary / NASA"),
    (6, 9, "Executive Summary / Moon to Mars"),
    (10, 14, "Executive Summary / Climate Research"),
    (15, 17, "Introduction / Economic Impact Methodology"),
    (18, 19, "Introduction / NASA Organization"),
    (20, 21, "Introduction / M2M and Climate Programs"),
    (22, 22, "Report Organization"),
    (23, 25, "Literature Survey"),
    (26, 28, "Data Analysis / NASA Employment"),
    (29, 31, "Data Analysis / NASA Procurement"),
    (32, 34, "Data Analysis / M2M Employment"),
]


# 3. Extract every page and attach metadata.
records = []

with pymupdf.open(pdf_path) as document:
    if len(document) != 34:
        raise ValueError(
            "This section mapping expects the supplied 34-page PDF."
        )

    for page_index in range(len(document)):
        page_number = page_index + 1

        page = document[page_index]

        text = page.get_text("text", sort=True).strip()

        # Find the section containing this page.
        section = "Unknown"

        for first_page, last_page, section_name in section_ranges:
            if first_page <= page_number <= last_page:
                section = section_name
                break

        # Our excerpt contains original PDF pages:
        # 1–14, followed by 50–69.
        if page_number <= 14:
            source_page = page_number
        else:
            source_page = page_number + 35

        record = {
            "text": text,
            "metadata": {
                "document": pdf_path.name,
                "page": page_number,
                "source_page": source_page,
                "section": section,
                "section_method": "reviewed_page_mapping",
                "content_type": "text",
            },
        }

        records.append(record)

        print(
            f"Page {page_number} | "
            f"Source page {source_page} | "
            f"{len(text)} characters | "
            f"{section}"
        )


# 4. Save all records as readable JSON.
with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        records,
        file,
        indent=2,
        ensure_ascii=False,
    )


# 5. Print a short processing summary.
empty_pages = [
    record["metadata"]["page"]
    for record in records
    if not record["text"]
]

print("\nSaved records:", len(records))
print("Pages without extracted text:", empty_pages)
print("Output file:", output_path)

print("\nMetadata for excerpt page 27:")
print(json.dumps(records[26]["metadata"], indent=2))