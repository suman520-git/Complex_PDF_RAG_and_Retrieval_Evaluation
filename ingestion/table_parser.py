import json
import re
from pathlib import Path

import pymupdf


# 1. Locate the input PDF and output file.
project_folder = Path(__file__).resolve().parent.parent

pdf_path = project_folder / "data" / "document.pdf"

output_folder = project_folder / "artifacts"
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "tables.json"


# 2. Identify which table appears on each page.
table_pages = {
    27: 1,
    28: 1,
    29: 1,
    30: 2,
    31: 2,
    32: 3,
    33: 3,
    34: 3,
}


# 3. Define column names, including their units.
table_headers = {
    1: [
        "State",
        "Jobs (FTE)",
        "Employment share (%)",
        "Labor income ($ thousands)",
        "Income share (%)",
    ],
    2: [
        "State",
        "Procurement ($ thousands)",
        "Procurement share (%)",
        "P-card ($ thousands)",
        "P-card share (%)",
        "Total ($ thousands)",
        "Total share (%)",
    ],
    3: [
        "State",
        "M2M jobs (FTE)",
        "Employment share (%)",
        "M2M labor income ($ thousands)",
        "Income share (%)",
    ],
}

table_titles = {
    1: "NASA labor force and associated labor income",
    2: "NASA procurement",
    3: "M2M campaign labor force and associated labor income",
}


# 4. Store one record for each table page.
table_records = []

with pymupdf.open(pdf_path) as document:
    if len(document) != 34:
        raise ValueError(
            "This table layout expects the supplied 34-page PDF."
        )

    for page_number, table_number in table_pages.items():
        page = document[page_number - 1]

        headers = table_headers[table_number]

        # Horizontal positions separating the table columns.
        # Coordinates are PDF points, not character positions.
        if table_number == 2:
            boundaries = [175, 245, 300, 370, 420, 490]
        else:
            boundaries = [235, 320, 390, 480]

        # Each word includes its position on the page.
        words = page.get_text("words")

        # Sort words from top to bottom, then left to right.
        words = sorted(
            words,
            key=lambda word: (word[1], word[0]),
        )

        # 5. Group words that appear on approximately the same line.
        line_groups = []

        for word in words:
            word_top = word[1]

            if (
                not line_groups
                or abs(line_groups[-1]["top"] - word_top) > 3
            ):
                line_groups.append({
                    "top": word_top,
                    "words": [],
                })

            line_groups[-1]["words"].append(word)

        # 6. Assign each word to a table column.
        rows = []

        for group in line_groups:
            cells = [""] * len(headers)

            line_words = sorted(
                group["words"],
                key=lambda word: word[0],
            )

            for word in line_words:
                word_left = word[0]
                word_text = word[4]

                column_index = 0

                for boundary in boundaries:
                    if word_left >= boundary:
                        column_index += 1

                if cells[column_index]:
                    cells[column_index] += " "

                cells[column_index] += word_text

            # 7. Accept rows with a label followed by numeric cells.
            # Examples accepted: 2,224, 12.5, 0, <0.1
            numeric_cells_valid = all(
                re.fullmatch(r"<?[\d,.]+", cell)
                for cell in cells[1:]
            )

            if cells[0] and numeric_cells_valid:
                row = dict(zip(headers, cells))
                rows.append(row)

        if not rows:
            raise ValueError(
                f"No table rows extracted from page {page_number}."
            )

        # 8. Save the rows with their provenance.
        table_record = {
            "metadata": {
                "document": pdf_path.name,
                "page": page_number,
                "source_page": page_number + 35,
                "section": table_titles[table_number],
                "content_type": "table",
                "table_id": f"table_{table_number}",
                "extraction_method": "reviewed_column_layout",
            },
            "columns": headers,
            "rows": rows,
        }

        table_records.append(table_record)

        print(
            f"Page {page_number} | "
            f"Table {table_number} | "
            f"Rows: {len(rows)}"
        )


# 9. Save the extracted tables.
with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        table_records,
        file,
        indent=2,
        ensure_ascii=False,
    )

total_rows = sum(
    len(record["rows"])
    for record in table_records
)

print("\nTable-page records:", len(table_records))
print("Total extracted rows:", total_rows)
print("Output file:", output_path)


# 10. Check important values against the supplied PDF.
expected_values = [
    (
        27,
        "Alabama",
        "Jobs (FTE)",
        "2,224",
    ),
    (
        27,
        "Alabama",
        "Labor income ($ thousands)",
        "422,863",
    ),
    (
        28,
        "Maryland",
        "Jobs (FTE)",
        "3,067",
    ),
    (
        28,
        "Maryland",
        "Labor income ($ thousands)",
        "650,169",
    ),
    (
        30,
        "California",
        "P-card ($ thousands)",
        "12,619",
    ),
    (
        30,
        "California",
        "Total ($ thousands)",
        "5,849,741",
    ),
    (
        31,
        "Texas",
        "Total ($ thousands)",
        "2,342,718",
    ),
    (
        33,
        "Texas",
        "M2M jobs (FTE)",
        "1,005",
    ),
]

print("\n--- Table validation ---")

for page_number, state, column, expected in expected_values:
    actual = None

    for record in table_records:
        if record["metadata"]["page"] != page_number:
            continue

        for row in record["rows"]:
            if row["State"] == state:
                actual = row[column]
                break

    if actual != expected:
        raise ValueError(
            f"Validation failed: page {page_number}, "
            f"{state}, {column}. "
            f"Expected {expected}, received {actual}."
        )

    print(
        f"PASS | Page {page_number} | "
        f"{state} | {column}: {actual}"
    )

print("\nAll 8 table-value checks passed.")