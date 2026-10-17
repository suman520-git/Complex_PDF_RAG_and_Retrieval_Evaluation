import json
from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_FOLDER / "artifacts"


def main():
    report = []

    for strategy in ("fixed", "paragraph"):
        path = ARTIFACTS / f"evaluation_{strategy}.json"

        with path.open(encoding="utf-8") as file:
            evaluation = json.load(file)

        details_by_id = {
            detail["question"]["id"]: detail
            for detail in evaluation["details"]
        }

        report.append(f"\n=== {strategy.upper()} ===")

        for row in evaluation["rows"]:
            is_negative = row["category"] == "negative"

            failed = (
                not row["negative_abstention_pass"]
                if is_negative
                else not row["all_evidence_at_5"]
            )

            if not failed:
                continue

            detail = details_by_id[row["id"]]
            question = detail["question"]

            report.append(
                f"\n{row['id']} | {row['category']}\n"
                f"Question: {row['query']}\n"
                f"Expected: {question['expected_answer']}"
            )

            if is_negative:
                response = detail["response"]

                report.append(
                    f"Status: {response['status']}\n"
                    f"Answer: {response['answer']}\n"
                    f"Review reason: "
                    f"{response.get('review_reason', 'Not provided')}"
                )

                report.append(
                    "Citations:\n"
                    + json.dumps(
                        response.get("citations", []),
                        indent=2,
                        ensure_ascii=False,
                    )
                )

                results = response.get("results", [])

            else:
                ranks = detail["evidence_ranks"]

                for number, (evidence, rank) in enumerate(
                    zip(question["evidence"], ranks),
                    start=1,
                ):
                    report.append(
                        f"Evidence group {number}: "
                        f"page={evidence['page']}, "
                        f"type={evidence['content_type']}, "
                        f"rank={rank}, "
                        f"required={evidence['contains']}"
                    )

                results = detail["results"]

            report.append("Retrieved passages:")

            for result in results:
                report.append(
                    f"\nRank {result['rank']} | "
                    f"Score {result['score']:.4f} | "
                    f"Page {result['page']} | "
                    f"Source page {result['source_page']} | "
                    f"Type {result['content_type']}\n"
                    f"Section: {result['section']}\n"
                    f"Text: {result['text']}"
                )

    report_text = "\n".join(report)

    output_path = ARTIFACTS / "failure_inspection.txt"
    output_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()