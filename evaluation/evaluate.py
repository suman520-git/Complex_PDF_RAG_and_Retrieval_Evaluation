import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from retrieval.search import search
from retrieval.rag import answer_question


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_FOLDER / "artifacts"
QUESTIONS_FILE = PROJECT_FOLDER / "evaluation" / "questions.json"


def normalize(text):
    """Ignore capitalization and differences in whitespace."""
    return " ".join(text.casefold().split())


def matches_evidence(result, evidence):
    """Require the right page, content type, and text fragments."""
    if result["page"] != evidence["page"]:
        return False

    if result["content_type"] != evidence["content_type"]:
        return False

    text = normalize(result["text"])

    return all(
        normalize(fragment) in text
        for fragment in evidence["contains"]
    )


def validate_questions(questions):
    """Check labels against ingestion units before making API calls."""
    if len(questions) != 25:
        raise ValueError("Expected exactly 25 questions.")

    ids = [question["id"] for question in questions]

    if len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be unique.")

    expected_categories = {
        "easy": 5,
        "multi_section": 5,
        "table": 5,
        "visual": 5,
        "negative": 5,
    }

    counts = dict(Counter(q["category"] for q in questions))

    if counts != expected_categories:
        raise ValueError(f"Unexpected category counts: {counts}")

    with (ARTIFACTS / "units.json").open(
        encoding="utf-8"
    ) as file:
        units = json.load(file)

    # Convert ingestion units to the shape used by the matcher.
    source_records = [
        {
            "page": unit["metadata"]["page"],
            "content_type": unit["metadata"]["content_type"],
            "text": unit["text"],
        }
        for unit in units
    ]

    errors = []

    for question in questions:
        if not question["query"].strip():
            errors.append(f"{question['id']}: empty query")

        evidence_groups = question["evidence"]

        if question["category"] == "negative":
            if evidence_groups:
                errors.append(
                    f"{question['id']}: negative question has evidence"
                )
            continue

        if not evidence_groups:
            errors.append(f"{question['id']}: missing evidence")
            continue

        for number, evidence in enumerate(evidence_groups, start=1):
            fragments = evidence["contains"]

            if not fragments or any(
                not fragment.strip() for fragment in fragments
            ):
                errors.append(
                    f"{question['id']}: empty evidence fragments"
                )
                continue

            found = any(
                matches_evidence(record, evidence)
                for record in source_records
            )

            if not found:
                errors.append(
                    f"{question['id']}: evidence group {number} "
                    f"not found on page {evidence['page']}"
                )

    if errors:
        raise ValueError("\n".join(errors))

    print("Questions: 25")
    print(f"Categories: {counts}")
    print("Answerable questions: 20")
    print("Negative questions: 5")
    print("Evidence-label validation: PASS")
    print("PDF review is still needed to confirm the ground truth.")


def evaluate_positive(question, strategy):
    results = search(
        question["query"],
        top_k=5,
        strategy=strategy,
    )

    # One earliest matching rank for each required evidence group.
    evidence_ranks = []

    for evidence in question["evidence"]:
        rank = next(
            (
                position
                for position, result in enumerate(results, start=1)
                if matches_evidence(result, evidence)
            ),
            None,
        )
        evidence_ranks.append(rank)

    found_ranks = [
        rank for rank in evidence_ranks
        if rank is not None
    ]

    all_found = len(found_ranks) == len(evidence_ranks)

    # Complete evidence is available only once every group is present.
    complete_rank = max(found_ranks) if all_found else None

    expected_pages = {
        evidence["page"] for evidence in question["evidence"]
    }
    retrieved_pages = {result["page"] for result in results}

    row = {
        "id": question["id"],
        "category": question["category"],
        "query": question["query"],
        "any_evidence_at_5": bool(found_ranks),
        "all_expected_pages_at_5": expected_pages <= retrieved_pages,
        "first_evidence_rank": min(found_ranks) if found_ranks else None,
        "complete_evidence_rank": complete_rank,
        "all_evidence_at_1": complete_rank is not None and complete_rank <= 1,
        "all_evidence_at_3": complete_rank is not None and complete_rank <= 3,
        "all_evidence_at_5": complete_rank is not None and complete_rank <= 5,
        "evidence_recall_at_5": len(found_ranks) / len(evidence_ranks),
        "negative_abstention_pass": None,
        "comments": (
            "All required evidence retrieved"
            if all_found
            else "Missing evidence groups: " + ", ".join(
                str(index + 1)
                for index, rank in enumerate(evidence_ranks)
                if rank is None
            )
        ),
    }

    details = {
        "question": question,
        "evidence_ranks": evidence_ranks,
        "results": results,
    }

    return row, details


def evaluate_negative(question, strategy):
    response = answer_question(
        question["query"],
        top_k=5,
        strategy=strategy,
    )

    passed = (
        response["status"] == "insufficient_evidence"
        and response["citations"] == []
    )

    row = {
        "id": question["id"],
        "category": question["category"],
        "query": question["query"],
        "any_evidence_at_5": None,
        "all_expected_pages_at_5": None,
        "first_evidence_rank": None,
        "complete_evidence_rank": None,
        "all_evidence_at_1": None,
        "all_evidence_at_3": None,
        "all_evidence_at_5": None,
        "evidence_recall_at_5": None,
        "negative_abstention_pass": passed,
        "comments": (
            f"status={response['status']}; "
            f"answer={response['answer']}"
        ),
    }

    details = {
        "question": question,
        "response": response,
    }

    return row, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["fixed", "paragraph"],
        default="paragraph",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    with QUESTIONS_FILE.open(encoding="utf-8") as file:
        questions = json.load(file)

    validate_questions(questions)

    if args.validate_only:
        return

    rows = []
    details = []

    for question in questions:
        print(f"Evaluating {question['id']}...", flush=True)

        # An API error stops the run rather than being recorded
        # incorrectly as a retrieval failure or successful abstention.
        if question["category"] == "negative":
            row, detail = evaluate_negative(question, args.strategy)
        else:
            row, detail = evaluate_positive(question, args.strategy)

        rows.append(row)
        details.append(detail)

        print(f"  {row['comments']}")

    positive_rows = [
        row for row in rows if row["category"] != "negative"
    ]
    negative_rows = [
        row for row in rows if row["category"] == "negative"
    ]

    summary = {
        "strategy": args.strategy,
        "questions_sha256": hashlib.sha256(
            QUESTIONS_FILE.read_bytes()
        ).hexdigest(),
        "total_questions": len(rows),
        "answerable_questions": len(positive_rows),
        "negative_questions": len(negative_rows),
    }

    for k in (1, 3, 5):
        successes = sum(
            row[f"all_evidence_at_{k}"] for row in positive_rows
        )
        summary[f"complete_evidence_successes_at_{k}"] = successes
        summary[f"complete_evidence_rate_at_{k}"] = (
            successes / len(positive_rows)
        )

    summary["any_evidence_hit_rate_at_5"] = sum(
        row["any_evidence_at_5"] for row in positive_rows
    ) / len(positive_rows)

    summary["mean_evidence_recall_at_5"] = sum(
        row["evidence_recall_at_5"] for row in positive_rows
    ) / len(positive_rows)

    summary["negative_abstention_rate"] = sum(
        row["negative_abstention_pass"] for row in negative_rows
    ) / len(negative_rows)

    summary["complete_evidence_rate_at_5_by_category"] = {}

    for category in ("easy", "multi_section", "table", "visual"):
        category_rows = [
            row for row in positive_rows
            if row["category"] == category
        ]
        summary["complete_evidence_rate_at_5_by_category"][category] = (
            sum(row["all_evidence_at_5"] for row in category_rows)
            / len(category_rows)
        )

    ARTIFACTS.mkdir(exist_ok=True)

    csv_path = ARTIFACTS / f"evaluation_{args.strategy}.csv"
    json_path = ARTIFACTS / f"evaluation_{args.strategy}.json"

    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "rows": rows,
                "details": details,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n--- Evaluation summary ---")
    print(json.dumps(summary, indent=2))
    print(f"Evaluation table: {csv_path}")
    print(f"Full evidence: {json_path}")


if __name__ == "__main__":
    main()