import json
from pathlib import Path

from retrieval.search import search


project_folder = Path(__file__).resolve().parent.parent
artifacts_folder = project_folder / "artifacts"


# These are exploratory checks, not our full 25-question evaluation.
test_questions = [
    {
        "category": "text",
        "query": (
            "How many jobs nationwide are supported by NASA "
            "according to the executive summary?"
        ),
        "expected_evidence": "304,803 jobs; excerpt page 4.",
    },
    {
        "category": "table",
        "query": (
            "In Table 2, what are California's P-card purchases "
            "and total procurement, in thousands of dollars?"
        ),
        "expected_evidence": (
            "California: P-card 12,619; total 5,849,741 "
            "thousand dollars; excerpt page 30."
        ),
    },
    {
        "category": "visual",
        "query": (
            "In executive-summary Figure 5, what percentage "
            "of Florida's NASA employment impacts is M2M?"
        ),
        "expected_evidence": (
            "Florida: 37%; corrected visual description "
            "on excerpt page 10."
        ),
    },
    {
        "category": "multi_section",
        "query": (
            "Compare the nationwide employment impacts of "
            "Moon to Mars and climate change research."
        ),
        "expected_evidence": (
            "M2M: 96,479 jobs on excerpt page 7; "
            "climate research: 32,900 jobs on excerpt page 11."
        ),
    },
]

runs = []

for question in test_questions:
    print("\n" + "=" * 60)
    print("Category:", question["category"])
    print("Question:", question["query"])
    print("Expected evidence:", question["expected_evidence"])

    for strategy in ["fixed", "paragraph"]:
        # Only the question goes into search.
        # Expected evidence is NOT passed to the retriever.
        results = search(
            question["query"],
            top_k=5,
            strategy=strategy,
        )

        print(f"\nStrategy: {strategy}")

        for result in results:
            print(
                f"Rank {result['rank']} | "
                f"Score {result['score']:.4f} | "
                f"Page {result['page']} | "
                f"Type {result['content_type']}"
            )

            preview = result["text"][:250].replace("\n", " ")
            print("Preview:", preview)

        runs.append({
            "category": question["category"],
            "query": question["query"],
            "expected_evidence": question["expected_evidence"],
            "strategy": strategy,
            "results": results,
        })


# Save complete retrieved text, not just the terminal previews.
output_path = artifacts_folder / "retrieval_checks.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        runs,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("\nSearch runs completed:", len(runs))
print("Full results saved to:", output_path)