import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_FOLDER / "artifacts"

API_URL = "http://127.0.0.1:8000/chat"

# Two questions from each of the five categories.
DEMO_IDS = [
    "Q01", "Q05",
    "Q06", "Q08",
    "Q11", "Q13",
    "Q16", "Q20",
    "Q21", "Q24",
]


def main():
    questions_path = PROJECT_FOLDER / "evaluation" / "questions.json"

    with questions_path.open(encoding="utf-8") as file:
        questions = json.load(file)

    questions_by_id = {
        question["id"]: question
        for question in questions
    }

    demonstrations = []
    ARTIFACTS.mkdir(exist_ok=True)
    output_path = ARTIFACTS / "live_demonstrations.json"

    for question_id in DEMO_IDS:
        question = questions_by_id[question_id]

        payload = {
            "query": question["query"],
            "top_k": 5,
            "strategy": "paragraph",
        }

        request = Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"\n{question_id}: {question['query']}", flush=True)

        record = {
            "id": question_id,
            "category": question["category"],
            "expected_answer": question["expected_answer"],
            "request": payload,
        }

        try:
            with urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
                record["http_status"] = response.status
                record["response"] = body

            pages = [
                citation["page"]
                for citation in body.get("citations", [])
            ]

            print(f"HTTP status: {record['http_status']}")
            print(f"Application status: {body.get('status')}")
            print(f"Answer: {body.get('answer')}")
            print(f"Cited pages: {pages}")

            if body.get("review_reason"):
                print(f"Review reason: {body['review_reason']}")

        except HTTPError as error:
            record["http_status"] = error.code
            record["error"] = error.read().decode(
                "utf-8", errors="replace"
            )
            print(f"HTTP error: {error.code}")
            print(record["error"])

        except (URLError, TimeoutError) as error:
            record["http_status"] = None
            record["error"] = str(error)
            print(f"Connection error: {error}")

        demonstrations.append(record)

        # Save after every request so completed results are preserved.
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                demonstrations,
                file,
                indent=2,
                ensure_ascii=False,
            )

    completed = sum(
        record.get("http_status") == 200
        for record in demonstrations
    )

    print(f"\nHTTP 200 responses: {completed}/{len(DEMO_IDS)}")
    print(f"Saved demonstrations: {output_path}")
    print("HTTP 200 means the request succeeded, not that the answer is correct.")


if __name__ == "__main__":
    main()