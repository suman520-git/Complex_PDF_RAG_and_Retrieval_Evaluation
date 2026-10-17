import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from retrieval.search import search

from retrieval.rag import answer_question

app = FastAPI(
    title="Complex PDF RAG and Retrieval Evaluation"
)

logger = logging.getLogger(__name__)


# Describe and validate the incoming JSON request.
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        strict=True,
    )

    strategy: Literal["fixed", "paragraph"] = "paragraph"

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Query must not be blank.")

        return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(request: QueryRequest):
    try:
        results = search(
            request.query,
            top_k=request.top_k,
            strategy=request.strategy,
        )

        return {
            "query": request.query,
            "strategy": request.strategy,
            "results": results,
        }

    except Exception:
        logger.exception("Search request failed.")

        raise HTTPException(
            status_code=503,
            detail=(
                "Search could not complete. "
                "Check the server terminal for details."
            ),
        )





@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    try:
        result = answer_question(
            request.query,
            top_k=request.top_k,
            strategy=request.strategy,
        )

        return {
            "query": request.query,
            "strategy": request.strategy,
            **result,
        }

    except Exception:
        logger.exception("Chat request failed.")

        raise HTTPException(
            status_code=503,
            detail=(
                "Answer generation could not complete. "
                "Check the server terminal for details."
            ),
        )    