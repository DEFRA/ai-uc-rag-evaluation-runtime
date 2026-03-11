from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import app.evaluation.evaluation_service as evaluation_service
import app.evaluation.rag_answer_service as rag_answer_service

router = APIRouter(tags=["evaluation"])


@router.get("/evaluation/rag")
async def trigger_rag_evaluation(
    group_id: Annotated[str, Query(..., description="Knowledge group ID")],
    query: Annotated[str, Query(..., description="Search query")],
    expected_answer: Annotated[str, Query(..., description="Expected answer")],
    max_results: Annotated[
        int, Query(ge=1, le=100, description="Maximum number of results")
    ] = 5,
) -> JSONResponse:
    """
    Trigger a RAG evaluation by querying the evaluation-data service snapshots endpoint.
    """
    answer = await rag_answer_service.answer_with_rag(query, group_id, max_results)

    evaluated_content = await evaluation_service.evaluate_pydantic(
        query,
        expected_answer,
        answer,
    )

    return JSONResponse(content=evaluated_content)


@router.get("/rag/answer")
async def rag_answer(
    group_id: Annotated[str, Query(description="Knowledge group ID")],
    query: Annotated[str, Query(description="Question to answer")],
    max_results: Annotated[
        int, Query(ge=1, le=100, description="Maximum number of context documents")
    ] = 5,
) -> JSONResponse:
    """Answer a question using RAG context retrieved from the evaluation-data service."""
    answer = await rag_answer_service.answer_with_rag(query, group_id, max_results)
    return JSONResponse(content={"answer": answer})
