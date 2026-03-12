from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import app.evaluation.evaluation_service as evaluation_service
import app.evaluation.rag_answer_service as rag_answer_service
import app.evaluation.sqs_service as sqs_service

router = APIRouter(tags=["evaluation"])


class QueueEvaluationRequest(BaseModel):
    group_id: str
    query: str
    expected_answer: str
    snapshot_id: str | None = None


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
    result = await evaluation_service.run_rag_evaluation(
        group_id, query, expected_answer, max_results
    )
    return JSONResponse(content=result)


@router.post("/evaluation/queue")
async def queue_evaluation(request: QueueEvaluationRequest) -> JSONResponse:
    """Enqueue a RAG evaluation request for background processing."""
    await sqs_service.enqueue_evaluation(
        request.group_id, request.query, request.expected_answer, request.snapshot_id
    )
    return JSONResponse(
        status_code=202,
        content={"queued": True, "group_id": request.group_id, "query": request.query},
    )


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
