from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import app.evaluation.evaluation_service as evaluation_service
import app.evaluation.rag_answer_service as rag_answer_service
import app.evaluation.runs_repository as runs_repository
import app.evaluation.sqs_service as sqs_service

router = APIRouter(tags=["evaluation"])


class EvaluationItem(BaseModel):
    query: str
    expected_answer: str


class QueueEvaluationRequest(BaseModel):
    group_id: str
    queries: list[EvaluationItem]
    snapshot_id: str | None = None
    rubrics: list[str] | None = None
    models: list[str] | None = None


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
    results = await evaluation_service.run_rag_evaluation(
        group_id, query, expected_answer, max_results
    )
    return JSONResponse(content=results[0])


@router.post("/evaluation/queue")
async def queue_evaluation(request: QueueEvaluationRequest) -> JSONResponse:
    """Enqueue a RAG evaluation request for background processing."""
    run_id = runs_repository.new_run_id()
    queries = [item.model_dump() for item in request.queries]
    await runs_repository.create_run(
        run_id,
        request.group_id,
        queries,
        request.snapshot_id,
        request.rubrics,
        request.models,
    )
    await sqs_service.enqueue_evaluation(
        run_id,
        request.group_id,
        queries,
        request.snapshot_id,
        request.rubrics,
        request.models,
    )
    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "status": "accepted"},
    )


@router.get("/evaluation/runs")
async def list_runs() -> JSONResponse:
    """List all evaluation runs with their run_id, status and a link to results."""
    runs = await runs_repository.list_runs()
    for run in runs:
        run["results_url"] = f"/evaluation/runs/{run['run_id']}/results"
    return JSONResponse(content={"runs": runs})


@router.get("/evaluation/runs/{run_id}/results")
async def get_run_results(run_id: str) -> JSONResponse:
    """Return the full results for a single evaluation run."""
    run = await runs_repository.get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "Run not found"})
    return JSONResponse(content=run)


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
