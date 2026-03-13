from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.evaluation import evaluation_service, runs_repository
from app.evaluation.models import EvaluationQuery

router = APIRouter(tags=["evaluation"])


class QueueEvaluationRequest(BaseModel):
    group_id: str
    queries: list[EvaluationQuery]
    snapshot_id: str
    rubrics: list[str] | None = None
    models: list[str]


@router.post("/evaluation")
async def queue_evaluation(request: QueueEvaluationRequest) -> JSONResponse:
    """Enqueue a RAG evaluation request for background processing."""
    run = await evaluation_service.create_evaluation_run(
        request.group_id,
        request.queries,
        request.snapshot_id,
        request.rubrics,
        request.models,
    )
    return JSONResponse(status_code=202, content=run.model_dump())


@router.get("/evaluation")
async def list_runs() -> JSONResponse:
    """List all evaluation runs with their run_id, status and a link to results."""
    runs = await runs_repository.list_runs()
    for run in runs:
        run["results_url"] = f"/evaluation/{run['run_id']}"
    return JSONResponse(content={"runs": runs})


@router.get("/evaluation/{run_id}")
async def get_run_results(run_id: str) -> JSONResponse:
    """Return the full results for a single evaluation run."""
    run = await runs_repository.get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "Run not found"})
    return JSONResponse(content=run.model_dump())
