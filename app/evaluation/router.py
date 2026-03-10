from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

import app.evaluation.evaluation_service as evaluation_service
import app.evaluation.rag_service as rag_service
from app.evaluation.exceptions import (
    EvaluationDataServiceError,
    EvaluationDataServiceNotConfiguredError,
)

router = APIRouter(tags=["evaluation"])


@router.get("/evaluation/rag")
async def trigger_rag_evaluation(
    group_id: str = Query(..., description="Knowledge group ID"),
    query: str = Query(..., description="Search query"),
    max_results: int = Query(5, ge=1, le=100, description="Maximum number of results"),
    expected_answer: str = Query(..., description="Expected answer"),
) -> JSONResponse:
    """
    Trigger a RAG evaluation by querying the evaluation-data service snapshots endpoint.
    """
    content = await rag_service.query_snapshot(group_id, query, max_results)

    evaluated_content = []
    for item in content:
        answer = item.get("content")
        evaluation = await evaluation_service.evaluate_pydantic(
            "",
            query,
            expected_answer,
            answer,
        )
        evaluated_content.append(evaluation)

    return JSONResponse(content=evaluated_content)
