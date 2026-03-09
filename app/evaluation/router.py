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
) -> JSONResponse:
    """
    Trigger a RAG evaluation by querying the evaluation-data service snapshots endpoint.
    """
    try:
        content = await rag_service.query_snapshot(group_id, query, max_results)
    except EvaluationDataServiceNotConfiguredError:
        raise HTTPException(
            status_code=503,
            detail="evaluation_data_service_url is not configured",
        ) from None
    except EvaluationDataServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc

    return JSONResponse(content=content)


@router.get("/testing")
async def testing() -> JSONResponse:
    """
    Testing endpoint for the evaluation service.
    """
    knowledge = "Pretty Please is the first album released by Hector on Stilts in 2000.Hector on Stilts (HOS) is an American Indie pop/rock band. The band was originally formed in Tucson, Arizona, in 1998, and currently resides in Albany, New York."
    question = "What band originally from Tucson, Arizona and currently from Albany, New York released their first album in 2000?"
    answer = "Hector on Stilts is a band that started in Arizona and moved to New York before releasing their first album."

    result = await evaluation_service.evaluate_pydantic(
        knowledge, question, answer, evaluation_service.settings
    )

    return JSONResponse(content=result)
