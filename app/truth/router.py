import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.truth import models, repository

router = APIRouter(tags=["truth"])


class CreateTruthDataSourceRequest(BaseModel):
    dataset_id: str
    question_answers: list[models.QuestionAnswer]


class UpdateQuestionAnswersRequest(BaseModel):
    question_answers: list[models.QuestionAnswer]


@router.get("/truth-sources")
async def list_truth_sources() -> JSONResponse:
    """List all truth data sources with id, dataset_id and a link to full details."""
    sources = await repository.list_sources()
    return JSONResponse(
        content={
            "sources": [
                {**s.model_dump(), "url": f"/truth-sources/{s.id}"} for s in sources
            ]
        }
    )


@router.post("/truth-sources")
async def create_truth_source(request: CreateTruthDataSourceRequest) -> JSONResponse:
    """Create a new truth data source."""
    source = models.TruthDataSource(
        id=str(uuid.uuid4()),
        dataset_id=request.dataset_id,
        question_answers=request.question_answers,
    )
    await repository.create(source)
    return JSONResponse(status_code=201, content=source.model_dump())


@router.put("/truth-sources/{source_id}/question-answers")
async def update_question_answers(
    source_id: str, request: UpdateQuestionAnswersRequest
) -> JSONResponse:
    """Replace the question/answer pairs for a truth data source."""
    found = await repository.update_question_answers(
        source_id, request.question_answers
    )
    if not found:
        return JSONResponse(
            status_code=404, content={"detail": "Truth data source not found"}
        )
    return JSONResponse(content={"id": source_id})


@router.get("/truth-sources/{source_id}")
async def get_truth_source(source_id: str) -> JSONResponse:
    """Retrieve a truth data source by ID."""
    source = await repository.get(source_id)
    if source is None:
        return JSONResponse(
            status_code=404, content={"detail": "Truth data source not found"}
        )
    return JSONResponse(content=source.model_dump())
