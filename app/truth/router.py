import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.truth import dependencies, models, repository

router = APIRouter(tags=["truth"])

TruthRepo = Annotated[
    repository.AbstractTruthDataSourceRepository,
    Depends(dependencies.get_truth_repository),
]


class CreateTruthDataSourceRequest(BaseModel):
    dataset_id: str
    question_answers: list[models.QuestionAnswer]


class UpdateQuestionAnswersRequest(BaseModel):
    question_answers: list[models.QuestionAnswer]


@router.get("/truth-sources")
async def list_truth_sources(repo: TruthRepo) -> JSONResponse:
    """List all truth data sources with id, dataset_id and a link to full details."""
    sources = await repo.list_sources()
    return JSONResponse(
        content={
            "sources": [
                {**s.model_dump(), "url": f"/truth-sources/{s.id}"} for s in sources
            ]
        }
    )


@router.post("/truth-sources")
async def create_truth_source(
    request: CreateTruthDataSourceRequest,
    repo: TruthRepo,
) -> JSONResponse:
    """Create a new truth data source."""
    source = models.TruthDataSource(
        id=str(uuid.uuid4()),
        dataset_id=request.dataset_id,
        question_answers=request.question_answers,
    )
    await repo.create(source)
    return JSONResponse(status_code=201, content=source.model_dump())


@router.put("/truth-sources/{source_id}/question-answers")
async def update_question_answers(
    source_id: str,
    request: UpdateQuestionAnswersRequest,
    repo: TruthRepo,
) -> JSONResponse:
    """Replace the question/answer pairs for a truth data source."""
    found = await repo.update_question_answers(source_id, request.question_answers)
    if not found:
        return JSONResponse(
            status_code=404, content={"detail": "Truth data source not found"}
        )
    return JSONResponse(content={"id": source_id})


@router.get("/truth-sources/{source_id}")
async def get_truth_source(source_id: str, repo: TruthRepo) -> JSONResponse:
    """Retrieve a truth data source by ID."""
    source = await repo.get(source_id)
    if source is None:
        return JSONResponse(
            status_code=404, content={"detail": "Truth data source not found"}
        )
    return JSONResponse(content=source.model_dump())
