from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.truth import models, repository
from app.truth.router import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)

_SOURCE_ID = "source-1"
_DATASET_ID = "dataset-1"
_QAS = [{"question": "q1", "answer": "a1"}]


def _make_source(**kwargs: object) -> models.TruthDataSource:
    defaults: dict = {
        "id": _SOURCE_ID,
        "dataset_id": _DATASET_ID,
        "question_answers": [models.QuestionAnswer(question="q1", answer="a1")],
    }
    return models.TruthDataSource(**{**defaults, **kwargs})


def test_list_truth_sources(mocker: MockerFixture) -> None:
    mocker.patch.object(
        repository,
        "list_sources",
        new=mocker.AsyncMock(
            return_value=[
                models.TruthDataSourceSummary(id="source-1", dataset_id="dataset-1"),
                models.TruthDataSourceSummary(id="source-2", dataset_id="dataset-2"),
            ]
        ),
    )

    response = client.get("/truth-sources")

    assert response.status_code == 200
    assert response.json() == {
        "sources": [
            {
                "id": "source-1",
                "dataset_id": "dataset-1",
                "url": "/truth-sources/source-1",
            },
            {
                "id": "source-2",
                "dataset_id": "dataset-2",
                "url": "/truth-sources/source-2",
            },
        ]
    }


def test_create_truth_source(mocker: MockerFixture) -> None:
    mocker.patch.object(repository, "create", new=mocker.AsyncMock())

    response = client.post(
        "/truth-sources",
        json={"dataset_id": _DATASET_ID, "question_answers": _QAS},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["dataset_id"] == _DATASET_ID
    assert data["question_answers"] == _QAS
    assert data["id"]
    repository.create.assert_awaited_once()  # type: ignore[attr-defined]


def test_update_question_answers_success(mocker: MockerFixture) -> None:
    mocker.patch.object(
        repository, "update_question_answers", new=mocker.AsyncMock(return_value=True)
    )

    response = client.put(
        f"/truth-sources/{_SOURCE_ID}/question-answers",
        json={"question_answers": _QAS},
    )

    assert response.status_code == 200
    assert response.json()["id"] == _SOURCE_ID
    repository.update_question_answers.assert_awaited_once_with(  # type: ignore[attr-defined]
        _SOURCE_ID, [models.QuestionAnswer(question="q1", answer="a1")]
    )


def test_update_question_answers_not_found(mocker: MockerFixture) -> None:
    mocker.patch.object(
        repository, "update_question_answers", new=mocker.AsyncMock(return_value=False)
    )

    response = client.put(
        f"/truth-sources/{_SOURCE_ID}/question-answers",
        json={"question_answers": _QAS},
    )

    assert response.status_code == 404


def test_get_truth_source_success(mocker: MockerFixture) -> None:
    source = _make_source()
    mocker.patch.object(repository, "get", new=mocker.AsyncMock(return_value=source))

    response = client.get(f"/truth-sources/{_SOURCE_ID}")

    assert response.status_code == 200
    assert response.json() == source.model_dump()
    repository.get.assert_awaited_once_with(_SOURCE_ID)  # type: ignore[attr-defined]


def test_get_truth_source_not_found(mocker: MockerFixture) -> None:
    mocker.patch.object(repository, "get", new=mocker.AsyncMock(return_value=None))

    response = client.get("/truth-sources/unknown")

    assert response.status_code == 404
