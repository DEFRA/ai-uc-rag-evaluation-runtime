from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.truth import dependencies, models, repository
from app.truth import router as truth_router

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


def _make_client(
    repo: repository.AbstractTruthDataSourceRepository,
) -> TestClient:
    app = FastAPI()
    app.include_router(truth_router.router)
    app.dependency_overrides[dependencies.get_truth_repository] = lambda: repo
    return TestClient(app)


def test_list_truth_sources(mocker: MockerFixture) -> None:
    repo = mocker.AsyncMock(spec=repository.AbstractTruthDataSourceRepository)
    repo.list_sources.return_value = [
        models.TruthDataSourceSummary(id="source-1", dataset_id="dataset-1"),
        models.TruthDataSourceSummary(id="source-2", dataset_id="dataset-2"),
    ]
    client = _make_client(repo)

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
    repo = mocker.AsyncMock(spec=repository.AbstractTruthDataSourceRepository)
    client = _make_client(repo)

    response = client.post(
        "/truth-sources",
        json={"dataset_id": _DATASET_ID, "question_answers": _QAS},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["dataset_id"] == _DATASET_ID
    assert data["question_answers"] == _QAS
    assert data["id"]
    repo.create.assert_awaited_once()


def test_update_question_answers_success(mocker: MockerFixture) -> None:
    repo = mocker.AsyncMock(spec=repository.AbstractTruthDataSourceRepository)
    repo.update_question_answers.return_value = True
    client = _make_client(repo)

    response = client.put(
        f"/truth-sources/{_SOURCE_ID}/question-answers",
        json={"question_answers": _QAS},
    )

    assert response.status_code == 200
    assert response.json()["id"] == _SOURCE_ID
    repo.update_question_answers.assert_awaited_once_with(
        _SOURCE_ID, [models.QuestionAnswer(question="q1", answer="a1")]
    )


def test_update_question_answers_not_found(mocker: MockerFixture) -> None:
    repo = mocker.AsyncMock(spec=repository.AbstractTruthDataSourceRepository)
    repo.update_question_answers.return_value = False
    client = _make_client(repo)

    response = client.put(
        f"/truth-sources/{_SOURCE_ID}/question-answers",
        json={"question_answers": _QAS},
    )

    assert response.status_code == 404


def test_get_truth_source_success(mocker: MockerFixture) -> None:
    source = _make_source()
    repo = mocker.AsyncMock(spec=repository.AbstractTruthDataSourceRepository)
    repo.get.return_value = source
    client = _make_client(repo)

    response = client.get(f"/truth-sources/{_SOURCE_ID}")

    assert response.status_code == 200
    assert response.json() == source.model_dump()
    repo.get.assert_awaited_once_with(_SOURCE_ID)


def test_get_truth_source_not_found(mocker: MockerFixture) -> None:
    repo = mocker.AsyncMock(spec=repository.AbstractTruthDataSourceRepository)
    repo.get.return_value = None
    client = _make_client(repo)

    response = client.get("/truth-sources/unknown")

    assert response.status_code == 404
