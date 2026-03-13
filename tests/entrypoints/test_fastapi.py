from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.entrypoints.fastapi import app
from app.evaluation import evaluation_service, runs_repository
from app.evaluation.models import EvaluationQuery, EvaluationRun

client = TestClient(app)


def test_lifespan(mocker: MockerFixture) -> None:
    mock_mongo_client = mocker.AsyncMock()
    mock_get_mongo = mocker.patch(
        "app.entrypoints.fastapi.get_mongo_client", return_value=mock_mongo_client
    )
    mocker.patch(
        "app.entrypoints.fastapi.listen",
        new=mocker.AsyncMock(return_value=None),
    )

    with client:
        pass

    mock_get_mongo.assert_awaited_once()
    mock_mongo_client.close.assert_awaited_once()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 404


def _make_run(**kwargs: object) -> EvaluationRun:
    defaults: dict = {
        "run_id": "test-run-id",
        "status": "accepted",
        "group_id": "g1",
        "queries": [
            EvaluationQuery(query="question 1", expected_answer="answer 1"),
            EvaluationQuery(query="question 2", expected_answer="answer 2"),
        ],
        "snapshot_id": "snap-1",
        "models": ["sonnet"],
    }
    return EvaluationRun(**{**defaults, **kwargs})


def test_queue_evaluation_success(mocker: MockerFixture) -> None:
    run = _make_run()
    mocker.patch.object(
        evaluation_service,
        "create_evaluation_run",
        new=mocker.AsyncMock(return_value=run),
    )

    response = client.post(
        "/evaluation",
        json={
            "group_id": "g1",
            "queries": [
                {"query": "question 1", "expected_answer": "answer 1"},
                {"query": "question 2", "expected_answer": "answer 2"},
            ],
            "models": ["sonnet"],
            "snapshot_id": "snap-1",
        },
    )

    assert response.status_code == 202
    assert response.json()["run_id"] == "test-run-id"
    assert response.json()["status"] == "accepted"
    evaluation_service.create_evaluation_run.assert_awaited_once()  # type: ignore[attr-defined]


def test_list_runs(mocker: MockerFixture) -> None:
    mocker.patch.object(
        runs_repository,
        "list_runs",
        new=mocker.AsyncMock(
            return_value=[
                {"run_id": "run-1", "status": "completed"},
                {"run_id": "run-2", "status": "in_progress"},
            ]
        ),
    )

    response = client.get("/evaluation")

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "run-1",
                "status": "completed",
                "results_url": "/evaluation/run-1",
            },
            {
                "run_id": "run-2",
                "status": "in_progress",
                "results_url": "/evaluation/run-2",
            },
        ]
    }


def test_get_run_results(mocker: MockerFixture) -> None:
    run = EvaluationRun(
        run_id="run-1",
        status="completed",
        group_id="g1",
        queries=[],
        snapshot_id="snapshot-id",
        models=["model1"],
    )
    mocker.patch.object(
        runs_repository,
        "get_run",
        new=mocker.AsyncMock(return_value=run),
    )

    response = client.get("/evaluation/run-1")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"
    assert response.json()["status"] == "completed"


def test_get_run_results_not_found(mocker: MockerFixture) -> None:
    mocker.patch.object(
        runs_repository,
        "get_run",
        new=mocker.AsyncMock(return_value=None),
    )

    response = client.get("/evaluation/unknown")

    assert response.status_code == 404


def test_queue_evaluation_not_configured(mocker: MockerFixture) -> None:
    mocker.patch.object(
        evaluation_service,
        "create_evaluation_run",
        new=mocker.AsyncMock(
            side_effect=ValueError("RAG_EVALUATION_START_QUEUE_URL is not configured")
        ),
    )

    response = client.post(
        "/evaluation",
        json={
            "group_id": "g1",
            "queries": [{"query": "q", "expected_answer": "a"}],
            "models": ["sonnet"],
            "snapshot_id": "snap-1",
        },
    )

    assert response.status_code == 503
    assert "RAG_EVALUATION_START_QUEUE_URL" in response.json()["detail"]
