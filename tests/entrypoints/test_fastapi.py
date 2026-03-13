from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

import app.evaluation.runs_repository as runs_repository
import app.evaluation.sqs_service as sqs_service
from app.entrypoints.fastapi import app
from app.evaluation.models import EvaluationRun

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

    # Using TestClient as a context manager triggers lifespan startup/shutdown
    with TestClient(app):
        mock_get_mongo.assert_called_once()  # Startup: connect called

    mock_mongo_client.close.assert_awaited_once()  # Shutdown: close called


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 404


def test_queue_evaluation_success(mocker: MockerFixture) -> None:
    mocker.patch.object(runs_repository, "new_run_id", return_value="test-run-id")
    mocker.patch.object(
        runs_repository, "create_run", new=mocker.AsyncMock(return_value=None)
    )
    mocker.patch.object(
        sqs_service, "enqueue_evaluation", new=mocker.AsyncMock(return_value=None)
    )

    queries = [
        {"query": "question 1", "expected_answer": "answer 1"},
        {"query": "question 2", "expected_answer": "answer 2"},
    ]

    response = client.post(
        "/evaluation",
        json={
            "group_id": "g1",
            "queries": queries,
            "models": ["sonnet"],
            "snapshot_id": "snap-1",
        },
    )

    assert response.status_code == 202
    assert response.json()["run_id"] == "test-run-id"
    assert response.json()["status"] == "accepted"
    call_args = runs_repository.create_run.call_args  # type: ignore[attr-defined]
    run = call_args.args[0]
    assert run.run_id == "test-run-id"
    assert run.group_id == "g1"
    assert [q.model_dump() for q in run.queries] == queries
    sqs_service.enqueue_evaluation.assert_awaited_once_with(  # type: ignore[attr-defined]
        "test-run-id"
    )


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

    response = client.get("/evaluation/runs/unknown/results")

    assert response.status_code == 404


def test_queue_evaluation_not_configured(mocker: MockerFixture) -> None:
    mocker.patch.object(runs_repository, "new_run_id", return_value="test-run-id")
    mocker.patch.object(
        runs_repository, "create_run", new=mocker.AsyncMock(return_value=None)
    )
    mocker.patch.object(
        sqs_service,
        "enqueue_evaluation",
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
