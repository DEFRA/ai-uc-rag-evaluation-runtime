from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

import app.evaluation.rag_answer_service as rag_answer_service
import app.evaluation.runs_repository as runs_repository
import app.evaluation.sqs_service as sqs_service
from app.entrypoints.fastapi import app

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


def test_rag_answer_success(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.rag_service.config.config.evaluation_data_service_url",
        "http://data-service.example/",
    )
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'[{"content": "Paris is the capital of France."}]'
    mock_response.json = mocker.MagicMock(
        return_value=[{"content": "Paris is the capital of France."}]
    )

    mock_client = mocker.MagicMock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "app.evaluation.rag_service.http_client.create_async_client",
        return_value=mock_client,
    )

    mock_result = mocker.MagicMock()
    mock_result.output = "The capital of France is Paris."
    mocker.patch.object(
        rag_answer_service._agent,
        "run",
        new=mocker.AsyncMock(return_value=mock_result),
    )

    response = client.get(
        "/rag/answer",
        params={
            "group_id": "g1",
            "query": "What is the capital of France?",
            "max_results": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "The capital of France is Paris."}
    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["json"] == {
        "groupId": "g1",
        "query": "What is the capital of France?",
        "maxResults": 3,
    }


def test_rag_answer_upstream_error(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.rag_service.config.config.evaluation_data_service_url",
        "http://data-service.example/",
    )
    mock_response = mocker.MagicMock()
    mock_response.status_code = 400
    mock_response.content = b'{"detail": "Knowledge group not found"}'
    mock_response.json = mocker.MagicMock(
        return_value={"detail": "Knowledge group not found"}
    )
    mock_response.text = "Knowledge group not found"

    mock_client = mocker.MagicMock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "app.evaluation.rag_service.http_client.create_async_client",
        return_value=mock_client,
    )

    response = client.get(
        "/rag/answer",
        params={"group_id": "g1", "query": "What is the capital of France?"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {"detail": "Knowledge group not found"}


def test_evaluation_rag_not_configured(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.rag_service.config.config.evaluation_data_service_url",
        None,
    )

    response = client.get(
        "/evaluation/rag",
        params={"group_id": "g1", "query": "test query", "expected_answer": "expected"},
    )

    assert response.status_code == 503
    assert "detail" in response.json()


def test_evaluation_rag_success(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.router.evaluation_service.run_rag_evaluation",
        new=mocker.AsyncMock(
            return_value=[
                {
                    "method": "Pydantic",
                    "score": 0.91,
                    "reason": "match",
                    "passed": True,
                }
            ]
        ),
    )

    response = client.get(
        "/evaluation/rag",
        params={
            "group_id": "g1",
            "query": "test query",
            "expected_answer": "expected answer",
            "max_results": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "method": "Pydantic",
        "score": 0.91,
        "reason": "match",
        "passed": True,
    }


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
        "/evaluation/queue",
        json={"group_id": "g1", "queries": queries},
    )

    assert response.status_code == 202
    assert response.json() == {"run_id": "test-run-id", "status": "accepted"}
    runs_repository.create_run.assert_awaited_once_with(  # type: ignore[attr-defined]
        "test-run-id", "g1", queries, None, None
    )
    sqs_service.enqueue_evaluation.assert_awaited_once_with(  # type: ignore[attr-defined]
        "test-run-id", "g1", queries, None, None
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

    response = client.get("/evaluation/runs")

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "run-1",
                "status": "completed",
                "results_url": "/evaluation/runs/run-1/results",
            },
            {
                "run_id": "run-2",
                "status": "in_progress",
                "results_url": "/evaluation/runs/run-2/results",
            },
        ]
    }


def test_get_run_results(mocker: MockerFixture) -> None:
    mocker.patch.object(
        runs_repository,
        "get_run",
        new=mocker.AsyncMock(
            return_value={
                "run_id": "run-1",
                "status": "completed",
                "group_id": "g1",
                "result": {"results": [{"score": 0.9}]},
            }
        ),
    )

    response = client.get("/evaluation/runs/run-1/results")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"
    assert response.json()["result"]["results"] == [{"score": 0.9}]


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
        "/evaluation/queue",
        json={
            "group_id": "g1",
            "queries": [{"query": "q", "expected_answer": "a"}],
        },
    )

    assert response.status_code == 503
    assert "RAG_EVALUATION_START_QUEUE_URL" in response.json()["detail"]


def test_evaluation_rag_upstream_error(mocker: MockerFixture) -> None:
    from app.evaluation.exceptions import EvaluationDataServiceError

    mocker.patch(
        "app.evaluation.router.evaluation_service.run_rag_evaluation",
        new=mocker.AsyncMock(
            side_effect=EvaluationDataServiceError(
                400, {"detail": "Knowledge group not found"}
            )
        ),
    )

    response = client.get(
        "/evaluation/rag",
        params={
            "group_id": "g1",
            "query": "test query",
            "expected_answer": "expected answer",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {"detail": "Knowledge group not found"}
