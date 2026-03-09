from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.entrypoints.fastapi import app

client = TestClient(app)


def test_lifespan(mocker: MockerFixture) -> None:
    mock_mongo_client = mocker.AsyncMock()
    mock_get_mongo = mocker.patch(
        "app.entrypoints.fastapi.get_mongo_client", return_value=mock_mongo_client
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


def test_trigger_rag_evaluation_not_configured(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.rag_service.config.config.evaluation_data_service_url",
        None,
    )
    response = client.get(
        "/evaluation/rag",
        params={"group_id": "g1", "query": "test query"},
    )
    assert response.status_code == 503
    assert "detail" in response.json()


def test_trigger_rag_evaluation_success(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.rag_service.config.config.evaluation_data_service_url",
        "http://data-service.example/",
    )
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'[{"content": "doc1", "similarityScore": 0.9}]'
    mock_response.json = mocker.MagicMock(
        return_value=[{"content": "doc1", "similarityScore": 0.9}]
    )

    mock_client = mocker.MagicMock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "app.evaluation.rag_service.http_client.create_async_client",
        return_value=mock_client,
    )

    response = client.get(
        "/evaluation/rag",
        params={"group_id": "g1", "query": "test query", "max_results": 3},
    )

    assert response.status_code == 200
    assert response.json() == [{"content": "doc1", "similarityScore": 0.9}]
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["json"] == {
        "groupId": "g1",
        "query": "test query",
        "maxResults": 3,
    }


def test_trigger_rag_evaluation_upstream_error(mocker: MockerFixture) -> None:
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
        "/evaluation/rag",
        params={"group_id": "g1", "query": "test query"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {"detail": "Knowledge group not found"}
