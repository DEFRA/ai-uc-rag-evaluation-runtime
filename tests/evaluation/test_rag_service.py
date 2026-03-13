from typing import Any

import pytest
from pytest_mock import MockerFixture

import app.evaluation.rag_service as rag_service
from app.evaluation.exceptions import (
    EvaluationDataServiceError,
    EvaluationDataServiceNotConfiguredError,
)

_URL = "http://data-service"


def _patch_url(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.rag_service.config.config.evaluation_data_service_url", _URL
    )


def _make_mock_client(
    mocker: MockerFixture, *, status_code: int, json_body: object
) -> Any:
    mock_response = mocker.MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_body
    mock_response.content = True

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
    mock_client.post = mocker.AsyncMock(return_value=mock_response)

    mocker.patch(
        "app.evaluation.rag_service.http_client.create_async_client",
        return_value=mock_client,
    )
    return mock_client


async def test_query_snapshot_returns_documents(mocker: MockerFixture) -> None:
    _patch_url(mocker)
    documents = [{"content": "doc one"}, {"content": "doc two"}]
    mock_client = _make_mock_client(mocker, status_code=200, json_body=documents)

    result = await rag_service.query_snapshot("group-1", "what is x?", 5)

    assert result == documents
    mock_client.post.assert_awaited_once_with(
        f"{_URL}/snapshots/query",
        json={"groupId": "group-1", "query": "what is x?", "maxResults": 5},
    )


async def test_query_snapshot_includes_snapshot_id(mocker: MockerFixture) -> None:
    _patch_url(mocker)
    mock_client = _make_mock_client(mocker, status_code=200, json_body=[])

    await rag_service.query_snapshot("group-1", "query", 3, snapshot_id="snap-1")

    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["snapshotId"] == "snap-1"


async def test_query_snapshot_omits_snapshot_id_when_none(
    mocker: MockerFixture,
) -> None:
    _patch_url(mocker)
    mock_client = _make_mock_client(mocker, status_code=200, json_body=[])

    await rag_service.query_snapshot("group-1", "query", 3, snapshot_id=None)

    _, kwargs = mock_client.post.call_args
    assert "snapshotId" not in kwargs["json"]


async def test_query_snapshot_raises_when_not_configured(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.rag_service.config.config.evaluation_data_service_url", None
    )

    with pytest.raises(EvaluationDataServiceNotConfiguredError):
        await rag_service.query_snapshot("group-1", "query", 5)


async def test_query_snapshot_raises_on_error_response(mocker: MockerFixture) -> None:
    _patch_url(mocker)
    _make_mock_client(mocker, status_code=500, json_body={"error": "server error"})

    with pytest.raises(EvaluationDataServiceError) as exc_info:
        await rag_service.query_snapshot("group-1", "query", 5)

    assert exc_info.value.status_code == 500


async def test_query_snapshot_raises_when_response_is_not_list(
    mocker: MockerFixture,
) -> None:
    _patch_url(mocker)
    _make_mock_client(mocker, status_code=200, json_body={"unexpected": "dict"})

    with pytest.raises(
        EvaluationDataServiceError, match="Expected snapshots response to be a list"
    ):
        await rag_service.query_snapshot("group-1", "query", 5)
