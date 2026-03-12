import asyncio
import contextlib
import json

from pytest_mock import MockerFixture

import app.evaluation.evaluation_service as evaluation_service
import app.evaluation.queue_listener as queue_listener


async def test_listen_disabled_when_not_configured(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.rag_evaluation_start_queue_url",
        None,
    )
    mock_warning = mocker.patch("app.evaluation.queue_listener.logger")

    await queue_listener.listen()

    mock_warning.warning.assert_called_once()


async def test_listen_logs_received_messages(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.rag_evaluation_start_queue_url",
        "http://localhost:4566/000000000000/rag_evaluation_start.fifo",
    )

    message_body = {
        "group_id": "group1",
        "query": "test query",
        "expected_answer": "expected",
    }
    mock_messages = [{"Body": json.dumps(message_body), "ReceiptHandle": "handle1"}]

    call_count = 0

    def fake_receive(_queue_url: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_messages[0]
        raise asyncio.CancelledError

    mocker.patch(
        "app.evaluation.queue_listener._receive",
        side_effect=fake_receive,
    )
    mocker.patch("app.evaluation.queue_listener._delete")
    mock_run = mocker.patch.object(
        evaluation_service,
        "run_rag_evaluation",
        new=mocker.AsyncMock(return_value={}),
    )
    mock_logger = mocker.patch("app.evaluation.queue_listener.logger")

    with contextlib.suppress(asyncio.CancelledError):
        await queue_listener.listen()

    mock_logger.info.assert_any_call("Received evaluation request: %s", message_body)
    mock_run.assert_awaited_once_with(
        message_body["group_id"],
        message_body["query"],
        message_body["expected_answer"],
        snapshot_id=None,
    )
