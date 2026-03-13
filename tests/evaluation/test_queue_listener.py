import asyncio
import contextlib
import json

from pytest_mock import MockerFixture

import app.evaluation.evaluation_service as evaluation_service
import app.evaluation.queue_listener as queue_listener
import app.evaluation.runs_repository as runs_repository


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

    message_body = {"run_id": "run-1"}
    mock_messages = [{"Body": json.dumps(message_body), "ReceiptHandle": "handle1"}]
    run_doc = {
        "run_id": "run-1",
        "group_id": "group1",
        "queries": [
            {"query": "question 1", "expected_answer": "answer 1"},
            {"query": "question 2", "expected_answer": "answer 2"},
        ],
        "snapshot_id": None,
        "rubrics": None,
        "models": None,
    }

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
    mocker.patch.object(
        runs_repository, "get_run", new=mocker.AsyncMock(return_value=run_doc)
    )
    mock_update_status = mocker.patch.object(
        runs_repository, "update_status", new=mocker.AsyncMock(return_value=None)
    )
    mock_append_result = mocker.patch.object(
        runs_repository, "append_result", new=mocker.AsyncMock(return_value=None)
    )
    mock_run = mocker.patch.object(
        evaluation_service,
        "run_rag_evaluation",
        new=mocker.AsyncMock(return_value=[{"score": 1.0}]),
    )
    mock_logger = mocker.patch("app.evaluation.queue_listener.logger")

    with contextlib.suppress(asyncio.CancelledError):
        await queue_listener.listen()

    mock_logger.info.assert_any_call("Received evaluation request: %s", "run-1")
    mock_update_status.assert_any_await("run-1", "in_progress")
    assert mock_run.await_count == 2
    mock_run.assert_any_await(
        "group1",
        "question 1",
        "answer 1",
        snapshot_id=None,
        rubrics=None,
        judge_model_keys=None,
    )
    mock_run.assert_any_await(
        "group1",
        "question 2",
        "answer 2",
        snapshot_id=None,
        rubrics=None,
        judge_model_keys=None,
    )
    assert mock_append_result.await_count == 2
    mock_append_result.assert_any_await("run-1", {"score": 1.0})
    mock_update_status.assert_any_await("run-1", "completed")
