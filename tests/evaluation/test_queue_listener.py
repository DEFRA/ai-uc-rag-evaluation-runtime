import asyncio
import contextlib
import json

from pytest_mock import MockerFixture

import app.evaluation.judge_service as evaluation_service
import app.evaluation.queue_listener as queue_listener
import app.evaluation.rag_answer_service as rag_answer_service
import app.evaluation.runs_repository as runs_repository
from app.evaluation.models import EvaluationQuery, EvaluationResult, EvaluationRun


def _make_run(**kwargs: object) -> EvaluationRun:
    defaults: dict = {
        "run_id": "run-1",
        "status": "accepted",
        "group_id": "group1",
        "queries": [
            EvaluationQuery(query="question 1", expected_answer="answer 1"),
            EvaluationQuery(query="question 2", expected_answer="answer 2"),
        ],
        "snapshot_id": "snapshot-id",
        "rubrics": None,
        "models": ["model1"],
        "results": [],
    }
    return EvaluationRun(**{**defaults, **kwargs})


def _make_result(**kwargs: object) -> EvaluationResult:
    defaults: dict = {
        "question": "question 1",
        "expected_answer": "answer 1",
        "actual_answer": "the answer",
        "model": "model1",
        "rubric": evaluation_service.DEFAULT_RUBRIC,
        "score": 1.0,
        "reason": "correct",
    }
    return EvaluationResult(**{**defaults, **kwargs})


async def test_listen_processes_queries(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.rag_evaluation_start_queue_url",
        "http://localhost:4566/000000000000/rag_evaluation_start.fifo",
    )

    call_count = 0

    def fake_receive(_queue_url: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"Body": json.dumps({"run_id": "run-1"}), "ReceiptHandle": "handle1"}
        raise asyncio.CancelledError

    mocker.patch(
        "app.evaluation.queue_listener._receive_message", side_effect=fake_receive
    )
    mocker.patch("app.evaluation.queue_listener._delete_message")
    mocker.patch.object(
        runs_repository, "get_run", new=mocker.AsyncMock(return_value=_make_run())
    )
    mock_update_status = mocker.patch.object(
        runs_repository, "update_status", new=mocker.AsyncMock(return_value=None)
    )
    mock_append_result = mocker.patch.object(
        runs_repository, "append_result", new=mocker.AsyncMock(return_value=None)
    )
    mock_answer = mocker.patch.object(
        rag_answer_service,
        "answer_with_rag",
        new=mocker.AsyncMock(return_value="the answer"),
    )
    mock_evaluate = mocker.patch.object(
        evaluation_service,
        "evaluate_with_judge",
        new=mocker.AsyncMock(return_value=_make_result()),
    )

    with contextlib.suppress(asyncio.CancelledError):
        await queue_listener.listen()

    mock_update_status.assert_any_await("run-1", "in_progress")
    assert mock_answer.await_count == 2
    mock_answer.assert_any_await("question 1", "group1", snapshot_id="snapshot-id")
    mock_answer.assert_any_await("question 2", "group1", snapshot_id="snapshot-id")
    assert mock_evaluate.await_count == 2
    assert mock_append_result.await_count == 2
    mock_update_status.assert_any_await("run-1", "completed")


async def test_listen_skips_run_not_found_in_mongo(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.rag_evaluation_start_queue_url",
        "http://localhost:4566/000000000000/rag_evaluation_start.fifo",
    )

    call_count = 0

    def fake_receive(_queue_url: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "Body": json.dumps({"run_id": "missing-run"}),
                "ReceiptHandle": "h1",
            }
        raise asyncio.CancelledError

    mocker.patch(
        "app.evaluation.queue_listener._receive_message", side_effect=fake_receive
    )
    mock_delete = mocker.patch("app.evaluation.queue_listener._delete_message")
    mocker.patch.object(
        runs_repository, "get_run", new=mocker.AsyncMock(return_value=None)
    )
    mock_update_status = mocker.patch.object(
        runs_repository, "update_status", new=mocker.AsyncMock(return_value=None)
    )
    mock_answer = mocker.patch.object(
        rag_answer_service,
        "answer_with_rag",
        new=mocker.AsyncMock(return_value="the answer"),
    )

    with contextlib.suppress(asyncio.CancelledError):
        await queue_listener.listen()

    mock_answer.assert_not_awaited()
    mock_update_status.assert_not_awaited()
    mock_delete.assert_called_once()


async def test_listen_continues_after_processing_exception(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.rag_evaluation_start_queue_url",
        "http://localhost:4566/000000000000/rag_evaluation_start.fifo",
    )

    call_count = 0

    def fake_receive(_queue_url: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"Body": json.dumps({"run_id": "run-1"}), "ReceiptHandle": "h1"}
        raise asyncio.CancelledError

    mocker.patch(
        "app.evaluation.queue_listener._receive_message", side_effect=fake_receive
    )
    mocker.patch("app.evaluation.queue_listener._delete_message")
    mocker.patch.object(
        runs_repository,
        "get_run",
        new=mocker.AsyncMock(side_effect=RuntimeError("db connection failed")),
    )
    mocker.patch("app.evaluation.queue_listener.asyncio.sleep", new=mocker.AsyncMock())

    with contextlib.suppress(asyncio.CancelledError):
        await queue_listener.listen()

    # listener recovered and looped again (second receive raised CancelledError)
    assert call_count == 2


async def test_listen_skips_already_completed_combinations(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.rag_evaluation_start_queue_url",
        "http://localhost:4566/000000000000/rag_evaluation_start.fifo",
    )

    model_key = "model1"
    existing_result = _make_result(model=model_key)
    run = _make_run(
        queries=[EvaluationQuery(query="question 1", expected_answer="answer 1")],
        results=[existing_result],
    )

    call_count = 0

    def fake_receive(_queue_url: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"Body": json.dumps({"run_id": "run-1"}), "ReceiptHandle": "h1"}
        raise asyncio.CancelledError

    mocker.patch(
        "app.evaluation.queue_listener._receive_message", side_effect=fake_receive
    )
    mocker.patch("app.evaluation.queue_listener._delete_message")
    mocker.patch.object(
        runs_repository, "get_run", new=mocker.AsyncMock(return_value=run)
    )
    mocker.patch.object(
        runs_repository, "update_status", new=mocker.AsyncMock(return_value=None)
    )
    mocker.patch.object(
        runs_repository, "append_result", new=mocker.AsyncMock(return_value=None)
    )
    mocker.patch.object(
        rag_answer_service,
        "answer_with_rag",
        new=mocker.AsyncMock(return_value="the answer"),
    )
    mock_evaluate = mocker.patch.object(
        evaluation_service,
        "evaluate_with_judge",
        new=mocker.AsyncMock(return_value=_make_result()),
    )

    with contextlib.suppress(asyncio.CancelledError):
        await queue_listener.listen()

    mock_evaluate.assert_not_awaited()
