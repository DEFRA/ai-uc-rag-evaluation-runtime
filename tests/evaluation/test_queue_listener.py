import asyncio
import contextlib
import json

from pytest_mock import MockerFixture

from app.evaluation import judge_service as evaluation_service
from app.evaluation import models, queue_listener, rag_answer_service, runs_repository
from app.truth import repository as truth_repository
from app.truth.models import QuestionAnswer, TruthDataSource


def _make_truth_source(**kwargs: object) -> TruthDataSource:
    defaults: dict = {
        "id": "truth-1",
        "dataset_id": "group1",
        "question_answers": [
            QuestionAnswer(question="question 1", answer="answer 1"),
            QuestionAnswer(question="question 2", answer="answer 2"),
        ],
    }
    return TruthDataSource(**{**defaults, **kwargs})


def _make_run(**kwargs: object) -> models.EvaluationRun:
    defaults: dict = {
        "run_id": "run-1",
        "status": "accepted",
        "group_id": "group1",
        "truth_source_id": "truth-1",
        "snapshot_id": "snapshot-id",
        "rubrics": None,
        "models": ["model1"],
        "results": [],
    }
    return models.EvaluationRun(**{**defaults, **kwargs})


def _make_result(**kwargs: object) -> models.EvaluationResult:
    defaults: dict = {
        "question": "question 1",
        "expected_answer": "answer 1",
        "actual_answer": "the answer",
        "model": "model1",
        "rubric": evaluation_service.DEFAULT_RUBRIC,
        "score": 1.0,
        "reason": "correct",
    }
    return models.EvaluationResult(**{**defaults, **kwargs})


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
    mocker.patch.object(
        truth_repository, "get", new=mocker.AsyncMock(return_value=_make_truth_source())
    )
    mock_update_status = mocker.patch.object(
        runs_repository, "update_status", new=mocker.AsyncMock(return_value=None)
    )
    mock_append_result = mocker.patch.object(
        runs_repository, "append_result", new=mocker.AsyncMock(return_value=None)
    )
    mocker.patch.object(runs_repository, "save_summary", new=mocker.AsyncMock())
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


async def test_listen_skips_when_truth_source_not_found(mocker: MockerFixture) -> None:
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
        runs_repository, "get_run", new=mocker.AsyncMock(return_value=_make_run())
    )
    mocker.patch.object(
        truth_repository, "get", new=mocker.AsyncMock(return_value=None)
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
    mock_update_status.assert_any_await("run-1", "failed")


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
    run = _make_run(results=[existing_result])
    truth_source = _make_truth_source(
        question_answers=[QuestionAnswer(question="question 1", answer="answer 1")]
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
        truth_repository, "get", new=mocker.AsyncMock(return_value=truth_source)
    )
    mocker.patch.object(
        runs_repository, "update_status", new=mocker.AsyncMock(return_value=None)
    )
    mocker.patch.object(
        runs_repository, "append_result", new=mocker.AsyncMock(return_value=None)
    )
    mocker.patch.object(runs_repository, "save_summary", new=mocker.AsyncMock())
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


def test_summarise_run_groups_by_model_and_rubric(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.llm_as_a_judge_config.score_threshold",
        0.5,
    )
    results = [
        _make_result(model="model1", rubric="rubric1", score=0.8),
        _make_result(model="model1", rubric="rubric1", score=0.6),
        _make_result(model="model2", rubric="rubric1", score=0.4),
    ]

    summary = queue_listener._summarise_run(results)

    assert len(summary) == 2
    m1 = next(s for s in summary if s.model == "model1")
    assert m1.average_score == 0.7
    assert m1.passed is True
    m2 = next(s for s in summary if s.model == "model2")
    assert m2.average_score == 0.4
    assert m2.passed is False


def test_summarise_run_ignores_none_scores(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.queue_listener.config.config.llm_as_a_judge_config.score_threshold",
        0.5,
    )
    results = [
        _make_result(model="model1", rubric="rubric1", score=None),
        _make_result(model="model1", rubric="rubric1", score=1.0),
    ]

    summary = queue_listener._summarise_run(results)

    assert len(summary) == 1
    assert summary[0].average_score == 1.0
