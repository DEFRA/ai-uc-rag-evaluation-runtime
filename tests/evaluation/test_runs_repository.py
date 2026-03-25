from typing import Any

from pytest_mock import MockerFixture

from app.evaluation import models, runs_repository


def _make_run(**kwargs: object) -> models.EvaluationRun:
    defaults: dict = {
        "run_id": "run-1",
        "status": "accepted",
        "group_id": "group-1",
        "truth_source_id": "truth-1",
        "snapshot_id": "snap-1",
        "models": ["model1"],
    }
    return models.EvaluationRun(**{**defaults, **kwargs})


def _make_result(**kwargs: object) -> models.EvaluationResult:
    defaults: dict = {
        "question": "q1",
        "expected_answer": "a1",
        "actual_answer": "actual",
        "model": "model1",
        "rubric": "rubric",
        "score": 0.9,
        "reason": "correct",
    }
    return models.EvaluationResult(**{**defaults, **kwargs})


async def _mock_collection(mocker: MockerFixture) -> Any:
    col = mocker.AsyncMock()
    mocker.patch(
        "app.evaluation.runs_repository._collection",
        new=mocker.AsyncMock(return_value=col),
    )
    return col


async def test_create_run_inserts_document(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    run = _make_run()

    await runs_repository.create_run(run)

    col.insert_one.assert_awaited_once_with(run.model_dump())


async def test_update_status_sets_status(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)

    await runs_repository.update_status("run-1", "in_progress")

    col.update_one.assert_awaited_once_with(
        {"run_id": "run-1"}, {"$set": {"status": "in_progress"}}
    )


async def test_append_result_pushes_result(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    result = _make_result()

    await runs_repository.append_result("run-1", result)

    col.update_one.assert_awaited_once_with(
        {"run_id": "run-1"}, {"$push": {"results": result.model_dump()}}
    )


async def test_save_summary_sets_summary(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    summary = [
        models.EvaluationSummary(
            model="model1", rubric="rubric", average_score=0.8, passed=True
        )
    ]

    await runs_repository.save_summary("run-1", summary)

    col.update_one.assert_awaited_once_with(
        {"run_id": "run-1"},
        {"$set": {"evaluation_summary": [s.model_dump() for s in summary]}},
    )


async def test_list_runs_returns_run_summaries(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    mock_cursor = mocker.MagicMock()
    mock_cursor.to_list = mocker.AsyncMock(
        return_value=[
            {"run_id": "run-1", "status": "completed"},
            {"run_id": "run-2", "status": "in_progress"},
        ]
    )
    col.find = mocker.MagicMock(return_value=mock_cursor)

    result = await runs_repository.list_runs()

    assert result == [
        {"run_id": "run-1", "status": "completed"},
        {"run_id": "run-2", "status": "in_progress"},
    ]
    col.find.assert_called_once_with({}, {"_id": 0, "run_id": 1, "status": 1})


async def test_get_run_returns_validated_run(mocker: MockerFixture) -> None:
    run = _make_run()
    col = await _mock_collection(mocker)
    col.find_one = mocker.AsyncMock(return_value=run.model_dump())

    result = await runs_repository.get_run("run-1")

    assert result == run
    col.find_one.assert_awaited_once_with({"run_id": "run-1"}, {"_id": 0})


async def test_get_run_returns_none_when_not_found(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    col.find_one = mocker.AsyncMock(return_value=None)

    result = await runs_repository.get_run("missing")

    assert result is None
