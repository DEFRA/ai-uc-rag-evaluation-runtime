from pytest_mock import MockerFixture

import app.evaluation.evaluation_service as evaluation_service


def _make_report(
    score: float | None, reason: str | None, mocker: MockerFixture
) -> object:
    score_entry = mocker.MagicMock()
    score_entry.value = score

    assertion = mocker.MagicMock()
    assertion.reason = reason

    case_result = mocker.MagicMock()
    case_result.scores = {"AnswerMatcher": score_entry} if score is not None else {}
    case_result.assertions = {"LLMJudge_pass": assertion} if reason is not None else {}

    report = mocker.MagicMock()
    report.cases = [case_result]
    return report


async def test_evaluate_pydantic_returns_pass_when_score_above_threshold(
    mocker: MockerFixture,
) -> None:
    report = _make_report(score=0.9, reason="Answer is correct.", mocker=mocker)
    mocker.patch(
        "app.evaluation.evaluation_service.pydantic_evals.Dataset.evaluate",
        new=mocker.AsyncMock(return_value=report),
    )

    result = await evaluation_service.evaluate_pydantic(
        "What is the capital of France?",
        "Paris",
        "The capital of France is Paris.",
    )

    assert result["method"] == "Pydantic"
    assert result["score"] == 0.9
    assert result["reason"] == "Answer is correct."
    assert result["passed"] is True


async def test_evaluate_pydantic_returns_fail_when_score_below_threshold(
    mocker: MockerFixture,
) -> None:
    report = _make_report(score=0.2, reason="Answer is mostly wrong.", mocker=mocker)
    mocker.patch(
        "app.evaluation.evaluation_service.pydantic_evals.Dataset.evaluate",
        new=mocker.AsyncMock(return_value=report),
    )

    result = await evaluation_service.evaluate_pydantic(
        "What is the capital of France?",
        "Paris",
        "The capital of France is Berlin.",
    )

    assert result["score"] == 0.2
    assert result["passed"] is False


async def test_evaluate_pydantic_handles_missing_score(
    mocker: MockerFixture,
) -> None:
    report = _make_report(score=None, reason=None, mocker=mocker)
    mocker.patch(
        "app.evaluation.evaluation_service.pydantic_evals.Dataset.evaluate",
        new=mocker.AsyncMock(return_value=report),
    )

    result = await evaluation_service.evaluate_pydantic(
        "question", "expected", "actual"
    )

    assert result["score"] is None
    assert result["reason"] == ""
    assert result["passed"] == -1
