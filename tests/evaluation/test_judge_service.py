import pytest
from pytest_mock import MockerFixture

from app.evaluation import judge_service

_MODEL_KEY = "test-model"


def _patch_models(mocker: MockerFixture) -> None:
    mocker.patch.dict(judge_service._models, {_MODEL_KEY: mocker.MagicMock()})


def _make_mock_dataset(
    mocker: MockerFixture, *, scores: dict, assertions: dict
) -> object:
    mock_case_result = mocker.MagicMock()
    mock_case_result.scores = scores
    mock_case_result.assertions = assertions

    mock_report = mocker.MagicMock()
    mock_report.cases = [mock_case_result]

    mock_dataset = mocker.MagicMock()
    mock_dataset.evaluate = mocker.AsyncMock(return_value=mock_report)

    mocker.patch(
        "app.evaluation.judge_service.pydantic_evals.Dataset", return_value=mock_dataset
    )
    return mock_dataset


async def test_evaluate_with_judge_returns_result(mocker: MockerFixture) -> None:
    _patch_models(mocker)

    mock_score = mocker.MagicMock()
    mock_score.value = 0.75
    mock_assertion = mocker.MagicMock()
    mock_assertion.reason = "mostly correct"

    _make_mock_dataset(
        mocker,
        scores={"AnswerMatcher": mock_score},
        assertions={"LLMJudge_pass": mock_assertion},
    )

    result = await judge_service.evaluate_with_judge(
        question="What is X?",
        expected_answer="X is Y",
        actual_answer="X is Y indeed",
        model_key=_MODEL_KEY,
        rubric=judge_service.DEFAULT_RUBRIC,
    )

    assert result.question == "What is X?"
    assert result.expected_answer == "X is Y"
    assert result.actual_answer == "X is Y indeed"
    assert result.model == _MODEL_KEY
    assert result.score == 0.75
    assert result.reason == "mostly correct"


async def test_evaluate_with_judge_raises_for_unknown_model() -> None:
    with pytest.raises(ValueError, match="No model configured for 'unknown'"):
        await judge_service.evaluate_with_judge(
            question="q",
            expected_answer="a",
            actual_answer="b",
            model_key="unknown",
            rubric=judge_service.DEFAULT_RUBRIC,
        )


async def test_evaluate_with_judge_handles_missing_score(mocker: MockerFixture) -> None:
    _patch_models(mocker)

    mock_assertion = mocker.MagicMock()
    mock_assertion.reason = "no score"

    _make_mock_dataset(
        mocker,
        scores={},
        assertions={"LLMJudge_pass": mock_assertion},
    )

    result = await judge_service.evaluate_with_judge(
        question="q",
        expected_answer="a",
        actual_answer="b",
        model_key=_MODEL_KEY,
        rubric=judge_service.DEFAULT_RUBRIC,
    )

    assert result.score is None


async def test_evaluate_with_judge_handles_missing_assertion(
    mocker: MockerFixture,
) -> None:
    _patch_models(mocker)

    mock_score = mocker.MagicMock()
    mock_score.value = 1.0

    _make_mock_dataset(
        mocker,
        scores={"AnswerMatcher": mock_score},
        assertions={},
    )

    result = await judge_service.evaluate_with_judge(
        question="q",
        expected_answer="a",
        actual_answer="b",
        model_key=_MODEL_KEY,
        rubric=judge_service.DEFAULT_RUBRIC,
    )

    assert result.reason == ""
