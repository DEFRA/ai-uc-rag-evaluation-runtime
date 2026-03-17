from logging import getLogger

import pydantic_evals
import pydantic_evals.evaluators
from pydantic_ai import models
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider

from app import config
from app.evaluation.models import EvaluationResult

logger = getLogger(__name__)

DEFAULT_RUBRIC = """
You are evaluating whether an answer correctly and completely addresses a question based on an expected answer.

Score the answer from 0 to 1 using these criteria:
- 1.0: Fully correct and complete — captures all key information from the expected answer
- 0.75: Mostly correct — captures the main points but misses minor details
- 0.5: Partially correct — contains some accurate information but is missing significant content or contains inaccuracies
- 0.25: Mostly incorrect — touches on the topic but is largely wrong or incomplete
- 0.0: Completely incorrect or irrelevant to the question

Focus on factual accuracy and completeness relative to the expected answer. Ignore differences in phrasing or style.
"""

_judge_config = config.config.llm_as_a_judge_config
_provider = BedrockProvider(region_name=config.config.aws_region)
_settings = (
    BedrockModelSettings(
        bedrock_guardrail_config={
            "guardrailIdentifier": _judge_config.guardrails_id,
            "guardrailVersion": _judge_config.guardrails_version,
            "trace": "enabled",
        }
    )
    if _judge_config.guardrails_id
    else None
)


def _build_configured_models() -> dict[str, models.Model]:
    return {
        key: BedrockConverseModel(
            entry["arn"] or entry["model_id"],
            provider=_provider,
            profile=_provider.model_profile(entry["model_id"]),
            settings=_settings,
        )
        for key, entry in config.config.llm_as_a_judge_config.models.items()
    }


_models = _build_configured_models()


def _make_judge(
    rubric: str, judge_model: models.Model
) -> pydantic_evals.evaluators.LLMJudge:
    return pydantic_evals.evaluators.LLMJudge(
        model=judge_model,
        rubric=rubric,
        score={"evaluation_name": "AnswerMatcher"},
        model_settings={
            "temperature": _judge_config.temperature,
            "max_tokens": _judge_config.max_tokens,
        },
        include_input=True,
        include_expected_output=True,
    )


async def evaluate_with_judge(
    question: str,
    expected_answer: str,
    actual_answer: str,
    model_key: str,
    rubric: str,
) -> EvaluationResult:
    if model_key not in _models:
        msg = f"No model configured for '{model_key}'"
        raise ValueError(msg)
    model = _models[model_key]
    judge = _make_judge(rubric, model)
    dataset = pydantic_evals.Dataset(
        cases=[pydantic_evals.Case(inputs=question, expected_output=expected_answer)],
        evaluators=[judge],
    )

    report = await dataset.evaluate(lambda _: actual_answer)

    case_result = report.cases[0]

    score_entry = case_result.scores.get("AnswerMatcher")

    logger.info("AnswerMatcher: %s", score_entry)

    score = score_entry.value if score_entry else None
    reason = case_result.assertions.get("LLMJudge_pass")
    return EvaluationResult(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        model=model_key,
        rubric=rubric,
        score=score,
        reason=reason.reason if reason else "",
    )
