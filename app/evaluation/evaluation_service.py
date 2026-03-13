from logging import getLogger

import pydantic_evals
import pydantic_evals.evaluators
from pydantic_ai import models
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider

import app.config as config
import app.evaluation.rag_answer_service as rag_answer_service

logger = getLogger(__name__)

_rubric = """
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

_model: models.Model = BedrockConverseModel(
    _judge_config.inference_profile_arn or _judge_config.model_id,
    provider=_provider,
    profile=_provider.model_profile(_judge_config.model_id),
    settings=_settings,
)


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


def _build_model_from_key(key: str) -> models.Model:
    entry = config.config.llm_as_a_judge_config.models.get(key)
    if not entry:
        msg = f"Model key '{key}' not found in LlmConfig.models"
        raise ValueError(msg)
    return BedrockConverseModel(
        entry["arn"] or entry["model_id"],
        provider=_provider,
        profile=_provider.model_profile(entry["model_id"]),
        settings=_settings,
    )


async def run_rag_evaluation(
    group_id: str,
    query: str,
    expected_answer: str,
    max_results: int = 5,
    snapshot_id: str | None = None,
    rubrics: list[str] | None = None,
    judge_model_keys: list[str] | None = None,
) -> list[dict]:
    answer = await rag_answer_service.answer_with_rag(
        query, group_id, max_results, snapshot_id
    )
    effective_rubrics: list[str] = rubrics if rubrics is not None else [_rubric]
    model_entries: list[tuple[str, models.Model]] = (
        [(key, _build_model_from_key(key)) for key in judge_model_keys]
        if judge_model_keys
        else [(_judge_config.model_id, _model)]
    )
    results = [
        await evaluate_pydantic(
            query, expected_answer, answer, rubric, model_key, judge_model
        )
        for rubric in effective_rubrics
        for model_key, judge_model in model_entries
    ]
    logger.info(
        "RAG evaluation complete for group %s: %d result(s)", group_id, len(results)
    )
    return results


async def evaluate_pydantic(
    question: str,
    expected_answer: str,
    actual_answer: str,
    rubric: str | None = None,
    model_key: str | None = None,
    judge_model: models.Model | None = None,
) -> dict:
    effective_rubric = rubric if rubric is not None else _rubric
    effective_model = judge_model if judge_model is not None else _model
    judge = _make_judge(effective_rubric, effective_model)
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
    return {
        "method": "Pydantic",
        "model": model_key if model_key is not None else _judge_config.model_id,
        "rubric": effective_rubric,
        "score": score,
        "reason": reason.reason if reason else "",
        "passed": score >= _judge_config.threshold if score else -1,
    }
