import pydantic_evals
import pydantic_evals.evaluators
from pydantic_ai import models
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

import app.config as config

_rubric = """
You are an agent to determine the correctness of the answer provided compared to the expected answer.
Return a score between 0 and 1 reflecting the correctness of the answer 0 is not very correct and 1 is very correct.
"""

_judge_config = config.config.llm_as_a_judge_config
_model: models.Model | str
if _judge_config.inference_profile_arn:
    _provider = BedrockProvider(region_name=config.config.aws_region)
    # _judge_config.model_id
    _profile = _provider.model_profile(_judge_config.model_id)

    _model = BedrockConverseModel(
        _judge_config.inference_profile_arn,
        provider=_provider,
        profile=_profile,
    )
else:
    _model = f"bedrock:{_judge_config.model_id}"

_judge = pydantic_evals.evaluators.LLMJudge(
    model=_model,
    rubric=_rubric,
    score={"evaluation_name": "AnswerMatcher"},
    model_settings={
        "temperature": _judge_config.temperature,
        "max_tokens": _judge_config.max_tokens,
    },
    include_input=True,
    include_expected_output=True,
)


async def evaluate_pydantic(
    question: str, expected_answer: str, actual_answer: str
) -> dict:
    dataset = pydantic_evals.Dataset(
        cases=[pydantic_evals.Case(inputs=question, expected_output=expected_answer)],
        evaluators=[_judge],
    )

    report = await dataset.evaluate(lambda _: actual_answer)

    case_result = report.cases[0]

    score_entry = case_result.scores.get("AnswerMatcher")

    score = score_entry.value if score_entry else None
    reason = case_result.assertions.get("LLMJudge_pass")
    return {
        "method": "Pydantic",
        "score": score,
        "reason": reason.reason if reason else "",
        "passed": score >= _judge_config.threshold if score else -1,
    }
