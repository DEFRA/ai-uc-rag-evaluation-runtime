from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

settings: dict[str, Any] = {}
settings["n_prompts"] = 10
settings["region"] = "eu-west-2"
settings["temperature"] = 0
settings["threshold"] = 0.5
settings["model_id"] = "anthropic.claude-3-haiku-20240307-v1:0"
settings["model_name"] = "Claude 3 Haiku"
settings["rubric"] = """
You are a huallucination detector.
You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge.
Return a score between 0 and 1 reflecting the likelihood that the answer is a hallucination where 0 is very unlikely and 1 is very likely.
"""

_judge = LLMJudge(
    model=f"bedrock:{settings['model_id']}",
    rubric=settings["rubric"],
    score={"evaluation_name": "HallucinationScore"},
    model_settings={
        "temperature": settings["temperature"],
        "max_tokens": 2048,
    },
    include_input=True,
    include_expected_output=True,
)


async def evaluate_pydantic(
    knowledge: str, question: str, answer: str, settings: dict
) -> dict:
    dataset = Dataset(
        cases=[Case(inputs=question, expected_output=knowledge)],
        evaluators=[_judge],
    )

    try:
        report = await dataset.evaluate(lambda _: answer)
    except Exception as exc:
        return {
            "method": "Pydantic",
            "score": None,
            "reason": str(exc),
            "passed": False,
        }

    case_result = report.cases[0]

    score_entry = case_result.scores.get("HallucinationScore")

    score = score_entry.value if score_entry else None
    reason = case_result.assertions.get("LLMJudge_pass")
    return {
        "method": "Pydantic",
        "score": score,
        "reason": reason.reason if reason else "",
        "passed": score >= settings["threshold"],
    }
