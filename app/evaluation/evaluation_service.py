import pydantic_evals
import pydantic_evals.evaluators
import app.config as config


rubric = """
You are an agent to determine the correctness of the answer privided compared to the expected answer.
Return a score between 0 and 1 reflecting the corectness of the answer 0 is not very correct and 1 is very correct.
"""

judge_config = config.config.llm_as_a_judge_config
_judge = pydantic_evals.evaluators.LLMJudge(
    model=f"bedrock:{judge_config.model_id}",
    rubric=rubric,
    score={"evaluation_name": "AnswerMatcher"},
    model_settings={
        "temperature": judge_config.temperature,
        "max_tokens": 2048,
    },
    include_input=True,
    include_expected_output=True,
)


async def evaluate_pydantic(
    knowledge: str, question: str, expected_answer: str, actual_answer: str
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
        "passed": score >= judge_config.threshold,
    }
