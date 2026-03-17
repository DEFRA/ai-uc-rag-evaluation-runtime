from pydantic import BaseModel, Field


class EvaluationQuery(BaseModel):
    query: str
    expected_answer: str


class EvaluationResult(BaseModel):
    question: str
    expected_answer: str
    actual_answer: str
    model: str
    rubric: str
    score: float | None
    reason: str


class EvaluationSummary(BaseModel):
    model: str
    rubric: str
    average_score: float
    passed: bool


class EvaluationRun(BaseModel):
    run_id: str
    status: str
    group_id: str
    queries: list[EvaluationQuery]
    snapshot_id: str
    rubrics: list[str] | None = None
    models: list[str]
    results: list[EvaluationResult] = Field(default_factory=list)
    evaluation_summary: list[EvaluationSummary] = Field(default_factory=list)
