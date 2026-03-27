from pydantic import BaseModel


class QuestionAnswer(BaseModel):
    question: str
    answer: str


class TruthDataSourceSummary(BaseModel):
    id: str
    dataset_id: str


class TruthDataSource(BaseModel):
    id: str
    dataset_id: str
    question_answers: list[QuestionAnswer]
