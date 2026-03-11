class EvaluationDataServiceNotConfiguredError(Exception):
    """Evaluation data service URL is not configured."""


class EvaluationDataServiceError(Exception):
    """The evaluation data service returned an error response."""

    def __init__(self, status_code: int, detail: str | dict) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Evaluation data service returned {status_code}: {detail}")
