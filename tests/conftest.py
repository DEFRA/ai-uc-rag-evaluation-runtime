import os

import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Set environment variables before any imports happen."""
    os.environ.update(
        {
            "LLM_AS_A_JUDGE_MODEL_ID": "test-judge-model-id",
            "AWS_REGION": "eu-west-2",
            "LLM_MODEL_ID": "test-model-id",
            "RAG_EVALUATION_START_QUEUE_URL": "test-queue-url",
        }
    )
