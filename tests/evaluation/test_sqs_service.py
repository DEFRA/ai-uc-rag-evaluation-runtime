import json

import pytest
from pytest_mock import MockerFixture

import app.evaluation.sqs_service as sqs_service

QUERIES = [
    {"query": "question 1", "expected_answer": "answer 1"},
    {"query": "question 2", "expected_answer": "answer 2"},
]


async def test_enqueue_evaluation_sends_message(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.sqs_service.config.config.rag_evaluation_start_queue_url",
        "http://localhost:4566/000000000000/rag_evaluation_start.fifo",
    )
    mock_client = mocker.MagicMock()
    mocker.patch("app.evaluation.sqs_service.boto3.client", return_value=mock_client)

    await sqs_service.enqueue_evaluation("run-1", "group1", QUERIES)

    mock_client.send_message.assert_called_once_with(
        QueueUrl="http://localhost:4566/000000000000/rag_evaluation_start.fifo",
        MessageBody=json.dumps(
            {
                "run_id": "run-1",
                "group_id": "group1",
                "queries": QUERIES,
                "snapshot_id": None,
                "rubrics": None,
                "models": None,
            }
        ),
        MessageGroupId="group1",
    )


async def test_enqueue_evaluation_raises_when_not_configured(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "app.evaluation.sqs_service.config.config.rag_evaluation_start_queue_url",
        None,
    )

    with pytest.raises(ValueError, match="RAG_EVALUATION_START_QUEUE_URL"):
        await sqs_service.enqueue_evaluation("run-1", "group1", QUERIES)
