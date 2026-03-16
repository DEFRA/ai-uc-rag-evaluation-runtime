import json

import pytest
from pytest_mock import MockerFixture

from app.evaluation import sqs_service


async def test_enqueue_evaluation_sends_message(mocker: MockerFixture) -> None:
    mocker.patch(
        "app.evaluation.sqs_service.config.config.rag_evaluation_start_queue_url",
        "http://localhost:4566/000000000000/rag_evaluation_start.fifo",
    )
    mock_client = mocker.MagicMock()
    mocker.patch.object(sqs_service, "_client", mock_client)

    await sqs_service.enqueue_evaluation("run-1")

    call_kwargs = mock_client.send_message.call_args.kwargs
    assert (
        call_kwargs["QueueUrl"]
        == "http://localhost:4566/000000000000/rag_evaluation_start.fifo"
    )
    assert call_kwargs["MessageBody"] == json.dumps({"run_id": "run-1"})
    assert call_kwargs["MessageGroupId"] == "run_evaluation"
    assert call_kwargs["MessageDeduplicationId"].startswith("run-1-")


async def test_enqueue_evaluation_raises_when_not_configured(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "app.evaluation.sqs_service.config.config.rag_evaluation_start_queue_url",
        None,
    )

    with pytest.raises(ValueError, match="RAG_EVALUATION_START_QUEUE_URL"):
        await sqs_service.enqueue_evaluation("run-1")
