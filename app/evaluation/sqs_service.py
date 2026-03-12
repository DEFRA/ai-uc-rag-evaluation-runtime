import asyncio
import json
from logging import getLogger

import boto3

import app.config as config

logger = getLogger(__name__)


def _get_client() -> object:
    return boto3.client(
        "sqs",
        region_name=config.config.aws_region,
        endpoint_url=config.config.sqs_endpoint_url,
    )


async def enqueue_evaluation(
    run_id: str,
    group_id: str,
    query: str,
    expected_answer: str,
    snapshot_id: str | None = None,
) -> None:
    queue_url = config.config.rag_evaluation_start_queue_url
    if not queue_url:
        msg = "RAG_EVALUATION_START_QUEUE_URL is not configured"
        raise ValueError(msg)

    message: dict[str, str | None] = {
        "run_id": run_id,
        "group_id": group_id,
        "query": query,
        "expected_answer": expected_answer,
        "snapshot_id": snapshot_id,
    }

    def _send() -> None:
        client = _get_client()
        client.send_message(  # type: ignore[attr-defined]
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
            MessageGroupId=group_id,
        )

    await asyncio.to_thread(_send)
