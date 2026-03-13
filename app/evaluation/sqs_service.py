import asyncio
import json
from logging import getLogger

import boto3
from mypy_boto3_sqs import SQSClient

import app.config as config

logger = getLogger(__name__)


def _get_client() -> SQSClient:
    return boto3.client(
        "sqs",
        region_name=config.config.aws_region,
        endpoint_url=config.config.sqs_endpoint_url,
    )


async def enqueue_evaluation(run_id: str) -> None:
    queue_url = config.config.rag_evaluation_start_queue_url
    if not queue_url:
        msg = "RAG_EVALUATION_START_QUEUE_URL is not configured"
        raise ValueError(msg)

    def _send() -> None:
        client = _get_client()
        client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps({"run_id": run_id}),
            MessageGroupId="run_evaluation",
        )

    await asyncio.to_thread(_send)
