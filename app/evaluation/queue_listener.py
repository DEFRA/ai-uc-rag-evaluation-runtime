import asyncio
import json
from logging import getLogger
from typing import Any

import boto3

import app.config as config
import app.evaluation.evaluation_service as evaluation_service
import app.evaluation.runs_repository as runs_repository

logger = getLogger(__name__)


def _get_client() -> Any:
    return boto3.client(
        "sqs",
        region_name=config.config.aws_region,
        endpoint_url=config.config.sqs_endpoint_url,
    )


def _receive(queue_url: str) -> dict[str, Any] | None:
    client = _get_client()
    response = client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=20,
    )
    messages: list[dict[str, Any]] = response.get("Messages", [])
    return messages[0] if messages else None


def _delete(queue_url: str, receipt_handle: str) -> None:
    client = _get_client()
    client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


async def listen() -> None:
    queue_url = config.config.rag_evaluation_start_queue_url
    if not queue_url:
        logger.warning(
            "RAG_EVALUATION_START_QUEUE_URL not configured; queue listener disabled"
        )
        return

    logger.info("Starting SQS queue listener on %s", queue_url)
    while True:
        try:
            msg = await asyncio.to_thread(_receive, queue_url)
            if msg is None:
                continue
            body: dict[str, Any] = json.loads(msg["Body"])
            run_id: str = body["run_id"]
            logger.info("Received evaluation request: %s", run_id)
            run = await runs_repository.get_run(run_id)
            if run is None:
                logger.error("Run %s not found in database; skipping", run_id)
                await asyncio.to_thread(_delete, queue_url, msg["ReceiptHandle"])
                continue
            await runs_repository.update_status(run_id, "in_progress")
            rubrics: list[str] | None = run.get("rubrics") or None
            judge_model_keys: list[str] | None = run.get("models") or None
            for item in run["queries"]:
                item_results = await evaluation_service.run_rag_evaluation(
                    run["group_id"],
                    item["query"],
                    item["expected_answer"],
                    snapshot_id=run.get("snapshot_id"),
                    rubrics=rubrics,
                    judge_model_keys=judge_model_keys,
                )
                for result in item_results:
                    await runs_repository.append_result(run_id, result)
            await runs_repository.update_status(run_id, "completed")
            await asyncio.to_thread(_delete, queue_url, msg["ReceiptHandle"])
        except asyncio.CancelledError:
            logger.info("Queue listener cancelled")
            raise
        except Exception:
            logger.exception("Error processing message; it will become visible again")
            await asyncio.sleep(5)
