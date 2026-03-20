import asyncio
import json
from logging import getLogger
from typing import Any

import boto3
from mypy_boto3_sqs import SQSClient
from mypy_boto3_sqs.type_defs import MessageTypeDef

from app import config
from app.evaluation import judge_service, models, rag_answer_service, runs_repository

logger = getLogger(__name__)


def _get_client() -> SQSClient:
    return boto3.client(
        "sqs",
        region_name=config.config.aws_region,
        endpoint_url=config.config.sqs_endpoint_url,
    )


def _receive_message(queue_url: str) -> MessageTypeDef | None:
    client = _get_client()
    response = client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=20,
    )
    messages = response.get("Messages", [])
    return messages[0] if messages else None


def _delete_message(queue_url: str, receipt_handle: str) -> None:
    client = _get_client()
    client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


def _get_run_id(msg: MessageTypeDef) -> str:
    body: dict[str, Any] = json.loads(msg["Body"])
    return str(body["run_id"])


async def _process(queue_url: str, msg: MessageTypeDef) -> None:
    run_id = _get_run_id(msg)
    logger.info("Received evaluation request: %s", run_id)

    run = await runs_repository.get_run(run_id)
    if run is None:
        logger.error("Run %s not found in database; skipping", run_id)
        await asyncio.to_thread(_delete_message, queue_url, msg["ReceiptHandle"])
        return

    await _execute_run(run, run_id)
    await asyncio.to_thread(_delete_message, queue_url, msg["ReceiptHandle"])


def _summarise_run(
    results: list[models.EvaluationResult],
) -> list[models.EvaluationSummary]:
    threshold = config.config.llm_as_a_judge_config.score_threshold
    groups: dict[tuple[str, str], list[float]] = {}
    for result in results:
        if result.score is not None:
            key = (result.model, result.rubric)
            groups.setdefault(key, []).append(result.score)
    summaries = []
    for (model, rubric), scores in groups.items():
        avg = sum(scores) / len(scores)
        passed = avg >= threshold
        summaries.append(
            models.EvaluationSummary(
                model=model,
                rubric=rubric,
                average_score=avg,
                passed=passed,
            )
        )
    return summaries


async def _execute_run(run: models.EvaluationRun, run_id: str) -> None:
    await runs_repository.update_status(run_id, "in_progress")

    effective_rubrics = run.rubrics or [judge_service.DEFAULT_RUBRIC]
    done = {(r.question, r.rubric, r.model) for r in run.results}
    for item in run.queries:
        answer = await rag_answer_service.answer_with_rag(
            item.query, run.group_id, snapshot_id=run.snapshot_id
        )
        for rubric in effective_rubrics:
            for model_key in run.models:
                if (item.query, rubric, model_key) in done:
                    continue
                result = await judge_service.evaluate_with_judge(
                    item.query, item.expected_answer, answer, model_key, rubric
                )
                await runs_repository.append_result(run_id, result)
    updated_run = await runs_repository.get_run(run_id)
    if updated_run:
        summary = _summarise_run(updated_run.results)
        await runs_repository.save_summary(run_id, summary)
    await runs_repository.update_status(run_id, "completed")


async def listen() -> None:
    queue_url = config.config.rag_evaluation_start_queue_url

    logger.info("Starting SQS queue listener on %s", queue_url)
    while True:
        try:
            msg = await asyncio.to_thread(_receive_message, queue_url)
            if msg is not None:
                await _process(queue_url, msg)
        except asyncio.CancelledError:
            logger.info("Queue listener cancelled")
            raise
        except Exception:
            logger.exception("Error processing message; it will become visible again")
            await asyncio.sleep(5)
