import uuid

import app.evaluation.runs_repository as runs_repository
import app.evaluation.sqs_service as sqs_service
from app.evaluation.models import EvaluationQuery, EvaluationRun


async def create_evaluation_run(
    group_id: str,
    queries: list[EvaluationQuery],
    snapshot_id: str,
    rubrics: list[str] | None,
    model_keys: list[str],
) -> EvaluationRun:
    run = EvaluationRun(
        run_id=str(uuid.uuid4()),
        status="accepted",
        group_id=group_id,
        queries=queries,
        snapshot_id=snapshot_id,
        rubrics=rubrics,
        models=model_keys,
    )
    await runs_repository.create_run(run)
    await sqs_service.enqueue_evaluation(run.run_id)
    return run
