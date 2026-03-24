import uuid

from app.evaluation import runs_repository, sqs_service
from app.evaluation.models import EvaluationRun


async def create_evaluation_run(
    group_id: str,
    truth_source_id: str,
    snapshot_id: str,
    rubrics: list[str] | None,
    model_keys: list[str],
) -> EvaluationRun:
    run = EvaluationRun(
        run_id=str(uuid.uuid4()),
        status="accepted",
        group_id=group_id,
        truth_source_id=truth_source_id,
        snapshot_id=snapshot_id,
        rubrics=rubrics,
        models=model_keys,
    )
    await runs_repository.create_run(run)
    await sqs_service.enqueue_evaluation(run.run_id)
    return run
