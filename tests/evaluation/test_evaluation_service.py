from pytest_mock import MockerFixture

from app.evaluation import evaluation_service, runs_repository, sqs_service
from app.evaluation.models import EvaluationQuery


async def test_create_evaluation_run_persists_and_enqueues(
    mocker: MockerFixture,
) -> None:
    mock_create_run = mocker.patch.object(
        runs_repository, "create_run", new=mocker.AsyncMock(return_value=None)
    )
    mock_enqueue = mocker.patch.object(
        sqs_service, "enqueue_evaluation", new=mocker.AsyncMock(return_value=None)
    )

    queries = [EvaluationQuery(query="q1", expected_answer="a1")]
    run = await evaluation_service.create_evaluation_run(
        group_id="g1",
        queries=queries,
        snapshot_id="snap-1",
        rubrics=["custom rubric"],
        model_keys=["sonnet"],
    )

    assert run.status == "accepted"
    assert run.group_id == "g1"
    assert run.queries == queries
    assert run.snapshot_id == "snap-1"
    assert run.rubrics == ["custom rubric"]
    assert run.models == ["sonnet"]
    mock_create_run.assert_awaited_once_with(run)
    mock_enqueue.assert_awaited_once_with(run.run_id)
