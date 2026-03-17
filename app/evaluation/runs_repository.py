from typing import Any

from app.common.mongo import get_mongo_client
from app.config import config
from app.evaluation import models

COLLECTION = "evaluation_runs"


async def _collection() -> Any:
    client = await get_mongo_client()
    db = client.get_database(config.mongo_database)
    return db[COLLECTION]


async def create_run(run: models.EvaluationRun) -> None:
    col = await _collection()
    await col.insert_one(run.model_dump())


async def update_status(run_id: str, status: str) -> None:
    col = await _collection()
    await col.update_one({"run_id": run_id}, {"$set": {"status": status}})


async def append_result(run_id: str, result: models.EvaluationResult) -> None:
    col = await _collection()
    await col.update_one(
        {"run_id": run_id}, {"$push": {"results": result.model_dump()}}
    )


async def save_summary(run_id: str, summary: list[models.EvaluationSummary]) -> None:
    col = await _collection()
    await col.update_one(
        {"run_id": run_id},
        {"$set": {"evaluation_summary": [s.model_dump() for s in summary]}},
    )


async def list_runs() -> list[dict[str, Any]]:
    col = await _collection()
    cursor = col.find({}, {"_id": 0, "run_id": 1, "status": 1})
    return list(await cursor.to_list())


async def get_run(run_id: str) -> models.EvaluationRun | None:
    col = await _collection()
    doc = await col.find_one({"run_id": run_id}, {"_id": 0})
    return models.EvaluationRun.model_validate(doc) if doc else None
