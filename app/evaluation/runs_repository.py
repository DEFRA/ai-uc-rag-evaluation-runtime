import uuid
from typing import Any

from app.common.mongo import get_mongo_client
from app.config import config

COLLECTION = "evaluation_runs"


async def _collection() -> Any:
    client = await get_mongo_client()
    db = client.get_database(config.mongo_database)
    return db[COLLECTION]


async def create_run(
    run_id: str,
    group_id: str,
    query: str,
    expected_answer: str,
    snapshot_id: str | None,
) -> None:
    col = await _collection()
    await col.insert_one(
        {
            "run_id": run_id,
            "status": "accepted",
            "group_id": group_id,
            "query": query,
            "expected_answer": expected_answer,
            "snapshot_id": snapshot_id,
        }
    )


async def update_status(run_id: str, status: str, result: dict | None = None) -> None:
    col = await _collection()
    update: dict[str, Any] = {"$set": {"status": status}}
    if result is not None:
        update["$set"]["result"] = result
    await col.update_one({"run_id": run_id}, update)


async def list_runs() -> list[dict[str, Any]]:
    col = await _collection()
    cursor = col.find({}, {"_id": 0, "run_id": 1, "status": 1})
    return list(await cursor.to_list())


def new_run_id() -> str:
    return str(uuid.uuid4())
