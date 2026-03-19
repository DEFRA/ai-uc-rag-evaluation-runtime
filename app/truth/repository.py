from typing import Any

from app import config
from app.common import mongo
from app.truth import models

COLLECTION = "truth_data_sources"


async def _collection() -> Any:
    client = await mongo.get_mongo_client()
    db = client.get_database(config.config.mongo_database)
    return db[COLLECTION]


async def create(source: models.TruthDataSource) -> None:
    col = await _collection()
    await col.insert_one(source.model_dump())


async def update_question_answers(
    source_id: str, question_answers: list[models.QuestionAnswer]
) -> bool:
    col = await _collection()
    result = await col.update_one(
        {"id": source_id},
        {"$set": {"question_answers": [qa.model_dump() for qa in question_answers]}},
    )
    return bool(result.matched_count > 0)


async def list_sources() -> list[models.TruthDataSourceSummary]:
    col = await _collection()
    cursor = col.find({}, {"_id": 0, "id": 1, "dataset_id": 1})
    return [
        models.TruthDataSourceSummary.model_validate(doc)
        for doc in await cursor.to_list()
    ]


async def get(source_id: str) -> models.TruthDataSource | None:
    col = await _collection()
    doc = await col.find_one({"id": source_id}, {"_id": 0})
    return models.TruthDataSource.model_validate(doc) if doc else None
