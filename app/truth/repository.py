import abc

import pymongo.asynchronous.collection
import pymongo.asynchronous.database

from app.truth import models


class AbstractTruthDataSourceRepository(abc.ABC):
    @abc.abstractmethod
    async def create(self, source: models.TruthDataSource) -> None:
        """Insert a new truth data source"""

    @abc.abstractmethod
    async def update_question_answers(
        self, source_id: str, question_answers: list[models.QuestionAnswer]
    ) -> bool:
        """Update question answers for a source; returns True if found"""

    @abc.abstractmethod
    async def list_sources(self) -> list[models.TruthDataSourceSummary]:
        """List all truth data source summaries"""

    @abc.abstractmethod
    async def get(self, source_id: str) -> models.TruthDataSource | None:
        """Get a truth data source by ID"""


class MongoTruthDataSourceRepository(AbstractTruthDataSourceRepository):
    def __init__(self, db: pymongo.asynchronous.database.AsyncDatabase):
        self.collection: pymongo.asynchronous.collection.AsyncCollection = db[
            "truth_data_sources"
        ]

    async def create(self, source: models.TruthDataSource) -> None:
        await self.collection.insert_one(source.model_dump())

    async def update_question_answers(
        self, source_id: str, question_answers: list[models.QuestionAnswer]
    ) -> bool:
        result = await self.collection.update_one(
            {"id": source_id},
            {
                "$set": {
                    "question_answers": [qa.model_dump() for qa in question_answers]
                }
            },
        )
        return bool(result.matched_count > 0)

    async def list_sources(self) -> list[models.TruthDataSourceSummary]:
        cursor = self.collection.find({}, {"_id": 0, "id": 1, "dataset_id": 1})
        return [
            models.TruthDataSourceSummary.model_validate(doc)
            for doc in await cursor.to_list()
        ]

    async def get(self, source_id: str) -> models.TruthDataSource | None:
        doc = await self.collection.find_one({"id": source_id}, {"_id": 0})
        return models.TruthDataSource.model_validate(doc) if doc else None
