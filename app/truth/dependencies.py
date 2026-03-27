"""Dependency injection factories for truth module."""

import fastapi
import pymongo.asynchronous.database

from app.common import mongo
from app.truth import repository


def get_truth_repository(
    db: pymongo.asynchronous.database.AsyncDatabase = fastapi.Depends(mongo.get_db),
) -> repository.AbstractTruthDataSourceRepository:
    """Dependency injection for MongoTruthDataSourceRepository."""
    return repository.MongoTruthDataSourceRepository(db)
