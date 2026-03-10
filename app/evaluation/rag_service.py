from logging import getLogger
from typing import Any, cast

import app.common.http_client as http_client
import app.config as config
from app.evaluation.exceptions import (
    EvaluationDataServiceError,
    EvaluationDataServiceNotConfiguredError,
)

logger = getLogger(__name__)


async def query_snapshot(
    group_id: str, query: str, max_results: int
) -> list[dict[str, Any]]:
    if not config.config.evaluation_data_service_url:
        raise EvaluationDataServiceNotConfiguredError()

    url = (
        f"{str(config.config.evaluation_data_service_url).rstrip('/')}/snapshots/query"
    )
    body = {
        "groupId": group_id,
        "query": query,
        "maxResults": max_results,
    }
    async with http_client.create_async_client() as client:
        response = await client.post(url, json=body)

    if response.status_code != 200:
        try:
            detail = (
                response.json()
                if response.content
                else response.text or "Unknown error"
            )
        except Exception:
            detail = response.text or "Unknown error"
        logger.error(
            "Error when contacting rag service %s %s", response.status_code, detail
        )
        raise EvaluationDataServiceError(
            status_code=response.status_code, detail=detail
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise EvaluationDataServiceError(
            status_code=response.status_code,
            detail="Expected snapshots response to be a list",
        )

    return cast(list[dict[str, Any]], payload)
