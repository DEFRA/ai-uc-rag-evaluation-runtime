import app.common.http_client as http_client
import app.config as config
from app.evaluation.exceptions import (
    EvaluationDataServiceError,
    EvaluationDataServiceNotConfiguredError,
)


async def query_snapshot(
    group_id: str, query: str, max_results: int
) -> tuple[int, list[dict]]:
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
        raise EvaluationDataServiceError(
            status_code=response.status_code, detail=detail
        )

    return response.json()
