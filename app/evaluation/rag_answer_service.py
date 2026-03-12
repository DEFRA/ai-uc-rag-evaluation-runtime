from logging import getLogger

from pydantic_ai import Agent, models
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider

import app.config as config
import app.evaluation.rag_service as rag_service

logger = getLogger(__name__)

_cfg = config.config.llm_config
_provider = BedrockProvider(region_name=config.config.aws_region)
_settings = (
    BedrockModelSettings(
        bedrock_guardrail_config={
            "guardrailIdentifier": _cfg.guardrails_id,
            "guardrailVersion": _cfg.guardrails_version,
            "trace": "enabled",
        }
    )
    if _cfg.guardrails_id
    else None
)

_model: models.Model = BedrockConverseModel(
    _cfg.inference_profile_arn or _cfg.model_id,
    provider=_provider,
    profile=_provider.model_profile(_cfg.model_id),
    settings=_settings,
)

_agent: Agent[None, str] = Agent(
    model=_model,
    system_prompt=(
        "You are a helpful assistant. Answer the question using only the provided "
        "context documents. If the context does not contain enough information to "
        "answer, say so clearly."
    ),
)


async def answer_with_rag(query: str, group_id: str, max_results: int = 5) -> str:
    documents = await rag_service.query_snapshot(group_id, query, max_results)
    context = "\n\n".join(str(doc.get("content", "")) for doc in documents)
    result = await _agent.run(f"Context documents:\n{context}\n\nQuestion: {query}")
    return result.output
