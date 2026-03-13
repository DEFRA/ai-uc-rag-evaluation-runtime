import os

from pydantic import Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmAsAJudgeConfig(BaseSettings):
    model_config = SettingsConfigDict()
    model_id: str = Field(..., alias="LLM_AS_A_JUDGE_MODEL_ID")
    inference_profile_arn: str | None = Field(
        None, alias="LLM_AS_A_JUDGE_INFERENCE_PROFILE_ARN"
    )
    temperature: float = Field(0, alias="LLM_AS_A_JUDGE_TEMPERATURE")
    n_prompts: int = Field(10, alias="LLM_AS_A_JUDGE_N_PROMPTS")
    max_tokens: int = Field(2048, alias="LLM_AS_A_JUDGE_MAX_TOKENS")
    guardrails_id: str | None = Field(None, alias="LLM_AS_A_JUDGE_GUARDRAILS_ID")
    guardrails_version: str | None = Field(
        None, alias="LLM_AS_A_JUDGE_GUARDRAILS_VERSION"
    )
    models: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def build_models(cls, values: dict) -> dict:
        models: dict[str, dict[str, str]] = {}
        for key, value in os.environ.items():
            if key.startswith("LLM_AS_A_JUDGE_MODEL_ID_"):
                name = key[len("LLM_AS_A_JUDGE_MODEL_ID_") :].lower()
                arn = os.environ.get(
                    f"LLM_AS_A_JUDGE_INFERENCE_PROFILE_ARN_{name.upper()}", ""
                )
                models[name] = {"model_id": value, "arn": arn}
        if models:
            values["models"] = models
        return values


class LlmConfig(BaseSettings):
    model_config = SettingsConfigDict()
    model_id: str = Field(..., alias="LLM_MODEL_ID")
    inference_profile_arn: str | None = Field(None, alias="LLM_INFERENCE_PROFILE_ARN")
    guardrails_id: str | None = Field(None, alias="LLM_GUARDRAILS_ID")
    guardrails_version: str | None = Field(None, alias="LLM_GUARDRAILS_VERSION")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict()
    aws_region: str = Field(..., alias="AWS_REGION")
    python_env: str | None = None
    host: str = "127.0.0.1"
    port: int = 8086
    log_config: str | None = None
    mongo_uri: str | None = None
    mongo_database: str = "ai-uc-rag-evaluation-runtime"
    mongo_truststore: str = "TRUSTSTORE_CDP_ROOT_CA"
    aws_endpoint_url: str | None = None
    sqs_endpoint_url: str | None = Field(None, alias="SQS_ENDPOINT_URL")
    http_proxy: HttpUrl | None = None
    enable_metrics: bool = False
    tracing_header: str = "x-cdp-request-id"
    evaluation_data_service_url: HttpUrl | None = None
    rag_evaluation_start_queue_url: str = Field(
        ..., alias="RAG_EVALUATION_START_QUEUE_URL"
    )
    llm_as_a_judge_config: LlmAsAJudgeConfig = Field(
        default_factory=lambda: LlmAsAJudgeConfig()
    )
    llm_config: LlmConfig = Field(default_factory=lambda: LlmConfig())


config = AppConfig()
